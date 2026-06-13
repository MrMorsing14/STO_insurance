"""
STO Pipeline v2: Dekomponering + per-delkrav evaluering.

Flow:
  Krav ind
    → Pre-filter (kortniveau kendt? under STO-grænse?)        [metadata]
    → Dekomponering i delkrav                                  [LLM]
    → Pr. delkrav:
        → Metadata-filter (dækket overhovedet?)                [metadata]
        → Vektorsøgning (chunks for delkravets dækningstype)   [ChromaDB]
        → Evaluering                                           [LLM]
        → Confidence routing                                   [thresholds]
    → Aggregering → samlet afgørelse + kundebesked             [deterministisk]
"""
from config.settings import CONFIDENCE_AUTO_APPROVE, CONFIDENCE_AUTO_REJECT
from src.aggregation import aggreger
from src.decomposition.claim_decomposer import KANONISKE_DÆKNINGSTYPER, ClaimDecomposer
from src.evaluation.llm_evaluator import LLMEvaluator
from src.models import (
    Afgørelse,
    DelAfgørelse,
    DelAfgørelseType,
    DelKrav,
    ForsikringsKrav,
    KravAfgørelse,
)
from src.retrieval.metadata_filter import MetadataFilter


class STOPipeline:
    def __init__(
        self,
        metadata_filter: MetadataFilter = None,
        vector_store=None,
        decomposer: ClaimDecomposer = None,
        llm_evaluator: LLMEvaluator = None,
    ):
        # Dependency injection så komponenter kan mockes i tests.
        # PolicyVectorStore importeres lazy — chromadb/sentence-transformers
        # er tunge dependencies, som mockede tests ikke behøver.
        self.metadata_filter = metadata_filter or MetadataFilter()
        if vector_store is None:
            from src.retrieval.vector_store import PolicyVectorStore
            vector_store = PolicyVectorStore()
        self.vector_store = vector_store
        self.decomposer = decomposer or ClaimDecomposer()
        self.llm_evaluator = llm_evaluator or LLMEvaluator()

    def process_claim(self, krav: ForsikringsKrav) -> KravAfgørelse:
        print(f"\n━━ Behandler krav {krav.krav_id} ({krav.kortniveau.value}, {krav.beløb_dkk} DKK) ━━")

        # ── Trin 0: Pre-filter på hele kravet ──
        eskaleringsgrund = self.metadata_filter.pre_filter_krav(krav)
        if eskaleringsgrund:
            print(f"  [PRE-FILTER] Eskaleret: {eskaleringsgrund}")
            return KravAfgørelse(
                krav_id=krav.krav_id,
                afgørelse=Afgørelse.MANUELT_REVIEW,
                begrundelse=eskaleringsgrund,
                kundebesked=(
                    f"Vedr. dit krav {krav.krav_id}:\n\nDit krav kræver en manuel "
                    "vurdering af en sagsbehandler. Du hører fra os hurtigst muligt."
                ),
                konfidens=1.0,
                ansøgt_beløb_dkk=krav.beløb_dkk,
                metadata_filtreret=True,
            )

        # ── Trin 1: Dekomponering ──
        delkrav_liste = self.decomposer.decompose(krav)
        print(f"  [DECOMP] {len(delkrav_liste)} delkrav:")
        for d in delkrav_liste:
            beløb = f"{d.beløb_dkk} DKK" if d.beløb_dkk is not None else "uspecificeret"
            print(f"    - {d.delkrav_id}: {d.dækningstype} ({beløb}) — {d.beskrivelse[:60]}")

        # ── Trin 2-4: Pr. delkrav ──
        delafgørelser: list[DelAfgørelse] = []
        for delkrav in delkrav_liste:
            delafgørelser.append(self._behandl_delkrav(krav, delkrav))

        # ── Trin 5: Aggregering ──
        resultat = aggreger(krav, delafgørelser)
        print(f"  [AGGREGERING] Samlet afgørelse: {resultat.afgørelse.value} "
              f"({resultat.godkendt_beløb_dkk:.2f} af {resultat.ansøgt_beløb_dkk:.2f} DKK)")
        return resultat

    def _behandl_delkrav(self, krav: ForsikringsKrav, delkrav) -> DelAfgørelse:
        delafgørelse = self._vurder_delkrav(krav, delkrav)

        # ── Omklassificerings-sikkerhedsnet ──
        # Hvis evaluatoren afviser med "hører under en anden dækningstype",
        # prøver vi ÉN gang under den foreslåede type. Det konverterer den
        # farligste fejl (selvsikker forkert afvisning pga. forkert kategori)
        # til enten en korrekt godkendelse eller en menneskelig vurdering.
        # Hårdt loft på 1 retry — ellers bygger man et system der "shopper"
        # efter en dækning der godkender.
        forslag = delafgørelse.forslag_til_anden_dækningstype
        if (
            delafgørelse.afgørelse == DelAfgørelseType.AFVIST
            and forslag
            and forslag in KANONISKE_DÆKNINGSTYPER
            and forslag != delkrav.dækningstype
        ):
            print(f"  [RECLASS] {delkrav.delkrav_id}: afvist under '{delkrav.dækningstype}', "
                  f"evaluator foreslår '{forslag}' — re-evaluerer (1 forsøg)")
            return self._omklassificer_og_vurder(krav, delkrav, delafgørelse, forslag)

        return delafgørelse

    def _omklassificer_og_vurder(
        self, krav: ForsikringsKrav, delkrav, original: DelAfgørelse, ny_type: str
    ) -> DelAfgørelse:
        nyt_delkrav = DelKrav(
            delkrav_id=delkrav.delkrav_id,
            beskrivelse=delkrav.beskrivelse,
            dækningstype=ny_type,
            beløb_dkk=delkrav.beløb_dkk,
        )
        retry = self._vurder_delkrav(krav, nyt_delkrav)
        retry.omklassificeret_fra = delkrav.dækningstype

        if retry.afgørelse == DelAfgørelseType.GODKENDT:
            # Korrekt kategori fundet — godkendelsen står
            retry.begrundelse = (
                f"[Omklassificeret fra '{delkrav.dækningstype}' til '{ny_type}'] "
                + retry.begrundelse
            )
            print(f"  [RECLASS] {delkrav.delkrav_id}: godkendt under '{ny_type}'")
            return retry

        if retry.metadata_filtreret and retry.afgørelse == DelAfgørelseType.AFVIST:
            # Den foreslåede type er deterministisk ikke dækket på kortet —
            # afvisningen er reel under begge kategorier
            retry.begrundelse = (
                f"Ikke dækket under '{delkrav.dækningstype}' ({original.begrundelse}). "
                f"Heller ikke dækket under '{ny_type}' på dette kortniveau."
            )
            print(f"  [RECLASS] {delkrav.delkrav_id}: '{ny_type}' ikke dækket på kortet — afvist")
            return retry

        # To LLM-vurderinger er uenige om kategorien, eller retry er usikker
        # → det afgør et menneske, ikke endnu et LLM-kald
        retry_udfald = retry.afgørelse.value
        retry.afgørelse = DelAfgørelseType.MANUELT_REVIEW
        retry.godkendt_beløb_dkk = None
        retry.begrundelse = (
            f"Klassificeringstvivl: afvist under '{delkrav.dækningstype}' "
            f"({original.begrundelse}) — re-evaluering under '{ny_type}' gav "
            f"'{retry_udfald}' ({retry.begrundelse})"
        )
        print(f"  [RECLASS] {delkrav.delkrav_id}: uafklaret → manuelt review")
        return retry

    def _vurder_delkrav(self, krav: ForsikringsKrav, delkrav) -> DelAfgørelse:
        # Metadata-filter: kan delkravet afgøres uden LLM?
        meta_result = self.metadata_filter.filter_delkrav(krav, delkrav)
        if meta_result is not None:
            print(f"  [METADATA] {delkrav.delkrav_id}: {meta_result.afgørelse.value} (uden LLM)")
            return meta_result

        # Retrieval målrettet delkravets dækningstype
        coverage_context = self.metadata_filter.get_coverage_context(
            krav.kortniveau.value, delkrav.dækningstype
        )
        policy_chunks = self.vector_store.search(
            query=delkrav.beskrivelse,
            kortniveau=self._map_kortniveau_for_search(krav.kortniveau.value),
            dækningstype=delkrav.dækningstype,
            n_results=5,
        )
        print(f"  [RETRIEVAL] {delkrav.delkrav_id}: {len(policy_chunks)} chunks")

        # LLM-evaluering af netop dette delkrav
        delafgørelse = self.llm_evaluator.evaluate_delkrav(
            krav, delkrav, policy_chunks, coverage_context
        )
        print(f"  [LLM] {delkrav.delkrav_id}: {delafgørelse.afgørelse.value} "
              f"(konfidens {delafgørelse.konfidens})")

        # Confidence routing pr. delkrav
        return self._apply_confidence_routing(delafgørelse)

    @staticmethod
    def _apply_confidence_routing(d: DelAfgørelse) -> DelAfgørelse:
        threshold = None
        if d.afgørelse == DelAfgørelseType.GODKENDT:
            threshold = CONFIDENCE_AUTO_APPROVE
        elif d.afgørelse == DelAfgørelseType.AFVIST:
            threshold = CONFIDENCE_AUTO_REJECT

        if threshold is not None and d.konfidens < threshold:
            d.afgørelse = DelAfgørelseType.MANUELT_REVIEW
            d.begrundelse += f" [Eskaleret: konfidens {d.konfidens} under threshold {threshold}]"
            d.godkendt_beløb_dkk = None
        return d

    @staticmethod
    def _map_kortniveau_for_search(kortniveau: str) -> str:
        """Family-varianter deler policytekst med primærkortet."""
        family_map = {
            "mastercard_gold_family": "mastercard_gold",
            "mastercard_platinum_family": "mastercard_platinum",
            "world_elite_family": "world_elite",
        }
        return family_map.get(kortniveau, kortniveau)
