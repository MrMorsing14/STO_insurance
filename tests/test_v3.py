"""
Unit tests for v3-funktionerne: itemiseret klassificering,
omklassificerings-sikkerhedsnettet og API-validering af poster.
Kræver ingen API-nøgle — alt mockes.

Kør: python -m pytest tests/test_v3.py -v
"""
import json
from datetime import date
from unittest.mock import MagicMock

from src.decomposition.claim_decomposer import ClaimDecomposer
from src.evaluation.llm_evaluator import LLMEvaluator
from src.models import (
    Afgørelse,
    DelAfgørelseType,
    ForsikringsKrav,
    Kortniveau,
    KravPost,
)
from src.pipeline import STOPipeline
from src.retrieval.metadata_filter import MetadataFilter


def lav_krav(**overrides) -> ForsikringsKrav:
    defaults = dict(
        krav_id="TEST-V3",
        kortniveau=Kortniveau.WORLD_ELITE,
        beløb_dkk=4000.0,
        hændelse_beskrivelse=(
            "Vores fly fra Billund til København var over 3 timer forsinket. "
            "Vi nåede ikke vores separate fly til Nairobi og måtte købe nye billetter."
        ),
        poster=[
            KravPost(
                beskrivelse="Nye flybilletter København til Nairobi efter mistet forbindelse",
                beløb_dkk=4000.0,
                dækningstype_hint="flyforsinkelse",  # kundens FORKERTE hint
            )
        ],
        hændelse_dato=date(2026, 5, 10),
        dokumentation="Boardingkort, forsinkelsesbekræftelse, kvittering for nye billetter",
    )
    defaults.update(overrides)
    return ForsikringsKrav(**defaults)


def mock_mistral(payload: dict) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = json.dumps(payload, ensure_ascii=False)
    client.chat.complete.return_value = MagicMock(choices=[MagicMock(message=msg)])
    return client


def mock_mistral_sequence(payloads: list[dict]) -> MagicMock:
    """Klient der returnerer payloads i rækkefølge (kald 1, 2, ...)."""
    client = MagicMock()
    responses = []
    for p in payloads:
        msg = MagicMock()
        msg.content = json.dumps(p, ensure_ascii=False)
        responses.append(MagicMock(choices=[MagicMock(message=msg)]))
    client.chat.complete.side_effect = responses
    return client


# ── Itemiseret klassificering ──────────────────────────────────────────

class TestKlassificering:
    def test_klassificering_overstyrer_kundens_hint(self):
        """Modellen siger forsinket_fremmøde selvom kunden hintede flyforsinkelse."""
        payload = {"klassificeringer": [{"post_nr": 1, "dækningstype": "forsinket_fremmøde"}]}
        d = ClaimDecomposer(client=mock_mistral(payload))
        delkrav = d.decompose(lav_krav())
        assert len(delkrav) == 1
        assert delkrav[0].dækningstype == "forsinket_fremmøde"
        assert delkrav[0].beløb_dkk == 4000.0  # kundens beløb røres ikke

    def test_ugyldig_klassificering_falder_tilbage_til_gyldigt_hint(self):
        payload = {"klassificeringer": [{"post_nr": 1, "dækningstype": "vrøvl"}]}
        d = ClaimDecomposer(client=mock_mistral(payload))
        delkrav = d.decompose(lav_krav())
        assert delkrav[0].dækningstype == "flyforsinkelse"  # hintet er gyldigt → bruges

    def test_ugyldig_klassificering_uden_hint_giver_ukendt(self):
        payload = {"klassificeringer": []}
        krav = lav_krav(poster=[KravPost(beskrivelse="noget mystisk", beløb_dkk=4000.0)])
        d = ClaimDecomposer(client=mock_mistral(payload))
        delkrav = d.decompose(krav)
        assert delkrav[0].dækningstype == "ukendt"

    def test_api_fejl_bruger_hints(self):
        client = MagicMock()
        client.chat.complete.side_effect = RuntimeError("API nede")
        d = ClaimDecomposer(client=client)
        delkrav = d.decompose(lav_krav())
        assert delkrav[0].dækningstype == "flyforsinkelse"  # hint som fallback

    def test_flere_poster_klassificeres_i_et_kald(self):
        krav = lav_krav(
            beløb_dkk=6000.0,
            poster=[
                KravPost(beskrivelse="nye billetter efter mistet forbindelse", beløb_dkk=4000.0),
                KravPost(beskrivelse="aftensmad og hotel under ventetiden", beløb_dkk=2000.0),
            ],
        )
        payload = {"klassificeringer": [
            {"post_nr": 1, "dækningstype": "forsinket_fremmøde"},
            {"post_nr": 2, "dækningstype": "flyforsinkelse"},
        ]}
        client = mock_mistral(payload)
        d = ClaimDecomposer(client=client)
        delkrav = d.decompose(krav)
        assert [x.dækningstype for x in delkrav] == ["forsinket_fremmøde", "flyforsinkelse"]
        assert client.chat.complete.call_count == 1  # ÉT kald for alle poster


# ── Omklassificerings-sikkerhedsnet ────────────────────────────────────

def byg_pipeline(decomp_payload, eval_payloads):
    vector_store = MagicMock()
    vector_store.search.return_value = [{"text": "policy chunk", "metadata": {"sektion": "9.0"}}]
    return STOPipeline(
        metadata_filter=MetadataFilter(),
        vector_store=vector_store,
        decomposer=ClaimDecomposer(client=mock_mistral(decomp_payload)),
        llm_evaluator=LLMEvaluator(client=mock_mistral_sequence(eval_payloads)),
    )


class TestOmklassificering:
    DECOMP = {"klassificeringer": [{"post_nr": 1, "dækningstype": "flyforsinkelse"}]}

    AFVIST_MED_FORSLAG = {
        "afgørelse": "afvist",
        "begrundelse": "Udgifter til flybilletter er undtaget (sektion 9.3)",
        "konfidens": 0.95,
        "relevante_betingelser": ["9.3"],
        "godkendt_beløb_dkk": 0,
        "forslag_til_anden_dækningstype": "forsinket_fremmøde",
    }

    def test_buddy_casen_godkendes_efter_omklassificering(self):
        """Afvist under flyforsinkelse, godkendt under forsinket_fremmøde."""
        retry_godkendt = {
            "afgørelse": "godkendt",
            "begrundelse": "Nye billetter for at indhente rejseruten er dækket (sektion 10.2)",
            "konfidens": 0.93,
            "relevante_betingelser": ["10.1", "10.2"],
            "godkendt_beløb_dkk": 4000,
            "forslag_til_anden_dækningstype": None,
        }
        pipeline = byg_pipeline(self.DECOMP, [self.AFVIST_MED_FORSLAG, retry_godkendt])
        resultat = pipeline.process_claim(lav_krav())

        assert resultat.afgørelse == Afgørelse.GODKENDT
        assert resultat.godkendt_beløb_dkk == 4000.0
        d = resultat.delafgørelser[0]
        assert d.dækningstype == "forsinket_fremmøde"
        assert d.omklassificeret_fra == "flyforsinkelse"
        assert "Omklassificeret" in d.begrundelse

    def test_to_uenige_llm_vurderinger_giver_manuelt_review(self):
        """Retry afviser også → klassificeringstvivl → menneske afgør."""
        retry_afvist = {
            "afgørelse": "afvist",
            "begrundelse": "Heller ikke dækket her",
            "konfidens": 0.9,
            "relevante_betingelser": [],
            "godkendt_beløb_dkk": 0,
            "forslag_til_anden_dækningstype": None,
        }
        pipeline = byg_pipeline(self.DECOMP, [self.AFVIST_MED_FORSLAG, retry_afvist])
        resultat = pipeline.process_claim(lav_krav())

        assert resultat.afgørelse == Afgørelse.MANUELT_REVIEW
        d = resultat.delafgørelser[0]
        assert d.afgørelse == DelAfgørelseType.MANUELT_REVIEW
        assert "Klassificeringstvivl" in d.begrundelse
        assert resultat.godkendt_beløb_dkk == 0.0

    def test_kun_en_retry_aldrig_kaskade(self):
        """Retry foreslår ENDNU en type — den må IKKE følges (loft på 1)."""
        retry_afvist_med_nyt_forslag = {
            "afgørelse": "afvist",
            "begrundelse": "Prøv en tredje type",
            "konfidens": 0.9,
            "relevante_betingelser": [],
            "godkendt_beløb_dkk": 0,
            "forslag_til_anden_dækningstype": "rejseulykke",
        }
        pipeline = byg_pipeline(self.DECOMP, [self.AFVIST_MED_FORSLAG, retry_afvist_med_nyt_forslag])
        resultat = pipeline.process_claim(lav_krav())

        # Præcis 2 evaluator-kald (original + 1 retry), ikke 3
        assert pipeline.llm_evaluator._client.chat.complete.call_count == 2
        assert resultat.afgørelse == Afgørelse.MANUELT_REVIEW

    def test_forslag_til_ikke_daekket_type_giver_afvisning(self):
        """Foreslået type er ikke dækket på kortet → reel afvisning under begge."""
        # Business dækker flyforsinkelse men IKKE rejseulykke
        # → original LLM-afvisning + metadata-afvisning på retryen
        afvist_forslag_rejseulykke = {
            **self.AFVIST_MED_FORSLAG,
            "forslag_til_anden_dækningstype": "rejseulykke",
        }
        pipeline = byg_pipeline(self.DECOMP, [afvist_forslag_rejseulykke])
        resultat = pipeline.process_claim(lav_krav(kortniveau=Kortniveau.BUSINESS))

        d = resultat.delafgørelser[0]
        assert d.afgørelse == DelAfgørelseType.AFVIST
        assert "Heller ikke dækket" in d.begrundelse
        # Kun 1 LLM-evaluering — retryen blev afgjort af metadata
        assert pipeline.llm_evaluator._client.chat.complete.call_count == 1

    def test_godkendelse_udloser_ingen_retry(self):
        godkendt = {
            "afgørelse": "godkendt", "begrundelse": "Dækket", "konfidens": 0.95,
            "relevante_betingelser": [], "godkendt_beløb_dkk": 4000,
            "forslag_til_anden_dækningstype": "forsinket_fremmøde",  # ignoreres ved godkendt
        }
        pipeline = byg_pipeline(self.DECOMP, [godkendt])
        resultat = pipeline.process_claim(lav_krav())
        assert resultat.afgørelse == Afgørelse.GODKENDT
        assert pipeline.llm_evaluator._client.chat.complete.call_count == 1


# ── API: poster-validering ─────────────────────────────────────────────

class TestAPIPoster:
    def setup_method(self):
        from fastapi.testclient import TestClient
        import app.api as api

        class FakePipeline:
            def process_claim(self, krav):
                from src.models import KravAfgørelse
                # Gem kravet så testen kan inspicere det
                self.sidste_krav = krav
                return KravAfgørelse(
                    krav_id=krav.krav_id, afgørelse=Afgørelse.GODKENDT,
                    begrundelse="ok", konfidens=1.0,
                    ansøgt_beløb_dkk=krav.beløb_dkk, godkendt_beløb_dkk=krav.beløb_dkk,
                )

        self.fake = FakePipeline()
        api._pipeline = self.fake
        self.client = TestClient(api.app)

    BASE = {
        "kortniveau": "world_elite",
        "hændelse_beskrivelse": "Forsinket fly, mistet forbindelse til Nairobi",
        "hændelse_dato": "2026-05-10",
    }

    def test_poster_sum_skal_matche_total(self):
        resp = self.client.post("/api/claims", json={
            **self.BASE, "beløb_dkk": 9999,
            "poster": [{"beskrivelse": "nye billetter", "beløb_dkk": 4000}],
        })
        assert resp.status_code == 422
        assert "matcher ikke" in resp.json()["detail"]

    def test_gyldige_poster_naar_pipelinen(self):
        resp = self.client.post("/api/claims", json={
            **self.BASE, "beløb_dkk": 4000,
            "poster": [{"beskrivelse": "nye billetter", "beløb_dkk": 4000,
                        "dækningstype_hint": "flyforsinkelse"}],
        })
        assert resp.status_code == 200
        assert self.fake.sidste_krav.poster[0].dækningstype_hint == "flyforsinkelse"

    def test_fritekst_uden_poster_virker_stadig(self):
        resp = self.client.post("/api/claims", json={**self.BASE, "beløb_dkk": 4000})
        assert resp.status_code == 200
        assert self.fake.sidste_krav.poster is None
