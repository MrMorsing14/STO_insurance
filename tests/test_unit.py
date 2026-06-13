"""
Unit tests UDEN API-nøgle og UDEN vektorstore.
Mocker Mistral-klienten og tester dekomponering, metadata-filter,
aggregering og kundebesked.

Kør: python -m pytest tests/test_unit.py -v
"""
import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from src.aggregation import aggreger
from src.decomposition.claim_decomposer import ClaimDecomposer
from src.models import (
    Afgørelse,
    DelAfgørelse,
    DelAfgørelseType,
    DelKrav,
    ForsikringsKrav,
    Kortniveau,
)
from src.retrieval.metadata_filter import MetadataFilter


def lav_krav(**overrides) -> ForsikringsKrav:
    defaults = dict(
        krav_id="TEST-001",
        kortniveau=Kortniveau.PLATINUM,
        beløb_dkk=4000.0,
        hændelse_beskrivelse=(
            "Jeg faldt på ferien og måtte tage en taxa til hospitalet (400 kr), "
            "betale for behandling (2100 kr), mistede en feriedag i sengen (1000 kr), "
            "og mine bukser blev ødelagt i faldet (500 kr)."
        ),
        hændelse_dato=date(2026, 5, 10),
        rejse_startdato=date(2026, 5, 8),
        rejse_slutdato=date(2026, 5, 15),
        dokumentation="Lægeerklæring, taxakvittering, behandlingsregning",
    )
    defaults.update(overrides)
    return ForsikringsKrav(**defaults)


def mock_mistral(response_payload: dict) -> MagicMock:
    """Mistral-klient der returnerer et fast JSON-svar."""
    client = MagicMock()
    msg = MagicMock()
    msg.content = json.dumps(response_payload, ensure_ascii=False)
    client.chat.complete.return_value = MagicMock(choices=[MagicMock(message=msg)])
    return client


# ── Dekomponering ──────────────────────────────────────────────────────

class TestClaimDecomposer:
    def test_blandet_krav_splittes_i_fire_delkrav(self):
        payload = {
            "delkrav": [
                {"beskrivelse": "taxa til hospitalet", "dækningstype": "sygdom_og_hjemtransport", "beløb_dkk": 400},
                {"beskrivelse": "lægebehandling", "dækningstype": "sygdom_og_hjemtransport", "beløb_dkk": 2100},
                {"beskrivelse": "ødelagt feriedag", "dækningstype": "feriekompensation", "beløb_dkk": 1000},
                {"beskrivelse": "ødelagte bukser", "dækningstype": "bagagedækning", "beløb_dkk": 500},
            ]
        }
        decomposer = ClaimDecomposer(client=mock_mistral(payload))
        delkrav = decomposer.decompose(lav_krav())

        assert len(delkrav) == 4
        assert delkrav[0].dækningstype == "sygdom_og_hjemtransport"
        assert delkrav[2].dækningstype == "feriekompensation"
        assert delkrav[3].dækningstype == "bagagedækning"
        assert sum(d.beløb_dkk for d in delkrav) == 4000.0

    def test_ukendt_dækningstype_normaliseres_til_ukendt(self):
        payload = {"delkrav": [{"beskrivelse": "noget mystisk", "dækningstype": "rumrejseforsikring", "beløb_dkk": 4000}]}
        decomposer = ClaimDecomposer(client=mock_mistral(payload))
        delkrav = decomposer.decompose(lav_krav())
        assert delkrav[0].dækningstype == "ukendt"

    def test_beløb_mismatch_giver_fallback(self):
        payload = {
            "delkrav": [
                {"beskrivelse": "taxa", "dækningstype": "sygdom_og_hjemtransport", "beløb_dkk": 400},
                {"beskrivelse": "bukser", "dækningstype": "bagagedækning", "beløb_dkk": 500},
            ]
        }  # sum 900 ≠ total 4000 → må ikke stole på opdelingen
        decomposer = ClaimDecomposer(client=mock_mistral(payload))
        delkrav = decomposer.decompose(lav_krav())
        assert len(delkrav) == 1
        assert delkrav[0].beløb_dkk == 4000.0

    def test_manglende_beløb_er_tilladt(self):
        payload = {
            "delkrav": [
                {"beskrivelse": "taxa", "dækningstype": "sygdom_og_hjemtransport", "beløb_dkk": None},
                {"beskrivelse": "bukser", "dækningstype": "bagagedækning", "beløb_dkk": None},
            ]
        }
        decomposer = ClaimDecomposer(client=mock_mistral(payload))
        delkrav = decomposer.decompose(lav_krav())
        assert len(delkrav) == 2  # ingen sum-validering når beløb mangler

    def test_uparserbart_svar_giver_fallback(self):
        client = MagicMock()
        msg = MagicMock()
        msg.content = "Jeg er en LLM der ikke følger instruktioner!"
        client.chat.complete.return_value = MagicMock(choices=[MagicMock(message=msg)])
        decomposer = ClaimDecomposer(client=client)
        delkrav = decomposer.decompose(lav_krav())
        assert len(delkrav) == 1
        assert delkrav[0].dækningstype == "ukendt"


# ── Metadata-filter ────────────────────────────────────────────────────

class TestMetadataFilter:
    @pytest.fixture
    def mf(self):
        return MetadataFilter()

    def test_beløb_over_sto_grænse_eskaleres(self, mf):
        grund = mf.pre_filter_krav(lav_krav(beløb_dkk=12000))
        assert grund is not None and "STO-grænsen" in grund

    def test_beløb_under_grænse_passerer(self, mf):
        assert mf.pre_filter_krav(lav_krav(beløb_dkk=4000)) is None

    def test_ikke_dækket_type_afvises_uden_llm(self, mf):
        # Gold har IKKE feriekompensation → deterministisk afvisning
        krav = lav_krav(kortniveau=Kortniveau.GOLD)
        delkrav = DelKrav(delkrav_id="X-D1", beskrivelse="ødelagt feriedag",
                          dækningstype="feriekompensation", beløb_dkk=1000)
        result = mf.filter_delkrav(krav, delkrav)
        assert result is not None
        assert result.afgørelse == DelAfgørelseType.AFVIST
        assert result.metadata_filtreret

    def test_dækket_type_går_videre_til_llm(self, mf):
        # Platinum HAR feriekompensation → None = LLM skal vurdere
        krav = lav_krav(kortniveau=Kortniveau.PLATINUM)
        delkrav = DelKrav(delkrav_id="X-D1", beskrivelse="ødelagt feriedag",
                          dækningstype="feriekompensation", beløb_dkk=1000)
        assert mf.filter_delkrav(krav, delkrav) is None

    def test_ukendt_type_til_manuelt_review(self, mf):
        krav = lav_krav()
        delkrav = DelKrav(delkrav_id="X-D1", beskrivelse="?", dækningstype="ukendt", beløb_dkk=100)
        result = mf.filter_delkrav(krav, delkrav)
        assert result.afgørelse == DelAfgørelseType.MANUELT_REVIEW

    def test_family_variant_arver_dækning(self, mf):
        krav = lav_krav(kortniveau=Kortniveau.PLATINUM_FAMILY)
        delkrav = DelKrav(delkrav_id="X-D1", beskrivelse="ødelagt feriedag",
                          dækningstype="feriekompensation", beløb_dkk=1000)
        assert mf.filter_delkrav(krav, delkrav) is None  # dækket via platinum

    def test_coverage_context_indeholder_udbetalingsregler(self, mf):
        ctx = mf.get_coverage_context("mastercard_platinum", "feriekompensation")
        assert ctx["dækning"]["dækket"] is True
        assert "udbetalingsregler" in ctx["dækning"]
        assert "generelle_undtagelser" in ctx


# ── Aggregering + kundebesked ──────────────────────────────────────────

def lav_delafgørelse(id, afg, beløb=None, godkendt=None, besk="x", konfidens=0.95, meta=False):
    return DelAfgørelse(
        delkrav_id=id, beskrivelse=besk, dækningstype="sygdom_og_hjemtransport",
        beløb_dkk=beløb, afgørelse=afg, begrundelse="testbegrundelse",
        konfidens=konfidens, godkendt_beløb_dkk=godkendt, metadata_filtreret=meta,
    )


class TestAggregering:
    def test_blandet_giver_delvist_godkendt(self):
        krav = lav_krav()
        delafgørelser = [
            lav_delafgørelse("D1", DelAfgørelseType.GODKENDT, 400, 400, "taxa til hospitalet"),
            lav_delafgørelse("D2", DelAfgørelseType.GODKENDT, 2100, 2100, "lægebehandling"),
            lav_delafgørelse("D3", DelAfgørelseType.GODKENDT, 1000, 1000, "ødelagt feriedag"),
            lav_delafgørelse("D4", DelAfgørelseType.AFVIST, 500, 0.0, "ødelagte bukser"),
        ]
        resultat = aggreger(krav, delafgørelser)
        assert resultat.afgørelse == Afgørelse.DELVIST_GODKENDT
        assert resultat.godkendt_beløb_dkk == 3500.0
        assert "bukser" in resultat.kundebesked
        assert "kan desværre ikke godkende" in resultat.kundebesked

    def test_et_review_eskalerer_hele_kravet(self):
        krav = lav_krav()
        delafgørelser = [
            lav_delafgørelse("D1", DelAfgørelseType.GODKENDT, 400, 400),
            lav_delafgørelse("D2", DelAfgørelseType.MANUELT_REVIEW, 2100),
        ]
        resultat = aggreger(krav, delafgørelser)
        assert resultat.afgørelse == Afgørelse.MANUELT_REVIEW
        assert resultat.godkendt_beløb_dkk == 0.0  # ingen auto-udbetaling før review

    def test_alle_godkendt(self):
        resultat = aggreger(lav_krav(), [
            lav_delafgørelse("D1", DelAfgørelseType.GODKENDT, 4000, 4000),
        ])
        assert resultat.afgørelse == Afgørelse.GODKENDT
        assert resultat.godkendt_beløb_dkk == 4000.0

    def test_alle_afvist(self):
        resultat = aggreger(lav_krav(), [
            lav_delafgørelse("D1", DelAfgørelseType.AFVIST, 4000, 0.0),
        ])
        assert resultat.afgørelse == Afgørelse.AFVIST
        assert "afvist i sin helhed" in resultat.kundebesked

    def test_samlet_konfidens_er_laveste_llm_konfidens(self):
        resultat = aggreger(lav_krav(), [
            lav_delafgørelse("D1", DelAfgørelseType.GODKENDT, 1000, 1000, konfidens=0.97),
            lav_delafgørelse("D2", DelAfgørelseType.GODKENDT, 3000, 3000, konfidens=0.88),
            lav_delafgørelse("D3", DelAfgørelseType.AFVIST, 0, 0.0, konfidens=1.0, meta=True),
        ])
        assert resultat.konfidens == 0.88  # metadata-afgørelser tæller ikke med


# ── Fuld pipeline med mocks (uden vektorstore og uden ægte LLM) ────────

class TestPipelineMocked:
    def test_bukse_scenariet_ende_til_ende(self):
        """Platinum-kunde: taxa + behandling + feriedag godkendes, bukser afvises."""
        from src.pipeline import STOPipeline

        decomp_payload = {
            "delkrav": [
                {"beskrivelse": "taxa til hospitalet", "dækningstype": "sygdom_og_hjemtransport", "beløb_dkk": 400},
                {"beskrivelse": "lægebehandling", "dækningstype": "sygdom_og_hjemtransport", "beløb_dkk": 2100},
                {"beskrivelse": "ødelagt feriedag", "dækningstype": "feriekompensation", "beløb_dkk": 1000},
                {"beskrivelse": "ødelagte bukser", "dækningstype": "bagagedækning", "beløb_dkk": 500},
            ]
        }
        eval_responses = {
            "taxa til hospitalet": {"afgørelse": "godkendt", "begrundelse": "Transport til behandling er dækket", "konfidens": 0.95, "relevante_betingelser": [], "godkendt_beløb_dkk": 400},
            "lægebehandling": {"afgørelse": "godkendt", "begrundelse": "Akut behandling er dækket", "konfidens": 0.95, "relevante_betingelser": [], "godkendt_beløb_dkk": 2100},
            "ødelagt feriedag": {"afgørelse": "godkendt", "begrundelse": "Sengeleje dokumenteret", "konfidens": 0.9, "relevante_betingelser": [], "godkendt_beløb_dkk": 1000},
            "ødelagte bukser": {"afgørelse": "afvist", "begrundelse": "Beskadigede ejendele under ulykke er ikke omfattet — henvises til indboforsikring", "konfidens": 0.92, "relevante_betingelser": [], "godkendt_beløb_dkk": 0},
        }

        from src.decomposition.claim_decomposer import ClaimDecomposer
        from src.evaluation.llm_evaluator import LLMEvaluator

        decomposer = ClaimDecomposer(client=mock_mistral(decomp_payload))

        eval_client = MagicMock()
        def eval_side_effect(model, messages, temperature):
            user_msg = messages[1]["content"]
            for key, payload in eval_responses.items():
                # Match KUN på delkrav-sektionen, ikke den fælles hændelsesbeskrivelse
                if f"- Beskrivelse: {key}" in user_msg:
                    msg = MagicMock(); msg.content = json.dumps(payload, ensure_ascii=False)
                    return MagicMock(choices=[MagicMock(message=msg)])
            raise AssertionError("Ukendt delkrav i prompt")
        eval_client.chat.complete.side_effect = eval_side_effect
        evaluator = LLMEvaluator(client=eval_client)

        vector_store = MagicMock()
        vector_store.search.return_value = [{"text": "dummy chunk", "metadata": {"sektion": "4.0"}}]

        pipeline = STOPipeline(
            metadata_filter=MetadataFilter(),
            vector_store=vector_store,
            decomposer=decomposer,
            llm_evaluator=evaluator,
        )

        resultat = pipeline.process_claim(lav_krav())

        assert resultat.afgørelse == Afgørelse.DELVIST_GODKENDT
        assert resultat.godkendt_beløb_dkk == 3500.0
        assert len(resultat.delafgørelser) == 4
        assert "bukser" in resultat.kundebesked
        assert "indboforsikring" in resultat.kundebesked

    def test_gold_kunde_får_feriedag_afvist_af_metadata(self):
        """Samme krav på GOLD: feriekompensation + bagagedækning afvises uden LLM."""
        from src.pipeline import STOPipeline

        decomp_payload = {
            "delkrav": [
                {"beskrivelse": "taxa til hospitalet", "dækningstype": "sygdom_og_hjemtransport", "beløb_dkk": 2500},
                {"beskrivelse": "ødelagt feriedag", "dækningstype": "feriekompensation", "beløb_dkk": 1000},
                {"beskrivelse": "ødelagte bukser", "dækningstype": "bagagedækning", "beløb_dkk": 500},
            ]
        }
        eval_payload = {"afgørelse": "godkendt", "begrundelse": "Dækket", "konfidens": 0.95,
                        "relevante_betingelser": [], "godkendt_beløb_dkk": 2500}

        from src.decomposition.claim_decomposer import ClaimDecomposer
        from src.evaluation.llm_evaluator import LLMEvaluator

        vector_store = MagicMock()
        vector_store.search.return_value = []

        pipeline = STOPipeline(
            metadata_filter=MetadataFilter(),
            vector_store=vector_store,
            decomposer=ClaimDecomposer(client=mock_mistral(decomp_payload)),
            llm_evaluator=LLMEvaluator(client=mock_mistral(eval_payload)),
        )

        resultat = pipeline.process_claim(lav_krav(kortniveau=Kortniveau.GOLD))

        assert resultat.afgørelse == Afgørelse.DELVIST_GODKENDT
        assert resultat.godkendt_beløb_dkk == 2500.0
        meta_afviste = [d for d in resultat.delafgørelser if d.metadata_filtreret and d.afgørelse == DelAfgørelseType.AFVIST]
        assert len(meta_afviste) == 2  # feriedag + bukser afvist UDEN LLM-kald
