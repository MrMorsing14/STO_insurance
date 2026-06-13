"""
Metadata-filter: Hurtige afgørelser via policy_metadata.json — UDEN LLM.

To niveauer:
1. pre_filter_krav: hele kravet (kendt kortniveau? under STO-grænsen?)
2. filter_delkrav: det enkelte delkrav (er dækningstypen overhovedet dækket på kortet?)

Dækningstyper med "dækket": false afvises øjeblikkeligt og deterministisk —
det sparer både LLM-kald og fjerner risikoen for at modellen "godkender af venlighed".
"""
import json
from typing import Optional

from config.settings import METADATA_PATH, STO_BELØB_GRÆNSE_DKK
from src.models import DelAfgørelse, DelAfgørelseType, DelKrav, ForsikringsKrav


class MetadataFilter:
    def __init__(self, metadata_path=METADATA_PATH):
        with open(metadata_path, encoding="utf-8") as f:
            self._meta = json.load(f)
        self._kortniveauer = self._meta.get("kortniveauer", {})

    # ── Niveau 1: hele kravet ──────────────────────────────────────────

    def pre_filter_krav(self, krav: ForsikringsKrav) -> Optional[str]:
        """
        Returnerer en eskaleringsgrund (str) hvis kravet IKKE kan STO-behandles,
        ellers None. Bemærk: vi afviser ikke her — vi eskalerer kun.
        """
        if self._resolve_kortniveau(krav.kortniveau.value) is None:
            return f"Ukendt kortniveau '{krav.kortniveau.value}' — kan ikke slå dækning op"

        if krav.beløb_dkk > STO_BELØB_GRÆNSE_DKK:
            return (
                f"Beløb {krav.beløb_dkk} DKK overstiger STO-grænsen "
                f"på {STO_BELØB_GRÆNSE_DKK} DKK"
            )
        return None

    # ── Niveau 2: det enkelte delkrav ──────────────────────────────────

    def filter_delkrav(self, krav: ForsikringsKrav, delkrav: DelKrav) -> Optional[DelAfgørelse]:
        """
        Returnerer en DelAfgørelse hvis delkravet kan afgøres på metadata alene:
        - dækningstype 'ukendt'        → manuelt_review
        - dækningstype ikke i metadata → manuelt_review (vores mapping kan have huller)
        - "dækket": false              → afvist (deterministisk)
        Ellers None → videre til retrieval + LLM.
        """
        if delkrav.dækningstype == "ukendt":
            return self._delafgørelse(
                delkrav,
                DelAfgørelseType.MANUELT_REVIEW,
                "Delkravet kunne ikke klassificeres til en kendt dækningstype",
            )

        coverage = self.get_coverage(krav.kortniveau.value, delkrav.dækningstype)
        if coverage is None:
            return self._delafgørelse(
                delkrav,
                DelAfgørelseType.MANUELT_REVIEW,
                f"Dækningstypen '{delkrav.dækningstype}' findes ikke i metadata for "
                f"kortniveau '{krav.kortniveau.value}'",
            )

        if not coverage.get("dækket", False):
            return self._delafgørelse(
                delkrav,
                DelAfgørelseType.AFVIST,
                f"Dækningstypen '{delkrav.dækningstype}' er ikke omfattet af "
                f"forsikringen på kortniveau '{krav.kortniveau.value}'",
            )

        return None  # dækket → LLM skal vurdere betingelserne

    # ── Kontekst til LLM ───────────────────────────────────────────────

    def get_coverage(self, kortniveau: str, dækningstype: str) -> Optional[dict]:
        kort = self._resolve_kortniveau(kortniveau)
        if kort is None:
            return None
        return kort.get("dækninger", {}).get(dækningstype)

    def get_coverage_context(self, kortniveau: str, dækningstype: str) -> dict:
        """Struktureret kontekst til LLM-evaluering af ét delkrav."""
        kort = self._resolve_kortniveau(kortniveau) or {}
        return {
            "kortniveau": kortniveau,
            "kort_info": {
                k: v for k, v in kort.items() if k != "dækninger"
            },
            "dækningstype": dækningstype,
            "dækning": kort.get("dækninger", {}).get(dækningstype, {}),
            "generelle_undtagelser": self._meta.get("generelle_undtagelser", {}),
        }

    def _resolve_kortniveau(self, kortniveau: str) -> Optional[dict]:
        """Family-varianter peger på samme dækninger som primærkortet."""
        if kortniveau in self._kortniveauer:
            return self._kortniveauer[kortniveau]
        family_map = {
            "mastercard_gold_family": "mastercard_gold",
            "mastercard_platinum_family": "mastercard_platinum",
            "world_elite_family": "world_elite",
        }
        primary = family_map.get(kortniveau)
        return self._kortniveauer.get(primary) if primary else None

    @staticmethod
    def _delafgørelse(delkrav: DelKrav, afgørelse: DelAfgørelseType, begrundelse: str) -> DelAfgørelse:
        return DelAfgørelse(
            delkrav_id=delkrav.delkrav_id,
            beskrivelse=delkrav.beskrivelse,
            dækningstype=delkrav.dækningstype,
            beløb_dkk=delkrav.beløb_dkk,
            afgørelse=afgørelse,
            begrundelse=begrundelse,
            konfidens=1.0,  # metadata-afgørelser er deterministiske
            godkendt_beløb_dkk=0.0 if afgørelse == DelAfgørelseType.AFVIST else None,
            metadata_filtreret=True,
        )
