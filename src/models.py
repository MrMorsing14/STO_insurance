"""
Datamodeller for forsikringskrav og afgørelser.

Nyt ift. tidligere version:
- DelKrav / DelAfgørelse: et krav dekomponeres i delkrav, som vurderes individuelt
- Afgørelse.DELVIST_GODKENDT: nogle delkrav godkendt, andre afvist
- KravAfgørelse indeholder nu itemiseret breakdown + kundebesked
"""
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Kortniveau(str, Enum):
    """De kortniveauer vi understøtter."""
    BLUE_SHOPPING = "mastercard_blue_shopping"
    GOLD = "mastercard_gold"
    GOLD_FAMILY = "mastercard_gold_family"
    PLATINUM = "mastercard_platinum"
    PLATINUM_FAMILY = "mastercard_platinum_family"
    WORLD_ELITE = "world_elite"
    WORLD_ELITE_FAMILY = "world_elite_family"
    BUSINESS = "mastercard_business"
    BUSINESS_PLATINUM = "mastercard_business_platinum"


class Afgørelse(str, Enum):
    """Mulige afgørelser for et samlet krav."""
    GODKENDT = "godkendt"
    DELVIST_GODKENDT = "delvist_godkendt"
    AFVIST = "afvist"
    MANUELT_REVIEW = "manuelt_review"


class DelAfgørelseType(str, Enum):
    """Mulige afgørelser for et enkelt delkrav."""
    GODKENDT = "godkendt"
    AFVIST = "afvist"
    MANUELT_REVIEW = "manuelt_review"


class KravPost(BaseModel):
    """Én udgiftspost angivet af kunden i den itemiserede formular."""
    beskrivelse: str = Field(min_length=3, description="Hvad udgiften dækker")
    beløb_dkk: float = Field(ge=0)
    dækningstype_hint: Optional[str] = Field(
        default=None,
        description="Kundens valgte type — KUN et hint, klassificeringen er ikke bundet af det",
    )


class ForsikringsKrav(BaseModel):
    """Et indkommende forsikringskrav."""
    krav_id: str = Field(description="Unikt ID for kravet")
    kortniveau: Kortniveau
    dækningstype: Optional[str] = Field(
        default=None,
        description="Kundens angivne primære dækningstype (kun et hint — dekomponeringen afgør de faktiske typer)",
    )
    beløb_dkk: float = Field(ge=0, description="Kravets samlede beløb i DKK")
    hændelse_beskrivelse: str = Field(description="Kundens beskrivelse af hændelsen")
    poster: Optional[list[KravPost]] = Field(
        default=None,
        description="Itemiserede udgiftsposter. Hvis sat, klassificeres disse direkte i stedet for fritekst-dekomponering",
    )
    hændelse_dato: date
    rejse_startdato: Optional[date] = None
    rejse_slutdato: Optional[date] = None
    dokumentation: Optional[str] = Field(default=None, description="Beskrivelse af vedlagt dokumentation")


class DelKrav(BaseModel):
    """Ét delkrav udskilt fra det samlede krav (fx 'taxa til hospital, 400 DKK')."""
    delkrav_id: str
    beskrivelse: str = Field(description="Hvad delkravet dækker, med kundens egne ord")
    dækningstype: str = Field(description="Kanonisk dækningstype, eller 'ukendt' hvis ingen passer")
    beløb_dkk: Optional[float] = Field(
        default=None,
        description="Beløb for dette delkrav, hvis det kan udledes af beskrivelsen. None = ikke specificeret",
    )


class DelAfgørelse(BaseModel):
    """Afgørelse for ét delkrav."""
    delkrav_id: str
    beskrivelse: str
    dækningstype: str
    beløb_dkk: Optional[float] = None
    afgørelse: DelAfgørelseType
    begrundelse: str
    goodwill: bool = False
    konfidens: float = Field(ge=0.0, le=1.0)
    godkendt_beløb_dkk: Optional[float] = Field(
        default=None, description="Beløb der godkendes for delkravet (kan være lavere end ansøgt)"
    )
    relevante_betingelser: list[str] = Field(default_factory=list)
    forslag_til_anden_dækningstype: Optional[str] = Field(
        default=None,
        description="Evaluatorens forslag hvis udgiften synes at høre under en ANDEN dækningstype",
    )
    omklassificeret_fra: Optional[str] = Field(
        default=None,
        description="Sat hvis delkravet blev re-evalueret under en ny dækningstype",
    )
    metadata_filtreret: bool = Field(
        default=False, description="Om delkravet blev afgjort via metadata alene (uden LLM)"
    )


class KravAfgørelse(BaseModel):
    """Samlet resultat for et krav, aggregeret fra delafgørelser."""
    krav_id: str
    afgørelse: Afgørelse
    begrundelse: str = Field(description="Intern begrundelse for den samlede afgørelse")
    kundebesked: str = Field(default="", description="Kundevendt besked, deterministisk genereret")
    konfidens: float = Field(ge=0.0, le=1.0, description="Laveste konfidens blandt LLM-vurderede delkrav")
    delafgørelser: list[DelAfgørelse] = Field(default_factory=list)
    ansøgt_beløb_dkk: float = 0.0
    godkendt_beløb_dkk: float = 0.0
    metadata_filtreret: bool = Field(default=False, description="Om hele kravet blev afgjort via metadata alene")
