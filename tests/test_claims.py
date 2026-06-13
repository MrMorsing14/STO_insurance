"""
Integrationstest med RIGTIG Mistral API + indekseret vektorstore.

Forudsætninger:
  1. .env med MISTRAL_API_KEY
  2. python -m src.ingestion.ingest_policies (indekserede policydokumenter)

Kør: python -m tests.test_claims
"""
from datetime import date

from src.models import ForsikringsKrav, Kortniveau, KravPost
from src.pipeline import STOPipeline

SCENARIER = [
    # Det centrale scenarie: blandet krav på Platinum
    ForsikringsKrav(
        krav_id="KRAV-001",
        kortniveau=Kortniveau.PLATINUM,
        beløb_dkk=4000.0,
        hændelse_beskrivelse=(
            "Jeg faldt på trappen på mit hotel i Spanien og slog mit knæ. "
            "Jeg tog en taxa til hospitalet (400 kr) og betalte for behandling (2100 kr). "
            "Lægen ordinerede sengeleje, så jeg mistede en feriedag (1000 kr). "
            "Mine bukser blev desuden flænset i faldet (500 kr)."
        ),
        hændelse_dato=date(2026, 5, 10),
        rejse_startdato=date(2026, 5, 8),
        rejse_slutdato=date(2026, 5, 15),
        dokumentation="Lægeerklæring med ordineret sengeleje, taxakvittering, behandlingsregning, foto af bukser",
    ),
    # Samme historie på Gold — feriekompensation og bagagedækning er IKKE dækket på Gold
    ForsikringsKrav(
        krav_id="KRAV-002",
        kortniveau=Kortniveau.GOLD,
        beløb_dkk=4000.0,
        hændelse_beskrivelse=(
            "Jeg faldt på trappen på mit hotel i Spanien og slog mit knæ. "
            "Taxa til hospitalet 400 kr, behandling 2100 kr, en mistet feriedag 1000 kr, "
            "og mine bukser blev ødelagt, 500 kr."
        ),
        hændelse_dato=date(2026, 5, 10),
        rejse_startdato=date(2026, 5, 8),
        rejse_slutdato=date(2026, 5, 15),
        dokumentation="Lægeerklæring, taxakvittering, behandlingsregning",
    ),
    # Simpelt enkelt-krav: bagageforsinkelse på World Elite
    ForsikringsKrav(
        krav_id="KRAV-003",
        kortniveau=Kortniveau.WORLD_ELITE,
        beløb_dkk=1800.0,
        hændelse_beskrivelse=(
            "Min indskrevne kuffert var 9 timer forsinket ved ankomst til New York. "
            "Jeg købte nødvendigt tøj og toiletartikler for 1800 kr."
        ),
        hændelse_dato=date(2026, 4, 2),
        rejse_startdato=date(2026, 4, 2),
        rejse_slutdato=date(2026, 4, 9),
        dokumentation="PIR-rapport fra luftfartsselskabet, kvitteringer for indkøb",
    ),
    # Over STO-grænsen → skal eskaleres uden LLM-kald
    ForsikringsKrav(
        krav_id="KRAV-004",
        kortniveau=Kortniveau.PLATINUM,
        beløb_dkk=14500.0,
        hændelse_beskrivelse="Hospitalsindlæggelse i USA efter cykeluheld.",
        hændelse_dato=date(2026, 3, 1),
        dokumentation="Hospitalsregning",
    ),
    # Bagageforsinkelse på HJEMREJSEN → undtagelse, bør afvises
    ForsikringsKrav(
        krav_id="KRAV-005",
        kortniveau=Kortniveau.PLATINUM,
        beløb_dkk=900.0,
        hændelse_beskrivelse=(
            "Min kuffert var 6 timer forsinket da jeg landede i Kastrup på vej HJEM "
            "fra ferie. Jeg købte tøj for 900 kr."
        ),
        hændelse_dato=date(2026, 5, 20),
        rejse_startdato=date(2026, 5, 13),
        rejse_slutdato=date(2026, 5, 20),
        dokumentation="PIR-rapport, kvitteringer",
    ),
    # "Buddy-casen": itemiseret post med FORKERT kunde-hint (flyforsinkelse).
    # Korrekt udfald: forsinket_fremmøde → godkendt — enten via klassificeringen
    # eller via omklassificerings-sikkerhedsnettet.
    ForsikringsKrav(
        krav_id="KRAV-006",
        kortniveau=Kortniveau.WORLD_ELITE,
        beløb_dkk=4000.0,
        hændelse_beskrivelse=(
            "Vores fly fra Billund til København var over 3 timer forsinket. "
            "Vi nåede derfor ikke vores separate fly fra København til Nairobi "
            "og måtte købe nye billetter for at komme videre på rejsen."
        ),
        poster=[
            KravPost(
                beskrivelse="Nye flybilletter København til Nairobi efter mistet forbindelse",
                beløb_dkk=4000.0,
                dækningstype_hint="flyforsinkelse",  # bevidst forkert hint
            )
        ],
        hændelse_dato=date(2026, 5, 10),
        rejse_startdato=date(2026, 5, 10),
        rejse_slutdato=date(2026, 5, 24),
        dokumentation="Boardingkort, forsinkelsesbekræftelse fra flyselskabet, kvittering for nye billetter",
    ),
]


def main():
    pipeline = STOPipeline()
    for krav in SCENARIER:
        resultat = pipeline.process_claim(krav)
        print(f"\n  Kundebesked:\n{_indent(resultat.kundebesked)}")
        print(f"\n{'═' * 60}")


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


if __name__ == "__main__":
    main()
