"""
Policy-ingestion: Indlæser forsikringsbetingelser (.pdf ELLER .md),
chunker efter sektioner, tagger med metadata, og gemmer i ChromaDB.

Kør: python -m src.ingestion.ingest_policies

Vigtige fixes ift. tidligere version:
1. SEKTIONS-KONTEKST: Sektion A og Sektion B genbruger sektionsnumre
   (A har "4.0 Krigs- og atomskader", B har "4.0 Sygdom og hjemtransport").
   Vi tracker nu hvilken sektion (A/B/C/D) en chunk tilhører:
     - Sektion A  → dækningstype "generelt"
     - Sektion B  → opslag i SEKTION_DÆKNING_MAP
     - Sektion C/D → "købsforsikring" (uden for rejse-scope, men tagget korrekt)
2. MARKDOWN-SUPPORT: .md-filer (fx fra PDF-konvertering) har overskrifter
   som "## **4.0 Sygdom og hjemtransport**" — regexen håndterer begge formater.
"""
import re
from pathlib import Path

from config.settings import POLICY_DIR

# Mapping: filnavn (uden extension) → hvilke kortniveauer dokumentet dækker
FIL_KORTNIVEAU_MAP = {
    "Mastercard_forsikringsbetingelser": [
        "mastercard_blue_shopping",
        "mastercard_gold",
        "mastercard_platinum",
    ],
    "World_Elite_forsikringsbetingelser": [
        "world_elite",
    ],
    "Mastercard_Business_forsikringsbetingelser": [
        "mastercard_business",
        "mastercard_business_platinum",
    ],
}

# Mapping: sektionsnumre i SEKTION B → dækningstype
# Kun hovedsektionen (fx "13") er nødvendig — undersektioner arver fra parent.
SEKTION_DÆKNING_MAP = {
    "gold_platinum": {
        "4": "sygdom_og_hjemtransport",
        "5": "sygeledsagelse",
        "6": "tilkaldelse",
        "7": "hjemkaldelse",
        "8": "privatansvarsforsikring",
        "9": "retshjælp_og_sikkerhedsstillelse",
        "10": "afbestillingsforsikring",
        "11": "overfald",
        "12": "rejseulykke",
        "13": "bagageforsinkelse",
        "14": "flyforsinkelse",
        "15": "forsinket_fremmøde",
        "16": "eftersøgning_og_redning",
        "17": "evakuering_og_ufrivilligt_ophold",
        "18": "bagagedækning",
        "19": "feriekompensation",
        "20": "forsikring_ved_billeje",
    },
    "world_elite": {
        "4": "sygdom_og_hjemtransport",
        "5": "krisehjælp",
        "6": "sygeledsagelse",
        "7": "tilkaldelse",
        "8": "hjemkaldelse",
        "9": "flyforsinkelse",
        "10": "forsinket_fremmøde",
        "11": "bagageforsinkelse",
        "12": "bagagedækning",
        "13": "privatansvarsforsikring",
        "14": "ferieboligsikring",
        "15": "retshjælp_og_sikkerhedsstillelse",
        "16": "overfald",
        "17": "forsikring_ved_billeje",
        "18": "feriekompensation_og_erstatningsrejse",
        "19": "eftersøgning_og_redning",
        "20": "evakuering_og_ufrivilligt_ophold",
        "21": "afbestillingsforsikring",
    },
    "business": {
        "4": "afbestillingsforsikring",
        "5": "sygdom_og_hjemtransport",
        "6": "sygeledsagelse",
        "7": "tilkaldelse",
        "8": "hjemkaldelse",
        "9": "flyforsinkelse",
        "10": "forsinket_fremmøde",
        "11": "bagageforsinkelse",
        "12": "bagagedækning",
        "13": "privatansvarsforsikring",
        "14": "retshjælp_og_sikkerhedsstillelse",
        "15": "overfald",
        "16": "forsikring_ved_billeje",
        "17": "rejseulykke",
    },
}

# Overskrift med sektionsnummer. Matcher både rå PDF-tekst ("4.0 Sygdom ...")
# og markdown ("## **4.0 Sygdom ...**" / "### 4.0 Sygdom ...")
SECTION_PATTERN = re.compile(
    r"^(?:#{1,6}\s*)?\**\s*(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s+([^\n*]+)",
    re.MULTILINE,
)

# Overskrifter der skifter sektions-kontekst (A/B/C/D)
SEKTION_KONTEKST_PATTERN = re.compile(
    r"^(?:#{1,6}\s*)?\**\s*SEKTION\s+([A-D])\b",
    re.MULTILINE | re.IGNORECASE,
)


def extract_text(path: Path) -> str:
    """Udtræk tekst fra .pdf (PyMuPDF) eller .md/.txt (rå tekst)."""
    if path.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF — kun importeret hvis der faktisk er PDF'er

        doc = fitz.open(str(path))
        full_text = "".join(page.get_text() + "\n" for page in doc)
        doc.close()
        return full_text
    return path.read_text(encoding="utf-8")


def find_sektion_kontekster(text: str) -> list[tuple[int, str]]:
    """Returnerer [(position, 'A'|'B'|'C'|'D'), ...] sorteret efter position."""
    return [(m.start(), m.group(1).upper()) for m in SEKTION_KONTEKST_PATTERN.finditer(text)]


def sektion_kontekst_ved(position: int, kontekster: list[tuple[int, str]]) -> str:
    """Hvilken sektion (A/B/C/D) gælder ved en given tekstposition?"""
    gældende = "A"  # alt før første markør behandles som generelt
    for pos, sektion in kontekster:
        if pos <= position:
            gældende = sektion
        else:
            break
    return gældende


def chunk_by_sections(text: str) -> list[dict]:
    """Split tekst i chunks pr. nummereret sektion, med sektions-kontekst (A/B/C/D)."""
    kontekster = find_sektion_kontekster(text)
    matches = list(SECTION_PATTERN.finditer(text))

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        if len(section_text) < 50:  # tomme headers
            continue

        sections.append({
            "sektion_nr": match.group(1),
            "sektion_titel": match.group(2).strip(),
            "sektion_kontekst": sektion_kontekst_ved(start, kontekster),
            "text": section_text,
        })
    return sections


def determine_gruppe(stem: str) -> str:
    lower = stem.lower()
    if "business" in lower:
        return "business"
    if "world_elite" in lower or "world elite" in lower:
        return "world_elite"
    return "gold_platinum"


def find_dækningstype(sektion_nr: str, sektion_kontekst: str, gruppe: str) -> str:
    """
    Sektion A → generelt. Sektion C/D → købsforsikring.
    Sektion B → opslag på hovedsektionsnummeret ("13.2" → "13").
    """
    if sektion_kontekst == "A":
        return "generelt"
    if sektion_kontekst in ("C", "D"):
        return "købsforsikring"

    hovednr = sektion_nr.split(".")[0]
    return SEKTION_DÆKNING_MAP.get(gruppe, {}).get(hovednr, "generelt")


def tag_chunk_with_metadata(
    chunk: dict, filename: str, kortniveauer: list[str], gruppe: str, chunk_index: int
) -> list[dict]:
    """Én chunk-entry pr. kortniveau, så vektorsøgning kan filtrere skarpt."""
    dækningstype = find_dækningstype(
        chunk["sektion_nr"], chunk["sektion_kontekst"], gruppe
    )

    return [
        {
            "id": f"{filename}_{chunk_index:04d}_{chunk['sektion_nr']}_{niveau}",
            "text": chunk["text"],
            "metadata": {
                "kortniveau": niveau,
                "dækningstype": dækningstype,
                "sektion": chunk["sektion_nr"],
                "sektion_kontekst": chunk["sektion_kontekst"],
                "sektion_titel": chunk["sektion_titel"][:100],
                "kilde": filename,
            },
        }
        for niveau in kortniveauer
    ]


def ingest_all(reset: bool = True):
    from src.retrieval.vector_store import PolicyVectorStore  # lazy: tung dependency

    policy_dir = Path(POLICY_DIR)
    if not policy_dir.exists():
        print(f"FEJL: Policy-mappen '{policy_dir}' findes ikke.")
        return

    files = sorted(
        p for p in policy_dir.iterdir() if p.suffix.lower() in (".pdf", ".md")
    )
    if not files:
        print(f"FEJL: Ingen .pdf eller .md filer fundet i '{policy_dir}'")
        return

    store = PolicyVectorStore()
    if reset:
        store.reset()
        print("Vektorstore nulstillet.")

    for path in files:
        stem = path.stem
        print(f"\n{'─' * 50}\nIndlæser: {path.name}")

        kortniveauer = FIL_KORTNIVEAU_MAP.get(stem)
        if not kortniveauer:
            print(f"  ADVARSEL: Ukendt fil, springer over: {path.name}")
            continue

        gruppe = determine_gruppe(stem)
        print(f"  Gruppe: {gruppe} | Kortniveauer: {kortniveauer}")

        text = extract_text(path)
        sections = chunk_by_sections(text)
        print(f"  Sektioner fundet: {len(sections)}")

        all_tagged = []
        for idx, section in enumerate(sections):
            all_tagged.extend(
                tag_chunk_with_metadata(section, path.name, kortniveauer, gruppe, idx)
            )

        added = store.add_chunks(all_tagged)
        print(f"  Chunks tilføjet til ChromaDB: {added}")

    print(f"\n{'=' * 50}\nDONE! Total chunks i vektorstore: {store.chunk_count()}")


if __name__ == "__main__":
    ingest_all()
