# STO Prototype v2 — Automatisk Forsikringsskadebehandling med dekomponering

AI-baseret Straight-Through Processing (STO) for Nykredit Mastercard forsikringskrav under 5.000 DKK.

**Nyt i v2:** Krav dekomponeres i delkrav, som vurderes individuelt. Et blandet krav
("taxa + behandling + ødelagt feriedag + ødelagte bukser") kan nu blive **delvist godkendt**
med en itemiseret kundebesked — i stedet for én alt-eller-intet afgørelse.

## Arkitektur

```
Krav ind
   │
   ▼
┌────────────────────┐
│  Pre-filter         │ ← policy_metadata.json
│  (hele kravet)      │   Kendt kortniveau? Under STO-grænsen (5.000 DKK)?
└─────────┬──────────┘
          │ passerer
          ▼
┌────────────────────┐
│  Dekomponering      │ ← Mistral API
│  (krav → delkrav)   │   "taxa, behandling, feriedag, bukser" → 4 delkrav
│                     │   med hver sin dækningstype + beløb
└─────────┬──────────┘
          │ pr. delkrav
          ▼
┌────────────────────┐
│  Metadata-filter    │   "dækket": false → AFVIST uden LLM
│  (pr. delkrav)      │   "ukendt" type   → manuelt review
└─────────┬──────────┘
          │ dækket
          ▼
┌────────────────────┐
│  Vector Search      │ ← ChromaDB + multilingual-e5-small
│  (pr. delkrav)      │   Chunks filtreret på kortniveau + delkravets dækningstype
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  LLM Evaluering     │ ← Mistral API
│  (pr. delkrav)      │   Betingelser, undtagelser, udbetalingsregler, beløb
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ Confidence Routing  │   Lav konfidens → delkrav eskaleres til manuelt review
│  (pr. delkrav)      │
└─────────┬──────────┘
          ▼
┌────────────────────┐
│  Aggregering        │   godkendt / delvist_godkendt / afvist / manuelt_review
│  + Kundebesked      │   Deterministisk template — ingen LLM i kundeteksten
└────────────────────┘
```

## Aggregeringsregler

| Delafgørelser | Samlet afgørelse |
|---|---|
| Mindst ét delkrav i manuelt review | `manuelt_review` (hele kravet — ingen delvis auto-udbetaling) |
| Alle godkendt | `godkendt` |
| Alle afvist | `afvist` |
| Blandet godkendt/afvist | `delvist_godkendt` + itemiseret kundebesked |

Rationale for review-reglen: en sagsbehandler skal alligevel røre sagen, og to
separate svar på samme krav (auto-udbetaling + senere manuel afgørelse) forvirrer
kunden. Den itemiserede analyse følger med til sagsbehandleren.

## Fail-safes

- **Dekomponering:** Hvis LLM-svaret ikke kan parses, eller summen af delbeløb
  afviger fra totalbeløbet, falder vi tilbage til ét samlet delkrav → typisk manuelt review.
- **Kundebesked:** Bygges af en deterministisk template, ikke en LLM. Begrundelser
  citeres fra evalueringen, men strukturen kan ikke hallucinere.
- **Godkendt uden beløb:** Et delkrav uden specificeret beløb kan aldrig auto-godkendes.

## Quick Start

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
pip install -r requirements.txt
cp .env.example .env            # indsæt din Mistral API-nøgle
```

### Læg policydokumenter i data/policies/
Ingesteren accepterer **både .pdf og .md** — markdown-konverterede betingelser
virker direkte (overskrifter som `## **4.0 Sygdom og hjemtransport**` parses).

### Indeksér
```bash
python -m src.ingestion.ingest_policies
```

### Kør tests
```bash
# Unit tests — kræver INGEN API-nøgle og INGEN vektorstore (LLM mockes)
python -m pytest tests/test_unit.py -v

# Fuld integrationstest — kræver Mistral API-nøgle + indekseret vektorstore
python -m tests.test_claims
```

### Start app'en (API + frontend)
```bash
uvicorn app.api:app --reload
```
Åbn http://127.0.0.1:8000 — frontenden serveres af FastAPI, så ingen CORS-bøvl.
Tjek http://127.0.0.1:8000/api/health for at se om vektorstoren er indekseret.

Første request er langsom (embedderen loades lazy). Et 4-delt krav tager
typisk 15-40 sekunder, da hvert delkrav evalueres med sit eget LLM-kald.

## Projektstruktur

```
sto-prototype/
├── app/api.py                    # FastAPI: POST /api/claims + serverer frontend
├── frontend/                     # index.html, style.css, app.js (vanilla)
├── config/settings.py            # Thresholds, paths, modelnavne
├── data/
│   ├── policies/                 # Forsikringsbetingelser (.pdf eller .md)
│   ├── policy_metadata.json      # Struktureret dækningsoversigt pr. kortniveau
│   └── chroma_db/                # Vektorstore (genereres ved indeksering)
├── src/
│   ├── models.py                 # ForsikringsKrav, DelKrav, DelAfgørelse, KravAfgørelse
│   ├── pipeline.py               # Orkestrering af hele flowet
│   ├── aggregation.py            # Delafgørelser → samlet afgørelse + kundebesked
│   ├── decomposition/
│   │   └── claim_decomposer.py   # NYT: krav → delkrav (LLM + fail-safes)
│   ├── ingestion/
│   │   └── ingest_policies.py    # PDF/MD → chunks → ChromaDB (m. Sektion A/B-fix)
│   ├── retrieval/
│   │   ├── metadata_filter.py    # Pre-LLM filter, nu pr. delkrav
│   │   └── vector_store.py       # ChromaDB wrapper (e5 query/passage-prefixes)
│   └── evaluation/
│       └── llm_evaluator.py      # Mistral-vurdering af ÉT delkrav ad gangen
├── tests/
│   ├── test_unit.py              # Mocked tests (ingen API-nøgle nødvendig)
│   └── test_claims.py            # Integrationsscenarier (kræver API-nøgle)
├── requirements.txt
└── .env.example
```

## Kendte begrænsninger / næste skridt

- [ ] LLM-kald pr. delkrav er sekventielle — kan paralleliseres med asyncio
- [ ] Feriekompensations-beregning (dagpris × ødelagte døgn) bør være deterministisk kode, ikke LLM
- [ ] Historiske testdata med kendte udfald → mål STO-rate og fejlrate
- [ ] FastAPI endpoint for integration
- [ ] Audit-log af alle LLM-prompts/-svar (compliance-krav ved rigtig drift)
