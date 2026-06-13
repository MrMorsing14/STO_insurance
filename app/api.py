"""
FastAPI-app: Eksponerer STO-pipelinen som REST API og serverer frontenden.

Kør:  uvicorn app.api:app --reload
Åbn:  http://127.0.0.1:8000

Endpoints:
  POST /api/claims   — behandl et forsikringskrav
  GET  /api/health   — status (pipeline klar? chunks indekseret?)
  GET  /              — frontend (statisk)
"""
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models import ForsikringsKrav, Kortniveau, KravPost

app = FastAPI(title="STO Prototype API", version="2.0")

# Pipeline initialiseres lazy ved første request — embedderen tager
# nogle sekunder at loade, og vi vil ikke blokere uvicorn-opstart.
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.pipeline import STOPipeline
        _pipeline = STOPipeline()
    return _pipeline


# ── Request/response-modeller ──────────────────────────────────────────

class PostInput(BaseModel):
    beskrivelse: str = Field(min_length=3)
    beløb_dkk: float = Field(gt=0)
    dækningstype_hint: Optional[str] = None


class ClaimRequest(BaseModel):
    kortniveau: Kortniveau
    dækningstype: Optional[str] = Field(default=None, description="Valgfrit hint — kun brugt i fritekst-mode")
    beløb_dkk: float = Field(ge=0, description="Samlet beløb. Med poster: skal matche summen af posterne")
    hændelse_beskrivelse: str = Field(min_length=10)
    poster: Optional[list[PostInput]] = Field(
        default=None, max_length=15,
        description="Itemiserede udgiftsposter (foretrukket). Uden poster bruges fritekst-dekomponering",
    )
    hændelse_dato: date
    rejse_startdato: Optional[date] = None
    rejse_slutdato: Optional[date] = None
    dokumentation: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Tjek om pipelinen kan initialiseres og vektorstoren har indhold."""
    try:
        pipeline = get_pipeline()
        chunks = pipeline.vector_store.chunk_count()
        return {
            "status": "ok" if chunks > 0 else "vektorstore_tom",
            "chunks_indekseret": chunks,
        }
    except ValueError as e:  # typisk manglende API-nøgle
        return {"status": "fejl", "detalje": str(e)}


@app.post("/api/claims")
def process_claim(req: ClaimRequest):
    # Med itemiserede poster skal summen matche totalbeløbet —
    # beløbsfordelingen er kundens ansvar, ikke modellens
    if req.poster:
        post_sum = sum(p.beløb_dkk for p in req.poster)
        if abs(post_sum - req.beløb_dkk) > 1.0:
            raise HTTPException(
                status_code=422,
                detail=f"Summen af poster ({post_sum} DKK) matcher ikke det samlede beløb ({req.beløb_dkk} DKK)",
            )

    krav = ForsikringsKrav(
        krav_id=f"STO-{uuid.uuid4().hex[:8].upper()}",
        kortniveau=req.kortniveau,
        dækningstype=req.dækningstype or None,
        beløb_dkk=req.beløb_dkk,
        hændelse_beskrivelse=req.hændelse_beskrivelse,
        poster=[KravPost(**p.model_dump()) for p in req.poster] if req.poster else None,
        hændelse_dato=req.hændelse_dato,
        rejse_startdato=req.rejse_startdato,
        rejse_slutdato=req.rejse_slutdato,
        dokumentation=req.dokumentation or None,
    )

    try:
        pipeline = get_pipeline()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    start = time.monotonic()
    try:
        resultat = pipeline.process_claim(krav)
    except Exception as e:
        # Ved uventede fejl må vi ALDRIG gætte en afgørelse
        raise HTTPException(
            status_code=500,
            detail=f"Pipelinefejl ({type(e).__name__}): {str(e)[:300]}",
        )

    payload = resultat.model_dump(mode="json")
    payload["behandlingstid_sek"] = round(time.monotonic() - start, 1)
    return payload


# ── Statisk frontend (mountes sidst så /api/* har forrang) ─────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
