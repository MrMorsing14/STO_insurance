"""
FastAPI-app: HTTP-interface til STO-pipelinen + serverer frontenden.

Kør:  uvicorn src.api:app --reload
Åbn:  http://localhost:8000

Bemærk: Pipelinen initialiseres ved opstart (embedding-modellen indlæses),
så første request er hurtig. process_claim er synkron og blokkerer i
10-60 sekunder (dekomponering + ét LLM-kald pr. delkrav) — FastAPI kører
den i en threadpool, så serveren forbliver responsiv.
"""
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.models import ForsikringsKrav, Kortniveau, KravAfgørelse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialisér pipelinen ved opstart, så embedding-modellen er varm."""
    global _pipeline
    from src.pipeline import STOPipeline
    print("Initialiserer STO-pipeline (indlæser embedding-model)…")
    _pipeline = STOPipeline()
    chunks = _pipeline.vector_store.chunk_count()
    print(f"Klar. Vektorstore indeholder {chunks} chunks.")
    if chunks == 0:
        print("ADVARSEL: Vektorstoren er tom — kør 'python -m src.ingestion.ingest_policies' først.")
    yield


app = FastAPI(title="STO Prototype", lifespan=lifespan)


class KravRequest(BaseModel):
    """Indkommende krav fra frontenden — krav_id genereres server-side."""
    kortniveau: Kortniveau
    dækningstype: Optional[str] = Field(default=None, description="Valgfrit hint fra kunden")
    beløb_dkk: float = Field(ge=0)
    hændelse_beskrivelse: str = Field(min_length=10)
    hændelse_dato: date
    rejse_startdato: Optional[date] = None
    rejse_slutdato: Optional[date] = None
    dokumentation: Optional[str] = None


@app.post("/api/krav", response_model=KravAfgørelse)
def behandl_krav(req: KravRequest) -> KravAfgørelse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipelinen er ikke initialiseret endnu")

    krav = ForsikringsKrav(
        krav_id=f"STO-{uuid.uuid4().hex[:8].upper()}",
        **req.model_dump(),
    )
    try:
        return _pipeline.process_claim(krav)
    except Exception as e:
        # Aldrig en rå stacktrace til klienten — men log den server-side
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Behandlingsfejl: {type(e).__name__}")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "chunks_i_vektorstore": _pipeline.vector_store.chunk_count() if _pipeline else 0,
    }


# Statisk frontend — mountes til sidst så /api/* har forrang
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
