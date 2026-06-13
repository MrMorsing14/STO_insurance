"""
Central konfiguration for STO-prototypen.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Policydokumenter — både .pdf og .md understøttes af ingesteren
POLICY_DIR = DATA_DIR / "policies"
PDF_DIR = POLICY_DIR  # bagudkompatibelt alias

CHROMA_DIR = DATA_DIR / "chroma_db"
METADATA_PATH = DATA_DIR / "policy_metadata.json"

# LLM — separate modeller pr. rolle:
#   Decomposer/klassificering: lettere opgave → billigere model
#   Evaluator: ræsonnement over betingelser/undtagelser → stærkeste model
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_MODEL_DECOMPOSER = os.getenv("MISTRAL_MODEL_DECOMPOSER", "mistral-medium-latest")
MISTRAL_MODEL_EVALUATOR = os.getenv("MISTRAL_MODEL_EVALUATOR", "mistral-large-latest")

# Bagudkompatibelt: hvis MISTRAL_MODEL er sat, bruges den som evaluator
_legacy = os.getenv("MISTRAL_MODEL")
if _legacy:
    MISTRAL_MODEL_EVALUATOR = _legacy
MISTRAL_MODEL = MISTRAL_MODEL_EVALUATOR  # alias for gammel kode

# Embeddings (e5-modeller kræver "query: " / "passage: " prefixes!)
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# STO-regler
STO_BELØB_GRÆNSE_DKK = 5000.0  # Krav over denne grænse går ALTID til manuelt review

# Confidence thresholds for auto-behandling af det enkelte delkrav
CONFIDENCE_AUTO_APPROVE = 0.85
CONFIDENCE_AUTO_REJECT = 0.85

# Dekomponering: hvor meget må summen af delkrav afvige fra totalbeløbet
BELØB_SUM_TOLERANCE_DKK = 1.0
