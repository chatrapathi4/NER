"""
app.py — FastAPI Application for Medical NER

Endpoints:
    GET  /        → Serves the frontend HTML page
    GET  /health  → Health check (is the service running, which model)
    POST /predict → Accepts medical text, returns extracted entities

The API loads the NER predictor once at startup and reuses it for all
requests (singleton pattern), avoiding the overhead of loading the model
on every request.
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    EntityResponse,
    HealthResponse,
)
from src.inference.predictor import MedicalNERPredictor

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Determine project root (two levels up from this file: src/api/app.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Global predictor instance (loaded once at startup)
predictor: MedicalNERPredictor = None


def get_model_type() -> str:
    """
    Determine which model type to use based on available trained models.

    Priority:
    1. CRF model (highest F1 score: 0.9565)
    2. Transformer model
    3. Rule-based fallback (always available)
    """
    crf_path = PROJECT_ROOT / "models" / "crf" / "crf_model.pkl"
    transformer_path = PROJECT_ROOT / "models" / "biobert" / "final"

    if crf_path.exists():
        return "crf"
    elif transformer_path.exists() and (transformer_path / "config.json").exists():
        return "transformer"
    else:
        return "rule_based"


# ──────────────────────────────────────────────────────────────────────────────
# Application Lifespan (startup/shutdown events)
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the NER model when the application starts.

    Using lifespan events (instead of @app.on_event) is the modern
    FastAPI approach for startup/shutdown logic.
    """
    global predictor

    model_type = get_model_type()
    logger.info(f"Loading NER predictor with model type: {model_type}")

    try:
        predictor = MedicalNERPredictor(
            model_type=model_type,
            transformer_model_dir=str(PROJECT_ROOT / "models" / "biobert" / "final"),
            crf_model_path=str(PROJECT_ROOT / "models" / "crf" / "crf_model.pkl"),
        )
        logger.info(f"Predictor loaded successfully (using: {predictor.model_type})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}. Using rule-based fallback.")
        predictor = MedicalNERPredictor(model_type="rule_based")

    yield  # Application runs here

    logger.info("Shutting down NER service.")


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medical NER API",
    description=(
        "Domain-Specific Named Entity Recognition for Medical Drug "
        "and Dosage Extraction. Extracts DRUG, DOSAGE, FREQUENCY, "
        "DURATION, ROUTE, and FORM entities from medical text."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — allows the frontend to call the API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
if FRONTEND_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="static",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """
    Serve the frontend HTML page.

    GET / returns the index.html file, which loads CSS and JS from /static/.
    """
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Place index.html in the frontend/ directory.",
        )
    return FileResponse(str(index_path))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns the service status and which model type is active.
    Useful for monitoring and load balancers.
    """
    return HealthResponse(
        status="healthy",
        model_type=predictor.model_type if predictor else "not_loaded",
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Extract medical entities from text.

    Accepts a JSON body with a "text" field containing medical/prescription
    text. Returns a list of extracted entities with their types and positions.

    Example request:
        {"text": "Take Paracetamol 500 mg twice daily for 5 days."}

    Example response:
        {
            "text": "Take Paracetamol 500 mg twice daily for 5 days.",
            "entities": [
                {"text": "Paracetamol", "label": "DRUG", "start": 5, "end": 16},
                {"text": "500 mg", "label": "DOSAGE", "start": 17, "end": 23},
                ...
            ],
            "model_used": "rule_based"
        }
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later.",
        )

    try:
        entities = predictor.extract_entities(request.text)
        return PredictionResponse(
            text=request.text,
            entities=[EntityResponse(**e) for e in entities],
            model_used=predictor.model_type,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error during entity extraction: {str(e)}",
        )
