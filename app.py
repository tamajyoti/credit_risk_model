"""
Part 2: FastAPI Inference Service for credit decision

Exposes POST apis which accepts customer features, returns default probability.

Base structure is obtained from app.py skeleton; extended with
customer_id in response, input validation for features, and startup health check.
"""

import warnings
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# Artifacts path
# artifacts/ sits alongside this file (both in project root)
MODEL_PATH  = Path(__file__).resolve().parent / "artifacts" / "model.joblib"
SCALER_PATH = Path(__file__).resolve().parent / "artifacts" / "scaler.joblib"

# The exact feature order the model was trained on — do not reorder
FLOAT_COLS = ["txn_count", "total_debit", "total_credit", "avg_amount"]
FLAG_COLS  = ["kw_rent", "kw_netflix", "kw_tesco", "kw_payroll", "kw_bonus"]
MODEL_FEATURES = FLOAT_COLS + FLAG_COLS

# App Initialization
app = FastAPI(
    title="Credit Risk API",
    description=(
        "Predicts 90-day default probability from customer transaction features.\n\n"
        "- `POST /predict` — raw features, no scaling\n"
        "- `POST /predict/scaled` — float features scaled via StandardScaler before inference"
    ),
    version="0.1.0",
)


# Schemas for input and output
class CustomerFeatures(BaseModel):
    customer_id:  str   = Field(..., example="CUST_0001")

    # Core transaction aggregates which are float features
    txn_count:    float = Field(..., ge=0,     example=3)
    total_debit:  float = Field(..., le=0,     example=-65.98,  description="Must be <= 0")
    total_credit: float = Field(..., ge=0,     example=2500.00, description="Must be >= 0")
    avg_amount:   float = Field(...,           example=811.34)

    # Merchant keyword flags 0 or 1 which are flags and no scaling is done
    kw_rent:      int   = Field(0, ge=0, le=1, example=0)
    kw_netflix:   int   = Field(0, ge=0, le=1, example=1)
    kw_tesco:     int   = Field(0, ge=0, le=1, example=1)
    kw_payroll:   int   = Field(0, ge=0, le=1, example=1)
    kw_bonus:     int   = Field(0, ge=0, le=1, example=0)


class PredictionResponse(BaseModel):
    customer_id: str
    probability: float = Field(..., description="Probability of defaulting within 90 days")
    prediction:  int   = Field(..., description="1 = probably default, 0 = probably no default")


# Model lifecycle
model  = None
scaler = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler
 
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. "
            "Place model.joblib in the artifacts/ directory."
        )
    model = joblib.load(MODEL_PATH)
 
    if not SCALER_PATH.exists():
        raise RuntimeError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run prepare_data.py first to generate scaler.joblib."
        )
    scaler = joblib.load(SCALER_PATH)

# ── Shared inference logic ─────────────────────────────────────────────────────
def _build_feature_vector(payload: CustomerFeatures, apply_scaling: bool) -> list:
    """
    Build the 9-element feature vector in MODEL_FEATURES order.
 
    If apply_scaling=True:
      - Float features (txn_count, total_debit, total_credit, avg_amount) are
        transformed using the fitted StandardScaler: x_scaled = (x - mean) / std
      - Keyword flags (kw_*) are passed through as-is (0 or 1)
 
    If apply_scaling=False:
      - All features are passed through raw — original behaviour.
    """
    float_vals = np.array([[getattr(payload, col) for col in FLOAT_COLS]])
    flag_vals  = [getattr(payload, col) for col in FLAG_COLS]
 
    if apply_scaling:
        float_vals = scaler.transform(float_vals)
 
    return [float_vals[0].tolist() + flag_vals]

def _run_inference(payload: CustomerFeatures, apply_scaling: bool) -> PredictionResponse:
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model or scaler not loaded")
 
    X     = _build_feature_vector(payload, apply_scaling)
    proba = float(model.predict_proba(X)[0][1])
    pred  = int(proba >= 0.5)
 
    return PredictionResponse(
        customer_id=payload.customer_id,
        probability=round(proba, 4),
        prediction=pred,
        scaled=apply_scaling,
    )


# App Routes
@app.get("/health", tags=["ops"])
def health():
    """Liveness check — also confirms the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict_raw(payload: CustomerFeatures) -> PredictionResponse:
    """
    Predict using **raw** feature values — no scaling applied.
 
    Use this route to compare against the scaled route .
    """
    return _run_inference(payload, apply_scaling=False)

@app.post("/predict/scaled", response_model=PredictionResponse, tags=["inference"])
def predict_scaled(payload: CustomerFeatures) -> PredictionResponse:
    """
    Predict with **StandardScaler** applied to float features before inference.
 
    Float features (txn_count, total_debit, total_credit, avg_amount) are
    transformed to mean=0, std=1 using parameters fitted on the training data.
    Keyword flags (kw_*) are passed through unchanged.
 
    This is the recommended route for production use.
    """
    return _run_inference(payload, apply_scaling=True)