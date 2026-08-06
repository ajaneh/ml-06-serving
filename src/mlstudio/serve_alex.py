# === Section 1. IMPORTS ===

import logging
from pathlib import Path
from typing import Any, Final

from datafun_toolkit.logger import get_logger, log_header
from fastapi import FastAPI, HTTPException
import joblib  # for serializing and deserializing the model
from sklearn.ensemble import RandomForestClassifier

__all__ = ["app", "predict_from_features", "predict"]

# === Section 2. CONFIGURE LOGGER ===

LOG: logging.Logger = get_logger("M06", level="DEBUG")
log_header(LOG, "M06")

# === Section 3. CONSTANTS AND CONFIGURATION ===

# The path to the saved model artifact.
MODEL_PATH: Final[Path] = Path("artifacts") / "model.joblib"

# The feature columns the model was trained on.
# These must match exactly what was used during training.
FEATURE_COLS: Final[list[str]] = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]

# === Section 4. LOAD THE MODEL ===

LOG.info(f"Loading model from: {MODEL_PATH}")

if not MODEL_PATH.exists():
    LOG.error(f"Model file not found: {MODEL_PATH}")
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run the training notebook or app_case.py first."
    )

MODEL = joblib.load(MODEL_PATH)
LOG.info("Model loaded successfully")

# === Section 5. CREATE THE APP ===

app = FastAPI(title="Penguin species classifier")

# === Section 6. DEFINE THE PREDICT ENDPOINT ===


def predict_from_features(
    model: RandomForestClassifier, payload: dict[str, Any]
) -> dict[str, Any]:
    """Pure prediction function - testable outside the web framework."""
    try:
        features = [float(payload[c]) for c in FEATURE_COLS]
    except KeyError as exc:
        raise ValueError(f"Missing required feature: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid feature value: {exc}") from exc

    label: str = str(model.predict([features])[0])
    LOG.info(f"Predicted label: {label} for features: {features}")
    return {"prediction": label}


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return predict_from_features(MODEL, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
