# === Section 1. IMPORTS ===

import logging
from pathlib import Path
from typing import Any, Final

from datafun_toolkit.logger import get_logger, log_header
from fastapi import FastAPI, HTTPException, status
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
    missing_features = [column for column in FEATURE_COLS if column not in payload]
    if missing_features:
        missing = ", ".join(missing_features)
        raise ValueError(f"Missing required feature: {missing}")

    features: list[float] = []
    for column in FEATURE_COLS:
        value = payload[column]
        try:
            features.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{column}' must be numeric") from exc

    label: str = str(model.predict([features])[0])
    probabilities: list[float] = model.predict_proba([features])[0]
    prob_dict: dict[str, float] = dict(
        zip(model.classes_.tolist(), probabilities.tolist(), strict=False)
    )
    LOG.info(f"Predicted probabilities: {prob_dict} for features: {features}")
    LOG.info(f"Predicted label: {label} for features: {features}")
    return {
        "prediction": label,
        "confidence": prob_dict[label],
        "probabilities": prob_dict,
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return predict_from_features(MODEL, payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        LOG.exception("Unexpected prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        ) from exc
