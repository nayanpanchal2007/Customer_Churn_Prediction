from __future__ import annotations

import joblib
import pandas as pd

from .config import MODEL_DIR, RISK_THRESHOLDS
from .feature_engineering import engineer_features


def load_model():
    path = MODEL_DIR / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python -m src.train_models"
        )
    return joblib.load(path)


def risk_level(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low_max"]:
        return "Low"
    if probability <= RISK_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"


def predict_customer(customer: dict, model=None):
    model = model or load_model()
    row = pd.DataFrame([customer])
    row = engineer_features(row)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.50)
    return {
        "prediction": prediction,
        "probability": probability,
        "risk": risk_level(probability),
    }
