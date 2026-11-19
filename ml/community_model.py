"""Scores a community pretrained delay model on our test months (PRD K-3).
Returns metrics, or a documented reason if the model cannot be compared fairly.

Run from flight-delay-ml/:  python -m ml.community_model
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import ROOT, DATASET_FILE, TEST_MONTHS, DELAY_THRESHOLD_MIN

try:
    from ml.evaluate import classification_metrics, regression_metrics
except ImportError:
    from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                                f1_score, mean_absolute_error, mean_squared_error)

    def classification_metrics(y_true, proba, threshold=0.5):
        pred = (np.asarray(proba) >= threshold).astype(int)
        return {"roc_auc": roc_auc_score(y_true, proba),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0)}

    def regression_metrics(y_true, y_pred):
        return {"mae": mean_absolute_error(y_true, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


COMMUNITY = {
    "name": "Barvero/flight-delay-predictor",
    "source_url": "https://huggingface.co/Barvero/flight-delay-predictor",
    "trained_on": "About 96K US domestic flights (year not stated); target is "
                "ARRIVAL_DELAY in minutes",
    "model_file": ROOT / "models" / "community" / "winning_model.pkl",
    "input_map": {},
}

# Based on the README only; the model file was not downloaded or run.
DOCUMENTED_REASON = (
    "Barvero/flight-delay-predictor (Hugging Face) is not comparable fairly: "
    "it was trained on ~96K US domestic flights (year not stated, not our 10-airport "
    "2023 set) using 33 scaled features (one-hot airline, frequency-encoded airports, "
    "IS_WEEKEND, a K-Means FLIGHT_CLUSTER) that the README does not list. "
    "The README shows only the model being exported, not the scaler, "
    "frequency tables or cluster model, so its inputs cannot be rebuilt from our data. "
    "It has no weather features, and its reported R2 of about 0.89 is high enough "
    "that leakage cannot be ruled out. Its reported metrics were not reproduced."
)


def _not_comparable(reason):
    return {"comparable": False, "reason": reason}


def evaluate_community_model(test_df):
    c = COMMUNITY
    if DOCUMENTED_REASON:
        return _not_comparable(DOCUMENTED_REASON)
    if not c["input_map"]:
        return _not_comparable("No input mapping defined for the community model.")
    missing = [v for v in c["input_map"].values() if v not in test_df.columns]
    if missing:
        return _not_comparable(f"Our dataset lacks columns needed by the model: {missing}")
    if not Path(c["model_file"]).exists():
        return _not_comparable(f"Model file not found: {c['model_file']}")

    df = test_df[test_df["month"].isin(TEST_MONTHS)]
    if df.empty:
        return _not_comparable("No rows in the test months.")
    X = df[list(c["input_map"].values())].copy()
    X.columns = list(c["input_map"].keys())

    try:
        model = joblib.load(c["model_file"])  # pickle: load only from a trusted source
        pred = np.clip(np.asarray(model.predict(X), dtype=float), 0, 300)
    except Exception as e:
        return _not_comparable(f"Model failed to load or predict: {type(e).__name__}: {e}")

    return {
        "comparable": True,
        "model": c["name"],
        "trained_on": c["trained_on"],
        "n_rows": int(len(df)),
        "regression": regression_metrics(df["delay_min"], pred),
        "classification": classification_metrics(
            df["delayed"], pred, threshold=DELAY_THRESHOLD_MIN),
        "note": "Predicted minutes were used as a delay score, not a calibrated probability.",
    }


if __name__ == "__main__":
    path = DATASET_FILE if DATASET_FILE.exists() else ROOT / "data" / "sample_flights.parquet"
    print("Scoring on:", path.name)
    print(evaluate_community_model(pd.read_parquet(path)))