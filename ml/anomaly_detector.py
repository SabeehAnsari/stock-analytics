import os
import pickle
import pandas as pd
import numpy as np

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "ml", "models")


def load_model(ticker: str):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


def score_row(ticker: str, row: dict) -> dict:
    """
    Score a single incoming tick against the trained model.
    Returns the row with an 'ml_anomaly' field added.
    """
    model = load_model(ticker)

    if model is None:
        row["ml_anomaly"] = False
        row["anomaly_score"] = 0.0
        return row

    # We only have single-row features here — use what's available
    # price_range_pct and daily_return are computable from one row
    # volume_zscore and MA ratios need history, so we approximate
    price_range_pct = (row["high"] - row["low"]) / row["close"] * 100
    daily_return    = 0.0  # No previous close available in stream context

    # Use neutral values for rolling features we can't compute on a single row
    volume_zscore = 0.0
    ma5_ratio     = 1.0
    ma20_ratio    = 1.0

    X = [[price_range_pct, daily_return, volume_zscore, ma5_ratio, ma20_ratio]]

    prediction    = model.predict(X)[0]       # -1 = anomaly, 1 = normal
    anomaly_score = model.decision_function(X)[0]  # Lower = more anomalous

    row["ml_anomaly"]    = bool(prediction == -1)
    row["anomaly_score"] = round(float(anomaly_score), 4)

    return row