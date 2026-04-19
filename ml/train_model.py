import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR  = os.path.join(BASE_DIR, "ml", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features the Isolation Forest will learn from."""

    df = df.copy()
    df.sort_values("date", inplace=True)

    # Price range as % of close — measures intraday volatility
    df["price_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100

    # Daily return — % change from previous close
    df["daily_return"] = df["close"].pct_change() * 100

    # Volume z-score — how unusual is today's volume vs 20-day average
    df["volume_zscore"] = (
        (df["volume"] - df["volume"].rolling(20).mean()) /
        df["volume"].rolling(20).std()
    )

    # 5-day and 20-day moving average ratio — detects price trend breaks
    df["ma5_ratio"]  = df["close"] / df["close"].rolling(5).mean()
    df["ma20_ratio"] = df["close"] / df["close"].rolling(20).mean()

    # Drop NaN rows created by rolling windows
    df.dropna(inplace=True)

    return df


def train_and_save(ticker: str, df: pd.DataFrame):
    print(f"Training model for {ticker}...")

    df = engineer_features(df)

    features = [
        "price_range_pct",
        "daily_return",
        "volume_zscore",
        "ma5_ratio",
        "ma20_ratio"
    ]

    X = df[features].values

    # Isolation Forest — contamination=0.05 means we expect ~5% anomalies
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X)

    # Save model to disk
    model_path = os.path.join(MODEL_DIR, f"{ticker}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Quick evaluation — how many anomalies did it find in training data?
    preds = model.predict(X)
    n_anomalies = (preds == -1).sum()
    print(f"  Trained on {len(X)} rows — flagged {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%)")
    print(f"  Saved to {model_path}")


if __name__ == "__main__":
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".parquet"):
            continue

        ticker = file.replace(".parquet", "")
        df = pd.read_parquet(os.path.join(DATA_DIR, file))
        train_and_save(ticker, df)

    print("\nAll models trained and saved.")