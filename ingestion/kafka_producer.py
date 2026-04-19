import pandas as pd
import json
import time
import os
from kafka import KafkaProducer

# Kafka config
KAFKA_TOPIC   = "stock-stream"
KAFKA_BROKER  = "localhost:9092"

# How fast to simulate the stream
TICK_INTERVAL = 0.5

# Absolute path to data/raw regardless of where script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")


def load_all_tickers() -> pd.DataFrame:
    frames = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".parquet"):
            df = pd.read_parquet(os.path.join(DATA_DIR, file))
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No parquet files found in {DATA_DIR}")

    combined = pd.concat(frames)

    # Sort by date so we replay in chronological order
    combined.sort_values("date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    return combined


def serialize(row: dict) -> bytes:
    return json.dumps(row, default=str).encode("utf-8")


def run_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=serialize
    )

    df = load_all_tickers()
    total = len(df)

    print(f"Starting stream — {total} ticks across {df['ticker'].nunique()} tickers")
    print("Press Ctrl+C to stop.\n")

    for i, row in df.iterrows():
        message = {
            "date":   str(row["date"]),
            "ticker": row["ticker"],
            "open":   round(float(row["open"]),   2),
            "high":   round(float(row["high"]),   2),
            "low":    round(float(row["low"]),    2),
            "close":  round(float(row["close"]),  2),
            "volume": int(row["volume"])
        }

        producer.send(KAFKA_TOPIC, value=message)

        print(f"[{i+1}/{total}] Sent → {message['ticker']} | {message['date']} | close: ${message['close']}")

        time.sleep(TICK_INTERVAL)

    producer.flush()
    print("\nStream complete.")


if __name__ == "__main__":
    run_producer()