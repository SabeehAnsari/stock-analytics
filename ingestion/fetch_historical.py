import pandas as pd
import os

# Paths
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "data", "raw", "all_stocks_5yr.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "raw")

# We'll focus on these 5 tickers — rest of the 500 are still in the CSV if needed
TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]


def convert_to_parquet():
    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Standardize column names to match the rest of our pipeline
    df.rename(columns={
        "date":   "date",
        "open":   "open",
        "high":   "high",
        "low":    "low",
        "close":  "close",
        "volume": "volume",
        "Name":   "ticker"
    }, inplace=True)

    # Keep only the columns we need
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]

    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])

    print(f"Total rows loaded: {len(df)}")
    print(f"Tickers available: {df['ticker'].nunique()}")

    # Save one Parquet file per ticker
    for ticker in TICKERS:
        ticker_df = df[df["ticker"] == ticker].copy()

        if ticker_df.empty:
            print(f"  WARNING: No data found for {ticker}, skipping...")
            continue

        ticker_df.sort_values("date", inplace=True)
        ticker_df.reset_index(drop=True, inplace=True)

        output_path = os.path.join(OUTPUT_DIR, f"{ticker}.parquet")
        ticker_df.to_parquet(output_path, index=False)

        print(f"  Saved {len(ticker_df)} rows to {output_path}")

    print("\nDone. Parquet files ready.")


if __name__ == "__main__":
    convert_to_parquet()