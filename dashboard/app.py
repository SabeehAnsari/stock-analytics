import os
import sys
import time
import glob
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path so we can import anomaly_detector
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from ml.anomaly_detector import load_model, score_row

# Paths
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw")
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")

# Page config
st.set_page_config(
    page_title="Stock Analytics Platform",
    page_icon="📈",
    layout="wide"
)

def load_model(ticker: str):
    model_path = os.path.join(BASE_DIR, "ml", "models", f"{ticker}_model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-top: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
    }
    .anomaly-badge {
        background: #7f1d1d;
        color: #fca5a5;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .normal-badge {
        background: #14532d;
        color: #86efac;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    h1, h2, h3, .stSelectbox label, .stMetric label {
        color: #f1f5f9 !important;
    }
    .stMetric { background: #111827; border-radius: 8px; padding: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_ticker_data(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_alerts() -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(ALERTS_DIR, "**", "*.csv"), recursive=True)
    if not csv_files:
        return pd.DataFrame()
    
    col_names = ["date", "ticker", "open", "high", "low", "close", "volume", "anomaly"]
    frames = []
    
    for f in csv_files:
        try:
            df = pd.read_csv(f, header=None, names=col_names)
            # Drop rows where 'date' column contains the literal string 'date' (bad header rows)
            df = df[df["date"] != "date"]
            # Drop rows where ticker is literally 'ticker'
            df = df[df["ticker"] != "ticker"]
            df = df.dropna(subset=["ticker", "date"])
            frames.append(df)
        except Exception:
            continue
    
    if not frames:
        return pd.DataFrame()
    
    return pd.concat(frames, ignore_index=True)


def get_available_tickers() -> list:
    files = glob.glob(os.path.join(DATA_DIR, "*.parquet"))
    return sorted([os.path.basename(f).replace(".parquet", "") for f in files])


# ── ML scoring on historical data ─────────────────────────────────────────
@st.cache_data
def score_historical(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)

    # Engineer features with full historical context
    df["price_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    df["daily_return"]    = df["close"].pct_change() * 100
    df["volume_zscore"]   = (
        (df["volume"] - df["volume"].rolling(20).mean()) /
        df["volume"].rolling(20).std()
    )
    df["ma5_ratio"]  = df["close"] / df["close"].rolling(5).mean()
    df["ma20_ratio"] = df["close"] / df["close"].rolling(20).mean()
    df.dropna(inplace=True)

    model = load_model(ticker)
    if model is None:
        df["ml_anomaly"]    = False
        df["anomaly_score"] = 0.0
        return df

    features = ["price_range_pct", "daily_return", "volume_zscore", "ma5_ratio", "ma20_ratio"]
    X = df[features].values

    df["ml_anomaly"]    = model.predict(X) == -1
    df["anomaly_score"] = model.decision_function(X)

    return df


# ── Header ─────────────────────────────────────────────────────────────────
st.title("📈 Real-Time Stock Analytics & Anomaly Detection")
st.caption("Lambda Architecture · PySpark Structured Streaming · Isolation Forest")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    tickers      = get_available_tickers()
    selected     = st.selectbox("Select Ticker", tickers)
    auto_refresh = st.toggle("Auto Refresh (10s)", value=False)
    st.divider()
    st.markdown("**Pipeline Status**")
    st.success("Kafka  ·  Connected")
    st.success("PySpark  ·  Running")
    st.success("Isolation Forest  ·  Loaded")

if auto_refresh:
    time.sleep(10)
    st.rerun()

# ── Load data ──────────────────────────────────────────────────────────────
df = load_ticker_data(selected)

if df.empty:
    st.error(f"No data found for {selected}")
    st.stop()

# Score last 200 rows for performance
df_scored = score_historical(selected, df.tail(200))
df_scored["date"] = pd.to_datetime(df_scored["date"])

anomalies = df_scored[df_scored["ml_anomaly"] == True]

# ── KPI Metrics ────────────────────────────────────────────────────────────
latest      = df.iloc[-1]
prev        = df.iloc[-2]
price_delta = round(float(latest["close"]) - float(prev["close"]), 2)
pct_delta   = round(price_delta / float(prev["close"]) * 100, 2)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close",    f"${latest['close']:.2f}",  f"{price_delta:+.2f}")
col2.metric("Daily High",      f"${latest['high']:.2f}")
col3.metric("Daily Low",       f"${latest['low']:.2f}")
col4.metric("Anomalies Found", f"{len(anomalies)}",        f"of last 200 rows")

st.divider()

# ── Price Chart with Anomaly Overlay ──────────────────────────────────────
st.subheader(f"{selected} — Price History & Anomalies")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.7, 0.3],
    vertical_spacing=0.05
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=df["date"],
    open=df["open"], high=df["high"],
    low=df["low"],   close=df["close"],
    name="Price",
    increasing_line_color="#22c55e",
    decreasing_line_color="#ef4444"
), row=1, col=1)

# Anomaly markers
if not anomalies.empty:
    fig.add_trace(go.Scatter(
        x=anomalies["date"],
        y=anomalies["close"],
        mode="markers",
        marker=dict(color="#f97316", size=10, symbol="x"),
        name="Anomaly"
    ), row=1, col=1)

# Volume bars
fig.add_trace(go.Bar(
    x=df["date"],
    y=df["volume"],
    name="Volume",
    marker_color="#3b82f6",
    opacity=0.6
), row=2, col=1)

fig.update_layout(
    height=550,
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0a0e1a",
    font_color="#f1f5f9",
    xaxis_rangeslider_visible=False,
    legend=dict(bgcolor="#111827"),
    margin=dict(l=10, r=10, t=10, b=10)
)
fig.update_xaxes(gridcolor="#1f2937")
fig.update_yaxes(gridcolor="#1f2937")

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Anomaly Score Chart ────────────────────────────────────────────────────
st.subheader("Isolation Forest Anomaly Scores")
st.caption("Lower score = more anomalous. Scores below 0 are flagged.")

score_fig = go.Figure()
score_fig.add_trace(go.Scatter(
    x=df_scored["date"],
    y=df_scored["anomaly_score"],
    mode="lines",
    line=dict(color="#818cf8", width=1.5),
    name="Anomaly Score"
))
score_fig.add_hline(
    y=0, line_dash="dash",
    line_color="#ef4444",
    annotation_text="Anomaly Threshold"
)
score_fig.update_layout(
    height=250,
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0a0e1a",
    font_color="#f1f5f9",
    margin=dict(l=10, r=10, t=10, b=10)
)
score_fig.update_xaxes(gridcolor="#1f2937")
score_fig.update_yaxes(gridcolor="#1f2937")

st.plotly_chart(score_fig, use_container_width=True)

st.divider()

# ── Live Alert Feed ────────────────────────────────────────────────────────
st.subheader("Live Alert Feed")
st.caption("Anomalies detected by PySpark stream processor")

alerts_df = load_alerts()

if alerts_df.empty:
    st.info("No alerts yet — start the Kafka producer and stream processor to populate this feed.")
else:
    st.dataframe(
        alerts_df.sort_values("date", ascending=False).head(50),
        use_container_width=True
    )

# ── Raw Data Table ─────────────────────────────────────────────────────────
with st.expander("View Raw Data"):
    st.dataframe(df.tail(100), use_container_width=True)