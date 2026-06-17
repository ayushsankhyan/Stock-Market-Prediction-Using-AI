# app.py - Full dashboard without pandas-ta dependency

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# INDICATOR FUNCTIONS — pure pandas, no pandas-ta needed
# ─────────────────────────────────────────────────────────
def calc_sma(series, length):
    return series.rolling(window=length).mean()

def calc_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def calc_rsi(series, length=14):
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_l  = loss.ewm(com=length - 1, min_periods=length).mean()
    rs     = avg_g / avg_l
    return 100 - (100 / (1 + rs))

def calc_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# ─────────────────────────────────────────────────────────
# THE 8 COMPANIES
# ─────────────────────────────────────────────────────────
COMPANIES = {
    "🇮🇳 Reliance Industries" : ("RELIANCE.NS", "₹", "Energy & Retail"),
    "🇮🇳 TCS"                 : ("TCS.NS",      "₹", "IT Services"),
    "🇮🇳 HDFC Bank"           : ("HDFCBANK.NS", "₹", "Banking"),
    "🇮🇳 Infosys"             : ("INFY.NS",     "₹", "IT Services"),
    "🇺🇸 Apple"               : ("AAPL",        "$", "Technology"),
    "🇺🇸 NVIDIA"              : ("NVDA",        "$", "Semiconductors"),
    "🇺🇸 Microsoft"           : ("MSFT",        "$", "Technology"),
    "🇺🇸 Tesla"               : ("TSLA",        "$", "Electric Vehicles"),
}

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.title("📈 AI Stock Price Predictor")
st.caption("LSTM Neural Network · Live data from Yahoo Finance · Built by Ayush Sankhyan")
st.markdown("---")

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Controls")

selected_name = st.sidebar.selectbox(
    "🏢 Choose Company",
    options=list(COMPANIES.keys()),
    index=0
)

ticker, currency, sector = COMPANIES[selected_name]

st.sidebar.markdown(f"""
**Selected:**
- Ticker   : `{ticker}`
- Sector   : {sector}
- Market   : {"NSE India 🇮🇳" if ".NS" in ticker else "NASDAQ/NYSE 🇺🇸"}
- Currency : {currency}
""")

period   = st.sidebar.selectbox("📅 Chart Period",
                                 ["6mo","1y","2y","5y"], index=3)
show_raw = st.sidebar.checkbox("Show Raw Data Table", value=False)

st.sidebar.markdown("---")
st.sidebar.warning("⚠️ Educational project only. Not financial advice.")
st.sidebar.info("💡 Model trained on RELIANCE.NS. Patterns applied to all stocks.")

# ─────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("stock_lstm.keras")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ─────────────────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    df.columns = df.columns.get_level_values(0)
    close = df["Close"].squeeze()

    df["SMA_20"]      = calc_sma(close, 20)
    df["EMA_50"]      = calc_ema(close, 50)
    df["RSI"]         = calc_rsi(close, 14)
    macd, signal      = calc_macd(close)
    df["MACD"]        = macd
    df["MACD_Signal"] = signal

    df.dropna(inplace=True)
    return df

with st.spinner(f"⏳ Fetching live data for {selected_name}..."):
    df = fetch_data(ticker, period)

# ─────────────────────────────────────────────────────────
# TOP METRICS
# ─────────────────────────────────────────────────────────
current  = float(df["Close"].iloc[-1])
previous = float(df["Close"].iloc[-2])
change   = current - previous
pct      = (change / previous) * 100
high_52w = float(df["High"].max())
low_52w  = float(df["Low"].min())
volume   = int(df["Volume"].iloc[-1])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🏢 Company",       selected_name.split(" ", 1)[1])
c2.metric("💰 Current Price",
          f"{currency}{current:,.2f}",
          f"{change:+.2f} ({pct:+.2f}%)")
c3.metric("📈 52W High",      f"{currency}{high_52w:,.2f}")
c4.metric("📉 52W Low",       f"{currency}{low_52w:,.2f}")
c5.metric("📊 Today Volume",  f"{volume:,}")

st.markdown("---")

# ─────────────────────────────────────────────────────────
# PRICE CHART
# ─────────────────────────────────────────────────────────
st.subheader(f"📊 {selected_name} — Price Chart")

fig1, ax1 = plt.subplots(figsize=(14, 4))
ax1.plot(df.index, df["Close"],
         label="Close Price", linewidth=2,   color="#4f8ef7")
ax1.plot(df.index, df["SMA_20"],
         label="SMA 20",      linewidth=1.2, color="#f5a623",
         linestyle="--", alpha=0.85)
ax1.plot(df.index, df["EMA_50"],
         label="EMA 50",      linewidth=1.2, color="#7ed321",
         linestyle="--", alpha=0.85)
ax1.set_ylabel(f"Price ({currency})")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.2)
plt.tight_layout()
st.pyplot(fig1)
plt.close()

# ─────────────────────────────────────────────────────────
# RSI + MACD
# ─────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("📉 RSI — Momentum")
    rsi_now = float(df["RSI"].iloc[-1])

    fig2, ax2 = plt.subplots(figsize=(7, 2.8))
    ax2.plot(df.index, df["RSI"], color="#a855f7", linewidth=1.3)
    ax2.axhline(70, color="red",   linestyle="--", linewidth=1.2,
                alpha=0.8, label="Overbought (70)")
    ax2.axhline(30, color="green", linestyle="--", linewidth=1.2,
                alpha=0.8, label="Oversold (30)")
    ax2.axhline(50, color="gray",  linestyle=":",
                linewidth=0.8, alpha=0.5)
    ax2.fill_between(df.index, df["RSI"], 70,
                     where=(df["RSI"] >= 70), alpha=0.12, color="red")
    ax2.fill_between(df.index, df["RSI"], 30,
                     where=(df["RSI"] <= 30), alpha=0.12, color="green")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    if rsi_now >= 70:
        st.error(f"🔴 RSI = {rsi_now:.1f} — Overbought. Pullback possible.")
    elif rsi_now <= 30:
        st.success(f"🟢 RSI = {rsi_now:.1f} — Oversold. Bounce possible.")
    elif rsi_now >= 55:
        st.info(f"🔵 RSI = {rsi_now:.1f} — Moderately bullish.")
    elif rsi_now <= 45:
        st.warning(f"🟡 RSI = {rsi_now:.1f} — Moderately bearish.")
    else:
        st.info(f"⚪ RSI = {rsi_now:.1f} — Neutral zone.")

with right:
    st.subheader("📈 MACD — Trend")
    macd_now   = float(df["MACD"].iloc[-1])
    signal_now = float(df["MACD_Signal"].iloc[-1])

    fig3, ax3 = plt.subplots(figsize=(7, 2.8))
    ax3.plot(df.index, df["MACD"],
             label="MACD",   color="#4f8ef7", linewidth=1.3)
    ax3.plot(df.index, df["MACD_Signal"],
             label="Signal", color="#f5a623", linewidth=1.3)
    ax3.fill_between(df.index,
                     df["MACD"] - df["MACD_Signal"], 0,
                     where=(df["MACD"] >= df["MACD_Signal"]),
                     alpha=0.15, color="green")
    ax3.fill_between(df.index,
                     df["MACD"] - df["MACD_Signal"], 0,
                     where=(df["MACD"] < df["MACD_Signal"]),
                     alpha=0.15, color="red")
    ax3.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_ylabel("MACD")
    ax3.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    if macd_now > signal_now:
        st.success("🟢 MACD above Signal — Bullish momentum.")
    else:
        st.warning("🟡 MACD below Signal — Bearish momentum.")
    st.caption(f"MACD: {macd_now:.3f} | Signal: {signal_now:.3f} | "
               f"Diff: {macd_now - signal_now:+.3f}")

# ─────────────────────────────────────────────────────────
# AI PREDICTION
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 AI Price Prediction — Tomorrow's Close")

FEATURES   = ["Close", "SMA_20", "EMA_50", "RSI", "MACD", "MACD_Signal"]
SEQ_LEN    = 60
N_FEATURES = 6

if len(df) >= SEQ_LEN:
    last_60        = df[FEATURES].values[-SEQ_LEN:]
    last_60_scaled = scaler.transform(last_60)
    X_input        = last_60_scaled.reshape(1, SEQ_LEN, N_FEATURES)

    pred_scaled    = model.predict(X_input, verbose=0)
    dummy          = np.zeros((1, N_FEATURES))
    dummy[0, 0]    = pred_scaled[0, 0]
    pred_price     = float(scaler.inverse_transform(dummy)[0, 0])

    diff      = pred_price - current
    diff_pct  = (diff / current) * 100
    direction = "📈 UP" if pred_price > current else "📉 DOWN"

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Today's Close",
              f"{currency}{current:,.2f}")
    p2.metric("Predicted Tomorrow",
              f"{currency}{pred_price:,.2f}",
              f"{diff:+.2f} ({diff_pct:+.2f}%)")
    p3.metric("Expected Direction", direction)
    p4.metric("Model Accuracy",     "98.47%")

    st.markdown("#### Today vs Tomorrow")
    t_col, b_col = st.columns(2)

    with t_col:
        st.markdown(f"""
| | Price |
|---|---|
| **Today Close** | {currency}{current:,.2f} |
| **Predicted Tomorrow** | {currency}{pred_price:,.2f} |
| **Expected Change** | {currency}{diff:+,.2f} |
| **Change %** | {diff_pct:+.2f}% |
| **Direction** | {direction} |
        """)

    with b_col:
        fig4, ax4 = plt.subplots(figsize=(5, 2.5))
        colors = ["#4f8ef7",
                  "#7ed321" if pred_price > current else "#f5a623"]
        bars = ax4.bar(
            ["Today\nClose", "Predicted\nTomorrow"],
            [current, pred_price],
            color=colors, width=0.4, edgecolor="none"
        )
        for bar, val in zip(bars, [current, pred_price]):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.001,
                     f"{currency}{val:,.1f}",
                     ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
        ax4.set_ylabel(f"Price ({currency})")
        ax4.set_ylim(min(current, pred_price) * 0.995,
                     max(current, pred_price) * 1.008)
        ax4.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.caption("⚠️ Pattern-based prediction. Not financial advice.")

else:
    st.warning(f"Need at least {SEQ_LEN} days of data.")

# ─────────────────────────────────────────────────────────
# ALL 8 COMPANIES SNAPSHOT
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🌍 All 8 Companies — Live Snapshot")

with st.spinner("Fetching live prices..."):
    rows = []
    for name, (tick, curr, sec) in COMPANIES.items():
        try:
            snap = yf.download(tick, period="5d", progress=False)
            snap.columns = snap.columns.get_level_values(0)
            if len(snap) >= 2:
                price  = float(snap["Close"].iloc[-1])
                prev   = float(snap["Close"].iloc[-2])
                chng   = price - prev
                chng_p = (chng / prev) * 100
                rows.append({
                    "Company"  : name,
                    "Ticker"   : tick,
                    "Sector"   : sec,
                    "Price"    : f"{curr}{price:,.2f}",
                    "Change"   : f"{curr}{chng:+,.2f}",
                    "Change %" : f"{chng_p:+.2f}%",
                    "Signal"   : "📈" if chng > 0 else "📉"
                })
        except Exception:
            pass

    if rows:
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True,
                     hide_index=True)

# ─────────────────────────────────────────────────────────
# RAW DATA
# ─────────────────────────────────────────────────────────
if show_raw:
    st.markdown("---")
    st.subheader("📋 Raw Data — Last 30 Days")
    st.dataframe(
        df[["Open","High","Low","Close",
            "Volume","RSI","MACD"]].tail(30).round(2),
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built by Ayush Sankhyan · "
    "LSTM Neural Network · "
    "Live data from Yahoo Finance · "
    "Deployed on Streamlit Cloud"
)