# indicators.py
# Step 3 — Add technical indicators to our stock data

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# DOWNLOAD DATA
# ─────────────────────────────────────────────────────────
print("📥 Downloading data...")
df = yf.download("RELIANCE.NS", period="5y")

# Fix MultiIndex that newer yfinance creates
df.columns   = df.columns.get_level_values(0)
close_series = df["Close"].squeeze()

print(f"✅ Downloaded {len(df)} days of data")

# ─────────────────────────────────────────────────────────
# ADD INDICATOR 1 — SMA 20
# ─────────────────────────────────────────────────────────
df["SMA_20"] = ta.sma(close_series, length=20)
print("✅ SMA_20 added")

# ─────────────────────────────────────────────────────────
# ADD INDICATOR 2 — EMA 50
# ─────────────────────────────────────────────────────────
df["EMA_50"] = ta.ema(close_series, length=50)
print("✅ EMA_50 added")

# ─────────────────────────────────────────────────────────
# ADD INDICATOR 3 — RSI 14
# ─────────────────────────────────────────────────────────
df["RSI"] = ta.rsi(close_series, length=14)
print("✅ RSI added")

# ─────────────────────────────────────────────────────────
# ADD INDICATOR 4 & 5 — MACD + MACD Signal
# ─────────────────────────────────────────────────────────
# Calculate MACD
macd_data = ta.macd(close_series)

# Print exactly what columns came back
print(f"\nMACD raw columns: {macd_data.columns.tolist()}")

# Safely grab the 3 MACD columns by position
# pandas_ta always returns them in this order:
# column 0 = MACD line
# column 1 = Histogram
# column 2 = Signal line
macd_cols = macd_data.columns.tolist()

df["MACD"]        = macd_data[macd_cols[0]].values
df["MACD_Signal"] = macd_data[macd_cols[2]].values

print(f"✅ MACD added        (column: {macd_cols[0]})")
print(f"✅ MACD_Signal added (column: {macd_cols[2]})")

# ─────────────────────────────────────────────────────────
# DROP ROWS WITH NaN
# ─────────────────────────────────────────────────────────
print(f"\nRows BEFORE dropping NaN : {len(df)}")
df.dropna(inplace=True)
print(f"Rows AFTER dropping NaN  : {len(df)}")

# ─────────────────────────────────────────────────────────
# PRINT SAMPLE OF ENRICHED DATA
# ─────────────────────────────────────────────────────────
print("\n━━━ LAST 5 ROWS — All indicators ━━━")
print(df[["Close", "SMA_20", "EMA_50", "RSI", "MACD", "MACD_Signal"]].tail().round(2))

print("\n━━━ CURRENT INDICATOR VALUES (Today) ━━━")
print(f"Close Price  : ₹{float(df['Close'].iloc[-1]):,.2f}")
print(f"SMA 20       : ₹{float(df['SMA_20'].iloc[-1]):,.2f}")
print(f"EMA 50       : ₹{float(df['EMA_50'].iloc[-1]):,.2f}")
print(f"RSI          : {float(df['RSI'].iloc[-1]):.2f}")
print(f"MACD         : {float(df['MACD'].iloc[-1]):.4f}")
print(f"MACD Signal  : {float(df['MACD_Signal'].iloc[-1]):.4f}")

# ─────────────────────────────────────────────────────────
# INTERPRETATION
# ─────────────────────────────────────────────────────────
rsi_val  = float(df["RSI"].iloc[-1])
macd_val = float(df["MACD"].iloc[-1])
sig_val  = float(df["MACD_Signal"].iloc[-1])

print("\n━━━ WHAT DO TODAY'S INDICATORS SAY? ━━━")

if rsi_val > 70:
    print(f"RSI {rsi_val:.1f} → 🔴 OVERBOUGHT — stock may pull back soon")
elif rsi_val < 30:
    print(f"RSI {rsi_val:.1f} → 🟢 OVERSOLD — stock may bounce up soon")
else:
    print(f"RSI {rsi_val:.1f} → ⚪ NEUTRAL — no extreme signal")

if macd_val > sig_val:
    print("MACD        → 🟢 BULLISH — MACD above Signal line")
else:
    print("MACD        → 🔴 BEARISH — MACD below Signal line")

# ─────────────────────────────────────────────────────────
# DRAW 3-PANEL CHART
# ─────────────────────────────────────────────────────────
print("\n📊 Drawing indicator charts...")

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.suptitle("Reliance Industries — Technical Indicators",
             fontsize=15, fontweight="bold")

# ── Panel 1: Price + SMA + EMA ──
axes[0].plot(df.index, df["Close"],
             label="Close Price", linewidth=2, color="royalblue")
axes[0].plot(df.index, df["SMA_20"],
             label="SMA 20", linewidth=1.3,
             linestyle="--", color="orange", alpha=0.85)
axes[0].plot(df.index, df["EMA_50"],
             label="EMA 50", linewidth=1.3,
             linestyle="--", color="green", alpha=0.85)
axes[0].set_title("Price + Moving Averages", fontweight="bold")
axes[0].set_ylabel("Price (₹)")
axes[0].legend(loc="upper left", fontsize=9)
axes[0].grid(alpha=0.25)

# ── Panel 2: RSI ──
axes[1].plot(df.index, df["RSI"],
             color="purple", linewidth=1.3, label="RSI (14)")
axes[1].axhline(70, color="red",   linestyle="--",
                linewidth=1.2, alpha=0.8, label="Overbought (70)")
axes[1].axhline(30, color="green", linestyle="--",
                linewidth=1.2, alpha=0.8, label="Oversold (30)")
axes[1].axhline(50, color="gray",  linestyle=":",
                linewidth=0.8, alpha=0.6, label="Neutral (50)")
axes[1].fill_between(df.index, df["RSI"], 70,
                     where=(df["RSI"] >= 70),
                     alpha=0.15, color="red")
axes[1].fill_between(df.index, df["RSI"], 30,
                     where=(df["RSI"] <= 30),
                     alpha=0.15, color="green")
axes[1].set_title("RSI — Momentum Indicator (0 to 100)", fontweight="bold")
axes[1].set_ylabel("RSI Value")
axes[1].set_ylim(0, 100)
axes[1].legend(loc="upper left", fontsize=9)
axes[1].grid(alpha=0.25)

# ── Panel 3: MACD ──
axes[2].plot(df.index, df["MACD"],
             label="MACD Line", color="royalblue", linewidth=1.3)
axes[2].plot(df.index, df["MACD_Signal"],
             label="Signal Line", color="tomato", linewidth=1.3)
axes[2].fill_between(df.index,
                     df["MACD"] - df["MACD_Signal"], 0,
                     where=(df["MACD"] >= df["MACD_Signal"]),
                     alpha=0.15, color="green", label="Bullish zone")
axes[2].fill_between(df.index,
                     df["MACD"] - df["MACD_Signal"], 0,
                     where=(df["MACD"] < df["MACD_Signal"]),
                     alpha=0.15, color="red", label="Bearish zone")
axes[2].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
axes[2].set_title("MACD — Trend & Momentum Indicator", fontweight="bold")
axes[2].set_ylabel("MACD Value")
axes[2].legend(loc="upper left", fontsize=9)
axes[2].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("indicators_chart.png", dpi=150)
print("✅ Chart saved as indicators_chart.png")

print("\n━━━ STEP 3 COMPLETE ━━━")
print("👉 Right-click indicators_chart.png → Open Preview")
print("👉 You'll see 3 panels — price, RSI, MACD")
print("👉 Ready for Step 4!")