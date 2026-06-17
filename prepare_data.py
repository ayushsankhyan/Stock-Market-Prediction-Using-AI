# prepare_data.py
# Scale data + create 60-day sliding windows for LSTM

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

print("📥 Downloading and processing data...")

# ─────────────────────────────────────────────────────────
# REBUILD THE DATASET 
# ─────────────────────────────────────────────────────────
df = yf.download("RELIANCE.NS", period="5y")

# Fix MultiIndex
df.columns   = df.columns.get_level_values(0)
close_series = df["Close"].squeeze()

# Add all indicators
df["SMA_20"] = ta.sma(close_series, length=20)
df["EMA_50"] = ta.ema(close_series, length=50)
df["RSI"]    = ta.rsi(close_series, length=14)

macd_data    = ta.macd(close_series)
macd_cols    = macd_data.columns.tolist()
df["MACD"]        = macd_data[macd_cols[0]].values
df["MACD_Signal"] = macd_data[macd_cols[2]].values

df.dropna(inplace=True)
print(f"✅ Data ready — {len(df)} days after cleaning")

# ─────────────────────────────────────────────────────────
# STEP A — SELECT THE 6 FEATURES
# ─────────────────────────────────────────────────────────
# These 6 columns are what the AI learns from
# Close       = actual price (also what we predict)
# SMA_20      = short term trend
# EMA_50      = medium term trend
# RSI         = overbought / oversold signal
# MACD        = momentum direction
# MACD_Signal = smoothed MACD crossover signal

FEATURES = ["Close", "SMA_20", "EMA_50", "RSI", "MACD", "MACD_Signal"]

# Extract as plain numpy array
# Shape: (1193, 6) → 1193 days, 6 numbers per day
data = df[FEATURES].values

print(f"\n━━━ STEP A — Feature Selection ━━━")
print(f"Features : {FEATURES}")
print(f"Shape    : {data.shape}  →  {data.shape[0]} days × {data.shape[1]} features")

# ─────────────────────────────────────────────────────────
# STEP B — SCALE EVERYTHING TO 0–1
# ─────────────────────────────────────────────────────────
# Why scale?
# Close price  = ₹1000 to ₹3000  (big numbers)
# RSI          = 0 to 100         (medium numbers)
# MACD         = tiny decimals    (small numbers)
#
# Without scaling, the AI thinks Close is 30x more important
# than RSI just because its numbers are bigger — that is WRONG
#
# MinMaxScaler fixes this:
# lowest value in each column  → 0.0
# highest value in each column → 1.0
# everything else scales in between proportionally

scaler      = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
# fit_transform:
# fit      → learns min and max of each column FROM this data
# transform → converts all values to 0-1 range

print(f"\n━━━ STEP B — Scaling Verification ━━━")
print(f"Before scaling — Close : ₹{data[:,0].min():.0f}  to  ₹{data[:,0].max():.0f}")
print(f"After  scaling — Close :  {scaled_data[:,0].min():.4f}  to   {scaled_data[:,0].max():.4f}")
print(f"Before scaling — RSI   :  {data[:,3].min():.1f}  to  {data[:,3].max():.1f}")
print(f"After  scaling — RSI   :  {scaled_data[:,3].min():.4f}  to   {scaled_data[:,3].max():.4f}")
print("✅ All columns now in range 0.0 → 1.0")

# ─────────────────────────────────────────────────────────
# STEP C — CREATE 60-DAY SLIDING WINDOWS
# ─────────────────────────────────────────────────────────
# LSTM needs sequences — not individual rows
# We create "windows" like this:
#
# Window 1 : days  0–59  → predict day  60 close price
# Window 2 : days  1–60  → predict day  61 close price
# Window 3 : days  2–61  → predict day  62 close price
# ...and so on
#
# Think of each window as one flashcard:
# Front = 60 days of all 6 features
# Back  = what the close price was the next day
#
# The AI studies all these flashcards and learns
# "when I see THIS pattern → the price usually does THIS"

SEQ_LEN = 60   # look back 60 trading days = roughly 3 months

X = []   # will hold all the 60-day windows (inputs)
y = []   # will hold all the next-day prices (targets)

for i in range(SEQ_LEN, len(scaled_data)):
    # X: 60 rows of all 6 features
    X.append(scaled_data[i - SEQ_LEN : i])

    # y: just the Close price of the NEXT day
    # index 0 = Close (first column in FEATURES list)
    y.append(scaled_data[i, 0])

# Convert Python lists → NumPy arrays (TensorFlow needs this)
X = np.array(X)
y = np.array(y)

print(f"\n━━━ STEP C — Sequence Creation ━━━")
print(f"SEQ_LEN (lookback window) : {SEQ_LEN} trading days")
print(f"X shape : {X.shape}")
print(f"          → {X.shape[0]} windows")
print(f"          → each window = {X.shape[1]} days × {X.shape[2]} features")
print(f"y shape : {y.shape}")
print(f"          → {y.shape[0]} target prices (one per window)")

# ─────────────────────────────────────────────────────────
# STEP D — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────
# 80% of windows → AI trains on these (learns the patterns)
# 20% of windows → AI is tested on these (never seen before)
#
# IMPORTANT: we do NOT shuffle the data
# Order matters for time series — day 500 must come after day 499
# Shuffling would mix future data into training = cheating

split       = int(len(X) * 0.80)

X_train     = X[:split]
X_test      = X[split:]
y_train     = y[:split]
y_test      = y[split:]

print(f"\n━━━ STEP D — Train / Test Split ━━━")
print(f"Total windows    : {len(X)}")
print(f"Training windows : {len(X_train)}  (80%) ← AI learns from these")
print(f"Test windows     : {len(X_test)}   (20%) ← AI is evaluated on these")
print(f"\nTraining covers  : {df.index[SEQ_LEN].date()}  →  {df.index[split + SEQ_LEN - 1].date()}")
print(f"Testing covers   : {df.index[split + SEQ_LEN].date()}  →  {df.index[-1].date()}")

# ─────────────────────────────────────────────────────────
# STEP E — SAVE EVERYTHING TO DISK
# ─────────────────────────────────────────────────────────
# We save these files so train_model.py and app.py
# can load them without re-downloading data every time
#
# .npy  = NumPy's native binary format (fast to save/load)
# .pkl  = Python pickle format (saves any Python object)

np.save("X_train.npy", X_train)
np.save("X_test.npy",  X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy",  y_test)

# CRITICAL: save the scaler
# We MUST use the EXACT same scaler later to convert
# scaled predictions back to real ₹ prices
# If we lose the scaler we lose the ability to interpret predictions
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\n━━━ STEP E — Files Saved ━━━")
print("✅ X_train.npy  — training input sequences")
print("✅ X_test.npy   — test input sequences")
print("✅ y_train.npy  — training target prices")
print("✅ y_test.npy   — test target prices")
print("✅ scaler.pkl   — the MinMaxScaler object")

