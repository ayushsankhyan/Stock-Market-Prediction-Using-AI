# explore.py
# Download and explore real stock data

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # saves chart to file instead of opening a window
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# DOWNLOAD 5 YEARS OF REAL STOCK DATA
# ─────────────────────────────────────────────────────────
# "RELIANCE.NS" = Reliance Industries on NSE
# .NS suffix = National Stock Exchange of India
# period="5y" = last 5 years of trading data

print("📥 Downloading data from Yahoo Finance...")
df = yf.download("RELIANCE.NS", period="5y")

print("✅ Download complete!")

# ─────────────────────────────────────────────────────────
# WHAT DOES THE DATA LOOK LIKE?
# ─────────────────────────────────────────────────────────

print("\n━━━ SHAPE OF DATA ━━━")
print(f"Rows (trading days) : {df.shape[0]}")
print(f"Columns             : {df.shape[1]}")
# Each ROW = one trading day
# Each COLUMN = a different price measurement

print("\n━━━ FIRST 5 ROWS (oldest data) ━━━")
print(df.head())
# Open   = price when market opened that morning at 9:15 AM
# High   = highest price the stock touched that entire day
# Low    = lowest price the stock touched that entire day
# Close  = price when market closed at 3:30 PM ← THIS IS WHAT WE PREDICT
# Volume = how many shares changed hands that day

print("\n━━━ LAST 5 ROWS (most recent data) ━━━")
print(df.tail())

print("\n━━━ DATE RANGE ━━━")
print(f"Oldest data point : {df.index[0].date()}")
print(f"Newest data point : {df.index[-1].date()}")

print("\n━━━ ANY MISSING VALUES? ━━━")
print(df.isnull().sum())
# All should be 0 — Yahoo Finance data is very clean

print("\n━━━ CLOSE PRICE STATISTICS ━━━")
print(df["Close"].describe())
# count = total trading days
# mean  = average closing price over 5 years
# min   = cheapest the stock ever was in this period
# max   = most expensive it ever got
# std   = standard deviation (how much it swings around the average)

# ─────────────────────────────────────────────────────────
# DRAW AND SAVE THE PRICE CHART
# ─────────────────────────────────────────────────────────
print("\n📊 Drawing price chart...")

plt.figure(figsize=(14, 5))

plt.plot(
    df.index,        # x-axis = dates
    df["Close"],     # y-axis = closing price
    color="royalblue",
    linewidth=1.5,
    label="Close Price"
)

plt.title("Reliance Industries — 5 Year Closing Price", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Price (₹)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

# Save the chart as an image file in your project folder
plt.savefig("price_chart.png", dpi=150)
print("✅ Chart saved as price_chart.png")

