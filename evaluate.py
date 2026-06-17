# evaluate.py
# Step 6 — Test how accurate the trained model is

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────────────────
# LOAD EVERYTHING
# ─────────────────────────────────────────────────────────
print("📂 Loading model and data...")

model  = tf.keras.models.load_model("stock_lstm.keras")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

print(f"✅ Model loaded")
print(f"✅ Test samples : {len(X_test)}")

# ─────────────────────────────────────────────────────────
# MAKE PREDICTIONS
# ─────────────────────────────────────────────────────────
print("\n🔮 Making predictions on test data...")
predictions_scaled = model.predict(X_test, verbose=1)
print(f"✅ Predictions done — shape: {predictions_scaled.shape}")

# ─────────────────────────────────────────────────────────
# CONVERT SCALED PREDICTIONS BACK TO REAL ₹ PRICES
# ─────────────────────────────────────────────────────────
# The scaler expects 6 columns (all features)
# but predictions only have 1 column (Close price)
#
# Solution:
# 1. Create a dummy array with 6 columns filled with zeros
# 2. Put our prediction in column 0 (Close)
# 3. Run inverse_transform on the whole thing
# 4. Extract only column 0 (Close price in real ₹)

N_FEATURES = 6

# Convert predictions
dummy_pred       = np.zeros((len(predictions_scaled), N_FEATURES))
dummy_pred[:, 0] = predictions_scaled[:, 0]
predictions_real = scaler.inverse_transform(dummy_pred)[:, 0]

# Convert actual values same way
dummy_actual       = np.zeros((len(y_test), N_FEATURES))
dummy_actual[:, 0] = y_test
actuals_real       = scaler.inverse_transform(dummy_actual)[:, 0]

print(f"\n✅ Converted back to real ₹ prices")
print(f"Actual price range    : ₹{actuals_real.min():,.0f} → ₹{actuals_real.max():,.0f}")
print(f"Predicted price range : ₹{predictions_real.min():,.0f} → ₹{predictions_real.max():,.0f}")

# ─────────────────────────────────────────────────────────
# CALCULATE ACCURACY METRICS
# ─────────────────────────────────────────────────────────
rmse     = np.sqrt(mean_squared_error(actuals_real, predictions_real))
mae      = np.mean(np.abs(actuals_real - predictions_real))
mape     = np.mean(np.abs((actuals_real - predictions_real) / actuals_real)) * 100
accuracy = 100 - mape

print(f"\n{'━'*50}")
print(f"  RESULTS ON {len(X_test)} UNSEEN TEST DAYS")
print(f"{'━'*50}")
print(f"  RMSE     : ₹{rmse:.2f}")
print(f"  → On average predictions are ₹{rmse:.0f} away from truth")
print(f"\n  MAE      : ₹{mae:.2f}")
print(f"  → Simpler average of absolute errors")
print(f"\n  MAPE     : {mape:.2f}%")
print(f"  → Average percentage error")
print(f"\n  Accuracy : {accuracy:.2f}%")
print(f"{'━'*50}")

# Rating
if mape < 2:
    print("  Rating : 🟢 EXCELLENT — under 2% average error")
elif mape < 4:
    print("  Rating : 🟡 GOOD — under 4% average error")
elif mape < 7:
    print("  Rating : 🟠 DECENT — under 7% average error")
else:
    print("  Rating : 🔴 NEEDS WORK — over 7% error")
    print("  Tip: retrain with more epochs or more data")

# ─────────────────────────────────────────────────────────
# SHOW LAST 15 PREDICTIONS IN A TABLE
# ─────────────────────────────────────────────────────────
print(f"\n━━━ Last 15 Predictions vs Actual ━━━")
print(f"{'Day':<5} {'Actual':>12} {'Predicted':>12} {'Error ₹':>10} {'Error %':>9}")
print("─" * 52)

for i, (a, p) in enumerate(zip(actuals_real[-15:], predictions_real[-15:])):
    err     = abs(a - p)
    err_pct = (err / a) * 100
    arrow   = "↑" if p > a else "↓"
    print(f"{i+1:<5} ₹{a:>10,.2f} ₹{p:>10,.2f} ₹{err:>8,.2f} {err_pct:>7.2f}% {arrow}")

# ─────────────────────────────────────────────────────────
# DRAW CHARTS
# ─────────────────────────────────────────────────────────
print("\n📊 Drawing prediction charts...")

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("LSTM Stock Price Prediction — Evaluation",
             fontsize=14, fontweight="bold")

# ── Chart 1: Full test period ──
axes[0].plot(actuals_real,
             label="Actual Price",    color="royalblue", linewidth=1.5)
axes[0].plot(predictions_real,
             label="Predicted Price", color="tomato",    linewidth=1.5, linestyle="--")
axes[0].set_title(f"Full Test Period ({len(X_test)} days) — Actual vs Predicted",
                  fontweight="bold")
axes[0].set_xlabel("Days (Test Set)")
axes[0].set_ylabel("Price (₹)")
axes[0].legend()
axes[0].grid(alpha=0.25)

# Add RMSE annotation
axes[0].annotate(f"RMSE: ₹{rmse:.2f}  |  MAPE: {mape:.2f}%  |  Accuracy: {accuracy:.2f}%",
                 xy=(0.02, 0.95), xycoords="axes fraction",
                 fontsize=9, color="darkgreen",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))

# ── Chart 2: Zoomed last 60 days ──
axes[1].plot(actuals_real[-60:],
             label="Actual Price",    color="royalblue", linewidth=2)
axes[1].plot(predictions_real[-60:],
             label="Predicted Price", color="tomato",    linewidth=2, linestyle="--")
axes[1].set_title("Last 60 Days — Zoomed In", fontweight="bold")
axes[1].set_xlabel("Days")
axes[1].set_ylabel("Price (₹)")
axes[1].legend()
axes[1].grid(alpha=0.25)

# ── Chart 3: Error per day ──
errors = np.abs(actuals_real - predictions_real)
axes[2].bar(range(len(errors)), errors,
            color=["green" if e < rmse else "tomato" for e in errors],
            alpha=0.7, width=1.0)
axes[2].axhline(rmse, color="black", linestyle="--",
                linewidth=1.5, label=f"RMSE = ₹{rmse:.2f}")
axes[2].set_title("Prediction Error Per Day (Green = below RMSE, Red = above RMSE)",
                  fontweight="bold")
axes[2].set_xlabel("Days (Test Set)")
axes[2].set_ylabel("Absolute Error (₹)")
axes[2].legend()
axes[2].grid(alpha=0.25)

plt.tight_layout()
plt.savefig("prediction_chart.png", dpi=150)
print("✅ Chart saved as prediction_chart.png")

print("\n━━━ STEP 6 COMPLETE ━━━")
print("👉 Open prediction_chart.png — 3 panels:")
print("   Panel 1 = full test period actual vs predicted")
print("   Panel 2 = zoomed last 60 days")
print("   Panel 3 = error per day (green = good, red = high error)")
print("\n👉 Ready for Step 7 — building the live dashboard!")