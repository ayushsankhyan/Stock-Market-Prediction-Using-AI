# train_model.py
# Step 5 — Build, train, and save the LSTM neural network

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # suppress oneDNN warnings

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────
print("📂 Loading prepared data...")

X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

print(f"✅ X_train: {X_train.shape}")
print(f"✅ X_test : {X_test.shape}")

# ─────────────────────────────────────────────────────────
# BUILD MODEL
# ─────────────────────────────────────────────────────────
print("\n🔧 Building model...")

inputs = tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2]))
x      = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
x      = tf.keras.layers.Dropout(0.2)(x)
x      = tf.keras.layers.LSTM(32, return_sequences=False)(x)
x      = tf.keras.layers.Dropout(0.2)(x)
output = tf.keras.layers.Dense(1)(x)

model  = tf.keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.summary()
print("✅ Model built")

# ─────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────
print("\n🏋️ Training started — watch loss go DOWN each epoch...")
print("This will take 5–15 minutes on CPU. Do not close terminal.\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ─────────────────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────────────────
model.save("stock_lstm.keras")
print("\n✅ Model saved as stock_lstm.keras")

# ─────────────────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ─────────────────────────────────────────────────────────
print("📊 Saving training chart...")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(history.history["loss"],
        label="Training Loss",   color="royalblue", linewidth=2)
ax.plot(history.history["val_loss"],
        label="Validation Loss", color="tomato",    linewidth=2, linestyle="--")
ax.set_title("LSTM Training Progress — Loss Should Decrease", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (MSE)")
ax.legend()
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)

print("✅ Chart saved as training_loss.png")

print("\n━━━ STEP 5 COMPLETE ━━━")
print("👉 stock_lstm.keras saved ✅")
print("👉 Open training_loss.png to see training progress")
print("👉 Ready for Step 6!")