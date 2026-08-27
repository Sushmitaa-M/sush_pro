"""
Training and evaluation script for Module 3 — Baseline vs Improved TCN.
Trains both models and produces comparison metrics.
"""
import os
import sys
import numpy as np

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from config import LOOK_BACK, HORIZON, PARAM_KEYS

# ============================================================
# 1. Load and preprocess data (with fixed scaler)
# ============================================================
print("=" * 70)
print("STEP 1: DATA LOADING AND PREPROCESSING")
print("=" * 70)

from modules.data_acquisition import acquire_and_validate
from modules.preprocessing import (
    preprocess_data, create_sequences, split_chronological, save_scaler
)

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "wastewater_10000.csv")
df = acquire_and_validate(DATASET_PATH, remove_dups=True)

df_clean = preprocess_data(df)
from config import PARAMETERS
param_cols = [PARAMETERS[k] for k in PARAM_KEYS]
raw_values = df_clean[param_cols].values

# Create sequences BEFORE scaling
X, y = create_sequences(raw_values)
print(f"Sequences: X={X.shape}, y={y.shape}")

# Split chronologically BEFORE scaling
X_train, X_test, y_train, y_test = split_chronological(X, y)

# Fit scaler ONLY on training data
scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
print("Scaler fitted on TRAINING data only (no test leakage).")

# Scale
X_train_sc = scaler.transform(X_train.reshape(-1, 5)).reshape(X_train.shape)
X_test_sc = scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)
y_train_sc = scaler.transform(y_train.reshape(-1, 5)).reshape(y_train.shape)
y_test_sc = scaler.transform(y_test.reshape(-1, 5)).reshape(y_test.shape)

# Save scaler
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
save_scaler(scaler, SCALER_PATH)

scaler_mean = scaler.mean_.astype(np.float32)
scaler_std = scaler.scale_.astype(np.float32)

# Train/val split
val_split = 0.1
si = int(len(X_train_sc) * (1 - val_split))
X_tr, X_val = X_train_sc[:si], X_train_sc[si:]
y_tr, y_val = y_train_sc[:si], y_train_sc[si:]
print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test_sc)}")

# ============================================================
# 2. Baseline TCN
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: BASELINE TCN TRAINING")
print("=" * 70)

from modules.tcn_prediction import build_tcn_model, train_tcn_model, evaluate_tcn_model

model_base = build_tcn_model(input_shape=(LOOK_BACK, 5))
history_base = train_tcn_model(
    model_base, X_tr, y_tr, X_val, y_val,
    epochs=20, batch_size=128, patience=8,
    model_path=os.path.join(PROJECT_ROOT, "models", "tcn_model.keras")
)

print(f"\nEpochs trained: {len(history_base.history['loss'])}")
print(f"Best val_loss: {min(history_base.history['val_loss']):.6f}")

results_base = evaluate_tcn_model(model_base, X_test_sc, y_test_sc)

# ============================================================
# 3. Improved TCN v2
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: IMPROVED TCN v2 TRAINING")
print("=" * 70)

from modules.tcn_prediction import (
    build_tcn_model_v2, train_tcn_model_v2,
    predict_next_24h_v2, evaluate_spikes, compare_metrics
)

model_v2 = build_tcn_model_v2(input_shape=(LOOK_BACK, 5))
model_v2, history_v2, sm, ss = train_tcn_model_v2(
    model_v2, X_tr, y_tr, X_val, y_val,
    scaler_mean=scaler_mean, scaler_std=scaler_std,
    epochs=30, batch_size=64, patience=10,
    model_path=os.path.join(PROJECT_ROOT, "models", "tcn_model_v2.keras"),
    stats_path=os.path.join(PROJECT_ROOT, "models", "scaler_stats.npz"),
)

print(f"\nEpochs trained: {len(history_v2.history['loss'])}")
print(f"Best val_loss: {min(history_v2.history['val_loss']):.6f}")

results_v2 = evaluate_tcn_model(model_v2, X_test_sc, y_test_sc)

# ============================================================
# 4. Comparison
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: METRIC COMPARISON")
print("=" * 70)

compare_metrics(results_base, results_v2)

# ============================================================
# 5. Spike Evaluation
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: SPIKE EVALUATION")
print("=" * 70)

from modules.tcn_prediction import predict_next_24h

n_test = X_test_sc.shape[0]
print(f"Generating predictions for {n_test} test samples...")

# Batch predict then inverse transform in bulk
y_pred_base_sc = model_base.predict(X_test_sc, verbose=0)
y_pred_v2_sc = model_v2.predict(X_test_sc, verbose=0)

# Inverse transform all at once (reshape to 2D for scaler, then reshape back)
all_pred_base_orig = scaler.inverse_transform(y_pred_base_sc.reshape(-1, 5)).reshape(n_test, 24, 5)
all_pred_v2_orig = scaler.inverse_transform(y_pred_v2_sc.reshape(-1, 5)).reshape(n_test, 24, 5)
all_true_orig = scaler.inverse_transform(y_test_sc.reshape(-1, 5)).reshape(n_test, 24, 5)

print("Predictions generated.")

spike_base = evaluate_spikes(
    all_true_orig, all_pred_base_orig, scaler_mean, scaler_std, spike_threshold=2.5)
spike_v2 = evaluate_spikes(
    all_true_orig, all_pred_v2_orig, scaler_mean, scaler_std, spike_threshold=2.5)

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n{'Metric':<22} {'Baseline':>12} {'Improved':>12}")
print("-" * 50)
for key in ["MSE", "MAE", "RMSE", "R2"]:
    b = results_base[key]
    v = results_v2[key]
    d = v - b
    sign = "+" if d > 0 else ""
    better = "*" if (key == "R2" and d > 0) or (key != "R2" and d < 0) else ""
    print(f"  {key:<20} {b:>12.6f} {v:>12.6f}  ({sign}{d:.4f}){better}")

print(f"\nSpike F1:")
print(f"  Baseline: {spike_base['overall']['Spike F1']:.4f}")
print(f"  Improved: {spike_v2['overall']['Spike F1']:.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
