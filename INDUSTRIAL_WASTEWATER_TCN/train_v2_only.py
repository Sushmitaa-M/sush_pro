"""Minimal script: train ONLY the v2 model, then evaluate both."""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

print("Loading data...", flush=True)
from modules.data_acquisition import acquire_and_validate
from modules.preprocessing import preprocess_data, create_sequences, split_chronological
from config import LOOK_BACK, HORIZON, PARAM_KEYS, PARAMETERS

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "wastewater_10000.csv")
df = acquire_and_validate(DATASET_PATH, remove_dups=True)
df_clean = preprocess_data(df)
param_cols = [PARAMETERS[k] for k in PARAM_KEYS]
raw_values = df_clean[param_cols].values

X, y = create_sequences(raw_values)
X_train, X_test, y_train, y_test = split_chronological(X, y)

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

X_train_sc = scaler.transform(X_train.reshape(-1, 5)).reshape(X_train.shape)
X_test_sc = scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)
y_train_sc = scaler.transform(y_train.reshape(-1, 5)).reshape(y_train.shape)
y_test_sc = scaler.transform(y_test.reshape(-1, 5)).reshape(y_test.shape)

val_split = 0.1
si = int(len(X_train_sc) * (1 - val_split))
X_tr, X_val = X_train_sc[:si], X_train_sc[si:]
y_tr, y_val = y_train_sc[:si], y_train_sc[si:]
print(f"Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_test_sc)}", flush=True)

scaler_mean = scaler.mean_.astype(np.float32)
scaler_std = scaler.scale_.astype(np.float32)

print("\nBuilding v2 model...", flush=True)
from modules.tcn_prediction import build_tcn_model_v2, train_tcn_model_v2
from modules.tcn_prediction import build_tcn_model, train_tcn_model

model_v2 = build_tcn_model_v2(input_shape=(LOOK_BACK, 5))
model_v2, history_v2, sm, ss = train_tcn_model_v2(
    model_v2, X_tr, y_tr, X_val, y_val,
    scaler_mean=scaler_mean, scaler_std=scaler_std,
    epochs=15, batch_size=128, patience=8,
    model_path=os.path.join(PROJECT_ROOT, "models", "tcn_model_v2.keras"),
    stats_path=os.path.join(PROJECT_ROOT, "models", "scaler_stats.npz"),
)
print(f"\nV2 epochs trained: {len(history_v2.history['loss'])}", flush=True)
print(f"Best val_loss: {min(history_v2.history['val_loss']):.6f}", flush=True)

print("\nEvaluating both models...", flush=True)
from modules.tcn_prediction import evaluate_tcn_model, evaluate_spikes, compare_metrics

model_base = build_tcn_model(input_shape=(LOOK_BACK, 5))
model_base.load_weights(os.path.join(PROJECT_ROOT, "models", "tcn_model.keras"))
# Workaround: build with dummy data then load full model
from tensorflow.keras import models as keras_models
model_base = keras_models.load_model(os.path.join(PROJECT_ROOT, "models", "tcn_model.keras"))

results_base = evaluate_tcn_model(model_base, X_test_sc, y_test_sc)
results_v2 = evaluate_tcn_model(model_v2, X_test_sc, y_test_sc)

compare_metrics(results_base, results_v2)

print("\nSpike evaluation...", flush=True)
n_test = X_test_sc.shape[0]
y_pred_base_sc = model_base.predict(X_test_sc, verbose=0)
y_pred_v2_sc = model_v2.predict(X_test_sc, verbose=0)

all_pred_base_orig = scaler.inverse_transform(y_pred_base_sc.reshape(-1, 5)).reshape(n_test, 24, 5)
all_pred_v2_orig = scaler.inverse_transform(y_pred_v2_sc.reshape(-1, 5)).reshape(n_test, 24, 5)
all_true_orig = scaler.inverse_transform(y_test_sc.reshape(-1, 5)).reshape(n_test, 24, 5)

spike_base = evaluate_spikes(all_true_orig, all_pred_base_orig, scaler_mean, scaler_std, spike_threshold=2.5)
spike_v2 = evaluate_spikes(all_true_orig, all_pred_v2_orig, scaler_mean, scaler_std, spike_threshold=2.5)

print(f"\nSpike F1 - Baseline: {spike_base['overall']['Spike F1']:.4f}")
print(f"Spike F1 - Improved: {spike_v2['overall']['Spike F1']:.4f}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60, flush=True)
