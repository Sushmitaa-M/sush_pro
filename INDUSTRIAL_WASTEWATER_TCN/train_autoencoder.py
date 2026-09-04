"""
Train and Evaluate Module 4 Autoencoder
=======================================

Pipeline:
  1. Load dataset & preprocess time-series sequences.
  2. Split chronologically into Train / Test sets.
  3. Load or fit the StandardScaler (train only).
  4. Build and train the 1D Convolutional Autoencoder.
  5. Compute reconstruction errors on training set and calibrate 95th percentile threshold.
  6. Evaluate anomaly detection on test set and inspect anomalous samples.
  7. Save model to models/autoencoder_model.keras and stats to models/autoencoder_stats.json.
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from config import LOOK_BACK, HORIZON, PARAM_KEYS, PARAMETERS, ANOMALY_PERCENTILE
from modules.data_acquisition import acquire_and_validate
from modules.preprocessing import (
    preprocess_data, create_sequences, split_chronological, load_scaler, save_scaler
)
from modules.autoencoder import (
    build_autoencoder_model, train_autoencoder, compute_reconstruction_error,
    calibrate_threshold, detect_anomalies, save_autoencoder, save_threshold_stats
)
from sklearn.preprocessing import StandardScaler


def main():
    print("=" * 70)
    print("MODULE 4: AUTOENCODER TRAINING & CALIBRATION")
    print("=" * 70)

    # 1. Load Data
    dataset_path = os.path.join(PROJECT_ROOT, "dataset", "wastewater_10000.csv")
    df = acquire_and_validate(dataset_path, remove_dups=True)
    df_clean = preprocess_data(df)

    param_cols = [PARAMETERS[k] for k in PARAM_KEYS]
    raw_values = df_clean[param_cols].values

    # 2. Sequence creation & chronological split
    X, y = create_sequences(raw_values, look_back=LOOK_BACK, horizon=HORIZON)
    X_train, X_test, y_train, y_test = split_chronological(X, y)

    scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = load_scaler(scaler_path)
        print(f"Loaded existing scaler from: {scaler_path}")
    else:
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, 5))
        save_scaler(scaler, scaler_path)

    # Scale sequences
    X_train_sc = scaler.transform(X_train.reshape(-1, 5)).reshape(X_train.shape)
    X_test_sc = scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)

    # Train / validation split
    val_split = 0.1
    si = int(len(X_train_sc) * (1 - val_split))
    X_tr, X_val = X_train_sc[:si], X_train_sc[si:]
    print(f"Autoencoder Split: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_test_sc)}")

    # 3. Build & Train Autoencoder
    model_path = os.path.join(PROJECT_ROOT, "models", "autoencoder_model.keras")
    stats_path = os.path.join(PROJECT_ROOT, "models", "autoencoder_stats.json")

    ae_model = build_autoencoder_model(input_shape=(LOOK_BACK, 5))
    print(ae_model.summary())

    print("\nStarting Autoencoder training...")
    history = train_autoencoder(
        ae_model,
        X_tr,
        X_val,
        epochs=35,
        batch_size=64,
        patience=8,
        model_path=model_path,
    )
    print(f"Training completed. Best val_loss: {min(history.history['val_loss']):.6f}")

    # 4. Calibrate Anomaly Threshold on Training Data
    train_errors, _, _ = compute_reconstruction_error(ae_model, X_train_sc)
    threshold = calibrate_threshold(train_errors, percentile=ANOMALY_PERCENTILE)
    mean_err = float(np.mean(train_errors))
    std_err = float(np.std(train_errors))
    max_err = float(np.max(train_errors))

    stats = {
        "anomaly_threshold": round(threshold, 6),
        "percentile": ANOMALY_PERCENTILE,
        "train_mean_error": round(mean_err, 6),
        "train_std_error": round(std_err, 6),
        "train_max_error": round(max_err, 6),
        "n_train_samples": len(train_errors),
        "parameters": PARAM_KEYS,
    }
    save_threshold_stats(stats, stats_path)

    # 5. Evaluate on Test Set
    test_errors, param_test_errors, _ = compute_reconstruction_error(ae_model, X_test_sc)
    anomalies_detected = int(np.sum(test_errors > threshold))
    anomaly_rate = float(anomalies_detected / len(test_errors) * 100.0)

    print("\n" + "=" * 70)
    print("MODULE 4 EVALUATION ON TEST SET")
    print("=" * 70)
    print(f"Test samples evaluated: {len(test_errors)}")
    print(f"Anomaly threshold:     {threshold:.6f}")
    print(f"Mean test error:       {np.mean(test_errors):.6f}")
    print(f"Detected anomalies:    {anomalies_detected} ({anomaly_rate:.2f}%)")

    # Sample single prediction check
    sample_res = detect_anomalies(ae_model, X_test_sc[0], threshold)
    print("\nSample Inspection (Test sequence #0):")
    print(f"  Is Anomaly:        {sample_res['is_anomaly']}")
    print(f"  Recon Error:       {sample_res['reconstruction_error']}")
    print(f"  Anomaly Score:     {sample_res['anomaly_score']}")
    print(f"  Param attribution: {sample_res['param_contributions']}")

    print("\n" + "=" * 70)
    print("MODULE 4 AUTOENCODER READY")
    print("=" * 70)


if __name__ == "__main__":
    main()
