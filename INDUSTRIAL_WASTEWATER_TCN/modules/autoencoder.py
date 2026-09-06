"""
Module 4 — Autoencoder Anomaly Detection
========================================

Responsibilities:
  • Build a 1D Convolutional Autoencoder for sequence reconstruction.
  • Train on historical operational sequences to learn normal operational patterns.
  • Compute reconstruction error (MSE) across all 5 wastewater parameters.
  • Calibrate an empirical anomaly threshold using the 95th percentile rule (from config.py).
  • Produce normalized anomaly scores (0.0 to 1.0) and parameter-level attribution.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from config import LOOK_BACK, PARAM_KEYS, ANOMALY_PERCENTILE


def build_autoencoder_model(
    input_shape=(LOOK_BACK, 5),
    n_features=5,
    learning_rate=0.001,
) -> tf.keras.Model:
    """
    Build a 1D Convolutional Autoencoder for time-series sequences.

    Architecture:
      Encoder:
        - Conv1D(32, 3, padding="same", activation="relu")
        - MaxPooling1D(2, padding="same") -> (12, 32)
        - Conv1D(16, 3, padding="same", activation="relu")
        - MaxPooling1D(2, padding="same") -> (6, 16) [Bottleneck: 96 dims]
      Decoder:
        - Conv1D(16, 3, padding="same", activation="relu")
        - UpSampling1D(2) -> (12, 16)
        - Conv1D(32, 3, padding="same", activation="relu")
        - UpSampling1D(2) -> (24, 32)
        - Conv1D(n_features, 3, padding="same", activation="linear") -> (24, 5)
    """
    inputs = layers.Input(shape=input_shape, name="ae_input")

    # --- Encoder ---
    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu", name="enc_conv1")(inputs)
    x = layers.BatchNormalization(name="enc_bn1")(x)
    x = layers.MaxPooling1D(pool_size=2, padding="same", name="enc_pool1")(x)

    x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.BatchNormalization(name="enc_bn2")(x)
    encoded = layers.MaxPooling1D(pool_size=2, padding="same", name="bottleneck")(x)

    # --- Decoder ---
    x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu", name="dec_conv1")(encoded)
    x = layers.BatchNormalization(name="dec_bn1")(x)
    x = layers.UpSampling1D(size=2, name="dec_up1")(x)

    x = layers.Conv1D(32, kernel_size=3, padding="same", activation="relu", name="dec_conv2")(x)
    x = layers.BatchNormalization(name="dec_bn2")(x)
    x = layers.UpSampling1D(size=2, name="dec_up2")(x)

    outputs = layers.Conv1D(n_features, kernel_size=3, padding="same", activation="linear", name="ae_reconstruction")(x)

    autoencoder = models.Model(inputs, outputs, name="Autoencoder_Wastewater")
    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return autoencoder


def train_autoencoder(
    model: tf.keras.Model,
    X_train: np.ndarray,
    X_val: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    patience: int = 8,
    model_path: str = None,
):
    """
    Train the autoencoder to reconstruct normal input sequences (X -> X).
    """
    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
    ]

    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        callback_list.append(
            callbacks.ModelCheckpoint(
                filepath=model_path, monitor="val_loss", save_best_only=True
            )
        )

    history = model.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1,
    )
    return history


def compute_reconstruction_error(
    model: tf.keras.Model, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate reconstruction errors for given sequences.

    Returns:
      overall_errors : (n_samples,) MSE per sequence
      param_errors   : (n_samples, 5) MSE per parameter across the 24 hours
      reconstructed  : (n_samples, 24, 5) predicted reconstruction
    """
    if X.ndim == 2:
        X = X[np.newaxis, ...]

    reconstructed = model.predict(X, verbose=0)
    squared_diff = np.square(X - reconstructed)

    overall_errors = np.mean(squared_diff, axis=(1, 2))
    param_errors = np.mean(squared_diff, axis=1)

    return overall_errors, param_errors, reconstructed


def calibrate_threshold(
    training_errors: np.ndarray, percentile: float = ANOMALY_PERCENTILE
) -> float:
    """
    Calibrate empirical anomaly threshold from training reconstruction errors.
    """
    threshold = float(np.percentile(training_errors, percentile))
    print(f"Calibrated Autoencoder Anomaly Threshold ({percentile}th percentile): {threshold:.6f}")
    return threshold


def detect_anomalies(
    model: tf.keras.Model,
    X: np.ndarray,
    threshold: float,
    param_keys: list = PARAM_KEYS,
) -> dict:
    """
    Run anomaly detection on a single sequence (24, 5) or batch of sequences (N, 24, 5).

    Returns a structured dictionary containing:
      - is_anomaly (bool or list of bool)
      - reconstruction_error (float or list of float)
      - anomaly_score (0.0 to 1.0 normalized value)
      - threshold (float)
      - param_contributions (dict mapping parameter to percentage contribution to total error)
      - reconstructed_sequence (np.ndarray)
    """
    is_single = (X.ndim == 2)
    if is_single:
        X = X[np.newaxis, ...]

    overall_errors, param_errors, reconstructed = compute_reconstruction_error(model, X)

    results = []
    for i in range(len(overall_errors)):
        err = float(overall_errors[i])
        is_anom = bool(err > threshold)

        # Scale anomaly score smoothly to 0.0 - 1.0 where 0.5 is at threshold
        if threshold > 0:
            raw_ratio = err / threshold
            norm_score = float(min(1.0, max(0.0, raw_ratio * 0.5)))
        else:
            norm_score = 0.0

        p_err = param_errors[i]
        sum_p = float(np.sum(p_err)) + 1e-8
        param_pct = {
            param_keys[p]: float(round((p_err[p] / sum_p) * 100.0, 2))
            for p in range(len(param_keys))
        }

        results.append({
            "is_anomaly": is_anom,
            "reconstruction_error": round(err, 6),
            "anomaly_score": round(norm_score, 4),
            "threshold": round(threshold, 6),
            "param_contributions": param_pct,
            "reconstructed": reconstructed[i].tolist(),
        })

    return results[0] if is_single else results


def save_autoencoder(model: tf.keras.Model, filepath: str):
    """Save autoencoder model weights/architecture."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Autoencoder saved to: {filepath}")


def load_autoencoder(filepath: str) -> tf.keras.Model:
    """Load autoencoder model with safe fallback."""
    try:
        model = models.load_model(filepath, compile=False)
    except Exception:
        model = build_autoencoder_model()
        model.load_weights(filepath)
    return model


def save_threshold_stats(stats: dict, filepath: str):
    """Save anomaly threshold and training error stats to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Autoencoder stats saved to: {filepath}")


def load_threshold_stats(filepath: str) -> dict:
    """Load anomaly threshold and training error stats from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)
