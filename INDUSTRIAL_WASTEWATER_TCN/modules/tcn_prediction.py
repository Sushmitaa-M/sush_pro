"""
Module 3 — TCN Prediction
==========================

Build a Temporal Convolutional Network (TCN) that predicts the next
24 hours of all 5 wastewater parameters.

This module provides:
  1. Baseline TCN (original) — for comparison
  2. Improved TCN with project-specific Spike-Aware Forecasting formulation

Forecasting Formulation:
  y_hat(t+h) = y(t) + alpha_h * T(t) + beta_h * A(t) + gamma_h * S(t)

  where:
    y(t)  = latest observed value (implicit in TCN features)
    T(t)  = temporal trend from recent history
    A(t)  = acceleration (change in trend)
    S(t)  = spike/anomaly signal derived from input window

  All components (alpha_h, beta_h, gamma_h) are LEARNED by the TCN,
  not manually fixed.

Architecture (improved):
  • 4 TCN residual blocks with dilation rates [1, 2, 4, 8]
  • Receptive field covers full 24-hour input window
  • Temporal decoder preserves all 24 timesteps (no x[:, -1, :] bottleneck)
  • Explicit trend + acceleration features concatenated with TCN output
  • Spike-aware weighted loss for improved spike sensitivity
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import PARAM_KEYS, LOOK_BACK, HORIZON


# ======================================================================
# Baseline TCN (original) — kept for comparison
# ======================================================================

def residual_block(x, dilation_rate, n_filters, kernel_size):
    """Single TCN residual block (baseline)."""
    residual = x
    conv1 = layers.Conv1D(n_filters, kernel_size, padding="causal",
                          dilation_rate=dilation_rate)(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv2 = layers.Conv1D(n_filters, kernel_size, padding="causal",
                          dilation_rate=dilation_rate)(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    if x.shape[-1] != n_filters:
        residual = layers.Conv1D(n_filters, kernel_size=1, padding="same")(residual)
    return layers.Add()([conv2, residual])


def build_tcn_model(
    input_shape=(LOOK_BACK, 5),
    n_features=5,
    n_filters=64,
    kernel_size=2,
    dilation_rates=(1, 2, 4),
    horizon=HORIZON,
) -> tf.keras.Model:
    """Build the baseline TCN model (original architecture)."""
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for rate in dilation_rates:
        x = residual_block(x, dilation_rate=rate, n_filters=n_filters,
                           kernel_size=kernel_size)
        x = layers.Dropout(0.1)(x)
    x = x[:, -1, :]
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(horizon * n_features)(x)
    outputs = layers.Reshape((horizon, n_features))(x)
    model = models.Model(inputs, outputs, name="TCN_Predictor")
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])
    print("Baseline TCN Model Architecture:")
    print(model.summary())
    return model


def train_tcn_model(
    model, X_train, y_train, X_val, y_val,
    epochs=50, batch_size=32, patience=10, model_path=None,
):
    """Train the baseline TCN model."""
    callback_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-6),
    ]
    if model_path:
        callback_list.append(
            callbacks.ModelCheckpoint(filepath=model_path, monitor="val_loss",
                                      save_best_only=True))
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callback_list, verbose=1)
    return history


def predict_next_24h(model, X_input, scaler):
    """Predict next 24 hours using the baseline model."""
    if X_input.ndim == 2:
        X_input = X_input[np.newaxis, ...]
    y_pred_scaled = model.predict(X_input, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled[0])
    return y_pred


def save_tcn_model(model, filepath):
    """Save the trained TCN model to disk."""
    model.save(filepath)
    print(f"TCN model saved to: {filepath}")


def load_tcn_model(filepath):
    """Load a TCN model from disk."""
    return models.load_model(filepath)


# ======================================================================
# Improved TCN — Spike-Aware Forecasting
# ======================================================================

class TemporalFeatureLayer(layers.Layer):
    """
    Compute trend and acceleration from the input sequence.

    Trend(t)  = x(t) - x(t-1)        (first-order difference)
    Accel(t)  = Trend(t) - Trend(t-1) (second-order difference)

    Both are zero-padded to preserve the original sequence length.
    No future values are used — only information from the input window.
    """

    def call(self, x):
        x_diff = x[:, 1:, :] - x[:, :-1, :]
        x_diff2 = x_diff[:, 1:, :] - x_diff[:, :-1, :]
        trend = tf.pad(x_diff, [[0, 0], [1, 0], [0, 0]])
        accel = tf.pad(x_diff2, [[0, 0], [2, 0], [0, 0]])
        return trend, accel


def spike_residual_block(x, dilation_rate, n_filters, kernel_size):
    """TCN residual block for the improved architecture."""
    residual = x
    conv1 = layers.Conv1D(n_filters, kernel_size, padding="causal",
                          dilation_rate=dilation_rate)(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv2 = layers.Conv1D(n_filters, kernel_size, padding="causal",
                          dilation_rate=dilation_rate)(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    if x.shape[-1] != n_filters:
        residual = layers.Conv1D(n_filters, kernel_size=1, padding="same")(residual)
    return layers.Add()([conv2, residual])


def temporal_decoder_block(x, n_filters, kernel_size, dilation_rate):
    """Temporal decoder block: Conv1D + BatchNorm + ReLU + Dropout."""
    x = layers.Conv1D(n_filters, kernel_size, padding="causal",
                      dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)
    return x


def build_tcn_model_v2(
    input_shape=(LOOK_BACK, 5),
    n_features=5,
    n_filters=64,
    kernel_size=2,
    dilation_rates=(1, 2, 4, 8),
    horizon=HORIZON,
) -> tf.keras.Model:
    """
    Build the improved TCN with Spike-Aware Forecasting formulation.

    Forecasting formulation:
      y_hat(t+h) = y(t) + alpha_h * T(t) + beta_h * A(t) + gamma_h * S(t)

      All coefficients (alpha, beta, gamma) are learned by the network.

    Architecture improvements over baseline:
      1. 4 TCN blocks with dilation [1,2,4,8] -> receptive field = 30 (full 24h)
      2. Temporal decoder with Conv1D preserves all 24 timesteps
      3. Explicit trend (T) and acceleration (A) features concatenated
      4. Spike signal (S) derived from input window statistics
      5. No x[:, -1, :] bottleneck — full temporal information flows to output
    """
    inputs = layers.Input(shape=input_shape)

    # --- Explicit feature computation via Keras layers ---
    trend, accel = TemporalFeatureLayer()(inputs)

    # Spike signal: z-score of each timestep relative to input statistics
    input_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(inputs)
    input_std = layers.Lambda(lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-8)(inputs)
    spike_signal = layers.Lambda(lambda parts: (parts[0] - parts[1]) / parts[2])([inputs, input_mean, input_std])

    # Stack: [input(5), trend(5), accel(5), spike(5)] = 20 features
    tcn_input = layers.Concatenate(axis=-1)([inputs, trend, accel, spike_signal])

    # --- TCN encoder: 4 residual blocks, dilation [1,2,4,8] ---
    x = tcn_input
    for rate in dilation_rates:
        x = spike_residual_block(x, dilation_rate=rate, n_filters=n_filters,
                                 kernel_size=kernel_size)
        x = layers.Dropout(0.1)(x)

    # --- Temporal decoder: preserves 24 timesteps ---
    # (unlike baseline which collapsed to x[:, -1, :])
    x = temporal_decoder_block(x, n_filters=64, kernel_size=3, dilation_rate=1)
    x = temporal_decoder_block(x, n_filters=32, kernel_size=3, dilation_rate=2)

    # --- Context features ---
    # Global context vector (summary of entire sequence)
    global_context = layers.GlobalAveragePooling1D()(x)

    # Per-timestep trend and acceleration context (last values, broadcast)
    trend_ctx = layers.Lambda(lambda t: t[:, -1:, :])(trend)
    accel_ctx = layers.Lambda(lambda t: t[:, -1:, :])(accel)
    trend_ctx = layers.Lambda(lambda t: tf.tile(t, [1, horizon, 1]))(trend_ctx)
    accel_ctx = layers.Lambda(lambda t: tf.tile(t, [1, horizon, 1]))(accel_ctx)

    # --- Output head: full temporal output ---
    temporal_out = layers.Reshape((horizon, -1))(x)   # (batch, 24, 32)
    global_rep = layers.Lambda(lambda t: tf.repeat(t[:, tf.newaxis, :], horizon, axis=1))(global_context)
    # global_rep is already (batch, horizon, 32) from tf.repeat

    combined = layers.Concatenate(axis=-1)(
        [temporal_out, global_rep, trend_ctx, accel_ctx])  # 32+32+5+5=74

    x = layers.Dense(128, activation="relu")(combined)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(n_features)(x)  # (batch, 24, 5) directly

    model = models.Model(inputs, outputs, name="TCN_SpikeAware_v2")
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss="mse", metrics=["mae"])

    print("Improved TCN (Spike-Aware) Architecture:")
    print(model.summary())
    return model


# ======================================================================
# Spike-Aware Loss Function
# ======================================================================

class SpikeAwareLoss(losses.Loss):
    """
    Spike-Aware Huber loss with adaptive sample weighting.

    Samples where any parameter deviates significantly from the training
    mean receive higher weight, encouraging the model to learn spike
    patterns rather than simply predicting the average.

    Uses Huber loss as the base (robust to outliers) combined with
    spike-weighting based on training statistics.
    """

    def __init__(self, train_mean, train_std, spike_factor=3.0,
                 spike_threshold=2.5, **kwargs):
        super().__init__(**kwargs)
        self.train_mean = tf.constant(train_mean, dtype=tf.float32)
        self.train_std = tf.constant(train_std, dtype=tf.float32)
        self.spike_factor = spike_factor
        self.spike_threshold = spike_threshold

    def call(self, y_true, y_pred):
        huber = tf.keras.losses.huber(y_true, y_pred, delta=1.0)

        # Per-sample deviation: max z-score across all parameters and timesteps
        deviations = tf.abs(y_true - self.train_mean) / (self.train_std + 1e-8)
        max_dev = tf.reduce_max(deviations, axis=[1, 2])

        # Weight: 1.0 for normal, spike_factor for spikes
        weights = tf.where(max_dev > self.spike_threshold,
                           self.spike_factor, 1.0)
        weights = tf.expand_dims(tf.expand_dims(weights, -1), -1)

        return tf.reduce_mean(huber * weights)


# ======================================================================
# Training (Improved Model)
# ======================================================================

def train_tcn_model_v2(
    model, X_train, y_train, X_val, y_val,
    scaler_mean, scaler_std,
    epochs=60, batch_size=32, patience=15,
    model_path=None, stats_path=None,
):
    """
    Train the improved TCN model with spike-aware loss.

    Parameters
    ----------
    model : tf.keras.Model
        The improved TCN model.
    X_train, y_train : np.ndarray
        Training sequences (scaled).
    X_val, y_val : np.ndarray
        Validation sequences (scaled).
    scaler_mean, scaler_std : np.ndarray
        Training set mean and std per parameter (for spike detection).
    epochs : int
        Maximum training epochs.
    batch_size : int
        Batch size.
    patience : int
        Early stopping patience.
    model_path : str, optional
        Path to save the best model.
    stats_path : str, optional
        Path to save scaler statistics (.npz).

    Returns
    -------
    model, history, scaler_mean, scaler_std
    """
    # Spike-aware loss: data is already standardized (mean=0, std=1)
    # so pass zeros/ones. A "spike" in scaled space = z-score > 2.5
    n_features = len(scaler_mean)
    spike_loss = SpikeAwareLoss(
        train_mean=np.zeros(n_features, dtype=np.float32),
        train_std=np.ones(n_features, dtype=np.float32),
        spike_factor=3.0,
        spike_threshold=2.5,
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=spike_loss, metrics=["mae"])

    callback_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=7, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1,
    )

    # Save model
    if model_path:
        model.save(model_path)
        print(f"Improved TCN model saved to: {model_path}")

    # Save scaler statistics
    if stats_path:
        np.savez(stats_path, mean=scaler_mean, std=scaler_std)
        print(f"Scaler stats saved to: {stats_path}")

    return model, history, scaler_mean, scaler_std


def predict_next_24h_v2(model, X_input, scaler):
    """Predict next 24 hours using the improved model."""
    if X_input.ndim == 2:
        X_input = X_input[np.newaxis, ...]
    y_pred_scaled = model.predict(X_input, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled[0])
    return y_pred


# ======================================================================
# Evaluation
# ======================================================================

def evaluate_tcn_model(model, X_test, y_test):
    """Evaluate TCN model on test set (works for both baseline and improved)."""
    y_pred = model.predict(X_test, verbose=0)
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])

    mse = mean_squared_error(y_test_flat, y_pred_flat)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_flat, y_pred_flat, multioutput="variance_weighted")

    per_param = {}
    for i, key in enumerate(PARAM_KEYS):
        per_param[key] = {
            "MAE": float(mean_absolute_error(y_test_flat[:, i], y_pred_flat[:, i])),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_flat[:, i], y_pred_flat[:, i]))),
            "R2": float(r2_score(y_test_flat[:, i], y_pred_flat[:, i])),
        }

    results = {"MSE": float(mse), "MAE": float(mae), "RMSE": float(rmse),
               "R2": float(r2), "per_parameter": per_param}

    print("\n--- TCN Test Evaluation ---")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")
    print("\nPer-parameter:")
    for key, m in per_param.items():
        print(f"  {key:12s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R²={m['R2']:.4f}")
    return results


# ======================================================================
# Spike Evaluation
# ======================================================================

def evaluate_spikes(y_true_orig, y_pred_orig, scaler_mean, scaler_std,
                    spike_threshold=2.5):
    """
    Evaluate spike detection performance using training-set statistics.

    A timestep is flagged as a spike if ANY parameter's value deviates
    from the training mean by more than `spike_threshold` standard deviations.

    Parameters
    ----------
    y_true_orig : np.ndarray
        True values in ORIGINAL scale, shape (n_samples, 24, 5).
    y_pred_orig : np.ndarray
        Predicted values in ORIGINAL scale, shape (n_samples, 24, 5).
    scaler_mean, scaler_std : np.ndarray
        Training set per-parameter mean and std.
    spike_threshold : float
        Z-score threshold for spike detection.

    Returns
    -------
    dict
        Overall and per-parameter spike metrics.
    """
    thresholds_upper = scaler_mean + spike_threshold * scaler_std
    thresholds_lower = scaler_mean - spike_threshold * scaler_std

    n_params = y_true_orig.shape[-1]

    def _spike_mask(arr):
        """Return (n_samples,) bool: True if any param/time is spike."""
        mask = np.zeros(arr.shape[0], dtype=bool)
        for p in range(n_params):
            hi = arr[:, :, p] > thresholds_upper[p]
            lo = arr[:, :, p] < thresholds_lower[p]
            mask |= np.any(hi | lo, axis=1)
        return mask

    actual_spikes = _spike_mask(y_true_orig)
    predicted_spikes = _spike_mask(y_pred_orig)

    n_actual = int(np.sum(actual_spikes))
    n_predicted = int(np.sum(predicted_spikes))
    tp = int(np.sum(actual_spikes & predicted_spikes))
    fp = int(np.sum(~actual_spikes & predicted_spikes))
    fn = int(np.sum(actual_spikes & ~predicted_spikes))
    tn = int(np.sum(~actual_spikes & ~predicted_spikes))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    overall = {
        "Actual spike windows": n_actual,
        "Predicted spike windows": n_predicted,
        "True positives": tp,
        "False positives": fp,
        "False negatives": fn,
        "True negatives": tn,
        "Spike precision": round(precision, 4),
        "Spike recall": round(recall, 4),
        "Spike F1": round(f1, 4),
    }

    per_param = {}
    for p in range(n_params):
        name = PARAM_KEYS[p]
        tp_p = int(np.sum((actual_spikes) & (predicted_spikes)))
        act_p = int(np.sum(y_true_orig[:, :, p] > thresholds_upper[p]))
        pred_p = int(np.sum(y_pred_orig[:, :, p] > thresholds_upper[p]))
        tp_pp = int(np.sum(
            (y_true_orig[:, :, p] > thresholds_upper[p]) &
            (y_pred_orig[:, :, p] > thresholds_upper[p])))
        fp_pp = int(np.sum(
            (y_true_orig[:, :, p] <= thresholds_upper[p]) &
            (y_pred_orig[:, :, p] > thresholds_upper[p])))
        fn_pp = int(np.sum(
            (y_true_orig[:, :, p] > thresholds_upper[p]) &
            (y_pred_orig[:, :, p] <= thresholds_upper[p])))
        prec_p = tp_pp / (tp_pp + fp_pp) if (tp_pp + fp_pp) > 0 else 0.0
        rec_p = tp_pp / (tp_pp + fn_pp) if (tp_pp + fn_pp) > 0 else 0.0
        f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0.0
        per_param[name] = {
            "Actual spikes": act_p,
            "Predicted spikes": pred_p,
            "TP": tp_pp, "FP": fp_pp, "FN": fn_pp,
            "Precision": round(prec_p, 4),
            "Recall": round(rec_p, 4),
            "F1": round(f1_p, 4),
        }

    print("\n--- Spike Detection Evaluation ---")
    print(f"Actual spike windows:     {n_actual}")
    print(f"Predicted spike windows:  {n_predicted}")
    print(f"True positives:           {tp}")
    print(f"False positives:          {fp}")
    print(f"Missed (false negatives): {fn}")
    print(f"Spike Precision:  {precision:.4f}")
    print(f"Spike Recall:     {recall:.4f}")
    print(f"Spike F1:         {f1:.4f}")
    print("\nPer-parameter spike detection:")
    for name, m in per_param.items():
        print(f"  {name:12s}  Act={m['Actual spikes']:3d}  "
              f"Pred={m['Predicted spikes']:3d}  "
              f"P={m['Precision']:.3f}  R={m['Recall']:.3f}  F1={m['F1']:.3f}")

    return {"overall": overall, "per_parameter": per_param}


def compare_metrics(results_base, results_improved):
    """Print a side-by-side comparison of baseline vs improved metrics."""
    print("\n" + "=" * 72)
    print("BASELINE vs IMPROVED — METRIC COMPARISON")
    print("=" * 72)
    print(f"{'Metric':<22} {'Baseline':>12} {'Improved':>12} {'Delta':>12} {'Change':>8}")
    print("-" * 72)

    for key in ["MSE", "MAE", "RMSE", "R2"]:
        b = results_base[key]
        i = results_improved[key]
        d = i - b
        pct = (d / abs(b) * 100) if b != 0 else 0
        sign = "+" if d > 0 else ""
        marker = " *" if (key == "R2" and d > 0) or (key != "R2" and d < 0) else ""
        print(f"  {key:<20} {b:>12.6f} {i:>12.6f} {sign}{d:>11.6f} {sign}{pct:>6.1f}%{marker}")

    print("-" * 72)
    print("\nPer-parameter comparison:")
    print(f"  {'Param':<12} {'Metric':<6} {'Baseline':>10} {'Improved':>10} {'Delta':>10}")
    print("  " + "-" * 54)
    for key in PARAM_KEYS:
        for metric in ["MAE", "RMSE", "R2"]:
            b = results_base["per_parameter"][key][metric]
            i = results_improved["per_parameter"][key][metric]
            d = i - b
            sign = "+" if d > 0 else ""
            print(f"  {key:<12} {metric:<6} {b:>10.4f} {i:>10.4f} {sign}{d:>9.4f}")
        print()
    print("=" * 72)


# ======================================================================
# Main pipeline (baseline)
# ======================================================================

if __name__ == "__main__":
    from modules.data_acquisition import acquire_and_validate
    from modules.preprocessing import (
        preprocess_pipeline, fit_scaler, scale_data, save_scaler, load_scaler,
    )

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "wastewater_10000.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "tcn_model.keras")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

    print("=" * 60)
    print("DATA PREPARATION (Modules 1 & 2)")
    print("=" * 60)
    df = acquire_and_validate(DATASET_PATH, remove_dups=True)
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        df, scaler_path=SCALER_PATH)

    val_split = 0.1
    split_idx = int(len(X_train) * (1 - val_split))
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    print(f"\nFinal split: train={len(X_tr)}, val={len(X_val)}, test={len(X_test)}")

    print("\n" + "=" * 60)
    print("BUILD TCN MODEL (Module 3 — Baseline)")
    print("=" * 60)
    model = build_tcn_model(input_shape=(LOOK_BACK, 5))

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    history = train_tcn_model(model, X_tr, y_tr, X_val, y_val,
                              epochs=40, batch_size=64, patience=10,
                              model_path=MODEL_PATH)

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    results = evaluate_tcn_model(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("MODULE 3 BASELINE COMPLETE")
    print("=" * 60)
    save_tcn_model(model, MODEL_PATH)
