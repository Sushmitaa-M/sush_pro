"""
Module 3 - TCN Prediction (Refactored Phase 3)
==============================================

Build, train, and evaluate Temporal Convolutional Networks (TCN) for predicting
the next 24 hours of all 5 wastewater parameters.

Refactoring Improvements (Phase 3):
  1. Receptive Field Expansion: kernel_size=3, dilation_rates=(1, 2, 4, 8) -> RF=61h.
  2. Aligned Multi-Horizon Projection Head: context extraction at final step t
     projecting strictly to future horizon target shape (batch, 24, 5).
  3. Asymmetric Spike-Weighted Loss (AsymmetricSpikeLoss): Custom Keras loss function
     that heavily penalizes under-prediction of high-magnitude spikes (5.0x to 10.0x weight)
     to prevent smooth mean outputs and maximize peak capture accuracy.
"""

import os
import sys
from typing import Optional, Tuple, Union

# Ensure project root is in sys.path when running standalone
if "__file__" in globals():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:
    PROJECT_ROOT = os.path.abspath(".")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import PARAM_KEYS, LOOK_BACK, HORIZON


# ======================================================================
# Custom Asymmetric Spike-Weighted Loss Function
# ======================================================================

@tf.keras.utils.register_keras_serializable(package="CustomLosses")
class AsymmetricSpikeLoss(losses.Loss):
    """
    Asymmetric Spike-Weighted Loss Function for peak optimization.

    Under-predicting high-magnitude spikes is penalized heavily (default 6.0x penalty),
    increasing gradient signal on rare, high-magnitude industrial wastewater spikes.
    Over-predicting spikes incurs a moderate penalty (2.0x weight).
    Non-spike baseline targets receive standard 1.0x MSE weighting.

    Default spike_z_threshold=2.0 to isolate true tail outliers in log-transformed
    RobustScaler space (avoids false-positive spike flags from log-compression).

    Loss Formula:
      L = mean( w_{i,h,p} * (y_{true} - y_{pred})^2 )

      w_{i,h,p} = underpredict_penalty  if y_true > threshold AND y_pred < y_true
                = overpredict_penalty   if y_true > threshold AND y_pred >= y_true
                = 1.0                   otherwise
    """

    def __init__(
        self,
        spike_z_threshold: float = 2.0,
        underpredict_penalty: float = 6.0,
        overpredict_penalty: float = 2.0,
        name: str = "asymmetric_spike_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.spike_z_threshold = float(spike_z_threshold)
        self.underpredict_penalty = float(underpredict_penalty)
        self.overpredict_penalty = float(overpredict_penalty)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Element-wise squared error
        error_sq = tf.square(y_true - y_pred)

        # Spike target mask: True where ground truth target exceeds threshold
        is_spike = y_true > self.spike_z_threshold

        # Under-prediction mask: True where model predicts lower than ground truth
        is_underpredict = y_pred < y_true

        # Determine weight per element (batch, horizon, n_features)
        weights = tf.where(
            is_spike & is_underpredict,
            self.underpredict_penalty,
            tf.where(
                is_spike & (~is_underpredict),
                self.overpredict_penalty,
                1.0,
            ),
        )

        weighted_errors = error_sq * weights
        return tf.reduce_mean(weighted_errors)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "spike_z_threshold": self.spike_z_threshold,
                "underpredict_penalty": self.underpredict_penalty,
                "overpredict_penalty": self.overpredict_penalty,
            }
        )
        return config


# ======================================================================
# Receptive Field Calculation Helper
# ======================================================================

def calculate_receptive_field(
    kernel_size: int = 3,
    dilation_rates: tuple = (1, 2, 4, 8),
    layers_per_block: int = 2,
) -> int:
    """
    Calculate the receptive field of a TCN with causal convolutions.

    RF = 1 + sum(layers_per_block * (kernel_size - 1) * dilation_rate)
    """
    rf = 1 + sum(layers_per_block * (kernel_size - 1) * d for d in dilation_rates)
    return rf


# ======================================================================
# TCN Architecture Components
# ======================================================================

def residual_block(
    x, dilation_rate: int, n_filters: int, kernel_size: int = 3, dropout_rate: float = 0.1
):
    """
    Single TCN residual block with 2 Causal Conv1D layers, BatchNorm, ReLU, and Dropout.
    """
    residual = x

    conv1 = layers.Conv1D(
        n_filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)
    conv1 = layers.Dropout(dropout_rate)(conv1)

    conv2 = layers.Conv1D(
        n_filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)
    conv2 = layers.Dropout(dropout_rate)(conv2)

    if x.shape[-1] != n_filters:
        residual = layers.Conv1D(n_filters, kernel_size=1, padding="same")(residual)

    return layers.Add()([conv2, residual])


def build_tcn_model(
    input_shape: tuple = (LOOK_BACK, 5),
    n_features: int = 5,
    n_filters: int = 64,
    kernel_size: int = 3,
    dilation_rates: tuple = (1, 2, 4, 8),
    horizon: int = HORIZON,
    dropout_rate: float = 0.1,
    loss_type: str = "asymmetric_spike",
) -> tf.keras.Model:
    """
    Build the refactored TCN architecture with expanded Receptive Field (RF=61h),
    aligned Multi-Horizon Projection Head, and Asymmetric Spike-Weighted Loss.

    Parameters
    ----------
    input_shape : tuple, default (24, 5)
        Input shape (lookback timesteps, n_features).
    n_features : int, default 5
        Number of target parameters.
    n_filters : int, default 64
        Number of convolutional filters.
    kernel_size : int, default 3
        Convolution kernel size (RF=61h with dilation 1,2,4,8).
    dilation_rates : tuple, default (1, 2, 4, 8)
        Dilation rates for 4 TCN residual blocks.
    horizon : int, default 24
        Forecast horizon timesteps to predict (t+1 ... t+24).
    dropout_rate : float, default 0.1
        Dropout rate for regularization.
    loss_type : str, default "asymmetric_spike"
        Loss function choice: 'asymmetric_spike', 'huber', or 'mse'.

    Returns
    -------
    tf.keras.Model
    """
    rf = calculate_receptive_field(kernel_size, dilation_rates, layers_per_block=2)
    print(f"Building TCN Model - Kernel Size: {kernel_size}, Dilations: {dilation_rates}")
    print(f"Calculated Receptive Field: {rf} hours (Coverage: {rf}h >= {input_shape[0]}h lookback)")

    inputs = layers.Input(shape=input_shape, name="input_sequence")
    x = inputs

    # TCN Encoder
    for rate in dilation_rates:
        x = residual_block(
            x,
            dilation_rate=rate,
            n_filters=n_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )

    # Correct Temporal Alignment & Decoder Projection Head
    last_step = layers.Lambda(lambda t: t[:, -1, :], name="last_timestep_context")(x)
    avg_pool = layers.GlobalAveragePooling1D(name="seq_avg_pooling")(x)
    max_pool = layers.GlobalMaxPooling1D(name="seq_max_pooling")(x)

    context = layers.Concatenate(axis=-1, name="context_vector")([last_step, avg_pool, max_pool])

    dense1 = layers.Dense(128, activation="relu", name="projection_dense1")(context)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(dropout_rate)(dense1)

    dense2 = layers.Dense(64, activation="relu", name="projection_dense2")(dense1)
    dense2 = layers.Dropout(dropout_rate)(dense2)

    flat_output = layers.Dense(horizon * n_features, name="flattened_forecast")(dense2)
    outputs = layers.Reshape((horizon, n_features), name="forecast_output")(flat_output)

    model = models.Model(inputs, outputs, name="TCN_MultiHorizon_Predictor")

    # Select configured loss function
    if loss_type == "asymmetric_spike":
        loss_fn = AsymmetricSpikeLoss(
            spike_z_threshold=1.5, underpredict_penalty=5.0, overpredict_penalty=2.0
        )
    elif loss_type == "huber":
        loss_fn = losses.Huber(delta=1.0)
    elif loss_type == "mse":
        loss_fn = "mse"
    else:
        raise ValueError(f"Unsupported loss_type: '{loss_type}'. Choose 'asymmetric_spike', 'huber', or 'mse'.")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=["mae"],
    )
    return model


# ======================================================================
# Spike-Aware Feature Layer & Improved TCN v2
# ======================================================================

class TemporalFeatureLayer(layers.Layer):
    """Compute first-order trend and second-order acceleration from input sequence."""

    def call(self, x):
        x_diff = x[:, 1:, :] - x[:, :-1, :]
        x_diff2 = x_diff[:, 1:, :] - x_diff[:, :-1, :]
        trend = tf.pad(x_diff, [[0, 0], [1, 0], [0, 0]])
        accel = tf.pad(x_diff2, [[0, 0], [2, 0], [0, 0]])
        return trend, accel


def build_tcn_model_v2(
    input_shape: tuple = (LOOK_BACK, 5),
    n_features: int = 5,
    n_filters: int = 64,
    kernel_size: int = 3,
    dilation_rates: tuple = (1, 2, 4, 8),
    horizon: int = HORIZON,
    dropout_rate: float = 0.1,
    loss_type: str = "asymmetric_spike",
) -> tf.keras.Model:
    """
    Build the improved TCN with explicit temporal features and Asymmetric Spike Loss.
    """
    inputs = layers.Input(shape=input_shape, name="input_sequence")

    trend, accel = TemporalFeatureLayer()(inputs)

    input_mean = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(inputs)
    input_std = layers.Lambda(lambda t: tf.math.reduce_std(t, axis=1, keepdims=True) + 1e-8)(inputs)
    spike_signal = layers.Lambda(lambda parts: (parts[0] - parts[1]) / parts[2])([inputs, input_mean, input_std])

    tcn_input = layers.Concatenate(axis=-1)([inputs, trend, accel, spike_signal])

    x = tcn_input
    for rate in dilation_rates:
        x = residual_block(
            x,
            dilation_rate=rate,
            n_filters=n_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )

    last_step = layers.Lambda(lambda t: t[:, -1, :])(x)
    avg_pool = layers.GlobalAveragePooling1D()(x)
    max_pool = layers.GlobalMaxPooling1D()(x)

    context = layers.Concatenate(axis=-1)([last_step, avg_pool, max_pool])

    h = layers.Dense(128, activation="relu")(context)
    h = layers.BatchNormalization()(h)
    h = layers.Dropout(dropout_rate)(h)

    h = layers.Dense(64, activation="relu")(h)
    flat_out = layers.Dense(horizon * n_features)(h)
    outputs = layers.Reshape((horizon, n_features))(flat_out)

    model = models.Model(inputs, outputs, name="TCN_SpikeAware_v2")

    if loss_type == "asymmetric_spike":
        loss_fn = AsymmetricSpikeLoss(
            spike_z_threshold=1.5, underpredict_penalty=5.0, overpredict_penalty=2.0
        )
    elif loss_type == "huber":
        loss_fn = losses.Huber(delta=1.0)
    elif loss_type == "mse":
        loss_fn = "mse"
    else:
        raise ValueError(f"Unsupported loss_type: '{loss_type}'")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=loss_fn,
        metrics=["mae"],
    )
    return model


# ======================================================================
# Multi-Task Learning (MTL) TCN: Dual-Head Regression + Classification
# ======================================================================

def build_mtl_tcn_model(
    input_shape: tuple = (LOOK_BACK, 5),
    n_features: int = 5,
    n_filters: int = 64,
    kernel_size: int = 3,
    dilation_rates: tuple = (1, 2, 4, 8),
    horizon: int = HORIZON,
    dropout_rate: float = 0.1,
) -> tf.keras.Model:
    """
    Build a Multi-Task Learning TCN with two parallel output heads:

      1. Regression head  ('regression')  : predicts continuous concentration values.
         Shape: (batch, horizon, n_features)

      2. Classification head ('classification'): predicts per-cell spike probability.
         Shape: (batch, horizon, n_features) with sigmoid activation.

    The shared dilated TCN encoder (RF=61h) feeds a pooled context vector that
    branches into two independent Dense projection paths.  The classification head
    provides an explicit binary gradient signal to break the recall plateau caused
    by training with regression-only losses.

    Architecture:
      Input (24, 5)
        |-- TCN Residual Blocks (dilations 1,2,4,8, kernel=3)
        |   RF = 1 + 2*(3-1)*(1+2+4+8) = 61h
        |
        +-- Context = Concat[last_step, GlobalAvgPool, GlobalMaxPool]  (3*n_filters)
              |
              +-- Dense(128->64)  [shared stem]
                    |
                    +-- reg_head:  Dense(120)->Reshape(24,5)   name='regression'
                    |
                    +-- cls_head:  Dense(120,sigmoid)->Reshape(24,5)  name='classification'

    Returns
    -------
    tf.keras.Model with output dict keys 'regression' and 'classification'.
    NOTE: Model is returned un-compiled; caller must compile with dual losses.
    """
    rf = calculate_receptive_field(kernel_size, dilation_rates, layers_per_block=2)
    print(f"Building MTL TCN Model -- Kernel={kernel_size}, Dilations={dilation_rates}, RF={rf}h")

    inputs = layers.Input(shape=input_shape, name="input_sequence")
    x = inputs

    # Shared dilated TCN encoder
    for rate in dilation_rates:
        x = residual_block(
            x,
            dilation_rate=rate,
            n_filters=n_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )

    # Shared temporal context pooling
    last_step = layers.Lambda(lambda t: t[:, -1, :], name="last_timestep_context")(x)
    avg_pool  = layers.GlobalAveragePooling1D(name="seq_avg_pooling")(x)
    max_pool  = layers.GlobalMaxPooling1D(name="seq_max_pooling")(x)
    context   = layers.Concatenate(axis=-1, name="context_vector")([last_step, avg_pool, max_pool])

    # Shared dense stem
    stem = layers.Dense(128, activation="relu", name="shared_dense1")(context)
    stem = layers.BatchNormalization(name="shared_bn1")(stem)
    stem = layers.Dropout(dropout_rate, name="shared_drop1")(stem)
    stem = layers.Dense(64, activation="relu", name="shared_dense2")(stem)
    stem = layers.Dropout(dropout_rate, name="shared_drop2")(stem)

    # --- Regression head ---
    reg = layers.Dense(horizon * n_features, name="reg_flat")(stem)
    reg_output = layers.Reshape((horizon, n_features), name="regression")(reg)

    # --- Classification head (sigmoid -> spike probability) ---
    cls = layers.Dense(horizon * n_features, activation="sigmoid", name="cls_flat")(stem)
    cls_output = layers.Reshape((horizon, n_features), name="classification")(cls)

    model = models.Model(inputs, [reg_output, cls_output], name="TCN_MTL_DualHead")
    return model


def evaluate_mtl_spikes(
    y_true_orig: np.ndarray,
    y_pred_reg_orig: np.ndarray,
    cls_probs: np.ndarray,
    thresholds: tuple = (0.20, 0.30, 0.40, 0.50),
    use_physical_thresholds: bool = True,
) -> dict:
    """
    Evaluate MTL model spike detection performance across multiple classification
    decision thresholds tau in `thresholds`.

    For each tau:
      - Spike prediction mask = cls_probs > tau  (any cell in window)
      - Compute Precision, Recall, F1 against physical threshold ground truth
      - Compute Physical Peak MAE on true-spike cells (using regression predictions)

    Parameters
    ----------
    y_true_orig : np.ndarray
        Ground truth in physical units, shape (n_samples, 24, 5).
    y_pred_reg_orig : np.ndarray
        Inverse-transformed regression predictions, shape (n_samples, 24, 5).
    cls_probs : np.ndarray
        Raw classification probabilities from sigmoid head, shape (n_samples, 24, 5).
    thresholds : tuple
        Decision thresholds to sweep.
    use_physical_thresholds : bool
        Use physical operating bounds from config.py THRESHOLDS.

    Returns
    -------
    dict keyed by tau float, each value containing overall and per-parameter metrics.
    """
    from config import THRESHOLDS, PARAM_KEYS

    n_params = y_true_orig.shape[-1]
    lower_bounds = np.zeros(n_params, dtype=np.float32)
    upper_bounds = np.zeros(n_params, dtype=np.float32)

    if use_physical_thresholds:
        for p in range(n_params):
            key = PARAM_KEYS[p]
            lower_bounds[p] = THRESHOLDS[key]["normal_min"]
            upper_bounds[p] = THRESHOLDS[key]["normal_max"]

    # Physical ground-truth spike mask
    all_true_spike_mask  = (y_true_orig < lower_bounds) | (y_true_orig > upper_bounds)
    sample_true_spikes   = np.any(all_true_spike_mask, axis=(1, 2))

    # Peak MAE on regression predictions at true-spike locations (threshold-independent)
    if np.any(all_true_spike_mask):
        peak_mae_reg = float(
            np.mean(np.abs(y_true_orig[all_true_spike_mask] - y_pred_reg_orig[all_true_spike_mask]))
        )
    else:
        peak_mae_reg = 0.0

    overall_mae  = float(np.mean(np.abs(y_true_orig - y_pred_reg_orig)))
    overall_rmse = float(np.sqrt(np.mean((y_true_orig - y_pred_reg_orig) ** 2)))

    results_by_tau = {}

    print("\n--- MTL Threshold Sweep (Physical Units) ---")
    print(f"  {'tau':>5}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  "
          f"{'PkMAE_reg':>12}  {'OvMAE':>9}  {'OvRMSE':>9}")
    print("  " + "-" * 70)

    for tau in thresholds:
        # Classification decision: flag window as spike if any cell prob > tau
        pred_spike_cells  = cls_probs > tau
        sample_pred_spikes = np.any(pred_spike_cells, axis=(1, 2))

        tp = int(np.sum(sample_true_spikes & sample_pred_spikes))
        fp = int(np.sum(~sample_true_spikes & sample_pred_spikes))
        fn = int(np.sum(sample_true_spikes & ~sample_pred_spikes))
        tn = int(np.sum(~sample_true_spikes & ~sample_pred_spikes))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        print(f"  {tau:>5.2f}  {prec:>10.4f}  {rec:>8.4f}  {f1:>8.4f}  "
              f"{peak_mae_reg:>12.2f}  {overall_mae:>9.4f}  {overall_rmse:>9.4f}")

        per_param = {}
        for p in range(n_params):
            key = PARAM_KEYS[p]
            p_true_mask = (
                (y_true_orig[:, :, p] < lower_bounds[p]) |
                (y_true_orig[:, :, p] > upper_bounds[p])
            )
            p_pred_mask = pred_spike_cells[:, :, p]

            # Window-level per parameter
            win_p_true = np.any(p_true_mask, axis=1)
            win_p_pred = np.any(p_pred_mask, axis=1)

            tp_p = int(np.sum(win_p_true & win_p_pred))
            fp_p = int(np.sum(~win_p_true & win_p_pred))
            fn_p = int(np.sum(win_p_true & ~win_p_pred))

            prec_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
            rec_p  = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0
            f1_p   = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0.0

            pk_mae_p = (
                float(np.mean(np.abs(
                    y_true_orig[:, :, p][p_true_mask] - y_pred_reg_orig[:, :, p][p_true_mask]
                )))
                if np.any(p_true_mask) else 0.0
            )
            mae_p = float(np.mean(np.abs(y_true_orig[:, :, p] - y_pred_reg_orig[:, :, p])))

            per_param[key] = {
                "Precision": round(prec_p, 4),
                "Recall": round(rec_p, 4),
                "F1": round(f1_p, 4),
                "Peak MAE": round(pk_mae_p, 4),
                "MAE": round(mae_p, 4),
                "TP": tp_p, "FP": fp_p, "FN": fn_p,
            }

        results_by_tau[tau] = {
            "overall": {
                "Spike Precision": round(prec, 4),
                "Spike Recall": round(rec, 4),
                "Spike F1": round(f1, 4),
                "Peak MAE (reg)": round(peak_mae_reg, 4),
                "Overall MAE": round(overall_mae, 4),
                "Overall RMSE": round(overall_rmse, 4),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            },
            "per_parameter": per_param,
        }

    # Per-parameter detail at each tau
    print("\nPer-parameter breakdown per tau:")
    for tau in thresholds:
        print(f"\n  tau={tau:.2f}:")
        print(f"    {'Param':<14} {'P':>7} {'R':>7} {'F1':>7} {'PkMAE':>10} {'MAE':>9}")
        print("    " + "-" * 55)
        for key in PARAM_KEYS:
            m = results_by_tau[tau]["per_parameter"][key]
            unit = THRESHOLDS[key]["units"] or ""
            print(f"    {key:<14} {m['Precision']:>7.4f} {m['Recall']:>7.4f} "
                  f"{m['F1']:>7.4f} {m['Peak MAE']:>10.2f} {m['MAE']:>9.4f}  {unit}")

    return results_by_tau


# Legacy SpikeAwareLoss maintained for loading legacy weights if needed

@tf.keras.utils.register_keras_serializable(package="CustomLosses")
class SpikeAwareLoss(losses.Loss):
    def __init__(self, train_mean, train_std, spike_factor=3.0, spike_threshold=2.5, **kwargs):
        super().__init__(**kwargs)
        self.train_mean = tf.constant(train_mean, dtype=tf.float32)
        self.train_std = tf.constant(train_std, dtype=tf.float32)
        self.spike_factor = spike_factor
        self.spike_threshold = spike_threshold

    def call(self, y_true, y_pred):
        huber = tf.keras.losses.huber(y_true, y_pred, delta=1.0)
        deviations = tf.abs(y_true - self.train_mean) / (self.train_std + 1e-8)
        max_dev = tf.reduce_max(deviations, axis=[1, 2])
        weights = tf.where(max_dev > self.spike_threshold, self.spike_factor, 1.0)
        weights = tf.expand_dims(tf.expand_dims(weights, -1), -1)
        return tf.reduce_mean(huber * weights)

    def get_config(self):
        config = super().get_config()
        config.update({
            "train_mean": self.train_mean.numpy().tolist(),
            "train_std": self.train_std.numpy().tolist(),
            "spike_factor": self.spike_factor,
            "spike_threshold": self.spike_threshold,
        })
        return config


# ======================================================================
# Training & Utility Functions
# ======================================================================

def train_tcn_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    model_path: Optional[str] = None,
):
    """Train TCN model with EarlyStopping and ReduceLROnPlateau callbacks."""
    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
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
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1,
    )
    return history


def train_tcn_model_v2(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    epochs: int = 60,
    batch_size: int = 32,
    patience: int = 15,
    model_path: Optional[str] = None,
    stats_path: Optional[str] = None,
):
    """Train improved TCN model with AsymmetricSpikeLoss."""
    asymmetric_loss = AsymmetricSpikeLoss(
        spike_z_threshold=1.5, underpredict_penalty=5.0, overpredict_penalty=2.0
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=asymmetric_loss,
        metrics=["mae"],
    )

    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
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
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callback_list,
        verbose=1,
    )

    if stats_path:
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        np.savez(stats_path, mean=scaler_mean, std=scaler_std)

    return model, history, scaler_mean, scaler_std


def predict_next_24h(model, X_input, scaler):
    """Predict next 24 hours using model and inverse transform."""
    if X_input.ndim == 2:
        X_input = X_input[np.newaxis, ...]
    y_pred_scaled = model.predict(X_input, verbose=0)

    from modules.preprocessing import inverse_transform_predictions
    y_pred = inverse_transform_predictions(y_pred_scaled[0], scaler)
    return y_pred


def predict_next_24h_v2(model, X_input, scaler):
    """Predict next 24 hours using improved model v2."""
    return predict_next_24h(model, X_input, scaler)


def save_tcn_model(model, filepath):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    model.save(filepath)
    print(f"Model saved to: {filepath}")


def load_tcn_model(filepath, safe_mode: bool = False):
    """Load model from disk with registered custom loss functions."""
    custom_objects = {
        "AsymmetricSpikeLoss": AsymmetricSpikeLoss,
        "SpikeAwareLoss": SpikeAwareLoss,
    }
    return models.load_model(filepath, custom_objects=custom_objects, safe_mode=safe_mode)


# ======================================================================
# Evaluation & Spike Metrics
# ======================================================================

def evaluate_tcn_model(model, X_test, y_test):
    """Evaluate TCN model performance metrics (MSE, MAE, RMSE, R2)."""
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

    results = {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "per_parameter": per_param,
    }

    print("\n--- TCN Test Evaluation ---")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R2   : {r2:.6f}")
    print("\nPer-parameter:")
    for key, m in per_param.items():
        print(f"  {key:12s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")
    return results


def evaluate_spikes(
    y_true_orig: np.ndarray,
    y_pred_orig: np.ndarray,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_std: Optional[np.ndarray] = None,
    use_physical_thresholds: bool = True,
    spike_threshold_z: float = 2.5,
) -> dict:
    """
    Evaluate spike detection performance in original physical units (pH, mg/L, degC).

    Compares true vs. predicted values against physical thresholds defined in config.py
    (e.g., COD > 250 mg/L, BOD > 100 mg/L, TDS > 1000 mg/L, pH outside 6.5-8.5, Temp outside 10-40 deg C).

    Parameters
    ----------
    y_true_orig : np.ndarray
        True ground truth array in ORIGINAL physical units, shape (n_samples, 24, 5).
    y_pred_orig : np.ndarray
        Predicted values array in ORIGINAL physical units, shape (n_samples, 24, 5).
    scaler_mean, scaler_std : np.ndarray, optional
        Used when use_physical_thresholds=False.
    use_physical_thresholds : bool, default True
        If True, evaluates strictly against physical limits in config.py.
    spike_threshold_z : float, default 2.5
        Z-score cutoff used when use_physical_thresholds=False.

    Returns
    -------
    dict
        Overall and per-parameter spike metrics including Peak MAE, Precision, Recall, F1.
    """
    from config import THRESHOLDS, PARAM_KEYS

    n_params = y_true_orig.shape[-1]
    lower_bounds = np.zeros(n_params, dtype=np.float32)
    upper_bounds = np.zeros(n_params, dtype=np.float32)

    for p in range(n_params):
        key = PARAM_KEYS[p]
        if use_physical_thresholds:
            lower_bounds[p] = THRESHOLDS[key]["normal_min"]
            upper_bounds[p] = THRESHOLDS[key]["normal_max"]
        else:
            if scaler_mean is not None and scaler_std is not None:
                lower_bounds[p] = scaler_mean[p] - spike_threshold_z * scaler_std[p]
                upper_bounds[p] = scaler_mean[p] + spike_threshold_z * scaler_std[p]

    # Element-wise spike masks
    all_true_spike_mask = (y_true_orig < lower_bounds) | (y_true_orig > upper_bounds)
    all_pred_spike_mask = (y_pred_orig < lower_bounds) | (y_pred_orig > upper_bounds)

    # Sample-level window spike masks (True if ANY step/param in window is a spike)
    sample_true_spikes = np.any(all_true_spike_mask, axis=(1, 2))
    sample_pred_spikes = np.any(all_pred_spike_mask, axis=(1, 2))

    tp = int(np.sum(sample_true_spikes & sample_pred_spikes))
    fp = int(np.sum(~sample_true_spikes & sample_pred_spikes))
    fn = int(np.sum(sample_true_spikes & ~sample_pred_spikes))
    tn = int(np.sum(~sample_true_spikes & ~sample_pred_spikes))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Peak MAE (MAE specifically on ground-truth spike timesteps)
    if np.any(all_true_spike_mask):
        peak_mae = float(np.mean(np.abs(y_true_orig[all_true_spike_mask] - y_pred_orig[all_true_spike_mask])))
    else:
        peak_mae = 0.0

    overall_mae = float(np.mean(np.abs(y_true_orig - y_pred_orig)))
    overall_rmse = float(np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2)))

    overall = {
        "Actual spike windows": int(np.sum(sample_true_spikes)),
        "Predicted spike windows": int(np.sum(sample_pred_spikes)),
        "True positives": tp,
        "False positives": fp,
        "False negatives": fn,
        "True negatives": tn,
        "Spike Precision": round(precision, 4),
        "Spike Recall": round(recall, 4),
        "Spike F1": round(f1, 4),
        "Peak MAE": round(peak_mae, 4),
        "Overall MAE": round(overall_mae, 4),
        "Overall RMSE": round(overall_rmse, 4),
    }

    per_param = {}
    for p in range(n_params):
        name = PARAM_KEYS[p]
        p_true_mask = (y_true_orig[:, :, p] < lower_bounds[p]) | (y_true_orig[:, :, p] > upper_bounds[p])
        p_pred_mask = (y_pred_orig[:, :, p] < lower_bounds[p]) | (y_pred_orig[:, :, p] > upper_bounds[p])

        tp_p = int(np.sum(p_true_mask & p_pred_mask))
        fp_p = int(np.sum(~p_true_mask & p_pred_mask))
        fn_p = int(np.sum(p_true_mask & ~p_pred_mask))

        prec_p = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0.0
        rec_p = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0.0
        f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0.0

        peak_mae_p = (
            float(np.mean(np.abs(y_true_orig[:, :, p][p_true_mask] - y_pred_orig[:, :, p][p_true_mask])))
            if np.any(p_true_mask)
            else 0.0
        )
        mae_p = float(np.mean(np.abs(y_true_orig[:, :, p] - y_pred_orig[:, :, p])))
        rmse_p = float(np.sqrt(np.mean((y_true_orig[:, :, p] - y_pred_orig[:, :, p]) ** 2)))

        per_param[name] = {
            "Actual spikes": int(np.sum(p_true_mask)),
            "Predicted spikes": int(np.sum(p_pred_mask)),
            "TP": tp_p,
            "FP": fp_p,
            "FN": fn_p,
            "Precision": round(prec_p, 4),
            "Recall": round(rec_p, 4),
            "F1": round(f1_p, 4),
            "Peak MAE": round(peak_mae_p, 4),
            "MAE": round(mae_p, 4),
            "RMSE": round(rmse_p, 4),
        }

    print("\n--- Spike Detection Evaluation (Physical Units) ---")
    print(f"Spike Precision: {precision:.4f}")
    print(f"Spike Recall:    {recall:.4f}")
    print(f"Spike F1:        {f1:.4f}")
    print(f"Peak MAE:        {peak_mae:.4f}")
    print(f"Overall MAE:     {overall_mae:.4f}")
    print(f"Overall RMSE:    {overall_rmse:.4f}")
    print("\nPer-parameter spike evaluation (Physical Units):")
    for name, m in per_param.items():
        unit = THRESHOLDS[name]["units"]
        unit_str = f" {unit}" if unit else ""
        print(
            f"  {name:12s}  P={m['Precision']:.3f}  R={m['Recall']:.3f}  F1={m['F1']:.3f}  "
            f"PeakMAE={m['Peak MAE']:.2f}{unit_str}  MAE={m['MAE']:.2f}{unit_str}"
        )

    return {"overall": overall, "per_parameter": per_param}


def compare_metrics(results_base, results_improved):
    """Print comparison table between baseline and improved models."""
    print("\n" + "=" * 72)
    print("BASELINE vs IMPROVED - METRIC COMPARISON")
    print("=" * 72)
    print(f"{'Metric':<22} {'Baseline':>12} {'Improved':>12} {'Delta':>12}")
    print("-" * 72)
    for key in ["MSE", "MAE", "RMSE", "R2"]:
        b = results_base[key]
        i = results_improved[key]
        d = i - b
        sign = "+" if d > 0 else ""
        print(f"  {key:<20} {b:>12.6f} {i:>12.6f} {sign}{d:>11.6f}")
    print("=" * 72)


# ======================================================================
# Dry-run Verification Execution
# ======================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 3 - TCN PREDICTION & ASYMMETRIC LOSS VERIFICATION")
    print("=" * 60)

    # 1. Test AsymmetricSpikeLoss Gradient & Penalty Behavior
    loss_func = AsymmetricSpikeLoss(
        spike_z_threshold=1.5, underpredict_penalty=5.0, overpredict_penalty=2.0
    )

    # Test cases:
    # Target 3.0 (spike > 1.5): under-predicting with 1.0 (diff -2.0, sq_err 4.0, weight 5.0 -> loss 20.0)
    # Target 3.0 (spike > 1.5): over-predicting with 4.0 (diff +1.0, sq_err 1.0, weight 2.0 -> loss 2.0)
    # Target 0.5 (normal <= 1.5): predicting 1.0 (diff +0.5, sq_err 0.25, weight 1.0 -> loss 0.25)
    y_true_test = tf.constant([[[3.0], [3.0], [0.5]]], dtype=tf.float32)
    y_pred_under = tf.constant([[[1.0], [4.0], [1.0]]], dtype=tf.float32)

    loss_under = loss_func(
        tf.constant([[[3.0]]], dtype=tf.float32),
        tf.constant([[[1.0]]], dtype=tf.float32),
    )
    loss_over = loss_func(
        tf.constant([[[3.0]]], dtype=tf.float32),
        tf.constant([[[4.0]]], dtype=tf.float32),
    )
    loss_normal = loss_func(
        tf.constant([[[0.5]]], dtype=tf.float32),
        tf.constant([[[1.0]]], dtype=tf.float32),
    )

    print(f"1. AsymmetricSpikeLoss Penalty Verification:")
    print(f"   * Spike Under-prediction Loss (y_true=3.0, y_pred=1.0): {loss_under.numpy():.4f} (Expected: 20.0000)")
    print(f"   * Spike Over-prediction Loss  (y_true=3.0, y_pred=4.0): {loss_over.numpy():.4f} (Expected: 2.0000)")
    print(f"   * Non-Spike Baseline Loss    (y_true=0.5, y_pred=1.0): {loss_normal.numpy():.4f} (Expected: 0.2500)")

    assert abs(loss_under.numpy() - 20.0) < 1e-4
    assert abs(loss_over.numpy() - 2.0) < 1e-4
    assert abs(loss_normal.numpy() - 0.25) < 1e-4
    print("   [PASS] Asymmetric loss penalty weighting verified.")

    # 2. Build TCN Model with AsymmetricSpikeLoss
    print("\n2. Building TCN Model with AsymmetricSpikeLoss...")
    model = build_tcn_model(input_shape=(LOOK_BACK, 5), loss_type="asymmetric_spike")
    print(f"   Model compiled successfully. Loss function: {model.loss.__class__.__name__}")

    # 3. Test Forward Pass with Dummy Batch Tensor
    dummy_input = np.random.randn(32, LOOK_BACK, 5).astype(np.float32)
    dummy_output = model.predict(dummy_input, verbose=0)
    print(f"\n3. Forward pass dummy output shape: {dummy_output.shape}")
    assert dummy_output.shape == (32, HORIZON, 5)
    print("   [PASS] Forward pass verified.")

    # 4. Verify Keras Serialization (Save & Load)
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, "test_tcn.keras")
    save_tcn_model(model, temp_model_path)
    loaded_model = load_tcn_model(temp_model_path)
    print(f"\n4. Keras Serialization Check:")
    print(f"   Loaded model loss class: {loaded_model.loss.__class__.__name__}")
    assert loaded_model.loss.__class__.__name__ == "AsymmetricSpikeLoss"
    print("   [PASS] Keras serialization (get_config / from_config) verified.")

    print("\n" + "=" * 60)
    print("MODULE 3 PHASE 3 VERIFICATION SUCCESSFUL")
    print("=" * 60)
