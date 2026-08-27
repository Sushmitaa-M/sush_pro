"""
Module 2 - Data Preprocessing (Refactored Phase 1)
==================================================

Responsibilities:
  * Clean raw dataset (timestamp conversion, chronological sorting, deduplication, interpolation).
  * Strict chronological train/test split of RAW time-series BEFORE scaling or sequence creation.
  * Scaler isolation (StandardScaler or RobustScaler fitted ONLY on raw training segment).
  * Optional log1p transformation for heavy-tailed parameters (COD, BOD, TDS).
  * Clean sliding-window sequence generation with proper historical context padding
    for test input features, eliminating target label data leakage.
"""

import os
import sys
import pickle
from typing import Tuple, Union, List, Optional

# Ensure project root is in sys.path when running standalone
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from config import PARAMETERS, PARAM_KEYS, LOOK_BACK, HORIZON, TEST_SIZE

# Default chemical parameters that exhibit heavy-tailed spike behavior
DEFAULT_LOG_PARAMS = ["COD", "BOD", "TDS"]


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw DataFrame: convert timestamps, sort chronologically,
    remove duplicates, interpolate missing values, and select the 5 parameter columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from Module 1.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame containing Timestamp + 5 parameter columns.
    """
    df = df.copy()

    # Convert Timestamp to datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    # Sort chronologically (required for time-series)
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate rows during preprocessing.")

    # Handle missing values - interpolate then clip
    param_cols = [PARAMETERS[k] for k in PARAM_KEYS]
    for col in param_cols:
        if df[col].isnull().any():
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    # Select Timestamp and the 5 wastewater parameters
    df_params = df[["Timestamp"] + param_cols].copy()

    print(f"Preprocessing complete. Cleaned shape: {df_params.shape}")
    return df_params


def get_log_param_indices(log_params: Optional[List[str]] = None) -> List[int]:
    """Get integer column indices for parameter keys that should receive log1p scaling."""
    if log_params is None:
        log_params = DEFAULT_LOG_PARAMS

    indices = []
    for p in log_params:
        if p in PARAM_KEYS:
            indices.append(PARAM_KEYS.index(p))
    return indices


def apply_log1p_transform(
    data: np.ndarray, log_indices: List[int]
) -> np.ndarray:
    """Apply log1p transformation to specified column indices."""
    transformed = data.copy()
    for idx in log_indices:
        transformed[..., idx] = np.log1p(np.maximum(0, transformed[..., idx]))
    return transformed


def inverse_log1p_transform(
    data: np.ndarray, log_indices: List[int]
) -> np.ndarray:
    """Apply expm1 inverse transformation to specified column indices."""
    transformed = data.copy()
    for idx in log_indices:
        transformed[..., idx] = np.expm1(transformed[..., idx])
    return transformed


def fit_scaler(
    train_data: np.ndarray, scaler_type: str = "standard"
) -> Union[StandardScaler, RobustScaler]:
    """
    Fit a StandardScaler or RobustScaler strictly on raw training data.

    Parameters
    ----------
    train_data : np.ndarray
        Raw training feature array of shape (n_train_samples, n_features).
    scaler_type : str, default "standard"
        Scaler type to fit ("standard" or "robust").

    Returns
    -------
    Scaler object (StandardScaler or RobustScaler).
    """
    if scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(
            f"Unsupported scaler_type: '{scaler_type}'. Choose 'standard' or 'robust'."
        )

    scaler.fit(train_data)

    # Ensure mean_ attribute exists for compatibility across scripts if RobustScaler is used
    if hasattr(scaler, "center_") and not hasattr(scaler, "mean_"):
        scaler.mean_ = scaler.center_

    print(
        f"{scaler.__class__.__name__} fitted strictly on raw training data "
        f"shape {train_data.shape}."
    )
    return scaler


def scale_data(
    scaler: Union[StandardScaler, RobustScaler], data: np.ndarray
) -> np.ndarray:
    """Transform parameter data using a fitted scaler."""
    shape = data.shape
    if len(shape) == 3:
        # Array of shape (samples, timesteps, features)
        reshaped = data.reshape(-1, shape[-1])
        scaled = scaler.transform(reshaped).reshape(shape)
    else:
        scaled = scaler.transform(data)
    return scaled


def create_sequences(
    data: np.ndarray, look_back: int = LOOK_BACK, horizon: int = HORIZON
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input/output sliding sequences from a 2D time series array (samples, features).

    For each starting index i:
        X[i] = data[i : i + look_back]                        -> (look_back, n_features)
        y[i] = data[i + look_back : i + look_back + horizon]  -> (horizon, n_features)

    Parameters
    ----------
    data : np.ndarray
        Time-series array of shape (n_samples, n_features).
    look_back : int
        Number of past timesteps as input (default 24).
    horizon : int
        Number of future timesteps to predict (default 24).

    Returns
    -------
    X : np.ndarray of shape (n_sequences, look_back, n_features)
    y : np.ndarray of shape (n_sequences, horizon, n_features)
    """
    X, y = [], []
    n_samples = len(data)
    for i in range(n_samples - look_back - horizon + 1):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back : i + look_back + horizon])

    X = np.array(X)
    y = np.array(y)
    return X, y


def create_test_sequences(
    scaled_train: np.ndarray,
    scaled_test: np.ndarray,
    look_back: int = LOOK_BACK,
    horizon: int = HORIZON,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test sequences without target label data leakage.

    Prepend the last `look_back` historical timesteps from `scaled_train`
    to `scaled_test` feature array so that the first test sequence (predicting starting at test index 0)
    has complete 24-hour historical context.

    Parameters
    ----------
    scaled_train : np.ndarray
        Scaled training set array of shape (n_train, n_features).
    scaled_test : np.ndarray
        Scaled testing set array of shape (n_test, n_features).
    look_back : int
        Lookback window size (default 24).
    horizon : int
        Forecast horizon (default 24).

    Returns
    -------
    X_test : np.ndarray of shape (n_test_sequences, look_back, n_features)
    y_test : np.ndarray of shape (n_test_sequences, horizon, n_features)
    """
    # Extended feature array for test inputs: last `look_back` train steps + all test steps
    extended_test_features = np.vstack([scaled_train[-look_back:], scaled_test])
    n_test = len(scaled_test)

    X_test, y_test = [], []
    for i in range(n_test - horizon + 1):
        # Input uses historical context starting from extended array
        X_test.append(extended_test_features[i : i + look_back])
        # Target uses strictly test set ground truth labels starting at index i
        y_test.append(scaled_test[i : i + horizon])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_test, y_test


def split_chronological(
    X: np.ndarray, y: np.ndarray, test_size: float = TEST_SIZE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy helper: Split pre-created sequence arrays chronologically into train/test.
    (Maintained for backwards compatibility; prefer splitting raw series in preprocess_pipeline).
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(
        f"Train/test sequence split (chronological): "
        f"train={X_train.shape[0]}, test={X_test.shape[0]}"
    )
    return X_train, X_test, y_train, y_test


def inverse_transform_predictions(
    y_pred_scaled: np.ndarray,
    scaler: Union[StandardScaler, RobustScaler],
    use_log1p: bool = False,
    log_params: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Inverse transform scaled predictions back to original physical units.

    Handles 2D arrays (samples, features) and 3D arrays (samples, timesteps, features),
    and un-applies log1p transformation if enabled.
    """
    shape = y_pred_scaled.shape
    if len(shape) == 3:
        reshaped = y_pred_scaled.reshape(-1, shape[-1])
        unscaled = scaler.inverse_transform(reshaped).reshape(shape)
    else:
        unscaled = scaler.inverse_transform(y_pred_scaled)

    if use_log1p:
        log_indices = get_log_param_indices(log_params)
        unscaled = inverse_log1p_transform(unscaled, log_indices)

    return unscaled


def save_scaler(scaler: Union[StandardScaler, RobustScaler], filepath: str) -> None:
    """Save fitted scaler to disk using pickle."""
    with open(filepath, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {filepath}")


def load_scaler(filepath: str) -> Union[StandardScaler, RobustScaler]:
    """Load scaler from disk."""
    with open(filepath, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def preprocess_pipeline(
    df: pd.DataFrame,
    scaler_path: Optional[str] = None,
    test_size: float = TEST_SIZE,
    look_back: int = LOOK_BACK,
    horizon: int = HORIZON,
    scaler_type: str = "standard",
    use_log1p: bool = False,
    log_params: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Union[StandardScaler, RobustScaler]]:
    """
    Full zero-leakage preprocessing pipeline:
      1. Clean DataFrame.
      2. Extract raw wastewater parameter matrix.
      3. Strict chronological split of RAW time series into train and test segments.
      4. Optional log1p transformation on heavy-tailed chemical parameters.
      5. Fit scaler ONLY on raw training segment.
      6. Transform train and test segments independently using training stats.
      7. Generate sliding window sequences for train and test without boundary data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated dataset from Module 1.
    scaler_path : str, optional
        Path to save fitted scaler (.pkl).
    test_size : float, default 0.2
        Fraction of raw time series reserved for testing.
    look_back : int, default 24
        Historical input sequence length.
    horizon : int, default 24
        Forecast horizon sequence length.
    scaler_type : str, default "standard"
        Choice of scaler: "standard" or "robust".
    use_log1p : bool, default False
        Whether to apply log1p transformation to heavy-tailed parameters before scaling.
    log_params : list of str, optional
        Parameter short keys to log-transform (default: COD, BOD, TDS).

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    # Step 1: Clean raw dataset
    df_clean = preprocess_data(df)

    # Step 2: Extract raw parameter values (n_samples, 5)
    param_cols = [PARAMETERS[k] for k in PARAM_KEYS]
    raw_values = df_clean[param_cols].values.astype(np.float32)

    # Step 3: Strict chronological train/test split of RAW time series BEFORE scaling
    split_idx = int(len(raw_values) * (1.0 - test_size))
    raw_train = raw_values[:split_idx].copy()
    raw_test = raw_values[split_idx:].copy()

    print(
        f"Raw data chronologically split: train_samples={len(raw_train)}, "
        f"test_samples={len(raw_test)} (split_idx={split_idx})"
    )

    # Step 4: Optional log1p transform on heavy-tailed parameters
    log_indices = get_log_param_indices(log_params)
    if use_log1p:
        print(f"Applying log1p transformation to parameters: {[PARAM_KEYS[i] for i in log_indices]}")
        raw_train = apply_log1p_transform(raw_train, log_indices)
        raw_test = apply_log1p_transform(raw_test, log_indices)

    # Step 5: Fit scaler ONLY on raw training segment (zero test data leakage)
    scaler = fit_scaler(raw_train, scaler_type=scaler_type)

    # Attach transformation metadata directly to scaler object for reference
    scaler.use_log1p = use_log1p
    scaler.log_indices = log_indices

    # Step 6: Scale train and test segments independently using training stats
    scaled_train = scaler.transform(raw_train)
    scaled_test = scaler.transform(raw_test)

    # Step 7: Create sliding window sequences
    # Training sequences: created strictly inside the training period
    X_train, y_train = create_sequences(scaled_train, look_back=look_back, horizon=horizon)

    # Testing sequences: input features receive context padding from train, target labels strictly test
    X_test, y_test = create_test_sequences(
        scaled_train, scaled_test, look_back=look_back, horizon=horizon
    )

    print(f"Generated clean sequences:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  X_test  shape: {X_test.shape}")
    print(f"  y_test  shape: {y_test.shape}")

    # Save scaler if path provided
    if scaler_path:
        save_scaler(scaler, scaler_path)

    return X_train, X_test, y_train, y_test, scaler


def create_spike_cls_targets(
    y_reg: np.ndarray,
    spike_z_threshold: float = 2.0,
) -> np.ndarray:
    """
    Generate binary spike classification mask from scaled regression targets.

    A timestep-parameter cell is labelled 1 (spike) when its scaled value exceeds
    spike_z_threshold standard deviations above zero (i.e. y_reg > spike_z_threshold),
    matching the threshold used by AsymmetricSpikeLoss in tcn_prediction.py.

    Parameters
    ----------
    y_reg : np.ndarray
        Scaled regression target array, shape (n_samples, horizon, n_features).
    spike_z_threshold : float, default 2.0
        z-score threshold above which a cell is considered a spike.

    Returns
    -------
    y_cls : np.ndarray of dtype float32, shape (n_samples, horizon, n_features)
        Binary mask: 1.0 where y_reg > spike_z_threshold, else 0.0.
    """
    y_cls = (y_reg > spike_z_threshold).astype(np.float32)
    n_spikes = int(y_cls.sum())
    n_total  = int(y_cls.size)
    print(f"Spike classification targets: {n_spikes} spike cells out of {n_total} "
          f"({100.0 * n_spikes / n_total:.2f}% positive rate, threshold={spike_z_threshold})")
    return y_cls



if __name__ == "__main__":
    import os
    from modules.data_acquisition import acquire_and_validate

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "wastewater_10000.csv")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

    # Load and validate (Module 1)
    df = acquire_and_validate(DATASET_PATH, remove_dups=True)

    # Preprocess pipeline dry-run check (Module 2)
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        df, scaler_path=SCALER_PATH, scaler_type="standard", use_log1p=False
    )

    print("\n--- Module 2 Dry-Run Verification ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_test  shape: {y_test.shape}")
    if hasattr(scaler, "mean_"):
        print(f"Scaler mean (5 features): {scaler.mean_[:5]}")
    if hasattr(scaler, "scale_"):
        print(f"Scaler scale (5 features): {scaler.scale_[:5]}")
