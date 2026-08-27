"""
Module 2 — Data Preprocessing
==============================

Responsibilities:
  • Convert Timestamp to datetime.
  • Sort data chronologically.
  • Remove duplicate records.
  • Handle missing values (interpolation).
  • Select the 5 wastewater parameters.
  • Scale parameters using StandardScaler.
  • Create 24-hour input / 24-hour output time-series sequences.
  • Split sequences into training and testing data (chronological).
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import PARAMETERS, PARAM_KEYS, LOOK_BACK, HORIZON, TEST_SIZE


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean an already-validated raw DataFrame:
    convert timestamp, sort, deduplicate, handle missing values,
    and select only the 5 parameter columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from Module 1 (must contain Timestamp + 5 params).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with only the 5 parameter columns,
        indexed by row order (Timestamp preserved separately).
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

    # Handle missing values — interpolate then clip
    param_cols = [PARAMETERS[k] for k in PARAM_KEYS]
    for col in param_cols:
        if df[col].isnull().any():
            df[col] = df[col].interpolate(method="linear", limit_direction="both")

    # Select the 5 wastewater parameters
    df_params = df[["Timestamp"] + param_cols].copy()

    print(f"Preprocessing complete. Shape: {df_params.shape}")
    return df_params


def fit_scaler(param_data: pd.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on the parameter columns.

    Parameters
    ----------
    param_data : pd.DataFrame
        DataFrame containing the 5 parameter columns.

    Returns
    -------
    StandardScaler
        Fitted scaler instance.
    """
    param_cols = [PARAMETERS[k] for k in PARAM_KEYS]
    scaler = StandardScaler()
    scaler.fit(param_data[param_cols].values)
    print("Scaler fitted on all parameter data.")
    return scaler


def scale_data(scaler: StandardScaler, param_data: pd.DataFrame) -> np.ndarray:
    """
    Transform parameter data using a fitted scaler.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted StandardScaler.
    param_data : pd.DataFrame
        DataFrame containing the 5 parameter columns.

    Returns
    -------
    np.ndarray
        Scaled data of shape (n_samples, 5).
    """
    param_cols = [PARAMETERS[k] for k in PARAM_KEYS]
    scaled = scaler.transform(param_data[param_cols].values)
    return scaled


def create_sequences(
    data: np.ndarray, look_back: int = LOOK_BACK, horizon: int = HORIZON
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create input/output sequences from a 1-D time series.

    For each starting index i:
        X[i] = data[i : i + look_back]       -> (look_back, 5)
        y[i] = data[i + look_back : i + look_back + horizon] -> (horizon, 5)

    Parameters
    ----------
    data : np.ndarray
        Scaled time-series data of shape (n_samples, n_features).
    look_back : int
        Number of past time steps as input (default 24).
    horizon : int
        Number of future time steps to predict (default 24).

    Returns
    -------
    X : np.ndarray
        Input sequences of shape (n_sequences, look_back, n_features).
    y : np.ndarray
        Target sequences of shape (n_sequences, horizon, n_features).
    """
    X, y = [], []
    n_samples = len(data)
    for i in range(n_samples - look_back - horizon + 1):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back : i + look_back + horizon])

    X = np.array(X)
    y = np.array(y)
    print(f"Created sequences: X={X.shape}, y={y.shape}")
    return X, y


def split_chronological(
    X: np.ndarray, y: np.ndarray, test_size: float = TEST_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences chronologically (no shuffling) into train/test.

    Parameters
    ----------
    X : np.ndarray
        Input sequences.
    y : np.ndarray
        Target sequences.
    test_size : float
        Fraction of data for testing (default 0.2).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(
        f"Train/test split (chronological): "
        f"train={X_train.shape[0]}, test={X_test.shape[0]}"
    )
    return X_train, X_test, y_train, y_test


def save_scaler(scaler: StandardScaler, filepath: str) -> None:
    """
    Save a fitted StandardScaler to disk using pickle.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted scaler.
    filepath : str
        Output path (.pkl).
    """
    with open(filepath, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {filepath}")


def load_scaler(filepath: str) -> StandardScaler:
    """
    Load a StandardScaler from disk.

    Parameters
    ----------
    filepath : str
        Path to the .pkl scaler file.

    Returns
    -------
    StandardScaler
    """
    with open(filepath, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def preprocess_pipeline(
    df: pd.DataFrame, scaler_path: str = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Full preprocessing pipeline:
    clean → create sequences → split → fit scaler on train only → scale.

    The scaler is fitted ONLY on training data to prevent data leakage
    from the test set into the scaling statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw validated dataset from Module 1.
    scaler_path : str, optional
        If provided, save the fitted scaler to this path.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
        Training/testing sequences and the fitted scaler.
    """
    # Step 1: Clean
    df_clean = preprocess_data(df)

    # Step 2: Create sequences from raw cleaned data
    param_cols = [PARAMETERS[k] for k in PARAM_KEYS]
    raw_values = df_clean[param_cols].values
    X, y = create_sequences(raw_values)

    # Step 3: Split chronologically BEFORE scaling
    X_train, X_test, y_train, y_test = split_chronological(X, y)

    # Step 4: Fit scaler ONLY on training data (no test leakage)
    scaler = StandardScaler()
    n_train = X_train.shape[0]
    train_rows = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(train_rows)
    print("Scaler fitted on training data only (no test leakage).")

    # Step 5: Scale train and test using the training-fitted scaler
    X_train = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_train = scaler.transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    y_test = scaler.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    print(f"Data scaled. Train: {X_train.shape}, Test: {X_test.shape}")

    # Save scaler if path provided
    if scaler_path:
        save_scaler(scaler, scaler_path)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    import os
    from modules.data_acquisition import acquire_and_validate

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "wastewater_10000.csv")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

    # Load and validate (Module 1)
    df = acquire_and_validate(DATASET_PATH, remove_dups=True)

    # Preprocess (Module 2)
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        df, scaler_path=SCALER_PATH
    )

    print("\n--- Module 2 Summary ---")
    print(f"X_train shape: {X_train.shape}  (samples, 24 hrs, 5 params)")
    print(f"y_train shape: {y_train.shape}  (samples, 24 hrs, 5 params)")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_test  shape: {y_test.shape}")
    print(f"Scaler mean: {scaler.mean_[:5]}")
    print(f"Scaler std:  {scaler.scale_[:5]}")
