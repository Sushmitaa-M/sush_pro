"""
Project Configuration
=====================

Central place for all project-wide constants and thresholds.

IMPORTANT: Thresholds are NOT derived from the reference dataset.
They are predefined based on standard industrial wastewater
quality guidelines (BIS / CPCB / standard industrial ranges).
"""

import numpy as np
from typing import Tuple, Dict

# ----------------------------------------------------------------------
# Wastewater parameters
# ----------------------------------------------------------------------
# Maps a short parameter key to its exact column name in the dataset.
PARAMETERS = {
    "pH": "pH",
    "COD": "COD_mg_L",
    "BOD": "BOD_mg_L",
    "TDS": "TDS_mg_L",
    "Temperature": "Temperature_C",
}

# Ordered list of parameter keys
PARAM_KEYS = list(PARAMETERS.keys())

# ----------------------------------------------------------------------
# Sequence generation settings
# ----------------------------------------------------------------------
LOOK_BACK = 24   # Use the previous 24 hours as input
HORIZON = 24     # Predict the next 24 hours
TEST_SIZE = 0.2  # 20 % of sequences reserved for testing

# ----------------------------------------------------------------------
# Per-parameter thresholds (Normal / Warning / Critical)
# ----------------------------------------------------------------------
THRESHOLDS = {
    "pH": {
        "normal_min": 6.5,
        "normal_max": 8.5,
        "warning_min": 6.0,
        "warning_max": 9.0,
        "critical_min": 0.0,
        "critical_max": 14.0,
        "units": "",
        "description": "pH level",
    },
    "COD": {
        "normal_min": 0.0,
        "normal_max": 250.0,
        "warning_min": 250.0,
        "warning_max": 400.0,
        "critical_min": 400.0,
        "critical_max": float("inf"),
        "units": "mg/L",
        "description": "Chemical Oxygen Demand",
    },
    "BOD": {
        "normal_min": 0.0,
        "normal_max": 100.0,
        "warning_min": 100.0,
        "warning_max": 200.0,
        "critical_min": 200.0,
        "critical_max": float("inf"),
        "units": "mg/L",
        "description": "Biological Oxygen Demand",
    },
    "TDS": {
        "normal_min": 0.0,
        "normal_max": 1000.0,
        "warning_min": 1000.0,
        "warning_max": 2000.0,
        "critical_min": 2000.0,
        "critical_max": float("inf"),
        "units": "mg/L",
        "description": "Total Dissolved Solids",
    },
    "Temperature": {
        "normal_min": 10.0,
        "normal_max": 40.0,
        "warning_min": 5.0,
        "warning_max": 45.0,
        "critical_min": 0.0,
        "critical_max": 50.0,
        "units": "°C",
        "description": "Water temperature",
    },
}

# ----------------------------------------------------------------------
# Anomaly detection (Autoencoder) & Risk engine settings
# ----------------------------------------------------------------------
ANOMALY_PERCENTILE = 95
ANOMALY_SCORE_WARNING = 0.02
ANOMALY_SCORE_CRITICAL = 0.05
SPIKE_HORIZON_HOURS = 24
RISK_LEVELS = ["Normal", "Warning", "Critical"]


# ----------------------------------------------------------------------
# Physical & Dynamic Scaled Threshold Helper Functions
# ----------------------------------------------------------------------

def get_physical_spike_thresholds() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get 1D arrays of lower and upper normal operating limits for the 5 parameters
    in original physical units.

    Returns
    -------
    lower_bounds, upper_bounds : np.ndarray of shape (5,)
    """
    lower_bounds = np.array([THRESHOLDS[k]["normal_min"] for k in PARAM_KEYS], dtype=np.float32)
    upper_bounds = np.array([THRESHOLDS[k]["normal_max"] for k in PARAM_KEYS], dtype=np.float32)
    return lower_bounds, upper_bounds


def get_scaled_spike_thresholds(
    scaler, use_log1p: bool = False, log_params: list = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert physical normal thresholds into scaled space (z-scores) dynamically
    using fitted scaler parameters and optional log1p transformation.

    Parameters
    ----------
    scaler : StandardScaler or RobustScaler
        Fitted scaler instance.
    use_log1p : bool
        Whether log1p transformation was applied prior to scaling.
    log_params : list of str, optional
        Keys of parameters that were log-transformed.

    Returns
    -------
    scaled_lower, scaled_upper : np.ndarray of shape (5,)
    """
    lower_phys, upper_phys = get_physical_spike_thresholds()

    if use_log1p:
        from modules.preprocessing import get_log_param_indices, apply_log1p_transform
        log_indices = get_log_param_indices(log_params)
        lower_phys = apply_log1p_transform(lower_phys.reshape(1, -1), log_indices)[0]
        upper_phys = apply_log1p_transform(upper_phys.reshape(1, -1), log_indices)[0]

    scaled_lower = scaler.transform(lower_phys.reshape(1, -1))[0]
    scaled_upper = scaler.transform(upper_phys.reshape(1, -1))[0]

    return scaled_lower, scaled_upper
