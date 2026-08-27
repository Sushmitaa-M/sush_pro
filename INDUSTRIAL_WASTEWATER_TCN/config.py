"""
Project Configuration
=====================

Central place for all project-wide constants and thresholds.

IMPORTANT: Thresholds are NOT derived from the reference dataset.
They are predefined based on standard industrial wastewater
quality guidelines (BIS / CPCB / standard industrial ranges).
"""

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
# Each parameter has:
#   - normal_min / normal_max : acceptable operating range
#   - warning_min / warning_max : alert band
#   - critical_min / critical_max : dangerous band
#
# A reading outside the normal range triggers a warning.
# A reading outside the warning range (into critical band)
# is flagged as Critical.

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
# Anomaly detection (Autoencoder)
# ----------------------------------------------------------------------
# Percentile of reconstruction errors above which a sample is
# considered anomalous.
ANOMALY_PERCENTILE = 95

# ----------------------------------------------------------------------
# HSRAE risk classification
# ----------------------------------------------------------------------
# Combined severity thresholds for the risk engine.
# anomaly_score is a 0-1 normalised value from the autoencoder.
ANOMALY_SCORE_WARNING = 0.02
ANOMALY_SCORE_CRITICAL = 0.05

# How far ahead (hours) a spike must be within to trigger a warning
SPIKE_HORIZON_HOURS = 24

# Risk levels
RISK_LEVELS = ["Normal", "Warning", "Critical"]
