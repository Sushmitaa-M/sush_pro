"""
FastAPI Backend for Industrial Wastewater AI & HSRAE
====================================================

Serves deep learning models (TCN v2, Autoencoder) and HSRAE risk decision engine
to the interactive React frontend.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from config import (
    PARAMETERS, PARAM_KEYS, THRESHOLDS, LOOK_BACK, HORIZON,
    ANOMALY_PERCENTILE, ANOMALY_SCORE_WARNING, ANOMALY_SCORE_CRITICAL
)
from modules.tcn_prediction import build_tcn_model_v2
from modules.autoencoder import (
    build_autoencoder_model, compute_reconstruction_error, load_threshold_stats
)
from modules.hsrae import assess_spike_risk

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Industrial Wastewater AI Engine API",
    description="24-Hour Forecasting, Anomaly Detection & HSRAE Risk Assessment",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Model State ---
state = {
    "scaler": None,
    "tcn_model": None,
    "autoencoder_model": None,
    "ae_stats": None,
    "dataset_samples": {},
    "is_ready": False,
}


@app.on_event("startup")
def load_all_models():
    print("Loading models and preprocessing artifacts...")
    try:
        # 1. Load Scaler
        scaler_path = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
        with open(scaler_path, "rb") as f:
            state["scaler"] = pickle.load(f)
        print("StandardScaler loaded successfully.")

        # 2. Load TCN v2
        tcn_path = os.path.join(PROJECT_ROOT, "models", "tcn_model_v2.keras")
        tcn = build_tcn_model_v2(input_shape=(LOOK_BACK, 5))
        tcn.load_weights(tcn_path)
        state["tcn_model"] = tcn
        print("TCN v2 model loaded successfully.")

        # 3. Load Autoencoder
        ae_path = os.path.join(PROJECT_ROOT, "models", "autoencoder_model.keras")
        ae = build_autoencoder_model(input_shape=(LOOK_BACK, 5))
        ae.load_weights(ae_path)
        state["autoencoder_model"] = ae
        print("Autoencoder model loaded successfully.")

        # 4. Load AE Stats
        stats_path = os.path.join(PROJECT_ROOT, "models", "autoencoder_stats.json")
        state["ae_stats"] = load_threshold_stats(stats_path)
        print("Autoencoder threshold stats loaded.")

        # 5. Extract Representative Scenarios from Dataset
        prepare_sample_scenarios()

        state["is_ready"] = True
        print("All backend models loaded & initialized!")
    except Exception as e:
        print(f"Error during model startup: {e}")
        state["is_ready"] = False


def prepare_sample_scenarios():
    """Extract 4 diverse real scenarios from the historical dataset and cache full matrix for streaming."""
    csv_path = os.path.join(PROJECT_ROOT, "dataset", "wastewater_10000.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    param_cols = [PARAMETERS[k] for k in PARAM_KEYS]

    # Cache full dataset matrix and timestamps for high-speed live streaming
    state["df"] = df
    state["param_cols"] = param_cols
    state["raw_param_matrix"] = df[param_cols].values.astype(np.float32)
    state["timestamps"] = df["Timestamp"].tolist()

    # Scenario 1: Normal baseline
    normal_slice = df.iloc[100:124]

    # Scenario 2: COD spike period
    cod_spike_idx = df[df["COD_mg_L"] > 250].index
    if len(cod_spike_idx) > 0:
        start_idx = max(0, cod_spike_idx[0] - 12)
        cod_slice = df.iloc[start_idx:start_idx + 24]
    else:
        cod_slice = normal_slice

    # Scenario 3: pH anomaly period
    ph_dev_idx = df[(df["pH"] < 6.8) | (df["pH"] > 8.2)].index
    if len(ph_dev_idx) > 0:
        start_idx = max(0, ph_dev_idx[0] - 10)
        ph_slice = df.iloc[start_idx:start_idx + 24]
    else:
        ph_slice = normal_slice

    # Scenario 4: High TDS / Salt surge
    tds_idx = df[df["TDS_mg_L"] > 850].index
    if len(tds_idx) > 0:
        start_idx = max(0, tds_idx[0] - 10)
        tds_slice = df.iloc[start_idx:start_idx + 24]
    else:
        tds_slice = normal_slice

    def _format_slice(sub_df, name, desc):
        return {
            "name": name,
            "description": desc,
            "timestamps": sub_df["Timestamp"].tolist(),
            "data": sub_df[param_cols].values.tolist(),
        }

    state["dataset_samples"] = {
        "normal": _format_slice(
            normal_slice,
            "Optimal Baseline",
            "Stable hydraulic and organic conditions. All 5 parameters operating within normal regulatory bands."
        ),
        "cod_surge": _format_slice(
            cod_slice,
            "High COD Organic Load Surge",
            "Chemical Oxygen Demand surging toward critical biological treatment threshold (> 250 mg/L)."
        ),
        "ph_drop": _format_slice(
            ph_slice,
            "Acidic Influent Excursion",
            "Rapid downward trend in pH threatening biological nitrification and requiring neutralizer dosing."
        ),
        "tds_spike": _format_slice(
            tds_slice,
            "Elevated TDS Salinity Inflow",
            "Dissolved solids peak indicating heavy industrial brine or RO reject effluent discharge."
        ),
    }


# --- Request/Response Models ---
class PredictRequest(BaseModel):
    sequence: List[List[float]]  # (24, 5) float values: pH, COD, BOD, TDS, Temperature


# --- API Endpoints ---
@app.get("/api/status")
def get_status():
    return {
        "status": "ready" if state["is_ready"] else "loading",
        "models": {
            "tcn_predictor": "TCN v2 (Spike-Aware Dilated Residual Network)",
            "anomaly_detector": "1D Convolutional Autoencoder",
            "risk_engine": "HSRAE (Hybrid Spike Risk Assessment Engine)",
        },
        "parameters": PARAM_KEYS,
        "input_window_hours": LOOK_BACK,
        "forecast_horizon_hours": HORIZON,
    }


@app.get("/api/config")
def get_config():
    sanitized_th = {}
    for k, v in THRESHOLDS.items():
        item = dict(v)
        if item.get("critical_max") is not None and (np.isinf(item["critical_max"]) or item["critical_max"] == float("inf")):
            item["critical_max"] = None
        sanitized_th[k] = item

    return {
        "parameters": PARAM_KEYS,
        "parameter_details": PARAMETERS,
        "thresholds": sanitized_th,
        "anomaly_threshold": state["ae_stats"].get("anomaly_threshold") if state["ae_stats"] else 0.163,
        "anomaly_score_warning": ANOMALY_SCORE_WARNING,
        "anomaly_score_critical": ANOMALY_SCORE_CRITICAL,
    }


@app.get("/api/samples")
def get_samples():
    return state["dataset_samples"]


def run_inference_pipeline(raw_input: np.ndarray) -> dict:
    """Core inference pipeline for TCN, Autoencoder, and HSRAE."""
    scaler = state["scaler"]
    tcn = state["tcn_model"]
    ae = state["autoencoder_model"]
    ae_stats = state["ae_stats"]
    threshold = float(ae_stats["anomaly_threshold"])

    # 1. Scale input (24, 5)
    scaled_input = scaler.transform(raw_input)
    X_input = scaled_input[np.newaxis, ...]  # (1, 24, 5)

    # 2. TCN v2 Forecast
    pred_scaled = tcn.predict(X_input, verbose=0)[0]  # (24, 5)
    forecast_orig = scaler.inverse_transform(pred_scaled)  # (24, 5)

    # 3. Autoencoder Reconstruction & Anomaly Detection
    recon_scaled = ae.predict(X_input, verbose=0)[0]  # (24, 5)
    recon_orig = scaler.inverse_transform(recon_scaled)

    # Reconstruction errors in scaled space
    sq_err = np.square(scaled_input - recon_scaled)
    overall_err = float(np.mean(sq_err))
    param_err = np.mean(sq_err, axis=0)  # (5,)

    is_anom = bool(overall_err > threshold)
    norm_score = float(min(1.0, max(0.0, (overall_err / threshold) * 0.5)))

    sum_p = float(np.sum(param_err)) + 1e-8
    param_contributions = {
        PARAM_KEYS[p]: float(round((param_err[p] / sum_p) * 100.0, 2))
        for p in range(len(PARAM_KEYS))
    }

    # 4. HSRAE Risk Assessment
    hsrae_result = assess_spike_risk(
        forecast_24h=forecast_orig,
        autoencoder_score=norm_score,
        current_values=raw_input[-1],
        param_keys=PARAM_KEYS,
    )

    # Prepare timeline response
    timeline = []
    for h in range(HORIZON):
        point = {
            "hour": h + 1,
            "label": f"+{h + 1}h",
        }
        for p, key in enumerate(PARAM_KEYS):
            point[f"{key}_forecast"] = round(float(forecast_orig[h, p]), 2)
        timeline.append(point)

    history_timeline = []
    for h in range(LOOK_BACK):
        point = {
            "hour": h - 23,
            "label": f"{h - 23}h",
        }
        for p, key in enumerate(PARAM_KEYS):
            point[f"{key}_actual"] = round(float(raw_input[h, p]), 2)
            point[f"{key}_recon"] = round(float(recon_orig[h, p]), 2)
        history_timeline.append(point)

    return {
        "current_values": {
            PARAM_KEYS[p]: round(float(raw_input[-1, p]), 2)
            for p in range(len(PARAM_KEYS))
        },
        "forecast_timeline": timeline,
        "history_timeline": history_timeline,
        "autoencoder": {
            "is_anomaly": is_anom,
            "reconstruction_error": round(overall_err, 5),
            "anomaly_score": round(norm_score, 4),
            "threshold": round(threshold, 5),
            "param_contributions": param_contributions,
        },
        "hsrae": hsrae_result,
    }


@app.post("/api/predict")
def predict_pipeline(req: PredictRequest):
    if not state["is_ready"]:
        raise HTTPException(status_code=503, detail="Models are still initializing. Please wait.")

    raw_input = np.array(req.sequence, dtype=np.float32)
    if raw_input.shape != (24, 5):
        raise HTTPException(
            status_code=400,
            detail=f"Expected input sequence of shape (24, 5), got {raw_input.shape}",
        )

    return run_inference_pipeline(raw_input)


@app.get("/api/stream/tick")
def stream_tick(index: int = 120, noise: float = 0.0, step: int = 1):
    """
    Auto Live Stream: Extracts a real 24-hour window from the dataset at the given row index,
    optionally applies realistic sensor noise, and returns the full pipeline forecast.
    """
    if not state["is_ready"]:
        raise HTTPException(status_code=503, detail="Models are still initializing. Please wait.")

    total_rows = len(state["timestamps"]) if state["timestamps"] else 0
    if total_rows == 0:
        raise HTTPException(status_code=500, detail="Dataset not loaded for streaming.")

    # Wrap index within valid bounds
    idx = max(23, min(total_rows - 1, index))

    # Slice the real 24-hour window (hours idx-23 through idx)
    raw_window = state["raw_param_matrix"][idx - 23 : idx + 1].copy()

    # Optional realistic sensor noise (transducer micro-fluctuations)
    if noise > 0.0:
        noise_factor = float(min(0.08, max(0.0, noise)))
        jitter = np.random.normal(0, noise_factor, size=raw_window.shape)
        raw_window = raw_window * (1.0 + jitter)
        # Keep physical bounds: pH [4, 10], non-negative for others
        raw_window[:, 0] = np.clip(raw_window[:, 0], 4.0, 10.0)
        raw_window[:, 1:] = np.maximum(0.0, raw_window[:, 1:])

    result = run_inference_pipeline(raw_window)
    result["timestamp"] = state["timestamps"][idx]
    result["current_index"] = idx
    result["next_index"] = (idx + step) if (idx + step) < total_rows else 24
    result["total_rows"] = total_rows
    result["noise_applied"] = noise > 0.0
    return result


@app.get("/api/metrics")
def get_model_metrics():
    return {
        "baseline_vs_v2": [
            {"metric": "MSE", "baseline": 0.354120, "improved_v2": 0.281400, "delta": "-20.5%", "status": "improved"},
            {"metric": "MAE", "baseline": 0.372100, "improved_v2": 0.312800, "delta": "-15.9%", "status": "improved"},
            {"metric": "RMSE", "baseline": 0.595080, "improved_v2": 0.530471, "delta": "-10.9%", "status": "improved"},
            {"metric": "R² Score", "baseline": 0.645900, "improved_v2": 0.718600, "delta": "+11.3%", "status": "improved"},
            {"metric": "Spike Precision", "baseline": 0.6820, "improved_v2": 0.8140, "delta": "+19.4%", "status": "improved"},
            {"metric": "Spike Recall", "baseline": 0.6140, "improved_v2": 0.7890, "delta": "+28.5%", "status": "improved"},
            {"metric": "Spike F1-Score", "baseline": 0.6462, "improved_v2": 0.8012, "delta": "+24.0%", "status": "improved"},
        ],
        "autoencoder_metrics": {
            "reconstruction_val_loss": 0.105456,
            "anomaly_threshold_95pct": 0.163175,
            "test_sample_anomaly_rate": "5.88%",
            "architecture": "1D Convolutional Autoencoder (32-16-16-32)",
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
