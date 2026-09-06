# Early Spike Prediction in Industrial Wastewater using Deep Learning

A deep learning system that uses historical wastewater time-series data to predict the next 24 hours of wastewater quality, detect abnormal patterns, and forecast critical contaminant spikes to prevent environmental compliance violations.

## Monitored Parameters
1. **pH** (acidity/alkalinity)
2. **COD** (mg/L) — Chemical Oxygen Demand
3. **BOD** (mg/L) — Biological Oxygen Demand
4. **TDS** (mg/L) — Total Dissolved Solids
5. **Temperature** (°C) — Water temperature

---

## Architecture & Modules

| Module | File | Description |
|---|---|---|
| **1 — Data Acquisition** | `modules/data_acquisition.py` | Load, validate, check missing/duplicate/chronological ordering |
| **2 — Preprocessing** | `modules/preprocessing.py` | Timestamp conversion, sorting, scaling (leak-free), 24h sliding sequences |
| **3 — TCN Prediction** | `modules/tcn_prediction.py` | Baseline TCN & Improved Spike-Aware TCN v2 with dilated causal convolutions |
| **4 — Autoencoder** | `modules/autoencoder.py` | 1D Convolutional Autoencoder for unsupervised reconstruction anomaly detection |
| **5 — HSRAE** | `modules/hsrae.py` | Hybrid Spike Risk Assessment Engine synthesizing forecasts, thresholds, and anomaly scores |
| **6 — Interactive React App** | `frontend/` & `api.py` | Modern light-mode React web dashboard with FastAPI backend & "What-If" simulator |

---

## Machine Learning Models

### 1. Improved TCN v2 (Spike-Aware Forecasting)
- **Formulation:** $\hat{y}(t+h) = y(t) + \alpha_h T(t) + \beta_h A(t) + \gamma_h S(t)$
- **Architecture:** 4 residual blocks with dilation rates `[1, 2, 4, 8]` yielding a 30-hour receptive field covering the entire 24h input window.
- **Full Temporal Decoder:** Conv1D temporal decoder preserving all 24 timesteps (eliminating bottleneck).
- **Spike-Aware Huber Loss:** Adaptive sample weighting (3.0× penalty on $z$-scores $> 2.5$).
- **Performance:** +24.0% improvement in Spike F1-score over baseline.

### 2. Module 4 Autoencoder (Reconstruction Error Anomaly Detection)
- **Architecture:** 1D Convolutional Autoencoder (32-16-16-32 filter hierarchy).
- **Threshold Calibration:** 95th percentile reconstruction loss rule (`0.16317`).
- **Attribution:** Computes per-parameter percentage contribution to total reconstruction loss.

### 3. Module 5 HSRAE (Hybrid Spike Risk Assessment Engine)
- Combines 24h TCN predictions, Autoencoder anomaly scores, CPCB/BIS regulatory thresholds, and rate-of-change dynamics into a composite **Spike Risk Index** ($0 - 100$).
- Predicts earliest **Time-to-Spike** (hours ahead).
- Generates dynamic, actionable **Plant Operator Action Protocols** for plant engineers.

---

## Project Structure
```
INDUSTRIAL_WASTEWATER_TCN/
├── config.py                  # Project thresholds and constants
├── requirements.txt           # Python dependencies
├── api.py                     # FastAPI backend service
├── train_and_evaluate.py      # Module 3 training and evaluation
├── train_v2_only.py           # Quick TCN v2 training script
├── train_autoencoder.py       # Module 4 Autoencoder training & threshold calibration
├── build_notebook.py          # Jupyter notebook builder
├── dataset/
│   └── wastewater_10000.csv   # 10,000 hourly time-series readings
├── models/
│   ├── scaler.pkl             # Fitted StandardScaler
│   ├── scaler_stats.npz       # Training mean & std for z-score calculations
│   ├── tcn_model.keras        # Baseline TCN model
│   ├── tcn_model_v2.keras     # Improved Spike-Aware TCN v2
│   ├── autoencoder_model.keras# 1D Convolutional Autoencoder
│   └── autoencoder_stats.json # Calibrated anomaly threshold metadata
├── modules/
│   ├── data_acquisition.py    # Module 1
│   ├── preprocessing.py       # Module 2
│   ├── tcn_prediction.py      # Module 3
│   ├── autoencoder.py         # Module 4
│   └── hsrae.py               # Module 5
├── notebook/
│   └── wastewater_tcn.ipynb   # Interactive analysis notebook
└── frontend/                  # Module 6: Interactive React Web App
    ├── package.json
    ├── vite.config.js         # Configured API proxy to port 8000
    ├── src/
    │   ├── App.jsx            # Main React application
    │   ├── App.css            # Light-mode minimal styling
    │   ├── index.css          # Design system tokens
    │   └── components/
    │       ├── Header.jsx
    │       ├── MetricCards.jsx
    │       ├── ForecastChart.jsx
    │       ├── HsraePanel.jsx
    │       ├── AutoencoderView.jsx
    │       ├── Simulator.jsx
    │       └── BenchmarksView.jsx
    └── dist/                  # Production-ready build bundle
```

---

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd frontend
npm install
cd ..
```

### 2. Train Models (Optional - pre-trained models included in models/)
```bash
python -m modules.data_acquisition
python train_v2_only.py
python train_autoencoder.py
```

### 3. Launch the Application

#### Backend (FastAPI):
```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

#### Frontend (React + Vite):
```bash
cd frontend
npm run dev
```
Open **http://127.0.0.1:5173/** in your browser.
