# Early Spike Prediction in Industrial Wastewater using Deep Learning

A deep learning system that uses historical wastewater time-series data to predict the next 24 hours of wastewater quality and detect abnormal patterns and future spikes.

## Parameters Monitored
1. **pH**
2. **COD** (mg/L) — Chemical Oxygen Demand
3. **BOD** (mg/L) — Biological Oxygen Demand
4. **TDS** (mg/L) — Total Dissolved Solids
5. **Temperature** (°C)

## Architecture

| Module | File | Description |
|--------|------|-------------|
| 1 — Data Acquisition | `modules/data_acquisition.py` | Load, validate, check missing/duplicate/chronological |
| 2 — Preprocessing | `modules/preprocessing.py` | Timestamp conversion, sorting, scaling, 24h sequence generation |
| 3 — TCN Prediction | `modules/tcn_prediction.py` | Temporal Convolutional Network for 24h forecasting |
| 4 — Autoencoder | `modules/autoencoder.py` | Anomaly detection via reconstruction error |
| 5 — HSRAE | `modules/hsrae.py` | Hybrid Spike Risk Assessment Engine |
| 6 — Dashboard | `modules/dashboard.py` | Streamlit dashboard |

### TCN Model
- Causal Conv1D layers with dilated convolutions (rates 1, 2, 4)
- Residual connections
- Batch Normalization
- ReLU activations
- Temporal feature aggregation
- Trained on 24-hour input windows to predict 24-hour futures

## Project Structure
```
INDUSTRIAL_WASTEWATER_TCN/
├── config.py              # Project thresholds and parameter definitions
├── requirements.txt       # Python dependencies
├── dataset/
│   └── wastewater_10000.csv
├── models/
│   ├── tcn_model.keras    # Trained TCN model
│   └── scaler.pkl         # Fitted StandardScaler
├── modules/
│   ├── data_acquisition.py
│   ├── preprocessing.py
│   └── tcn_prediction.py
├── notebook/
│   └── wastewater_tcn.ipynb
└── app.py                 # Streamlit app entry point
```

## Setup
```bash
pip install -r requirements.txt
python -m modules.data_acquisition
python -m modules.tcn_prediction
```

## Usage
```bash
streamlit run app.py
```

## Threshold Configuration
Project thresholds are defined in `config.py` and are NOT derived from the dataset.
