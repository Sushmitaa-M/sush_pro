# Early Spike Prediction in Industrial Wastewater using Deep Learning

A deep learning system that uses historical wastewater time-series data to predict the next 24 hours of wastewater quality and detect abnormal patterns and future spikes.

## Parameters Monitored
1. **pH** (normal range: 6.5 - 8.5)
2. **COD** (mg/L) - Chemical Oxygen Demand (threshold: <= 250 mg/L)
3. **BOD** (mg/L) - Biological Oxygen Demand (threshold: <= 100 mg/L)
4. **TDS** (mg/L) - Total Dissolved Solids (threshold: <= 1000 mg/L)
5. **Temperature** (deg C) - (normal range: 10 - 40 deg C)

## Architecture

| Module / Pipeline | File | Description |
|:---|:---|:---|
| 1 - Data Acquisition | `modules/data_acquisition.py` | Load, validate, check missing/duplicate/chronological ordering |
| 2 - Preprocessing | `modules/preprocessing.py` | Chronological splitting, log1p transformation, RobustScaler, sliding-window generation |
| 3 - TCN Prediction | `modules/tcn_prediction.py` | Dilated TCN backbone, Multi-Task projection heads (regression + classification) |
| 4 - Training Pipeline | `train_and_evaluate.py` | Two-stage training and evaluation script with threshold sweeping |

### Multi-Task TCN Model Details
- **dilated convolutions**: kernel_size=3, dilation_rates=(1, 2, 4, 8) -> Receptive Field = 61h (covers full 24h lookback window).
- **temporal context pooling**: concatenates the final timestep representation, global average pooling, and global max pooling.
- **dual output heads**:
  - **regression head**: outputs continuous concentration levels (24 horizon, 5 parameters), trained via `AsymmetricSpikeLoss` to penalize under-prediction of spike values.
  - **classification head**: outputs per-cell spike probabilities (24 horizon, 5 parameters), trained via Binary Cross-Entropy (BCE) to break the spike recall plateau.

## Project Structure
```
INDUSTRIAL_WASTEWATER_TCN/
-- config.py              # Project thresholds, physical metrics, and scaling helper functions
-- requirements.txt       # Python dependencies
-- train_and_evaluate.py  # End-to-end two-stage MTL training and sweep evaluation pipeline
-- dataset/
   -- wastewater_10000.csv
-- models/
   -- tcn_model.keras    # Two-stage optimized dual-head model
   -- tcn_model_mse.keras# Single-head Baseline MSE reference model
   -- scaler.pkl         # Fitted RobustScaler
-- modules/
    -- data_acquisition.py
    -- preprocessing.py
    -- tcn_prediction.py
```

## Setup & Training
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end two-stage training & evaluation pipeline
python train_and_evaluate.py
```

## Core Preprocessing & Loss Optimization
1. **Zero-Leakage Preprocessing**: The dataset is chronologically split *before* scaling. RobustScaler is fitted strictly on the train segment. Feature extraction for test sequences prepends context without leakage.
2. **Heavy-Tail Adjustments**: Applies log1p transformation to heavily right-skewed parameters (COD, BOD, TDS) followed by RobustScaler to compress scale variance.
3. **Two-Stage Training**:
   - **Stage 1 (Warmup)**: Trains the TCN with MSE (regression) + BCE (classification) loss using Adam(lr=1e-3) for 25 epochs.
   - **Stage 2 (Fine-tuning)**: Recompiles with AsymmetricSpikeLoss (reg) + BCE (cls) using Adam(lr=5e-4) for 45 epochs.
4. **Swept Decision Threshold**: Evaluates classification probabilities using a decision threshold sweep (tau in [0.20, 0.30, 0.40, 0.50]) to balance Precision and Recall for wastewater alarm calibration.
