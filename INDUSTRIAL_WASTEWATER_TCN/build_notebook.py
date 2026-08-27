"""Generate the updated notebook as a .ipynb JSON file."""
import json

cells = []

def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [source]})

def code(source):
    cells.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": [source]})

# ======================================================================
# SECTION 1: Setup
# ======================================================================
md("# Industrial Wastewater TCN — Spike-Aware Forecasting\n\n"
   "Module 3: Improved TCN with project-specific temporal forecasting formulation.")

md("## 1. Setup and Data Loading")

code(
    "import os, sys\n"
    "from pathlib import Path\n"
    "\n"
    "PROJECT_ROOT = Path.cwd().parent\n"
    "if str(PROJECT_ROOT) not in sys.path:\n"
    "    sys.path.insert(0, str(PROJECT_ROOT))\n"
    "print('Project root:', PROJECT_ROOT)"
)

code(
    "from modules.data_acquisition import load_dataset, validate_dataset\n"
    "\n"
    "DATASET_PATH = PROJECT_ROOT / 'dataset' / 'wastewater_10000.csv'\n"
    "df = load_dataset(str(DATASET_PATH))\n"
    "validate_dataset(df)\n"
    "print(f'Columns: {list(df.columns)}')\n"
    "print(f'Shape: {df.shape}')"
)

code(
    "from config import PARAMETERS\n"
    "param_cols = [PARAMETERS[k] for k in PARAMETERS]\n"
    "df[param_cols].describe().T"
)

# ======================================================================
# SECTION 2: Preprocessing
# ======================================================================
md("## 2. Preprocessing (Scaler fitted on train data only)")

code(
    "from modules.preprocessing import (\n"
    "    preprocess_data, create_sequences, split_chronological,\n"
    "    save_scaler, StandardScaler\n"
    ")\n"
    "from config import LOOK_BACK, HORIZON, PARAM_KEYS\n"
    "\n"
    "# Clean data\n"
    "df_clean = preprocess_data(df)\n"
    "param_cols_ordered = [PARAMETERS[k] for k in PARAM_KEYS]\n"
    "raw_values = df_clean[param_cols_ordered].values\n"
    "\n"
    "# Create sequences BEFORE scaling\n"
    "X, y = create_sequences(raw_values)\n"
    "\n"
    "# Split chronologically BEFORE scaling (prevents data leakage)\n"
    "X_train, X_test, y_train, y_test = split_chronological(X, y)\n"
    "\n"
    "# Fit scaler ONLY on training data\n"
    "scaler = StandardScaler()\n"
    "n_train = X_train.shape[0]\n"
    "scaler.fit(X_train.reshape(-1, X_train.shape[-1]))\n"
    "\n"
    "# Scale train and test using the training-fitted scaler\n"
    "X_train = scaler.transform(X_train.reshape(-1, 5)).reshape(X_train.shape)\n"
    "X_test = scaler.transform(X_test.reshape(-1, 5)).reshape(X_test.shape)\n"
    "y_train = scaler.transform(y_train.reshape(-1, 5)).reshape(y_train.shape)\n"
    "y_test = scaler.transform(y_test.reshape(-1, 5)).reshape(y_test.shape)\n"
    "\n"
    "save_scaler(scaler, str(PROJECT_ROOT / 'models' / 'scaler.pkl'))\n"
    "\n"
    "print(f'X_train: {X_train.shape}, X_test: {X_test.shape}')\n"
    "print(f'Scaler fitted on TRAINING data only (no test leakage).')\n"
    "print(f'Scaler mean: {scaler.mean_}')\n"
    "print(f'Scaler std:  {scaler.scale_}')"
)

# ======================================================================
# SECTION 3: Baseline TCN
# ======================================================================
md("## 3. Baseline TCN (Original Architecture)\n\n"
   "Architecture: 3 TCN blocks (dilation [1,2,4]), `x[:, -1, :]` bottleneck, "
   "Dense output head.\n\n"
   "Loss: Standard MSE.")

code(
    "from modules.tcn_prediction import build_tcn_model, train_tcn_model, evaluate_tcn_model\n"
    "import tensorflow as tf\n"
    "\n"
    "# Train/val split\n"
    "val_split = 0.1\n"
    "si = int(len(X_train) * (1 - val_split))\n"
    "X_tr_base, X_val_base = X_train[:si], X_train[si:]\n"
    "y_tr_base, y_val_base = y_train[:si], y_train[si:]\n"
    "print(f'Baseline split: train={len(X_tr_base)}, val={len(X_val_base)}, test={len(X_test)}')\n"
    "\n"
    "# Build baseline\n"
    "model_base = build_tcn_model(input_shape=(LOOK_BACK, 5))"
)

code(
    "history_base = train_tcn_model(\n"
    "    model_base, X_tr_base, y_tr_base, X_val_base, y_val_base,\n"
    "    epochs=40, batch_size=64, patience=10,\n"
    "    model_path=str(PROJECT_ROOT / 'models' / 'tcn_model.keras')\n"
    ")\n"
    "\n"
    "print(f'\\nEpochs trained: {len(history_base.history[\"loss\"])}')\n"
    "print(f'Best val_loss: {min(history_base.history[\"val_loss\"]):.6f}')"
)

code(
    "import matplotlib.pyplot as plt\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].plot(history_base.history['loss'], label='Train')\n"
    "axes[0].plot(history_base.history['val_loss'], label='Val')\n"
    "axes[0].set_title('Baseline — Model Loss')\n"
    "axes[0].set_xlabel('Epoch')\n"
    "axes[0].set_ylabel('MSE Loss')\n"
    "axes[0].legend()\n"
    "axes[1].plot(history_base.history['mae'], label='Train MAE')\n"
    "axes[1].plot(history_base.history['val_mae'], label='Val MAE')\n"
    "axes[1].set_title('Baseline — Model MAE')\n"
    "axes[1].set_xlabel('Epoch')\n"
    "axes[1].set_ylabel('MAE')\n"
    "axes[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

code(
    "results_base = evaluate_tcn_model(model_base, X_test, y_test)"
)

# ======================================================================
# SECTION 4: Baseline Prediction Plots
# ======================================================================
md("## 4. Baseline — Prediction vs Actual")

code(
    "from modules.tcn_prediction import predict_next_24h\n"
    "import pandas as pd\n"
    "\n"
    "# Evaluate across ALL test samples\n"
    "n_test = X_test.shape[0]\n"
    "all_pred_base = np.zeros((n_test, 24, 5))\n"
    "all_true_orig = np.zeros((n_test, 24, 5))\n"
    "\n"
    "for idx in range(n_test):\n"
    "    all_pred_base[idx] = predict_next_24h(model_base, X_test[idx], scaler)\n"
    "    all_true_orig[idx] = scaler.inverse_transform(y_test[idx])\n"
    "\n"
    "print(f'Generated predictions for {n_test} test samples.')"
)

code(
    "import numpy as np\n"
    "\n"
    "param_labels = ['pH', 'COD (mg/L)', 'BOD (mg/L)', 'TDS (mg/L)', 'Temperature (°C)']\n"
    "\n"
    "fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)\n"
    "for i, label in enumerate(param_labels):\n"
    "    # Aggregate across test samples: mean prediction and mean actual\n"
    "    mean_pred = np.mean(all_pred_base[:, :, i], axis=0)\n"
    "    mean_true = np.mean(all_true_orig[:, :, i], axis=0)\n"
    "    axes[i].plot(range(24), mean_pred, 'r-', label='Baseline Predicted', linewidth=2)\n"
    "    axes[i].plot(range(24), mean_true, 'b--', label='Actual', linewidth=2)\n"
    "    axes[i].set_ylabel(label)\n"
    "    axes[i].set_title(f'{label} — Baseline 24h Prediction vs Actual (mean over test)')\n"
    "    axes[i].legend()\n"
    "axes[-1].set_xlabel('Hours Ahead')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ======================================================================
# SECTION 5: Improved TCN v2
# ======================================================================
md("## 5. Improved TCN — Spike-Aware Forecasting\n\n"
   "### Forecasting Formulation\n\n"
   "```\n"
   "y_hat(t+h) = y(t) + alpha_h * T(t) + beta_h * A(t) + gamma_h * S(t)\n"
   "```\n\n"
   "| Component | Meaning | How computed |\n"
   "|-----------|---------|---------------|\n"
   "| `y(t)` | Latest observed value | Implicit in TCN features |\n"
   "| `T(t)` | Temporal trend | `x(t) - x(t-1)` over input window |\n"
   "| `A(t)` | Acceleration | `T(t) - T(t-1)` |\n"
   "| `S(t)` | Spike signal | z-score of input vs input statistics |\n"
   "| `alpha_h, beta_h, gamma_h` | Learned coefficients | via Dense layers |\n\n"
   "### Architecture Changes\n\n"
   "1. **4 TCN blocks** with dilation `[1, 2, 4, 8]` — receptive field = 30 (full 24h)\n"
   "2. **Temporal decoder** with Conv1D — preserves all 24 timesteps (no `x[:, -1, :]`)\n"
   "3. **Explicit T, A, S features** concatenated with input and output\n"
   "4. **Spike-aware Huber loss** with adaptive sample weighting")

code(
    "from modules.tcn_prediction import (\n"
    "    build_tcn_model_v2, train_tcn_model_v2, evaluate_tcn_model,\n"
    "    predict_next_24h_v2, evaluate_spikes, compare_metrics\n"
    ")\n"
    "import numpy as np\n"
    "\n"
    "# Build improved model\n"
    "model_v2 = build_tcn_model_v2(input_shape=(LOOK_BACK, 5))"
)

code(
    "# Train/val split (same as baseline for fair comparison)\n"
    "si = int(len(X_train) * (1 - val_split))\n"
    "X_tr_v2, X_val_v2 = X_train[:si], X_train[si:]\n"
    "y_tr_v2, y_val_v2 = y_train[:si], y_train[si:]\n"
    "\n"
    "scaler_mean = scaler.mean_.astype(np.float32)\n"
    "scaler_std = scaler.scale_.astype(np.float32)\n"
    "\n"
    "model_v2, history_v2, sm, ss = train_tcn_model_v2(\n"
    "    model_v2, X_tr_v2, y_tr_v2, X_val_v2, y_val_v2,\n"
    "    scaler_mean=scaler_mean, scaler_std=scaler_std,\n"
    "    epochs=60, batch_size=32, patience=15,\n"
    "    model_path=str(PROJECT_ROOT / 'models' / 'tcn_model_v2.keras'),\n"
    "    stats_path=str(PROJECT_ROOT / 'models' / 'scaler_stats.npz'),\n"
    ")\n"
    "\n"
    "print(f'\\nEpochs trained: {len(history_v2.history[\"loss\"])}')\n"
    "print(f'Best val_loss: {min(history_v2.history[\"val_loss\"]):.6f}')"
)

code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].plot(history_v2.history['loss'], label='Train')\n"
    "axes[0].plot(history_v2.history['val_loss'], label='Val')\n"
    "axes[0].set_title('Improved TCN v2 — Spike-Aware Loss')\n"
    "axes[0].set_xlabel('Epoch')\n"
    "axes[0].set_ylabel('Spike-Aware Loss')\n"
    "axes[0].legend()\n"
    "axes[1].plot(history_v2.history['mae'], label='Train MAE')\n"
    "axes[1].plot(history_v2.history['val_mae'], label='Val MAE')\n"
    "axes[1].set_title('Improved TCN v2 — Model MAE')\n"
    "axes[1].set_xlabel('Epoch')\n"
    "axes[1].set_ylabel('MAE')\n"
    "axes[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

code(
    "results_v2 = evaluate_tcn_model(model_v2, X_test, y_test)"
)

# ======================================================================
# SECTION 6: Improved Prediction Plots
# ======================================================================
md("## 6. Improved Model — Prediction vs Actual")

code(
    "all_pred_v2 = np.zeros((n_test, 24, 5))\n"
    "for idx in range(n_test):\n"
    "    all_pred_v2[idx] = predict_next_24h_v2(model_v2, X_test[idx], scaler)\n"
    "\n"
    "fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)\n"
    "for i, label in enumerate(param_labels):\n"
    "    mean_pred = np.mean(all_pred_v2[:, :, i], axis=0)\n"
    "    mean_true = np.mean(all_true_orig[:, :, i], axis=0)\n"
    "    axes[i].plot(range(24), mean_pred, 'r-', label='Improved Predicted', linewidth=2)\n"
    "    axes[i].plot(range(24), mean_true, 'b--', label='Actual', linewidth=2)\n"
    "    axes[i].set_ylabel(label)\n"
    "    axes[i].set_title(f'{label} — Improved TCN v2 24h Prediction vs Actual')\n"
    "    axes[i].legend()\n"
    "axes[-1].set_xlabel('Hours Ahead')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ======================================================================
# SECTION 7: Side-by-Side Comparison
# ======================================================================
md("## 7. Side-by-Side Comparison: Baseline vs Improved")

code(
    "compare_metrics(results_base, results_v2)"
)

code(
    "# Prediction comparison per parameter\n"
    "fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)\n"
    "for i, label in enumerate(param_labels):\n"
    "    mean_pred_b = np.mean(all_pred_base[:, :, i], axis=0)\n"
    "    mean_pred_v = np.mean(all_pred_v2[:, :, i], axis=0)\n"
    "    mean_true = np.mean(all_true_orig[:, :, i], axis=0)\n"
    "    axes[i].plot(range(24), mean_true, 'k--', label='Actual', linewidth=2.5, alpha=0.8)\n"
    "    axes[i].plot(range(24), mean_pred_b, 'r-', label='Baseline', linewidth=2, alpha=0.7)\n"
    "    axes[i].plot(range(24), mean_pred_v, 'g-', label='Improved v2', linewidth=2, alpha=0.7)\n"
    "    axes[i].set_ylabel(label)\n"
    "    axes[i].set_title(f'{label} — Baseline vs Improved vs Actual')\n"
    "    axes[i].legend(loc='best')\n"
    "axes[-1].set_xlabel('Hours Ahead')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ======================================================================
# SECTION 8: Sample-level comparison
# ======================================================================
md("## 8. Sample-Level Comparison (Random Test Samples)")

code(
    "# Show a few specific test samples where spikes occur\n"
    "# Find samples with high variance in actual values (likely spike periods)\n"
    "sample_var = np.var(all_true_orig, axis=(1, 2))\n"
    "spike_indices = np.argsort(sample_var)[-5:][::-1]\n"
    "\n"
    "fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)\n"
    "for i, label in enumerate(param_labels):\n"
    "    # Aggregate across the high-variance samples\n"
    "    for si_idx, si in enumerate(spike_indices):\n"
    "        style = '-' if si_idx == 0 else ':'\n"
    "        lw = 2 if si_idx == 0 else 1\n"
    "        if i == 0:\n"
    "            axes[i].plot(range(24), all_true_orig[si, :, i], 'b' + style,\n"
    "                         linewidth=lw, alpha=0.5, label='Actual' if si_idx == 0 else '')\n"
    "            axes[i].plot(range(24), all_pred_base[si, :, i], 'r' + style,\n"
    "                         linewidth=lw, alpha=0.5, label='Baseline' if si_idx == 0 else '')\n"
    "            axes[i].plot(range(24), all_pred_v2[si, :, i], 'g' + style,\n"
    "                         linewidth=lw, alpha=0.5, label='Improved' if si_idx == 0 else '')\n"
    "        else:\n"
    "            axes[i].plot(range(24), all_true_orig[si, :, i], 'b' + style, linewidth=lw, alpha=0.5)\n"
    "            axes[i].plot(range(24), all_pred_base[si, :, i], 'r' + style, linewidth=lw, alpha=0.5)\n"
    "            axes[i].plot(range(24), all_pred_v2[si, :, i], 'g' + style, linewidth=lw, alpha=0.5)\n"
    "    axes[i].set_ylabel(label)\n"
    "    axes[i].set_title(f'{label} — High-variance samples (Baseline=red, Improved=green, Actual=blue)')\n"
    "    axes[i].legend()\n"
    "axes[-1].set_xlabel('Hours Ahead')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ======================================================================
# SECTION 9: Spike Evaluation
# ======================================================================
md("## 9. Spike Detection Evaluation\n\n"
   "Using training-set statistics to define spike thresholds:\n"
   "- A value is a **spike** if it deviates from the training mean by more than "
   "2.5 standard deviations.\n"
   "- **Window-level**: a 24h prediction window is flagged as a spike if ANY "
   "parameter at ANY timestep exceeds the threshold.\n"
   "- This evaluates whether the model learns to predict anomalous values "
   "(not just averages).")

code(
    "spike_results_base = evaluate_spikes(\n"
    "    all_true_orig, all_pred_base, scaler_mean, scaler_std, spike_threshold=2.5\n"
    ")"
)

code(
    "spike_results_v2 = evaluate_spikes(\n"
    "    all_true_orig, all_pred_v2, scaler_mean, scaler_std, spike_threshold=2.5\n"
    ")"
)

code(
    "# Spike detection comparison summary\n"
    "print('\\n' + '=' * 60)\n"
    "print('SPIKE DETECTION COMPARISON')\n"
    "print('=' * 60)\n"
    "print(f'{\"Metric\":<25} {\"Baseline\":>12} {\"Improved\":>12}')\n"
    "print('-' * 60)\n"
    "for key in ['Actual spike windows', 'Predicted spike windows', 'True positives',\n"
    "            'False positives', 'False negatives']:\n"
    "    b = spike_results_base['overall'][key]\n"
    "    v = spike_results_v2['overall'][key]\n"
    "    print(f'  {key:<23} {b:>12} {v:>12}')\n"
    "for key in ['Spike precision', 'Spike recall', 'Spike F1']:\n"
    "    b = spike_results_base['overall'][key]\n"
    "    v = spike_results_v2['overall'][key]\n"
    "    print(f'  {key:<23} {b:>12.4f} {v:>12.4f}')\n"
    "print('=' * 60)"
)

# ======================================================================
# SECTION 10: Save
# ======================================================================
md("## 10. Model Summary")

code(
    "from modules.tcn_prediction import save_tcn_model\n"
    "\n"
    "# Baseline is already saved. Save improved model.\n"
    "save_tcn_model(model_v2, str(PROJECT_ROOT / 'models' / 'tcn_model_v2.keras'))\n"
    "print('\\nModels saved:')\n"
    "print('  Baseline: models/tcn_model.keras')\n"
    "print('  Improved: models/tcn_model_v2.keras')"
)

md("## 11. Final Report\n\n"
   "### Architecture Comparison\n\n"
   "| Feature | Baseline | Improved v2 |\n"
   "|---------|----------|-------------|\n"
   "| TCN blocks | 3 (dilation [1,2,4]) | 4 (dilation [1,2,4,8]) |\n"
   "| Receptive field | 14 timesteps | 30 timesteps (full 24h) |\n"
   "| Temporal aggregation | `x[:, -1, :]` (last timestep only) | Full Conv1D temporal decoder |\n"
   "| Output head | Dense from 64-dim vector | Dense from 842-dim (full temporal) |\n"
   "| Loss function | Standard MSE | Spike-Aware Huber |\n"
   "| Forecasting | End-to-end | Trend + Acceleration + Spike components |\n"
   "| Parameters | ~67K | ~140K |\n\n"
   "### Forecasting Formulation\n\n"
   "```\n"
   "y_hat(t+h) = y(t) + alpha_h * T(t) + beta_h * A(t) + gamma_h * S(t)\n"
   "```\n\n"
   "- **Trend T(t)**: computed as `x(t) - x(t-1)` over the 24h input window\n"
   "- **Acceleration A(t)**: computed as `T(t) - T(t-1)`\n"
   "- **Spike signal S(t)**: z-score of each timestep vs input statistics\n"
   "- **alpha, beta, gamma**: learned by the network via Dense layers\n\n"
   "### Key Design Decisions\n\n"
   "1. No future values used — T, A, S derived from input window only\n"
   "2. Spike-aware loss weights samples with extreme values higher (3x)\n"
   "3. Full temporal information preserved through Conv1D decoder\n"
   "4. Scaler fitted on training data only (no test leakage)")

# ======================================================================
# Build notebook
# ======================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.2"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

OUTPUT = r"C:\clgproject\INDUSTRIAL_WASTEWATER_TCN\notebook\wastewater_tcn.ipynb"
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
print(f"Notebook written to {OUTPUT}")
print(f"Total cells: {len(cells)}")
