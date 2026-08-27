"""
End-to-End MTL Two-Stage Training & Evaluation Pipeline
========================================================

Integrates all 4 phases with Multi-Task Learning (MTL) dual-head architecture:

  * Phase 1: RobustScaler + log1p (COD, BOD, TDS), zero-leakage chronological split.
  * Phase 2: RF=61h dilated TCN encoder shared across both heads.
  * Phase 3: Two-stage MTL training:
      - Stage 1 (25 ep): MSE + BCE warm-up, Adam(lr=1e-3)
      - Stage 2 (45 ep): AsymmetricSpikeLoss + BCE fine-tuning, Adam(lr=5e-4)
  * Phase 4: Physical unit evaluation with classification threshold sweep
             tau in [0.20, 0.30, 0.40, 0.50] to show Precision-Recall trade-off.

MTL Dual-Head Architecture:
  Shared TCN encoder (RF=61h)
    +-- context = Concat[last_step, GlobalAvgPool, GlobalMaxPool]
          +-- Dense(128->64)  [shared stem]
                +-- reg_output  : Dense(120) -> Reshape(24,5)   (regression)
                +-- cls_output  : Dense(120, sigmoid) -> Reshape(24,5)  (classification)
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import optimizers, callbacks

from config import LOOK_BACK, HORIZON, PARAM_KEYS, THRESHOLDS
from modules.data_acquisition import acquire_and_validate
from modules.preprocessing import (
    preprocess_pipeline,
    inverse_transform_predictions,
    get_log_param_indices,
    create_spike_cls_targets,
)
from modules.tcn_prediction import (
    build_tcn_model,
    build_mtl_tcn_model,
    evaluate_tcn_model,
    evaluate_spikes,
    evaluate_mtl_spikes,
    AsymmetricSpikeLoss,
)

CLS_SPIKE_THRESHOLD = 2.0   # z-score threshold in RobustScaler+log1p space


# ======================================================================
# Helpers
# ======================================================================

def make_callbacks(model_path, es_patience, rlr_patience=4):
    """Return standard [EarlyStopping, ReduceLROnPlateau, ModelCheckpoint] list."""
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=rlr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


def print_summary(tag, history_list):
    total_epochs = sum(len(h.history["loss"]) for h in history_list)
    best_val     = min(min(h.history["val_loss"]) for h in history_list)
    print(f"  [{tag}]  Epochs={total_epochs}  Best val_loss={best_val:.6f}")


# ======================================================================
# MAIN PIPELINE
# ======================================================================

def run_pipeline():
    print("=" * 75)
    print("INDUSTRIAL WASTEWATER TCN - MTL DUAL-HEAD TWO-STAGE PIPELINE")
    print("=" * 75)

    # ------------------------------------------------------------------
    # STEP 1: PREPROCESSING
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: PREPROCESSING - RobustScaler + log1p (Phase 1)")
    print("=" * 70)

    dataset_path = os.path.join(PROJECT_ROOT, "dataset", "wastewater_10000.csv")
    scaler_path  = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

    df = acquire_and_validate(dataset_path, remove_dups=True)

    X_train_sc, X_test_sc, y_train_sc, y_test_sc, scaler = preprocess_pipeline(
        df,
        scaler_path=scaler_path,
        scaler_type="robust",
        use_log1p=True,
        log_params=["COD", "BOD", "TDS"],
    )

    # Chronological val split (last 10% of train)
    val_idx  = int(len(X_train_sc) * 0.9)
    X_tr, X_val = X_train_sc[:val_idx], X_train_sc[val_idx:]
    y_tr, y_val = y_train_sc[:val_idx], y_train_sc[val_idx:]

    print(f"\nPartitions: Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_test_sc)}")
    print(f"Input -> Target: {X_tr.shape} -> {y_tr.shape}")

    # ------------------------------------------------------------------
    # STEP 2: BUILD BINARY CLASSIFICATION TARGETS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 2: BINARY SPIKE CLASSIFICATION TARGETS (threshold={CLS_SPIKE_THRESHOLD})")
    print("=" * 70)

    print("Training set:")
    y_tr_cls  = create_spike_cls_targets(y_tr,  spike_z_threshold=CLS_SPIKE_THRESHOLD)
    print("Validation set:")
    y_val_cls = create_spike_cls_targets(y_val, spike_z_threshold=CLS_SPIKE_THRESHOLD)
    print("Test set:")
    y_test_cls = create_spike_cls_targets(y_test_sc, spike_z_threshold=CLS_SPIKE_THRESHOLD)

    # ------------------------------------------------------------------
    # STEP 3: BASELINE TCN (MSE only, single-head, reference)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: BASELINE TCN (MSE Loss - Single Head - Reference)")
    print("=" * 70)

    model_base_path = os.path.join(PROJECT_ROOT, "models", "tcn_model_mse.keras")
    model_base = build_tcn_model(
        input_shape=(LOOK_BACK, 5), n_features=5, n_filters=64,
        kernel_size=3, dilation_rates=(1, 2, 4, 8), horizon=HORIZON, loss_type="mse",
    )
    hist_base = model_base.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=80, batch_size=32,
        callbacks=make_callbacks(model_base_path, es_patience=15, rlr_patience=5),
        verbose=1,
    )
    print_summary("Baseline MSE", [hist_base])
    print(f"  Saved -> {model_base_path}")

    # ------------------------------------------------------------------
    # STEP 4A: MTL TCN - STAGE 1 (MSE + BCE Warm-up)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4A: MTL TCN - STAGE 1 (MSE + BCE Warm-up, max 25 epochs)")
    print("=" * 70)

    model_mtl_path  = os.path.join(PROJECT_ROOT, "models", "tcn_model.keras")
    stage1_ckpt     = os.path.join(PROJECT_ROOT, "models", "_mtl_stage1.keras")

    model_mtl = build_mtl_tcn_model(
        input_shape=(LOOK_BACK, 5), n_features=5, n_filters=64,
        kernel_size=3, dilation_rates=(1, 2, 4, 8), horizon=HORIZON,
    )

    # Stage 1 compile: MSE regression + BCE classification, warm-up LR
    model_mtl.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={"regression": "mse", "classification": "binary_crossentropy"},
        loss_weights={"regression": 1.0, "classification": 0.3},
        metrics={"regression": "mae"},
    )
    print("Stage 1: MSE + BCE, Adam(lr=1e-3), loss_weights={reg:1.0, cls:0.3}")

    hist_s1 = model_mtl.fit(
        X_tr,
        {"regression": y_tr, "classification": y_tr_cls},
        validation_data=(X_val, {"regression": y_val, "classification": y_val_cls}),
        epochs=25,
        batch_size=32,
        callbacks=make_callbacks(stage1_ckpt, es_patience=8, rlr_patience=4),
        verbose=1,
    )
    print_summary("MTL Stage 1 (MSE+BCE)", [hist_s1])

    # ------------------------------------------------------------------
    # STEP 4B: MTL TCN - STAGE 2 (AsymmetricSpikeLoss + BCE Fine-Tune)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4B: MTL TCN - STAGE 2 (AsymmetricSpikeLoss + BCE, max 45 epochs)")
    print(f"  spike_z_threshold={CLS_SPIKE_THRESHOLD}, underpredict_penalty=6.0, Adam(lr=5e-4)")
    print("=" * 70)

    spike_loss = AsymmetricSpikeLoss(
        spike_z_threshold=CLS_SPIKE_THRESHOLD,
        underpredict_penalty=6.0,
        overpredict_penalty=2.0,
    )

    # Stage 2: recompile with asymmetric regression loss, reduced LR
    model_mtl.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss={"regression": spike_loss, "classification": "binary_crossentropy"},
        loss_weights={"regression": 1.0, "classification": 0.3},
        metrics={"regression": "mae"},
    )
    print("Stage 2: AsymmetricSpikeLoss + BCE, Adam(lr=5e-4), loss_weights={reg:1.0, cls:0.3}")

    hist_s2 = model_mtl.fit(
        X_tr,
        {"regression": y_tr, "classification": y_tr_cls},
        validation_data=(X_val, {"regression": y_val, "classification": y_val_cls}),
        epochs=45,
        batch_size=32,
        callbacks=make_callbacks(model_mtl_path, es_patience=12, rlr_patience=4),
        verbose=1,
    )
    print_summary("MTL Stage 2 (SpikeLoss+BCE)", [hist_s2])
    print_summary("MTL Combined total", [hist_s1, hist_s2])
    print(f"  Saved -> {model_mtl_path}")

    # ------------------------------------------------------------------
    # STEP 5: PHYSICAL-UNIT EVALUATION (Phase 4)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: PHYSICAL UNITS EVALUATION (Phase 4)")
    print("=" * 70)

    print(f"Generating predictions for {len(X_test_sc)} test samples...")

    # Baseline (single-head regression)
    y_pred_base_sc = model_base.predict(X_test_sc, verbose=0)

    # MTL model returns [reg_output, cls_output]
    mtl_preds = model_mtl.predict(X_test_sc, verbose=0)
    y_pred_mtl_reg_sc = mtl_preds[0]   # regression head
    cls_probs_sc      = mtl_preds[1]   # classification probabilities (no inv-transform needed)

    # Inverse transform regression predictions to physical units
    all_pred_base_orig = inverse_transform_predictions(
        y_pred_base_sc, scaler, use_log1p=True, log_params=["COD", "BOD", "TDS"]
    )
    all_pred_mtl_orig = inverse_transform_predictions(
        y_pred_mtl_reg_sc, scaler, use_log1p=True, log_params=["COD", "BOD", "TDS"]
    )
    all_true_orig = inverse_transform_predictions(
        y_test_sc, scaler, use_log1p=True, log_params=["COD", "BOD", "TDS"]
    )

    print("Inverse transform complete. Physical value ranges:")
    for i, key in enumerate(PARAM_KEYS):
        unit = THRESHOLDS[key]["units"]
        print(f"  {key:<14} min={all_true_orig[:,:,i].min():8.2f}  "
              f"max={all_true_orig[:,:,i].max():8.2f}  {unit}")

    # Scaled-space regression metrics
    print("\n--- Baseline (MSE) scaled-space metrics ---")
    results_base = evaluate_tcn_model(model_base, X_test_sc, y_test_sc)

    # For MTL model pass only the regression output for scaled-space metric
    print("\n--- MTL Two-Stage scaled-space regression metrics ---")
    y_test_flat = y_test_sc.reshape(-1, y_test_sc.shape[-1])
    y_pred_mtl_flat = y_pred_mtl_reg_sc.reshape(-1, y_pred_mtl_reg_sc.shape[-1])
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mtl_mse  = float(mean_squared_error(y_test_flat, y_pred_mtl_flat))
    mtl_mae  = float(mean_absolute_error(y_test_flat, y_pred_mtl_flat))
    mtl_rmse = float(np.sqrt(mtl_mse))
    mtl_r2   = float(r2_score(y_test_flat, y_pred_mtl_flat, multioutput="variance_weighted"))
    print(f"  MSE={mtl_mse:.4f}  MAE={mtl_mae:.4f}  RMSE={mtl_rmse:.4f}  R2={mtl_r2:.4f}")
    results_mtl = {"MAE": mtl_mae, "RMSE": mtl_rmse, "R2": mtl_r2}

    # Physical spike evaluation - baseline (regression-only detection)
    print("\n--- Physical spike evaluation - Baseline (MSE, regression detection) ---")
    spike_base = evaluate_spikes(all_true_orig, all_pred_base_orig, use_physical_thresholds=True)

    # Physical spike evaluation - MTL classification head threshold sweep
    print("\n--- MTL Classification Head - Threshold Sweep ---")
    mtl_sweep = evaluate_mtl_spikes(
        y_true_orig=all_true_orig,
        y_pred_reg_orig=all_pred_mtl_orig,
        cls_probs=cls_probs_sc,
        thresholds=(0.20, 0.30, 0.40, 0.50),
        use_physical_thresholds=True,
    )

    # ------------------------------------------------------------------
    # STEP 6: FINAL COMPARISON TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("FINAL COMPARISON: Baseline (MSE)  vs  MTL Two-Stage (reg+cls heads)")
    print("=" * 75)

    print(f"\n{'Metric':<30} {'Baseline':>14} {'MTL Two-Stage':>16}")
    print("-" * 63)

    # Regression metrics
    for label, bk, mk in [
        ("MAE  (scaled)",  "MAE",  "MAE"),
        ("RMSE (scaled)",  "RMSE", "RMSE"),
        ("R2 Score",       "R2",   "R2"),
    ]:
        bv = results_base[bk]
        mv = results_mtl[mk]
        better = mv < bv if mk != "R2" else mv > bv
        tag = "BETTER" if better else "WORSE "
        print(f"  {label:<28} {bv:>14.4f} {mv:>14.4f}  ({tag} {abs(mv-bv):.4f})")

    print("-" * 63)

    # Physical spike metrics at each tau
    print(f"\n{'MTL Classification Threshold Sweep vs Baseline':}")
    base_p = spike_base["overall"]["Spike Precision"]
    base_r = spike_base["overall"]["Spike Recall"]
    base_f = spike_base["overall"]["Spike F1"]
    base_pk = spike_base["overall"]["Peak MAE"]
    base_rmse = spike_base["overall"]["Overall RMSE"]
    print(f"  {'Source':<26} {'Precision':>10} {'Recall':>8} {'F1':>8} "
          f"{'PkMAE':>10} {'RMSE':>9}")
    print("  " + "-" * 72)
    print(f"  {'Baseline (MSE)':<26} {base_p:>10.4f} {base_r:>8.4f} {base_f:>8.4f} "
          f"{base_pk:>10.2f} {base_rmse:>9.2f}")
    for tau in (0.20, 0.30, 0.40, 0.50):
        ov = mtl_sweep[tau]["overall"]
        flag_p = "(*)" if ov["Spike Recall"] > base_r else "   "
        flag_f = "(+)" if ov["Spike F1"] > base_f else "   "
        print(f"  {'MTL tau='+str(tau):<26} {ov['Spike Precision']:>10.4f} "
              f"{ov['Spike Recall']:>8.4f}{flag_p} {ov['Spike F1']:>8.4f}{flag_f} "
              f"{ov['Peak MAE (reg)']:>10.2f} {ov['Overall RMSE']:>9.2f}")
    print("  (*) = recall improvement over baseline    (+) = F1 improvement")

    print("\n" + "=" * 75)
    print(f"  MTL two-stage model : {model_mtl_path}")
    print(f"  Baseline MSE model  : {model_base_path}")
    print(f"  RobustScaler        : {scaler_path}")
    print("=" * 75)
    print("MTL TWO-STAGE PIPELINE COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
