"""
Module 5 — HSRAE (Hybrid Spike Risk Assessment Engine)
======================================================

Responsibilities:
  • Synthesize 24-hour TCN multi-parameter forecasts with regulatory thresholds.
  • Integrate Autoencoder unsupervised anomaly scores.
  • Analyze temporal rate-of-change (velocity/acceleration) across parameters.
  • Compute a composite 0-100 Spike Risk Index and classify into Normal, Warning, or Critical.
  • Calculate earliest Time-to-Spike (hours until threshold breach).
  • Generate automated, parameter-specific plant operator mitigation protocols.
"""

import numpy as np
from config import (
    THRESHOLDS, PARAM_KEYS, PARAMETERS,
    ANOMALY_SCORE_WARNING, ANOMALY_SCORE_CRITICAL,
    RISK_LEVELS, SPIKE_HORIZON_HOURS
)


def evaluate_parameter_threshold(param_key: str, value: float) -> tuple[str, float]:
    """
    Evaluate a single reading against the defined Normal/Warning/Critical bands.

    Returns:
      status : "Normal" | "Warning" | "Critical"
      severity : float in [0.0, 1.0] (0 = well within normal, 1.0 = deep critical)
    """
    th = THRESHOLDS.get(param_key)
    if not th:
        return "Normal", 0.0

    n_min, n_max = th["normal_min"], th["normal_max"]
    w_min, w_max = th["warning_min"], th["warning_max"]
    c_min, c_max = th["critical_min"], th["critical_max"]

    # Check Bilateral (pH, Temperature)
    if param_key in ["pH", "Temperature"]:
        if value < w_min or value > w_max:
            # Critical
            if value < w_min:
                dist = (w_min - value) / max(1e-4, w_min - c_min)
            else:
                dist = (value - w_max) / max(1e-4, c_max - w_max)
            severity = min(1.0, 0.70 + 0.30 * min(1.0, max(0.0, dist)))
            return "Critical", float(severity)
        elif value < n_min or value > n_max:
            # Warning
            if value < n_min:
                dist = (n_min - value) / max(1e-4, n_min - w_min)
            else:
                dist = (value - n_max) / max(1e-4, w_max - n_max)
            severity = 0.35 + 0.34 * min(1.0, max(0.0, dist))
            return "Warning", float(severity)
        else:
            # Normal
            center = (n_min + n_max) / 2.0
            radius = max(1e-4, (n_max - n_min) / 2.0)
            dev = abs(value - center) / radius
            severity = 0.30 * min(1.0, dev)
            return "Normal", float(severity)

    # One-sided metrics (COD, BOD, TDS): normal up to n_max, warning n_max to w_max, critical above w_max
    else:
        if value > w_max:
            # Critical
            dist = (value - w_max) / max(1e-4, w_max * 0.5)
            severity = min(1.0, 0.70 + 0.30 * min(1.0, dist))
            return "Critical", float(severity)
        elif value > n_max:
            # Warning
            dist = (value - n_max) / max(1e-4, w_max - n_max)
            severity = 0.35 + 0.34 * min(1.0, dist)
            return "Warning", float(severity)
        else:
            # Normal
            ratio = max(0.0, value / max(1e-4, n_max))
            severity = 0.30 * min(1.0, ratio)
            return "Normal", float(severity)


def generate_mitigation_protocols(breaches: list, ae_score: float) -> list[str]:
    """
    Generate actionable plant mitigation protocols based on detected breaches and anomaly score.
    """
    protocols = []

    if not breaches and ae_score < ANOMALY_SCORE_WARNING:
        return [
            "Normal Operating State: Maintain standard aerobic basin dissolved oxygen (2.0-2.5 mg/L).",
            "Keep baseline hydraulic retention time (HRT) and standard clarifier return activated sludge (RAS) rates.",
            "Routine telemetry logging active; next scheduled sensor diagnostic in 4 hours."
        ]

    for b in breaches:
        param = b["param"]
        val = b["peak_value"]
        hr = b["hour_of_breach"]
        level = b["level"]
        unit = THRESHOLDS[param]["units"]
        unit_str = f" {unit}" if unit else ""

        if param == "COD":
            if level == "Critical":
                protocols.append(
                    f"CRITICAL COD SPIKE ({val:.1f}{unit_str} in {hr}h): Divert peak influent to Equalization Basin #1. "
                    "Ramp up secondary aeration blowers to prevent dissolved oxygen depletion."
                )
            else:
                protocols.append(
                    f"COD WARNING ({val:.1f}{unit_str} in {hr}h): Increase bio-selector recirculation rate by 15%. "
                    "Alert secondary clarifier operators for potential sludge bulking."
                )

        elif param == "BOD":
            protocols.append(
                f"BOD Surge ({val:.1f}{unit_str} in {hr}h): Biological load increasing. "
                "Adjust return activated sludge (RAS) ratio to sustain food-to-microorganism (F/M) equilibrium."
            )

        elif param == "pH":
            if val < 6.0:
                protocols.append(
                    f"ACIDIC INFLUENT ALERT (pH {val:.2f} in {hr}h): Activate chemical feed pumps — inject 10% sodium hydroxide (NaOH) / hydrated lime at flash mixer."
                )
            else:
                protocols.append(
                    f"ALKALINE INFLUENT ALERT (pH {val:.2f} in {hr}h): Initiate controlled sulfuric acid (H2SO4) neutralizer dosing to protect downstream nitrifying bacteria."
                )

        elif param == "TDS":
            protocols.append(
                f"TDS Elevated ({val:.1f}{unit_str} in {hr}h): High dissolved salts. Inspect upstream desalination/demineralizer regenerant discharge; verify membrane flux to prevent scaling."
            )

        elif param == "Temperature":
            protocols.append(
                f"Thermal Alert ({val:.1f}{unit_str} in {hr}h): Water temperature excursion. Activate cooling tower spray headers or verify heat exchanger efficiency."
            )

    if ae_score >= ANOMALY_SCORE_CRITICAL:
        protocols.append(
            f"HIGH OPERATIONAL ANOMALY (Autoencoder Score: {ae_score:.3f}): Unrecognized sensor pattern detected. "
            "Dispatch technician to verify optical sensor fouling and perform laboratory verification test."
        )
    elif ae_score >= ANOMALY_SCORE_WARNING:
        protocols.append(
            f"Mild Operational Anomaly (Autoencoder Score: {ae_score:.3f}): Slight multivariate divergence from standard profile."
        )

    return protocols


def assess_spike_risk(
    forecast_24h: np.ndarray,
    autoencoder_score: float = 0.0,
    current_values: np.ndarray = None,
    param_keys: list = PARAM_KEYS,
) -> dict:
    """
    Comprehensive Hybrid Spike Risk Assessment.

    Parameters:
      forecast_24h       : (24, 5) predicted parameter trajectories in original scale
      autoencoder_score  : float in [0, 1] normalized score from Autoencoder
      current_values     : (5,) latest observed parameter values (optional)
      param_keys         : list of parameter names

    Returns structured dictionary with:
      - overall_risk_level   : "Normal" | "Warning" | "Critical"
      - risk_score           : float (0.0 to 100.0)
      - earliest_spike_hour  : int or None (hours from now)
      - primary_driver       : str (e.g. "COD", "pH", etc.)
      - parameter_evaluations: list of dicts with peak forecasts and status
      - breached_parameters  : list of breached parameters
      - autoencoder_impact   : dict with AE score and risk contribution
      - protocols            : list of actionable mitigation steps
    """
    forecast_24h = np.array(forecast_24h)
    n_hours, n_params = forecast_24h.shape

    param_evals = []
    breaches = []
    max_forecast_severity = 0.0
    primary_driver = "None"
    earliest_spike_hour = None

    for p, key in enumerate(param_keys):
        series = forecast_24h[:, p]
        peak_idx = int(np.argmax(series)) if key != "pH" else int(np.argmax(np.abs(series - 7.0)))
        peak_val = float(series[peak_idx])
        curr_val = float(current_values[p]) if current_values is not None else float(series[0])

        # Evaluate peak forecasted point
        status, severity = evaluate_parameter_threshold(key, peak_val)

        # Track earliest breach across the 24 hours
        breach_hour = None
        breach_level = "Normal"
        for h in range(n_hours):
            h_status, _ = evaluate_parameter_threshold(key, series[h])
            if h_status in ["Warning", "Critical"]:
                breach_hour = h + 1
                breach_level = h_status
                break

        if severity > max_forecast_severity:
            max_forecast_severity = severity
            primary_driver = key

        eval_data = {
            "param": key,
            "units": THRESHOLDS[key]["units"],
            "current_value": round(curr_val, 2),
            "peak_forecast_value": round(peak_val, 2),
            "hour_of_peak": peak_idx + 1,
            "status": status,
            "severity_score": round(severity, 3),
            "threshold_normal": f"{THRESHOLDS[key]['normal_min']} - {THRESHOLDS[key]['normal_max']}",
            "threshold_warning": f"{THRESHOLDS[key]['warning_min']} - {THRESHOLDS[key]['warning_max']}",
        }
        param_evals.append(eval_data)

        if breach_hour is not None:
            breaches.append({
                "param": key,
                "peak_value": peak_val,
                "hour_of_breach": breach_hour,
                "level": breach_level,
            })
            if earliest_spike_hour is None or breach_hour < earliest_spike_hour:
                earliest_spike_hour = breach_hour

    # Autoencoder contribution (normalized 0.0 - 1.0, where 0.5 = 95th percentile threshold)
    ae_norm = min(1.0, max(0.0, float(autoencoder_score)))
    ae_warning_th = 0.50
    ae_critical_th = 0.75

    # Composite Risk Index Formula:
    # 55% from peak forecasted threshold breach severity
    # 30% from Autoencoder anomaly pattern deviation
    # 15% velocity/trend surge factor
    velocity_factor = 0.0
    if current_values is not None:
        for p, key in enumerate(param_keys):
            if key in ["COD", "BOD", "TDS"]:
                delta = float(forecast_24h[5, p] - current_values[p])  # 6h trajectory slope
                norm_band = THRESHOLDS[key]["normal_max"] - THRESHOLDS[key]["normal_min"]
                if delta > 0 and norm_band > 0:
                    velocity_factor = max(velocity_factor, min(1.0, delta / norm_band))

    raw_risk = (
        0.55 * max_forecast_severity +
        0.30 * ae_norm +
        0.15 * velocity_factor
    )
    risk_score = float(round(raw_risk * 100.0, 1))

    # Overall Classification Rules
    any_critical_breach = any(b["level"] == "Critical" for b in breaches)
    any_warning_breach = any(b["level"] == "Warning" for b in breaches)

    if any_critical_breach or risk_score >= 70.0 or ae_norm >= ae_critical_th:
        overall_level = "Critical"
    elif any_warning_breach or risk_score >= 35.0 or ae_norm >= ae_warning_th:
        overall_level = "Warning"
    else:
        overall_level = "Normal"

    protocols = generate_mitigation_protocols(breaches, ae_norm)

    return {
        "overall_risk_level": overall_level,
        "risk_score": risk_score,
        "earliest_spike_hour": earliest_spike_hour,
        "primary_driver": primary_driver if primary_driver != "None" else "All Stable",
        "parameter_evaluations": param_evals,
        "breached_parameters": breaches,
        "autoencoder_impact": {
            "score": round(ae_norm, 4),
            "status": "Critical" if ae_norm >= ae_critical_th else ("Warning" if ae_norm >= ae_warning_th else "Normal"),
        },
        "protocols": protocols,
    }
