"""
predictor.py — FloodSight Inference Engine
==========================================
Loads the trained LSTM + scaler and runs predictions.
Handles both single-step (current conditions) and
multi-step (24-hour ahead) forecasts.
"""

import os
import numpy as np
import torch
import joblib

from model import FloodLSTM, FEATURE_COLS, SEQ_LEN, level_to_risk, build_model

MODEL_PATH  = os.getenv("MODEL_PATH",  "models/flood_lstm.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")

_model:  FloodLSTM | None = None
_scaler = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load model and scaler into module-level cache (called once at startup)."""
    global _model, _scaler

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python train.py"
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. Run: python train.py"
        )

    _scaler = joblib.load(SCALER_PATH)
    _model  = build_model(_device)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    _model.eval()
    print(f"[predictor] Model loaded from {MODEL_PATH}  (device={_device})")


def _build_sequence(feature_dict: dict) -> np.ndarray:
    """
    Build a (1, SEQ_LEN, n_features) tensor from a single feature snapshot
    by repeating it SEQ_LEN times. In production you'd pass real historical
    observations; this is a sensible fallback for live single-step calls.
    """
    row = np.array([feature_dict[col] for col in FEATURE_COLS], dtype=np.float32)
    seq = np.tile(row, (SEQ_LEN, 1))               # (24, 10)
    seq_scaled = _scaler.transform(seq)             # (24, 10)
    return seq_scaled[np.newaxis, ...]              # (1, 24, 10)


def predict_level(feature_dict: dict) -> dict:
    """
    Run single-step flood level prediction.

    Parameters
    ----------
    feature_dict : dict with keys matching FEATURE_COLS

    Returns
    -------
    dict:
        predicted_level_m, risk_category, confidence_pct, feature_importance
    """
    if _model is None:
        load_model()

    seq = _build_sequence(feature_dict)
    tensor = torch.tensor(seq, dtype=torch.float32).to(_device)

    with torch.no_grad():
        pred = _model(tensor).item()

    pred = max(0.0, pred)
    risk = level_to_risk(pred)

    # ── Simple feature importance via finite differences ──────────────────
    importances = {}
    for i, col in enumerate(FEATURE_COLS):
        perturbed = seq.copy()
        perturbed[0, :, i] += 0.5          # nudge feature
        t_pert = torch.tensor(perturbed, dtype=torch.float32).to(_device)
        with torch.no_grad():
            pert_pred = _model(t_pert).item()
        importances[col] = round(abs(pert_pred - pred), 4)

    # Normalise importances to percentages
    total = sum(importances.values()) or 1.0
    importances = {k: round(v / total * 100, 1) for k, v in importances.items()}

    # Rough confidence: inverse of prediction spread (placeholder)
    confidence = round(min(98.0, max(60.0, 92.0 - abs(pred - 2.0) * 5)), 1)

    return {
        "predicted_level_m": round(pred, 3),
        "risk_category":     risk,
        "confidence_pct":    confidence,
        "feature_importance": importances,
    }


def predict_forecast(feature_dict: dict, steps: int = 24) -> list[dict]:
    """
    Autoregressive 24-step (hour-ahead) forecast.
    Each step uses the previous predicted level as upstream_level_m.
    """
    if _model is None:
        load_model()

    current = feature_dict.copy()
    results = []

    for step in range(1, steps + 1):
        # Simple rainfall decay (in reality you'd use forecast data)
        current["rainfall_mm"]         = max(0, current["rainfall_mm"] * 0.92)
        current["cumulative_rain_24h"] = max(0, current["cumulative_rain_24h"] * 0.98)
        current["cumulative_rain_72h"] = max(0, current["cumulative_rain_72h"] * 0.99)

        result = predict_level(current)
        pred_level = result["predicted_level_m"]

        # Feed prediction back as upstream proxy
        current["upstream_level_m"]     = pred_level * 0.85
        current["river_discharge_m3s"]  = 50 + pred_level * 30
        current["soil_moisture_pct"]    = min(100, current["soil_moisture_pct"] + current["rainfall_mm"] * 0.2)

        results.append({
            "hour":              step,
            "predicted_level_m": pred_level,
            "risk_category":     result["risk_category"],
            "confidence_pct":    result["confidence_pct"],
        })

    return results
