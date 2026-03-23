"""
train.py — FloodSight LSTM Training Script
==========================================
Generates realistic synthetic flood data, trains the LSTM,
and saves weights + feature scaler for inference.

Usage:
    python train.py                   # train with default settings
    python train.py --epochs 50       # custom epoch count
    python train.py --csv data.csv    # train on your own CSV file

CSV format expected (if --csv is used):
    datetime, rainfall_mm, temperature_c, humidity_pct, wind_speed_ms,
    pressure_hpa, river_discharge_m3s, soil_moisture_pct,
    upstream_level_m, cumulative_rain_24h, cumulative_rain_72h,
    water_level_m
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import joblib

from model import FloodLSTM, FEATURE_COLS, LABEL_COL, SEQ_LEN, build_model


# ── Config ────────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 64
EPOCHS       = 40
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "flood_lstm.pt")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")


# ── Synthetic Data Generator ──────────────────────────────────────────────────

def generate_synthetic_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """
    Generate 2 years of realistic hourly hydrological data.
    Physics-inspired: water level responds to lagged rainfall,
    soil saturation, upstream inflow, and seasonal patterns.
    """
    rng  = np.random.default_rng(seed)
    hours = n_days * 24
    t    = np.arange(hours)

    # ── Seasonal baselines ───────────────────────────────────────────────────
    season        = np.sin(2 * math.pi * t / (365.25 * 24))      # annual cycle
    diurnal       = np.sin(2 * math.pi * t / 24)                 # daily cycle

    # ── Rainfall (mm/hr): sporadic bursts + seasonal weighting ───────────────
    rain_prob     = 0.08 + 0.06 * (season + 1) / 2               # higher in wet season
    rain_mask     = rng.random(hours) < rain_prob
    rain_intensity = rng.exponential(scale=4.0, size=hours)
    # Typhoon events (random intense bursts)
    typhoon_starts = rng.integers(0, hours - 72, size=8)
    for ts in typhoon_starts:
        rain_intensity[ts:ts+72] += rng.exponential(30, size=72)
    rainfall = rain_mask * rain_intensity
    rainfall = np.clip(rainfall, 0, 120)

    # ── Cumulative rainfall ───────────────────────────────────────────────────
    cum_24h  = np.array([rainfall[max(0, i-24):i].sum()  for i in range(hours)])
    cum_72h  = np.array([rainfall[max(0, i-72):i].sum()  for i in range(hours)])

    # ── Temperature (°C) ─────────────────────────────────────────────────────
    temperature = 25 + 8 * season + 3 * diurnal + rng.normal(0, 1.5, hours)

    # ── Humidity (%) ─────────────────────────────────────────────────────────
    humidity = 70 + 15 * season - 5 * diurnal + 0.3 * rainfall + rng.normal(0, 4, hours)
    humidity = np.clip(humidity, 20, 100)

    # ── Wind speed (m/s) ─────────────────────────────────────────────────────
    wind = 3 + 2 * np.abs(season) + rng.exponential(1.5, hours)
    for ts in typhoon_starts:
        wind[ts:ts+72] += rng.uniform(15, 35, size=72)
    wind = np.clip(wind, 0, 60)

    # ── Pressure (hPa) ───────────────────────────────────────────────────────
    pressure = 1013 - 5 * season + rng.normal(0, 3, hours)
    for ts in typhoon_starts:
        pressure[ts:ts+72] -= rng.uniform(10, 30, size=72)

    # ── Soil moisture (%) — accumulates with rain, drains slowly ─────────────
    soil = np.zeros(hours)
    soil[0] = 40.0
    for i in range(1, hours):
        drain = 0.02 * soil[i-1]
        soil[i] = np.clip(soil[i-1] + rainfall[i] * 0.4 - drain, 0, 100)

    # ── Upstream level (m) — correlated with upstream rainfall ───────────────
    upstream = np.zeros(hours)
    upstream[0] = 1.0
    for i in range(1, hours):
        inflow    = cum_24h[i] * 0.015
        recession = 0.05 * upstream[i-1]
        upstream[i] = np.clip(upstream[i-1] + inflow - recession + rng.normal(0, 0.05), 0, 8)

    # ── River discharge (m³/s) ────────────────────────────────────────────────
    discharge = 50 + 30 * upstream + 5 * cum_24h * 0.1 + rng.normal(0, 10, hours)
    discharge = np.clip(discharge, 0, 5000)

    # ── Target: water level (m) — physics-inspired composite ─────────────────
    # Manning's equation influence + lagged rain + soil saturation
    base_level  = 0.8
    rain_contrib  = 0.012 * cum_24h + 0.004 * cum_72h
    soil_contrib  = 0.008 * soil
    upstream_contrib = 0.35 * upstream
    discharge_contrib = 0.0005 * discharge
    noise        = rng.normal(0, 0.08, hours)
    water_level  = base_level + rain_contrib + soil_contrib + upstream_contrib + discharge_contrib + noise
    water_level  = np.clip(water_level, 0, 10)

    df = pd.DataFrame({
        "rainfall_mm":          rainfall,
        "temperature_c":        temperature,
        "humidity_pct":         humidity,
        "wind_speed_ms":        wind,
        "pressure_hpa":         pressure,
        "river_discharge_m3s":  discharge,
        "soil_moisture_pct":    soil,
        "upstream_level_m":     upstream,
        "cumulative_rain_24h":  cum_24h,
        "cumulative_rain_72h":  cum_72h,
        "water_level_m":        water_level,
    })
    print(f"[data] Generated {len(df):,} synthetic hourly records over {n_days} days")
    return df


# ── Dataset ────────────────────────────────────────────────────────────────────

class FloodDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_sequences(df: pd.DataFrame, scaler: StandardScaler | None = None):
    """
    Slide a window of SEQ_LEN over the dataframe to build (X, y) arrays.
    Returns X, y, fitted_scaler.
    """
    features = df[FEATURE_COLS].values
    labels   = df[LABEL_COL].values

    # Fit scaler on features only
    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)

    X, y = [], []
    for i in range(SEQ_LEN, len(df)):
        X.append(features_scaled[i - SEQ_LEN : i])
        y.append(labels[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler


# ── Training loop ─────────────────────────────────────────────────────────────

def train(epochs: int = EPOCHS, csv_path: str | None = None):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    if csv_path and os.path.exists(csv_path):
        print(f"[data] Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["datetime"])
        df = df.dropna()
    else:
        df = generate_synthetic_data(n_days=730)

    X, y, scaler = make_sequences(df)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[data] Scaler saved → {SCALER_PATH}")
    print(f"[data] Sequences: {X.shape}  |  Labels range: {y.min():.2f} – {y.max():.2f} m")

    dataset = FloodDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(DEVICE)
    print(f"[model] Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[train] Device: {DEVICE}  |  Epochs: {epochs}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=0.5)   # robust to outliers

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= train_size

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(yb)
        val_loss /= val_size
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n[done] Best val loss: {best_val_loss:.4f}")
    print(f"[done] Model saved → {MODEL_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FloodSight LSTM")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--csv",    type=str, default=None,
                        help="Path to a real CSV dataset (optional)")
    args = parser.parse_args()
    train(epochs=args.epochs, csv_path=args.csv)
