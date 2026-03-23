"""
main.py — FloodSight FastAPI Backend
=====================================
Run:  uvicorn main:app --reload --port 8000

Endpoints
---------
GET  /                          Health check
GET  /api/weather/{city}        Live weather + derived features
GET  /api/weather/all           Summary for all pre-configured cities
POST /api/predict               Single-step flood prediction
POST /api/forecast              24-hour flood forecast
GET  /api/model/info            Model metadata
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from data_fetch import fetch_current_weather, fetch_multi_city_summary, WeatherFetchError
from predictor  import load_model, predict_level, predict_forecast
from model      import FEATURE_COLS, SEQ_LEN, RISK_THRESHOLDS


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model once at startup."""
    try:
        load_model()
        print("[startup] LSTM model ready ✓")
    except FileNotFoundError as e:
        print(f"[startup] WARNING: {e}")
        print("[startup] Run `python train.py` to train the model first.")
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FloodSight API",
    description="AI-powered flood prediction backend — LSTM + OpenWeatherMap",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the frontend (any origin during dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Lock this down in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class FeatureInput(BaseModel):
    """Manual feature override — all values optional; defaults are moderate conditions."""
    rainfall_mm:          float = Field(default=5.0,    ge=0,    le=200,  description="Current rainfall mm/hr")
    temperature_c:        float = Field(default=25.0,   ge=-10,  le=50)
    humidity_pct:         float = Field(default=75.0,   ge=0,    le=100)
    wind_speed_ms:        float = Field(default=5.0,    ge=0,    le=90)
    pressure_hpa:         float = Field(default=1010.0, ge=800,  le=1100)
    river_discharge_m3s:  float = Field(default=100.0,  ge=0)
    soil_moisture_pct:    float = Field(default=50.0,   ge=0,    le=100)
    upstream_level_m:     float = Field(default=1.5,    ge=0)
    cumulative_rain_24h:  float = Field(default=20.0,   ge=0)
    cumulative_rain_72h:  float = Field(default=40.0,   ge=0)


class PredictRequest(BaseModel):
    city:     Optional[str]          = Field(default=None, description="Fetch live weather for this city (overrides manual features)")
    features: Optional[FeatureInput] = Field(default=None, description="Manual feature override")


class ForecastRequest(PredictRequest):
    steps: int = Field(default=24, ge=1, le=72, description="Hours ahead to forecast")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {
        "status": "ok",
        "service": "FloodSight API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/api/weather/{city}", tags=["weather"])
async def get_weather(city: str):
    """
    Fetch current weather + 24-h forecast for a city from OpenWeatherMap.
    Also returns the derived feature vector ready for the LSTM.

    Cities: tainan, taipei, kaohsiung, tokyo, bangkok, jakarta, manila
    """
    try:
        data = await fetch_current_weather(city)
    except WeatherFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return data


@app.get("/api/weather", tags=["weather"])
async def get_all_weather():
    """Brief snapshot for all configured cities."""
    try:
        data = await fetch_multi_city_summary()
    except WeatherFetchError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"cities": data}


@app.post("/api/predict", tags=["prediction"])
async def predict(req: PredictRequest):
    """
    Single-step flood level prediction.

    If `city` is provided, fetches live weather from OpenWeatherMap.
    Otherwise uses the `features` block (or defaults).

    Returns:
        predicted_level_m, risk_category, confidence_pct, feature_importance,
        weather_data (if city was provided)
    """
    weather_data = None

    if req.city:
        try:
            weather_data = await fetch_current_weather(req.city)
            feat_dict = weather_data["model_features"]
        except WeatherFetchError as e:
            raise HTTPException(status_code=502, detail=str(e))
    else:
        feat_input = req.features or FeatureInput()
        feat_dict  = feat_input.model_dump()

    try:
        result = predict_level(feat_dict)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e) + " — run python train.py first")

    response = {
        "input_features": feat_dict,
        **result,
    }
    if weather_data:
        response["weather"] = {
            k: weather_data[k]
            for k in ("city", "temperature_c", "humidity_pct", "wind_speed_ms",
                       "rain_1h_mm", "description", "icon", "forecast_24h")
        }
    return response


@app.post("/api/forecast", tags=["prediction"])
async def forecast(req: ForecastRequest):
    """
    Autoregressive N-step (default 24-hour) flood forecast.

    Returns a list of {hour, predicted_level_m, risk_category, confidence_pct}.
    """
    weather_data = None

    if req.city:
        try:
            weather_data = await fetch_current_weather(req.city)
            feat_dict = weather_data["model_features"]
        except WeatherFetchError as e:
            raise HTTPException(status_code=502, detail=str(e))
    else:
        feat_input = req.features or FeatureInput()
        feat_dict  = feat_input.model_dump()

    try:
        steps = predict_forecast(feat_dict, steps=req.steps)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e) + " — run python train.py first")

    city_label = weather_data["city"] if weather_data else "Manual Input"
    return {
        "city":     city_label,
        "steps":    req.steps,
        "forecast": steps,
    }


@app.get("/api/model/info", tags=["model"])
async def model_info():
    """Returns model architecture metadata and feature descriptions."""
    return {
        "architecture":    "Stacked Bi-directional LSTM with Attention Pooling",
        "framework":       "PyTorch",
        "sequence_length": SEQ_LEN,
        "input_features":  FEATURE_COLS,
        "output":          "water_level_m (regression)",
        "risk_thresholds": {k: list(v) for k, v in RISK_THRESHOLDS.items()},
        "training_data":   "730-day synthetic hydrological simulation (replace with real gauging data)",
        "physics_basis":   [
            "Manning's equation (open-channel flow)",
            "SCS Curve Number (runoff estimation)",
            "Horton infiltration model",
            "Lagged rainfall response",
        ],
    }
