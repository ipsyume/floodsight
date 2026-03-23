"""
data_fetch.py — FloodSight Weather Data Fetcher
================================================
Fetches real-time and forecast weather data from OpenWeatherMap
and converts it into the feature vector expected by the LSTM.

APIs used:
  • Current Weather  : api.openweathermap.org/data/2.5/weather
  • 5-day/3-hour     : api.openweathermap.org/data/2.5/forecast
  • Air Pollution    : api.openweathermap.org/data/2.5/air_pollution

Set OPENWEATHER_API_KEY in your .env file.
"""

import os
import httpx
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

OWM_KEY      = os.getenv("OPENWEATHER_API_KEY", "")
OWM_BASE     = "https://api.openweathermap.org/data/2.5"
TIMEOUT_SEC  = 10

# Supported cities pre-configured (extend as needed)
CITY_COORDS = {
    "tainan":     {"lat": 22.9997,  "lon": 120.2270, "label": "Tainan"},
    "taipei":     {"lat": 25.0330,  "lon": 121.5654, "label": "Taipei"},
    "kaohsiung":  {"lat": 22.6273,  "lon": 120.3014, "label": "Kaohsiung"},
    "tokyo":      {"lat": 35.6762,  "lon": 139.6503, "label": "Tokyo"},
    "bangkok":    {"lat": 13.7563,  "lon": 100.5018, "label": "Bangkok"},
    "jakarta":    {"lat": -6.2088,  "lon": 106.8456, "label": "Jakarta"},
    "manila":     {"lat": 14.5995,  "lon": 120.9842, "label": "Manila"},
}


class WeatherFetchError(Exception):
    pass


async def _get(client: httpx.AsyncClient, url: str, params: dict) -> dict:
    params["appid"] = OWM_KEY
    try:
        resp = await client.get(url, params=params, timeout=TIMEOUT_SEC)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise WeatherFetchError(f"OWM HTTP error {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        raise WeatherFetchError(f"Network error: {e}")


async def fetch_current_weather(city_key: str) -> dict:
    """
    Returns a dict with current weather + derived features for the model.
    """
    if not OWM_KEY:
        raise WeatherFetchError("OPENWEATHER_API_KEY not set in .env")

    coords = CITY_COORDS.get(city_key.lower())
    if not coords:
        raise WeatherFetchError(f"Unknown city '{city_key}'. Available: {list(CITY_COORDS)}")

    lat, lon = coords["lat"], coords["lon"]

    async with httpx.AsyncClient() as client:
        # ── Current weather ───────────────────────────────────────────────
        curr = await _get(client, f"{OWM_BASE}/weather", {
            "lat": lat, "lon": lon, "units": "metric"
        })

        # ── 5-day forecast (3-hour intervals) ─────────────────────────────
        fcast = await _get(client, f"{OWM_BASE}/forecast", {
            "lat": lat, "lon": lon, "units": "metric", "cnt": 16  # ~48 hours
        })

    # ── Parse current ─────────────────────────────────────────────────────
    rain_1h       = curr.get("rain", {}).get("1h", 0.0)
    temperature   = curr["main"]["temp"]
    humidity      = curr["main"]["humidity"]
    pressure      = curr["main"]["pressure"]
    wind_speed    = curr["wind"]["speed"]
    weather_desc  = curr["weather"][0]["description"] if curr.get("weather") else "n/a"
    weather_icon  = curr["weather"][0]["icon"]         if curr.get("weather") else "01d"
    clouds        = curr["clouds"]["all"]
    visibility    = curr.get("visibility", 10000) / 1000  # → km

    # ── Cumulative rain from forecast history (24h / 72h from past items) ─
    rain_vals = [entry.get("rain", {}).get("3h", 0.0) for entry in fcast["list"]]
    # 3h intervals: 8 items ≈ 24h, 24 items ≈ 72h
    cum_24h   = sum(rain_vals[:8])
    cum_72h   = sum(rain_vals[:24])

    # ── Derive soil moisture proxy from cumulative rain ────────────────────
    soil_moisture_proxy = min(100.0, 30 + cum_72h * 0.3)

    # ── Derive upstream level proxy (simplified hydraulic model) ──────────
    upstream_proxy = 0.8 + cum_24h * 0.015

    # ── Derive river discharge proxy ──────────────────────────────────────
    discharge_proxy = 50 + upstream_proxy * 30 + cum_24h * 0.5

    # ── Forecast summary (next 24 h) ──────────────────────────────────────
    forecast_24h = []
    for entry in fcast["list"][:8]:
        forecast_24h.append({
            "time":        entry["dt_txt"],
            "temp_c":      entry["main"]["temp"],
            "humidity":    entry["main"]["humidity"],
            "rain_3h_mm":  entry.get("rain", {}).get("3h", 0.0),
            "wind_ms":     entry["wind"]["speed"],
            "description": entry["weather"][0]["description"],
            "icon":        entry["weather"][0]["icon"],
        })

    return {
        # Raw weather for display
        "city":          coords["label"],
        "lat":           lat,
        "lon":           lon,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "temperature_c": round(temperature, 1),
        "humidity_pct":  humidity,
        "pressure_hpa":  pressure,
        "wind_speed_ms": round(wind_speed, 2),
        "rain_1h_mm":    round(rain_1h, 2),
        "clouds_pct":    clouds,
        "visibility_km": round(visibility, 1),
        "description":   weather_desc,
        "icon":          weather_icon,
        "forecast_24h":  forecast_24h,

        # LSTM feature vector (same order as FEATURE_COLS)
        "model_features": {
            "rainfall_mm":          round(rain_1h, 3),
            "temperature_c":        round(temperature, 2),
            "humidity_pct":         round(humidity, 2),
            "wind_speed_ms":        round(wind_speed, 3),
            "pressure_hpa":         round(pressure, 2),
            "river_discharge_m3s":  round(discharge_proxy, 2),
            "soil_moisture_pct":    round(soil_moisture_proxy, 2),
            "upstream_level_m":     round(upstream_proxy, 3),
            "cumulative_rain_24h":  round(cum_24h, 3),
            "cumulative_rain_72h":  round(cum_72h, 3),
        },
    }


async def fetch_multi_city_summary() -> list[dict]:
    """Fetch a brief weather snapshot for all pre-configured cities."""
    results = []
    for key in CITY_COORDS:
        try:
            data = await fetch_current_weather(key)
            results.append({
                "city_key":      key,
                "city":          data["city"],
                "temperature_c": data["temperature_c"],
                "humidity_pct":  data["humidity_pct"],
                "rain_1h_mm":    data["rain_1h_mm"],
                "wind_speed_ms": data["wind_speed_ms"],
                "description":   data["description"],
                "icon":          data["icon"],
            })
        except WeatherFetchError:
            results.append({"city_key": key, "city": CITY_COORDS[key]["label"], "error": "fetch_failed"})
    return results
