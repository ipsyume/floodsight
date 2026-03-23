# FloodSight 🌊 — Full-Stack AI Flood Intelligence

A frontend flood simulation dashboard powered by a **PyTorch LSTM** backend served via **FastAPI**, with live weather data from **OpenWeatherMap**.

---

## 📁 Project Structure

```
FloodSight/
├── index.html              ← Frontend (open in browser)
├── style.css               ← All styles
├── app.js                  ← Original simulation engine
├── api.js                  ← Frontend ↔ Python backend bridge (NEW)
│
└── backend/
    ├── main.py             ← FastAPI server  ⭐
    ├── model.py            ← PyTorch LSTM architecture
    ├── train.py            ← Training script
    ├── predictor.py        ← Inference engine
    ├── data_fetch.py       ← OpenWeatherMap integration
    ├── requirements.txt
    ├── .env.example        ← Copy to .env and add your API key
    └── models/             ← Auto-created after training
        ├── flood_lstm.pt
        └── scaler.pkl
```

---

## 🚀 Quick Start

### Step 1 — Get an OpenWeatherMap API key
1. Sign up free at https://openweathermap.org/api
2. Copy your key

### Step 2 — Set up the Python backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and paste your OpenWeatherMap key
```

### Step 3 — Train the LSTM model

```bash
# Train on 2 years of synthetic data (~2 minutes on CPU)
python train.py

# Optional: more epochs for better accuracy
python train.py --epochs 80

# Optional: train on your own CSV data
python train.py --csv /path/to/your/gauging_data.csv
```

### Step 4 — Start the backend server

```bash
uvicorn main:app --reload --port 8000
```

The API is now live at: **http://localhost:8000**  
Interactive docs at: **http://localhost:8000/docs**

### Step 5 — Open the frontend

Add `api.js` to your `index.html` **before** `app.js`:

```html
<script src="api.js"></script>
<script src="app.js"></script>
```

Then just open `index.html` in your browser. The frontend will automatically:
- Detect the Python backend
- Fetch live weather when you click a city
- Replace simulated water levels with real LSTM predictions
- Show the 24-hour AI forecast

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/api/weather/{city}` | Live weather + features |
| `GET`  | `/api/weather` | All cities snapshot |
| `POST` | `/api/predict` | Single-step LSTM prediction |
| `POST` | `/api/forecast` | 24-hour flood forecast |
| `GET`  | `/api/model/info` | Model metadata |

### Example: predict for Tainan
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"city": "tainan"}'
```

### Example: manual feature input
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "rainfall_mm": 45.0,
      "temperature_c": 28.0,
      "humidity_pct": 92.0,
      "wind_speed_ms": 18.0,
      "pressure_hpa": 995.0,
      "river_discharge_m3s": 850.0,
      "soil_moisture_pct": 88.0,
      "upstream_level_m": 3.2,
      "cumulative_rain_24h": 120.0,
      "cumulative_rain_72h": 280.0
    }
  }'
```

---

## 🧠 Model Architecture

```
Input (24h × 10 features)
        │
   Input Projection (Linear)
        │
   ┌────┴────────────────────────────┐
   │  Bi-LSTM Layer 1  (128 hidden)  │
   │  + LayerNorm + Dropout + Skip   │
   │  Bi-LSTM Layer 2  (128 hidden)  │
   │  + LayerNorm + Dropout + Skip   │
   │  Bi-LSTM Layer 3  (128 hidden)  │
   │  + LayerNorm + Dropout + Skip   │
   └────┬────────────────────────────┘
        │
   Attention Pooling (over time)
        │
   FC Head: 256 → 128 → 64 → 1
        │
   Predicted Water Level (m)
```

**Input Features** (10 per timestep):
- `rainfall_mm` — current hourly rainfall
- `temperature_c` — air temperature
- `humidity_pct` — relative humidity
- `wind_speed_ms` — wind speed
- `pressure_hpa` — atmospheric pressure
- `river_discharge_m3s` — river flow rate
- `soil_moisture_pct` — soil saturation
- `upstream_level_m` — upstream gauge reading
- `cumulative_rain_24h` — 24-hour accumulated rainfall
- `cumulative_rain_72h` — 72-hour accumulated rainfall

**Risk Categories**:
| Level | Range |
|-------|-------|
| 🟢 Safe | 0.0 – 1.5 m |
| 🟡 Watch | 1.5 – 2.5 m |
| 🟠 Warning | 2.5 – 3.5 m |
| 🔴 Danger | 3.5 – 4.5 m |
| ⛔ Critical | > 4.5 m |

---

## 📊 Using Real Data

Replace synthetic training data with real gauge readings from:
- **NOAA** — https://water.noaa.gov
- **Taiwan Water Resources Agency** — https://www.wra.gov.tw
- **Global Runoff Data Centre** — https://www.bafg.de/GRDC

CSV format:
```csv
datetime,rainfall_mm,temperature_c,humidity_pct,wind_speed_ms,pressure_hpa,
river_discharge_m3s,soil_moisture_pct,upstream_level_m,cumulative_rain_24h,
cumulative_rain_72h,water_level_m
2023-01-01 00:00:00,0.2,22.4,78.0,3.1,1015.0,82.0,42.0,1.1,4.5,9.2,1.23
...
```

Then train:
```bash
python train.py --csv your_data.csv --epochs 100
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML + CSS + Vanilla JS |
| Backend | Python 3.11 + FastAPI |
| ML Model | PyTorch 2.3 (LSTM) |
| Weather API | OpenWeatherMap |
| Feature scaling | scikit-learn StandardScaler |
| ASGI server | Uvicorn |
