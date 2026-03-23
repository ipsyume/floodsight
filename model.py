"""
model.py — FloodSight LSTM Model (PyTorch)
==========================================
Stacked Bi-LSTM with attention pooling.
Clean, simple architecture — no residual connections.
"""

import torch
import torch.nn as nn

# ── Constants shared across the project ──────────────────────────────────────
FEATURE_COLS = [
    "rainfall_mm",
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "pressure_hpa",
    "river_discharge_m3s",
    "soil_moisture_pct",
    "upstream_level_m",
    "cumulative_rain_24h",
    "cumulative_rain_72h",
]

LABEL_COL  = "water_level_m"
SEQ_LEN    = 24
INPUT_SIZE = len(FEATURE_COLS)   # 10

RISK_THRESHOLDS = {
    "Safe":     (0.0,  1.5),
    "Watch":    (1.5,  2.5),
    "Warning":  (2.5,  3.5),
    "Danger":   (3.5,  4.5),
    "Critical": (4.5,  float("inf")),
}


def level_to_risk(level_m: float) -> str:
    for label, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= level_m < hi:
            return label
    return "Critical"


# ── Model ─────────────────────────────────────────────────────────────────────

class FloodLSTM(nn.Module):
    def __init__(
        self,
        input_size:    int   = INPUT_SIZE,
        hidden_size:   int   = 128,
        num_layers:    int   = 3,
        dropout:       float = 0.25,
        bidirectional: bool  = True,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        D = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * D   # 256

        self.attn = nn.Sequential(
            nn.Linear(lstm_out_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.fc_head = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        scores  = self.attn(lstm_out)
        weights = torch.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        out = self.fc_head(context).squeeze(-1)
        return out


def build_model(device: str = "cpu") -> FloodLSTM:
    model = FloodLSTM()
    return model.to(device)