"""Central configuration for the Crop Yield Prediction (LSTM) project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "crop_yield.csv"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs"

# Monthly time-series inputs (one value per month -> T = 12 time steps)
WEATHER_COLS = ["rainfall", "tmax", "tmin", "humidity"]
# Season-level inputs repeated at every time step
STATIC_COLS = ["fertiliser", "irrigated_pct", "prev_yield"]
TARGET = "yield"
TIMESTEPS = 12

SEED = 42
SPLIT = (0.70, 0.15, 0.15)  # chronological: train / validation / test (by year)

LSTM_UNITS = (64, 32)
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 1e-3
