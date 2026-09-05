"""
config.py — Central configuration for the Pearls AQI Predictor.

Everything that the rest of the project needs to know (where files live,
which city we forecast, API settings, AQI categories) is defined here in
one place, so you only ever change settings in a single file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the environment.
load_dotenv()

# ── Project paths ──────────────────────────────────────────────────────
# ROOT is the top-level project folder (the parent of this src/ folder).
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"

# Make sure the folders exist (harmless if they already do).
for _d in (RAW_DIR, FEATURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# The "feature store" is just a parquet file on disk in local mode.
FEATURE_STORE_PATH = FEATURES_DIR / "aqi_features.parquet"
# The "model registry" lives in this folder (one file per artefact).
MODEL_PATH = MODELS_DIR / "aqi_model.joblib"
TF_MODEL_PATH = MODELS_DIR / "aqi_model_tf.keras"
METRICS_PATH = MODELS_DIR / "metrics.json"
REGISTRY_PATH = MODELS_DIR / "registry.json"

# ── City ───────────────────────────────────────────────────────────────
CITY_NAME = os.getenv("CITY_NAME", "Karachi")
CITY_LAT = float(os.getenv("CITY_LAT", "24.8607"))
CITY_LON = float(os.getenv("CITY_LON", "67.0011"))

# ── Data source ────────────────────────────────────────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()
# "auto"  -> use the API when a key is set, otherwise fall back to sample data
# "sample"-> always use synthetic sample data (great for learning / offline)
# "api"   -> force the live API (errors if no key)
DATA_MODE = os.getenv("DATA_MODE", "auto").strip().lower()

# OpenWeather Air Pollution API endpoints
OW_AIR_NOW_URL = "http://api.openweathermap.org/data/2.5/air_pollution"
OW_AIR_HISTORY_URL = "http://api.openweathermap.org/data/2.5/air_pollution/history"
OW_AIR_FORECAST_URL = "http://api.openweathermap.org/data/2.5/air_pollution/forecast"
OW_WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# ── Modelling settings ─────────────────────────────────────────────────
FORECAST_HORIZON_DAYS = 3          # predict AQI for the next 3 days
BACKFILL_DAYS = 365                # how much history to build for training
RANDOM_SEED = 42

# Columns we treat as raw pollutant measurements (µg/m³) from the API.
POLLUTANT_COLS = ["pm2_5", "pm10", "co", "no2", "so2", "o3", "no", "nh3"]
# Weather columns.
WEATHER_COLS = ["temp", "humidity", "pressure", "wind_speed"]

# ── AQI categories (US EPA scale, 0–500) ───────────────────────────────
# Each entry: (upper_bound, label, hex_colour)
AQI_CATEGORIES = [
    (50, "Good", "#009966"),
    (100, "Moderate", "#ffde33"),
    (150, "Unhealthy for Sensitive Groups", "#ff9933"),
    (200, "Unhealthy", "#cc0033"),
    (300, "Very Unhealthy", "#660099"),
    (500, "Hazardous", "#7e0023"),
]
# Send an alert when a forecast AQI is at/above this value ("Unhealthy").
HAZARD_THRESHOLD = 151


def aqi_category(aqi: float) -> tuple[str, str]:
    """Return (label, colour) for a given AQI value."""
    if aqi is None:
        return ("Unknown", "#888888")
    for upper, label, colour in AQI_CATEGORIES:
        if aqi <= upper:
            return (label, colour)
    return ("Hazardous", "#7e0023")


def using_live_api() -> bool:
    """True when we should call the real OpenWeather API."""
    if DATA_MODE == "sample":
        return False
    if DATA_MODE == "api":
        return True
    # auto
    return bool(OPENWEATHER_API_KEY)


if __name__ == "__main__":
    # Quick sanity check: `python src/config.py`
    print(f"City            : {CITY_NAME} ({CITY_LAT}, {CITY_LON})")
    print(f"Data mode       : {DATA_MODE}")
    print(f"Using live API  : {using_live_api()}")
    print(f"Project root    : {ROOT}")
    print(f"Feature store   : {FEATURE_STORE_PATH}")
