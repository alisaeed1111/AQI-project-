"""
data_fetch.py — Get raw weather + pollutant data for the city.

Two sources, chosen automatically (see config.DATA_MODE):

  • Live  : OpenWeather Air Pollution API (needs a free API key).
  • Sample: realistic *synthetic* data generated on the fly, so the whole
            project runs end-to-end with zero setup. Great for learning
            and for offline development.

Both paths return the SAME tidy daily table, so nothing downstream has to
care where the data came from:

    date | pm2_5 pm10 co no no2 o3 so2 nh3 | temp humidity pressure wind_speed | aqi
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from config import (
    BACKFILL_DAYS,
    CITY_LAT,
    CITY_LON,
    OPENWEATHER_API_KEY,
    OW_AIR_HISTORY_URL,
    OW_AIR_NOW_URL,
    OW_WEATHER_URL,
    POLLUTANT_COLS,
    RANDOM_SEED,
    WEATHER_COLS,
    using_live_api,
)
from aqi import compute_aqi, compute_aqi_series

ALL_POLLUTANTS = ["pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3"]


# ══════════════════════════════════════════════════════════════════════
#  SAMPLE (synthetic) DATA
# ══════════════════════════════════════════════════════════════════════
def _synthetic_daily(days: int) -> pd.DataFrame:
    """Generate a realistic daily pollutant + weather series for the city.

    Captures the patterns a real Karachi series shows: worse air in winter,
    slightly worse on weekdays, day-to-day persistence, and random noise.
    Deterministic (seeded) so results are reproducible.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    today = pd.Timestamp(datetime.now().date())
    dates = pd.date_range(end=today, periods=days, freq="D")

    doy = dates.dayofyear.to_numpy()
    # Seasonal signal: +1 in mid-January (worst), -1 in mid-summer (best).
    seasonal = np.cos(2 * np.pi * (doy - 15) / 365.0)

    # Weekday effect (Mon–Fri worse than the weekend).
    dow = dates.dayofweek.to_numpy()
    weekly = np.where(dow < 5, 4.0, -6.0)

    # AR(1) persistence so consecutive days are correlated (like real air).
    noise = rng.normal(0.0, 9.0, size=days)
    ar = np.zeros(days)
    phi = 0.6
    for i in range(1, days):
        ar[i] = phi * ar[i - 1] + noise[i]

    pm25 = np.clip(58.0 + 26.0 * seasonal + weekly + ar, 6.0, 320.0)
    pm10 = np.clip(pm25 * 1.7 + rng.normal(0, 12, days), 10.0, 500.0)

    # Other pollutants — loosely tied to PM2.5 with their own noise.
    scale = pm25 / 60.0
    co = np.clip(420 * scale + rng.normal(0, 40, days), 100, 3000)
    no = np.clip(3 * scale + rng.normal(0, 1.5, days), 0, 50)
    no2 = np.clip(28 * scale + rng.normal(0, 6, days), 1, 200)
    o3 = np.clip(45 - 10 * seasonal + rng.normal(0, 8, days), 1, 200)
    so2 = np.clip(14 * scale + rng.normal(0, 4, days), 1, 150)
    nh3 = np.clip(8 * scale + rng.normal(0, 2, days), 0, 100)

    # Weather.
    temp = 28.0 - 8.0 * seasonal + rng.normal(0, 1.5, days)      # °C
    humidity = np.clip(60 - 15 * seasonal + rng.normal(0, 6, days), 10, 100)
    pressure = 1010 + rng.normal(0, 3, days)                     # hPa
    wind_speed = np.clip(3.2 - 1.2 * seasonal + rng.normal(0, 0.8, days), 0.2, 15)

    df = pd.DataFrame(
        {
            "date": dates,
            "pm2_5": pm25, "pm10": pm10, "co": co, "no": no, "no2": no2,
            "o3": o3, "so2": so2, "nh3": nh3,
            "temp": temp, "humidity": humidity, "pressure": pressure,
            "wind_speed": wind_speed,
        }
    )
    df["aqi"] = compute_aqi_series(df["pm2_5"], df["pm10"]).values
    return df


# ══════════════════════════════════════════════════════════════════════
#  LIVE  (OpenWeather API)
# ══════════════════════════════════════════════════════════════════════
def _get_json(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _air_records_to_df(records: list[dict]) -> pd.DataFrame:
    """Convert OpenWeather air-pollution 'list' items into a tidy frame."""
    rows = []
    for item in records:
        comp = item.get("components", {})
        rows.append(
            {
                "dt": datetime.fromtimestamp(item["dt"], tz=timezone.utc),
                "pm2_5": comp.get("pm2_5"), "pm10": comp.get("pm10"),
                "co": comp.get("co"), "no": comp.get("no"),
                "no2": comp.get("no2"), "o3": comp.get("o3"),
                "so2": comp.get("so2"), "nh3": comp.get("nh3"),
            }
        )
    return pd.DataFrame(rows)


def _fetch_air_history_daily(days: int) -> pd.DataFrame:
    """Fetch hourly pollutant history from OpenWeather and aggregate to daily.

    The free Air Pollution History endpoint returns hourly data back to
    2020-11-27. We request it in monthly chunks to keep responses small.
    """
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    all_records: list[dict] = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=30), end)
        data = _get_json(
            OW_AIR_HISTORY_URL,
            {
                "lat": CITY_LAT, "lon": CITY_LON,
                "start": int(chunk_start.timestamp()),
                "end": int(chunk_end.timestamp()),
                "appid": OPENWEATHER_API_KEY,
            },
        )
        all_records.extend(data.get("list", []))
        chunk_start = chunk_end
        time.sleep(0.2)  # be polite to the API

    if not all_records:
        raise RuntimeError("OpenWeather returned no air-pollution history.")

    hourly = _air_records_to_df(all_records)
    hourly["date"] = pd.to_datetime(hourly["dt"]).dt.floor("D").dt.tz_localize(None)
    daily = hourly.groupby("date", as_index=False)[ALL_POLLUTANTS].mean()
    daily["aqi"] = compute_aqi_series(daily["pm2_5"], daily["pm10"]).values
    return daily


def _fetch_current_pollution() -> dict:
    data = _get_json(
        OW_AIR_NOW_URL,
        {"lat": CITY_LAT, "lon": CITY_LON, "appid": OPENWEATHER_API_KEY},
    )
    comp = data["list"][0]["components"]
    return {k: comp.get(k) for k in ALL_POLLUTANTS}


def _fetch_current_weather() -> dict:
    try:
        data = _get_json(
            OW_WEATHER_URL,
            {"lat": CITY_LAT, "lon": CITY_LON, "appid": OPENWEATHER_API_KEY,
             "units": "metric"},
        )
        return {
            "temp": data["main"].get("temp"),
            "humidity": data["main"].get("humidity"),
            "pressure": data["main"].get("pressure"),
            "wind_speed": data.get("wind", {}).get("speed"),
        }
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════
def get_raw_daily(days: int | None = None) -> pd.DataFrame:
    """Return the daily raw table, from the live API or synthetic sample.

    Falls back to synthetic data automatically if the API call fails, so a
    beginner never ends up staring at a stack trace.
    """
    days = days or BACKFILL_DAYS
    if using_live_api():
        try:
            print(f"[data] Fetching {days} days of live data from OpenWeather…")
            return _fetch_air_history_daily(days)
        except Exception as exc:  # noqa: BLE001
            print(f"[data] Live fetch failed ({exc}). Falling back to sample data.")
    else:
        print("[data] Using synthetic sample data (no API key set).")
    return _synthetic_daily(days)


def get_current() -> dict:
    """Return the most recent observation as a dict (for the dashboard header)."""
    if using_live_api():
        try:
            obs = _fetch_current_pollution()
            obs.update(_fetch_current_weather())
            obs["aqi"] = compute_aqi(obs.get("pm2_5"), obs.get("pm10"))
            obs["date"] = pd.Timestamp(datetime.now().date())
            return obs
        except Exception as exc:  # noqa: BLE001
            print(f"[data] Live current fetch failed ({exc}); using sample.")
    last = _synthetic_daily(BACKFILL_DAYS).iloc[-1].to_dict()
    return last


if __name__ == "__main__":
    df = get_raw_daily(120)
    print(f"\nRows: {len(df)}  |  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print("\nLast 5 days:")
    cols = ["date", "pm2_5", "pm10", "temp", "humidity", "wind_speed", "aqi"]
    print(df[cols].tail().to_string(index=False))
    print(f"\nAQI  min={df['aqi'].min():.0f}  mean={df['aqi'].mean():.0f}  max={df['aqi'].max():.0f}")
