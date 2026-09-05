"""
features.py — Turn the raw daily table into model-ready features.

We create three families of features (all computed only from the current
day and the PAST, so there is no "peeking into the future" / data leakage):

  • Time-based   : day-of-week, month, weekend flag, cyclical encodings.
  • Lag          : AQI 1/2/3/7 days ago, PM2.5 yesterday.
  • Rolling      : 3- and 7-day rolling mean / std of AQI.
  • Change rate  : how fast AQI is moving (today minus yesterday).
  • Same-day     : today's pollutant + weather readings.

Targets (what we want to predict) are AQI 1, 2 and 3 days into the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLS = ["target_d1", "target_d2", "target_d3"]

# Same-day columns we keep as features when they are present in the data.
_CANDIDATE_SAME_DAY = [
    "pm2_5", "pm10", "co", "no", "no2", "o3", "so2", "nh3",
    "temp", "humidity", "pressure", "wind_speed", "aqi",
]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the engineered feature table from the raw daily table."""
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Fill any calendar gaps (real API data can miss days), then interpolate.
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full)
    df.index.name = "date"
    numeric = df.select_dtypes(include="number").columns
    df[numeric] = df[numeric].interpolate(limit_direction="both")
    df = df.reset_index()

    # ── Time-based features ────────────────────────────────────────────
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # Cyclical encodings so the model knows Dec is next to Jan.
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

    # ── Lag features ───────────────────────────────────────────────────
    for lag in (1, 2, 3, 7):
        df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
    df["pm2_5_lag_1"] = df["pm2_5"].shift(1)

    # ── Rolling features ───────────────────────────────────────────────
    df["aqi_roll_mean_3"] = df["aqi"].rolling(3, min_periods=1).mean()
    df["aqi_roll_mean_7"] = df["aqi"].rolling(7, min_periods=1).mean()
    df["aqi_roll_std_7"] = df["aqi"].rolling(7, min_periods=2).std()
    df["pm2_5_roll_mean_3"] = df["pm2_5"].rolling(3, min_periods=1).mean()

    # ── Change rate ────────────────────────────────────────────────────
    df["aqi_change"] = df["aqi"] - df["aqi"].shift(1)
    df["aqi_change_3"] = df["aqi"] - df["aqi"].shift(3)

    # ── Targets: AQI 1/2/3 days ahead ─────────────────────────────────
    df["target_d1"] = df["aqi"].shift(-1)
    df["target_d2"] = df["aqi"].shift(-2)
    df["target_d3"] = df["aqi"].shift(-3)

    # Drop early rows that don't have complete lag/rolling features.
    feat_cols = feature_columns(df)
    df = df[df[feat_cols].notna().all(axis=1)].reset_index(drop=True)

    # Guard against any stray infinities.
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df = df[df[feat_cols].notna().all(axis=1)].reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model input columns = everything except the date and the targets."""
    exclude = set(["date"] + TARGET_COLS)
    return [c for c in df.columns if c not in exclude]


def latest_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """The most recent row — used to forecast the next 3 days."""
    return df.sort_values("date").iloc[[-1]].reset_index(drop=True)


if __name__ == "__main__":
    from data_fetch import get_raw_daily

    raw = get_raw_daily(120)
    feats = build_features(raw)
    fc = feature_columns(feats)
    print(f"Raw rows: {len(raw)}  ->  feature rows: {len(feats)}")
    print(f"Feature count: {len(fc)}")
    print(f"Features: {fc}")
    print(f"\nAny NaN in features? {feats[fc].isna().any().any()}")
    print("\nLast row (targets are NaN because those days are still in the future):")
    print(feats[["date", "aqi"] + TARGET_COLS].tail(4).to_string(index=False))
