"""
feature_store.py — A tiny local "Feature Store".

In the full serverless design this role is played by Hopsworks (or Vertex
AI). Here we keep the exact same *idea* — one central place to write and
read engineered features — but back it with a file on disk so the project
runs with no cloud account.

Swapping in Hopsworks later means changing ONLY this file: replace the
save/load bodies with `feature_group.insert(df)` / `feature_view.get_...()`.
Nothing else in the project needs to change.
"""
from __future__ import annotations

import pandas as pd

from config import FEATURES_DIR

PARQUET_PATH = FEATURES_DIR / "aqi_features.parquet"
CSV_PATH = FEATURES_DIR / "aqi_features.csv"


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def save_features(df: pd.DataFrame):
    """Persist the engineered feature table. Returns the path written."""
    if _parquet_available():
        df.to_parquet(PARQUET_PATH, index=False)
        return PARQUET_PATH
    df.to_csv(CSV_PATH, index=False)
    return CSV_PATH


def load_features() -> pd.DataFrame:
    """Load the engineered feature table (parquet preferred, else CSV)."""
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH)
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH, parse_dates=["date"])
    raise FileNotFoundError(
        "No features found yet. Run the feature pipeline first:\n"
        "    python pipeline.py backfill"
    )


def features_exist() -> bool:
    return PARQUET_PATH.exists() or CSV_PATH.exists()
