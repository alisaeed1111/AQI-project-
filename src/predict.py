"""
predict.py — Load the trained model + latest features and forecast the
Air Quality Index for the next 3 days.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import MODEL_PATH, REGISTRY_PATH, aqi_category
from feature_store import load_features
from features import latest_feature_row


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Train one first:\n    python pipeline.py train"
        )
    return json.loads(REGISTRY_PATH.read_text())


def _load_predictor():
    """Return (predict_fn, feature_names, registry).

    predict_fn takes a DataFrame of features and returns an (n, 3) array.
    Works for both the scikit-learn and TensorFlow model types.
    """
    import joblib

    reg = load_registry()
    bundle = joblib.load(MODEL_PATH)
    feat_names = bundle["feature_names"]
    kind = bundle["model_type"]

    if kind == "sklearn":
        model = bundle["model"]

        def predict_fn(X: pd.DataFrame) -> np.ndarray:
            return np.asarray(model.predict(X[feat_names]), dtype="float64").reshape(-1, 3)

    else:  # tensorflow
        import tensorflow as tf

        scaler = bundle["scaler"]
        model = tf.keras.models.load_model(bundle["tf_path"])

        def predict_fn(X: pd.DataFrame) -> np.ndarray:
            Xs = scaler.transform(X[feat_names])
            return np.asarray(model.predict(Xs, verbose=0), dtype="float64").reshape(-1, 3)

    return predict_fn, feat_names, reg


def forecast() -> tuple[pd.DataFrame, dict, pd.Timestamp]:
    """Forecast AQI for the next 3 days.

    Returns (forecast_df, registry, last_observed_date). forecast_df has
    columns: date, day_offset, aqi, category, color.
    """
    df = load_features()
    predict_fn, feat_names, reg = _load_predictor()

    row = latest_feature_row(df)
    X = row.reindex(columns=feat_names).fillna(0.0)
    yhat = predict_fn(X)[0]  # (3,)

    last_date = pd.to_datetime(row["date"].iloc[0])
    records = []
    for i, val in enumerate(yhat, start=1):
        aqi_val = float(np.clip(val, 0, 500))
        label, color = aqi_category(aqi_val)
        records.append(
            {
                "date": last_date + pd.Timedelta(days=i),
                "day_offset": i,
                "aqi": round(aqi_val, 1),
                "category": label,
                "color": color,
            }
        )
    return pd.DataFrame(records), reg, last_date


def forecast_with_history(history_days: int = 30):
    """Combined table of recent actual AQI + the 3-day forecast (for charts)."""
    df = load_features().sort_values("date")
    hist = df[["date", "aqi"]].tail(history_days).copy()
    hist["kind"] = "actual"

    fc, reg, last_date = forecast()
    fc_part = fc[["date", "aqi"]].copy()
    fc_part["kind"] = "forecast"

    combined = pd.concat([hist, fc_part], ignore_index=True)
    return combined, fc, reg


if __name__ == "__main__":
    fc, reg, last_date = forecast()
    print(f"Model: {reg['model_name']} ({reg['model_type']})  |  last observed: {last_date.date()}")
    print("\n3-day AQI forecast for the city:")
    for _, r in fc.iterrows():
        print(f"  {r['date'].date()}  AQI {r['aqi']:>5.0f}  — {r['category']}")
