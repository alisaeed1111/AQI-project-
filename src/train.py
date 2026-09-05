"""
train.py — The training pipeline.

Steps (matching the project spec):
  1. Load engineered features from the Feature Store.
  2. Build the (X, Y) training matrix — Y has 3 columns (AQI 1/2/3 days out).
  3. Split chronologically into train / test (never shuffle time series!).
  4. Train several models: Ridge Regression, Random Forest, and a small
     TensorFlow neural network.
  5. Evaluate each with RMSE, MAE and R² (per horizon and on average).
  6. Save the BEST model to the local Model Registry, plus a metrics file.

Heavy libraries (scikit-learn / TensorFlow / joblib) are imported *inside*
the functions that use them, so the data-prep helpers can be imported and
unit-tested even in a minimal environment.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_SEED,
    REGISTRY_PATH,
    TF_MODEL_PATH,
)
from features import TARGET_COLS, feature_columns
from feature_store import load_features


# ── Pure data-prep helpers (no heavy deps) ─────────────────────────────
def make_xy(df: pd.DataFrame):
    """Return (X, Y, feature_names) for rows that have all 3 targets."""
    feat_cols = feature_columns(df)
    train_df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    X = train_df[feat_cols].astype("float64").copy()
    Y = train_df[TARGET_COLS].astype("float64").copy()
    return X, Y, feat_cols


def time_split(n: int, test_frac: float = 0.2):
    """Chronological split: earliest rows train, most recent rows test."""
    n_test = max(1, int(round(n * test_frac)))
    idx = np.arange(n)
    return idx[:-n_test], idx[-n_test:]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE / MAE / R² per horizon plus the average across horizons."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    out = {}
    for i, name in enumerate(["d1", "d2", "d3"]):
        out[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
    out["mean"] = {
        k: float(np.mean([out[h][k] for h in ["d1", "d2", "d3"]]))
        for k in ("rmse", "mae", "r2")
    }
    return out


# ── Model builders ─────────────────────────────────────────────────────
def _build_ridge():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=RANDOM_SEED))


def _build_random_forest():
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def _train_tf(X_train, Y_train, X_test, Y_test):
    """Train a small Keras MLP. Returns (model, scaler, y_pred_test) or None."""
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # noqa: BLE001
        print(f"[train] TensorFlow not available ({exc}); skipping NN model.")
        return None

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(Xtr.shape[1],)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(3),  # 3 forecast horizons
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    es = tf.keras.callbacks.EarlyStopping(
        patience=20, restore_best_weights=True, monitor="val_loss"
    )
    model.fit(
        Xtr, Y_train.values,
        validation_split=0.2, epochs=300, batch_size=16,
        callbacks=[es], verbose=0,
    )
    y_pred = model.predict(Xte, verbose=0)
    return model, scaler, y_pred


# ── Main training routine ──────────────────────────────────────────────
def train(verbose: bool = True) -> dict:
    df = load_features()
    X, Y, feat_cols = make_xy(df)
    if len(X) < 30:
        raise RuntimeError(
            f"Only {len(X)} training rows — run the backfill first "
            f"(python pipeline.py backfill)."
        )
    tr, te = time_split(len(X))
    X_train, X_test = X.iloc[tr], X.iloc[te]
    Y_train, Y_test = Y.iloc[tr], Y.iloc[te]

    import joblib

    results = {}          # model_name -> metrics dict
    candidates = {}       # model_name -> ("sklearn"/"tf", payload)

    # --- Ridge & Random Forest -------------------------------------------------
    for name, builder in (("Ridge", _build_ridge), ("RandomForest", _build_random_forest)):
        model = builder()
        model.fit(X_train, Y_train)
        preds = np.asarray(model.predict(X_test))
        results[name] = _metrics(Y_test.values, preds)
        candidates[name] = ("sklearn", model)
        if verbose:
            m = results[name]["mean"]
            print(f"[train] {name:<13} RMSE={m['rmse']:6.2f}  MAE={m['mae']:6.2f}  R²={m['r2']:.3f}")

    # --- TensorFlow neural net (optional) --------------------------------------
    tf_out = _train_tf(X_train, Y_train, X_test, Y_test)
    if tf_out is not None:
        tf_model, tf_scaler, tf_pred = tf_out
        results["TensorFlow"] = _metrics(Y_test.values, tf_pred)
        candidates["TensorFlow"] = ("tf", (tf_model, tf_scaler))
        if verbose:
            m = results["TensorFlow"]["mean"]
            print(f"[train] {'TensorFlow':<13} RMSE={m['rmse']:6.2f}  MAE={m['mae']:6.2f}  R²={m['r2']:.3f}")

    # --- Pick the best model (lowest mean RMSE) --------------------------------
    best_name = min(results, key=lambda k: results[k]["mean"]["rmse"])
    kind, payload = candidates[best_name]
    if verbose:
        print(f"[train] Best model: {best_name}")

    # --- Save to the model registry --------------------------------------------
    if kind == "sklearn":
        joblib.dump(
            {"model": payload, "feature_names": feat_cols,
             "model_type": "sklearn", "model_name": best_name},
            MODEL_PATH,
        )
        artifact = MODEL_PATH.name
    else:  # tf
        tf_model, tf_scaler = payload
        tf_model.save(TF_MODEL_PATH)
        joblib.dump(
            {"scaler": tf_scaler, "feature_names": feat_cols,
             "model_type": "tf", "model_name": best_name,
             "tf_path": str(TF_MODEL_PATH)},
            MODEL_PATH,
        )
        artifact = TF_MODEL_PATH.name

    registry = {
        "model_name": best_name,
        "model_type": kind,
        "artifact": artifact,
        "feature_names": feat_cols,
        "target_cols": TARGET_COLS,
        "metrics": results[best_name],
        "all_metrics": results,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    if verbose:
        print(f"[train] Saved best model + metadata to {MODEL_PATH.parent}")
    return registry


if __name__ == "__main__":
    train()
