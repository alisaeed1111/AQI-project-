"""
explain.py — Feature-importance explanations for the AQI model.

Two methods:
  • Permutation importance — model-agnostic, always available. We shuffle
    one feature at a time and measure how much the error (RMSE) grows; the
    bigger the growth, the more important the feature.
  • SHAP — richer, game-theory based attributions for the tree model
    (Random Forest). Used when the `shap` package and a tree model are
    available; otherwise we simply fall back to permutation importance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import MODEL_PATH, RANDOM_SEED
from feature_store import load_features
from features import TARGET_COLS
from train import make_xy, time_split


# ── Universal permutation importance (pure numpy/pandas) ───────────────
def permutation_importance(predict_fn, X: pd.DataFrame, Y: pd.DataFrame,
                           n_repeats: int = 5, seed: int = RANDOM_SEED) -> pd.Series:
    """Return a Series of importances (mean RMSE increase) indexed by feature."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(Y, dtype="float64")

    def rmse(pred):
        pred = np.asarray(pred, dtype="float64").reshape(y_true.shape)
        return float(np.sqrt(np.mean((y_true - pred) ** 2)))

    base = rmse(predict_fn(X))
    importances = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(rmse(predict_fn(Xp)) - base)
        importances[col] = float(np.mean(drops))
    return pd.Series(importances).sort_values(ascending=False)


def model_feature_importance(top_n: int = 15, sample_n: int = 120) -> pd.Series:
    """Permutation importance for the currently registered model."""
    from predict import _load_predictor

    df = load_features()
    X, Y, feat_cols = make_xy(df)
    _, te = time_split(len(X))
    X_eval, Y_eval = X.iloc[te], Y.iloc[te]
    if len(X_eval) > sample_n:
        X_eval = X_eval.iloc[-sample_n:]
        Y_eval = Y_eval.iloc[-sample_n:]

    predict_fn, feat_names, _ = _load_predictor()
    imp = permutation_importance(predict_fn, X_eval[feat_names], Y_eval)
    return imp.head(top_n)


# ── SHAP (optional, for the tree model) ────────────────────────────────
def shap_bar(top_n: int = 15, sample_n: int = 150):
    """Return (feature_names, mean_abs_shap) for the tree model, or None."""
    try:
        import joblib
        import shap

        bundle = joblib.load(MODEL_PATH)
        if bundle.get("model_type") != "sklearn":
            return None
        model = bundle["model"]
        feat_names = bundle["feature_names"]
        # SHAP's TreeExplainer needs the underlying tree estimator.
        if not hasattr(model, "estimators_") and not hasattr(model, "feature_importances_"):
            return None

        df = load_features()
        X, _, _ = make_xy(df)
        X_sample = X[feat_names].tail(sample_n)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        # Multi-output -> list of arrays (one per horizon); average |value|.
        if isinstance(shap_vals, list):
            arr = np.mean([np.abs(v) for v in shap_vals], axis=0)
        else:
            arr = np.abs(shap_vals)
            if arr.ndim == 3:  # (samples, features, outputs)
                arr = arr.mean(axis=2)
        mean_abs = arr.mean(axis=0)
        s = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False).head(top_n)
        return list(s.index), list(s.values)
    except Exception as exc:  # noqa: BLE001
        print(f"[explain] SHAP unavailable ({exc}); using permutation importance.")
        return None


if __name__ == "__main__":
    imp = model_feature_importance()
    print("Top features driving the AQI forecast (permutation importance):")
    for name, val in imp.items():
        print(f"  {name:<20} {val:6.2f}")
