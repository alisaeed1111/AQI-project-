"""
dashboard.py — Pearls AQI Predictor web dashboard (Streamlit).

Run it from the project root with:

    streamlit run app/dashboard.py

Features
  • Latest observed AQI with its colour-coded health category.
  • 3-day AQI forecast (cards + chart with AQI health bands).
  • Hazardous-air alerts.
  • Exploratory Data Analysis: history, seasonal pattern, distribution.
  • Feature-importance explanations (SHAP for the tree model, else
    permutation importance).
  • Model metrics (RMSE / MAE / R²) and a model-comparison table.
  • One-click buttons to fetch data, build features and train models —
    so you never need the terminal if you don't want it.
"""
from __future__ import annotations

import os
import sys

# Make the src/ package importable no matter where Streamlit is launched from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    AQI_CATEGORIES,
    CITY_NAME,
    HAZARD_THRESHOLD,
    MODEL_PATH,
    REGISTRY_PATH,
    aqi_category,
    using_live_api,
)
import explain as explain_mod
import feature_pipeline
import predict as predict_mod
import train as train_mod
from feature_store import features_exist, load_features

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🌫️", layout="wide")


# ── Small helpers ──────────────────────────────────────────────────────
def aqi_badge(value: float, subtitle: str = "") -> str:
    label, color = aqi_category(value)
    sub = f"<div style='font-size:0.8rem;color:#555'>{subtitle}</div>" if subtitle else ""
    return (
        f"<div style='background:{color};padding:16px 20px;border-radius:14px;"
        f"color:white;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.15)'>"
        f"<div style='font-size:2.6rem;font-weight:800;line-height:1'>{value:.0f}</div>"
        f"<div style='font-size:0.95rem;font-weight:600'>{label}</div>{sub}</div>"
    )


def forecast_chart(combined: pd.DataFrame):
    """History + forecast line chart with shaded AQI health bands."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for low, high, color in [
        (0, 50, "#009966"), (50, 100, "#ffde33"), (100, 150, "#ff9933"),
        (150, 200, "#cc0033"), (200, 300, "#660099"), (300, 500, "#7e0023"),
    ]:
        ax.axhspan(low, high, color=color, alpha=0.10)

    actual = combined[combined["kind"] == "actual"]
    fc = combined[combined["kind"] == "forecast"]
    ax.plot(actual["date"], actual["aqi"], color="#222", lw=2, label="Observed AQI")
    if not actual.empty and not fc.empty:
        bridge = pd.concat([actual.tail(1), fc])
        ax.plot(bridge["date"], bridge["aqi"], color="#d6336c", lw=2.5,
                ls="--", marker="o", label="Forecast (next 3 days)")

    ax.axhline(HAZARD_THRESHOLD, color="#cc0033", lw=1, ls=":", alpha=0.7)
    ax.set_ylabel("AQI")
    ax.set_ylim(0, max(220, combined["aqi"].max() * 1.15))
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.15)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def run_pipeline_and_train(do_backfill: bool = True):
    if do_backfill:
        with st.spinner("Fetching data and building features…"):
            feature_pipeline.run(verbose=False)
    with st.spinner("Training models (Ridge, Random Forest, TensorFlow)…"):
        reg = train_mod.train(verbose=False)
    st.cache_data.clear()
    return reg


# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")
    st.caption(f"City: **{CITY_NAME}**")
    st.caption("Data source: " + ("🌐 Live OpenWeather" if using_live_api() else "🧪 Sample (synthetic)"))
    st.divider()
    if st.button("① Fetch data & build features", use_container_width=True):
        with st.spinner("Building features…"):
            feature_pipeline.run(verbose=False)
        st.cache_data.clear()
        st.success("Features updated.")
    if st.button("② Train models", use_container_width=True, disabled=not features_exist()):
        run_pipeline_and_train(do_backfill=False)
        st.success("Training complete.")
    if st.button("🔄 Do everything (build + train)", use_container_width=True, type="primary"):
        run_pipeline_and_train(do_backfill=True)
        st.success("Done! Dashboard updated.")
    st.divider()
    st.caption("Tip: add a free OpenWeather API key in `.env` to switch from "
               "sample data to live data.")


# ── Cached data accessors (keyed on model train time) ──────────────────
@st.cache_data(show_spinner=False)
def _forecast(cache_key: str):
    return predict_mod.forecast_with_history(history_days=45)


@st.cache_data(show_spinner=False)
def _importance(cache_key: str):
    shap_res = explain_mod.shap_bar(top_n=12)
    if shap_res is not None:
        names, vals = shap_res
        return pd.Series(vals, index=names), "SHAP (mean |impact|)"
    imp = explain_mod.model_feature_importance(top_n=12)
    return imp, "Permutation importance (RMSE increase)"


# ── Header ─────────────────────────────────────────────────────────────
st.title("🌫️ Pearls AQI Predictor")
st.caption(f"3-day Air Quality Index forecast for **{CITY_NAME}** — an end-to-end serverless ML pipeline.")

model_ready = REGISTRY_PATH.exists() and MODEL_PATH.exists()

if not model_ready:
    st.info(
        "👋 **Welcome!** No trained model yet. Click the button below to build "
        "sample data and train the models — it takes under a minute."
    )
    if st.button("🚀 Build sample data & train model now", type="primary"):
        run_pipeline_and_train(do_backfill=True)
        st.rerun()
    st.stop()

# ── Load everything ────────────────────────────────────────────────────
reg = predict_mod.load_registry()
cache_key = reg.get("trained_at", "")
combined, fc, _ = _forecast(cache_key)
feat_df = load_features().sort_values("date")
latest = feat_df.iloc[-1]

# ── Alert banner ───────────────────────────────────────────────────────
bad_days = fc[fc["aqi"] >= HAZARD_THRESHOLD]
if not bad_days.empty:
    days_txt = ", ".join(f"{d.date()} (AQI {a:.0f})" for d, a in zip(bad_days["date"], bad_days["aqi"]))
    st.error(f"⚠️ **Air quality alert:** unhealthy AQI expected on {days_txt}. "
             f"Sensitive groups should limit outdoor activity.")
else:
    st.success("✅ No hazardous AQI expected in the next 3 days.")

# ── Top row: current + 3-day forecast cards ────────────────────────────
st.subheader("Current & forecast")
cols = st.columns(4)
with cols[0]:
    st.markdown(aqi_badge(float(latest["aqi"]), f"Latest — {pd.to_datetime(latest['date']).date()}"),
                unsafe_allow_html=True)
for i, (_, r) in enumerate(fc.iterrows(), start=1):
    with cols[i]:
        st.markdown(aqi_badge(r["aqi"], f"Day +{r['day_offset']} — {r['date'].date()}"),
                    unsafe_allow_html=True)

st.pyplot(forecast_chart(combined))

# ── EDA ────────────────────────────────────────────────────────────────
st.subheader("📊 Exploratory data analysis")
c1, c2, c3 = st.columns(3)
c1.metric("Average AQI (history)", f"{feat_df['aqi'].mean():.0f}")
c2.metric("Worst day", f"{feat_df['aqi'].max():.0f}")
c3.metric("Best day", f"{feat_df['aqi'].min():.0f}")

e1, e2 = st.columns(2)
with e1:
    st.markdown("**AQI by month (seasonal pattern)**")
    monthly = feat_df.groupby(feat_df["date"].dt.month)["aqi"].mean()
    monthly.index = [pd.Timestamp(2020, m, 1).strftime("%b") for m in monthly.index]
    st.bar_chart(monthly)
with e2:
    st.markdown("**AQI distribution**")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(feat_df["aqi"], bins=25, color="#3b82f6", alpha=0.85)
    ax.set_xlabel("AQI")
    ax.set_ylabel("Days")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    st.pyplot(fig)

# ── Explainability ─────────────────────────────────────────────────────
st.subheader("🔎 What drives the forecast?")
try:
    imp, method = _importance(cache_key)
    st.caption(f"Method: {method}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp_sorted = imp.sort_values()
    ax.barh(imp_sorted.index, imp_sorted.values, color="#6366f1")
    ax.set_xlabel("Importance")
    ax.grid(alpha=0.15, axis="x")
    fig.tight_layout()
    st.pyplot(fig)
except Exception as exc:  # noqa: BLE001
    st.warning(f"Could not compute feature importance: {exc}")

# ── Model performance ──────────────────────────────────────────────────
st.subheader("🧠 Model performance")
m = reg["metrics"]["mean"]
p1, p2, p3, p4 = st.columns(4)
p1.metric("Best model", reg["model_name"])
p2.metric("RMSE", f"{m['rmse']:.2f}")
p3.metric("MAE", f"{m['mae']:.2f}")
p4.metric("R²", f"{m['r2']:.3f}")

with st.expander("Per-horizon metrics & model comparison"):
    per = pd.DataFrame(reg["metrics"]).T[["rmse", "mae", "r2"]].round(3)
    st.markdown("**Best model, per forecast horizon:**")
    st.dataframe(per, use_container_width=True)
    st.markdown("**All models (mean across horizons):**")
    comp = pd.DataFrame({name: mm["mean"] for name, mm in reg["all_metrics"].items()}).T.round(3)
    st.dataframe(comp, use_container_width=True)

st.caption(f"Model trained at {reg['trained_at']} · "
           f"{reg['n_train']} train / {reg['n_test']} test rows.")
