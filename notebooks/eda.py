"""
eda.py — Standalone Exploratory Data Analysis for the AQI dataset.

Run it with:   python notebooks/eda.py

It builds the feature set if needed, prints summary statistics, and saves a
few charts into the `assets/` folder so you can drop them into a report.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import feature_pipeline
from config import CITY_NAME
from feature_store import features_exist, load_features

ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)


def main():
    if not features_exist():
        print("[eda] No features yet — building them first…")
        feature_pipeline.run(verbose=False)

    df = load_features().sort_values("date").reset_index(drop=True)
    print(f"\n===== EDA for {CITY_NAME} =====")
    print(f"Rows: {len(df)}   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print("\nAQI summary:")
    print(df["aqi"].describe().round(1).to_string())

    # 1) AQI over time
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["date"], df["aqi"], color="#2563eb", lw=1.3)
    ax.set_title(f"{CITY_NAME} — AQI over time")
    ax.set_ylabel("AQI"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "eda_aqi_timeseries.png"), dpi=110)
    plt.close(fig)

    # 2) Average AQI by month (seasonal pattern)
    monthly = df.groupby(df["date"].dt.month)["aqi"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([pd.Timestamp(2020, m, 1).strftime("%b") for m in monthly.index],
           monthly.values, color="#f59e0b")
    ax.set_title(f"{CITY_NAME} — average AQI by month")
    ax.set_ylabel("Mean AQI"); ax.grid(alpha=0.2, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "eda_monthly.png"), dpi=110)
    plt.close(fig)

    # 3) AQI distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["aqi"], bins=30, color="#10b981", alpha=0.85)
    ax.set_title(f"{CITY_NAME} — AQI distribution")
    ax.set_xlabel("AQI"); ax.set_ylabel("Days"); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "eda_distribution.png"), dpi=110)
    plt.close(fig)

    # 4) What correlates with AQI?
    candidates = [c for c in ["pm2_5", "pm10", "co", "no2", "so2", "o3",
                              "temp", "humidity", "pressure", "wind_speed"] if c in df.columns]
    corr = df[candidates + ["aqi"]].corr()["aqi"].drop("aqi").sort_values()
    print("\nCorrelation of each feature with AQI:")
    print(corr.round(2).to_string())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(corr.index, corr.values, color="#6366f1")
    ax.set_title(f"{CITY_NAME} — correlation with AQI")
    ax.axvline(0, color="#333", lw=0.8); ax.grid(alpha=0.2, axis="x")
    fig.tight_layout(); fig.savefig(os.path.join(ASSETS, "eda_correlation.png"), dpi=110)
    plt.close(fig)

    print(f"\n[eda] Charts saved to: {ASSETS}")


if __name__ == "__main__":
    main()
