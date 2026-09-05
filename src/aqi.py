"""
aqi.py — Convert pollutant concentrations into the US EPA Air Quality Index.

The AQI is a 0–500 scale most people recognise ("AQI 180 = Unhealthy").
It is computed from a pollutant concentration using the EPA's pi–ecewise
linear formula:

    AQI = (I_high - I_low) / (C_high - C_low) * (C - C_low) + I_low

We compute a sub-index for PM2.5 and PM10 (the two pollutants that usually
drive urban AQI) and take the worse (max) of the two as the overall AQI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (C_low, C_high, I_low, I_high) — concentrations in µg/m³ (24-hour average).
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]


def _conc_to_aqi(conc, breakpoints) -> float:
    """Convert a single concentration to its AQI sub-index."""
    if conc is None:
        return np.nan
    try:
        conc = float(conc)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(conc) or conc < 0:
        return np.nan
    for c_low, c_high, i_low, i_high in breakpoints:
        if conc <= c_high:
            return float(round((i_high - i_low) / (c_high - c_low) * (conc - c_low) + i_low))
    # Above the top of the table -> clamp to the maximum AQI.
    return 500.0


def pm25_to_aqi(conc) -> float:
    return _conc_to_aqi(conc, PM25_BREAKPOINTS)


def pm10_to_aqi(conc) -> float:
    return _conc_to_aqi(conc, PM10_BREAKPOINTS)


def compute_aqi(pm25, pm10=None) -> float:
    """Overall AQI = the worse (max) of the PM2.5 and PM10 sub-indices."""
    a = pm25_to_aqi(pm25)
    if pm10 is None:
        return a
    b = pm10_to_aqi(pm10)
    vals = [v for v in (a, b) if v is not None and not np.isnan(v)]
    return float(max(vals)) if vals else np.nan


def compute_aqi_series(pm25, pm10) -> pd.Series:
    """Vectorised AQI over pandas Series / array-likes."""
    pm25 = pd.Series(list(pm25)).reset_index(drop=True)
    pm10 = pd.Series(list(pm10)).reset_index(drop=True)
    return pd.Series([compute_aqi(a, b) for a, b in zip(pm25, pm10)], dtype="float64")


if __name__ == "__main__":
    # Sanity check against known reference points.
    checks = [
        (9.0, None, "Good"),          # ~ AQI 38
        (35.4, None, "Moderate"),     # AQI 100
        (55.5, None, "Unhealthy"),    # AQI 151
        (250.5, None, "Very Unhealthy"),  # AQI 301
    ]
    for pm25, pm10, note in checks:
        print(f"PM2.5={pm25:>6} -> AQI {compute_aqi(pm25, pm10):>5.0f}  ({note})")
