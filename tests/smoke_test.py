"""
smoke_test.py — Fast health-check of the core AQI logic.

Runs the parts that don't need scikit-learn / TensorFlow, so it works in any
environment. Great for catching problems quickly.

    python tests/smoke_test.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

from aqi import compute_aqi
from data_fetch import get_raw_daily
from features import TARGET_COLS, build_features, feature_columns
from train import make_xy, time_split

PASS, FAIL = "✅ PASS", "❌ FAIL"
_failures = 0


def check(name, condition):
    global _failures
    print(f"{PASS if condition else FAIL}  {name}")
    if not condition:
        _failures += 1


def run():
    # AQI computation against EPA reference points.
    check("AQI(PM2.5=35.4) == 100", compute_aqi(35.4) == 100)
    check("AQI(PM2.5=55.5) == 151", compute_aqi(55.5) == 151)
    check("AQI clamps at 500", compute_aqi(9999) == 500)

    # Raw data.
    raw = get_raw_daily(90)
    check("raw has 90 rows", len(raw) == 90)
    check("raw has pm2_5 & aqi columns", {"pm2_5", "aqi"}.issubset(raw.columns))
    check("aqi within 0..500", raw["aqi"].between(0, 500).all())

    # Features.
    feats = build_features(raw)
    fc = feature_columns(feats)
    check("feature rows produced", len(feats) > 0)
    check("no NaN in feature columns", not feats[fc].isna().any().any())
    check("3 target columns present", all(t in feats.columns for t in TARGET_COLS))

    # Training matrix.
    X, Y, cols = make_xy(feats)
    check("Y has 3 columns", Y.shape[1] == 3)
    check("X and Y row counts match", len(X) == len(Y))
    tr, te = time_split(len(X))
    check("split is chronological", tr.max() < te.min())

    print()
    if _failures:
        print(f"{_failures} check(s) failed.")
        sys.exit(1)
    print("All smoke tests passed. 🎉")


if __name__ == "__main__":
    run()
