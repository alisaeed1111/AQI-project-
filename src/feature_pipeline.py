"""
feature_pipeline.py — Fetch raw data → engineer features → store them.

This is the "feature pipeline" from the project spec. In production it runs
automatically every hour (see .github/workflows/feature-pipeline.yml).

Running it over a long window (`days = BACKFILL_DAYS`) also performs the
HISTORICAL BACKFILL: it (re)builds the full training table in one shot.
"""
from __future__ import annotations

from config import BACKFILL_DAYS
from data_fetch import get_raw_daily
from features import build_features, feature_columns
from feature_store import save_features


def run(days: int | None = None, verbose: bool = True):
    """Run the feature pipeline and write results to the feature store."""
    days = days or BACKFILL_DAYS
    raw = get_raw_daily(days)
    feats = build_features(raw)
    path = save_features(feats)
    if verbose:
        n_feats = len(feature_columns(feats))
        print(
            f"[feature_pipeline] Stored {len(feats)} rows × {n_feats} features "
            f"({feats['date'].min().date()} → {feats['date'].max().date()})"
        )
        print(f"[feature_pipeline] Feature store: {path}")
    return feats


if __name__ == "__main__":
    run()
