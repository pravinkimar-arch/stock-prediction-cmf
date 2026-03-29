"""
Step 2: Time-Series Feature Engineering
========================================
Thesis Objective 2: Compute numeric OHLCV indicators and Chart2Tokens
event-based features for every symbol in the universe.

This script:
  1. Loads daily OHLCV from cache/completed_symbols.json universe
  2. Computes numeric features (log return, ATR, volatility, MA ratio, volume z-score)
  3. Detects Chart2Token events (breakout, gap, volume burst, round touch, engulfing)
  4. Computes token summary features (count_W, recency_W, time_since_last)
  5. Creates next-day direction labels
  6. Saves feature matrix to cache/features_all.parquet

Inputs:
  - cache/completed_symbols.json  (from Step 1)
  - price_data/*.csv              (minute-level OHLCV)

Outputs:
  - cache/features_all.parquet    (full feature matrix with labels)

Usage:
    python pipeline/step2_feature_engineering.py
"""

import argparse, json, logging, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, setup_logging
from src.data.ohlcv_loader import load_universe_ohlcv
from src.features.numeric import compute_numeric_features, NUMERIC_FEATURE_COLS
from src.features.chart2tokens import (
    compute_chart2tokens, get_token_feature_cols, TOKEN_NAMES,
)
from src.features.text_sentiment import TEXT_FEATURE_COLS
from pipeline._common import create_labels


def main():
    parser = argparse.ArgumentParser(description="Step 2: Feature Engineering")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip writing any files")
    args = parser.parse_args()
    dry_run = args.dry_run

    cfg = load_config("configs/default.yaml")
    setup_logging("INFO")
    set_seed(42)
    t0 = time.time()

    data_cfg = cfg["data"]
    cache_dir = Path(data_cfg["cache_dir"])
    ohlcv_map = cfg.get("ohlcv_columns", {})

    # ── 1. Load universe from Step 1 ──
    print("=" * 80)
    print("STEP 2: Time-Series Feature Engineering")
    if dry_run:
        print("  ** DRY RUN — no files will be written **")
    print("=" * 80)

    symbols_path = cache_dir / "completed_symbols.json"
    if not symbols_path.exists():
        print(f"ERROR: {symbols_path} not found. Run pipeline/step1_data_assembly.py first.")
        sys.exit(1)

    symbols = json.loads(symbols_path.read_text())
    print(f"\nUniverse: {len(symbols)} symbols")

    # ── 2. Load OHLCV ──
    print("\n[1/4] Loading OHLCV data...")
    daily = load_universe_ohlcv(
        data_cfg["price_data_dir"], symbols, ohlcv_map,
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
    )
    print(f"  {len(daily)} daily rows, {daily['symbol'].nunique()} symbols")
    print(f"  Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")

    # ── 3. Compute numeric features ──
    print("\n[2/4] Computing numeric features...")
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])
    print(f"  Numeric features: {NUMERIC_FEATURE_COLS}")

    # ── 4. Compute Chart2Tokens ──
    print("\n[3/4] Computing Chart2Tokens...")
    daily = compute_chart2tokens(daily, cfg["features"])
    W = cfg["features"]["token_summary"]["lookback_W"]
    token_cols = get_token_feature_cols(W)
    print(f"  Token events: {TOKEN_NAMES}")
    print(f"  Token summary features ({len(token_cols)}): count_{W}, recency_{W}, tsl per token")

    # ── 5. Create labels ──
    print("\n[4/4] Creating next-day direction labels...")
    daily = create_labels(daily)

    # Merge cached text features if available (so features_all.parquet is complete)
    text_cache = cache_dir / "daily_text_features.parquet"
    if text_cache.exists():
        print("\n  Merging cached text features from Step 1...")
        text_feats = pd.read_parquet(text_cache)
        text_feats = text_feats[text_feats["symbol"].isin(symbols)]
        text_feats["date"] = pd.to_datetime(text_feats["date"])
        daily = daily.merge(text_feats, on=["symbol", "date"], how="left")

    # Fill NaN text features
    for col in TEXT_FEATURE_COLS:
        if col in daily.columns:
            daily[col] = daily[col].fillna(0)
        else:
            daily[col] = 0

    # Drop rows with NaN in core features or labels
    daily = daily.dropna(subset=["label"] + NUMERIC_FEATURE_COLS).reset_index(drop=True)

    # ── Save ──
    out_path = cache_dir / "features_all.parquet"
    if not dry_run:
        daily.to_parquet(out_path, index=False)
    else:
        print(f"  [DRY RUN] Would write: {out_path}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("FEATURE ENGINEERING SUMMARY")
    print(f"{'='*80}")
    print(f"  Symbols:          {daily['symbol'].nunique()}")
    print(f"  Rows:             {len(daily)}")
    print(f"  Date range:       {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"  Numeric features: {len(NUMERIC_FEATURE_COLS)}")
    print(f"  Token features:   {len(token_cols)}")
    print(f"  Text features:    {len(TEXT_FEATURE_COLS)}")
    print(f"  Label dist:       {daily['label'].value_counts().to_dict()}")
    print(f"  Output:           {out_path}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
