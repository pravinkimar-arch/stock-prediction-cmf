"""
Step 3: Filings Modality — Text Sentiment Features
===================================================
Thesis Objective 3: Run frozen FinBERT sentiment inference on corporate
filings, pool to daily features, and compute text memory signals.

This script:
  1. Loads raw filings with first-seen session alignment (from Step 1)
  2. Runs frozen FinBERT on filing text (with per-document caching)
  3. Pools per-document sentiment to daily features per symbol
  4. Computes text memory features (filing count, recency, time-since-last)
  5. Saves daily text features to cache/daily_text_features.parquet

Inputs:
  - cache/completed_symbols.json  (from Step 1)
  - filings_data/                 (raw filings CSVs)
  - cache/sentiment_cache.parquet (optional, speeds up re-runs)

Outputs:
  - cache/filings_processed.parquet     (filings with sentiment scores)
  - cache/sentiment_cache.parquet       (per-document FinBERT cache)
  - cache/daily_text_features.parquet   (daily text features per symbol)

Usage:
    python pipeline/step3_filings_modality.py
    python pipeline/step3_filings_modality.py --dry-run   # no files written
"""

import argparse
import json, logging, os, sys, time
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, setup_logging
from src.data.ohlcv_loader import load_universe_ohlcv
from src.data.filings_loader import load_filings
from src.features.text_sentiment import compute_daily_text_features, TEXT_FEATURE_COLS


def main():
    parser = argparse.ArgumentParser(description="Step 3: Filings Modality")
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
    filings_map = cfg.get("filings_columns", {})

    # ── 1. Load universe ──
    print("=" * 80)
    print("STEP 3: Filings Modality — Text Sentiment Features")
    if dry_run:
        print("  ** DRY RUN — no files will be written **")
    print("=" * 80)

    symbols_path = cache_dir / "completed_symbols.json"
    if not symbols_path.exists():
        print(f"ERROR: {symbols_path} not found. Run pipeline/step1_data_assembly.py first.")
        sys.exit(1)

    symbols = json.loads(symbols_path.read_text())
    print(f"\nUniverse: {len(symbols)} symbols")

    # ── 2. Load daily dates (needed for text feature alignment) ──
    print("\n[1/3] Loading OHLCV for daily date grid...")
    daily = load_universe_ohlcv(
        data_cfg["price_data_dir"], symbols, ohlcv_map,
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
        dry_run=dry_run,
    )
    daily_dates = daily[["symbol", "date"]].drop_duplicates()
    print(f"  {len(daily_dates)} symbol-date pairs")

    # ── 3. Load filings with first-seen alignment ──
    print("\n[2/3] Loading filings with first-seen session alignment...")
    nse = cfg.get("nse_session", {})
    filings = load_filings(
        data_cfg["filings_data_dir"], symbols, filings_map,
        market_open=nse.get("market_open", "09:15"),
        market_close=nse.get("market_close", "15:30"),
        dry_run=dry_run,
    )

    if filings.empty:
        print("  WARNING: No filings loaded. Exiting.")
        return

    print(f"  {len(filings)} filings for {filings['symbol'].nunique()} symbols")
    print(f"  Date range: {filings['assigned_date'].min().date()} to {filings['assigned_date'].max().date()}")

    # ── 4. Run FinBERT + compute daily text features ──
    print("\n[3/3] Running FinBERT sentiment + daily pooling + memory features...")
    text_cfg = cfg["features"]["text"]
    text_cfg["cache_dir"] = str(cache_dir)

    text_feats = compute_daily_text_features(filings, daily_dates, text_cfg,
                                              dry_run=dry_run)

    # ── Summary ──
    filing_days = text_feats[text_feats["doc_count"] > 0]
    total_days = len(text_feats)

    print(f"\n{'='*80}")
    print("TEXT FEATURES SUMMARY")
    print(f"{'='*80}")
    print(f"  Symbols:        {text_feats['symbol'].nunique()}")
    print(f"  Total days:     {total_days}")
    print(f"  Filing days:    {len(filing_days)} ({len(filing_days)/total_days*100:.1f}%)")
    print(f"  Feature cols:   {len(TEXT_FEATURE_COLS)}")
    print(f"  Output:         {cache_dir / 'daily_text_features.parquet'}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
