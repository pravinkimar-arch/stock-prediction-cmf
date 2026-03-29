"""
Step 1: Data & Label Assembly
=============================
Thesis Objective 1: Pre-process/sanitize the dataset fetched from Kaggle and NSE.
Normalize time zones, apply first-seen alignment based on next session.

This script:
  1. Loads daily OHLCV from minute-level price data (price_data/)
  2. Loads corporate filings with first-seen session alignment (filings_data/)
  3. Runs frozen FinBERT sentiment inference on filing text
  4. Caches all intermediate artifacts for downstream steps
  5. Creates next-day direction labels: label = 1 if log(C_{t+1}/C_t) > 0

Outputs (in cache/):
  - completed_symbols.json    — symbols with both price + filing data
  - filings_raw.parquet       — raw loaded filings
  - filings_processed.parquet — filings with sentiment scores
  - sentiment_cache.parquet   — per-document FinBERT scores
  - daily_text_features.parquet — daily text features per symbol

Usage:
    python pipeline/step1_data_assembly.py
"""

import argparse, json, logging, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, setup_logging
from src.data.ohlcv_loader import load_universe_ohlcv
from src.data.filings_loader import load_filings
from src.features.text_sentiment import compute_daily_text_features, TEXT_FEATURE_COLS


def main():
    parser = argparse.ArgumentParser(description="Step 1: Data & Label Assembly")
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
    if not dry_run:
        cache_dir.mkdir(parents=True, exist_ok=True)
    ohlcv_map = cfg.get("ohlcv_columns", {})
    filings_map = cfg.get("filings_columns", {})

    # ── 1. Discover universe: symbols with both price + filings ──
    print("=" * 80)
    print("STEP 1: Data & Label Assembly")
    if dry_run:
        print("  ** DRY RUN — no files will be written **")
    print("=" * 80)

    price_dir = Path(data_cfg["price_data_dir"])
    filings_dir = Path(data_cfg["filings_data_dir"])

    price_symbols = {f.stem.replace("_minute", "").replace("_daily", "")
                     for f in price_dir.glob("*.csv")}
    filing_symbols = {d.name for f in filings_dir.iterdir() if f.is_dir()
                      for d in f.glob("*.csv")
                      for d in [Path(d.stem.rsplit("_", 1)[0])]}
    # Simpler: just use configured symbols filtered by price data
    configured = set(cfg["universe"]["symbols"])
    available = configured & price_symbols
    cached_symbols = sorted(available)

    print(f"\nConfigured symbols: {len(configured)}")
    print(f"With price data:   {len(price_symbols)}")
    print(f"Available:         {len(available)}")

    # ── 2. Load OHLCV ──
    print("\nLoading OHLCV data...")
    daily = load_universe_ohlcv(
        data_cfg["price_data_dir"], cached_symbols, ohlcv_map,
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
    )
    print(f"  {len(daily)} daily rows for {daily['symbol'].nunique()} symbols")
    print(f"  Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")

    # ── 3. Load filings with first-seen alignment ──
    print("\nLoading filings with first-seen session alignment...")
    nse = cfg.get("nse_session", {})
    filings = load_filings(
        data_cfg["filings_data_dir"], cached_symbols, filings_map,
        market_open=nse.get("market_open", "09:15"),
        market_close=nse.get("market_close", "15:30"),
    )
    if not filings.empty:
        print(f"  {len(filings)} filings for {filings['symbol'].nunique()} symbols")
        print(f"  Date range: {filings['assigned_date'].min().date()} to {filings['assigned_date'].max().date()}")
        print(f"  Categories: {filings['filing_category'].nunique()} types")
        if not dry_run:
            filings.to_parquet(cache_dir / "filings_raw.parquet", index=False)
        else:
            print(f"  [DRY RUN] Would write: {cache_dir / 'filings_raw.parquet'}")
    else:
        print("  WARNING: No filings loaded")

    # ── 4. Run FinBERT sentiment + compute daily text features ──
    print("\nComputing text features (FinBERT sentiment + daily pooling)...")
    daily_dates = daily[["symbol", "date"]].drop_duplicates()
    text_cfg = cfg["features"]["text"]
    text_cfg["cache_dir"] = str(cache_dir)
    text_feats = compute_daily_text_features(filings, daily_dates, text_cfg,
                                              dry_run=dry_run)
    print(f"  Text features: {text_feats.shape}")

    # Filter to symbols that have both price and text data
    symbols_with_text = set(text_feats[text_feats["doc_count"] > 0]["symbol"].unique())
    cached_symbols = sorted(set(cached_symbols) & symbols_with_text)
    print(f"\nFinal universe (price + filings): {len(cached_symbols)} symbols")

    # ── 5. Save symbol list ──
    if not dry_run:
        (cache_dir / "completed_symbols.json").write_text(json.dumps(cached_symbols, indent=2))
        print(f"Saved: {cache_dir}/completed_symbols.json")
    else:
        print(f"  [DRY RUN] Would write: {cache_dir}/completed_symbols.json")

    # ── 6. Summary statistics ──
    daily_sub = daily[daily["symbol"].isin(cached_symbols)]
    text_sub = text_feats[text_feats["symbol"].isin(cached_symbols)]
    filing_days = text_sub[text_sub["doc_count"] > 0]

    print(f"\n{'='*80}")
    print(f"DATA SUMMARY")
    print(f"{'='*80}")
    print(f"  Symbols:      {len(cached_symbols)}")
    print(f"  Daily rows:   {len(daily_sub)}")
    print(f"  Date range:   {daily_sub['date'].min().date()} to {daily_sub['date'].max().date()}")
    print(f"  Filing days:  {len(filing_days)} ({len(filing_days)/len(text_sub)*100:.1f}%)")
    print(f"  Total filings:{len(filings[filings['symbol'].isin(cached_symbols)])}")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
