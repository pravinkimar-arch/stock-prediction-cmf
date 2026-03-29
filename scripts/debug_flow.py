"""Step-by-step walkthrough of the Category-Aware Feature Engineering pipeline.

Run each step independently:
    python scripts/debug_flow.py --step 1
    python scripts/debug_flow.py --step 2
    ...up to step 11

Each step prints the data at that stage so you can see exactly what happens.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def step1_raw_filings():
    """STEP 1: Load raw filing CSVs and see what they look like before any processing."""
    from src.utils.config import load_config
    cfg = load_config("configs/default.yaml")

    filings_dir = Path(cfg["data"]["filings_data_dir"])
    print("=" * 80)
    print("STEP 1: RAW FILING DATA (before any processing)")
    print(f"Source: {filings_dir}")
    print(f"Code:   src/data/filings_loader.py:210-320")
    print("=" * 80)

    # Find one CSV to show raw structure
    date_dirs = sorted([d for d in filings_dir.iterdir() if d.is_dir()])
    print(f"\nTotal date folders: {len(date_dirs)}")
    print(f"First 5: {[d.name for d in date_dirs[:5]]}")

    # Read first CSV with data
    for date_dir in date_dirs:
        csvs = list(date_dir.glob("*.csv"))
        if csvs:
            df = pd.read_csv(csvs[0], encoding="utf-8", on_bad_lines="skip")
            if not df.empty:
                print(f"\nSample file: {csvs[0]}")
                print(f"Columns: {list(df.columns)}")
                print(f"\nFirst row:")
                for col in df.columns:
                    val = str(df.iloc[0][col])
                    if len(val) > 200:
                        val = val[:200] + "..."
                    print(f"  {col:20s}: {val}")
                break

    print("\n>> This is the RAW data. Next step: session assignment + classification.")


def step2_session_assignment():
    """STEP 2: Show how filing timestamps get assigned to trading sessions."""
    from src.data.nse_calendar import assign_filing_to_session
    from datetime import time as dtime
    import pytz

    IST = pytz.timezone("Asia/Kolkata")
    mo = dtime(9, 15)
    mc = dtime(15, 30)

    print("=" * 80)
    print("STEP 2: SESSION ASSIGNMENT")
    print("Code: src/data/nse_calendar.py → assign_filing_to_session()")
    print("Code: src/data/filings_loader.py:285")
    print("=" * 80)

    # Test cases
    test_cases = [
        ("During market hours (trading day)",    "2024-01-15 11:30:00"),
        ("After market close (trading day)",     "2024-01-15 16:45:00"),
        ("Before market open (trading day)",     "2024-01-15 08:00:00"),
        ("Saturday",                             "2024-01-13 14:00:00"),
        ("Sunday evening",                       "2024-01-14 20:00:00"),
        ("Holiday (Republic Day)",               "2024-01-26 10:00:00"),
    ]

    print(f"\nMarket hours: {mo} - {mc} IST")
    print(f"{'Scenario':45s} {'Raw Timestamp':25s} {'Assigned Date':15s}")
    print("-" * 85)

    for label, ts_str in test_cases:
        ts = IST.localize(pd.Timestamp(ts_str))
        assigned = assign_filing_to_session(ts, mo, mc)
        print(f"{label:45s} {ts_str:25s} {str(assigned):15s}")

    print("\n>> Filings after 15:30 go to NEXT trading day (prevents data leakage).")


def step3_classification():
    """STEP 3: Show the two-stage filing classification."""
    from src.data.filings_loader import _classify_base_type, _scan_body_direction, classify_filing

    print("=" * 80)
    print("STEP 3: FILING CLASSIFICATION (Two-Stage)")
    print("Code: src/data/filings_loader.py:27-186")
    print("=" * 80)

    test_cases = [
        (
            "Financial Results for Quarter Ended Dec 2023",
            "Revenue grew 18% year-on-year. Net profit after tax increased 12% to Rs 17265 crore. "
            "Strong performance driven by robust growth in digital services. Operating margins improved 150 bps.",
        ),
        (
            "Financial Results for Quarter Ended Sep 2023",
            "Revenue declined 5% year-on-year. Net profit fell 8% due to challenging market conditions. "
            "Margins compressed by 200 bps. Weak performance across segments.",
        ),
        (
            "Outcome of Board Meeting",
            "The Board approved the quarterly results showing strong growth. Revenue increased 20%.",
        ),
        (
            "Declaration of Dividend",
            "The Board has declared an interim dividend of Rs 10 per share.",
        ),
        (
            "Acquisition of Subsidiary",
            "The company has entered into an agreement for acquisition of 51% stake in XYZ Ltd.",
        ),
        (
            "Loss of Share Certificate",
            "Intimation of loss of share certificate by a shareholder.",
        ),
    ]

    for subject, body in test_cases:
        base_type = _classify_base_type(subject)
        direction = _scan_body_direction(body)
        final_cat = classify_filing(subject, body)

        print(f"\nSubject: \"{subject}\"")
        print(f"  Stage 1 (regex on subject) → base_type = \"{base_type}\"")
        print(f"  Stage 2 (body scan)        → direction = \"{direction}\"")
        print(f"  Final category             → \"{final_cat}\"")


def step4_loaded_filings():
    """STEP 4: Show the processed filings with categories from cache."""
    cache_dir = Path("cache")
    filings = pd.read_parquet(cache_dir / "filings_processed.parquet")

    print("=" * 80)
    print("STEP 4: PROCESSED FILINGS (with categories, from cache)")
    print(f"Code:   src/data/filings_loader.py:210-320")
    print(f"Source: cache/filings_processed.parquet")
    print("=" * 80)

    print(f"\nTotal filings: {len(filings)}")
    print(f"Symbols: {filings['symbol'].nunique()}")
    print(f"Date range: {filings['assigned_date'].min()} to {filings['assigned_date'].max()}")

    print(f"\nColumns: {list(filings.columns)}")

    print(f"\n--- Category distribution ---")
    print(filings["filing_category"].value_counts().to_string())

    print(f"\n--- Sample rows (first 5) ---")
    sample_cols = ["symbol", "assigned_date", "filing_category", "p_pos", "p_neg", "polarity"]
    available = [c for c in sample_cols if c in filings.columns]
    print(filings[available].head(5).to_string(index=False))

    # Pick one symbol to show detail
    sym = "RELIANCE" if "RELIANCE" in filings["symbol"].values else filings["symbol"].iloc[0]
    sym_df = filings[filings["symbol"] == sym].sort_values("assigned_date")
    print(f"\n--- {sym}: first 10 filings ---")
    print(sym_df[available].head(10).to_string(index=False))


def step5_finbert_scores():
    """STEP 5: Show FinBERT sentiment scores on the processed filings."""
    cache_dir = Path("cache")
    filings = pd.read_parquet(cache_dir / "filings_processed.parquet")

    print("=" * 80)
    print("STEP 5: FinBERT SENTIMENT SCORES")
    print("Code: src/features/text_sentiment.py:101-199 (infer_sentiment)")
    print("       → Chunk text → FinBERT forward pass → avg logits → softmax → polarity")
    print("=" * 80)

    print(f"\nPolarity = p_pos - p_neg  (range: -1 to +1)")
    print(f"\n--- Score distributions ---")
    for col in ["p_pos", "p_neg", "polarity"]:
        if col in filings.columns:
            print(f"  {col:12s}: mean={filings[col].mean():.4f}, "
                  f"std={filings[col].std():.4f}, "
                  f"min={filings[col].min():.4f}, "
                  f"max={filings[col].max():.4f}")

    print(f"\n--- Polarity by category ---")
    cat_stats = (
        filings.groupby("filing_category")["polarity"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    print(f"{'Category':35s} {'Mean Pol':>10s} {'Std':>8s} {'Count':>8s}")
    print("-" * 65)
    for cat, row in cat_stats.iterrows():
        print(f"{cat:35s} {row['mean']:10.4f} {row['std']:8.4f} {row['count']:8.0f}")

    print(f"\n--- Sample: Most positive filings ---")
    top5 = filings.nlargest(5, "polarity")
    for _, r in top5.iterrows():
        print(f"  {r['symbol']:12s} {str(r['assigned_date'])[:10]}  "
              f"cat={r['filing_category']:30s}  pol={r['polarity']:.4f}  "
              f"p_pos={r['p_pos']:.4f}  p_neg={r['p_neg']:.4f}")

    print(f"\n--- Sample: Most negative filings ---")
    bot5 = filings.nsmallest(5, "polarity")
    for _, r in bot5.iterrows():
        print(f"  {r['symbol']:12s} {str(r['assigned_date'])[:10]}  "
              f"cat={r['filing_category']:30s}  pol={r['polarity']:.4f}  "
              f"p_pos={r['p_pos']:.4f}  p_neg={r['p_neg']:.4f}")


def step6_first_groupby():
    """STEP 6: First aggregation — per (symbol, date, filing_category)."""
    cache_dir = Path("cache")
    filings = pd.read_parquet(cache_dir / "filings_processed.parquet")
    filings["date"] = pd.to_datetime(filings["assigned_date"])

    HIGH_IMPACT_GROUPS = {
        "earnings": ["earnings_positive", "earnings_negative", "earnings_neutral"],
        "board_outcome": ["board_outcome_positive", "board_outcome_negative", "board_outcome_neutral"],
        "dividend": ["dividend_announced"],
        "m_and_a": ["m_and_a_activity"],
    }
    all_hi_cats = [c for cats in HIGH_IMPACT_GROUPS.values() for c in cats]

    print("=" * 80)
    print("STEP 6: FIRST GROUPBY — per (symbol, date, filing_category)")
    print("Code: run_category_cross_modal.py:116-121")
    print("=" * 80)

    # Filter to high-impact categories only
    hi_filings = filings[filings["filing_category"].isin(all_hi_cats)]
    print(f"\nHigh-impact filings: {len(hi_filings)} out of {len(filings)} total")

    cat_agg = (
        hi_filings.groupby(["symbol", "date", "filing_category"])
        .agg(p_pos=("p_pos", "mean"), p_neg=("p_neg", "mean"),
             polarity=("polarity", "mean"), count=("polarity", "count"))
        .reset_index()
    )

    print(f"After groupby: {len(cat_agg)} rows")
    print(f"\n--- Days with MULTIPLE filings in same category (where averaging happens) ---")
    multi = cat_agg[cat_agg["count"] > 1].sort_values("count", ascending=False)
    if len(multi) > 0:
        print(f"Found {len(multi)} such cases. Top 10:")
        print(multi.head(10).to_string(index=False))
    else:
        print("None found — each (symbol, date, category) has exactly 1 filing.")

    # Show a sample day with multiple categories
    day_counts = cat_agg.groupby(["symbol", "date"]).size().reset_index(name="n_categories")
    multi_cat_days = day_counts[day_counts["n_categories"] > 1]
    if len(multi_cat_days) > 0:
        sample = multi_cat_days.iloc[0]
        sym, dt = sample["symbol"], sample["date"]
        print(f"\n--- Sample day with multiple categories: {sym} on {dt.date()} ---")
        day_data = cat_agg[(cat_agg["symbol"] == sym) & (cat_agg["date"] == dt)]
        print(day_data.to_string(index=False))
        print("\n>> These will be AVERAGED together in Step 7 if they're in the same group.")


def step7_second_groupby():
    """STEP 7: Second aggregation — per group per (symbol, date)."""
    cache_dir = Path("cache")
    filings = pd.read_parquet(cache_dir / "filings_processed.parquet")
    filings["date"] = pd.to_datetime(filings["assigned_date"])

    HIGH_IMPACT_GROUPS = {
        "earnings": ["earnings_positive", "earnings_negative", "earnings_neutral"],
        "board_outcome": ["board_outcome_positive", "board_outcome_negative", "board_outcome_neutral"],
        "dividend": ["dividend_announced"],
        "m_and_a": ["m_and_a_activity"],
    }

    print("=" * 80)
    print("STEP 7: SECOND GROUPBY — per group per (symbol, date)")
    print("Code: run_category_cross_modal.py:127-146")
    print("=" * 80)

    cat_agg = (
        filings.groupby(["symbol", "date", "filing_category"])
        .agg(p_pos=("p_pos", "mean"), p_neg=("p_neg", "mean"),
             polarity=("polarity", "mean"), count=("polarity", "count"))
        .reset_index()
    )

    for group_name, cats in HIGH_IMPACT_GROUPS.items():
        group_data = cat_agg[cat_agg["filing_category"].isin(cats)]
        group_daily = (
            group_data.groupby(["symbol", "date"])
            .agg(p_pos=("p_pos", "mean"), p_neg=("p_neg", "mean"),
                 polarity=("polarity", "mean"))
            .reset_index()
        )
        group_daily = group_daily.rename(columns={
            "p_pos": f"p_pos_{group_name}",
            "p_neg": f"p_neg_{group_name}",
            "polarity": f"polarity_{group_name}",
        })
        group_daily[f"has_{group_name}"] = 1

        print(f"\n--- {group_name.upper()} group ---")
        print(f"  Categories included: {cats}")
        print(f"  Days with this group: {len(group_daily)}")
        if len(group_daily) > 0:
            pol_col = f"polarity_{group_name}"
            print(f"  Polarity stats: mean={group_daily[pol_col].mean():.4f}, "
                  f"std={group_daily[pol_col].std():.4f}")
            print(f"  Sample rows:")
            print(f"  {group_daily.head(5).to_string(index=False)}")

    # Show a concrete example of the averaging
    print(f"\n{'='*80}")
    print("CONCRETE EXAMPLE: Finding a day where earnings subcategories get averaged")
    print("=" * 80)
    earnings_cats = HIGH_IMPACT_GROUPS["earnings"]
    earn_filings = cat_agg[cat_agg["filing_category"].isin(earnings_cats)]
    earn_day_counts = earn_filings.groupby(["symbol", "date"]).size().reset_index(name="n_subcats")
    multi = earn_day_counts[earn_day_counts["n_subcats"] > 1]
    if len(multi) > 0:
        sample = multi.iloc[0]
        sym, dt = sample["symbol"], sample["date"]
        print(f"\n{sym} on {dt.date()} has {sample['n_subcats']} earnings subcategories:")
        before = earn_filings[(earn_filings["symbol"] == sym) & (earn_filings["date"] == dt)]
        print(f"\nBEFORE averaging (individual subcategories):")
        print(before.to_string(index=False))

        after = before.agg({"p_pos": "mean", "p_neg": "mean", "polarity": "mean"})
        print(f"\nAFTER averaging (group-level):")
        print(f"  polarity_earnings = mean({', '.join(f'{v:.4f}' for v in before['polarity'])}) = {after['polarity']:.4f}")
        print(f"  p_pos_earnings    = mean({', '.join(f'{v:.4f}' for v in before['p_pos'])}) = {after['p_pos']:.4f}")
        print(f"  p_neg_earnings    = mean({', '.join(f'{v:.4f}' for v in before['p_neg'])}) = {after['p_neg']:.4f}")
        print(f"  has_earnings      = 1")
    else:
        print("No days found with multiple earnings subcategories.")


def step8_aggregate_features():
    """STEP 8: has_high_impact and polarity_high_impact."""
    cache_dir = Path("cache")
    filings = pd.read_parquet(cache_dir / "filings_processed.parquet")
    filings["date"] = pd.to_datetime(filings["assigned_date"])

    # Load daily OHLCV for the merge base
    daily = pd.read_parquet(cache_dir / "daily_ohlcv.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    HIGH_IMPACT_GROUPS = {
        "earnings": ["earnings_positive", "earnings_negative", "earnings_neutral"],
        "board_outcome": ["board_outcome_positive", "board_outcome_negative", "board_outcome_neutral"],
        "dividend": ["dividend_announced"],
        "m_and_a": ["m_and_a_activity"],
    }

    print("=" * 80)
    print("STEP 8: AGGREGATE FEATURES — has_high_impact, polarity_high_impact")
    print("Code: run_category_cross_modal.py:148-163")
    print("=" * 80)

    # Build category features (replicating the function)
    cat_agg = (
        filings.groupby(["symbol", "date", "filing_category"])
        .agg(p_pos=("p_pos", "mean"), p_neg=("p_neg", "mean"),
             polarity=("polarity", "mean"))
        .reset_index()
    )

    merged = daily.copy()
    for group_name, cats in HIGH_IMPACT_GROUPS.items():
        group_data = cat_agg[cat_agg["filing_category"].isin(cats)]
        group_daily = (
            group_data.groupby(["symbol", "date"])
            .agg(p_pos=("p_pos", "mean"), p_neg=("p_neg", "mean"),
                 polarity=("polarity", "mean"))
            .reset_index()
        )
        group_daily = group_daily.rename(columns={
            "p_pos": f"p_pos_{group_name}",
            "p_neg": f"p_neg_{group_name}",
            "polarity": f"polarity_{group_name}",
        })
        group_daily[f"has_{group_name}"] = 1
        merged = merged.merge(group_daily, on=["symbol", "date"], how="left")
        for col in [f"has_{group_name}", f"p_pos_{group_name}",
                     f"p_neg_{group_name}", f"polarity_{group_name}"]:
            merged[col] = merged[col].fillna(0).astype(np.float32)

    # Aggregate
    hi_cols = [f"has_{g}" for g in HIGH_IMPACT_GROUPS]
    merged["has_high_impact"] = (merged[hi_cols].sum(axis=1) > 0).astype(np.float32)

    pol_cols = [f"polarity_{g}" for g in HIGH_IMPACT_GROUPS]
    has_cols = [f"has_{g}" for g in HIGH_IMPACT_GROUPS]
    pol_arr = merged[pol_cols].values
    has_arr = merged[has_cols].values
    with np.errstate(invalid="ignore"):
        merged["polarity_high_impact"] = np.where(
            has_arr.sum(axis=1) > 0,
            (pol_arr * has_arr).sum(axis=1) / np.maximum(has_arr.sum(axis=1), 1),
            0.0,
        ).astype(np.float32)

    # Show results
    filing_days = merged[merged["has_high_impact"] == 1]
    non_filing = merged[merged["has_high_impact"] == 0]

    print(f"\nTotal rows: {len(merged)}")
    print(f"Filing days (has_high_impact=1): {len(filing_days)} ({len(filing_days)/len(merged)*100:.1f}%)")
    print(f"Non-filing days (has_high_impact=0): {len(non_filing)} ({len(non_filing)/len(merged)*100:.1f}%)")

    print(f"\n--- Filing days: polarity_high_impact distribution ---")
    print(f"  mean={filing_days['polarity_high_impact'].mean():.4f}")
    print(f"  std ={filing_days['polarity_high_impact'].std():.4f}")
    print(f"  min ={filing_days['polarity_high_impact'].min():.4f}")
    print(f"  max ={filing_days['polarity_high_impact'].max():.4f}")

    # Show a concrete example with multiple groups active
    multi_group = merged[merged[has_cols].sum(axis=1) > 1]
    if len(multi_group) > 0:
        sample = multi_group.iloc[0]
        print(f"\n--- CONCRETE EXAMPLE: {sample['symbol']} on {sample['date'].date()} ---")
        print(f"  (Multiple groups active on same day)\n")
        for g in HIGH_IMPACT_GROUPS:
            print(f"  has_{g:15s} = {sample[f'has_{g}']:.0f}    polarity_{g:15s} = {sample[f'polarity_{g}']:.4f}")

        active = [(g, sample[f"polarity_{g}"], sample[f"has_{g}"]) for g in HIGH_IMPACT_GROUPS if sample[f"has_{g}"] > 0]
        num = sum(pol * has for g, pol, has in active)
        den = sum(has for g, pol, has in active)
        print(f"\n  polarity_high_impact calculation:")
        parts = " + ".join(f"{pol:.4f}×{has:.0f}" for g, pol, has in active)
        print(f"    Numerator:   {parts} = {num:.4f}")
        print(f"    Denominator: {den:.0f} active groups")
        print(f"    Result:      {num:.4f} / {den:.0f} = {num/den:.4f}")
        print(f"    Stored:      {sample['polarity_high_impact']:.4f}")
    else:
        # Show single-group example
        sample = filing_days.iloc[0]
        print(f"\n--- CONCRETE EXAMPLE: {sample['symbol']} on {sample['date'].date()} ---")
        for g in HIGH_IMPACT_GROUPS:
            print(f"  has_{g:15s} = {sample[f'has_{g}']:.0f}    polarity_{g:15s} = {sample[f'polarity_{g}']:.4f}")
        print(f"  has_high_impact      = {sample['has_high_impact']:.0f}")
        print(f"  polarity_high_impact = {sample['polarity_high_impact']:.4f}")


def step9_numeric_features():
    """STEP 9: Numeric features — the other side of cross-modal interactions."""
    from src.utils.config import load_config
    from src.data.ohlcv_loader import load_universe_ohlcv
    from src.features.numeric import compute_numeric_features, NUMERIC_FEATURE_COLS

    cfg = load_config("configs/default.yaml")
    cache_dir = Path("cache")
    cached_symbols = json.loads((cache_dir / "completed_symbols.json").read_text())

    print("=" * 80)
    print("STEP 9: NUMERIC FEATURES (the other modality)")
    print("Code: src/features/numeric.py:15-78")
    print("=" * 80)

    daily = load_universe_ohlcv(
        cfg["data"]["price_data_dir"], cached_symbols,
        cfg.get("ohlcv_columns", {}),
        start_date=cfg["data"].get("start_date"),
        end_date=cfg["data"].get("end_date"),
    )
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])

    print(f"\nNumeric feature columns: {NUMERIC_FEATURE_COLS}")
    print(f"\nFormulas:")
    print(f"  log_return   = ln(close_t / close_{{t-1}})                    [line 37]")
    print(f"  atr          = rolling_mean(TrueRange, 14 days)              [line 48]")
    print(f"  rolling_vol  = rolling_std(log_return, 20 days)              [line 51]")
    print(f"  ma_ratio     = SMA(5) / SMA(20)  — >1=uptrend, <1=downtrend [line 56]")
    print(f"  ma_long_slope= (SMA20_t - SMA20_{{t-5}}) / SMA20_{{t-5}}       [line 59]")
    print(f"  volume_zscore= (vol - rolling_mean) / rolling_std, 20 days   [line 62-64]")

    # Show sample
    sym = "RELIANCE" if "RELIANCE" in daily["symbol"].values else daily["symbol"].iloc[0]
    sym_df = daily[daily["symbol"] == sym].dropna(subset=NUMERIC_FEATURE_COLS).tail(10)

    print(f"\n--- {sym}: last 10 trading days ---")
    show_cols = ["date", "close", "volume"] + NUMERIC_FEATURE_COLS
    print(sym_df[show_cols].to_string(index=False, float_format="%.4f"))

    print(f"\n--- Interpretation of last row ---")
    last = sym_df.iloc[-1]
    print(f"  log_return    = {last['log_return']:.4f} → {'up' if last['log_return'] > 0 else 'down'} {abs(last['log_return'])*100:.2f}%")
    print(f"  atr           = {last['atr']:.2f} → avg daily price range is Rs {last['atr']:.2f}")
    print(f"  volume_zscore = {last['volume_zscore']:.4f} → volume is {abs(last['volume_zscore']):.1f} std {'above' if last['volume_zscore'] > 0 else 'below'} normal")
    print(f"  ma_ratio      = {last['ma_ratio']:.4f} → {'uptrend' if last['ma_ratio'] > 1 else 'downtrend'} (5-day MA {'above' if last['ma_ratio'] > 1 else 'below'} 20-day MA)")


def step10_cross_modal_interactions():
    """STEP 10: Cross-modal interaction features — text × numeric."""
    from src.utils.config import load_config
    from src.data.ohlcv_loader import load_universe_ohlcv
    from src.features.numeric import compute_numeric_features

    cfg = load_config("configs/default.yaml")
    cache_dir = Path("cache")
    cached_symbols = json.loads((cache_dir / "completed_symbols.json").read_text())

    INTERACTION_DEFS = [
        ("xmod_earn_vol",   "polarity_earnings",      "volume_zscore"),
        ("xmod_hi_atr",     "has_high_impact",         "atr"),
        ("xmod_pol_ret",    "polarity_high_impact",    "log_return"),
        ("xmod_pol_trend",  "polarity_high_impact",    "ma_ratio"),
        ("xmod_earn_trend", "polarity_earnings",       "ma_ratio"),
        ("xmod_board_vol",  "polarity_board_outcome",  "volume_zscore"),
    ]

    HIGH_IMPACT_GROUPS = {
        "earnings": ["earnings_positive", "earnings_negative", "earnings_neutral"],
        "board_outcome": ["board_outcome_positive", "board_outcome_negative", "board_outcome_neutral"],
        "dividend": ["dividend_announced"],
        "m_and_a": ["m_and_a_activity"],
    }

    print("=" * 80)
    print("STEP 10: CROSS-MODAL INTERACTION FEATURES")
    print("Code: run_category_cross_modal.py:79-87 (definitions)")
    print("Code: run_category_cross_modal.py:165-172 (computation)")
    print("=" * 80)

    # Load and compute numeric
    daily = load_universe_ohlcv(
        cfg["data"]["price_data_dir"], cached_symbols,
        cfg.get("ohlcv_columns", {}),
        start_date=cfg["data"].get("start_date"),
        end_date=cfg["data"].get("end_date"),
    )
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])
    daily["date"] = pd.to_datetime(daily["date"])

    # Load filings and build category features
    filings = pd.read_parquet(cache_dir / "filings_processed.parquet")
    filings["date"] = pd.to_datetime(filings["assigned_date"])

    cat_agg = (
        filings.groupby(["symbol", "date", "filing_category"])
        .agg(p_pos=("p_pos", "mean"), p_neg=("p_neg", "mean"),
             polarity=("polarity", "mean"))
        .reset_index()
    )

    merged = daily.copy()
    for group_name, cats in HIGH_IMPACT_GROUPS.items():
        group_data = cat_agg[cat_agg["filing_category"].isin(cats)]
        group_daily = (
            group_data.groupby(["symbol", "date"])
            .agg(p_pos=("p_pos", "mean"), p_neg=("p_neg", "mean"),
                 polarity=("polarity", "mean"))
            .reset_index()
        )
        group_daily = group_daily.rename(columns={
            "p_pos": f"p_pos_{group_name}",
            "p_neg": f"p_neg_{group_name}",
            "polarity": f"polarity_{group_name}",
        })
        group_daily[f"has_{group_name}"] = 1
        merged = merged.merge(group_daily, on=["symbol", "date"], how="left")
        for col in [f"has_{group_name}", f"p_pos_{group_name}",
                     f"p_neg_{group_name}", f"polarity_{group_name}"]:
            merged[col] = merged[col].fillna(0).astype(np.float32)

    hi_cols = [f"has_{g}" for g in HIGH_IMPACT_GROUPS]
    merged["has_high_impact"] = (merged[hi_cols].sum(axis=1) > 0).astype(np.float32)
    pol_cols = [f"polarity_{g}" for g in HIGH_IMPACT_GROUPS]
    has_cols = [f"has_{g}" for g in HIGH_IMPACT_GROUPS]
    pol_arr = merged[pol_cols].values
    has_arr = merged[has_cols].values
    with np.errstate(invalid="ignore"):
        merged["polarity_high_impact"] = np.where(
            has_arr.sum(axis=1) > 0,
            (pol_arr * has_arr).sum(axis=1) / np.maximum(has_arr.sum(axis=1), 1),
            0.0,
        ).astype(np.float32)

    # Compute interactions
    interaction_cols = []
    for feat_name, text_col, num_col in INTERACTION_DEFS:
        if text_col in merged.columns and num_col in merged.columns:
            merged[feat_name] = (
                merged[text_col].values * merged[num_col].values
            ).astype(np.float32)
            interaction_cols.append(feat_name)

    print(f"\nInteraction definitions:")
    print(f"  {'Feature':20s} = {'Text Column':25s} × {'Numeric Column':15s}")
    print(f"  {'-'*65}")
    for feat_name, text_col, num_col in INTERACTION_DEFS:
        print(f"  {feat_name:20s} = {text_col:25s} × {num_col:15s}")

    # Show filing days with interactions
    filing_days = merged[merged["has_high_impact"] == 1].dropna(subset=["atr"])
    if len(filing_days) > 0:
        sample = filing_days.iloc[0]
        print(f"\n{'='*80}")
        print(f"CONCRETE EXAMPLE: {sample['symbol']} on {sample['date'].date()}")
        print(f"{'='*80}")

        print(f"\n  Text-side values (from Steps 7-8):")
        for g in HIGH_IMPACT_GROUPS:
            if sample[f"has_{g}"] > 0:
                print(f"    has_{g} = {sample[f'has_{g}']:.0f}, "
                      f"polarity_{g} = {sample[f'polarity_{g}']:.4f}")
        print(f"    has_high_impact      = {sample['has_high_impact']:.0f}")
        print(f"    polarity_high_impact = {sample['polarity_high_impact']:.4f}")

        print(f"\n  Numeric-side values (from Step 9):")
        print(f"    volume_zscore = {sample['volume_zscore']:.4f}")
        print(f"    atr           = {sample['atr']:.4f}")
        print(f"    log_return    = {sample['log_return']:.4f}")
        print(f"    ma_ratio      = {sample['ma_ratio']:.4f}")

        print(f"\n  Interaction computation:")
        for feat_name, text_col, num_col in INTERACTION_DEFS:
            t_val = sample.get(text_col, 0)
            n_val = sample.get(num_col, 0)
            result = t_val * n_val
            print(f"    {feat_name:20s} = {text_col}({t_val:.4f}) × {num_col}({n_val:.4f}) = {result:.4f}")

        # Contrast with a non-filing day
        non_filing = merged[(merged["has_high_impact"] == 0) &
                            (merged["symbol"] == sample["symbol"])].dropna(subset=["atr"])
        if len(non_filing) > 0:
            nf = non_filing.iloc[0]
            print(f"\n  CONTRAST: Non-filing day ({nf['symbol']} on {nf['date'].date()}):")
            for feat_name, text_col, num_col in INTERACTION_DEFS:
                t_val = nf.get(text_col, 0)
                n_val = nf.get(num_col, 0)
                result = t_val * n_val
                print(f"    {feat_name:20s} = {text_col}({t_val:.4f}) × {num_col}({n_val:.4f}) = {result:.4f}")
            print(f"\n  >> ALL interactions = 0 on non-filing days because text values are 0.")


def step11_final_feature_vector():
    """STEP 11: The complete feature vector that goes into the model."""
    from scripts.exploratory.run_category_cross_modal import build_category_features
    from src.utils.config import load_config
    from src.data.ohlcv_loader import load_universe_ohlcv
    from src.features.numeric import compute_numeric_features, NUMERIC_FEATURE_COLS
    from src.features.chart2tokens import compute_chart2tokens, get_token_feature_cols

    cfg = load_config("configs/default.yaml")
    cache_dir = Path("cache")
    cached_symbols = json.loads((cache_dir / "completed_symbols.json").read_text())

    print("=" * 80)
    print("STEP 11: FINAL FEATURE VECTOR (M4a Early Fusion)")
    print("Code: run_category_cross_modal.py:406-407")
    print("=" * 80)

    # Load and compute everything
    daily = load_universe_ohlcv(
        cfg["data"]["price_data_dir"], cached_symbols,
        cfg.get("ohlcv_columns", {}),
        start_date=cfg["data"].get("start_date"),
        end_date=cfg["data"].get("end_date"),
    )
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])
    daily = compute_chart2tokens(daily, cfg["features"])
    daily["date"] = pd.to_datetime(daily["date"])

    filings = pd.read_parquet(cache_dir / "filings_processed.parquet")
    daily, cat_feature_cols, interaction_cols = build_category_features(filings, daily)

    W = cfg["features"]["token_summary"]["lookback_W"]
    numeric_cols = list(NUMERIC_FEATURE_COLS)
    token_cols = get_token_feature_cols(W)
    m4a_cols = list(dict.fromkeys(numeric_cols + token_cols + cat_feature_cols + interaction_cols))

    print(f"\nM4a feature vector has {len(m4a_cols)} features total:")
    print(f"\n  Numeric features ({len(numeric_cols)}):")
    for c in numeric_cols:
        print(f"    - {c}")
    print(f"\n  Token features ({len(token_cols)}):")
    for c in token_cols:
        print(f"    - {c}")
    print(f"\n  Category sentiment features ({len(cat_feature_cols)}):")
    for c in cat_feature_cols:
        print(f"    - {c}")
    print(f"\n  Cross-modal interactions ({len(interaction_cols)}):")
    for c in interaction_cols:
        print(f"    - {c}")

    # Show one filing-day row
    filing_days = daily[daily["has_high_impact"] == 1].dropna(subset=m4a_cols)
    if len(filing_days) > 0:
        sample = filing_days.iloc[0]
        print(f"\n{'='*80}")
        print(f"SAMPLE ROW: {sample['symbol']} on {sample['date'].date()}")
        print(f"{'='*80}")
        for c in m4a_cols:
            val = sample[c]
            tag = ""
            if c in interaction_cols:
                tag = " [CROSS-MODAL]"
            elif c in cat_feature_cols:
                tag = " [CATEGORY]"
            elif c in token_cols:
                tag = " [TOKEN]"
            elif c in numeric_cols:
                tag = " [NUMERIC]"
            print(f"  {c:35s} = {val:12.4f}{tag}")

        print(f"\n>> This entire vector goes into LightGBM/LogReg to predict:")
        print(f"   'Will tomorrow's close be higher than today's close?'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step-by-step pipeline walkthrough")
    parser.add_argument("--step", type=int, required=True, choices=range(1, 12),
                        help="Step number (1-11)")
    args = parser.parse_args()

    steps = {
        1: step1_raw_filings,
        2: step2_session_assignment,
        3: step3_classification,
        4: step4_loaded_filings,
        5: step5_finbert_scores,
        6: step6_first_groupby,
        7: step7_second_groupby,
        8: step8_aggregate_features,
        9: step9_numeric_features,
        10: step10_cross_modal_interactions,
        11: step11_final_feature_vector,
    }

    print(f"\n{'#'*80}")
    print(f"# RUNNING STEP {args.step}")
    print(f"{'#'*80}\n")
    steps[args.step]()
    print(f"\n{'#'*80}")
    print(f"# STEP {args.step} COMPLETE")
    print(f"# Next: python scripts/debug_flow.py --step {min(args.step + 1, 11)}")
    print(f"{'#'*80}")
