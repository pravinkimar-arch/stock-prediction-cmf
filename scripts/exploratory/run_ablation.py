"""Feature ablation: prune noisy text features, check if fusion improves.

Tests 5 text feature tiers (from most to least pruned):
  T1: core sentiment only (3 features)
  T2: + memory/EMA (7 features)  
  T3: + filing metadata (10 features)
  T4: + high-impact categories only (16 features)
  T5: all 41 text features (baseline)

Runs both pooled (all stocks) and per-stock analysis.

Usage:
    python scripts/run_ablation.py
"""

import json, logging, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, setup_logging
from src.data.ohlcv_loader import load_universe_ohlcv
from src.features.numeric import compute_numeric_features, NUMERIC_FEATURE_COLS
from src.features.chart2tokens import compute_chart2tokens, get_token_feature_cols
from src.features.text_sentiment import TEXT_FEATURE_COLS
from src.splits.walk_forward import generate_walk_forward_splits, apply_purge_embargo
from src.models.training import train_logistic_regression, predict_logistic_regression
from src.models.calibration import get_calibrator
from src.models.fusion import get_fusion_model
from src.evaluation.metrics import compute_all_metrics


# ============================================================
# Feature tiers
# ============================================================

# T1: Core sentiment signal — what FinBERT actually tells us
TIER1_CORE = ["mean_polarity", "max_polarity", "polarity_ema"]

# T2: + memory features — how sentiment evolves over time
TIER2_MEMORY = TIER1_CORE + [
    "polarity_count_20", "polarity_recency_20", "polarity_tsl",
    "filing_recency_20",
]

# T3: + filing metadata — volume and presence signals
TIER3_META = TIER2_MEMORY + [
    "doc_count", "no_filings_day", "filing_count_20",
]

# T4: + high-impact category dummies only (earnings, board, dividend, M&A)
HIGH_IMPACT_CATS = [
    "cat_earnings_positive", "cat_earnings_negative", "cat_earnings_neutral",
    "cat_board_outcome_positive", "cat_board_outcome_negative", "cat_board_outcome_neutral",
    "cat_dividend_announced", "cat_m_and_a_activity",
    "cat_credit_rating_positive", "cat_credit_rating_negative",
]
TIER4_HICAT = TIER3_META + HIGH_IMPACT_CATS

# T5: all text features (baseline — what we had before)
TIER5_ALL = TEXT_FEATURE_COLS

TIERS = {
    "T1_core_sentiment": TIER1_CORE,
    "T2_+memory": TIER2_MEMORY,
    "T3_+metadata": TIER3_META,
    "T4_+hi_categories": TIER4_HICAT,
    "T5_all_features": TIER5_ALL,
}



def create_labels(daily):
    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        df["next_close"] = df["close"].shift(-1)
        df["label"] = (np.log(df["next_close"] / df["close"]) > 0).astype(float)
        df.loc[df["next_close"].isna(), "label"] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_variant_lr(feature_cols, data, splits, cfg):
    """Run LR variant, return list of (fold, auc, brier) tuples."""
    cal_method = cfg["calibration"]["method"]
    purge_days = cfg["splits"]["purge_days"]
    embargo_days = cfg["splits"]["embargo_days"]
    results = []
    for split in splits:
        tr, va, te = apply_purge_embargo(data, split, purge_days, embargo_days)
        if len(tr) < 30 or len(va) < 10 or len(te) < 10:
            continue
        if len(np.unique(tr["label"])) < 2 or len(np.unique(te["label"])) < 2:
            continue
        X_tr = tr[feature_cols].values.astype(np.float32)
        y_tr = tr["label"].values.astype(np.float32)
        X_va = va[feature_cols].values.astype(np.float32)
        y_va = va["label"].values.astype(np.float32)
        X_te = te[feature_cols].values.astype(np.float32)
        y_te = te["label"].values.astype(np.float32)
        for X in [X_tr, X_va, X_te]:
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            m, s = train_logistic_regression(X_tr, y_tr, cfg["models"]["logistic_regression"])
            p_va = predict_logistic_regression(m, s, X_va)
            p_te = predict_logistic_regression(m, s, X_te)
            cal = get_calibrator(cal_method)
            cal.fit(p_va, y_va)
            p_te_c = cal.transform(p_te)
            met = compute_all_metrics(p_te_c, y_te)
            if not np.isnan(met["roc_auc"]):
                results.append((split.fold_id, met["roc_auc"], met["brier"]))
        except Exception:
            continue
    return results


def run_fusion_lr(data, nt_cols, text_cols, splits, cfg):
    """Run fusion with LR, return list of (fold, auc, brier) tuples."""
    cal_method = cfg["calibration"]["method"]
    fusion_method = cfg["fusion"]["method"]
    purge_days = cfg["splits"]["purge_days"]
    embargo_days = cfg["splits"]["embargo_days"]
    results = []
    for split in splits:
        tr, va, te = apply_purge_embargo(data, split, purge_days, embargo_days)
        if len(tr) < 30 or len(va) < 10 or len(te) < 10:
            continue
        if len(np.unique(tr["label"])) < 2 or len(np.unique(te["label"])) < 2:
            continue
        y_tr = tr["label"].values.astype(np.float32)
        y_va = va["label"].values.astype(np.float32)
        y_te = te["label"].values.astype(np.float32)
        try:
            # TS branch
            X_ts = [tr[nt_cols].values.astype(np.float32),
                    va[nt_cols].values.astype(np.float32),
                    te[nt_cols].values.astype(np.float32)]
            for X in X_ts:
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            ts_m, ts_s = train_logistic_regression(X_ts[0], y_tr, cfg["models"]["logistic_regression"])
            p_ts_va = predict_logistic_regression(ts_m, ts_s, X_ts[1])
            p_ts_te = predict_logistic_regression(ts_m, ts_s, X_ts[2])
            ts_cal = get_calibrator(cal_method)
            ts_cal.fit(p_ts_va, y_va)
            p_ts_va_c = ts_cal.transform(p_ts_va)
            p_ts_te_c = ts_cal.transform(p_ts_te)

            # Text branch
            X_tx = [tr[text_cols].values.astype(np.float32),
                    va[text_cols].values.astype(np.float32),
                    te[text_cols].values.astype(np.float32)]
            for X in X_tx:
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            tx_m, tx_s = train_logistic_regression(X_tx[0], y_tr, cfg["models"]["logistic_regression"])
            p_tx_va = predict_logistic_regression(tx_m, tx_s, X_tx[1])
            p_tx_te = predict_logistic_regression(tx_m, tx_s, X_tx[2])
            tx_cal = get_calibrator(cal_method)
            tx_cal.fit(p_tx_va, y_va)
            p_tx_va_c = tx_cal.transform(p_tx_va)
            p_tx_te_c = tx_cal.transform(p_tx_te)

            # Fuse
            fus = get_fusion_model(fusion_method)
            fus.fit(p_ts_va_c, p_tx_va_c, y_va)
            p_fused = fus.predict(p_ts_te_c, p_tx_te_c)
            met = compute_all_metrics(p_fused, y_te)
            if not np.isnan(met["roc_auc"]):
                results.append((split.fold_id, met["roc_auc"], met["brier"]))
        except Exception:
            continue
    return results



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Text feature ablation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip writing any files")
    args = parser.parse_args()
    dry_run = args.dry_run

    cfg = load_config("configs/default.yaml")
    setup_logging("WARNING")
    set_seed(42)
    t0 = time.time()

    cache_dir = Path(cfg["data"]["cache_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    cached_symbols = json.loads((cache_dir / "completed_symbols.json").read_text())
    print(f"Feature ablation on {len(cached_symbols)} symbols\n")

    data_cfg = cfg["data"]
    ohlcv_map = cfg.get("ohlcv_columns", {})

    # Load data
    print("Loading data...")
    daily = load_universe_ohlcv(
        data_cfg["price_data_dir"], cached_symbols, ohlcv_map,
        start_date=data_cfg.get("start_date"), end_date=data_cfg.get("end_date"),
    )
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])
    daily = compute_chart2tokens(daily, cfg["features"])
    daily = create_labels(daily)

    text_feats = pd.read_parquet(cache_dir / "daily_text_features.parquet")
    text_feats = text_feats[text_feats["symbol"].isin(cached_symbols)]
    text_feats["date"] = pd.to_datetime(text_feats["date"])
    daily = daily.merge(text_feats, on=["symbol", "date"], how="left")

    for col in TEXT_FEATURE_COLS:
        if col in daily.columns:
            daily[col] = daily[col].fillna(0)
        else:
            daily[col] = 0

    daily = daily.dropna(subset=["label"] + NUMERIC_FEATURE_COLS).reset_index(drop=True)

    W = cfg["features"]["token_summary"]["lookback_W"]
    numeric_cols = NUMERIC_FEATURE_COLS
    token_cols = get_token_feature_cols(W)
    nt_cols = numeric_cols + token_cols

    print(f"Data: {len(daily)} rows, {daily['symbol'].nunique()} symbols")
    print(f"\nFeature tiers:")
    for name, cols in TIERS.items():
        available = [c for c in cols if c in daily.columns]
        print(f"  {name:25s} {len(available):3d} features")

    # ============================================================
    # PART 1: Pooled (all stocks together)
    # ============================================================
    print("\n" + "=" * 100)
    print("PART 1: POOLED ABLATION (all stocks)")
    print("=" * 100)

    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"], cfg["splits"]["val_months"],
        cfg["splits"]["test_months"], cfg["splits"]["step_months"],
    )
    print(f"Walk-forward: {len(splits)} folds\n")

    # Baselines
    num_res = run_variant_lr(numeric_cols, daily, splits, cfg)
    nt_res = run_variant_lr(nt_cols, daily, splits, cfg)
    auc_num = np.mean([r[1] for r in num_res]) if num_res else np.nan
    auc_nt = np.mean([r[1] for r in nt_res]) if nt_res else np.nan
    brier_num = np.mean([r[2] for r in num_res]) if num_res else np.nan
    brier_nt = np.mean([r[2] for r in nt_res]) if nt_res else np.nan

    pooled_rows = [
        {"variant": "numeric_only", "n_text_feats": 0, "auc": auc_num, "brier": brier_num},
        {"variant": "numeric_tokens", "n_text_feats": 0, "auc": auc_nt, "brier": brier_nt},
    ]

    # Fusion with each tier
    for tier_name, tier_cols in TIERS.items():
        available = [c for c in tier_cols if c in daily.columns]
        if not available:
            continue
        fus_res = run_fusion_lr(daily, nt_cols, available, splits, cfg)
        auc_fus = np.mean([r[1] for r in fus_res]) if fus_res else np.nan
        brier_fus = np.mean([r[2] for r in fus_res]) if fus_res else np.nan
        pooled_rows.append({
            "variant": f"fusion_{tier_name}", "n_text_feats": len(available),
            "auc": auc_fus, "brier": brier_fus,
        })
        print(f"  {tier_name:25s} ({len(available):2d} feats) -> AUC={auc_fus:.4f}  Brier={brier_fus:.4f}")

    pooled_df = pd.DataFrame(pooled_rows)

    print(f"\n{'Variant':40s} {'#Text':>5s} {'AUC':>8s} {'Brier':>8s} {'dAUC vs NT':>11s}")
    print("-" * 80)
    for _, r in pooled_df.iterrows():
        delta = r["auc"] - auc_nt if not np.isnan(r["auc"]) else np.nan
        d_str = f"{delta:+.4f}" if not np.isnan(delta) else "   N/A"
        print(f"  {r['variant']:38s} {r['n_text_feats']:5.0f} {r['auc']:8.4f} {r['brier']:8.4f} {d_str:>11s}")

    # ============================================================
    # PART 2: Per-stock ablation
    # ============================================================
    print("\n" + "=" * 100)
    print("PART 2: PER-STOCK ABLATION")
    print("=" * 100)

    stock_rows = []
    for i, sym in enumerate(cached_symbols):
        sym_data = daily[daily["symbol"] == sym].copy()
        if len(sym_data) < 100:
            continue

        sym_splits = generate_walk_forward_splits(
            sym_data["date"],
            cfg["splits"]["train_months"], cfg["splits"]["val_months"],
            cfg["splits"]["test_months"], cfg["splits"]["step_months"],
        )
        if len(sym_splits) < 2:
            continue

        # Baselines
        s_num = run_variant_lr(numeric_cols, sym_data, sym_splits, cfg)
        s_nt = run_variant_lr(nt_cols, sym_data, sym_splits, cfg)
        a_num = np.mean([r[1] for r in s_num]) if s_num else np.nan
        a_nt = np.mean([r[1] for r in s_nt]) if s_nt else np.nan

        row = {"symbol": sym, "auc_numeric": a_num, "auc_tokens": a_nt}

        # Each tier
        best_tier = None
        best_auc = -1
        for tier_name, tier_cols in TIERS.items():
            available = [c for c in tier_cols if c in daily.columns]
            if not available:
                continue
            fus_res = run_fusion_lr(sym_data, nt_cols, available, sym_splits, cfg)
            a_fus = np.mean([r[1] for r in fus_res]) if fus_res else np.nan
            row[f"auc_{tier_name}"] = a_fus
            if not np.isnan(a_fus) and a_fus > best_auc:
                best_auc = a_fus
                best_tier = tier_name

        row["best_tier"] = best_tier
        row["best_fusion_auc"] = best_auc if best_auc > -1 else np.nan

        # Check hypothesis at best tier
        if best_tier and not np.isnan(a_nt) and not np.isnan(a_num):
            row["hypothesis_at_best"] = (a_nt > a_num) and (best_auc > a_nt)
        else:
            row["hypothesis_at_best"] = False

        stock_rows.append(row)

        # Print progress
        tier_aucs = " ".join(
            f"{row.get(f'auc_{tn}', np.nan):.4f}" for tn in TIERS.keys()
        )
        flag = "YES" if row.get("hypothesis_at_best", False) else " "
        print(f"  [{i+1:2d}/{len(cached_symbols)}] {sym:15s} num={a_num:.4f} tok={a_nt:.4f} | {tier_aucs} | best={best_tier} {flag}")

    stock_df = pd.DataFrame(stock_rows)

    # Summary table
    print("\n" + "=" * 100)
    print("PER-STOCK RESULTS — BEST TIER PER STOCK")
    print("=" * 100)
    print(f"{'Symbol':15s} {'AUC_num':>8s} {'AUC_tok':>8s} "
          f"{'T1':>7s} {'T2':>7s} {'T3':>7s} {'T4':>7s} {'T5':>7s} "
          f"{'Best tier':>20s} {'Best AUC':>9s} {'Hyp?':>5s}")
    print("-" * 115)

    for _, r in stock_df.sort_values("best_fusion_auc", ascending=False).iterrows():
        t_vals = []
        for tn in TIERS.keys():
            v = r.get(f"auc_{tn}", np.nan)
            t_vals.append(f"{v:.4f}" if not np.isnan(v) else "   N/A")
        flag = "YES" if r.get("hypothesis_at_best", False) else ""
        print(f"  {r['symbol']:13s} {r['auc_numeric']:8.4f} {r['auc_tokens']:8.4f} "
              f"{'  '.join(t_vals)} "
              f"{str(r['best_tier']):>20s} {r['best_fusion_auc']:9.4f} {flag:>5s}")

    # Count improvements
    n_total = len(stock_df)
    n_hyp_t5 = stock_df.apply(
        lambda r: (not np.isnan(r.get("auc_T5_all_features", np.nan))
                   and not np.isnan(r["auc_tokens"])
                   and not np.isnan(r["auc_numeric"])
                   and r["auc_tokens"] > r["auc_numeric"]
                   and r.get("auc_T5_all_features", 0) > r["auc_tokens"]),
        axis=1
    ).sum()
    n_hyp_best = stock_df["hypothesis_at_best"].sum()

    print(f"\n  Full hypothesis with T5 (all features):  {n_hyp_t5}/{n_total} stocks")
    print(f"  Full hypothesis at best tier:            {n_hyp_best}/{n_total} stocks")

    # Best tier distribution
    print(f"\n  Best tier distribution:")
    tier_counts = stock_df["best_tier"].value_counts()
    for tier, count in tier_counts.items():
        print(f"    {tier:25s} {count:3d} stocks ({count/n_total*100:.0f}%)")

    # Average AUC by tier
    print(f"\n  Mean AUC across all stocks by tier:")
    for tn in TIERS.keys():
        col = f"auc_{tn}"
        if col in stock_df.columns:
            mean_auc = stock_df[col].mean()
            print(f"    {tn:25s} {mean_auc:.4f}")
    print(f"    {'numeric_only':25s} {stock_df['auc_numeric'].mean():.4f}")
    print(f"    {'numeric_tokens':25s} {stock_df['auc_tokens'].mean():.4f}")

    # Save
    if not dry_run:
        stock_df.to_csv(output_dir / "ablation_per_stock.csv", index=False)
        pooled_df.to_csv(output_dir / "ablation_pooled.csv", index=False)
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/ablation_per_stock.csv")
        print(f"  [DRY RUN] Would write: {output_dir}/ablation_pooled.csv")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print(f"Saved: {output_dir}/ablation_*.csv")


if __name__ == "__main__":
    main()
