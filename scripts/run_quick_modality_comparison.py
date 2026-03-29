"""Run experiment on the 24 cached symbols only.

Uses pre-built caches to skip FinBERT inference.
Compares: numeric_only vs numeric_tokens vs ts_text_fusion.

Usage:
    python scripts/run_quick_modality_comparison.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, setup_logging
from src.data.ohlcv_loader import load_universe_ohlcv
from src.features.numeric import compute_numeric_features, NUMERIC_FEATURE_COLS
from src.features.chart2tokens import compute_chart2tokens, get_token_feature_cols
from src.features.text_sentiment import TEXT_FEATURE_COLS
from src.splits.walk_forward import generate_walk_forward_splits, apply_purge_embargo
from src.models.training import (
    train_logistic_regression, predict_logistic_regression,
    train_lightgbm, predict_lightgbm,
)
from src.models.calibration import get_calibrator
from src.models.fusion import get_fusion_model
from src.evaluation.metrics import (
    compute_all_metrics, build_results_table, compute_delta_metrics,
    compute_reliability_stats,
)

logger = logging.getLogger(__name__)


def create_labels(daily: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        df["next_close"] = df["close"].shift(-1)
        df["label"] = (np.log(df["next_close"] / df["close"]) > 0).astype(float)
        df.loc[df["next_close"].isna(), "label"] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)



def run_variant(name, feature_cols, daily, splits, cfg):
    """Run a single variant across all walk-forward splits. Returns list of result dicts."""
    model_type = cfg["models"].get("primary_model", "lightgbm")
    cal_method = cfg["calibration"]["method"]
    purge_days = cfg["splits"]["purge_days"]
    embargo_days = cfg["splits"]["embargo_days"]
    results = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge_days, embargo_days)
        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df["label"].values.astype(np.float32)
        X_val = val_df[feature_cols].values.astype(np.float32)
        y_val = val_df["label"].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)
        y_test = test_df["label"].values.astype(np.float32)

        for X in [X_train, X_val, X_test]:
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if model_type == "logistic_regression":
            model, scaler = train_logistic_regression(X_train, y_train, cfg["models"]["logistic_regression"])
            p_val_raw = predict_logistic_regression(model, scaler, X_val)
            p_test_raw = predict_logistic_regression(model, scaler, X_test)
        else:
            model = train_lightgbm(X_train, y_train, X_val, y_val, cfg["models"]["lightgbm"], feature_cols)
            p_val_raw = predict_lightgbm(model, X_val)
            p_test_raw = predict_lightgbm(model, X_test)

        calibrator = get_calibrator(cal_method)
        calibrator.fit(p_val_raw, y_val)
        p_test = calibrator.transform(p_test_raw)

        test_metrics = compute_all_metrics(p_test, y_test)
        test_rel = compute_reliability_stats(p_test, y_test)

        results.append({
            "fold": split.fold_id, "variant": name, "split": "test",
            "metrics": test_metrics, "reliability": test_rel,
        })
    return results


def run_fusion(daily, numeric_token_cols, text_cols, splits, cfg):
    """Run late fusion variant. Returns list of result dicts."""
    model_type = cfg["models"].get("primary_model", "lightgbm")
    cal_method = cfg["calibration"]["method"]
    fusion_method = cfg["fusion"]["method"]
    purge_days = cfg["splits"]["purge_days"]
    embargo_days = cfg["splits"]["embargo_days"]
    results = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge_days, embargo_days)
        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        y_train = train_df["label"].values.astype(np.float32)
        y_val = val_df["label"].values.astype(np.float32)
        y_test = test_df["label"].values.astype(np.float32)

        # TS branch (numeric + tokens)
        X_ts_tr = train_df[numeric_token_cols].values.astype(np.float32)
        X_ts_va = val_df[numeric_token_cols].values.astype(np.float32)
        X_ts_te = test_df[numeric_token_cols].values.astype(np.float32)
        for X in [X_ts_tr, X_ts_va, X_ts_te]:
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if model_type == "logistic_regression":
            ts_m, ts_s = train_logistic_regression(X_ts_tr, y_train, cfg["models"]["logistic_regression"])
            p_ts_va = predict_logistic_regression(ts_m, ts_s, X_ts_va)
            p_ts_te = predict_logistic_regression(ts_m, ts_s, X_ts_te)
        else:
            ts_m = train_lightgbm(X_ts_tr, y_train, X_ts_va, y_val, cfg["models"]["lightgbm"])
            p_ts_va = predict_lightgbm(ts_m, X_ts_va)
            p_ts_te = predict_lightgbm(ts_m, X_ts_te)

        ts_cal = get_calibrator(cal_method)
        ts_cal.fit(p_ts_va, y_val)
        p_ts_va_c = ts_cal.transform(p_ts_va)
        p_ts_te_c = ts_cal.transform(p_ts_te)

        # Text branch
        X_tx_tr = train_df[text_cols].values.astype(np.float32)
        X_tx_va = val_df[text_cols].values.astype(np.float32)
        X_tx_te = test_df[text_cols].values.astype(np.float32)
        for X in [X_tx_tr, X_tx_va, X_tx_te]:
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        if model_type == "logistic_regression":
            tx_m, tx_s = train_logistic_regression(X_tx_tr, y_train, cfg["models"]["logistic_regression"])
            p_tx_va = predict_logistic_regression(tx_m, tx_s, X_tx_va)
            p_tx_te = predict_logistic_regression(tx_m, tx_s, X_tx_te)
        else:
            tx_m = train_lightgbm(X_tx_tr, y_train, X_tx_va, y_val, cfg["models"]["lightgbm"])
            p_tx_va = predict_lightgbm(tx_m, X_tx_va)
            p_tx_te = predict_lightgbm(tx_m, X_tx_te)

        tx_cal = get_calibrator(cal_method)
        tx_cal.fit(p_tx_va, y_val)
        p_tx_va_c = tx_cal.transform(p_tx_va)
        p_tx_te_c = tx_cal.transform(p_tx_te)

        # Fuse
        fusion = get_fusion_model(fusion_method)
        fusion.fit(p_ts_va_c, p_tx_va_c, y_val)
        p_fused_test = fusion.predict(p_ts_te_c, p_tx_te_c)

        test_metrics = compute_all_metrics(p_fused_test, y_test)
        test_rel = compute_reliability_stats(p_fused_test, y_test)

        results.append({
            "fold": split.fold_id, "variant": "ts_text_fusion", "split": "test",
            "metrics": test_metrics, "reliability": test_rel,
        })
    return results



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick modality comparison")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip writing any files")
    args = parser.parse_args()
    dry_run = args.dry_run

    cfg = load_config("configs/default.yaml")
    setup_logging("INFO")
    set_seed(cfg.get("seed", 42))
    t0 = time.time()

    cache_dir = Path(cfg["data"]["cache_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load the 24 cached symbols
    cached_symbols = json.loads((cache_dir / "completed_symbols.json").read_text())
    print(f"Running on {len(cached_symbols)} cached symbols: {cached_symbols}")

    # Override config universe
    cfg["universe"]["symbols"] = cached_symbols
    symbols = cached_symbols
    data_cfg = cfg["data"]
    ohlcv_map = cfg.get("ohlcv_columns", {})

    # Step 1: Load OHLCV + numeric + chart2tokens + labels
    print("\n[1/4] Loading OHLCV & computing numeric + token features...")
    daily = load_universe_ohlcv(
        data_cfg["price_data_dir"], symbols, ohlcv_map,
        start_date=data_cfg.get("start_date"),
        end_date=data_cfg.get("end_date"),
    )
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])
    daily = compute_chart2tokens(daily, cfg["features"])
    daily = create_labels(daily)
    print(f"  OHLCV: {len(daily)} rows, {daily['symbol'].nunique()} symbols")

    # Step 2: Load cached text features
    print("[2/4] Loading cached text features...")
    text_cache_path = cache_dir / "daily_text_features.parquet"
    if not text_cache_path.exists():
        print(f"  ERROR: {text_cache_path} not found. Run scripts/build_cache.py first.")
        sys.exit(1)

    text_feats = pd.read_parquet(text_cache_path)
    text_feats = text_feats[text_feats["symbol"].isin(symbols)]
    text_feats["date"] = pd.to_datetime(text_feats["date"])
    daily = daily.merge(text_feats, on=["symbol", "date"], how="left")

    # Fill NaN text features
    for col in TEXT_FEATURE_COLS:
        if col in daily.columns:
            daily[col] = daily[col].fillna(0)
        else:
            daily[col] = 0

    daily = daily.dropna(subset=["label"] + NUMERIC_FEATURE_COLS).reset_index(drop=True)
    print(f"  Final dataset: {daily.shape}, date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"  Label dist: {daily['label'].value_counts().to_dict()}")

    # Step 3: Define feature sets
    W = cfg["features"]["token_summary"]["lookback_W"]
    numeric_cols = NUMERIC_FEATURE_COLS
    token_cols = get_token_feature_cols(W)
    text_cols = [c for c in TEXT_FEATURE_COLS if c in daily.columns]
    numeric_token_cols = numeric_cols + token_cols

    print(f"\n  Feature sets: numeric={len(numeric_cols)}, tokens={len(token_cols)}, text={len(text_cols)}")

    # Step 4: Walk-forward splits
    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"],
        cfg["splits"]["val_months"],
        cfg["splits"]["test_months"],
        cfg["splits"]["step_months"],
    )
    print(f"  Walk-forward splits: {len(splits)}")

    if not splits:
        print("ERROR: No valid splits. Check date range.")
        return

    # Run all 3 variants
    all_results = []

    print(f"\n[3/4] Running 3 variants across {len(splits)} folds...")

    print("\n  >>> Variant 1/3: numeric_only")
    t1 = time.time()
    all_results.extend(run_variant("numeric_only", numeric_cols, daily, splits, cfg))
    print(f"      Done in {time.time()-t1:.1f}s")

    print("  >>> Variant 2/3: numeric_tokens")
    t2 = time.time()
    all_results.extend(run_variant("numeric_tokens", numeric_token_cols, daily, splits, cfg))
    print(f"      Done in {time.time()-t2:.1f}s")

    # Robustly detect presence of any non-zero text feature value
    has_text = False
    if text_cols:
        try:
            has_text = bool(daily[text_cols].ne(0).any(axis=None))
        except Exception:
            has_text = any((c in daily.columns) and (daily[c].abs().sum() > 0) for c in text_cols)
    if has_text:
        print("  >>> Variant 3/3: ts_text_fusion")
        t3 = time.time()
        all_results.extend(run_fusion(daily, numeric_token_cols, text_cols, splits, cfg))
        print(f"      Done in {time.time()-t3:.1f}s")
    else:
        print("  >>> Skipping fusion (no text signal)")

    # Build results
    print(f"\n[4/4] Building results...")
    results_df = build_results_table(all_results)

    # Summary table
    metric_cols = ["roc_auc", "pr_auc", "f1", "brier", "ece",
                   "reliability_slope", "reliability_intercept"]
    summary = results_df.groupby("variant")[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]

    # Delta metrics
    deltas = compute_delta_metrics(results_df)

    if not dry_run:
        results_df.to_csv(output_dir / "results_cached24_per_fold.csv", index=False)
        summary.to_csv(output_dir / "results_cached24_summary.csv")
        if not deltas.empty:
            deltas.to_csv(output_dir / "delta_cached24.csv", index=False)
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/results_cached24_per_fold.csv")
        print(f"  [DRY RUN] Would write: {output_dir}/results_cached24_summary.csv")
        print(f"  [DRY RUN] Would write: {output_dir}/delta_cached24.csv")

    total_min = (time.time() - t0) / 60

    # Print results
    print("\n" + "=" * 90)
    print(f"RESULTS SUMMARY — {len(cached_symbols)} stocks, {len(splits)} folds (test splits)")
    print("=" * 90)
    print(summary.round(4).to_string())

    if not deltas.empty:
        print("\n" + "=" * 90)
        print("DELTA METRICS (per-fold incremental value)")
        print("=" * 90)

        # Aggregate deltas
        delta_summary = deltas.groupby("comparison").agg(
            **{col: (col, "mean") for col in deltas.columns if col.startswith("delta_")}
        )
        delta_std = deltas.groupby("comparison").agg(
            **{f"{col}_std": (col, "std") for col in deltas.columns if col.startswith("delta_")}
        )
        delta_combined = pd.concat([delta_summary, delta_std], axis=1)
        print(delta_combined.round(4).to_string())

        print("\nInterpretation:")
        print("  tokens_vs_numeric:  (numeric+tokens) - (numeric_only)")
        print("  fusion_vs_tokens:   (ts+text fusion) - (numeric+tokens)")
        print("  higher ROC-AUC, PR-AUC, F1 = better | lower Brier, ECE = better")

    print(f"\nTotal time: {total_min:.1f} min")
    print(f"Outputs: {output_dir}/results_cached24_*.csv")


if __name__ == "__main__":
    main()
