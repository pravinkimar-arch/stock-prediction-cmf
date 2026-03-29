"""Filing Days vs Non-Filing Days comparison (Table 16).

Runs the basic 3-variant walk-forward (numeric_only, numeric_tokens,
ts_text_fusion) on the full cached universe, then splits test-set
predictions into filing-day and non-filing-day subsets.

Outputs:
  - outputs/filing_days_pooled.csv   (per-fold, per-subset metrics)

Usage:
    python scripts/run_filing_days_comparison.py
"""

import json, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
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
from src.evaluation.metrics import compute_all_metrics

MC = ["roc_auc", "pr_auc", "f1", "brier", "ece"]


def create_labels(daily):
    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        df["next_close"] = df["close"].shift(-1)
        df["label"] = (np.log(df["next_close"] / df["close"]) > 0).astype(float)
        df.loc[df["next_close"].isna(), "label"] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def eval_sub(probs, labels, min_n=10):
    if len(probs) < min_n or len(np.unique(labels)) < 2:
        return {m: np.nan for m in MC}
    r = compute_all_metrics(probs, labels)
    r["n"] = len(probs)
    return r



def run_variant_with_subsets(name, feature_cols, daily, splits, cfg):
    """Run a single-branch LR variant, return per-fold metrics split by filing/no-filing."""
    cal_method = cfg["calibration"]["method"]
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    lr_cfg = cfg["models"]["logistic_regression"]
    results = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        train_df = train_df.dropna(subset=["label"])
        val_df = val_df.dropna(subset=["label"])
        test_df = test_df.dropna(subset=["label"])

        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        X_tr = train_df[feature_cols].values.astype(np.float32)
        y_tr = train_df["label"].values.astype(np.float32)
        X_va = val_df[feature_cols].values.astype(np.float32)
        y_va = val_df["label"].values.astype(np.float32)
        X_te = test_df[feature_cols].values.astype(np.float32)
        y_te = test_df["label"].values.astype(np.float32)
        for X in [X_tr, X_va, X_te]:
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            model, scaler = train_logistic_regression(X_tr, y_tr, lr_cfg)
            p_va = predict_logistic_regression(model, scaler, X_va)
            p_te = predict_logistic_regression(model, scaler, X_te)
            cal = get_calibrator(cal_method)
            cal.fit(p_va, y_va)
            p_te_cal = cal.transform(p_te)
        except Exception:
            continue

        # Subsets
        has_filing = (test_df["doc_count"].values > 0) if "doc_count" in test_df.columns else np.zeros(len(test_df), dtype=bool)
        no_filing = ~has_filing

        for subset_name, mask in [("all_days", None), ("filing_days", has_filing), ("no_filing_days", no_filing)]:
            if mask is not None:
                p, y = p_te_cal[mask], y_te[mask]
            else:
                p, y = p_te_cal, y_te
            met = eval_sub(p, y)
            met.update({"fold": split.fold_id, "variant": name, "subset": subset_name})
            results.append(met)

    return results




def run_fusion_with_subsets(daily, ts_cols, text_cols, splits, cfg):
    """Late fusion: TS branch + Text branch, split by filing/no-filing."""
    cal_method = cfg["calibration"]["method"]
    fusion_method = cfg["fusion"]["method"]
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    lgbm_cfg = cfg["models"]["lightgbm"]
    results = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        train_df = train_df.dropna(subset=["label"])
        val_df = val_df.dropna(subset=["label"])
        test_df = test_df.dropna(subset=["label"])

        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        y_tr = train_df["label"].values.astype(np.float32)
        y_va = val_df["label"].values.astype(np.float32)
        y_te = test_df["label"].values.astype(np.float32)

        try:
            # TS branch
            X_ts_tr = train_df[ts_cols].values.astype(np.float32)
            X_ts_va = val_df[ts_cols].values.astype(np.float32)
            X_ts_te = test_df[ts_cols].values.astype(np.float32)
            for X in [X_ts_tr, X_ts_va, X_ts_te]:
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            ts_m = train_lightgbm(X_ts_tr, y_tr, X_ts_va, y_va, lgbm_cfg)
            p_ts_va = predict_lightgbm(ts_m, X_ts_va)
            p_ts_te = predict_lightgbm(ts_m, X_ts_te)
            ts_cal = get_calibrator(cal_method)
            ts_cal.fit(p_ts_va, y_va)
            p_ts_va_c = ts_cal.transform(p_ts_va)
            p_ts_te_c = ts_cal.transform(p_ts_te)

            # Text branch
            X_tx_tr = train_df[text_cols].values.astype(np.float32)
            X_tx_va = val_df[text_cols].values.astype(np.float32)
            X_tx_te = test_df[text_cols].values.astype(np.float32)
            for X in [X_tx_tr, X_tx_va, X_tx_te]:
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            tx_m = train_lightgbm(X_tx_tr, y_tr, X_tx_va, y_va, lgbm_cfg)
            p_tx_va = predict_lightgbm(tx_m, X_tx_va)
            p_tx_te = predict_lightgbm(tx_m, X_tx_te)
            tx_cal = get_calibrator(cal_method)
            tx_cal.fit(p_tx_va, y_va)
            p_tx_va_c = tx_cal.transform(p_tx_va)
            p_tx_te_c = tx_cal.transform(p_tx_te)

            # Fuse
            fusion = get_fusion_model(fusion_method)
            fusion.fit(p_ts_va_c, p_tx_va_c, y_va)
            p_fused = fusion.predict(p_ts_te_c, p_tx_te_c)
        except Exception:
            continue

        has_filing = (test_df["doc_count"].values > 0) if "doc_count" in test_df.columns else np.zeros(len(test_df), dtype=bool)
        no_filing = ~has_filing

        for subset_name, mask in [("all_days", None), ("filing_days", has_filing), ("no_filing_days", no_filing)]:
            if mask is not None:
                p, y = p_fused[mask], y_te[mask]
            else:
                p, y = p_fused, y_te
            met = eval_sub(p, y)
            met.update({"fold": split.fold_id, "variant": "ts_text_fusion", "subset": subset_name})
            results.append(met)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filing days comparison")
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

    # Use all 98 symbols with FinBERT sentiment
    cached_symbols = json.loads((cache_dir / "completed_symbols.json").read_text())
    data_cfg = cfg["data"]
    ohlcv_map = cfg.get("ohlcv_columns", {})

    print("=" * 90)
    print("FILING DAYS vs NON-FILING DAYS COMPARISON (Table 16)")
    print(f"Universe: {len(cached_symbols)} symbols | 10-year data")
    print("=" * 90)

    # Load data
    print("\nLoading OHLCV + features...")
    daily = load_universe_ohlcv(
        data_cfg["price_data_dir"], cached_symbols, ohlcv_map,
        start_date=data_cfg.get("start_date"), end_date=data_cfg.get("end_date"),
    )
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])
    daily = compute_chart2tokens(daily, cfg["features"])
    daily = create_labels(daily)

    # Load text features
    print("Loading text features...")
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

    # Filter to 2024-2025 (same as thesis experiment)
    DATE_START = "2024-01-01"
    DATE_END = "2025-12-31"
    daily = daily[(daily["date"] >= DATE_START) & (daily["date"] <= DATE_END)].reset_index(drop=True)

    filing_count = (daily["doc_count"] > 0).sum() if "doc_count" in daily.columns else 0
    print(f"  {len(daily)} rows, {daily['symbol'].nunique()} symbols")
    print(f"  Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"  Filing days: {filing_count} ({filing_count/len(daily)*100:.1f}%)")

    # Feature sets
    W = cfg["features"]["token_summary"]["lookback_W"]
    numeric_cols = NUMERIC_FEATURE_COLS
    token_cols = get_token_feature_cols(W)
    text_cols = [c for c in TEXT_FEATURE_COLS if c in daily.columns]
    numeric_token_cols = numeric_cols + token_cols

    print(f"  Features: numeric={len(numeric_cols)}, tokens={len(token_cols)}, text={len(text_cols)}")

    # Walk-forward splits
    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"], cfg["splits"]["val_months"],
        cfg["splits"]["test_months"], cfg["splits"]["step_months"],
    )
    print(f"  Walk-forward: {len(splits)} folds")

    # Run variants
    all_results = []

    print("\n  >>> numeric_only")
    t1 = time.time()
    all_results.extend(run_variant_with_subsets("numeric_only", numeric_cols, daily, splits, cfg))
    print(f"      Done in {time.time()-t1:.1f}s")

    print("  >>> numeric_tokens")
    t2 = time.time()
    all_results.extend(run_variant_with_subsets("numeric_tokens", numeric_token_cols, daily, splits, cfg))
    print(f"      Done in {time.time()-t2:.1f}s")

    has_text = any(daily[c].sum() > 0 for c in text_cols if c in daily.columns)
    if has_text:
        print("  >>> ts_text_fusion")
        t3 = time.time()
        all_results.extend(run_fusion_with_subsets(daily, numeric_token_cols, text_cols, splits, cfg))
        print(f"      Done in {time.time()-t3:.1f}s")

    # Build results
    pdf = pd.DataFrame(all_results)
    if not dry_run:
        pdf.to_csv(output_dir / "filing_days_pooled.csv", index=False)
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/filing_days_pooled.csv")

    # Print Table 16
    print(f"\n{'='*90}")
    print("TABLE 16: Filing Days vs Non-Filing Days (averaged across folds)")
    print(f"{'='*90}")
    print(f"  {'Variant':25s} {'Filing AUC':>12s} {'Non-Filing AUC':>16s} {'Delta':>8s}")
    print("  " + "-" * 65)

    for v in ["numeric_only", "numeric_tokens", "ts_text_fusion"]:
        f_sub = pdf[(pdf["variant"] == v) & (pdf["subset"] == "filing_days")]
        nf_sub = pdf[(pdf["variant"] == v) & (pdf["subset"] == "no_filing_days")]
        if f_sub.empty or nf_sub.empty:
            continue
        f_auc = f_sub["roc_auc"].mean()
        nf_auc = nf_sub["roc_auc"].mean()
        delta = f_auc - nf_auc
        print(f"  {v:25s} {f_auc:12.3f} {nf_auc:16.3f} {delta:+8.3f}")

    total_min = (time.time() - t0) / 60
    print(f"\nTotal time: {total_min:.1f} min")
    print(f"Saved: {output_dir}/filing_days_pooled.csv")


if __name__ == "__main__":
    main()
