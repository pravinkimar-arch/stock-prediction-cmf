"""
Step 4: Modeling, Calibration, Fusion & Evaluation
===================================================
Thesis Objective 4: Train baselines and fusion models under walk-forward
evaluation with purge/embargo, calibrate probabilities, and quantify the
incremental value of each modality.

This script:
  1. Loads the feature matrix from cache/features_all.parquet (Step 2)
  2. Generates walk-forward splits with purge/embargo
  3. Trains three model variants per fold:
     a) Numeric-only baseline (LightGBM on OHLCV indicators)
     b) Numeric + Chart2Tokens (adds token summary features)
     c) Late fusion: TS branch + Text branch → fused probabilities
  4. Calibrates all probabilities (Platt scaling)
  5. Evaluates: ROC-AUC, PR-AUC, F1, Brier, ECE, reliability
  6. Computes delta metrics (incremental value of tokens and text)

Inputs:
  - cache/features_all.parquet    (from Step 2)

Outputs (in outputs/):
  - results_per_fold.csv          (per-fold metrics for each variant)
  - results_summary.csv           (aggregated mean ± std)
  - delta_metrics.csv             (incremental value of each modality)
  - reliability_curves.png        (calibration plots)
  - metrics_comparison.png        (bar chart comparison)

Usage:
    python pipeline/step4_modeling_evaluation.py [--skip-text] [--config configs/default.yaml]
"""

import argparse, logging, os, sys, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, setup_logging, log_environment
from src.features.numeric import NUMERIC_FEATURE_COLS
from src.features.chart2tokens import get_token_feature_cols
from src.features.text_sentiment import TEXT_FEATURE_COLS
from src.splits.walk_forward import generate_walk_forward_splits, apply_purge_embargo
from src.models.training import (
    train_lightgbm, predict_lightgbm,
    train_logistic_regression, predict_logistic_regression,
)
from src.models.calibration import get_calibrator
from src.models.fusion import get_fusion_model
from src.evaluation.metrics import (
    compute_all_metrics, build_results_table, compute_delta_metrics,
    compute_reliability_stats,
)
from src.evaluation.plots import plot_reliability_curves, plot_metrics_comparison

logger = logging.getLogger(__name__)


# ─── Training helpers ────────────────────────────────────────────────────────

def _train_and_predict(X_train, y_train, X_val, y_val, X_test, cfg):
    """Train a single model branch and return raw val/test probabilities."""
    model_type = cfg["models"].get("primary_model", "lightgbm")

    if model_type == "logistic_regression":
        lr_cfg = cfg["models"]["logistic_regression"]
        model, scaler = train_logistic_regression(X_train, y_train, lr_cfg)
        p_val = predict_logistic_regression(model, scaler, X_val)
        p_test = predict_logistic_regression(model, scaler, X_test)
    else:
        lgb_cfg = cfg["models"]["lightgbm"]
        model = train_lightgbm(X_train, y_train, X_val, y_val, lgb_cfg)
        p_val = predict_lightgbm(model, X_val)
        p_test = predict_lightgbm(model, X_test)

    return p_val, p_test


def _calibrate(p_val, p_test, y_val, method):
    """Fit calibrator on val, transform both val and test."""
    cal = get_calibrator(method)
    cal.fit(p_val, y_val)
    return cal.transform(p_val), cal.transform(p_test)


def _prepare_arrays(train_df, val_df, test_df, feature_cols):
    """Extract feature arrays and clean NaN/Inf."""
    X_tr = train_df[feature_cols].values.astype(np.float32)
    X_va = val_df[feature_cols].values.astype(np.float32)
    X_te = test_df[feature_cols].values.astype(np.float32)
    for X in [X_tr, X_va, X_te]:
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y_tr = train_df["label"].values.astype(np.float32)
    y_va = val_df["label"].values.astype(np.float32)
    y_te = test_df["label"].values.astype(np.float32)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ─── Variant runners ────────────────────────────────────────────────────────

def run_variant(name, feature_cols, daily, splits, cfg):
    """Run a single-branch variant across all walk-forward folds."""
    cal_method = cfg["calibration"]["method"]
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    results = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            logger.warning(f"  Fold {split.fold_id}: insufficient data, skipping")
            continue

        X_tr, y_tr, X_va, y_va, X_te, y_te = _prepare_arrays(
            train_df, val_df, test_df, feature_cols,
        )
        p_val_raw, p_test_raw = _train_and_predict(X_tr, y_tr, X_va, y_va, X_te, cfg)
        _, p_test = _calibrate(p_val_raw, p_test_raw, y_va, cal_method)

        metrics = compute_all_metrics(p_test, y_te)
        rel = compute_reliability_stats(p_test, y_te)

        results.append({
            "fold": split.fold_id, "variant": name, "split": "test",
            "metrics": metrics, "reliability": rel,
        })
        logger.info(
            f"  Fold {split.fold_id} [{name}]: "
            f"AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f} "
            f"Brier={metrics['brier']:.4f}"
        )

    return results


def run_fusion_variant(daily, ts_cols, text_cols, splits, cfg):
    """Run late fusion: separate TS and Text branches → fuse probabilities."""
    cal_method = cfg["calibration"]["method"]
    fusion_method = cfg["fusion"]["method"]
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    results = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        y_va = val_df["label"].values.astype(np.float32)
        y_te = test_df["label"].values.astype(np.float32)

        # TS branch
        X_ts_tr, y_tr, X_ts_va, _, X_ts_te, _ = _prepare_arrays(
            train_df, val_df, test_df, ts_cols,
        )
        p_ts_va_raw, p_ts_te_raw = _train_and_predict(
            X_ts_tr, y_tr, X_ts_va, y_va, X_ts_te, cfg,
        )
        p_ts_va, p_ts_te = _calibrate(p_ts_va_raw, p_ts_te_raw, y_va, cal_method)

        # Text branch
        X_tx_tr, _, X_tx_va, _, X_tx_te, _ = _prepare_arrays(
            train_df, val_df, test_df, text_cols,
        )
        p_tx_va_raw, p_tx_te_raw = _train_and_predict(
            X_tx_tr, y_tr, X_tx_va, y_va, X_tx_te, cfg,
        )
        p_tx_va, p_tx_te = _calibrate(p_tx_va_raw, p_tx_te_raw, y_va, cal_method)

        # Fuse
        fusion = get_fusion_model(fusion_method)
        fusion.fit(p_ts_va, p_tx_va, y_va)
        p_fused = fusion.predict(p_ts_te, p_tx_te)

        metrics = compute_all_metrics(p_fused, y_te)
        rel = compute_reliability_stats(p_fused, y_te)

        results.append({
            "fold": split.fold_id, "variant": "ts_text_fusion", "split": "test",
            "metrics": metrics, "reliability": rel,
        })
        logger.info(
            f"  Fold {split.fold_id} [fusion]: "
            f"AUC={metrics['roc_auc']:.4f} F1={metrics['f1']:.4f} "
            f"Brier={metrics['brier']:.4f}"
        )

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 4: Modeling & Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-text", action="store_true", help="Skip fusion variant")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip writing any files")
    args = parser.parse_args()
    dry_run = args.dry_run

    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    set_seed(cfg.get("seed", 42))
    t0 = time.time()

    cache_dir = Path(cfg["data"]["cache_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_environment(str(output_dir))

    # ── 1. Load feature matrix ──
    print("=" * 80)
    print("STEP 4: Modeling, Calibration, Fusion & Evaluation")
    if dry_run:
        print("  ** DRY RUN — no files will be written **")
    print("=" * 80)

    feat_path = cache_dir / "features_all.parquet"
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found. Run pipeline/step2_feature_engineering.py first.")
        sys.exit(1)

    daily = pd.read_parquet(feat_path)
    daily["date"] = pd.to_datetime(daily["date"])
    print(f"\nLoaded {len(daily)} rows, {daily['symbol'].nunique()} symbols")
    print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")

    # ── 2. Define feature sets ──
    W = cfg["features"]["token_summary"]["lookback_W"]
    numeric_cols = NUMERIC_FEATURE_COLS
    token_cols = get_token_feature_cols(W)
    text_cols = [c for c in TEXT_FEATURE_COLS if c in daily.columns]
    numeric_token_cols = numeric_cols + token_cols

    print(f"\nFeature sets:")
    print(f"  Numeric:        {len(numeric_cols)}")
    print(f"  Tokens:         {len(token_cols)}")
    print(f"  Text:           {len(text_cols)}")
    print(f"  Numeric+Tokens: {len(numeric_token_cols)}")

    # ── 3. Walk-forward splits ──
    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"],
        cfg["splits"]["val_months"],
        cfg["splits"]["test_months"],
        cfg["splits"]["step_months"],
    )
    print(f"  Walk-forward folds: {len(splits)}")

    if not splits:
        print("ERROR: No valid splits. Check date range and split config.")
        return

    # ── 4. Run variants ──
    all_results = []

    print(f"\n>>> Variant 1/3: numeric_only")
    t1 = time.time()
    all_results.extend(run_variant("numeric_only", numeric_cols, daily, splits, cfg))
    print(f"    Done in {time.time()-t1:.1f}s")

    print(f"\n>>> Variant 2/3: numeric_tokens")
    t2 = time.time()
    all_results.extend(run_variant("numeric_tokens", numeric_token_cols, daily, splits, cfg))
    print(f"    Done in {time.time()-t2:.1f}s")

    has_text = not args.skip_text and any(
        daily[c].sum() > 0 for c in text_cols if c in daily.columns
    )
    if has_text:
        print(f"\n>>> Variant 3/3: ts_text_fusion (late fusion)")
        t3 = time.time()
        all_results.extend(
            run_fusion_variant(daily, numeric_token_cols, text_cols, splits, cfg)
        )
        print(f"    Done in {time.time()-t3:.1f}s")
    else:
        print("\n>>> Skipping fusion variant (no text data or --skip-text)")

    # ── 5. Build results ──
    results_df = build_results_table(all_results)

    # Aggregated summary
    metric_cols = [
        "roc_auc", "pr_auc", "f1", "brier", "ece",
        "reliability_slope", "reliability_intercept",
    ]
    summary = results_df.groupby("variant")[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]

    # Delta metrics
    deltas = compute_delta_metrics(results_df)

    if not dry_run:
        results_df.to_csv(output_dir / "results_per_fold.csv", index=False)
        summary.to_csv(output_dir / "results_summary.csv")
        if not deltas.empty:
            deltas.to_csv(output_dir / "delta_metrics.csv", index=False)
        plot_reliability_curves(all_results, str(output_dir))
        plot_metrics_comparison(results_df, str(output_dir))
    else:
        print(f"\n  [DRY RUN] Would write: {output_dir}/results_per_fold.csv")
        print(f"  [DRY RUN] Would write: {output_dir}/results_summary.csv")
        print(f"  [DRY RUN] Would write: {output_dir}/delta_metrics.csv")
        print(f"  [DRY RUN] Would write: {output_dir}/reliability_curves.png")
        print(f"  [DRY RUN] Would write: {output_dir}/metrics_comparison.png")

    # ── 6. Print results ──
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY (test splits, {len(splits)} folds)")
    print(f"{'='*80}")
    print(summary.round(4).to_string())

    if not deltas.empty:
        print(f"\n{'='*80}")
        print("DELTA METRICS (incremental value)")
        print(f"{'='*80}")
        delta_agg = deltas.groupby("comparison").agg(
            **{col: (col, "mean") for col in deltas.columns if col.startswith("delta_")}
        )
        print(delta_agg.round(4).to_string())
        print("\n  tokens_vs_numeric:  (numeric+tokens) − (numeric_only)")
        print("  fusion_vs_tokens:   (ts+text fusion) − (numeric+tokens)")
        print("  ↑ AUC, PR-AUC, F1 = better | ↓ Brier, ECE = better")

    total_min = (time.time() - t0) / 60
    print(f"\nAll outputs saved to {output_dir}/")
    print(f"Total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
