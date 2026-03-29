"""Validate Chart2Tokens v2 (best events + milestone) vs v1 (legacy).

Compares:
  - M1: Numeric only (baseline)
  - M2_v1: Numeric + legacy 21 tokens
  - M2_v2_best: Numeric + best 14 (pruned events + top-5 milestone)
  - M2_v2_full: Numeric + v2 29 (pruned events + all milestone)
  - M2_ms_only: Numeric + milestone only (20)
  - Fusion variants with text (M3)

Runs on both 2024-2025 and 10-year data.

Usage:
    python scripts/run_v2_validation.py
"""

import json, logging, os, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, setup_logging
from src.data.ohlcv_loader import load_universe_ohlcv
from src.features.numeric import compute_numeric_features, NUMERIC_FEATURE_COLS
from src.features.chart2tokens import (
    compute_chart2tokens,
    get_token_feature_cols,
    get_milestone_feature_cols,
    get_best_token_feature_cols,
    get_chart2tokens_v2_cols,
)
from src.features.text_sentiment import TEXT_FEATURE_COLS
from src.splits.walk_forward import generate_walk_forward_splits, apply_purge_embargo
from src.models.training import (
    train_logistic_regression, predict_logistic_regression,
    train_lightgbm, predict_lightgbm,
)
from src.models.calibration import get_calibrator
from src.evaluation.metrics import compute_all_metrics
from sklearn.linear_model import LogisticRegression as MetaLR

MC = ["roc_auc", "pr_auc", "f1", "brier", "ece"]

TEXT_T2 = [
    "mean_polarity", "max_polarity", "polarity_ema",
    "polarity_count_20", "polarity_recency_20", "polarity_tsl",
    "filing_recency_20",
]


def create_labels(daily):
    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        df["next_close"] = df["close"].shift(-1)
        df["label"] = (np.log(df["next_close"] / df["close"]) > 0).astype(float)
        df.loc[df["next_close"].isna(), "label"] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_data(cfg, start_date, end_date):
    cache_dir = Path(cfg["data"]["cache_dir"])
    cached_symbols = json.loads((cache_dir / "completed_symbols.json").read_text())
    data_cfg = cfg["data"]
    ohlcv_map = cfg.get("ohlcv_columns", {})

    daily = load_universe_ohlcv(
        data_cfg["price_data_dir"], list(cached_symbols), ohlcv_map,
        start_date=data_cfg.get("start_date"), end_date=data_cfg.get("end_date"),
    )
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])
    daily = compute_chart2tokens(daily, cfg["features"])
    daily = create_labels(daily)

    text_feats = pd.read_parquet(cache_dir / "daily_text_features.parquet")
    text_feats = text_feats[text_feats["symbol"].isin(list(cached_symbols))]
    text_feats["date"] = pd.to_datetime(text_feats["date"])
    daily = daily.merge(text_feats, on=["symbol", "date"], how="left")
    for col in TEXT_FEATURE_COLS:
        if col in daily.columns:
            daily[col] = daily[col].fillna(0)
        else:
            daily[col] = 0
    daily = daily.dropna(subset=["label"] + NUMERIC_FEATURE_COLS).reset_index(drop=True)

    if start_date and end_date:
        daily = daily[(daily["date"] >= start_date) & (daily["date"] <= end_date)].reset_index(drop=True)
    return daily


def eval_model(daily, cols, splits, cfg, model_type="lgbm"):
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    cal_method = cfg["calibration"]["method"]
    fold_metrics = []
    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        train_df = train_df.dropna(subset=["label"])
        val_df = val_df.dropna(subset=["label"])
        test_df = test_df.dropna(subset=["label"])
        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue
        X_tr = train_df[cols].values.astype(np.float32)
        y_tr = train_df["label"].values.astype(np.float32)
        X_va = val_df[cols].values.astype(np.float32)
        y_va = val_df["label"].values.astype(np.float32)
        X_te = test_df[cols].values.astype(np.float32)
        y_te = test_df["label"].values.astype(np.float32)
        for X in [X_tr, X_va, X_te]:
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            if model_type == "lr":
                m, s = train_logistic_regression(X_tr, y_tr, cfg["models"]["logistic_regression"])
                p_va = predict_logistic_regression(m, s, X_va)
                p_te = predict_logistic_regression(m, s, X_te)
            else:
                m = train_lightgbm(X_tr, y_tr, X_va, y_va, cfg["models"]["lightgbm"])
                p_va = predict_lightgbm(m, X_va)
                p_te = predict_lightgbm(m, X_te)
            cal = get_calibrator(cal_method)
            cal.fit(p_va, y_va)
            fold_metrics.append(compute_all_metrics(cal.transform(p_te), y_te))
        except Exception:
            pass
    if not fold_metrics:
        return {m: np.nan for m in MC}
    return {m: np.mean([f[m] for f in fold_metrics]) for m in MC}


def eval_crossmodal(daily, branches, splits, cfg):
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    cal_method = cfg["calibration"]["method"]
    fold_metrics = []
    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        train_df = train_df.dropna(subset=["label"])
        val_df = val_df.dropna(subset=["label"])
        test_df = test_df.dropna(subset=["label"])
        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue
        if len(np.unique(train_df["label"].values)) < 2:
            continue
        y_tr = train_df["label"].values.astype(np.float32)
        y_va = val_df["label"].values.astype(np.float32)
        y_te = test_df["label"].values.astype(np.float32)
        all_va, all_te = [], []
        ok = True
        for bcols in branches:
            X_tr = train_df[bcols].values.astype(np.float32)
            X_va = val_df[bcols].values.astype(np.float32)
            X_te = test_df[bcols].values.astype(np.float32)
            for X in [X_tr, X_va, X_te]:
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                m, s = train_logistic_regression(X_tr, y_tr, cfg["models"]["logistic_regression"])
                cal = get_calibrator(cal_method)
                cal.fit(predict_logistic_regression(m, s, X_va), y_va)
                all_va.append(cal.transform(predict_logistic_regression(m, s, X_va)))
                all_te.append(cal.transform(predict_logistic_regression(m, s, X_te)))
            except: ok = False; break
            try:
                gm = train_lightgbm(X_tr, y_tr, X_va, y_va, cfg["models"]["lightgbm"])
                cal = get_calibrator(cal_method)
                cal.fit(predict_lightgbm(gm, X_va), y_va)
                all_va.append(cal.transform(predict_lightgbm(gm, X_va)))
                all_te.append(cal.transform(predict_lightgbm(gm, X_te)))
            except: ok = False; break
        if not ok or len(all_va) < len(branches) * 2:
            continue
        try:
            meta = MetaLR(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            meta.fit(np.column_stack(all_va), y_va)
            fold_metrics.append(compute_all_metrics(
                meta.predict_proba(np.column_stack(all_te))[:, 1], y_te))
        except: pass
    if not fold_metrics:
        return {m: np.nan for m in MC}
    return {m: np.mean([f[m] for f in fold_metrics]) for m in MC}


def run_period(daily, period, cfg):
    W = cfg["features"]["token_summary"]["lookback_W"]
    m1 = list(NUMERIC_FEATURE_COLS)
    m2_v1 = get_token_feature_cols(W)
    m2_best = get_best_token_feature_cols(W)
    m2_v2 = get_chart2tokens_v2_cols(W)
    ms_only = get_milestone_feature_cols(W)
    m3 = [c for c in TEXT_T2 if c in daily.columns]

    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"], cfg["splits"]["val_months"],
        cfg["splits"]["test_months"], cfg["splits"]["step_months"],
    )
    print(f"  {len(daily)} rows, {daily['symbol'].nunique()} symbols, {len(splits)} folds")
    print(f"  M1={len(m1)} | M2_v1={len(m2_v1)} | M2_best={len(m2_best)} | M2_v2={len(m2_v2)} | MS={len(ms_only)} | M3={len(m3)}")

    results = []
    configs = [
        # === Single model baselines ===
        ("M1 numeric (LR)",                   m1, "lr"),
        ("M1 numeric (LGBM)",                 m1, "lgbm"),
        # === M1 + legacy C2T v1 ===
        ("M1 + C2T_v1 legacy (LR)",           m1 + m2_v1, "lr"),
        ("M1 + C2T_v1 legacy (LGBM)",         m1 + m2_v1, "lgbm"),
        # === M1 + C2T v2 best (pruned events + top-5 milestone) ===
        ("M1 + C2T_v2_best (LR)",             m1 + m2_best, "lr"),
        ("M1 + C2T_v2_best (LGBM)",           m1 + m2_best, "lgbm"),
        # === M1 + C2T v2 full (pruned events + all milestone) ===
        ("M1 + C2T_v2_full (LR)",             m1 + m2_v2, "lr"),
        ("M1 + C2T_v2_full (LGBM)",           m1 + m2_v2, "lgbm"),
        # === M1 + milestone only ===
        ("M1 + milestone_only (LR)",           m1 + ms_only, "lr"),
        ("M1 + milestone_only (LGBM)",         m1 + ms_only, "lgbm"),
        # === M1 + M3 text ===
        ("M1 + M3 text (LR)",                 m1 + m3, "lr"),
        ("M1 + M3 text (LGBM)",               m1 + m3, "lgbm"),
        # === M1 + C2T_v2_best + M3 ===
        ("M1 + C2T_v2_best + M3 (LR)",        m1 + m2_best + m3, "lr"),
        ("M1 + C2T_v2_best + M3 (LGBM)",      m1 + m2_best + m3, "lgbm"),
        # === M1 + C2T_v1 + M3 (legacy full) ===
        ("M1 + C2T_v1 + M3 legacy (LR)",      m1 + m2_v1 + m3, "lr"),
        ("M1 + C2T_v1 + M3 legacy (LGBM)",    m1 + m2_v1 + m3, "lgbm"),
    ]

    lr_base = None
    lgbm_base = None

    print(f"\n  {'Config':42s} {'#F':>3s} {'AUC':>7s} {'PR':>7s} {'F1':>7s} {'Brier':>7s} {'dAUC':>7s}")
    print("  " + "-" * 82)

    for name, cols, mt in configs:
        met = eval_model(daily, cols, splits, cfg, mt)
        if name == "M1 numeric (LR)": lr_base = met["roc_auc"]
        if name == "M1 numeric (LGBM)": lgbm_base = met["roc_auc"]
        base = lr_base if mt == "lr" else lgbm_base
        d = met["roc_auc"] - base if base else np.nan
        results.append({"config": name, "n_feat": len(cols), "period": period, **met})
        print(f"  {name:42s} {len(cols):3d} {met['roc_auc']:7.4f} {met['pr_auc']:7.4f} "
              f"{met['f1']:7.4f} {met['brier']:7.4f} {d:+7.4f}")

    # Cross-modal variants
    print(f"\n  Cross-modal variants:")
    print(f"  {'Config':42s} {'#F':>3s} {'AUC':>7s} {'PR':>7s} {'F1':>7s} {'Brier':>7s} {'dAUC':>7s}")
    print("  " + "-" * 82)

    cx_configs = [
        ("CX: M1 | C2T_v1 (legacy)",          [m1, m2_v1]),
        ("CX: M1 | C2T_v2_best",              [m1, m2_best]),
        ("CX: M1 | C2T_v2_full",              [m1, m2_v2]),
        ("CX: M1 | milestone",                [m1, ms_only]),
        ("CX: M1 | M3",                       [m1, m3]),
        ("CX: M1 | C2T_v2_best | M3",         [m1, m2_best, m3]),
        ("CX: M1 | C2T_v1 | M3 (legacy)",     [m1, m2_v1, m3]),
        ("CX: M1 | milestone | M3",           [m1, ms_only, m3]),
        ("CX: M1 | C2T_v2_best+M3 concat",    [m1, m2_best + m3]),
        ("CX: M1 | milestone+M3 concat",      [m1, ms_only + m3]),
    ]

    for name, branches in cx_configs:
        met = eval_crossmodal(daily, branches, splits, cfg)
        n = sum(len(b) for b in branches)
        d = met["roc_auc"] - lr_base if lr_base else np.nan
        results.append({"config": name, "n_feat": n, "period": period, **met})
        print(f"  {name:42s} {n:3d} {met['roc_auc']:7.4f} {met['pr_auc']:7.4f} "
              f"{met['f1']:7.4f} {met['brier']:7.4f} {d:+7.4f}")

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chart2Tokens v2 validation")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip writing any files")
    args = parser.parse_args()
    dry_run = args.dry_run

    cfg = load_config("configs/default.yaml")
    setup_logging("WARNING")
    set_seed(42)
    t0 = time.time()
    output_dir = Path(cfg["data"]["output_dir"])

    all_dfs = []
    for period, start, end in [
        ("2024-2025", "2024-01-01", "2025-12-31"),
        ("10-year",   "2016-01-01", "2026-01-31"),
    ]:
        print("\n" + "=" * 100)
        print(f"  CHART2TOKENS v2 VALIDATION — {period}")
        print("=" * 100)
        daily = load_data(cfg, start, end)
        df = run_period(daily, period, cfg)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    if not dry_run:
        combined.to_csv(output_dir / "chart2tokens_v2_validation.csv", index=False)
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/chart2tokens_v2_validation.csv")
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    if not dry_run:
        print(f"Saved: {output_dir}/chart2tokens_v2_validation.csv")


if __name__ == "__main__":
    main()
