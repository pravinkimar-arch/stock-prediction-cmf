"""Cross-modal fusion with Chart2Tokens + Text concatenated as one branch.

Branch A: Numeric (M1)
Branch B: Visual + Text concatenated (M2+M3)

Cross-modal: LR(A) + LGBM(A) + LR(B) + LGBM(B) → meta-LR

Also tests pruned variants of branch B.

Usage:
    python scripts/run_concat_crossmodal.py
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
from src.features.chart2tokens import compute_chart2tokens, get_token_feature_cols
from src.features.text_sentiment import TEXT_FEATURE_COLS
from src.splits.walk_forward import generate_walk_forward_splits, apply_purge_embargo
from src.models.training import (
    train_logistic_regression, predict_logistic_regression,
    train_lightgbm, predict_lightgbm,
)
from src.models.calibration import get_calibrator
from src.evaluation.metrics import compute_all_metrics
from sklearn.linear_model import LogisticRegression as MetaLR

logger = logging.getLogger(__name__)
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

    if start_date and end_date:
        daily = daily[(daily["date"] >= start_date) & (daily["date"] <= end_date)].reset_index(drop=True)

    return daily


def run_crossmodal(daily, branch_a_cols, branch_b_cols, splits, cfg, label=""):
    """Cross-modal: LR+LGBM per branch → meta-LR."""
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    cal_method = cfg["calibration"]["method"]
    lr_cfg = cfg["models"]["logistic_regression"]
    lgbm_cfg = cfg["models"]["lightgbm"]
    fold_metrics = []
    fold_filing_metrics = []

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

        all_va_probs = []
        all_te_probs = []
        ok = True

        for bcols in [branch_a_cols, branch_b_cols]:
            X_tr = train_df[bcols].values.astype(np.float32)
            X_va = val_df[bcols].values.astype(np.float32)
            X_te = test_df[bcols].values.astype(np.float32)
            for X in [X_tr, X_va, X_te]:
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # LR branch
            try:
                m, s = train_logistic_regression(X_tr, y_tr, lr_cfg)
                p_va = predict_logistic_regression(m, s, X_va)
                p_te = predict_logistic_regression(m, s, X_te)
                cal = get_calibrator(cal_method)
                cal.fit(p_va, y_va)
                all_va_probs.append(cal.transform(p_va))
                all_te_probs.append(cal.transform(p_te))
            except Exception:
                ok = False; break

            # LGBM branch
            try:
                gm = train_lightgbm(X_tr, y_tr, X_va, y_va, lgbm_cfg)
                p_va = predict_lightgbm(gm, X_va)
                p_te = predict_lightgbm(gm, X_te)
                cal = get_calibrator(cal_method)
                cal.fit(p_va, y_va)
                all_va_probs.append(cal.transform(p_va))
                all_te_probs.append(cal.transform(p_te))
            except Exception:
                ok = False; break

        if not ok or len(all_va_probs) < 4:
            continue

        try:
            X_meta_va = np.column_stack(all_va_probs)
            X_meta_te = np.column_stack(all_te_probs)
            meta = MetaLR(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            meta.fit(X_meta_va, y_va)
            p_fused = meta.predict_proba(X_meta_te)[:, 1]

            metrics = compute_all_metrics(p_fused, y_te)
            fold_metrics.append(metrics)

            # Filing days subset
            if "doc_count" in test_df.columns:
                fd_mask = test_df["doc_count"].values > 0
                if fd_mask.sum() >= 10 and len(np.unique(y_te[fd_mask])) >= 2:
                    fd_met = compute_all_metrics(p_fused[fd_mask], y_te[fd_mask])
                    fold_filing_metrics.append(fd_met)
        except Exception:
            pass

    all_res = {m: np.mean([f[m] for f in fold_metrics]) for m in MC} if fold_metrics else {m: np.nan for m in MC}
    all_res["n_folds"] = len(fold_metrics)

    fd_res = {m: np.mean([f[m] for f in fold_filing_metrics]) for m in MC} if fold_filing_metrics else {m: np.nan for m in MC}
    fd_res["n_folds"] = len(fold_filing_metrics)

    return all_res, fd_res


def run_single(daily, cols, splits, cfg, model_type="lr"):
    """Single model baseline."""
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
            metrics = compute_all_metrics(cal.transform(p_te), y_te)
            fold_metrics.append(metrics)
        except Exception:
            pass

    return {m: np.mean([f[m] for f in fold_metrics]) for m in MC} if fold_metrics else {m: np.nan for m in MC}


def run_experiment(daily, period_label, cfg):
    """Run all configs for one time period."""
    W = cfg["features"]["token_summary"]["lookback_W"]
    m1_cols = list(NUMERIC_FEATURE_COLS)
    m2_cols = get_token_feature_cols(W)
    m3_cols = [c for c in TEXT_T2 if c in daily.columns]

    # Top features (by importance from pruning experiment)
    top3_m1 = ["rolling_vol", "ma_ratio", "ma_long_slope"]
    top5_m2 = [c for c in ["gap_down_tsl", "gap_down_recency_20", "gap_up_tsl",
                            "gap_up_recency_20", "engulfing_bear_recency_20"] if c in m2_cols]
    top3_m3 = [c for c in ["polarity_ema", "polarity_recency_20", "max_polarity"] if c in m3_cols]

    # Concatenated branches (M2+M3)
    m2m3_full = m2_cols + m3_cols
    m2m3_pruned_top = top5_m2 + top3_m3
    m2m3_pruned_m2only = m2_cols
    m2m3_pruned_m3only = m3_cols

    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"], cfg["splits"]["val_months"],
        cfg["splits"]["test_months"], cfg["splits"]["step_months"],
    )
    print(f"  {len(daily)} rows, {daily['symbol'].nunique()} symbols, {len(splits)} folds")

    results = []

    # Baselines
    print("\n  Baselines:")
    for name, cols, mt in [
        ("LR_M1_full", m1_cols, "lr"),
        ("LR_M1_top3", top3_m1, "lr"),
        ("LGBM_M1_full", m1_cols, "lgbm"),
        ("LR_M1M2M3_concat", m1_cols + m2_cols + m3_cols, "lr"),
        ("LGBM_M1M2M3_concat", m1_cols + m2_cols + m3_cols, "lgbm"),
    ]:
        met = run_single(daily, cols, splits, cfg, mt)
        results.append({"config": name, "branch_a": "n/a", "branch_b": "n/a",
                        "n_feat": len(cols), "subset": "all", **met})
        print(f"    {name:40s} AUC={met['roc_auc']:.4f}  Brier={met['brier']:.4f}")

    # Cross-modal: 3-branch (original: M1 / M2 / M3)
    print("\n  Cross-modal 3-branch (original: M1 | M2 | M3):")
    # Can't directly do 3-branch in our 2-branch function, so do it manually
    # ... skip, already tested. Focus on 2-branch with M2+M3 concat.

    # Cross-modal 2-branch configs
    configs = [
        ("CX: M1_full | M2M3_full",       m1_cols, m2m3_full),
        ("CX: M1_full | M2M3_top5+top3",  m1_cols, m2m3_pruned_top),
        ("CX: M1_full | M2_only",         m1_cols, m2m3_pruned_m2only),
        ("CX: M1_full | M3_only",         m1_cols, m2m3_pruned_m3only),
        ("CX: M1_top3 | M2M3_full",       top3_m1, m2m3_full),
        ("CX: M1_top3 | M2M3_top5+top3",  top3_m1, m2m3_pruned_top),
        ("CX: M1_top3 | M3_only",         top3_m1, m2m3_pruned_m3only),
        ("CX: M1_top3 | M2_only",         top3_m1, m2m3_pruned_m2only),
    ]

    print("\n  Cross-modal 2-branch (M1 | M2+M3 concat):")
    print(f"  {'Config':45s} {'#A':>3s} {'#B':>3s} {'AUC(all)':>9s} {'AUC(fd)':>9s} {'Brier(all)':>10s} {'Brier(fd)':>10s}")
    print("  " + "-" * 90)

    for name, branch_a, branch_b in configs:
        all_met, fd_met = run_crossmodal(daily, branch_a, branch_b, splits, cfg, name)
        results.append({"config": name, "branch_a": len(branch_a), "branch_b": len(branch_b),
                        "n_feat": len(branch_a) + len(branch_b), "subset": "all", **all_met})
        results.append({"config": name + " [filing]", "branch_a": len(branch_a), "branch_b": len(branch_b),
                        "n_feat": len(branch_a) + len(branch_b), "subset": "filing", **fd_met})
        print(f"  {name:45s} {len(branch_a):3d} {len(branch_b):3d} "
              f"{all_met['roc_auc']:9.4f} {fd_met['roc_auc']:9.4f} "
              f"{all_met['brier']:10.4f} {fd_met['brier']:10.4f}")

    # Also test 3-branch cross-modal for comparison
    print("\n  Cross-modal 3-branch (M1 | M2 | M3) for comparison:")
    for name, branches in [
        ("CX3: M1_full | M2_full | M3_full", [m1_cols, m2_cols, m3_cols]),
        ("CX3: M1_top3 | M2_full | M3_full", [top3_m1, m2_cols, m3_cols]),
        ("CX3: M1_full | top5M2 | top3M3",   [m1_cols, top5_m2, top3_m3]),
    ]:
        # 3-branch cross-modal
        purge = cfg["splits"]["purge_days"]
        embargo = cfg["splits"]["embargo_days"]
        cal_method = cfg["calibration"]["method"]
        fold_all = []
        fold_fd = []

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
                    p_va = predict_logistic_regression(m, s, X_va)
                    p_te = predict_logistic_regression(m, s, X_te)
                    cal = get_calibrator(cal_method); cal.fit(p_va, y_va)
                    all_va.append(cal.transform(p_va)); all_te.append(cal.transform(p_te))
                except: ok = False; break
                try:
                    gm = train_lightgbm(X_tr, y_tr, X_va, y_va, cfg["models"]["lightgbm"])
                    p_va = predict_lightgbm(gm, X_va)
                    p_te = predict_lightgbm(gm, X_te)
                    cal = get_calibrator(cal_method); cal.fit(p_va, y_va)
                    all_va.append(cal.transform(p_va)); all_te.append(cal.transform(p_te))
                except: ok = False; break

            if not ok or len(all_va) < 6: continue
            try:
                meta = MetaLR(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
                meta.fit(np.column_stack(all_va), y_va)
                p = meta.predict_proba(np.column_stack(all_te))[:, 1]
                fold_all.append(compute_all_metrics(p, y_te))
                if "doc_count" in test_df.columns:
                    fd = test_df["doc_count"].values > 0
                    if fd.sum() >= 10 and len(np.unique(y_te[fd])) >= 2:
                        fold_fd.append(compute_all_metrics(p[fd], y_te[fd]))
            except: pass

        a = {m: np.mean([f[m] for f in fold_all]) for m in MC} if fold_all else {m: np.nan for m in MC}
        f = {m: np.mean([f[m] for f in fold_fd]) for m in MC} if fold_fd else {m: np.nan for m in MC}
        n_feat = sum(len(b) for b in branches)
        print(f"  {name:45s} {n_feat:3d}     "
              f"{a['roc_auc']:9.4f} {f['roc_auc']:9.4f} "
              f"{a['brier']:10.4f} {f['brier']:10.4f}")

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Cross-modal concat fusion")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip writing any files")
    args = parser.parse_args()
    dry_run = args.dry_run

    cfg = load_config("configs/default.yaml")
    setup_logging("WARNING")
    set_seed(42)
    t0 = time.time()
    output_dir = Path(cfg["data"]["output_dir"])
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    # Run on both time periods
    for period, start, end in [
        ("2024-2025", "2024-01-01", "2025-12-31"),
        ("10-year",   "2016-01-01", "2026-01-31"),
    ]:
        print("\n" + "=" * 110)
        print(f"  PERIOD: {period} ({start} to {end})")
        print("=" * 110)
        daily = load_data(cfg, start, end)
        df = run_experiment(daily, period, cfg)
        df["period"] = period
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    if not dry_run:
        combined.to_csv(output_dir / "concat_crossmodal_results.csv", index=False)
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/concat_crossmodal_results.csv")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    if not dry_run:
        print(f"Saved: {output_dir}/concat_crossmodal_results.csv")


if __name__ == "__main__":
    main()
