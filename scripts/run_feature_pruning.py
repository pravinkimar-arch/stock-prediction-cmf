"""Feature Pruning Experiment (2024-2025 data).

Crosses multiple feature selection strategies with all model variants
to find if removing underperforming features improves prediction.

Strategies:
  S1: Full features (baseline)
  S2: LGBM importance top-K (per modality)
  S3: Drop zero-importance features
  S4: Correlation-based pruning (remove redundant features)
  S5: Ablation per feature group (chart event types, text tiers)
  S6: Minimal numeric (top-3 by importance)

Model variants tested for each strategy:
  LR single-modal, LGBM single-modal, Early fusion, Late fusion, Cross-modal

Usage:
    python scripts/run_feature_pruning.py
"""

import json, logging, os, sys, time, warnings
from pathlib import Path
from collections import defaultdict
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

logger = logging.getLogger(__name__)
MC = ["roc_auc", "pr_auc", "f1", "brier", "ece"]

DATE_START = os.environ.get("PRUNE_START", "2024-01-01")
DATE_END = os.environ.get("PRUNE_END", "2025-12-31")

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


def load_data(cfg):
    """Load and filter data to 2024-2025."""
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

    # Filter to 2024-2025
    daily = daily[(daily["date"] >= DATE_START) & (daily["date"] <= DATE_END)].reset_index(drop=True)
    return daily


def get_feature_importance(daily, feature_cols, cfg, splits):
    """Get average LGBM feature importance across folds."""
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    importance = np.zeros(len(feature_cols))
    n_folds = 0

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        if len(train_df) < 50 or len(val_df) < 10:
            continue

        X_tr = train_df[feature_cols].values.astype(np.float32)
        y_tr = train_df["label"].values.astype(np.float32)
        X_va = val_df[feature_cols].values.astype(np.float32)
        y_va = val_df["label"].values.astype(np.float32)
        np.nan_to_num(X_tr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(X_va, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        model = train_lightgbm(X_tr, y_tr, X_va, y_va, cfg["models"]["lightgbm"])
        importance += model.feature_importances_
        n_folds += 1

    if n_folds > 0:
        importance /= n_folds
    return dict(zip(feature_cols, importance))


def eval_variant_across_folds(daily, feature_cols, splits, cfg, model_type="lgbm"):
    """Evaluate a single-branch model across folds. Returns mean metrics."""
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    cal_method = cfg["calibration"]["method"]
    fold_metrics = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
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
            p_te_cal = cal.transform(p_te)
            metrics = compute_all_metrics(p_te_cal, y_te)
            fold_metrics.append(metrics)
        except Exception:
            pass

    if not fold_metrics:
        return {m: np.nan for m in MC}
    return {m: np.mean([f[m] for f in fold_metrics]) for m in MC}


def eval_early_fusion(daily, all_cols, splits, cfg, model_type="lgbm"):
    """Early fusion = concatenate features, train one model."""
    return eval_variant_across_folds(daily, all_cols, splits, cfg, model_type)


def eval_late_fusion(daily, branch_cols_list, splits, cfg, model_type="lr"):
    """Late fusion: train separate branches, meta-LR on calibrated probs."""
    from sklearn.linear_model import LogisticRegression as MetaLR
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    cal_method = cfg["calibration"]["method"]
    fold_metrics = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        y_tr = train_df["label"].values.astype(np.float32)
        y_va = val_df["label"].values.astype(np.float32)
        y_te = test_df["label"].values.astype(np.float32)

        branch_va_probs = []
        branch_te_probs = []
        ok = True

        for bcols in branch_cols_list:
            X_tr = train_df[bcols].values.astype(np.float32)
            X_va = val_df[bcols].values.astype(np.float32)
            X_te = test_df[bcols].values.astype(np.float32)
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
                branch_va_probs.append(cal.transform(p_va))
                branch_te_probs.append(cal.transform(p_te))
            except Exception:
                ok = False
                break

        if not ok or len(branch_va_probs) < 2:
            continue

        try:
            X_meta_va = np.column_stack(branch_va_probs)
            X_meta_te = np.column_stack(branch_te_probs)
            meta = MetaLR(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            meta.fit(X_meta_va, y_va)
            p_fused = meta.predict_proba(X_meta_te)[:, 1]
            metrics = compute_all_metrics(p_fused, y_te)
            fold_metrics.append(metrics)
        except Exception:
            pass

    if not fold_metrics:
        return {m: np.nan for m in MC}
    return {m: np.mean([f[m] for f in fold_metrics]) for m in MC}


def eval_cross_modal(daily, branch_cols_list, splits, cfg):
    """Cross-modal: LR + LGBM per branch, meta-LR on all streams."""
    from sklearn.linear_model import LogisticRegression as MetaLR
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    cal_method = cfg["calibration"]["method"]
    fold_metrics = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        y_tr = train_df["label"].values.astype(np.float32)
        y_va = val_df["label"].values.astype(np.float32)
        y_te = test_df["label"].values.astype(np.float32)

        all_va_probs = []
        all_te_probs = []
        ok = True

        for bcols in branch_cols_list:
            X_tr = train_df[bcols].values.astype(np.float32)
            X_va = val_df[bcols].values.astype(np.float32)
            X_te = test_df[bcols].values.astype(np.float32)
            for X in [X_tr, X_va, X_te]:
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            # LR branch
            try:
                m, s = train_logistic_regression(X_tr, y_tr, cfg["models"]["logistic_regression"])
                p_va = predict_logistic_regression(m, s, X_va)
                p_te = predict_logistic_regression(m, s, X_te)
                cal = get_calibrator(cal_method)
                cal.fit(p_va, y_va)
                all_va_probs.append(cal.transform(p_va))
                all_te_probs.append(cal.transform(p_te))
            except Exception:
                ok = False
                break

            # LGBM branch
            try:
                gm = train_lightgbm(X_tr, y_tr, X_va, y_va, cfg["models"]["lightgbm"])
                p_va = predict_lightgbm(gm, X_va)
                p_te = predict_lightgbm(gm, X_te)
                cal = get_calibrator(cal_method)
                cal.fit(p_va, y_va)
                all_va_probs.append(cal.transform(p_va))
                all_te_probs.append(cal.transform(p_te))
            except Exception:
                ok = False
                break

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
        except Exception:
            pass

    if not fold_metrics:
        return {m: np.nan for m in MC}
    return {m: np.mean([f[m] for f in fold_metrics]) for m in MC}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Feature pruning experiment")
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

    print("=" * 110)
    print(f"FEATURE PRUNING EXPERIMENT — {DATE_START} to {DATE_END}")
    print("=" * 110)

    # Load data
    print("\nLoading data...")
    daily = load_data(cfg)
    print(f"  {len(daily)} rows, {daily['symbol'].nunique()} symbols")

    W = cfg["features"]["token_summary"]["lookback_W"]
    m1_cols = list(NUMERIC_FEATURE_COLS)
    m2_cols = get_token_feature_cols(W)
    m3_cols = [c for c in TEXT_T2 if c in daily.columns]
    all_cols = m1_cols + m2_cols + m3_cols

    print(f"  M1 numeric: {len(m1_cols)} | M2 visual: {len(m2_cols)} | M3 text: {len(m3_cols)} | Total: {len(all_cols)}")

    # Walk-forward splits
    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"], cfg["splits"]["val_months"],
        cfg["splits"]["test_months"], cfg["splits"]["step_months"],
    )
    print(f"  Walk-forward: {len(splits)} folds\n")

    # ================================================================
    # STEP 1: Feature Importance Analysis
    # ================================================================
    print("=" * 110)
    print("STEP 1: FEATURE IMPORTANCE (LGBM gain, averaged across folds)")
    print("=" * 110)

    imp_all = get_feature_importance(daily, all_cols, cfg, splits)
    imp_m1 = {k: v for k, v in imp_all.items() if k in m1_cols}
    imp_m2 = {k: v for k, v in imp_all.items() if k in m2_cols}
    imp_m3 = {k: v for k, v in imp_all.items() if k in m3_cols}

    print("\n  M1 Numeric features:")
    for f, v in sorted(imp_m1.items(), key=lambda x: -x[1]):
        print(f"    {f:30s} {v:10.1f}")

    print("\n  M2 Visual (Chart2Tokens) features:")
    for f, v in sorted(imp_m2.items(), key=lambda x: -x[1]):
        bar = "*" * int(v / max(max(imp_m2.values(), default=1), 1) * 20)
        print(f"    {f:40s} {v:10.1f}  {bar}")

    print("\n  M3 Text features:")
    for f, v in sorted(imp_m3.items(), key=lambda x: -x[1]):
        print(f"    {f:30s} {v:10.1f}")

    # Identify zero/low importance features
    zero_feats = [f for f, v in imp_all.items() if v == 0]
    low_feats = [f for f, v in sorted(imp_all.items(), key=lambda x: x[1]) if v < np.percentile(list(imp_all.values()), 25)]
    print(f"\n  Zero-importance features ({len(zero_feats)}): {zero_feats}")
    print(f"  Bottom-25% features ({len(low_feats)}): {low_feats}")

    # ================================================================
    # STEP 2: Define Pruning Strategies
    # ================================================================
    print("\n" + "=" * 110)
    print("STEP 2: PRUNING STRATEGIES")
    print("=" * 110)

    # Sort features by importance
    sorted_all = sorted(imp_all.items(), key=lambda x: -x[1])
    sorted_m1 = sorted(imp_m1.items(), key=lambda x: -x[1])
    sorted_m2 = sorted(imp_m2.items(), key=lambda x: -x[1])
    sorted_m3 = sorted(imp_m3.items(), key=lambda x: -x[1])

    # Top-K numeric
    top3_m1 = [f for f, _ in sorted_m1[:3]]
    top4_m1 = [f for f, _ in sorted_m1[:4]]

    # Top-K visual
    top5_m2 = [f for f, _ in sorted_m2[:5]]
    top10_m2 = [f for f, _ in sorted_m2[:10]]
    top15_m2 = [f for f, _ in sorted_m2[:15]]
    nonzero_m2 = [f for f, v in sorted_m2 if v > 0]

    # Top-K text
    top3_m3 = [f for f, _ in sorted_m3[:3]]
    top5_m3 = [f for f, _ in sorted_m3[:5]]
    nonzero_m3 = [f for f, v in sorted_m3 if v > 0]

    # Top-K overall (cross-modal)
    top10_all = [f for f, _ in sorted_all[:10]]
    top15_all = [f for f, _ in sorted_all[:15]]
    top20_all = [f for f, _ in sorted_all[:20]]
    nonzero_all = [f for f, v in sorted_all if v > 0]

    # Chart2Tokens grouped by event type
    event_types = ["breakout", "gap", "volume_burst", "round_touch", "engulfing"]
    token_groups = {}
    for et in event_types:
        token_groups[et] = [c for c in m2_cols if et in c]

    strategies = {}

    # S0: Baselines (full features)
    strategies["S0_full_M1"] = {"m1": m1_cols, "m2": None, "m3": None}
    strategies["S0_full_M1M2"] = {"m1": m1_cols, "m2": m2_cols, "m3": None}
    strategies["S0_full_all"] = {"m1": m1_cols, "m2": m2_cols, "m3": m3_cols}

    # S1: Top-K numeric
    strategies["S1_top3_M1"] = {"m1": top3_m1, "m2": None, "m3": None}
    strategies["S1_top4_M1"] = {"m1": top4_m1, "m2": None, "m3": None}

    # S2: Pruned visual
    strategies["S2_M1+top5_M2"] = {"m1": m1_cols, "m2": top5_m2, "m3": None}
    strategies["S2_M1+top10_M2"] = {"m1": m1_cols, "m2": top10_m2, "m3": None}
    strategies["S2_M1+top15_M2"] = {"m1": m1_cols, "m2": top15_m2, "m3": None}
    strategies["S2_M1+nonzero_M2"] = {"m1": m1_cols, "m2": nonzero_m2, "m3": None}

    # S3: Pruned text
    strategies["S3_M1+top3_M3"] = {"m1": m1_cols, "m2": None, "m3": top3_m3}
    strategies["S3_M1+top5_M3"] = {"m1": m1_cols, "m2": None, "m3": top5_m3}
    strategies["S3_M1+nonzero_M3"] = {"m1": m1_cols, "m2": None, "m3": nonzero_m3}

    # S4: Pruned all three
    strategies["S4_M1+top5M2+top3M3"] = {"m1": m1_cols, "m2": top5_m2, "m3": top3_m3}
    strategies["S4_M1+top10M2+top5M3"] = {"m1": m1_cols, "m2": top10_m2, "m3": top5_m3}
    strategies["S4_M1+nzM2+nzM3"] = {"m1": m1_cols, "m2": nonzero_m2, "m3": nonzero_m3}

    # S5: Top-K overall (ignoring modality boundaries)
    strategies["S5_top10_global"] = {"flat": top10_all}
    strategies["S5_top15_global"] = {"flat": top15_all}
    strategies["S5_top20_global"] = {"flat": top20_all}
    strategies["S5_nonzero_global"] = {"flat": nonzero_all}

    # S6: Drop each event type one at a time
    for et in event_types:
        remaining = [c for c in m2_cols if c not in token_groups[et]]
        strategies[f"S6_M1+M2_drop_{et}"] = {"m1": m1_cols, "m2": remaining, "m3": None}

    # S7: Minimal numeric + text only (skip visual)
    strategies["S7_top3M1+top3M3"] = {"m1": top3_m1, "m2": None, "m3": top3_m3}
    strategies["S7_top4M1+allM3"] = {"m1": top4_m1, "m2": None, "m3": m3_cols}

    for sname, spec in strategies.items():
        if "flat" in spec:
            n = len(spec["flat"])
        else:
            n = sum(len(v) for v in spec.values() if v is not None)
        print(f"  {sname:40s} {n:3d} features")

    # ================================================================
    # STEP 3: Evaluate all strategies x model variants
    # ================================================================
    print("\n" + "=" * 110)
    print("STEP 3: EVALUATION (strategy x model variant)")
    print("=" * 110)

    results = []

    for sname, spec in strategies.items():
        if "flat" in spec:
            flat_cols = spec["flat"]
            m1_s = flat_cols
            m2_s = None
            m3_s = None
            concat_cols = flat_cols
        else:
            m1_s = spec.get("m1")
            m2_s = spec.get("m2")
            m3_s = spec.get("m3")
            concat_cols = []
            if m1_s: concat_cols += m1_s
            if m2_s: concat_cols += m2_s
            if m3_s: concat_cols += m3_s

        n_feats = len(concat_cols)
        print(f"\n  --- {sname} ({n_feats} feats) ---")

        # LR on concatenated
        met = eval_variant_across_folds(daily, concat_cols, splits, cfg, "lr")
        row = {"strategy": sname, "variant": "LR_concat", "n_features": n_feats}
        row.update(met)
        results.append(row)
        print(f"    LR_concat:      AUC={met['roc_auc']:.4f}  Brier={met['brier']:.4f}")

        # LGBM on concatenated
        met = eval_variant_across_folds(daily, concat_cols, splits, cfg, "lgbm")
        row = {"strategy": sname, "variant": "LGBM_concat", "n_features": n_feats}
        row.update(met)
        results.append(row)
        print(f"    LGBM_concat:    AUC={met['roc_auc']:.4f}  Brier={met['brier']:.4f}")

        # Late fusion (only if we have 2+ modalities as separate branches)
        branches = []
        if m1_s and len(m1_s) > 0: branches.append(m1_s)
        if m2_s and len(m2_s) > 0: branches.append(m2_s)
        if m3_s and len(m3_s) > 0: branches.append(m3_s)

        if len(branches) >= 2:
            # Late fusion with LR branches
            met = eval_late_fusion(daily, branches, splits, cfg, "lr")
            row = {"strategy": sname, "variant": "Late_LR", "n_features": n_feats}
            row.update(met)
            results.append(row)
            print(f"    Late_LR:        AUC={met['roc_auc']:.4f}  Brier={met['brier']:.4f}")

            # Late fusion with LGBM branches
            met = eval_late_fusion(daily, branches, splits, cfg, "lgbm")
            row = {"strategy": sname, "variant": "Late_LGBM", "n_features": n_feats}
            row.update(met)
            results.append(row)
            print(f"    Late_LGBM:      AUC={met['roc_auc']:.4f}  Brier={met['brier']:.4f}")

            # Cross-modal (LR+LGBM per branch, meta-LR)
            met = eval_cross_modal(daily, branches, splits, cfg)
            row = {"strategy": sname, "variant": "CrossModal", "n_features": n_feats}
            row.update(met)
            results.append(row)
            print(f"    CrossModal:     AUC={met['roc_auc']:.4f}  Brier={met['brier']:.4f}")

    # ================================================================
    # STEP 4: Results Summary
    # ================================================================
    rdf = pd.DataFrame(results)
    if not dry_run:
        rdf.to_csv(output_dir / "feature_pruning_results.csv", index=False)
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/feature_pruning_results.csv")

    print("\n" + "=" * 110)
    print("RESULTS: BEST CONFIGURATIONS (sorted by AUC)")
    print("=" * 110)

    # Overall top-20
    top = rdf.nlargest(25, "roc_auc")
    print(f"\n  {'Strategy':40s} {'Variant':15s} {'#Feat':>5s} {'AUC':>7s} {'PR':>7s} {'F1':>7s} {'Brier':>7s} {'ECE':>7s}")
    print("  " + "-" * 95)
    for _, r in top.iterrows():
        print(f"  {r['strategy']:40s} {r['variant']:15s} {r['n_features']:5.0f} "
              f"{r['roc_auc']:7.4f} {r['pr_auc']:7.4f} {r['f1']:7.4f} {r['brier']:7.4f} {r['ece']:7.4f}")

    # Baseline comparison
    print("\n" + "=" * 110)
    print("IMPROVEMENT OVER BASELINES")
    print("=" * 110)

    baselines = {
        "LR_M1": rdf[(rdf["strategy"] == "S0_full_M1") & (rdf["variant"] == "LR_concat")]["roc_auc"].values,
        "LGBM_M1": rdf[(rdf["strategy"] == "S0_full_M1") & (rdf["variant"] == "LGBM_concat")]["roc_auc"].values,
        "LGBM_M1M2": rdf[(rdf["strategy"] == "S0_full_M1M2") & (rdf["variant"] == "LGBM_concat")]["roc_auc"].values,
        "LGBM_all": rdf[(rdf["strategy"] == "S0_full_all") & (rdf["variant"] == "LGBM_concat")]["roc_auc"].values,
    }
    for bname, bvals in baselines.items():
        if len(bvals) > 0:
            print(f"\n  Baseline {bname}: AUC = {bvals[0]:.4f}")

    # Best per variant type
    print("\n  Best per model variant:")
    for vtype in rdf["variant"].unique():
        vdf = rdf[rdf["variant"] == vtype].nlargest(3, "roc_auc")
        print(f"\n    {vtype}:")
        for _, r in vdf.iterrows():
            lgbm_m1_auc = baselines.get("LGBM_M1", [np.nan])[0]
            delta = r["roc_auc"] - lgbm_m1_auc if not np.isnan(lgbm_m1_auc) else np.nan
            print(f"      {r['strategy']:40s} AUC={r['roc_auc']:.4f} (dAUC={delta:+.4f}) Brier={r['brier']:.4f}")

    # Event type ablation summary
    print("\n" + "=" * 110)
    print("EVENT TYPE ABLATION (drop one event type from M2)")
    print("=" * 110)
    s0_m1m2_lgbm = rdf[(rdf["strategy"] == "S0_full_M1M2") & (rdf["variant"] == "LGBM_concat")]["roc_auc"].values
    base_auc = s0_m1m2_lgbm[0] if len(s0_m1m2_lgbm) > 0 else np.nan

    print(f"\n  {'Event dropped':25s} {'AUC':>7s} {'dAUC':>8s} {'Verdict':>10s}")
    print("  " + "-" * 55)
    for et in event_types:
        sname = f"S6_M1+M2_drop_{et}"
        val = rdf[(rdf["strategy"] == sname) & (rdf["variant"] == "LGBM_concat")]["roc_auc"].values
        if len(val) > 0:
            d = val[0] - base_auc
            verdict = "HELPS" if d > 0.001 else ("HURTS" if d < -0.001 else "neutral")
            print(f"  drop {et:20s} {val[0]:7.4f} {d:+8.4f} {verdict:>10s}")
    print(f"  {'(full M1+M2)':25s} {base_auc:7.4f} {'':>8s} {'baseline':>10s}")

    total_min = (time.time() - t0) / 60
    print(f"\nTotal time: {total_min:.1f} min")
    print(f"Saved: {output_dir}/feature_pruning_results.csv")


if __name__ == "__main__":
    main()
