"""M4: Category-Aware Cross-Modal Fusion experiment.

Extends the thesis pipeline with M4 — a category-aware fusion variant that
replaces pooled text sentiment with category-specific sentiment features and
cross-modal interaction terms. Compares M1-M4 under identical walk-forward
conditions using both LR and LightGBM.

Model variants (identical walk-forward, purge/embargo, Platt calibration):
  M1: Numeric-only (6 OHLCV indicators)
  M2: Numeric + Chart2Tokens (6 + 15 token summary features)
  M3: Numeric + Tokens + Pooled Text late fusion (existing pipeline)
  M4a: Early fusion — Numeric + Tokens + Category Sentiment + Interactions
  M4b: Late fusion — TS branch + Category-Text branch, fused at prob level

M4 Feature engineering (category-aware):
  - Category-specific sentiment: for each high-impact filing group
    (earnings, board_outcome, dividend, m_and_a), creates:
      has_{group}, polarity_{group}, p_pos_{group}, p_neg_{group}
  - Aggregate: has_high_impact, polarity_high_impact
  - Cross-modal interaction features (text x numeric):
      earnings_polarity x volume_zscore
      has_high_impact x atr
      polarity_high_impact x log_return
      polarity_high_impact x ma_ratio
      earnings_polarity x ma_ratio
      board_outcome_polarity x volume_zscore
  - All features are leak-safe: sentiment uses assigned_date (post-15:30
    shifted to next session), interactions use day-t values only.

Usage:
    python scripts/exploratory/run_category_cross_modal.py

Outputs:
    outputs/m4_cross_modal_results.csv     -- per-fold metrics for M1-M4
    outputs/m4_cross_modal_summary.csv     -- mean +/- std per variant
    outputs/m4_feature_importance.csv      -- LightGBM feature importances for M4
"""

import json
import logging
import os
import sys
import time
import warnings
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
from src.models.training import (
    train_logistic_regression, predict_logistic_regression,
    train_lightgbm, predict_lightgbm,
)
from src.models.calibration import get_calibrator
from src.models.fusion import get_fusion_model
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)
MC = ["roc_auc", "pr_auc", "f1", "brier", "ece"]

# High-impact filing categories (market-moving events)
HIGH_IMPACT_GROUPS = {
    "earnings": ["earnings_positive", "earnings_negative", "earnings_neutral"],
    "board_outcome": ["board_outcome_positive", "board_outcome_negative", "board_outcome_neutral"],
    "dividend": ["dividend_announced"],
    "m_and_a": ["m_and_a_activity"],
}

# Cross-modal interaction definitions: (name, text_col, numeric_col)
INTERACTION_DEFS = [
    ("xmod_earn_vol", "polarity_earnings", "volume_zscore"),
    ("xmod_hi_atr", "has_high_impact", "atr"),
    ("xmod_pol_ret", "polarity_high_impact", "log_return"),
    ("xmod_pol_trend", "polarity_high_impact", "ma_ratio"),
    ("xmod_earn_trend", "polarity_earnings", "ma_ratio"),
    ("xmod_board_vol", "polarity_board_outcome", "volume_zscore"),
]

# Pooled text features (T2 tier from ablation — best signal subset)
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


def build_category_features(filings, daily):
    """Build category-specific sentiment + cross-modal interaction features.

    Returns: (daily_merged, cat_feature_cols, interaction_cols)
    """
    filings_cp = filings.copy()
    filings_cp["date"] = pd.to_datetime(filings_cp["assigned_date"])

    cat_agg = (
        filings_cp.groupby(["symbol", "date", "filing_category"])
        .agg(p_pos=("p_pos", "mean"), p_neg=("p_neg", "mean"),
             polarity=("polarity", "mean"), count=("polarity", "count"))
        .reset_index()
    )

    merged = daily.copy()
    merged["date"] = pd.to_datetime(merged["date"])

    cat_feature_cols = []
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
            cat_feature_cols.append(col)

    # Aggregate high-impact flag and polarity
    hi_cols = [f"has_{g}" for g in HIGH_IMPACT_GROUPS]
    merged["has_high_impact"] = (merged[hi_cols].sum(axis=1) > 0).astype(np.float32)
    cat_feature_cols.append("has_high_impact")

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
    cat_feature_cols.append("polarity_high_impact")

    # Cross-modal interaction features
    interaction_cols = []
    for feat_name, text_col, num_col in INTERACTION_DEFS:
        if text_col in merged.columns and num_col in merged.columns:
            merged[feat_name] = (
                merged[text_col].values * merged[num_col].values
            ).astype(np.float32)
            interaction_cols.append(feat_name)

    return merged, cat_feature_cols, interaction_cols


def _prep(df, cols):
    """Extract and clean feature arrays."""
    X = df[cols].values.astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def run_fold_all_variants(data, split, feature_sets, cfg):
    """Run all M1-M4 variants on a single walk-forward fold.

    feature_sets: dict of {variant_name: feature_col_list}
    Returns list of result dicts.
    """
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    cal_m = cfg["calibration"]["method"]
    lr_cfg = cfg["models"]["logistic_regression"]
    lgbm_cfg = cfg["models"]["lightgbm"]

    train_df, val_df, test_df = apply_purge_embargo(data, split, purge, embargo)
    if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
        return [], {}
    train_df = train_df.dropna(subset=["label"])
    val_df = val_df.dropna(subset=["label"])
    test_df = test_df.dropna(subset=["label"])
    if len(np.unique(train_df["label"].values)) < 2:
        return [], {}

    y_tr = train_df["label"].values.astype(np.float32)
    y_va = val_df["label"].values.astype(np.float32)
    y_te = test_df["label"].values.astype(np.float32)

    has_filing = (test_df["doc_count"].values > 0) if "doc_count" in test_df.columns else np.zeros(len(test_df), dtype=bool)

    results = []
    importances = {}  # variant -> {feature: importance}

    for vname, vcols in feature_sets.items():
        X_tr = _prep(train_df, vcols)
        X_va = _prep(val_df, vcols)
        X_te = _prep(test_df, vcols)

        # --- LightGBM ---
        try:
            gm = train_lightgbm(X_tr, y_tr, X_va, y_va, lgbm_cfg)
            p_va = predict_lightgbm(gm, X_va)
            p_te = predict_lightgbm(gm, X_te)
            cal = get_calibrator(cal_m)
            cal.fit(p_va, y_va)
            p_te_cal = cal.transform(p_te)

            # All days
            met = compute_all_metrics(p_te_cal, y_te)
            met.update({"fold": split.fold_id, "variant": f"gb_{vname}", "subset": "all_days", "n": len(y_te)})
            results.append(met)

            # Filing days
            if has_filing.sum() >= 10:
                met_fd = compute_all_metrics(p_te_cal[has_filing], y_te[has_filing])
                met_fd.update({"fold": split.fold_id, "variant": f"gb_{vname}", "subset": "filing_days", "n": int(has_filing.sum())})
                results.append(met_fd)

            # Feature importance
            if hasattr(gm, "feature_importances_"):
                importances[f"gb_{vname}"] = dict(zip(vcols, gm.feature_importances_))
        except Exception:
            pass

        # --- Logistic Regression ---
        try:
            m, s = train_logistic_regression(X_tr, y_tr, lr_cfg)
            p_va = predict_logistic_regression(m, s, X_va)
            p_te = predict_logistic_regression(m, s, X_te)
            cal = get_calibrator(cal_m)
            cal.fit(p_va, y_va)
            p_te_cal = cal.transform(p_te)

            met = compute_all_metrics(p_te_cal, y_te)
            met.update({"fold": split.fold_id, "variant": f"lr_{vname}", "subset": "all_days", "n": len(y_te)})
            results.append(met)

            if has_filing.sum() >= 10:
                met_fd = compute_all_metrics(p_te_cal[has_filing], y_te[has_filing])
                met_fd.update({"fold": split.fold_id, "variant": f"lr_{vname}", "subset": "filing_days", "n": int(has_filing.sum())})
                results.append(met_fd)
        except Exception:
            pass

    # --- M3 Late Fusion: TS branch + Text branch ---
    if "M2_num_tok" in feature_sets and "M3_text_only" in feature_sets:
        ts_cols = feature_sets["M2_num_tok"]
        tx_cols = feature_sets["M3_text_only"]
        try:
            X_ts_tr, X_ts_va, X_ts_te = _prep(train_df, ts_cols), _prep(val_df, ts_cols), _prep(test_df, ts_cols)
            X_tx_tr, X_tx_va, X_tx_te = _prep(train_df, tx_cols), _prep(val_df, tx_cols), _prep(test_df, tx_cols)

            # TS branch
            gm_ts = train_lightgbm(X_ts_tr, y_tr, X_ts_va, y_va, lgbm_cfg)
            p_ts_va = predict_lightgbm(gm_ts, X_ts_va)
            p_ts_te = predict_lightgbm(gm_ts, X_ts_te)
            cal_ts = get_calibrator(cal_m)
            cal_ts.fit(p_ts_va, y_va)
            p_ts_va_c, p_ts_te_c = cal_ts.transform(p_ts_va), cal_ts.transform(p_ts_te)

            # Text branch
            gm_tx = train_lightgbm(X_tx_tr, y_tr, X_tx_va, y_va, lgbm_cfg)
            p_tx_va = predict_lightgbm(gm_tx, X_tx_va)
            p_tx_te = predict_lightgbm(gm_tx, X_tx_te)
            cal_tx = get_calibrator(cal_m)
            cal_tx.fit(p_tx_va, y_va)
            p_tx_va_c, p_tx_te_c = cal_tx.transform(p_tx_va), cal_tx.transform(p_tx_te)

            # Fuse
            fusion = get_fusion_model(cfg["fusion"]["method"])
            fusion.fit(p_ts_va_c, p_tx_va_c, y_va)
            p_fused = fusion.predict(p_ts_te_c, p_tx_te_c)

            met = compute_all_metrics(p_fused, y_te)
            met.update({"fold": split.fold_id, "variant": "M3_late_fusion", "subset": "all_days", "n": len(y_te)})
            results.append(met)

            if has_filing.sum() >= 10:
                met_fd = compute_all_metrics(p_fused[has_filing], y_te[has_filing])
                met_fd.update({"fold": split.fold_id, "variant": "M3_late_fusion", "subset": "filing_days", "n": int(has_filing.sum())})
                results.append(met_fd)
        except Exception:
            pass

    # --- M4b Late Fusion: TS branch + Category-Text branch ---
    if "M2_num_tok" in feature_sets and "M4_cat_text_only" in feature_sets:
        ts_cols = feature_sets["M2_num_tok"]
        tx_cols = feature_sets["M4_cat_text_only"]
        try:
            X_ts_tr, X_ts_va, X_ts_te = _prep(train_df, ts_cols), _prep(val_df, ts_cols), _prep(test_df, ts_cols)
            X_tx_tr, X_tx_va, X_tx_te = _prep(train_df, tx_cols), _prep(val_df, tx_cols), _prep(test_df, tx_cols)

            gm_ts = train_lightgbm(X_ts_tr, y_tr, X_ts_va, y_va, lgbm_cfg)
            p_ts_va = predict_lightgbm(gm_ts, X_ts_va)
            p_ts_te = predict_lightgbm(gm_ts, X_ts_te)
            cal_ts = get_calibrator(cal_m)
            cal_ts.fit(p_ts_va, y_va)
            p_ts_va_c, p_ts_te_c = cal_ts.transform(p_ts_va), cal_ts.transform(p_ts_te)

            gm_tx = train_lightgbm(X_tx_tr, y_tr, X_tx_va, y_va, lgbm_cfg)
            p_tx_va = predict_lightgbm(gm_tx, X_tx_va)
            p_tx_te = predict_lightgbm(gm_tx, X_tx_te)
            cal_tx = get_calibrator(cal_m)
            cal_tx.fit(p_tx_va, y_va)
            p_tx_va_c, p_tx_te_c = cal_tx.transform(p_tx_va), cal_tx.transform(p_tx_te)

            fusion = get_fusion_model(cfg["fusion"]["method"])
            fusion.fit(p_ts_va_c, p_tx_va_c, y_va)
            p_fused = fusion.predict(p_ts_te_c, p_tx_te_c)

            met = compute_all_metrics(p_fused, y_te)
            met.update({"fold": split.fold_id, "variant": "M4b_cat_late_fusion", "subset": "all_days", "n": len(y_te)})
            results.append(met)

            if has_filing.sum() >= 10:
                met_fd = compute_all_metrics(p_fused[has_filing], y_te[has_filing])
                met_fd.update({"fold": split.fold_id, "variant": "M4b_cat_late_fusion", "subset": "filing_days", "n": int(has_filing.sum())})
                results.append(met_fd)
        except Exception:
            pass

    return results, importances


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Category-aware cross-modal fusion")
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
    print("=" * 90)
    print("M4: CATEGORY-AWARE CROSS-MODAL FUSION EXPERIMENT")
    print("=" * 90)
    print(f"Thesis: Multimodal Stock Prediction via Cross-Modal Fusion")
    print(f"Symbols: {len(cached_symbols)}\n")

    # --- Load data ---
    print("Loading OHLCV + computing features...")
    daily = load_universe_ohlcv(
        cfg["data"]["price_data_dir"], cached_symbols,
        cfg.get("ohlcv_columns", {}),
        start_date=cfg["data"].get("start_date"),
        end_date=cfg["data"].get("end_date"),
    )
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])
    daily = compute_chart2tokens(daily, cfg["features"])
    daily = create_labels(daily)

    # Load pooled text features for M3
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

    # Load filings for category features
    filings = pd.read_parquet(cache_dir / "filings_processed.parquet")
    filings = filings[filings["symbol"].isin(cached_symbols)].copy()
    filings["assigned_date"] = pd.to_datetime(filings["assigned_date"])

    # Build category-aware features
    print("Building category-aware cross-modal features...")
    daily, cat_feature_cols, interaction_cols = build_category_features(filings, daily)

    # --- Define feature sets ---
    W = cfg["features"]["token_summary"]["lookback_W"]
    numeric_cols = list(NUMERIC_FEATURE_COLS)
    token_cols = get_token_feature_cols(W)
    numeric_token_cols = numeric_cols + token_cols
    text_t2_cols = [c for c in TEXT_T2 if c in daily.columns]

    # M4 category-aware text features (for early fusion and late-fusion text branch)
    m4_cat_text_cols = list(dict.fromkeys(cat_feature_cols + interaction_cols))

    # M4a early fusion: numeric + tokens + category sentiment + interactions
    m4a_early_cols = list(dict.fromkeys(numeric_token_cols + cat_feature_cols + interaction_cols))

    # Verify columns exist
    for c in m4a_early_cols:
        if c not in daily.columns:
            daily[c] = 0

    feature_sets = {
        "M1_numeric": numeric_cols,
        "M2_num_tok": numeric_token_cols,
        "M3_text_only": text_t2_cols,           # text branch for M3 late fusion
        "M4a_cat_early": m4a_early_cols,         # M4a: all features in single model
        "M4_cat_text_only": m4_cat_text_cols,    # text branch for M4b late fusion
    }

    total = len(daily)
    filing = (daily["doc_count"] > 0).sum() if "doc_count" in daily.columns else 0
    print(f"\nData: {total} rows | Filing days: {filing} ({filing/total*100:.1f}%)")
    print(f"Label pos rate: {daily['label'].mean():.3f}")
    print(f"\nFeature sets:")
    for k, v in feature_sets.items():
        print(f"  {k:25s}: {len(v)} features")
    print(f"  M3_late_fusion         : TS({len(numeric_token_cols)}) + Text({len(text_t2_cols)}) -> fuse")
    print(f"  M4b_cat_late_fusion    : TS({len(numeric_token_cols)}) + CatText({len(m4_cat_text_cols)}) -> fuse")

    # --- Walk-forward splits ---
    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"], cfg["splits"]["val_months"],
        cfg["splits"]["test_months"], cfg["splits"]["step_months"],
    )
    print(f"\nWalk-forward: {len(splits)} folds\n")

    # --- Run all folds ---
    all_results = []
    all_importances = []

    for i, split in enumerate(splits):
        fold_results, fold_imps = run_fold_all_variants(daily, split, feature_sets, cfg)
        all_results.extend(fold_results)
        for vname, imp in fold_imps.items():
            imp["fold"] = split.fold_id
            imp["variant"] = vname
            all_importances.append(imp)
        if (i + 1) % 5 == 0 or i == len(splits) - 1:
            print(f"  Fold {i+1}/{len(splits)} done")

    results_df = pd.DataFrame(all_results)

    # --- Print results ---
    DISPLAY_VARIANTS = [
        "gb_M1_numeric", "lr_M1_numeric",
        "gb_M2_num_tok", "lr_M2_num_tok",
        "M3_late_fusion",
        "gb_M4a_cat_early", "lr_M4a_cat_early",
        "M4b_cat_late_fusion",
    ]
    VLABELS = {
        "gb_M1_numeric":       "GB  M1: Numeric-only",
        "lr_M1_numeric":       "LR  M1: Numeric-only",
        "gb_M2_num_tok":       "GB  M2: Numeric+Tokens",
        "lr_M2_num_tok":       "LR  M2: Numeric+Tokens",
        "M3_late_fusion":      "M3: TS+Text Late Fusion (pooled)",
        "gb_M4a_cat_early":    "GB  M4a: Cat-Aware Early Fusion",
        "lr_M4a_cat_early":    "LR  M4a: Cat-Aware Early Fusion",
        "M4b_cat_late_fusion": "M4b: Cat-Aware Late Fusion",
    }

    print(f"\n{'='*100}")
    print("RESULTS: M1 vs M2 vs M3 vs M4")
    print(f"{'='*100}")

    for subset in ["all_days", "filing_days"]:
        print(f"\n--- {subset.upper()} ---")
        print(f"  {'Variant':40s} {'AUC':>8s} {'PR-AUC':>8s} {'F1':>8s} {'Brier':>8s} {'ECE':>8s} {'N':>5s}")
        print(f"  {'-'*85}")

        for v in DISPLAY_VARIANTS:
            sub = results_df[(results_df["variant"] == v) & (results_df["subset"] == subset)]
            if sub.empty:
                continue
            vals = {m: sub[m].mean() for m in MC}
            n = sub["n"].mean()
            label = VLABELS.get(v, v)
            print(f"  {label:40s} {vals['roc_auc']:8.4f} {vals['pr_auc']:8.4f} "
                  f"{vals['f1']:8.4f} {vals['brier']:8.4f} {vals['ece']:8.4f} {n:5.0f}")

    # --- Delta metrics ---
    print(f"\n{'='*100}")
    print("DELTA vs M1 BASELINE (LightGBM, all_days)")
    print(f"{'='*100}")
    m1_gb = results_df[(results_df["variant"] == "gb_M1_numeric") & (results_df["subset"] == "all_days")]
    if not m1_gb.empty:
        m1_auc = m1_gb["roc_auc"].mean()
        m1_brier = m1_gb["brier"].mean()
        print(f"\n  {'Variant':40s} {'d_AUC':>8s} {'d_Brier':>10s}")
        print(f"  {'-'*62}")
        for v in DISPLAY_VARIANTS:
            if v == "gb_M1_numeric":
                continue
            sub = results_df[(results_df["variant"] == v) & (results_df["subset"] == "all_days")]
            if sub.empty:
                continue
            d_auc = sub["roc_auc"].mean() - m1_auc
            d_brier = sub["brier"].mean() - m1_brier
            marker = ""
            if d_auc > 0.005:
                marker = " <-- LIFT"
            elif d_auc < -0.005:
                marker = " <-- HURT"
            label = VLABELS.get(v, v)
            print(f"  {label:40s} {d_auc:+8.4f}   {d_brier:+8.4f}{marker}")

    # --- Delta on filing days ---
    print(f"\n{'='*100}")
    print("DELTA vs M1 BASELINE (LightGBM, filing_days only)")
    print(f"{'='*100}")
    m1_fd = results_df[(results_df["variant"] == "gb_M1_numeric") & (results_df["subset"] == "filing_days")]
    if not m1_fd.empty:
        m1_auc_fd = m1_fd["roc_auc"].mean()
        m1_brier_fd = m1_fd["brier"].mean()
        print(f"\n  {'Variant':40s} {'d_AUC':>8s} {'d_Brier':>10s}")
        print(f"  {'-'*62}")
        for v in DISPLAY_VARIANTS:
            if v == "gb_M1_numeric":
                continue
            sub = results_df[(results_df["variant"] == v) & (results_df["subset"] == "filing_days")]
            if sub.empty:
                continue
            d_auc = sub["roc_auc"].mean() - m1_auc_fd
            d_brier = sub["brier"].mean() - m1_brier_fd
            marker = ""
            if d_auc > 0.005:
                marker = " <-- LIFT"
            elif d_auc < -0.005:
                marker = " <-- HURT"
            label = VLABELS.get(v, v)
            print(f"  {label:40s} {d_auc:+8.4f}   {d_brier:+8.4f}{marker}")

    # --- Feature importance for M4a ---
    if all_importances:
        m4_imps = [imp for imp in all_importances if imp.get("variant") == "gb_M4a_cat_early"]
        if m4_imps:
            imp_df = pd.DataFrame(m4_imps)
            feat_cols = [c for c in imp_df.columns if c not in ("fold", "variant")]
            imp_mean = imp_df[feat_cols].mean().sort_values(ascending=False)

            print(f"\n{'='*100}")
            print("M4a FEATURE IMPORTANCE (LightGBM gain, top 25)")
            print(f"{'='*100}")
            for feat, val in imp_mean.head(25).items():
                tag = ""
                if feat.startswith("xmod_"):
                    tag = " [CROSS-MODAL]"
                elif any(feat.startswith(p) for p in ["has_", "polarity_", "p_pos_", "p_neg_"]):
                    if feat not in NUMERIC_FEATURE_COLS and feat not in token_cols:
                        tag = " [CATEGORY]"
                print(f"  {feat:35s} {val:10.1f}{tag}")

            imp_out = imp_mean.reset_index()
            imp_out.columns = ["feature", "importance"]
            if not dry_run:
                imp_out.to_csv(output_dir / "m4_feature_importance.csv", index=False)

    # --- Save ---
    summary_rows = []
    for v in results_df["variant"].unique():
        for ss in results_df["subset"].unique():
            sub = results_df[(results_df["variant"] == v) & (results_df["subset"] == ss)]
            if sub.empty:
                continue
            row = {"variant": v, "subset": ss, "n_folds": len(sub)}
            for m in MC:
                row[f"{m}_mean"] = sub[m].mean()
                row[f"{m}_std"] = sub[m].std()
            summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    if not dry_run:
        results_df.to_csv(output_dir / "m4_cross_modal_results.csv", index=False)
        summary_df.to_csv(output_dir / "m4_cross_modal_summary.csv", index=False)
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/m4_cross_modal_results.csv")
        print(f"  [DRY RUN] Would write: {output_dir}/m4_cross_modal_summary.csv")

    total_min = (time.time() - t0) / 60
    print(f"\nOutputs saved to {output_dir}/")
    print(f"Total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
