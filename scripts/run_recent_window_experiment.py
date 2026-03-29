"""Run ALL experiments filtered to 2024-2025 data only.

Mirrors run_thesis_experiment.py but restricts data to 2024-01-01 .. 2025-12-31.
Also includes the basic 3-variant walk-forward (M1/M2/M3) from run_cached24.py.

Outputs (in outputs/):
  - 2024_2025_pooled.csv        (pooled fold-level results, all variants x subsets)
  - 2024_2025_per_stock.csv     (per-stock results)
  - 2024_2025_summary.csv       (aggregated mean +/- std for 3-variant comparison)
  - 2024_2025_delta.csv         (incremental deltas)

Usage:
    python scripts/run_2024_2025.py
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
from src.models.fusion import get_fusion_model
from src.evaluation.metrics import (
    compute_all_metrics, build_results_table, compute_delta_metrics,
    compute_reliability_stats,
)

logger = logging.getLogger(__name__)
MC = ["roc_auc", "pr_auc", "f1", "brier", "ece"]

# Date filter
DATE_START = "2024-01-01"
DATE_END = "2025-12-31"

# T2 text features (sweet spot from ablation)
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


def eval_sub(probs, labels, min_n=10):
    if len(probs) < min_n or len(np.unique(labels)) < 2:
        return {m: np.nan for m in MC} | {"n": len(probs)}
    r = compute_all_metrics(probs, labels)
    r["n"] = len(probs)
    return r


def run_fold(data, split, modalities, cfg):
    """One walk-forward fold. Train on all data, evaluate on subsets."""
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    cal_m = cfg["calibration"]["method"]
    lr_cfg = cfg["models"]["logistic_regression"]
    lgbm_cfg = cfg["models"]["lightgbm"]

    train_df, val_df, test_df = apply_purge_embargo(data, split, purge, embargo)

    for df in [train_df, val_df, test_df]:
        if "label" not in df.columns or df["label"].isna().all():
            return []

    train_df = train_df.dropna(subset=["label"])
    val_df = val_df.dropna(subset=["label"])
    test_df = test_df.dropna(subset=["label"])

    if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
        return []
    if len(np.unique(train_df["label"].values)) < 2:
        return []

    y_tr = train_df["label"].values.astype(np.float32)
    y_va = val_df["label"].values.astype(np.float32)
    y_te = test_df["label"].values.astype(np.float32)

    has_filing = (test_df["doc_count"].values > 0) if "doc_count" in test_df.columns else np.zeros(len(test_df), dtype=bool)
    no_filing = ~has_filing
    subsets = [("all", None), ("filing", has_filing), ("no_filing", no_filing)]

    results = []
    cal_probs = {}

    def _record(variant, p_te_cal):
        for sn, mask in subsets:
            if mask is not None and mask.sum() < 5:
                continue
            if mask is not None:
                met = eval_sub(p_te_cal[mask], y_te[mask])
            else:
                met = eval_sub(p_te_cal, y_te)
            met.update({"fold": split.fold_id, "variant": variant, "subset": sn})
            results.append(met)

    def _prep(cols):
        X_tr = train_df[cols].values.astype(np.float32)
        X_va = val_df[cols].values.astype(np.float32)
        X_te = test_df[cols].values.astype(np.float32)
        for X in [X_tr, X_va, X_te]:
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X_tr, X_va, X_te

    m1 = modalities["M1_numeric"]
    m2 = modalities["M2_visual"]
    m3 = modalities["M3_text"]

    # -- Single-modal LR --
    for mname, mcols in modalities.items():
        X_tr, X_va, X_te = _prep(mcols)
        try:
            m, s = train_logistic_regression(X_tr, y_tr, lr_cfg)
            p_va = predict_logistic_regression(m, s, X_va)
            p_te = predict_logistic_regression(m, s, X_te)
            cal = get_calibrator(cal_m); cal.fit(p_va, y_va)
            cal_probs[f"lr_{mname}"] = (cal.transform(p_va), cal.transform(p_te))
            _record(f"lr_{mname}", cal.transform(p_te))
        except Exception:
            pass

    # -- Single-modal LGBM --
    for mname, mcols in modalities.items():
        X_tr, X_va, X_te = _prep(mcols)
        try:
            gm = train_lightgbm(X_tr, y_tr, X_va, y_va, lgbm_cfg)
            p_va = predict_lightgbm(gm, X_va)
            p_te = predict_lightgbm(gm, X_te)
            cal = get_calibrator(cal_m); cal.fit(p_va, y_va)
            cal_probs[f"gb_{mname}"] = (cal.transform(p_va), cal.transform(p_te))
            _record(f"gb_{mname}", cal.transform(p_te))
        except Exception:
            pass

    # -- Early fusion (concatenated features) --
    for ename, ecols in [("early_M1M2", m1+m2), ("early_all", m1+m2+m3)]:
        X_tr, X_va, X_te = _prep(ecols)
        try:
            ms = train_logistic_regression(X_tr, y_tr, lr_cfg)
            p_va = predict_logistic_regression(ms[0], ms[1], X_va)
            p_te = predict_logistic_regression(ms[0], ms[1], X_te)
            cal = get_calibrator(cal_m); cal.fit(p_va, y_va)
            _record(f"lr_{ename}", cal.transform(p_te))
        except Exception:
            pass
        try:
            gm = train_lightgbm(X_tr, y_tr, X_va, y_va, lgbm_cfg)
            p_va = predict_lightgbm(gm, X_va)
            p_te = predict_lightgbm(gm, X_te)
            cal = get_calibrator(cal_m); cal.fit(p_va, y_va)
            _record(f"gb_{ename}", cal.transform(p_te))
        except Exception:
            pass

    # -- Late fusion: 2-branch (M1+M2) weighted avg --
    if "lr_M1_numeric" in cal_probs and "lr_M2_visual" in cal_probs:
        try:
            from src.models.fusion import WeightedAverageFusion
            fus = WeightedAverageFusion()
            fus.fit(cal_probs["lr_M1_numeric"][0], cal_probs["lr_M2_visual"][0], y_va)
            _record("late_M1M2", fus.predict(cal_probs["lr_M1_numeric"][1], cal_probs["lr_M2_visual"][1]))
        except Exception:
            pass

    # -- Late fusion: 3-branch meta-LR (M1+M2+M3) --
    lr_keys = [f"lr_{mn}" for mn in modalities]
    if all(k in cal_probs for k in lr_keys):
        try:
            from sklearn.linear_model import LogisticRegression as MetaLR
            X_meta_va = np.column_stack([cal_probs[k][0] for k in lr_keys])
            X_meta_te = np.column_stack([cal_probs[k][1] for k in lr_keys])
            meta = MetaLR(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            meta.fit(X_meta_va, y_va)
            _record("late_all", meta.predict_proba(X_meta_te)[:, 1])
        except Exception:
            pass

    # -- Cross-modal: meta-LR on all 6 streams (LR x 3 + LGBM x 3) --
    all_keys = lr_keys + [f"gb_{mn}" for mn in modalities]
    if all(k in cal_probs for k in all_keys):
        try:
            from sklearn.linear_model import LogisticRegression as MetaLR
            X_meta_va = np.column_stack([cal_probs[k][0] for k in all_keys])
            X_meta_te = np.column_stack([cal_probs[k][1] for k in all_keys])
            meta = MetaLR(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            meta.fit(X_meta_va, y_va)
            _record("cross_modal", meta.predict_proba(X_meta_te)[:, 1])
        except Exception:
            pass

    return results


# -- Basic 3-variant runner (for summary table) --
def run_basic_variant(name, feature_cols, daily, splits, cfg):
    """Run numeric_only / numeric_tokens / fusion variants (like run_cached24)."""
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


def run_basic_fusion(daily, numeric_token_cols, text_cols, splits, cfg):
    """Late fusion variant (like run_cached24)."""
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

        # TS branch
        X_ts_tr = train_df[numeric_token_cols].values.astype(np.float32)
        X_ts_va = val_df[numeric_token_cols].values.astype(np.float32)
        X_ts_te = test_df[numeric_token_cols].values.astype(np.float32)
        for X in [X_ts_tr, X_ts_va, X_ts_te]:
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

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
    parser = argparse.ArgumentParser(description="2024-2025 window experiment")
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

    print("=" * 100)
    print(f"ALL EXPERIMENTS — FILTERED TO {DATE_START} .. {DATE_END}")
    print(f"Universe: {len(cached_symbols)} NSE stocks | Label: next-day direction")
    print("=" * 100)

    data_cfg = cfg["data"]
    ohlcv_map = cfg.get("ohlcv_columns", {})

    # ── Load & filter data ──
    print("\nLoading data...")
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

    pre_filter = len(daily)
    print(f"  Before filter: {pre_filter} rows, date range: {daily['date'].min().date()} to {daily['date'].max().date()}")

    # *** APPLY 2024-2025 FILTER ***
    daily = daily[(daily["date"] >= DATE_START) & (daily["date"] <= DATE_END)].reset_index(drop=True)
    print(f"  After filter:  {len(daily)} rows, date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"  Symbols: {daily['symbol'].nunique()}")
    print(f"  Label dist: {daily['label'].value_counts().to_dict()}")

    # ── Feature sets ──
    W = cfg["features"]["token_summary"]["lookback_W"]
    m1_cols = NUMERIC_FEATURE_COLS
    m2_cols = get_token_feature_cols(W)
    m3_cols = [c for c in TEXT_T2 if c in daily.columns]
    text_cols_full = [c for c in TEXT_FEATURE_COLS if c in daily.columns]
    numeric_token_cols = m1_cols + m2_cols

    modalities = {
        "M1_numeric": m1_cols,
        "M2_visual":  m2_cols,
        "M3_text":    m3_cols,
    }

    VARIANTS = [
        "lr_M1_numeric", "lr_M2_visual", "lr_M3_text",
        "gb_M1_numeric", "gb_M2_visual", "gb_M3_text",
        "lr_early_M1M2", "lr_early_all",
        "gb_early_M1M2", "gb_early_all",
        "late_M1M2", "late_all", "cross_modal",
    ]
    VL = {
        "lr_M1_numeric": "LR Numeric (M1)",
        "lr_M2_visual":  "LR Visual (M2)",
        "lr_M3_text":    "LR Text (M3)",
        "gb_M1_numeric": "LGBM Numeric (M1)",
        "gb_M2_visual":  "LGBM Visual (M2)",
        "gb_M3_text":    "LGBM Text (M3)",
        "lr_early_M1M2": "Early LR (M1+M2)",
        "lr_early_all":  "Early LR (M1+M2+M3)",
        "gb_early_M1M2": "Early GB (M1+M2)",
        "gb_early_all":  "Early GB (M1+M2+M3)",
        "late_M1M2":     "Late Fusion (M1+M2)",
        "late_all":      "Late Fusion (M1+M2+M3)",
        "cross_modal":   "Cross-Modal (6-way)",
    }
    VS = {v: v.replace("_numeric","n").replace("_visual","v").replace("_text","t")
              .replace("lr_M1","lm1").replace("lr_M2","lm2").replace("lr_M3","lm3")
              .replace("gb_M1","gm1").replace("gb_M2","gm2").replace("gb_M3","gm3")
              .replace("lr_early_M1M2","le12").replace("lr_early_all","le3")
              .replace("gb_early_M1M2","ge12").replace("gb_early_all","ge3")
              .replace("late_M1M2","lt12").replace("late_all","lt3")
              .replace("cross_modal","cx")
          for v in VARIANTS}

    filing = (daily["doc_count"] > 0).sum() if "doc_count" in daily.columns else 0
    print(f"\n  M1: {len(m1_cols)} feats | M2: {len(m2_cols)} feats | M3: {len(m3_cols)} feats")
    print(f"  Filing days: {filing} ({filing/len(daily)*100:.1f}%)")

    # ================================================================
    # PART A: BASIC 3-VARIANT COMPARISON (like run_cached24)
    # ================================================================
    print("\n" + "=" * 100)
    print("PART A: BASIC 3-VARIANT COMPARISON (LGBM walk-forward)")
    print("=" * 100)

    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"], cfg["splits"]["val_months"],
        cfg["splits"]["test_months"], cfg["splits"]["step_months"],
    )
    print(f"Walk-forward: {len(splits)} folds")

    if not splits:
        print("ERROR: No valid splits for 2024-2025 range.")
        return

    for s in splits:
        print(f"  Fold {s.fold_id}: train [{s.train_start.date()}..{s.train_end.date()}] "
              f"val [{s.val_start.date()}..{s.val_end.date()}] "
              f"test [{s.test_start.date()}..{s.test_end.date()}]")

    basic_results = []

    print("\n  >>> numeric_only")
    basic_results.extend(run_basic_variant("numeric_only", m1_cols, daily, splits, cfg))

    print("  >>> numeric_tokens")
    basic_results.extend(run_basic_variant("numeric_tokens", numeric_token_cols, daily, splits, cfg))

    has_text = any(daily[c].sum() > 0 for c in text_cols_full if c in daily.columns)
    if has_text:
        print("  >>> ts_text_fusion")
        basic_results.extend(run_basic_fusion(daily, numeric_token_cols, text_cols_full, splits, cfg))
    else:
        print("  >>> Skipping fusion (no text signal)")

    if basic_results:
        basic_df = build_results_table(basic_results)
        metric_cols = ["roc_auc", "pr_auc", "f1", "brier", "ece",
                       "reliability_slope", "reliability_intercept"]
        summary = basic_df.groupby("variant")[metric_cols].agg(["mean", "std"])
        summary.columns = ["_".join(c) for c in summary.columns]

        deltas = compute_delta_metrics(basic_df)

        if not dry_run:
            basic_df.to_csv(output_dir / "2024_2025_per_fold.csv", index=False)
            summary.to_csv(output_dir / "2024_2025_summary.csv")
            if not deltas.empty:
                deltas.to_csv(output_dir / "2024_2025_delta.csv", index=False)
        else:
            print(f"  [DRY RUN] Would write: {output_dir}/2024_2025_per_fold.csv")
            print(f"  [DRY RUN] Would write: {output_dir}/2024_2025_summary.csv")
            print(f"  [DRY RUN] Would write: {output_dir}/2024_2025_delta.csv")

        print(f"\n{'='*90}")
        print(f"3-VARIANT SUMMARY — {len(cached_symbols)} stocks, {len(splits)} folds (2024-2025)")
        print("=" * 90)
        print(summary.round(4).to_string())

        if not deltas.empty:
            print(f"\n  DELTA METRICS:")
            delta_summary = deltas.groupby("comparison").agg(
                **{col: (col, "mean") for col in deltas.columns if col.startswith("delta_")}
            )
            print(delta_summary.round(4).to_string())

    # ================================================================
    # PART B: FULL THESIS EXPERIMENT (all variants, all/filing subsets)
    # ================================================================
    print("\n" + "=" * 100)
    print("PART B: THESIS EXPERIMENT — ALL VARIANTS (2024-2025)")
    print("=" * 100)

    pooled = []
    for i, split in enumerate(splits):
        pooled.extend(run_fold(daily, split, modalities, cfg))
        if (i + 1) % 5 == 0 or i == len(splits) - 1:
            print(f"  Fold {i+1}/{len(splits)} done")

    pdf = pd.DataFrame(pooled)

    if pdf.empty:
        print("  No results from thesis experiment folds.")
    else:
        # Table: all-days vs filing-days
        print(f"\n{'':30s} {'--- ALL DAYS ---':^42s}  {'-- FILING DAYS --':^42s}")
        print(f"{'Variant':30s} {'AUC':>7s} {'PR':>7s} {'F1':>7s} {'Brier':>7s} {'ECE':>7s}  "
              f"{'AUC':>7s} {'PR':>7s} {'F1':>7s} {'Brier':>7s} {'ECE':>7s}  {'dAUC':>6s}")
        print("-" * 120)

        for v in VARIANTS:
            all_sub = pdf[(pdf["variant"] == v) & (pdf["subset"] == "all")]
            fd_sub = pdf[(pdf["variant"] == v) & (pdf["subset"] == "filing")]
            if all_sub.empty:
                continue
            a_vals = [all_sub[m].mean() for m in MC]
            f_vals = [fd_sub[m].mean() for m in MC] if not fd_sub.empty else [np.nan]*5
            d_auc = f_vals[0] - a_vals[0] if not np.isnan(f_vals[0]) else np.nan
            print(f"  {VL.get(v,v):28s} {a_vals[0]:7.4f} {a_vals[1]:7.4f} {a_vals[2]:7.4f} {a_vals[3]:7.4f} {a_vals[4]:7.4f}  "
                  f"{f_vals[0]:7.4f} {f_vals[1]:7.4f} {f_vals[2]:7.4f} {f_vals[3]:7.4f} {f_vals[4]:7.4f}  {d_auc:+6.3f}")
            if v in ["lr_M3_text", "gb_M3_text", "gb_early_all"]:
                print()

        # Key finding: filing-day deltas
        print("\n" + "=" * 100)
        print("KEY FINDING: Fusion improvement on filing days vs all days")
        print("=" * 100)
        baseline = "lr_M1_numeric"
        base_all = pdf[(pdf["variant"] == baseline) & (pdf["subset"] == "all")]
        base_fd = pdf[(pdf["variant"] == baseline) & (pdf["subset"] == "filing")]

        if not base_all.empty:
            print(f"\nBaseline: {VL[baseline]}")
            print(f"{'Variant':30s} {'dAUC(all)':>10s} {'dAUC(filing)':>13s} {'dBrier(all)':>12s} {'dBrier(filing)':>15s}")
            print("-" * 85)

            for v in VARIANTS:
                if v == baseline:
                    continue
                comp_all = pdf[(pdf["variant"] == v) & (pdf["subset"] == "all")]
                comp_fd = pdf[(pdf["variant"] == v) & (pdf["subset"] == "filing")]
                if comp_all.empty:
                    continue
                da_all = comp_all["roc_auc"].mean() - base_all["roc_auc"].mean()
                da_fd = comp_fd["roc_auc"].mean() - base_fd["roc_auc"].mean() if not comp_fd.empty and not base_fd.empty else np.nan
                db_all = comp_all["brier"].mean() - base_all["brier"].mean()
                db_fd = comp_fd["brier"].mean() - base_fd["brier"].mean() if not comp_fd.empty and not base_fd.empty else np.nan
                print(f"  {VL.get(v,v):28s} {da_all:+10.4f} {da_fd:+13.4f} {db_all:+12.4f} {db_fd:+15.4f}")

    # ================================================================
    # PART C: PER-STOCK RESULTS
    # ================================================================
    print("\n" + "=" * 100)
    print("PART C: PER-STOCK RESULTS (2024-2025)")
    print("=" * 100)

    stock_rows = []
    for idx, sym in enumerate(cached_symbols):
        sym_data = daily[daily["symbol"] == sym].copy()
        if len(sym_data) < 100:
            print(f"  [{idx+1:2d}] {sym:15s} SKIP (only {len(sym_data)} rows)")
            continue

        sym_splits = generate_walk_forward_splits(
            sym_data["date"],
            cfg["splits"]["train_months"], cfg["splits"]["val_months"],
            cfg["splits"]["test_months"], cfg["splits"]["step_months"],
        )
        if len(sym_splits) < 2:
            print(f"  [{idx+1:2d}] {sym:15s} SKIP (only {len(sym_splits)} folds)")
            continue

        sym_results = []
        for split in sym_splits:
            sym_results.extend(run_fold(sym_data, split, modalities, cfg))
        if not sym_results:
            continue

        sdf = pd.DataFrame(sym_results)
        n_fd = (sym_data["doc_count"] > 0).sum() if "doc_count" in sym_data.columns else 0
        row = {"symbol": sym, "n_days": len(sym_data), "n_filing": n_fd, "n_folds": len(sym_splits)}

        for v in VARIANTS:
            vs = VS[v]
            for ss in ["all", "filing"]:
                sub = sdf[(sdf["variant"] == v) & (sdf["subset"] == ss)]
                for m in MC:
                    row[f"{m}_{vs}_{ss}"] = sub[m].mean() if not sub.empty else np.nan

        stock_rows.append(row)
        baseline = "lr_M1_numeric"
        a_m1 = row.get(f"roc_auc_{VS[baseline]}_all", np.nan)
        a_cx = row.get(f"roc_auc_{VS['cross_modal']}_all", np.nan)
        a_m1f = row.get(f"roc_auc_{VS[baseline]}_filing", np.nan)
        a_cxf = row.get(f"roc_auc_{VS['cross_modal']}_filing", np.nan)
        print(f"  [{idx+1:2d}] {sym:15s} days={len(sym_data):4d} folds={len(sym_splits):2d} "
              f"AUC(all): M1={a_m1:.3f} CX={a_cx:.3f} | "
              f"AUC(filing): M1={a_m1f:.3f} CX={a_cxf:.3f}")

    if stock_rows:
        sdf_all = pd.DataFrame(stock_rows)
        n = len(sdf_all)

        # Mean metrics
        print(f"\n  Mean metrics across {n} stocks (ALL days):")
        key_variants = [baseline, "lr_early_M1M2", "lr_early_all", "late_M1M2", "late_all", "cross_modal"]
        print(f"  {'Variant':30s} {'AUC':>8s} {'PR-AUC':>8s} {'F1':>8s} {'Brier':>8s}")
        print(f"  {'-'*60}")
        for v in key_variants:
            vs = VS[v]
            vals = [sdf_all[f"{m}_{vs}_all"].mean() if f"{m}_{vs}_all" in sdf_all.columns else np.nan for m in MC[:4]]
            print(f"  {VL.get(v,v):30s} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f}")

        print(f"\n  Mean metrics across {n} stocks (FILING days):")
        print(f"  {'Variant':30s} {'AUC':>8s} {'PR-AUC':>8s} {'F1':>8s} {'Brier':>8s}")
        print(f"  {'-'*60}")
        for v in key_variants:
            vs = VS[v]
            vals = [sdf_all[f"{m}_{vs}_filing"].mean() if f"{m}_{vs}_filing" in sdf_all.columns else np.nan for m in MC[:4]]
            print(f"  {VL.get(v,v):30s} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f}")

        # Save
        if not dry_run:
            sdf_all.to_csv(output_dir / "2024_2025_per_stock.csv", index=False)
        else:
            print(f"  [DRY RUN] Would write: {output_dir}/2024_2025_per_stock.csv")

    if not pdf.empty:
        if not dry_run:
            pdf.to_csv(output_dir / "2024_2025_pooled.csv", index=False)
        else:
            print(f"  [DRY RUN] Would write: {output_dir}/2024_2025_pooled.csv")

    total_min = (time.time() - t0) / 60
    print(f"\nTotal time: {total_min:.1f} min")
    print(f"Saved: {output_dir}/2024_2025_*.csv")


if __name__ == "__main__":
    main()
