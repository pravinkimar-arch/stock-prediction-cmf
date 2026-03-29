"""Milestone Attraction Feature Experiment.

Tests whether continuous "milestone attraction" features improve prediction.
Unlike the binary round_touch event, these capture:
  - Distance to nearest milestone (normalized)
  - Direction of movement toward/away from milestone
  - Gravitational pull (stronger when closer)
  - Crossing frequency in lookback window

Milestones: round numbers at multiple scales (50, 100, 500, 1000)

Usage:
    python scripts/run_milestone_attraction.py
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

MILESTONE_STEPS = [50, 100, 500, 1000]


def compute_milestone_features(daily, lookback=20):
    """Compute milestone attraction features per symbol.

    Features (per milestone scale):
      - ms_{step}_dist:      normalized distance to nearest milestone (dist / ATR), shifted
      - ms_{step}_gravity:   exp(-dist/ATR), stronger when closer, shifted
      - ms_{step}_direction: +1 if closing toward milestone, -1 if away, shifted
      - ms_{step}_cross_{lookback}: count of milestone crossings in lookback window, shifted

    Aggregate features:
      - ms_min_dist:      min distance across all scales (normalized)
      - ms_max_gravity:   max gravity across all scales
      - ms_cross_total:   total crossings across all scales
      - ms_squeeze:       price between two milestones, distance to both < ATR
    """
    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        close = df["close"].values
        atr = df["atr"].values

        for step in MILESTONE_STEPS:
            # Distance to nearest milestone (use previous day's close - shift(1))
            prev_close = np.roll(close, 1)
            prev_close[0] = np.nan

            remainder = prev_close % step
            dist = np.minimum(remainder, step - remainder)

            # Normalize by ATR
            prev_atr = np.roll(atr, 1)
            prev_atr[0] = np.nan
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_dist = np.where(prev_atr > 0, dist / prev_atr, np.nan)

            df[f"ms_{step}_dist"] = norm_dist

            # Gravity: exp(-norm_dist), capped
            df[f"ms_{step}_gravity"] = np.exp(-np.clip(norm_dist, 0, 10))

            # Direction: is today's close moving toward or away from nearest milestone?
            # Use shift(1) for previous close, shift(2) for the one before
            prev2_close = np.roll(close, 2)
            prev2_close[:2] = np.nan
            prev_remainder = prev2_close % step
            prev_dist = np.minimum(prev_remainder, step - prev_remainder)
            # direction = +1 if getting closer, -1 if getting farther
            direction = np.sign(prev_dist - dist)
            df[f"ms_{step}_direction"] = direction

            # Crossing count in lookback window
            # A crossing occurs when floor(close/step) changes
            milestone_level = np.floor(prev_close / step)
            crossing = (milestone_level != np.roll(milestone_level, 1)).astype(float)
            crossing[0] = 0
            cross_series = pd.Series(crossing)
            df[f"ms_{step}_cross_{lookback}"] = cross_series.shift(1).rolling(
                lookback, min_periods=lookback
            ).sum().values

        # Aggregate features
        dist_cols = [f"ms_{s}_dist" for s in MILESTONE_STEPS]
        grav_cols = [f"ms_{s}_gravity" for s in MILESTONE_STEPS]
        cross_cols = [f"ms_{s}_cross_{lookback}" for s in MILESTONE_STEPS]

        df["ms_min_dist"] = df[dist_cols].min(axis=1)
        df["ms_max_gravity"] = df[grav_cols].max(axis=1)
        df["ms_cross_total"] = df[cross_cols].sum(axis=1)

        # Squeeze: close is within 1 ATR of milestones on both sides
        prev_close_s = df["close"].shift(1)
        prev_atr_s = df["atr"].shift(1)
        above = np.ceil(prev_close_s / 100) * 100
        below = np.floor(prev_close_s / 100) * 100
        squeeze = ((above - prev_close_s) < prev_atr_s) & ((prev_close_s - below) < prev_atr_s)
        df["ms_squeeze"] = squeeze.astype(float)

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def get_milestone_feature_cols(lookback=20):
    """Return list of milestone feature column names."""
    cols = []
    for step in MILESTONE_STEPS:
        cols.extend([
            f"ms_{step}_dist",
            f"ms_{step}_gravity",
            f"ms_{step}_direction",
            f"ms_{step}_cross_{lookback}",
        ])
    cols.extend(["ms_min_dist", "ms_max_gravity", "ms_cross_total", "ms_squeeze"])
    return cols


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

    # Add milestone features
    daily = compute_milestone_features(daily, lookback=20)

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


def eval_single(daily, cols, splits, cfg, model_type="lgbm"):
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

    return {m: np.mean([f[m] for f in fold_metrics]) for m in MC} if fold_metrics else {m: np.nan for m in MC}


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

        if not ok or len(all_va) < len(branches) * 2:
            continue
        try:
            meta = MetaLR(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            meta.fit(np.column_stack(all_va), y_va)
            p = meta.predict_proba(np.column_stack(all_te))[:, 1]
            fold_metrics.append(compute_all_metrics(p, y_te))
        except: pass

    return {m: np.mean([f[m] for f in fold_metrics]) for m in MC} if fold_metrics else {m: np.nan for m in MC}


def get_importance(daily, feature_cols, cfg, splits):
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    importance = np.zeros(len(feature_cols))
    n = 0
    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        if len(train_df) < 50 or len(val_df) < 10: continue
        X_tr = train_df[feature_cols].values.astype(np.float32)
        y_tr = train_df["label"].values.astype(np.float32)
        X_va = val_df[feature_cols].values.astype(np.float32)
        y_va = val_df["label"].values.astype(np.float32)
        np.nan_to_num(X_tr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(X_va, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        m = train_lightgbm(X_tr, y_tr, X_va, y_va, cfg["models"]["lightgbm"])
        importance += m.feature_importances_
        n += 1
    if n > 0: importance /= n
    return dict(zip(feature_cols, importance))


def run_period(daily, period, cfg):
    W = cfg["features"]["token_summary"]["lookback_W"]
    m1_cols = list(NUMERIC_FEATURE_COLS)
    m2_cols = get_token_feature_cols(W)
    m3_cols = [c for c in TEXT_T2 if c in daily.columns]
    ms_cols = [c for c in get_milestone_feature_cols(20) if c in daily.columns]
    top3_m1 = ["rolling_vol", "ma_ratio", "ma_long_slope"]

    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"], cfg["splits"]["val_months"],
        cfg["splits"]["test_months"], cfg["splits"]["step_months"],
    )
    print(f"  {len(daily)} rows, {daily['symbol'].nunique()} symbols, {len(splits)} folds")
    print(f"  Milestone features: {len(ms_cols)}")

    # Feature importance for milestone features
    print("\n  Milestone feature importance (LGBM):")
    all_feats = m1_cols + ms_cols
    imp = get_importance(daily, all_feats, cfg, splits)
    ms_imp = {k: v for k, v in imp.items() if k.startswith("ms_")}
    for f, v in sorted(ms_imp.items(), key=lambda x: -x[1]):
        bar = "*" * int(v / max(max(ms_imp.values(), default=1), 1) * 20)
        print(f"    {f:30s} {v:8.1f}  {bar}")

    # Top milestone features
    sorted_ms = sorted(ms_imp.items(), key=lambda x: -x[1])
    top5_ms = [f for f, _ in sorted_ms[:5]]
    top8_ms = [f for f, _ in sorted_ms[:8]]
    top10_ms = [f for f, _ in sorted_ms[:10]]
    nonzero_ms = [f for f, v in sorted_ms if v > 0]

    results = []

    configs = [
        # Baselines
        ("LR M1 full",                 m1_cols, "lr"),
        ("LR M1 top3",                 top3_m1, "lr"),
        ("LGBM M1 full",               m1_cols, "lgbm"),
        ("LGBM M1+M2+M3 full",         m1_cols + m2_cols + m3_cols, "lgbm"),

        # Milestone only
        ("LR milestone only",          ms_cols, "lr"),
        ("LGBM milestone only",        ms_cols, "lgbm"),

        # M1 + milestone
        ("LR M1 + milestone full",     m1_cols + ms_cols, "lr"),
        ("LGBM M1 + milestone full",   m1_cols + ms_cols, "lgbm"),
        ("LR M1 + milestone top5",     m1_cols + top5_ms, "lr"),
        ("LGBM M1 + milestone top5",   m1_cols + top5_ms, "lgbm"),
        ("LR M1 + milestone top8",     m1_cols + top8_ms, "lr"),
        ("LGBM M1 + milestone top8",   m1_cols + top8_ms, "lgbm"),
        ("LR M1top3 + milestone top5", top3_m1 + top5_ms, "lr"),
        ("LGBM M1top3 + milestone top5", top3_m1 + top5_ms, "lgbm"),

        # M1 + milestone + text
        ("LR M1 + ms_top5 + M3",       m1_cols + top5_ms + m3_cols, "lr"),
        ("LGBM M1 + ms_top5 + M3",     m1_cols + top5_ms + m3_cols, "lgbm"),

        # M1 + milestone + M2 + M3 (kitchen sink)
        ("LGBM all + milestone",        m1_cols + m2_cols + m3_cols + ms_cols, "lgbm"),
        ("LR all + milestone",          m1_cols + m2_cols + m3_cols + ms_cols, "lr"),
    ]

    print(f"\n  {'Config':40s} {'Model':>5s} {'#Feat':>5s} {'AUC':>7s} {'Brier':>7s} {'dAUC':>7s}")
    print("  " + "-" * 80)

    baseline_lr = None
    baseline_lgbm = None

    for name, cols, mt in configs:
        met = eval_single(daily, cols, splits, cfg, mt)
        results.append({"config": name, "model": mt, "n_features": len(cols), **met})

        if name == "LR M1 full": baseline_lr = met["roc_auc"]
        if name == "LGBM M1 full": baseline_lgbm = met["roc_auc"]

        base = baseline_lr if mt == "lr" else baseline_lgbm
        d = met["roc_auc"] - base if base else np.nan
        print(f"  {name:40s} {mt:>5s} {len(cols):5d} {met['roc_auc']:7.4f} {met['brier']:7.4f} {d:+7.4f}")

    # Cross-modal with milestone as a branch
    print(f"\n  Cross-modal with milestone branch:")
    print(f"  {'Config':50s} {'AUC':>7s} {'Brier':>7s} {'dAUC':>7s}")
    print("  " + "-" * 70)

    cx_configs = [
        ("CX: M1 | milestone",                [m1_cols, ms_cols]),
        ("CX: M1 | milestone_top5",            [m1_cols, top5_ms]),
        ("CX: M1 | M3",                        [m1_cols, m3_cols]),
        ("CX: M1 | milestone + M3",            [m1_cols, ms_cols + m3_cols]),
        ("CX: M1 | milestone_top5 + M3",       [m1_cols, top5_ms + m3_cols]),
        ("CX: M1 | milestone | M3",            [m1_cols, ms_cols, m3_cols]),
        ("CX: M1 | milestone_top5 | M3",       [m1_cols, top5_ms, m3_cols]),
        ("CX: M1 | M2 | M3 (original)",        [m1_cols, m2_cols, m3_cols]),
        ("CX: M1 | milestone | M2 | M3",       [m1_cols, ms_cols, m2_cols, m3_cols]),
        ("CX: M1 | milestone+M2 | M3",         [m1_cols, ms_cols + m2_cols, m3_cols]),
    ]

    for name, branches in cx_configs:
        met = eval_crossmodal(daily, branches, splits, cfg)
        n = sum(len(b) for b in branches)
        d = met["roc_auc"] - baseline_lr if baseline_lr else np.nan
        results.append({"config": name, "model": "cx", "n_features": n, **met})
        print(f"  {name:50s} {met['roc_auc']:7.4f} {met['brier']:7.4f} {d:+7.4f}")

    return pd.DataFrame(results)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Milestone attraction experiment")
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
        print("\n" + "=" * 110)
        print(f"  MILESTONE ATTRACTION EXPERIMENT — {period} ({start} to {end})")
        print("=" * 110)
        daily = load_data(cfg, start, end)
        df = run_period(daily, period, cfg)
        df["period"] = period
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    if not dry_run:
        combined.to_csv(output_dir / "milestone_attraction_results.csv", index=False)
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/milestone_attraction_results.csv")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    if not dry_run:
        print(f"Saved: {output_dir}/milestone_attraction_results.csv")


if __name__ == "__main__":
    main()
