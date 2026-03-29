"""
Volatility Prediction Experiment (Option 1)
============================================
Instead of predicting next-day direction (binary), predict next-day
realized volatility. The hypothesis: filing sentiment affects uncertainty
even when it doesn't move price directionally.

Labels:
  - vol_hl:   log(High_{t+1} / Low_{t+1})  — next-day high-low range
  - vol_abs:  |log(Close_{t+1} / Close_t)|  — next-day absolute return

Models: LightGBM Regressor (same hyperparams, adapted for regression)
Evaluation: R², MAE, RMSE, Spearman correlation, plus a binary AUC
  for "high vol vs low vol" (above/below median) to compare with direction AUC.

Same walk-forward framework with purge/embargo. Same feature sets.
Reuses cache/features_all.parquet + cache/daily_text_features.parquet.

Usage:
    python scripts/exploratory/run_volatility_prediction.py
"""

import argparse, logging, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, setup_logging
from src.features.numeric import NUMERIC_FEATURE_COLS
from src.features.chart2tokens import get_token_feature_cols
from src.features.text_sentiment import TEXT_FEATURE_COLS
from src.splits.walk_forward import generate_walk_forward_splits, apply_purge_embargo

logger = logging.getLogger(__name__)


# ─── Volatility labels ───────────────────────────────────────────────────────

def create_volatility_labels(daily: pd.DataFrame) -> pd.DataFrame:
    """Create next-day volatility labels.

    vol_hl:  log(High_{t+1} / Low_{t+1})  — intraday range
    vol_abs: |log(Close_{t+1} / Close_t)|  — absolute return
    vol_high: binary — is next-day vol_hl above rolling 20-day median?
    """
    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        # Next-day high-low range (log scale)
        df["next_high"] = df["high"].shift(-1)
        df["next_low"] = df["low"].shift(-1)
        df["next_close"] = df["close"].shift(-1)
        df["vol_hl"] = np.log(df["next_high"] / df["next_low"])
        # Next-day absolute return
        df["vol_abs"] = np.abs(np.log(df["next_close"] / df["close"]))
        # Binary: above rolling median (expanding, shift(1) to avoid lookahead)
        rolling_med = df["vol_hl"].shift(1).expanding(min_periods=20).median()
        df["vol_high"] = (df["vol_hl"] > rolling_med).astype(float)
        # NaN last row
        df.loc[df["next_high"].isna(), ["vol_hl", "vol_abs", "vol_high"]] = np.nan
        df.drop(columns=["next_high", "next_low", "next_close"], inplace=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ─── Regression model ────────────────────────────────────────────────────────

def train_lgbm_regressor(X_train, y_train, X_val, y_val, cfg):
    """Train LightGBM regressor with same hyperparams as classifier."""
    import lightgbm as lgb

    lgb_cfg = cfg["models"]["lightgbm"]
    model = lgb.LGBMRegressor(
        n_estimators=lgb_cfg.get("n_estimators", 300),
        max_depth=lgb_cfg.get("max_depth", 4),
        learning_rate=lgb_cfg.get("learning_rate", 0.05),
        num_leaves=lgb_cfg.get("num_leaves", 15),
        min_child_samples=lgb_cfg.get("min_child_samples", 20),
        subsample=lgb_cfg.get("subsample", 0.8),
        colsample_bytree=lgb_cfg.get("colsample_bytree", 0.8),
        reg_alpha=lgb_cfg.get("reg_alpha", 0.1),
        reg_lambda=lgb_cfg.get("reg_lambda", 0.1),
        random_state=42,
        verbose=-1,
    )
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return model


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_vol_metrics(y_true, y_pred, vol_high_true=None):
    """Compute regression + ranking metrics for volatility prediction."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]
    if len(y_t) < 10:
        return {k: np.nan for k in ["r2", "mae", "rmse", "spearman", "spearman_p", "auc_vol_high"]}

    r2 = r2_score(y_t, y_p)
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    sp_corr, sp_p = spearmanr(y_t, y_p)

    # Binary AUC: can the model rank high-vol days above low-vol days?
    auc_vh = np.nan
    if vol_high_true is not None:
        vh = vol_high_true[mask]
        if len(np.unique(vh[np.isfinite(vh)])) == 2:
            vh_mask = np.isfinite(vh)
            auc_vh = roc_auc_score(vh[vh_mask], y_p[vh_mask])

    return {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "spearman": float(sp_corr),
        "spearman_p": float(sp_p),
        "auc_vol_high": float(auc_vh),
    }


# ─── Prepare arrays ─────────────────────────────────────────────────────────

def _prepare(train_df, val_df, test_df, feature_cols, label_col):
    """Extract feature/label arrays, clean NaN/Inf in features."""
    Xs, ys = [], []
    for df in [train_df, val_df, test_df]:
        X = df[feature_cols].values.astype(np.float32)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        Xs.append(X)
        ys.append(df[label_col].values.astype(np.float32))
    return Xs[0], ys[0], Xs[1], ys[1], Xs[2], ys[2]


# ─── Variant runners ─────────────────────────────────────────────────────────

def _filter_filing_days(df, filing_days_only):
    """Filter to filing days only if requested."""
    if not filing_days_only:
        return df
    if "doc_count" in df.columns:
        return df[df["doc_count"] > 0].copy()
    return df


def run_variant(name, feature_cols, daily, splits, cfg, label_col="vol_hl",
                filing_days_only=False):
    """Run a single-branch regression variant across all folds."""
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    results = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        # Drop rows with NaN labels
        train_df = train_df.dropna(subset=[label_col, "vol_high"])
        val_df = val_df.dropna(subset=[label_col, "vol_high"])
        test_df = test_df.dropna(subset=[label_col, "vol_high"])

        # Filter to filing days only
        train_df = _filter_filing_days(train_df, filing_days_only)
        val_df = _filter_filing_days(val_df, filing_days_only)
        test_df = _filter_filing_days(test_df, filing_days_only)

        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        X_tr, y_tr, X_va, y_va, X_te, y_te = _prepare(
            train_df, val_df, test_df, feature_cols, label_col
        )
        model = train_lgbm_regressor(X_tr, y_tr, X_va, y_va, cfg)
        y_pred = model.predict(X_te)

        metrics = compute_vol_metrics(
            y_te, y_pred, test_df["vol_high"].values.astype(np.float32)
        )
        metrics["fold"] = split.fold_id
        metrics["variant"] = name
        metrics["n_test"] = len(y_te)
        results.append(metrics)

    return results


def run_fusion_variant(daily, ts_cols, text_cols, splits, cfg, label_col="vol_hl",
                       filing_days_only=False):
    """Late fusion for volatility: TS branch + Text branch → average predictions."""
    purge = cfg["splits"]["purge_days"]
    embargo = cfg["splits"]["embargo_days"]
    results = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(daily, split, purge, embargo)
        train_df = train_df.dropna(subset=[label_col, "vol_high"])
        val_df = val_df.dropna(subset=[label_col, "vol_high"])
        test_df = test_df.dropna(subset=[label_col, "vol_high"])

        # Filter to filing days only
        train_df = _filter_filing_days(train_df, filing_days_only)
        val_df = _filter_filing_days(val_df, filing_days_only)
        test_df = _filter_filing_days(test_df, filing_days_only)

        if len(train_df) < 50 or len(val_df) < 10 or len(test_df) < 10:
            continue

        # TS branch
        X_ts_tr, y_tr, X_ts_va, y_va, X_ts_te, y_te = _prepare(
            train_df, val_df, test_df, ts_cols, label_col
        )
        model_ts = train_lgbm_regressor(X_ts_tr, y_tr, X_ts_va, y_va, cfg)
        pred_ts_val = model_ts.predict(X_ts_va)
        pred_ts_test = model_ts.predict(X_ts_te)

        # Text branch
        X_tx_tr, _, X_tx_va, _, X_tx_te, _ = _prepare(
            train_df, val_df, test_df, text_cols, label_col
        )
        model_tx = train_lgbm_regressor(X_tx_tr, y_tr, X_tx_va, y_va, cfg)
        pred_tx_val = model_tx.predict(X_tx_va)
        pred_tx_test = model_tx.predict(X_tx_te)

        # Learn optimal weight on validation set (grid search)
        best_w, best_mae = 0.5, np.inf
        for w in np.arange(0.0, 1.05, 0.05):
            combo = w * pred_ts_val + (1 - w) * pred_tx_val
            m = mean_absolute_error(y_va, combo)
            if m < best_mae:
                best_mae = m
                best_w = w

        y_pred = best_w * pred_ts_test + (1 - best_w) * pred_tx_test

        metrics = compute_vol_metrics(
            y_te, y_pred, test_df["vol_high"].values.astype(np.float32)
        )
        metrics["fold"] = split.fold_id
        metrics["variant"] = "ts_text_fusion"
        metrics["n_test"] = len(y_te)
        metrics["fusion_weight_ts"] = float(best_w)
        results.append(metrics)

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def build_features_from_cache(cfg, symbols):
    """Build full feature matrix from daily_ohlcv cache for given symbols."""
    from src.features.numeric import compute_numeric_features
    from src.features.chart2tokens import compute_chart2tokens

    cache_dir = Path(cfg["data"]["cache_dir"])
    daily = pd.read_parquet(cache_dir / "daily_ohlcv.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    # Filter to requested symbols
    daily = daily[daily["symbol"].isin(symbols)].copy()
    print(f"  Loaded daily OHLCV: {len(daily)} rows, {daily['symbol'].nunique()} symbols")

    # Compute numeric features
    daily = compute_numeric_features(daily, cfg["features"]["numeric"])

    # Compute Chart2Tokens
    daily = compute_chart2tokens(daily, cfg["features"])

    # Merge text features
    text_path = cache_dir / "daily_text_features.parquet"
    has_text = False
    if text_path.exists():
        text_feats = pd.read_parquet(text_path)
        text_feats["date"] = pd.to_datetime(text_feats["date"])
        text_feats = text_feats[text_feats["symbol"].isin(symbols)]
        daily = daily.merge(text_feats, on=["symbol", "date"], how="left")
        for col in TEXT_FEATURE_COLS:
            if col in daily.columns:
                daily[col] = daily[col].fillna(0)
        has_text = any(daily[c].sum() > 0 for c in TEXT_FEATURE_COLS if c in daily.columns)

    return daily, has_text


def main():
    parser = argparse.ArgumentParser(description="Volatility Prediction Experiment")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--label", default="vol_hl", choices=["vol_hl", "vol_abs"],
                        help="Which volatility label to predict")
    parser.add_argument("--filing-days-only", action="store_true",
                        help="Train and test only on days with filings")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip writing any files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    set_seed(cfg.get("seed", 42))
    t0 = time.time()

    cache_dir = Path(cfg["data"]["cache_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    dry_run = args.dry_run
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    label_col = args.label
    filing_days_only = args.filing_days_only

    # All 98 symbols with FinBERT sentiment
    import json
    cache_dir = Path(cfg["data"]["cache_dir"])
    all_symbols = json.loads((cache_dir / "completed_symbols.json").read_text())
    mode_str = "FILING DAYS ONLY" if filing_days_only else "ALL DAYS"
    print("=" * 80)
    print(f"VOLATILITY PREDICTION EXPERIMENT  (label={label_col}, {mode_str})")
    print(f"  Symbols: {len(all_symbols)}")
    print("=" * 80)

    # ── 1. Build features from daily OHLCV cache ──
    daily, has_text = build_features_from_cache(cfg, all_symbols)
    print(f"Loaded {len(daily)} rows, {daily['symbol'].nunique()} symbols")
    print(f"Text features available: {has_text}")

    # ── 2. Create volatility labels ──
    daily = create_volatility_labels(daily)
    valid = daily[label_col].notna()
    print(f"Valid volatility labels: {valid.sum()} / {len(daily)}")
    print(f"  {label_col} mean={daily.loc[valid, label_col].mean():.6f}, "
          f"std={daily.loc[valid, label_col].std():.6f}, "
          f"median={daily.loc[valid, label_col].median():.6f}")

    # ── 3. Feature sets ──
    W = cfg["features"]["token_summary"]["lookback_W"]
    numeric_cols = NUMERIC_FEATURE_COLS
    token_cols = get_token_feature_cols(W)
    text_cols = [c for c in TEXT_FEATURE_COLS if c in daily.columns]
    numeric_token_cols = numeric_cols + token_cols

    print(f"\nFeature sets: Numeric={len(numeric_cols)}, Tokens={len(token_cols)}, "
          f"Text={len(text_cols)}")

    # ── 4. Walk-forward splits ──
    splits = generate_walk_forward_splits(
        daily["date"],
        cfg["splits"]["train_months"],
        cfg["splits"]["val_months"],
        cfg["splits"]["test_months"],
        cfg["splits"]["step_months"],
    )
    print(f"Walk-forward folds: {len(splits)}")

    # ── 5. Run variants ──
    all_results = []

    if filing_days_only:
        filing_count = (daily["doc_count"] > 0).sum() if "doc_count" in daily.columns else 0
        print(f"\nFiling days in dataset: {filing_count} / {len(daily)} "
              f"({100*filing_count/len(daily):.1f}%)")

    print(f"\n>>> Variant 1/3: numeric_only")
    t1 = time.time()
    all_results.extend(run_variant("numeric_only", numeric_cols, daily, splits, cfg,
                                   label_col, filing_days_only))
    print(f"    Done in {time.time()-t1:.1f}s")

    print(f"\n>>> Variant 2/3: numeric_tokens")
    t2 = time.time()
    all_results.extend(run_variant("numeric_tokens", numeric_token_cols, daily, splits, cfg,
                                   label_col, filing_days_only))
    print(f"    Done in {time.time()-t2:.1f}s")

    if has_text:
        print(f"\n>>> Variant 3/3: ts_text_fusion (late fusion)")
        t3 = time.time()
        all_results.extend(
            run_fusion_variant(daily, numeric_token_cols, text_cols, splits, cfg,
                              label_col, filing_days_only)
        )
        print(f"    Done in {time.time()-t3:.1f}s")
    else:
        print("\n>>> Skipping fusion (no text features)")

    # ── 6. Results ──
    results_df = pd.DataFrame(all_results)
    suffix = f"_{label_col}_filing_days" if filing_days_only else f"_{label_col}"

    metric_cols = ["r2", "mae", "rmse", "spearman", "auc_vol_high"]
    summary = results_df.groupby("variant")[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]

    if not dry_run:
        results_df.to_csv(output_dir / f"volatility{suffix}_per_fold.csv", index=False)
        summary.to_csv(output_dir / f"volatility{suffix}_summary.csv")
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/volatility{suffix}_per_fold.csv")
        print(f"  [DRY RUN] Would write: {output_dir}/volatility{suffix}_summary.csv")

    # Delta metrics
    print(f"\n{'='*80}")
    print(f"VOLATILITY PREDICTION RESULTS  (label={label_col}, {len(splits)} folds)")
    print(f"{'='*80}")
    print(summary.round(4).to_string())

    # Compute deltas
    print(f"\n{'='*80}")
    print("DELTA METRICS (incremental value)")
    print(f"{'='*80}")
    for metric in metric_cols:
        nums = results_df[results_df["variant"] == "numeric_only"][metric].values
        toks = results_df[results_df["variant"] == "numeric_tokens"][metric].values
        n = min(len(nums), len(toks))
        if n > 0:
            delta = np.mean(toks[:n]) - np.mean(nums[:n])
            print(f"  Tokens vs Numeric  d_{metric}: {delta:+.4f}")

        if has_text:
            fus = results_df[results_df["variant"] == "ts_text_fusion"][metric].values
            n2 = min(len(toks), len(fus))
            if n2 > 0:
                delta2 = np.mean(fus[:n2]) - np.mean(toks[:n2])
                print(f"  Fusion vs Tokens   d_{metric}: {delta2:+.4f}")
        print()

    # Fusion weights summary
    if has_text:
        fus_rows = results_df[results_df["variant"] == "ts_text_fusion"]
        if "fusion_weight_ts" in fus_rows.columns:
            w_mean = fus_rows["fusion_weight_ts"].mean()
            print(f"  Avg fusion weight (TS branch): {w_mean:.2f}")

    total_min = (time.time() - t0) / 60
    print(f"\nOutputs saved to {output_dir}/")
    print(f"Total time: {total_min:.1f} min")


if __name__ == "__main__":
    main()
