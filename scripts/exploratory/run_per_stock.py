"""Per-stock experiment: train each symbol independently, check hypothesis.

Hypothesis: numeric_only < numeric_tokens < fusion (on ROC-AUC).
For per-stock models with ~460 rows, we use logistic regression (LightGBM overfits).

Usage:
    python scripts/run_per_stock.py
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
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def create_labels(daily):
    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        df["next_close"] = df["close"].shift(-1)
        df["label"] = (np.log(df["next_close"] / df["close"]) > 0).astype(float)
        df.loc[df["next_close"].isna(), "label"] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)



def run_single_variant(name, feature_cols, sym_data, splits, cfg):
    """Run variant for a single stock. Returns dict of mean test metrics across folds."""
    cal_method = cfg["calibration"]["method"]
    purge_days = cfg["splits"]["purge_days"]
    embargo_days = cfg["splits"]["embargo_days"]
    fold_metrics = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(sym_data, split, purge_days, embargo_days)
        if len(train_df) < 30 or len(val_df) < 10 or len(test_df) < 10:
            continue
        # Need both classes in train and test
        if len(np.unique(train_df["label"].values)) < 2:
            continue
        if len(np.unique(test_df["label"].values)) < 2:
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
            model, scaler = train_logistic_regression(X_tr, y_tr, cfg["models"]["logistic_regression"])
            p_va = predict_logistic_regression(model, scaler, X_va)
            p_te = predict_logistic_regression(model, scaler, X_te)

            cal = get_calibrator(cal_method)
            cal.fit(p_va, y_va)
            p_te_cal = cal.transform(p_te)

            m = compute_all_metrics(p_te_cal, y_te)
            if not np.isnan(m["roc_auc"]):
                fold_metrics.append(m)
        except Exception:
            continue

    if not fold_metrics:
        return {"roc_auc": np.nan, "brier": np.nan, "f1": np.nan, "ece": np.nan}
    return {
        "roc_auc": np.mean([m["roc_auc"] for m in fold_metrics]),
        "brier": np.mean([m["brier"] for m in fold_metrics]),
        "f1": np.mean([m["f1"] for m in fold_metrics]),
        "ece": np.mean([m["ece"] for m in fold_metrics]),
    }


def run_single_fusion(sym_data, nt_cols, text_cols, splits, cfg):
    """Run fusion for a single stock. Returns dict of mean test metrics across folds."""
    cal_method = cfg["calibration"]["method"]
    fusion_method = cfg["fusion"]["method"]
    purge_days = cfg["splits"]["purge_days"]
    embargo_days = cfg["splits"]["embargo_days"]
    fold_metrics = []

    for split in splits:
        train_df, val_df, test_df = apply_purge_embargo(sym_data, split, purge_days, embargo_days)
        if len(train_df) < 30 or len(val_df) < 10 or len(test_df) < 10:
            continue
        if len(np.unique(train_df["label"].values)) < 2:
            continue
        if len(np.unique(test_df["label"].values)) < 2:
            continue

        y_tr = train_df["label"].values.astype(np.float32)
        y_va = val_df["label"].values.astype(np.float32)
        y_te = test_df["label"].values.astype(np.float32)

        try:
            # TS branch
            X_ts_tr = train_df[nt_cols].values.astype(np.float32)
            X_ts_va = val_df[nt_cols].values.astype(np.float32)
            X_ts_te = test_df[nt_cols].values.astype(np.float32)
            for X in [X_ts_tr, X_ts_va, X_ts_te]:
                np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

            ts_m, ts_s = train_logistic_regression(X_ts_tr, y_tr, cfg["models"]["logistic_regression"])
            p_ts_va = predict_logistic_regression(ts_m, ts_s, X_ts_va)
            p_ts_te = predict_logistic_regression(ts_m, ts_s, X_ts_te)
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

            tx_m, tx_s = train_logistic_regression(X_tx_tr, y_tr, cfg["models"]["logistic_regression"])
            p_tx_va = predict_logistic_regression(tx_m, tx_s, X_tx_va)
            p_tx_te = predict_logistic_regression(tx_m, tx_s, X_tx_te)
            tx_cal = get_calibrator(cal_method)
            tx_cal.fit(p_tx_va, y_va)
            p_tx_va_c = tx_cal.transform(p_tx_va)
            p_tx_te_c = tx_cal.transform(p_tx_te)

            # Fuse
            fusion = get_fusion_model(fusion_method)
            fusion.fit(p_ts_va_c, p_tx_va_c, y_va)
            p_fused = fusion.predict(p_ts_te_c, p_tx_te_c)

            m = compute_all_metrics(p_fused, y_te)
            if not np.isnan(m["roc_auc"]):
                fold_metrics.append(m)
        except Exception:
            continue

    if not fold_metrics:
        return {"roc_auc": np.nan, "brier": np.nan, "f1": np.nan, "ece": np.nan}
    return {
        "roc_auc": np.mean([m["roc_auc"] for m in fold_metrics]),
        "brier": np.mean([m["brier"] for m in fold_metrics]),
        "f1": np.mean([m["f1"] for m in fold_metrics]),
        "ece": np.mean([m["ece"] for m in fold_metrics]),
    }



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Per-stock analysis")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip writing any files")
    args = parser.parse_args()
    dry_run = args.dry_run

    cfg = load_config("configs/default.yaml")
    setup_logging("WARNING")  # quiet
    set_seed(42)
    t0 = time.time()

    cache_dir = Path(cfg["data"]["cache_dir"])
    output_dir = Path(cfg["data"]["output_dir"])
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    cached_symbols = json.loads((cache_dir / "completed_symbols.json").read_text())
    print(f"Per-stock analysis on {len(cached_symbols)} symbols\n")

    data_cfg = cfg["data"]
    ohlcv_map = cfg.get("ohlcv_columns", {})

    # Load all data once
    print("Loading data...")
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

    W = cfg["features"]["token_summary"]["lookback_W"]
    numeric_cols = NUMERIC_FEATURE_COLS
    token_cols = get_token_feature_cols(W)
    text_cols = [c for c in TEXT_FEATURE_COLS if c in daily.columns]
    nt_cols = numeric_cols + token_cols

    print(f"Data ready: {len(daily)} rows, {daily['symbol'].nunique()} symbols")
    print(f"Features: numeric={len(numeric_cols)}, tokens={len(token_cols)}, text={len(text_cols)}")

    # Per-stock walk-forward
    rows = []
    for i, sym in enumerate(cached_symbols):
        sym_data = daily[daily["symbol"] == sym].copy()
        n_days = len(sym_data)
        n_filings = int(sym_data["doc_count"].sum()) if "doc_count" in sym_data.columns else 0

        if n_days < 100:
            print(f"  [{i+1}/{len(cached_symbols)}] {sym:15s} — skipped (only {n_days} days)")
            continue

        # Generate splits for this stock's date range
        splits = generate_walk_forward_splits(
            sym_data["date"],
            cfg["splits"]["train_months"],
            cfg["splits"]["val_months"],
            cfg["splits"]["test_months"],
            cfg["splits"]["step_months"],
        )

        if len(splits) < 2:
            print(f"  [{i+1}/{len(cached_symbols)}] {sym:15s} — skipped (<2 folds)")
            continue

        m_num = run_single_variant("numeric_only", numeric_cols, sym_data, splits, cfg)
        m_nt = run_single_variant("numeric_tokens", nt_cols, sym_data, splits, cfg)
        m_fus = run_single_fusion(sym_data, nt_cols, text_cols, splits, cfg)

        auc_num, auc_nt, auc_fus = m_num["roc_auc"], m_nt["roc_auc"], m_fus["roc_auc"]

        # Check hypothesis: numeric < tokens < fusion
        tok_beats_num = (not np.isnan(auc_nt) and not np.isnan(auc_num) and auc_nt > auc_num)
        fus_beats_tok = (not np.isnan(auc_fus) and not np.isnan(auc_nt) and auc_fus > auc_nt)
        full_hypothesis = tok_beats_num and fus_beats_tok

        flag = "YES" if full_hypothesis else ("~ partial" if (tok_beats_num or fus_beats_tok) else "no")

        def _safe_delta(a, b):
            return a - b if not (np.isnan(a) or np.isnan(b)) else np.nan

        rows.append({
            "symbol": sym, "n_days": n_days, "n_filings": n_filings, "n_folds": len(splits),
            "auc_numeric": auc_num, "auc_tokens": auc_nt, "auc_fusion": auc_fus,
            "f1_numeric": m_num["f1"], "f1_tokens": m_nt["f1"], "f1_fusion": m_fus["f1"],
            "brier_numeric": m_num["brier"], "brier_tokens": m_nt["brier"], "brier_fusion": m_fus["brier"],
            "ece_numeric": m_num["ece"], "ece_tokens": m_nt["ece"], "ece_fusion": m_fus["ece"],
            "delta_auc_tok_num": _safe_delta(auc_nt, auc_num),
            "delta_auc_fus_tok": _safe_delta(auc_fus, auc_nt),
            "delta_f1_tok_num": _safe_delta(m_nt["f1"], m_num["f1"]),
            "delta_f1_fus_tok": _safe_delta(m_fus["f1"], m_nt["f1"]),
            "delta_brier_tok_num": _safe_delta(m_nt["brier"], m_num["brier"]),
            "delta_brier_fus_tok": _safe_delta(m_fus["brier"], m_nt["brier"]),
            "delta_ece_tok_num": _safe_delta(m_nt["ece"], m_num["ece"]),
            "delta_ece_fus_tok": _safe_delta(m_fus["ece"], m_nt["ece"]),
            "tok_beats_num": tok_beats_num,
            "fus_beats_tok": fus_beats_tok,
            "full_hypothesis": full_hypothesis,
        })

        print(
            f"  [{i+1}/{len(cached_symbols)}] {sym:15s} "
            f"AUC: num={auc_num:.4f} tok={auc_nt:.4f} fus={auc_fus:.4f} "
            f"| F1: num={m_num['f1']:.3f} tok={m_nt['f1']:.3f} fus={m_fus['f1']:.3f} "
            f"| Brier: num={m_num['brier']:.4f} tok={m_nt['brier']:.4f} fus={m_fus['brier']:.4f} "
            f"| {flag}"
        )

    # Build results table
    df = pd.DataFrame(rows)
    df = df.sort_values("auc_fusion", ascending=False).reset_index(drop=True)
    if not dry_run:
        df.to_csv(output_dir / "per_stock_results.csv", index=False)
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/per_stock_results.csv")

    # Print summary
    total = len(df)
    n_full = df["full_hypothesis"].sum()
    n_tok = df["tok_beats_num"].sum()
    n_fus = df["fus_beats_tok"].sum()

    print("\n" + "=" * 160)
    print("PER-STOCK RESULTS (sorted by fusion AUC)")
    print("=" * 160)
    print(f"{'Symbol':15s} {'Days':>5s} {'Folds':>5s} "
          f"{'AUC_num':>8s} {'AUC_tok':>8s} {'AUC_fus':>8s} "
          f"{'F1_num':>7s} {'F1_tok':>7s} {'F1_fus':>7s} "
          f"{'Bri_num':>8s} {'Bri_tok':>8s} {'Bri_fus':>8s} "
          f"{'ECE_num':>8s} {'ECE_tok':>8s} {'ECE_fus':>8s} "
          f"{'Hyp':>6s}")
    print("-" * 160)

    for _, r in df.iterrows():
        flag = "FULL" if r["full_hypothesis"] else ("~ part" if (r["tok_beats_num"] or r["fus_beats_tok"]) else "no")
        print(
            f"{r['symbol']:15s} {r['n_days']:5d} {r['n_folds']:5d} "
            f"{r['auc_numeric']:8.4f} {r['auc_tokens']:8.4f} {r['auc_fusion']:8.4f} "
            f"{r['f1_numeric']:7.3f} {r['f1_tokens']:7.3f} {r['f1_fusion']:7.3f} "
            f"{r['brier_numeric']:8.4f} {r['brier_tokens']:8.4f} {r['brier_fusion']:8.4f} "
            f"{r['ece_numeric']:8.4f} {r['ece_tokens']:8.4f} {r['ece_fusion']:8.4f} "
            f"{flag:>6s}"
        )

    print("-" * 160)
    print(f"\nHypothesis check (numeric < tokens < fusion on ROC-AUC):")
    print(f"  Tokens beat numeric:     {n_tok}/{total} stocks ({n_tok/total*100:.0f}%)")
    print(f"  Fusion beats tokens:     {n_fus}/{total} stocks ({n_fus/total*100:.0f}%)")
    print(f"  Full hypothesis holds:   {n_full}/{total} stocks ({n_full/total*100:.0f}%)")

    # F1 improvement counts
    f1_tok_better = (df["delta_f1_tok_num"] > 0).sum()
    f1_fus_better = (df["delta_f1_fus_tok"] > 0).sum()
    # Brier improvement = lower is better, so negative delta is better
    brier_tok_better = (df["delta_brier_tok_num"] < 0).sum()
    brier_fus_better = (df["delta_brier_fus_tok"] < 0).sum()
    # ECE improvement = lower is better
    ece_tok_better = (df["delta_ece_tok_num"] < 0).sum()
    ece_fus_better = (df["delta_ece_fus_tok"] < 0).sum()

    print(f"\nF1 improvement (higher = better):")
    print(f"  Tokens beat numeric:     {f1_tok_better}/{total} stocks ({f1_tok_better/total*100:.0f}%)")
    print(f"  Fusion beats tokens:     {f1_fus_better}/{total} stocks ({f1_fus_better/total*100:.0f}%)")
    print(f"  Mean F1:  numeric={df['f1_numeric'].mean():.4f}  tokens={df['f1_tokens'].mean():.4f}  fusion={df['f1_fusion'].mean():.4f}")

    print(f"\nBrier improvement (lower = better):")
    print(f"  Tokens beat numeric:     {brier_tok_better}/{total} stocks ({brier_tok_better/total*100:.0f}%)")
    print(f"  Fusion beats tokens:     {brier_fus_better}/{total} stocks ({brier_fus_better/total*100:.0f}%)")
    print(f"  Mean Brier: numeric={df['brier_numeric'].mean():.4f}  tokens={df['brier_tokens'].mean():.4f}  fusion={df['brier_fusion'].mean():.4f}")

    print(f"\nECE improvement (lower = better):")
    print(f"  Tokens beat numeric:     {ece_tok_better}/{total} stocks ({ece_tok_better/total*100:.0f}%)")
    print(f"  Fusion beats tokens:     {ece_fus_better}/{total} stocks ({ece_fus_better/total*100:.0f}%)")
    print(f"  Mean ECE:  numeric={df['ece_numeric'].mean():.4f}  tokens={df['ece_tokens'].mean():.4f}  fusion={df['ece_fusion'].mean():.4f}")

    if n_full > 0:
        winners = df[df["full_hypothesis"]]
        print(f"\n  Stocks where full hypothesis holds (AUC):")
        for _, r in winners.iterrows():
            print(f"    {r['symbol']:15s} AUC: {r['auc_numeric']:.4f} -> {r['auc_tokens']:.4f} -> {r['auc_fusion']:.4f} "
                  f"| F1: {r['f1_numeric']:.3f} -> {r['f1_tokens']:.3f} -> {r['f1_fusion']:.3f} "
                  f"| Brier: {r['brier_numeric']:.4f} -> {r['brier_tokens']:.4f} -> {r['brier_fusion']:.4f}")

    # Also show partial winners
    partial_tok = df[df["tok_beats_num"] & ~df["full_hypothesis"]]
    partial_fus = df[df["fus_beats_tok"] & ~df["full_hypothesis"]]
    if len(partial_tok) > 0:
        print(f"\n  Stocks where only tokens beat numeric (AUC) but fusion doesn't:")
        for _, r in partial_tok.iterrows():
            print(f"    {r['symbol']:15s} AUC: {r['auc_numeric']:.4f} -> {r['auc_tokens']:.4f} -> {r['auc_fusion']:.4f} "
                  f"| F1: {r['f1_numeric']:.3f} -> {r['f1_tokens']:.3f} -> {r['f1_fusion']:.3f}")
    if len(partial_fus) > 0:
        print(f"\n  Stocks where only fusion beats tokens (AUC) but tokens don't beat numeric:")
        for _, r in partial_fus.iterrows():
            print(f"    {r['symbol']:15s} AUC: {r['auc_numeric']:.4f} -> {r['auc_tokens']:.4f} -> {r['auc_fusion']:.4f} "
                  f"| F1: {r['f1_numeric']:.3f} -> {r['f1_tokens']:.3f} -> {r['f1_fusion']:.3f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print(f"Saved: {output_dir / 'per_stock_results.csv'}")


if __name__ == "__main__":
    main()
