"""Sensitivity analysis over {W, half-life h} for Chart2Tokens.

Usage:
    python scripts/sensitivity_grid.py [--config configs/default.yaml]

Runs the numeric+tokens variant across a grid of (W, h) values
and outputs a tidy results table.
"""

import argparse
import sys
import os
import logging
import copy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed, setup_logging
from src.data.ohlcv_loader import load_universe_ohlcv
from src.features.numeric import compute_numeric_features, NUMERIC_FEATURE_COLS
from src.features.chart2tokens import compute_chart2tokens, get_token_feature_cols
from src.splits.walk_forward import generate_walk_forward_splits, apply_purge_embargo
from src.models.training import train_lightgbm, predict_lightgbm
from src.models.calibration import get_calibrator
from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Sensitivity grid over W and h")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run full pipeline but skip writing any files")
    args = parser.parse_args()
    dry_run = args.dry_run

    cfg = load_config(args.config)
    setup_logging(cfg.get("log_level", "INFO"))
    set_seed(cfg.get("seed", 42))

    output_dir = Path(cfg["data"]["output_dir"])
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare base data (numeric features only, tokens recomputed per grid point)
    symbols = cfg["universe"]["symbols"]
    ohlcv_map = cfg.get("ohlcv_columns", {})
    daily_base = load_universe_ohlcv(
        cfg["data"]["price_data_dir"], symbols, ohlcv_map,
        start_date=cfg["data"].get("start_date"),
        end_date=cfg["data"].get("end_date"),
    )
    daily_base = compute_numeric_features(daily_base, cfg["features"]["numeric"])

    # Create labels
    frames = []
    for sym, gdf in daily_base.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        df["next_close"] = df["close"].shift(-1)
        df["label"] = (np.log(df["next_close"] / df["close"]) > 0).astype(float)
        df.loc[df["next_close"].isna(), "label"] = np.nan
        frames.append(df)
    daily_base = pd.concat(frames, ignore_index=True)

    W_values = cfg["sensitivity"]["W_values"]
    h_values = cfg["sensitivity"]["h_values"]

    logger.info(f"Sensitivity grid: W={W_values}, h={h_values}")
    logger.info(f"Total combinations: {len(W_values) * len(h_values)}")

    results = []

    for W in W_values:
        for h in h_values:
            logger.info(f"\n--- W={W}, h={h} ---")

            # Recompute tokens with this W and h
            feat_cfg = copy.deepcopy(cfg["features"])
            feat_cfg["token_summary"]["lookback_W"] = W
            feat_cfg["token_summary"]["half_life_h"] = h

            daily = compute_chart2tokens(daily_base.copy(), feat_cfg)
            daily = daily.dropna(subset=["label"] + NUMERIC_FEATURE_COLS).reset_index(drop=True)

            token_cols = get_token_feature_cols(W)
            feature_cols = NUMERIC_FEATURE_COLS + token_cols

            # Run one pass of walk-forward
            splits = generate_walk_forward_splits(
                daily["date"],
                cfg["splits"]["train_months"],
                cfg["splits"]["val_months"],
                cfg["splits"]["test_months"],
                cfg["splits"]["step_months"],
            )

            fold_metrics = []
            for split in splits:
                train_df, val_df, test_df = apply_purge_embargo(
                    daily, split, cfg["splits"]["purge_days"], cfg["splits"]["embargo_days"],
                )

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

                model = train_lightgbm(X_train, y_train, X_val, y_val, cfg["models"]["lightgbm"])
                p_test = predict_lightgbm(model, X_test)

                cal = get_calibrator(cfg["calibration"]["method"])
                cal.fit(predict_lightgbm(model, X_val), y_val)
                p_test_cal = cal.transform(p_test)

                metrics = compute_all_metrics(p_test_cal, y_test)
                fold_metrics.append(metrics)

            if fold_metrics:
                avg = {k: np.mean([m[k] for m in fold_metrics])
                       for k in fold_metrics[0] if isinstance(fold_metrics[0][k], (int, float))}
                avg["W"] = W
                avg["h"] = h
                avg["n_folds"] = len(fold_metrics)
                results.append(avg)
                logger.info(f"  W={W}, h={h}: AUC={avg.get('roc_auc', 0):.4f}, Brier={avg.get('brier', 0):.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    if not dry_run:
        results_df.to_csv(output_dir / "sensitivity_grid.csv", index=False)
        logger.info(f"\nSensitivity grid saved to {output_dir / 'sensitivity_grid.csv'}")
    else:
        print(f"  [DRY RUN] Would write: {output_dir}/sensitivity_grid.csv")

    print("\n" + "=" * 80)
    print("SENSITIVITY GRID RESULTS")
    print("=" * 80)
    if not results_df.empty:
        pivot = results_df.pivot_table(values="roc_auc", index="W", columns="h")
        print("\nROC-AUC by (W, h):")
        print(pivot.round(4).to_string())

        pivot_brier = results_df.pivot_table(values="brier", index="W", columns="h")
        print("\nBrier Score by (W, h):")
        print(pivot_brier.round(4).to_string())


if __name__ == "__main__":
    main()
