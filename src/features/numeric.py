"""Numeric time-series feature engineering (leak-safe).

All features for day t use only data available up to and including day t.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_numeric_features(
    daily: pd.DataFrame,
    cfg: Dict,
) -> pd.DataFrame:
    """Compute numeric indicators per symbol.

    Input: daily OHLCV with columns [date, open, high, low, close, volume, symbol].
    Output: same DataFrame with added feature columns.

    All rolling computations use min_periods=window to avoid partial-window leakage.
    """
    atr_w = cfg.get("atr_window", 14)
    vol_w = cfg.get("volatility_window", 20)
    ma_s = cfg.get("ma_short", 5)
    ma_l = cfg.get("ma_long", 20)
    vz_w = cfg.get("volume_zscore_window", 20)

    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()

        # 1-day log return
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # True Range and ATR
        df["_prev_close"] = df["close"].shift(1)
        df["_tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                np.abs(df["high"] - df["_prev_close"]),
                np.abs(df["low"] - df["_prev_close"]),
            ),
        )
        df["atr"] = df["_tr"].rolling(window=atr_w, min_periods=atr_w).mean()

        # Rolling volatility (std of log returns)
        df["rolling_vol"] = df["log_return"].rolling(window=vol_w, min_periods=vol_w).std()

        # MA ratio (short/long) as trend proxy
        df["ma_short"] = df["close"].rolling(window=ma_s, min_periods=ma_s).mean()
        df["ma_long"] = df["close"].rolling(window=ma_l, min_periods=ma_l).mean()
        df["ma_ratio"] = df["ma_short"] / df["ma_long"]

        # MA slope (5-day slope of ma_long, normalized)
        df["ma_long_slope"] = (df["ma_long"] - df["ma_long"].shift(5)) / (df["ma_long"].shift(5) + 1e-9)

        # Volume z-score
        vol_mean = df["volume"].rolling(window=vz_w, min_periods=vz_w).mean()
        vol_std = df["volume"].rolling(window=vz_w, min_periods=vz_w).std()
        df["volume_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

        # Cleanup temp columns
        df = df.drop(columns=["_prev_close", "_tr", "ma_short", "ma_long"])

        frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    logger.info(f"Numeric features computed: {result.shape}")
    return result


NUMERIC_FEATURE_COLS = [
    "log_return", "atr", "rolling_vol", "ma_ratio", "ma_long_slope", "volume_zscore",
]
