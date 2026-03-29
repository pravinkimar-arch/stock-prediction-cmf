"""Intraday reaction features: first-hour market reaction on filing days.

Computes features from the first hour of trading (9:15-10:15 IST):
- First-hour return (open to 10:15 close)
- First-hour range / ATR (normalized volatility)
- First-hour volume ratio vs 20-day average first-hour volume
- First-hour high/low position (where did 10:15 close land in the range)
- Gap from previous close to today's open
- Chart2Token-style binary signals for the first hour

Label: direction from 10:15 price to EOD close.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIRST_HOUR_FEATURE_COLS = [
    "fh_return",           # log(close_1015 / open_0915)
    "fh_range_norm",       # (high - low) / atr_daily
    "fh_volume_ratio",     # first_hour_vol / rolling_avg_first_hour_vol
    "fh_close_position",   # (close_1015 - low) / (high - low) — where in range
    "fh_gap",              # log(open_today / close_yesterday)
    "fh_body_ratio",       # abs(close - open) / (high - low) — candle body
    "fh_upper_wick",       # (high - max(open,close)) / (high - low)
    "fh_trend_strength",   # fh_return / fh_range_norm — directional conviction
]

# Structural features only — no directional leakage (no return, no close position)
# These tell you HOW the market is reacting (volatility, volume, gap) but not WHERE
STRUCTURAL_FEATURE_COLS = [
    "fh_range_norm",       # how volatile is the first hour vs normal
    "fh_volume_ratio",     # how much volume vs normal first hour
    "fh_gap",              # overnight gap from prev close to open
    "fh_body_ratio",       # how decisive is the candle (body vs wicks)
    "fh_upper_wick",       # upper wick ratio — rejection from highs
]


def load_first_hour_bars(
    price_data_dir: str,
    symbol: str,
    col_map: Dict[str, str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    hour_end: str = "10:15",
) -> pd.DataFrame:
    """Load minute data and compute first-hour OHLCV per day.

    Returns DataFrame with columns:
        date, fh_open, fh_high, fh_low, fh_close, fh_volume, day_open
    """
    fpath = Path(price_data_dir) / f"{symbol}_minute.csv"
    if not fpath.exists():
        return pd.DataFrame()

    df = pd.read_csv(fpath)

    # Map columns
    rename = {}
    for std_name, file_col in col_map.items():
        if file_col in df.columns:
            rename[file_col] = std_name
    df = df.rename(columns=rename)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    if start_date:
        df = df[df["datetime"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["datetime"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]

    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.time

    from datetime import time as dt_time
    market_open = dt_time(9, 15)
    hour_h, hour_m = int(hour_end.split(":")[0]), int(hour_end.split(":")[1])
    cutoff = dt_time(hour_h, hour_m)

    # First hour: 9:15 to 10:15
    first_hour = df[(df["time"] >= market_open) & (df["time"] <= cutoff)].copy()

    if first_hour.empty:
        return pd.DataFrame()

    fh_daily = first_hour.groupby("date").agg(
        fh_open=("open", "first"),
        fh_high=("high", "max"),
        fh_low=("low", "min"),
        fh_close=("close", "last"),
        fh_volume=("volume", "sum"),
    ).reset_index()

    # Also get the day's open (first bar)
    day_open = df.groupby("date").agg(day_open=("open", "first")).reset_index()
    fh_daily = fh_daily.merge(day_open, on="date", how="left")

    fh_daily["date"] = pd.to_datetime(fh_daily["date"])
    fh_daily["symbol"] = symbol

    return fh_daily


def compute_intraday_features(
    fh_bars: pd.DataFrame,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    """Compute first-hour reaction features.

    Args:
        fh_bars: first-hour OHLCV from load_first_hour_bars
        daily: daily OHLCV with 'atr' column (from numeric features)

    Returns:
        DataFrame with [symbol, date] + FIRST_HOUR_FEATURE_COLS + eod_label
    """
    if fh_bars.empty:
        return pd.DataFrame()

    # Merge with daily for previous close and ATR
    daily_sub = daily[["symbol", "date", "close", "atr"]].copy()
    daily_sub = daily_sub.rename(columns={"close": "day_close"})

    merged = fh_bars.merge(daily_sub, on=["symbol", "date"], how="inner")

    # Previous day's close
    merged = merged.sort_values(["symbol", "date"])
    merged["prev_close"] = merged.groupby("symbol")["day_close"].shift(1)

    # Drop rows without previous close
    merged = merged.dropna(subset=["prev_close", "atr"])

    # Features
    fh_range = merged["fh_high"] - merged["fh_low"]
    fh_range_safe = fh_range.replace(0, np.nan)

    merged["fh_return"] = np.log(merged["fh_close"] / merged["fh_open"])
    merged["fh_range_norm"] = fh_range / merged["atr"].replace(0, np.nan)
    merged["fh_close_position"] = (merged["fh_close"] - merged["fh_low"]) / fh_range_safe
    merged["fh_gap"] = np.log(merged["day_open"] / merged["prev_close"])
    merged["fh_body_ratio"] = (merged["fh_close"] - merged["fh_open"]).abs() / fh_range_safe
    merged["fh_upper_wick"] = (
        merged["fh_high"] - merged[["fh_open", "fh_close"]].max(axis=1)
    ) / fh_range_safe
    merged["fh_trend_strength"] = merged["fh_return"] / merged["fh_range_norm"].replace(0, np.nan)

    # Rolling first-hour volume average (20-day)
    merged["_fh_vol_ma"] = merged.groupby("symbol")["fh_volume"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).mean()
    )
    merged["fh_volume_ratio"] = merged["fh_volume"] / merged["_fh_vol_ma"].replace(0, np.nan)

    # EOD label: direction from 10:15 close to day close
    merged["eod_label"] = (merged["day_close"] > merged["fh_close"]).astype(float)

    # Reversal label: did the stock reverse from its first-hour direction by EOD?
    # first-hour direction: up if fh_close > fh_open, down otherwise
    # EOD direction (from open): up if day_close > day_open, down otherwise
    fh_direction_up = merged["fh_close"] > merged["fh_open"]
    eod_direction_up = merged["day_close"] > merged["day_open"]
    merged["reversal_label"] = (fh_direction_up != eod_direction_up).astype(float)

    # Clean up
    for col in FIRST_HOUR_FEATURE_COLS:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
            merged[col] = merged[col].replace([np.inf, -np.inf], 0)

    result = merged[["symbol", "date"] + FIRST_HOUR_FEATURE_COLS + ["eod_label", "reversal_label"]].copy()
    return result


def load_universe_intraday_features(
    price_data_dir: str,
    symbols: List[str],
    col_map: Dict[str, str],
    daily: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    hour_end: str = "10:15",
) -> pd.DataFrame:
    """Load first-hour features for all symbols."""
    frames = []
    for sym in symbols:
        try:
            fh = load_first_hour_bars(price_data_dir, sym, col_map, start_date, end_date, hour_end)
            if fh.empty:
                continue
            feats = compute_intraday_features(fh, daily)
            if not feats.empty:
                frames.append(feats)
        except Exception as e:
            logger.warning(f"Skipping intraday for {sym}: {e}")

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    logger.info(f"Intraday features: {result.shape}, {result['symbol'].nunique()} symbols")
    return result
