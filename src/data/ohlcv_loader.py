"""OHLCV data loader: reads per-symbol minute CSVs, resamples to daily."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.data.nse_calendar import is_trading_day

logger = logging.getLogger(__name__)


def _find_file(data_dir: Path, symbol: str) -> Optional[Path]:
    """Find CSV or Parquet file for a symbol."""
    for pattern in [f"{symbol}_minute.csv", f"{symbol}_daily.csv",
                    f"{symbol}.csv", f"{symbol}.parquet",
                    f"{symbol}_minute.parquet", f"{symbol}_daily.parquet"]:
        p = data_dir / pattern
        if p.exists():
            return p
    return None


def load_symbol_ohlcv(
    data_dir: str,
    symbol: str,
    col_map: Dict[str, str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load and resample a single symbol's OHLCV to daily.

    Returns DataFrame with columns: date, open, high, low, close, volume, symbol
    indexed by date (DatetimeIndex).
    """
    data_path = Path(data_dir)
    fpath = _find_file(data_path, symbol)
    if fpath is None:
        raise FileNotFoundError(
            f"No OHLCV file found for {symbol} in {data_dir}. "
            f"Expected patterns: {symbol}_minute.csv, {symbol}.csv, etc."
        )

    logger.info(f"Loading {symbol} from {fpath}")

    if fpath.suffix == ".parquet":
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath)

    # Map columns
    rename = {}
    for std_name, file_col in col_map.items():
        if file_col in df.columns:
            rename[file_col] = std_name
    df = df.rename(columns=rename)

    required = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} for {symbol}. "
            f"Available: {list(df.columns)}. Check ohlcv_columns in config."
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Filter date range
    if start_date:
        df = df[df["datetime"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["datetime"] <= pd.Timestamp(end_date)]

    # Check if already daily (< 2 rows per day on average)
    df["_date"] = df["datetime"].dt.date
    rows_per_day = df.groupby("_date").size().median()

    if rows_per_day > 2:
        # Minute data → resample to daily
        logger.info(f"  {symbol}: minute data ({rows_per_day:.0f} bars/day), resampling to daily")
        daily = _resample_to_daily(df)
    else:
        logger.info(f"  {symbol}: already daily")
        daily = df.groupby("_date").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).reset_index()
        daily = daily.rename(columns={"_date": "date"})

    daily["date"] = pd.to_datetime(daily["date"])
    daily["symbol"] = symbol

    # Remove non-trading days (shouldn't exist in market data, but safety check)
    daily = daily[daily["date"].apply(lambda x: is_trading_day(x.date()))]
    daily = daily.sort_values("date").reset_index(drop=True)

    logger.info(f"  {symbol}: {len(daily)} trading days loaded")
    return daily


def _resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample minute bars to daily OHLCV."""
    df = df.copy()
    df["_date"] = df["datetime"].dt.date
    daily = df.groupby("_date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    daily = daily.rename(columns={"_date": "date"})
    # Drop days with zero volume or NaN
    daily = daily.dropna(subset=["open", "high", "low", "close"])
    daily = daily[daily["volume"] > 0]
    return daily


def load_cached_ohlcv(
    cache_path: str,
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load pre-computed daily OHLCV from a parquet cache file.

    Use this when raw minute-level CSVs are not available but the
    resampled daily cache (cache/daily_ohlcv.parquet) exists.
    """
    df = pd.read_parquet(cache_path)
    df["date"] = pd.to_datetime(df["date"])
    if symbols:
        if isinstance(symbols, dict):
            symbols = list(symbols.keys())
        df = df[df["symbol"].isin(symbols)]
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} cached daily rows for {df['symbol'].nunique()} symbols")
    return df


def load_universe_ohlcv(
    data_dir: str,
    symbols: List[str],
    col_map: Dict[str, str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Load daily OHLCV for all symbols, return stacked DataFrame.

    First tries cache/daily_ohlcv.parquet; falls back to raw CSV files.

    If dry_run=True, returns an empty DataFrame with the correct schema
    instead of reading any data files.
    """
    if dry_run:
        logger.info("[DRY RUN] Skipping OHLCV data loading")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])

    # Try cached daily OHLCV first
    cache_path = Path("cache/daily_ohlcv.parquet")
    if cache_path.exists():
        logger.info("Loading from cached daily_ohlcv.parquet")
        if isinstance(symbols, dict):
            symbols = list(symbols.keys())
        return load_cached_ohlcv(str(cache_path), symbols, start_date, end_date)

    # Fall back to raw minute-level CSV files
    frames = []
    for sym in symbols:
        try:
            df = load_symbol_ohlcv(data_dir, sym, col_map, start_date, end_date)
            frames.append(df)
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {sym}: {e}")

    if not frames:
        raise RuntimeError("No OHLCV data loaded for any symbol.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info(f"Loaded {len(combined)} total daily rows for {combined['symbol'].nunique()} symbols")
    return combined
