"""Tests for Chart2Tokens detectors on synthetic candles."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.features.chart2tokens import (
    detect_breakout, detect_gap, detect_volume_burst,
    detect_round_number_touch, detect_engulfing,
)


def make_synthetic_daily(n=50):
    """Create synthetic daily OHLCV data."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 1000 + np.cumsum(np.random.randn(n) * 5)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    open_ = close + np.random.randn(n) * 2
    volume = np.random.randint(100000, 500000, n).astype(float)

    df = pd.DataFrame({
        "date": dates, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "symbol": "TEST",
    })
    # ATR proxy
    df["atr"] = (df["high"] - df["low"]).rolling(14, min_periods=14).mean()
    return df


def test_breakout_detection():
    """Breakout should fire when close exceeds 20-day high for first time."""
    df = make_synthetic_daily(50)
    # Force a breakout at index 25
    df.loc[25, "close"] = df["high"].iloc[5:25].max() + 10
    df.loc[24, "close"] = df["high"].iloc[4:24].max() - 5  # prev day below

    result = detect_breakout(df, window=20)
    assert result.iloc[25] == 1, "Breakout should fire at index 25"


def test_gap_detection():
    """Gap up should fire when open > prev high by ATR threshold."""
    df = make_synthetic_daily(50)
    df["atr"] = 10.0  # Fixed ATR for testing

    # Force gap up at index 30
    df.loc[30, "open"] = df.loc[29, "high"] + 10  # gap > 0.5 * ATR(10) = 5

    gaps = detect_gap(df, df["atr"], atr_threshold=0.5)
    assert gaps["gap_up"].iloc[30] == 1, "Gap up should fire at index 30"


def test_volume_burst():
    """Volume burst should fire when volume >= 3x rolling mean."""
    df = make_synthetic_daily(50)
    df["volume"] = 100000.0  # Constant baseline

    # Spike at index 30
    df.loc[30, "volume"] = 400000.0  # 4x > 3x threshold

    result = detect_volume_burst(df, multiplier=3.0, norm="mean", window=20)
    assert result.iloc[30] == 1, "Volume burst should fire at index 30"


def test_round_number_touch():
    """Round number touch should fire near multiples of step."""
    df = make_synthetic_daily(50)
    df["atr"] = 10.0

    # Force close near 1000 (round number)
    df.loc[30, "close"] = 1001.5  # within 0.3 * 10 = 3 of 1000

    result = detect_round_number_touch(df, df["atr"], step=100, atr_fraction=0.3)
    assert result.iloc[30] == 1, "Round touch should fire near 1000"


def test_engulfing_bullish():
    """Bullish engulfing: prev bearish candle engulfed by current bullish."""
    df = make_synthetic_daily(50)

    # Set up bearish candle at 29, bullish engulfing at 30
    df.loc[29, "open"] = 1010
    df.loc[29, "close"] = 1000  # bearish
    df.loc[30, "open"] = 999   # open <= prev close
    df.loc[30, "close"] = 1011  # close >= prev open

    result = detect_engulfing(df)
    assert result["engulfing_bull"].iloc[30] == 1, "Bullish engulfing should fire"


def test_engulfing_bearish():
    """Bearish engulfing: prev bullish candle engulfed by current bearish."""
    df = make_synthetic_daily(50)

    df.loc[29, "open"] = 1000
    df.loc[29, "close"] = 1010  # bullish
    df.loc[30, "open"] = 1011  # open >= prev close
    df.loc[30, "close"] = 999   # close <= prev open

    result = detect_engulfing(df)
    assert result["engulfing_bear"].iloc[30] == 1, "Bearish engulfing should fire"


def test_no_lookahead_in_breakout():
    """Breakout uses shift(1) so day t only sees data up to t-1."""
    df = make_synthetic_daily(50)
    result = detect_breakout(df, window=20)
    # First 21 rows should be 0 (need 20 days of history + 1 shift)
    assert result.iloc[:21].sum() == 0, "No breakout signal in warmup period"


if __name__ == "__main__":
    test_breakout_detection()
    test_gap_detection()
    test_volume_burst()
    test_round_number_touch()
    test_engulfing_bullish()
    test_engulfing_bearish()
    test_no_lookahead_in_breakout()
    print("✓ All Chart2Tokens tests passed!")
