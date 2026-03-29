"""Chart2Tokens: deterministic event detectors + token summary features.

Two families of chart-derived features:

1. Binary event tokens (legacy): detect discrete chart events per day,
   then summarize with count/recency/time-since-last in a lookback window.
   Only the best-performing events are retained: engulfing (bull/bear)
   and gap_down.

2. Milestone attraction (continuous): measure price proximity to
   psychological round-number levels at multiple scales (50/100/500/1000).
   Features: normalized distance, gravitational pull, direction of
   movement, and crossing frequency — all leak-safe via shift(1).
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---- Milestone attraction constants ----

MILESTONE_STEPS = [50, 100, 500, 1000]

# ---- Token detectors ----


def detect_breakout(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Breakout: first close above rolling 20-day high (excluding current day).

    close[t] > max(high[t-window : t-1]) AND close[t-1] <= max(high[t-window-1 : t-2])
    """
    rolling_high = df["high"].shift(1).rolling(window=window, min_periods=window).max()
    prev_rolling_high = df["high"].shift(2).rolling(window=window, min_periods=window).max()
    breakout = (df["close"] > rolling_high) & (df["close"].shift(1) <= prev_rolling_high)
    return breakout.astype(int).fillna(0)


def detect_gap(df: pd.DataFrame, atr: pd.Series, atr_threshold: float = 0.5) -> pd.DataFrame:
    """Gap up/down: open outside prior day's range, filtered by ATR threshold.

    Returns DataFrame with columns: gap_up, gap_down.
    """
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    gap_size_up = df["open"] - prev_high
    gap_size_down = prev_low - df["open"]
    threshold = atr * atr_threshold

    gap_up = ((gap_size_up > 0) & (gap_size_up >= threshold)).astype(int).fillna(0)
    gap_down = ((gap_size_down > 0) & (gap_size_down >= threshold)).astype(int).fillna(0)
    return pd.DataFrame({"gap_up": gap_up, "gap_down": gap_down}, index=df.index)


def detect_volume_burst(
    df: pd.DataFrame,
    multiplier: float = 3.0,
    norm: str = "mean",
    window: int = 20,
) -> pd.Series:
    """Volume burst: volume >= multiplier × rolling norm."""
    if norm == "median":
        rolling_norm = df["volume"].shift(1).rolling(window=window, min_periods=window).median()
    else:
        rolling_norm = df["volume"].shift(1).rolling(window=window, min_periods=window).mean()
    burst = (df["volume"] >= multiplier * rolling_norm).astype(int).fillna(0)
    return burst


def detect_round_number_touch(
    df: pd.DataFrame,
    atr: pd.Series,
    step: int = 100,
    atr_fraction: float = 0.3,
) -> pd.Series:
    """Round-number touch: price within atr_fraction * ATR of a round level."""
    threshold = atr * atr_fraction

    # Check if any of OHLC is near a round number
    def near_round(price, thresh):
        remainder = price % step
        dist = np.minimum(remainder, step - remainder)
        return dist <= thresh

    touch = (
        near_round(df["high"], threshold)
        | near_round(df["low"], threshold)
        | near_round(df["close"], threshold)
    ).astype(int).fillna(0)
    return touch


def detect_engulfing(df: pd.DataFrame) -> pd.DataFrame:
    """Bullish and bearish engulfing candle patterns.

    Returns DataFrame with columns: engulfing_bull, engulfing_bear.
    """
    body = df["close"] - df["open"]
    prev_body = body.shift(1)
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)

    # Bullish engulfing: prev bearish, current bullish, current body engulfs prev
    bull = (
        (prev_body < 0)
        & (body > 0)
        & (df["open"] <= prev_close)
        & (df["close"] >= prev_open)
    ).astype(int).fillna(0)

    # Bearish engulfing: prev bullish, current bearish, current body engulfs prev
    bear = (
        (prev_body > 0)
        & (body < 0)
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
    ).astype(int).fillna(0)

    return pd.DataFrame({"engulfing_bull": bull, "engulfing_bear": bear}, index=df.index)


# ---- Milestone attraction ----


def compute_milestone_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Compute continuous milestone attraction features for a single symbol.

    All features use shift(1) to avoid look-ahead leakage.

    Per milestone scale (50, 100, 500, 1000):
      - ms_{step}_dist:      normalized distance to nearest milestone (dist / ATR)
      - ms_{step}_gravity:   exp(-dist/ATR), stronger when closer
      - ms_{step}_direction: +1 if closing toward milestone, -1 if away
      - ms_{step}_cross_{lookback}: milestone crossings in lookback window

    Aggregates:
      - ms_min_dist:      min distance across all scales
      - ms_max_gravity:   max gravity across all scales
      - ms_cross_total:   total crossings across all scales
      - ms_squeeze:       price between two 100-level milestones within 1 ATR

    Args:
        df: DataFrame with close, atr columns, sorted by date (single symbol).
        lookback: window for crossing count.

    Returns:
        DataFrame with milestone feature columns added.
    """
    close = df["close"].values
    atr = df["atr"].values

    for step in MILESTONE_STEPS:
        prev_close = np.roll(close, 1)
        prev_close[0] = np.nan
        prev_atr = np.roll(atr, 1)
        prev_atr[0] = np.nan

        # Distance to nearest milestone
        remainder = prev_close % step
        dist = np.minimum(remainder, step - remainder)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm_dist = np.where(prev_atr > 0, dist / prev_atr, np.nan)

        df[f"ms_{step}_dist"] = norm_dist
        df[f"ms_{step}_gravity"] = np.exp(-np.clip(norm_dist, 0, 10))

        # Direction: getting closer (+1) or farther (-1)?
        prev2_close = np.roll(close, 2)
        prev2_close[:2] = np.nan
        prev_remainder = prev2_close % step
        prev_dist = np.minimum(prev_remainder, step - prev_remainder)
        df[f"ms_{step}_direction"] = np.sign(prev_dist - dist)

        # Crossing count in lookback window
        milestone_level = np.floor(prev_close / step)
        crossing = (milestone_level != np.roll(milestone_level, 1)).astype(float)
        crossing[0] = 0
        cross_series = pd.Series(crossing, index=df.index)
        df[f"ms_{step}_cross_{lookback}"] = (
            cross_series.shift(1)
            .rolling(lookback, min_periods=lookback)
            .sum()
            .values
        )

    # Aggregate features
    dist_cols = [f"ms_{s}_dist" for s in MILESTONE_STEPS]
    grav_cols = [f"ms_{s}_gravity" for s in MILESTONE_STEPS]
    cross_cols = [f"ms_{s}_cross_{lookback}" for s in MILESTONE_STEPS]

    df["ms_min_dist"] = df[dist_cols].min(axis=1)
    df["ms_max_gravity"] = df[grav_cols].max(axis=1)
    df["ms_cross_total"] = df[cross_cols].sum(axis=1)

    # Squeeze: close between two 100-level milestones, both within 1 ATR
    prev_close_s = df["close"].shift(1)
    prev_atr_s = df["atr"].shift(1)
    above = np.ceil(prev_close_s / 100) * 100
    below = np.floor(prev_close_s / 100) * 100
    squeeze = ((above - prev_close_s) < prev_atr_s) & (
        (prev_close_s - below) < prev_atr_s
    )
    df["ms_squeeze"] = squeeze.astype(float)

    return df


# ---- Token summary features ----

# All legacy event tokens (kept for backward compatibility)
TOKEN_NAMES_ALL = [
    "breakout", "gap_up", "gap_down", "volume_burst",
    "round_touch", "engulfing_bull", "engulfing_bear",
]

# Best-performing subset: engulfing + gap_down (validated by feature pruning)
TOKEN_NAMES_BEST = ["gap_down", "engulfing_bull", "engulfing_bear"]

# Default: use best-performing subset
TOKEN_NAMES = TOKEN_NAMES_BEST


def compute_token_summaries(
    signals: pd.DataFrame,
    token_cols: List[str],
    W: int = 20,
    h: float = 5.0,
    max_time_since: int = 60,
) -> pd.DataFrame:
    """Compute count, recency-weighted count, and time-since-last for each token.

    All computations use only data up to and including day t (no look-ahead).

    Args:
        signals: DataFrame with binary token columns, sorted by date per symbol.
        token_cols: list of token column names.
        W: lookback window in days.
        h: half-life for exponential decay (in days).
        max_time_since: cap for time_since_last.

    Returns:
        DataFrame with 3 features per token: {token}_count_W, {token}_recency_W, {token}_tsl
    """
    result_frames = []

    for sym, gdf in signals.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        n = len(df)

        for tok in token_cols:
            sig = df[tok].values.astype(float)

            counts = np.full(n, np.nan)
            recency = np.full(n, np.nan)
            tsl = np.full(n, np.nan)

            for i in range(n):
                start = max(0, i - W + 1)
                window = sig[start: i + 1]

                # Count
                counts[i] = np.nansum(window)

                # Recency-weighted count: weight = 0.5^(age/h)
                ages = np.arange(len(window) - 1, -1, -1, dtype=float)
                weights = np.power(0.5, ages / h)
                recency[i] = np.nansum(window * weights)

                # Time since last occurrence
                occurrences = np.where(window > 0)[0]
                if len(occurrences) > 0:
                    tsl[i] = len(window) - 1 - occurrences[-1]
                else:
                    tsl[i] = min(len(window), max_time_since)

            df[f"{tok}_count_{W}"] = counts
            df[f"{tok}_recency_{W}"] = recency
            df[f"{tok}_tsl"] = tsl

        result_frames.append(df)

    return pd.concat(result_frames, ignore_index=True)


def compute_chart2tokens(daily: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Full Chart2Tokens pipeline: detect events + milestone attraction.

    Computes:
      1. All legacy binary event tokens (for backward compatibility)
      2. Token summary features (count/recency/tsl) for best-performing events
      3. Milestone attraction continuous features

    Args:
        daily: DataFrame with OHLCV + atr column, sorted by [symbol, date].
        cfg: config dict with chart2tokens and token_summary sections.

    Returns:
        DataFrame with all original columns + token signals + summary + milestone features.
    """
    c2t = cfg.get("chart2tokens", {})
    ts_cfg = cfg.get("token_summary", {})

    W = ts_cfg.get("lookback_W", 20)
    h = ts_cfg.get("half_life_h", 5.0)
    max_tsl = ts_cfg.get("max_time_since", 60)

    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()

        # Ensure ATR exists
        if "atr" not in df.columns:
            raise ValueError(f"ATR column missing for {sym}. Run numeric features first.")

        # Detect all legacy tokens (kept for backward compat)
        df["breakout"] = detect_breakout(df, c2t.get("breakout_window", 20))

        gaps = detect_gap(df, df["atr"], c2t.get("gap_atr_threshold", 0.5))
        df["gap_up"] = gaps["gap_up"]
        df["gap_down"] = gaps["gap_down"]

        df["volume_burst"] = detect_volume_burst(
            df,
            c2t.get("volume_burst_multiplier", 3.0),
            c2t.get("volume_burst_norm", "mean"),
            c2t.get("volume_burst_window", 20),
        )

        df["round_touch"] = detect_round_number_touch(
            df, df["atr"],
            c2t.get("round_number_step", 100),
            c2t.get("round_number_atr_fraction", 0.3),
        )

        engulf = detect_engulfing(df)
        df["engulfing_bull"] = engulf["engulfing_bull"]
        df["engulfing_bear"] = engulf["engulfing_bear"]

        # Milestone attraction features
        df = compute_milestone_features(df, lookback=W)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Compute summary features for ALL tokens (legacy compat)
    result = compute_token_summaries(combined, TOKEN_NAMES_ALL, W, h, max_tsl)

    logger.info(f"Chart2Tokens computed: {result.shape}")
    return result


def get_token_feature_cols(W: int = 20) -> List[str]:
    """Return legacy token summary feature columns (all 7 events, 21 features).

    Kept for backward compatibility with existing pipeline scripts.
    """
    cols = []
    for tok in TOKEN_NAMES_ALL:
        cols.extend([f"{tok}_count_{W}", f"{tok}_recency_{W}", f"{tok}_tsl"])
    return cols


def get_milestone_feature_cols(W: int = 20) -> List[str]:
    """Return milestone attraction feature column names (20 features)."""
    cols = []
    for step in MILESTONE_STEPS:
        cols.extend([
            f"ms_{step}_dist",
            f"ms_{step}_gravity",
            f"ms_{step}_direction",
            f"ms_{step}_cross_{W}",
        ])
    cols.extend(["ms_min_dist", "ms_max_gravity", "ms_cross_total", "ms_squeeze"])
    return cols


def get_best_token_feature_cols(W: int = 20) -> List[str]:
    """Return best-performing Chart2Tokens features (pruned events + milestone).

    Combines:
      - Token summaries for best events: gap_down, engulfing_bull, engulfing_bear (9 feats)
      - Top milestone attraction features (5 feats): ms_500_dist, ms_1000_dist,
        ms_100_dist, ms_100_gravity, ms_500_gravity
    """
    # Best binary event summaries
    token_cols = []
    for tok in TOKEN_NAMES_BEST:
        token_cols.extend([f"{tok}_count_{W}", f"{tok}_recency_{W}", f"{tok}_tsl"])

    # Top milestone features (by importance across both time periods)
    milestone_cols = [
        "ms_500_dist", "ms_1000_dist", "ms_100_dist",
        "ms_100_gravity", "ms_500_gravity",
    ]

    return token_cols + milestone_cols


def get_chart2tokens_v2_cols(W: int = 20) -> List[str]:
    """Return all Chart2Tokens v2 features (pruned events + all milestone).

    Combines:
      - Token summaries for best events: gap_down, engulfing_bull, engulfing_bear (9 feats)
      - All milestone attraction features (20 feats)
    """
    token_cols = []
    for tok in TOKEN_NAMES_BEST:
        token_cols.extend([f"{tok}_count_{W}", f"{tok}_recency_{W}", f"{tok}_tsl"])
    return token_cols + get_milestone_feature_cols(W)
