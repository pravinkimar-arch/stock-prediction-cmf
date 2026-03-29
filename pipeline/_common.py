"""Shared utilities for pipeline steps.

Centralizes functions duplicated across scripts (e.g. label creation).
"""

import os
import sys

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_labels(daily: pd.DataFrame) -> pd.DataFrame:
    """Create next-day direction labels.

    y_{t+1} = 1 if log(C_{t+1} / C_t) > 0, else 0.
    The last row per symbol gets NaN (no next-day close available).
    """
    frames = []
    for sym, gdf in daily.groupby("symbol"):
        df = gdf.sort_values("date").copy()
        df["next_close"] = df["close"].shift(-1)
        df["label"] = (np.log(df["next_close"] / df["close"]) > 0).astype(float)
        df.loc[df["next_close"].isna(), "label"] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
