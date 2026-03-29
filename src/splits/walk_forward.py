"""Walk-forward splitter with purge and embargo.

Implements rolling walk-forward evaluation ensuring no look-ahead leakage.
Purge removes training rows whose lookback windows overlap with val/test labels.
Embargo adds a buffer gap between train and val/test.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardSplit:
    """A single walk-forward split."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walk_forward_splits(
    dates: pd.Series,
    train_months: int = 12,
    val_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
) -> List[WalkForwardSplit]:
    """Generate walk-forward date splits.

    Args:
        dates: sorted Series of unique trading dates.
        train_months, val_months, test_months: period lengths.
        step_months: how far to slide forward each fold.

    Returns:
        List of WalkForwardSplit objects.
    """
    dates = pd.to_datetime(dates).sort_values().unique()
    min_date = dates[0]
    max_date = dates[-1]

    splits = []
    fold_id = 0
    train_start = min_date

    while True:
        train_end = train_start + pd.DateOffset(months=train_months) - timedelta(days=1)
        val_start = train_end + timedelta(days=1)
        val_end = val_start + pd.DateOffset(months=val_months) - timedelta(days=1)
        test_start = val_end + timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - timedelta(days=1)

        # Check if test period has data
        if test_start > max_date:
            break

        # Clip test_end to available data
        test_end = min(test_end, max_date)

        splits.append(WalkForwardSplit(
            fold_id=fold_id,
            train_start=pd.Timestamp(train_start),
            train_end=pd.Timestamp(train_end),
            val_start=pd.Timestamp(val_start),
            val_end=pd.Timestamp(val_end),
            test_start=pd.Timestamp(test_start),
            test_end=pd.Timestamp(test_end),
        ))

        fold_id += 1
        train_start = train_start + pd.DateOffset(months=step_months)

    logger.info(f"Generated {len(splits)} walk-forward splits")
    for s in splits:
        logger.info(
            f"  Fold {s.fold_id}: train [{s.train_start.date()}..{s.train_end.date()}] "
            f"val [{s.val_start.date()}..{s.val_end.date()}] "
            f"test [{s.test_start.date()}..{s.test_end.date()}]"
        )
    return splits


def apply_purge_embargo(
    df: pd.DataFrame,
    split: WalkForwardSplit,
    purge_days: int = 25,
    embargo_days: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test with purge and embargo.

    Purge: remove training rows within purge_days before val/test start.
    Embargo: remove rows within embargo_days after train end.

    This ensures no training row's lookback window overlaps with
    validation/test label windows.

    Args:
        df: DataFrame with 'date' column.
        split: WalkForwardSplit defining date boundaries.
        purge_days: number of days to purge from end of training.
        embargo_days: additional embargo gap.

    Returns:
        (train_df, val_df, test_df) with purge/embargo applied.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Define boundaries
    purge_cutoff = split.val_start - timedelta(days=purge_days)

    # Train: [train_start, min(purge_cutoff, train_end) - embargo_days]
    effective_train_end = min(purge_cutoff, split.train_end) - timedelta(days=embargo_days)
    train_mask = (df["date"] >= split.train_start) & (df["date"] <= effective_train_end)

    # Val: [val_start, val_end]
    val_mask = (df["date"] >= split.val_start) & (df["date"] <= split.val_end)

    # Test: [test_start, test_end]
    test_mask = (df["date"] >= split.test_start) & (df["date"] <= split.test_end)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(
        f"  Fold {split.fold_id} after purge/embargo: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)} "
        f"(purge removed {purge_days}d before val, embargo={embargo_days}d)"
    )

    return train_df, val_df, test_df
