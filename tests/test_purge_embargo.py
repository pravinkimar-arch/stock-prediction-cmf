"""Tests for purge/embargo ensuring no overlap leakage."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.splits.walk_forward import (
    WalkForwardSplit, apply_purge_embargo, generate_walk_forward_splits,
)


def make_test_data(start="2020-01-01", end="2023-12-31"):
    """Create test DataFrame with daily dates."""
    dates = pd.bdate_range(start, end)
    df = pd.DataFrame({
        "date": dates,
        "close": np.random.randn(len(dates)).cumsum() + 100,
        "label": np.random.randint(0, 2, len(dates)),
        "symbol": "TEST",
    })
    return df


def test_no_overlap_train_val():
    """Train end + purge must not overlap with val start."""
    df = make_test_data()
    split = WalkForwardSplit(
        fold_id=0,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2021-12-31"),
        val_start=pd.Timestamp("2022-01-01"),
        val_end=pd.Timestamp("2022-06-30"),
        test_start=pd.Timestamp("2022-07-01"),
        test_end=pd.Timestamp("2022-12-31"),
    )

    train_df, val_df, test_df = apply_purge_embargo(df, split, purge_days=25, embargo_days=5)

    if len(train_df) > 0 and len(val_df) > 0:
        train_max = train_df["date"].max()
        val_min = val_df["date"].min()
        gap = (val_min - train_max).days
        assert gap >= 25, f"Gap between train and val must be >= purge_days, got {gap}"


def test_no_overlap_train_test():
    """Train data must not overlap with test data."""
    df = make_test_data()
    split = WalkForwardSplit(
        fold_id=0,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2021-12-31"),
        val_start=pd.Timestamp("2022-01-01"),
        val_end=pd.Timestamp("2022-06-30"),
        test_start=pd.Timestamp("2022-07-01"),
        test_end=pd.Timestamp("2022-12-31"),
    )

    train_df, val_df, test_df = apply_purge_embargo(df, split, purge_days=25, embargo_days=5)

    if len(train_df) > 0 and len(test_df) > 0:
        train_dates = set(train_df["date"].dt.date)
        test_dates = set(test_df["date"].dt.date)
        overlap = train_dates & test_dates
        assert len(overlap) == 0, f"Train and test overlap: {overlap}"


def test_val_test_no_overlap():
    """Val and test must not overlap."""
    df = make_test_data()
    split = WalkForwardSplit(
        fold_id=0,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2021-12-31"),
        val_start=pd.Timestamp("2022-01-01"),
        val_end=pd.Timestamp("2022-06-30"),
        test_start=pd.Timestamp("2022-07-01"),
        test_end=pd.Timestamp("2022-12-31"),
    )

    train_df, val_df, test_df = apply_purge_embargo(df, split, purge_days=25, embargo_days=5)

    if len(val_df) > 0 and len(test_df) > 0:
        val_dates = set(val_df["date"].dt.date)
        test_dates = set(test_df["date"].dt.date)
        overlap = val_dates & test_dates
        assert len(overlap) == 0, f"Val and test overlap: {overlap}"


def test_purge_removes_rows():
    """Purge should actually remove rows from training."""
    df = make_test_data()
    split = WalkForwardSplit(
        fold_id=0,
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2021-12-31"),
        val_start=pd.Timestamp("2022-01-01"),
        val_end=pd.Timestamp("2022-06-30"),
        test_start=pd.Timestamp("2022-07-01"),
        test_end=pd.Timestamp("2022-12-31"),
    )

    train_no_purge, _, _ = apply_purge_embargo(df, split, purge_days=0, embargo_days=0)
    train_with_purge, _, _ = apply_purge_embargo(df, split, purge_days=25, embargo_days=5)

    assert len(train_with_purge) < len(train_no_purge), "Purge should reduce training set size"


def test_splits_generation():
    """Walk-forward splits should be generated correctly."""
    dates = pd.bdate_range("2020-01-01", "2023-12-31")
    splits = generate_walk_forward_splits(
        pd.Series(dates), train_months=12, val_months=3, test_months=3, step_months=3,
    )
    assert len(splits) > 0, "Should generate at least one split"

    for s in splits:
        assert s.train_start < s.train_end < s.val_start < s.val_end < s.test_start
        assert s.test_start <= s.test_end


if __name__ == "__main__":
    test_no_overlap_train_val()
    test_no_overlap_train_test()
    test_val_test_no_overlap()
    test_purge_removes_rows()
    test_splits_generation()
    print("✓ All purge/embargo tests passed!")
