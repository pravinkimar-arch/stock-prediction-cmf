"""Tests for first-seen session assignment."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, date
import pytz
from src.data.nse_calendar import assign_filing_to_session, is_trading_day, next_trading_day

IST = pytz.timezone("Asia/Kolkata")


def test_during_session():
    """Filing during trading hours → same day."""
    ts = IST.localize(datetime(2024, 1, 3, 11, 30, 0))  # Wednesday 11:30 AM
    assert assign_filing_to_session(ts) == date(2024, 1, 3)


def test_after_close():
    """Filing after market close → next trading day."""
    ts = IST.localize(datetime(2024, 1, 3, 16, 0, 0))  # Wednesday 4 PM
    result = assign_filing_to_session(ts)
    assert result == date(2024, 1, 4)  # Thursday


def test_before_open():
    """Filing before market open on trading day → same day."""
    ts = IST.localize(datetime(2024, 1, 3, 8, 0, 0))  # Wednesday 8 AM
    assert assign_filing_to_session(ts) == date(2024, 1, 3)


def test_saturday():
    """Filing on Saturday → next Monday (if Monday is trading day)."""
    ts = IST.localize(datetime(2024, 1, 6, 10, 0, 0))  # Saturday
    result = assign_filing_to_session(ts)
    assert result == date(2024, 1, 8)  # Monday


def test_sunday():
    """Filing on Sunday → next Monday."""
    ts = IST.localize(datetime(2024, 1, 7, 14, 0, 0))  # Sunday
    result = assign_filing_to_session(ts)
    assert result == date(2024, 1, 8)  # Monday


def test_friday_after_close():
    """Filing Friday after close → next Monday."""
    ts = IST.localize(datetime(2024, 1, 5, 16, 30, 0))  # Friday 4:30 PM
    result = assign_filing_to_session(ts)
    assert result == date(2024, 1, 8)  # Monday


def test_holiday():
    """Filing on Republic Day (holiday) → next trading day."""
    ts = IST.localize(datetime(2024, 1, 26, 10, 0, 0))  # Republic Day
    assert not is_trading_day(date(2024, 1, 26))
    result = assign_filing_to_session(ts)
    assert is_trading_day(result)
    assert result > date(2024, 1, 26)


def test_at_close_boundary():
    """Filing exactly at 15:30 → same day (close time is inclusive)."""
    ts = IST.localize(datetime(2024, 1, 3, 15, 30, 0))
    assert assign_filing_to_session(ts) == date(2024, 1, 3)


if __name__ == "__main__":
    test_during_session()
    test_after_close()
    test_before_open()
    test_saturday()
    test_sunday()
    test_friday_after_close()
    test_holiday()
    test_at_close_boundary()
    print("✓ All first-seen assignment tests passed!")
