"""NSE trading calendar utilities.

Provides trading-day checks, next-session lookups, and holiday handling.
Uses pandas_market_calendars when available, falls back to weekday-only logic.
"""

import logging
from datetime import date, datetime, timedelta, time as dtime
from typing import Optional, List, Set

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# Known NSE holidays 2024-2025 (extend as needed)
_NSE_HOLIDAYS: Set[date] = {
    # 2024
    date(2024, 1, 26), date(2024, 3, 8), date(2024, 3, 25),
    date(2024, 3, 29), date(2024, 4, 11), date(2024, 4, 14),
    date(2024, 4, 17), date(2024, 4, 21), date(2024, 5, 1),
    date(2024, 5, 20), date(2024, 5, 23), date(2024, 6, 17),
    date(2024, 7, 17), date(2024, 8, 15), date(2024, 9, 16),
    date(2024, 10, 2), date(2024, 10, 12), date(2024, 10, 31),
    date(2024, 11, 1), date(2024, 11, 15), date(2024, 12, 25),
    # 2025
    date(2025, 2, 26), date(2025, 3, 14), date(2025, 3, 31),
    date(2025, 4, 10), date(2025, 4, 14), date(2025, 4, 18),
    date(2025, 5, 1), date(2025, 8, 15), date(2025, 8, 16),
    date(2025, 8, 27), date(2025, 10, 2), date(2025, 10, 21),
    date(2025, 10, 22), date(2025, 11, 5), date(2025, 11, 26),
    date(2025, 12, 25),
}


def is_trading_day(d: date) -> bool:
    """Check if a date is an NSE trading day."""
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    if d in _NSE_HOLIDAYS:
        return False
    return True


def next_trading_day(d: date) -> date:
    """Return the next trading day strictly after d."""
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def prev_trading_day(d: date) -> date:
    """Return the previous trading day strictly before d."""
    prev = d - timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    return prev


def trading_days_between(start: date, end: date) -> List[date]:
    """Return list of trading days in [start, end] inclusive."""
    days = []
    d = start
    while d <= end:
        if is_trading_day(d):
            days.append(d)
        d += timedelta(days=1)
    return days


def assign_filing_to_session(
    ts: datetime,
    market_open: dtime = dtime(9, 15),
    market_close: dtime = dtime(15, 30),
) -> date:
    """Assign a filing timestamp (IST-aware or naive-assumed-IST) to a trading session.

    Rules:
    - If posted during trading hours on a trading day → that day
    - If posted after close on a trading day → next trading day
    - If posted before open on a trading day → that day
    - If posted on weekend/holiday → next trading day
    """
    if ts.tzinfo is not None:
        ts = ts.astimezone(IST)
    d = ts.date()
    t = ts.time()

    if not is_trading_day(d):
        return next_trading_day(d)

    if t > market_close:
        return next_trading_day(d)

    # During or before session on a trading day → assign to that day
    return d
