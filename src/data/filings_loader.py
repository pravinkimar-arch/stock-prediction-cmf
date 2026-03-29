"""Filings / announcements loader with first-seen session assignment.

Includes a two-stage filing classifier:
  1. Subject-based type classification (earnings, dividend, m_and_a, etc.)
  2. Body-text directional signal extraction for high-impact types
     → produces influencing-factor labels like earnings_positive, dividend_announced, etc.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytz

from src.data.nse_calendar import assign_filing_to_session, IST

logger = logging.getLogger(__name__)


# ============================================================
# Stage 1: Subject → base type
# ============================================================

_CATEGORY_RULES = [
    (r"financial.?result|quarterly.?result|annual.?result|half.?yearly",
     "earnings"),
    (r"dividend", "dividend"),
    (r"outcome.?of.?board|board.?meeting", "board_outcome"),
    (r"acqui|merger|amalgam|scheme.?of.?arrange|takeover|disinvest|sale.?or.?disposal",
     "m_and_a"),
    (r"change.?in.?director|change.?in.?management|appointment|resignation|cessation|"
     r"company.?secretary|compliance.?officer",
     "mgmt_change"),
    (r"analyst|investor|con\.?\s*call|presentation|institutional",
     "analyst_meet"),
    (r"esop|esos|esps|allotment.?of.?secur|forfeiture",
     "share_action"),
    (r"credit.?rating", "credit_rating"),
    (r"sebi|trading.?window|trading.?plan|pit|depositories|certificate.?under",
     "regulatory"),
    (r"litigation|dispute|pendency|action.*taken|action.*initiated|order.*passed|clarification",
     "legal"),
    (r"press.?release|news.?verif|newspaper|copy.?of.?news",
     "press_release"),
    (r"shareholder|agm|egm|postal.?ballot",
     "shareholder_meet"),
    (r"record.?date", "record_date"),
    (r"related.?party", "related_party"),
    (r"update|general|business|monthly",
     "general_update"),
    (r"loss.?of.?share|duplicate",
     "admin_noise"),
]


def _classify_base_type(subject: str) -> str:
    """Classify a filing subject into a base type."""
    if not subject or not isinstance(subject, str):
        return "other"
    s = subject.lower().strip()
    for pattern, category in _CATEGORY_RULES:
        if re.search(pattern, s):
            return category
    return "other"


# ============================================================
# Stage 2: Body text → directional signal
# ============================================================

# Positive financial keywords (near profit/revenue/income context)
_POS_PATTERNS = [
    r"profit.{0,30}(?:increas|grew|growth|higher|surge|jump|rose|up\b|improv|beat|exceed|record)",
    r"(?:increas|grew|growth|higher|surge|jump|rose|improv).{0,30}(?:profit|revenue|income|earning)",
    r"revenue.{0,30}(?:increas|grew|growth|higher|surge|jump|rose|up\b|improv)",
    r"(?:net\s+)?profit\s+(?:after|before)\s+tax.{0,50}(?:increas|grew|higher|up\b)",
    r"(?:strong|robust|healthy|solid|stellar|exceptional)\s+(?:performance|result|quarter|growth)",
    r"margin.{0,20}(?:improv|expand|increas|higher|widen)",
    r"(?:beat|exceed|surpass).{0,20}(?:estimate|expectation|consensus|street)",
    r"(?:upgrade|upgrad|rais).{0,20}(?:rating|target|outlook|guidance)",
    r"(?:record|highest|all.time).{0,20}(?:revenue|profit|income|earning|quarter)",
    # YoY/QoQ growth patterns common in Indian filings
    r"(?:grew|growth|up|increas).{0,15}(?:\d+\.?\d*\s*%)",
    r"\d+\.?\d*\s*%\s*(?:growth|increase|higher|up\b|yoy|y-o-y|year.on.year)",
    r"(?:nii|net interest income|operating profit|pat|pbt).{0,30}(?:increas|grew|up\b|higher|growth)",
    r"(?:highest|record|best).{0,15}(?:ever|quarter|annual|yearly)",
]

# Negative financial keywords
_NEG_PATTERNS = [
    r"profit.{0,30}(?:decreas|declin|fell|drop|lower|down\b|slump|miss|below|weak|shrunk|contract)",
    r"(?:decreas|declin|fell|drop|lower|slump|contract).{0,30}(?:profit|revenue|income|earning)",
    r"revenue.{0,30}(?:decreas|declin|fell|drop|lower|down\b|slump|miss|contract)",
    r"(?:net\s+)?(?:loss|losses)\s+(?:of|at|worth|amount)",
    r"(?:weak|disappointing|subdued|muted|challenging|difficult)\s+(?:performance|result|quarter)",
    r"margin.{0,20}(?:compress|contract|declin|narrow|squeez|lower|shrunk)",
    r"(?:miss|below|short).{0,20}(?:estimate|expectation|consensus|street)",
    r"(?:downgrad|cut|lower).{0,20}(?:rating|target|outlook|guidance)",
    r"(?:impair|write.?off|write.?down|provision).{0,20}(?:loss|charge|expense)",
    # YoY/QoQ decline patterns
    r"(?:declin|fell|drop|decreas|down).{0,15}(?:\d+\.?\d*\s*%)",
    r"\d+\.?\d*\s*%\s*(?:decline|decrease|drop|fall|lower|down\b)",
    r"(?:nii|net interest income|operating profit|pat|pbt).{0,30}(?:declin|fell|drop|lower|decreas)",
    r"(?:sequential|qoq|q-o-q).{0,20}(?:declin|drop|fell|lower|decreas)",
]


def _scan_body_direction(body: str) -> str:
    """Scan body text for positive/negative financial signals.

    Returns: 'positive', 'negative', or 'neutral'
    """
    if not body or len(body) < 100:
        return "neutral"

    # Scan first 15000 chars — Indian filings often have tables before highlights
    text = body[:15000].lower()

    pos_hits = sum(1 for p in _POS_PATTERNS if re.search(p, text))
    neg_hits = sum(1 for p in _NEG_PATTERNS if re.search(p, text))

    if pos_hits > neg_hits and pos_hits >= 1:
        return "positive"
    elif neg_hits > pos_hits and neg_hits >= 1:
        return "negative"
    return "neutral"


# ============================================================
# Stage 3: Combine type + direction → influencing factor
# ============================================================

# High-impact types that get directional suffixes
_DIRECTIONAL_TYPES = {
    "earnings", "board_outcome", "press_release", "general_update",
}

# Types with inherent direction (no body scan needed)
_INHERENT_FACTOR = {
    "dividend": "dividend_announced",
    "m_and_a": "m_and_a_activity",
    "mgmt_change": "mgmt_change",
    "analyst_meet": "analyst_meet",
    "share_action": "share_action",
    "credit_rating": "credit_rating",  # will get direction from body
    "regulatory": "regulatory",
    "legal": "legal_action",
    "shareholder_meet": "shareholder_meet",
    "record_date": "record_date",
    "related_party": "related_party",
    "admin_noise": "admin_noise",
    "other": "other",
}


def classify_filing(subject: str, body: str = "") -> str:
    """Classify a filing into an influencing-factor category.

    Combines subject-based type with body-text directional signals.
    Examples:
        - earnings + positive body → earnings_positive
        - earnings + negative body → earnings_negative
        - earnings + neutral body  → earnings_neutral
        - dividend (any)           → dividend_announced
        - credit_rating + positive → credit_rating_positive
        - admin_noise              → admin_noise
    """
    base_type = _classify_base_type(subject)

    # Credit rating also gets directional suffix
    if base_type == "credit_rating":
        direction = _scan_body_direction(body)
        if direction != "neutral":
            return f"credit_rating_{direction}"
        return "credit_rating_neutral"

    # High-impact types: append direction from body scan
    if base_type in _DIRECTIONAL_TYPES:
        direction = _scan_body_direction(body)
        return f"{base_type}_{direction}"

    # Inherent-factor types
    return _INHERENT_FACTOR.get(base_type, "other")


# All possible influencing-factor categories
FILING_CATEGORIES = [
    # Directional earnings/board/press/updates
    "earnings_positive", "earnings_negative", "earnings_neutral",
    "board_outcome_positive", "board_outcome_negative", "board_outcome_neutral",
    "press_release_positive", "press_release_negative", "press_release_neutral",
    "general_update_positive", "general_update_negative", "general_update_neutral",
    "credit_rating_positive", "credit_rating_negative", "credit_rating_neutral",
    # Inherent-factor types
    "dividend_announced", "m_and_a_activity",
    "mgmt_change", "analyst_meet", "share_action",
    "regulatory", "legal_action",
    "shareholder_meet", "record_date", "related_party",
    "admin_noise", "other",
]


# ============================================================
# Loader
# ============================================================

def load_filings(
    filings_dir: str,
    symbols: List[str],
    col_map: Dict[str, str],
    market_open: str = "09:15",
    market_close: str = "15:30",
    dry_run: bool = False,
) -> pd.DataFrame:
    """Load all filings from date-organized folder structure.

    Expected structure: filings_dir/<date>/<SYMBOL>_<HHMMSS>.csv
    Each CSV has columns: symbol, date, timestamp, subject, body_text, pdf_text, xbrl_text, ...

    Returns DataFrame with columns:
        symbol, raw_timestamp, assigned_date, headline, body,
        has_body, headline_only, filing_category

    If dry_run=True, returns an empty DataFrame with the correct schema
    instead of reading any data files.
    """
    if dry_run:
        logger.info("[DRY RUN] Skipping filings data loading")
        return pd.DataFrame(columns=[
            "symbol", "raw_timestamp", "assigned_date",
            "headline", "body", "has_body", "headline_only", "filing_category"
        ])

    filings_path = Path(filings_dir)
    if not filings_path.exists():
        raise FileNotFoundError(f"Filings directory not found: {filings_dir}")

    symbols_upper = {s.upper() for s in symbols}
    from datetime import time as dtime
    mo = dtime(*map(int, market_open.split(":")))
    mc = dtime(*map(int, market_close.split(":")))

    all_rows = []
    date_dirs = sorted([d for d in filings_path.iterdir() if d.is_dir()])
    logger.info(f"Scanning {len(date_dirs)} date folders in {filings_dir}")

    for date_dir in date_dirs:
        for csv_file in date_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, encoding="utf-8", on_bad_lines="skip")
            except Exception as e:
                logger.debug(f"Skipping {csv_file}: {e}")
                continue

            if df.empty:
                continue

            sym_col = col_map.get("symbol", "symbol")
            ts_col = col_map.get("timestamp", "timestamp")
            headline_col = col_map.get("headline", "subject")
            body_col = col_map.get("body", "pdf_text")
            body_fb_col = col_map.get("body_fallback", "body_text")

            for _, row in df.iterrows():
                sym = str(row.get(sym_col, "")).strip().upper()
                if sym not in symbols_upper:
                    continue

                # Parse timestamp
                raw_ts = row.get(ts_col, "")
                try:
                    ts = pd.to_datetime(raw_ts)
                    if ts.tzinfo is None:
                        ts = IST.localize(ts)
                except Exception:
                    continue

                # Headline
                headline = str(row.get(headline_col, "") or "").strip()

                # Body: prefer pdf_text, fallback to body_text
                body = str(row.get(body_col, "") or "").strip()
                if not body or body == "nan":
                    body = str(row.get(body_fb_col, "") or "").strip()
                    if body == "nan":
                        body = ""

                has_body = len(body) > 50
                headline_only = not has_body
                text = body if has_body else headline

                # First-seen session assignment
                assigned = assign_filing_to_session(ts, mo, mc)

                # Classify filing → influencing factor
                filing_cat = classify_filing(headline, text)

                all_rows.append({
                    "symbol": sym,
                    "raw_timestamp": ts,
                    "assigned_date": pd.Timestamp(assigned),
                    "headline": headline,
                    "body": text,
                    "has_body": has_body,
                    "headline_only": headline_only,
                    "filing_category": filing_cat,
                })

    if not all_rows:
        logger.warning("No filings loaded. Check filings_dir and symbol list.")
        return pd.DataFrame(columns=[
            "symbol", "raw_timestamp", "assigned_date",
            "headline", "body", "has_body", "headline_only", "filing_category"
        ])

    result = pd.DataFrame(all_rows)
    result = result.sort_values(
        ["symbol", "assigned_date", "raw_timestamp"]
    ).reset_index(drop=True)

    logger.info(
        f"Loaded {len(result)} filings for {result['symbol'].nunique()} symbols, "
        f"date range: {result['assigned_date'].min().date()} to "
        f"{result['assigned_date'].max().date()}"
    )
    cat_counts = result["filing_category"].value_counts()
    logger.info(f"Filing influencing factors:\n{cat_counts.to_string()}")
    return result
