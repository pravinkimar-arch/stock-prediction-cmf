"""Text sentiment features using frozen FinBERT.

Extracts per-document sentiment, pools to daily features per symbol,
and computes memory features (count, recency, time-since-last).
Caches per-document sentiment scores to avoid re-running FinBERT.
"""

import hashlib
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy-loaded model globals
_tokenizer = None
_model = None
_device = None


def _load_model(model_name: str = "ProsusAI/finbert"):
    """Load frozen FinBERT model (singleton)."""
    global _tokenizer, _model, _device
    if _model is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    logger.info(f"Loading frozen sentiment model: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSequenceClassification.from_pretrained(model_name)
    _model.eval()
    for param in _model.parameters():
        param.requires_grad = False
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)
    logger.info(f"Model loaded on {_device}")


def _chunk_text(text: str, max_tokens: int = 510, overlap: int = 50) -> List[str]:
    """Split text into chunks that fit within model's token limit."""
    words = text.split()
    if not words:
        return [text]

    approx_words_per_chunk = int(max_tokens / 1.3)
    overlap_words = int(overlap / 1.3)

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + approx_words_per_chunk, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap_words

    return chunks if chunks else [text]


def _text_hash(text: str) -> str:
    """Short hash of text for cache key."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _load_sentiment_cache(cache_path: Path) -> Dict[str, Dict[str, float]]:
    """Load cached sentiment scores keyed by text hash."""
    if not cache_path.exists():
        return {}
    try:
        df = pd.read_parquet(cache_path)
        cache = {}
        for _, row in df.iterrows():
            cache[row["text_hash"]] = {
                "p_pos": row["p_pos"], "p_neg": row["p_neg"],
                "p_neu": row["p_neu"], "polarity": row["polarity"],
            }
        logger.info(f"Loaded {len(cache)} cached sentiment scores from {cache_path}")
        return cache
    except Exception as e:
        logger.warning(f"Failed to load sentiment cache: {e}")
        return {}


def _save_sentiment_cache(
    cache: Dict[str, Dict[str, float]], cache_path: Path
):
    """Save sentiment cache to parquet."""
    rows = [{"text_hash": k, **v} for k, v in cache.items()]
    if rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(cache_path, index=False)
        logger.info(f"Saved {len(rows)} sentiment scores to cache: {cache_path}")


def infer_sentiment(
    texts: List[str],
    model_name: str = "ProsusAI/finbert",
    max_chunk_tokens: int = 510,
    chunk_overlap: int = 50,
    batch_size: int = 8,
    cache_dir: Optional[str] = None,
    dry_run: bool = False,
) -> List[Dict[str, float]]:
    """Run frozen FinBERT on a list of texts, with caching.

    For each text:
    1. Check cache by text hash
    2. If miss: chunk, run model, average logits, softmax
    3. Save new scores to cache

    Returns list of dicts with keys: p_pos, p_neg, p_neu, polarity
    """
    import torch

    # Load cache
    cache_path = Path(cache_dir or "cache") / "sentiment_cache.parquet"
    cache = _load_sentiment_cache(cache_path)

    # Identify which texts need inference
    hashes = [_text_hash(t) for t in texts]
    results = [None] * len(texts)
    to_infer = []  # (index, text) pairs for uncached texts

    for i, (h, text) in enumerate(zip(hashes, texts)):
        if h in cache:
            results[i] = cache[h]
        else:
            to_infer.append((i, text))

    cached_count = len(texts) - len(to_infer)
    logger.info(f"Sentiment cache: {cached_count}/{len(texts)} hits, {len(to_infer)} to infer")

    if to_infer:
        _load_model(model_name)

        total = len(to_infer)
        t_start = time.time()
        log_interval = max(1, total // 20)
        save_interval = 50  # save cache every N docs so Ctrl+C doesn't lose progress
        unsaved_count = 0

        for step, (idx, text) in enumerate(to_infer):
            if not text or len(text.strip()) < 5:
                score = {"p_pos": 0.33, "p_neg": 0.33, "p_neu": 0.34, "polarity": 0.0}
            else:
                chunks = _chunk_text(text, max_chunk_tokens, chunk_overlap)
                all_logits = []

                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i: i + batch_size]
                    inputs = _tokenizer(
                        batch, return_tensors="pt", truncation=True,
                        max_length=max_chunk_tokens + 2, padding=True,
                    )
                    inputs = {k: v.to(_device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = _model(**inputs)
                        all_logits.append(outputs.logits.cpu().numpy())

                avg_logits = np.mean(np.concatenate(all_logits, axis=0), axis=0)
                exp_logits = np.exp(avg_logits - avg_logits.max())
                probs = exp_logits / exp_logits.sum()

                score = {
                    "p_pos": float(probs[0]), "p_neg": float(probs[1]),
                    "p_neu": float(probs[2]), "polarity": float(probs[0] - probs[1]),
                }

            results[idx] = score
            cache[hashes[idx]] = score
            unsaved_count += 1

            # Progress + ETA
            done = step + 1
            if done == 1 or done == total or done % log_interval == 0:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta_min = ((total - done) / rate / 60) if rate > 0 else 0
                pct = done / total * 100
                logger.info(
                    f"  Sentiment: {done}/{total} new docs ({pct:.0f}%) | "
                    f"{rate:.1f} docs/sec | ETA: {eta_min:.1f} min"
                )

            # Periodic save — survives Ctrl+C
            if not dry_run and unsaved_count >= save_interval:
                _save_sentiment_cache(cache, cache_path)
                unsaved_count = 0

        # Final save
        if not dry_run:
            _save_sentiment_cache(cache, cache_path)
        else:
            logger.info("Dry run: skipping sentiment cache write")

    return results


def compute_daily_text_features(
    filings: pd.DataFrame,
    daily_dates: pd.DataFrame,
    text_cfg: Dict,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Compute daily text features per symbol.

    Caches:
      - Per-document sentiment scores: cache/sentiment_cache.parquet
      - Processed filings with sentiment + categories: cache/filings_processed.parquet
      - Final daily text features: cache/daily_text_features.parquet

    Args:
        filings: DataFrame with columns [symbol, assigned_date, body, has_body, headline_only]
        daily_dates: DataFrame with [symbol, date] for all trading days
        text_cfg: config dict for text features

    Returns:
        DataFrame with [symbol, date] + text feature columns
    """
    model_name = text_cfg.get("model_name", "ProsusAI/finbert")
    max_chunk = text_cfg.get("max_chunk_tokens", 510)
    overlap = text_cfg.get("chunk_overlap_tokens", 50)
    batch_sz = text_cfg.get("batch_size", 8)
    mem_W = text_cfg.get("text_memory_W", 20)
    mem_h = text_cfg.get("text_memory_h", 5.0)
    cache_dir_str = text_cfg.get("cache_dir", "cache")
    cache_dir = Path(cache_dir_str)
    if not dry_run:
        cache_dir.mkdir(parents=True, exist_ok=True)

    if filings.empty:
        logger.warning("No filings data. Returning empty text features.")
        return _empty_text_features(daily_dates)

    # --- Check daily text features cache ---
    daily_cache_path = cache_dir / "daily_text_features.parquet"
    if daily_cache_path.exists():
        try:
            cached = pd.read_parquet(daily_cache_path)
            # Validate: same symbols and date range
            cached_syms = set(cached["symbol"].unique())
            needed_syms = set(daily_dates["symbol"].unique())
            if needed_syms.issubset(cached_syms) and len(cached) > 0:
                # Filter to requested symbols/dates
                result = daily_dates[["symbol", "date"]].merge(
                    cached, on=["symbol", "date"], how="left",
                )
                missing_cols = [c for c in TEXT_FEATURE_COLS if c not in result.columns]
                if not missing_cols:
                    for col in TEXT_FEATURE_COLS:
                        result[col] = result[col].fillna(0)
                    logger.info(
                        f"Loaded daily text features from cache: {daily_cache_path} "
                        f"({len(result)} rows)"
                    )
                    return result
                else:
                    logger.info(
                        f"Cache missing columns {missing_cols}, recomputing..."
                    )
        except Exception as e:
            logger.warning(f"Failed to load daily text features cache: {e}")

    # --- Check processed filings cache ---
    filings_cache_path = cache_dir / "filings_processed.parquet"
    filings_cached = False
    if filings_cache_path.exists():
        try:
            filings_proc = pd.read_parquet(filings_cache_path)
            needed_cols = {"symbol", "assigned_date", "p_pos", "p_neg",
                           "p_neu", "polarity", "headline_only", "filing_category"}
            if needed_cols.issubset(set(filings_proc.columns)):
                cached_count = len(filings_proc)
                current_count = len(filings)
                # Use cache if same size (no new filings added)
                if cached_count == current_count:
                    logger.info(
                        f"Loaded processed filings from cache: "
                        f"{filings_cache_path} ({cached_count} docs)"
                    )
                    filings_cached = True
                else:
                    logger.info(
                        f"Filings count changed ({cached_count} → {current_count}), "
                        f"recomputing sentiment..."
                    )
        except Exception as e:
            logger.warning(f"Failed to load filings cache: {e}")

    if not filings_cached:
        # Step 1: Run sentiment on all documents (with per-doc caching)
        logger.info(f"Running sentiment inference on {len(filings)} documents...")
        texts = filings["body"].tolist()
        sentiments = infer_sentiment(
            texts, model_name, max_chunk, overlap, batch_sz,
            cache_dir=cache_dir_str, dry_run=dry_run,
        )

        filings_proc = filings.copy()
        filings_proc["p_pos"] = [s["p_pos"] for s in sentiments]
        filings_proc["p_neg"] = [s["p_neg"] for s in sentiments]
        filings_proc["p_neu"] = [s["p_neu"] for s in sentiments]
        filings_proc["polarity"] = [s["polarity"] for s in sentiments]

        # Save processed filings cache (drop heavy body text to save space)
        cache_cols = [
            "symbol", "assigned_date", "headline", "has_body", "headline_only",
            "filing_category", "p_pos", "p_neg", "p_neu", "polarity",
        ]
        save_cols = [c for c in cache_cols if c in filings_proc.columns]
        if not dry_run:
            filings_proc[save_cols].to_parquet(filings_cache_path, index=False)
            logger.info(f"Cached processed filings to {filings_cache_path}")
        else:
            logger.info("Dry run: skipping filings_processed.parquet write")

    # Step 2: Daily pooling per symbol — sentiment aggregates
    filings_proc["assigned_date"] = pd.to_datetime(filings_proc["assigned_date"])
    daily_pool = filings_proc.groupby(["symbol", "assigned_date"]).agg(
        doc_count=("polarity", "size"),
        mean_polarity=("polarity", "mean"),
        max_polarity=("polarity", "max"),
        mean_p_pos=("p_pos", "mean"),
        mean_p_neg=("p_neg", "mean"),
        any_headline_only=("headline_only", "max"),
    ).reset_index()
    daily_pool = daily_pool.rename(columns={"assigned_date": "date"})

    # Step 2b: Filing category counts per symbol per day
    from src.data.filings_loader import FILING_CATEGORIES
    if "filing_category" in filings_proc.columns:
        cat_dummies = pd.get_dummies(filings_proc["filing_category"], prefix="cat")
        cat_df = filings_proc[["symbol", "assigned_date"]].copy()
        cat_df = pd.concat([cat_df, cat_dummies], axis=1)
        cat_pool = cat_df.groupby(["symbol", "assigned_date"]).sum().reset_index()
        cat_pool = cat_pool.rename(columns={"assigned_date": "date"})
        # Ensure all category columns exist
        for cat in FILING_CATEGORIES:
            col = f"cat_{cat}"
            if col not in cat_pool.columns:
                cat_pool[col] = 0
        daily_pool = daily_pool.merge(cat_pool, on=["symbol", "date"], how="left")

    # Step 3: Merge with all trading days
    result = daily_dates[["symbol", "date"]].copy()
    result = result.merge(daily_pool, on=["symbol", "date"], how="left")

    # Fill missing days
    result["no_filings_day"] = result["doc_count"].isna().astype(int)
    result["doc_count"] = result["doc_count"].fillna(0)
    result["mean_polarity"] = result["mean_polarity"].fillna(0)
    result["max_polarity"] = result["max_polarity"].fillna(0)
    result["mean_p_pos"] = result["mean_p_pos"].fillna(0)
    result["mean_p_neg"] = result["mean_p_neg"].fillna(0)
    result["any_headline_only"] = result["any_headline_only"].fillna(0).astype(int)
    # Fill category columns
    for cat in FILING_CATEGORIES:
        col = f"cat_{cat}"
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)
        else:
            result[col] = 0

    # Step 4: Memory features + recency-weighted polarity EMA
    result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
    result = _compute_text_memory(result, mem_W, mem_h)

    # Cache final daily text features
    if not dry_run:
        try:
            result.to_parquet(daily_cache_path, index=False)
            logger.info(f"Cached daily text features to {daily_cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache daily text features: {e}")
    else:
        logger.info("Dry run: skipping daily_text_features.parquet write")

    logger.info(f"Daily text features computed: {result.shape}")
    return result


def _compute_text_memory(df: pd.DataFrame, W: int, h: float) -> pd.DataFrame:
    """Compute memory features for text signals + recency-weighted polarity EMA."""
    alpha = 1 - np.power(0.5, 1.0 / h)  # EMA decay factor from half-life

    frames = []
    for sym, gdf in df.groupby("symbol"):
        sdf = gdf.sort_values("date").copy()
        n = len(sdf)

        for signal_col, prefix in [("doc_count", "filing"), ("mean_polarity", "polarity")]:
            sig = sdf[signal_col].values
            has_signal = (sig > 0).astype(float) if signal_col == "doc_count" else (sig != 0).astype(float)

            counts = np.full(n, np.nan)
            recency = np.full(n, np.nan)
            tsl = np.full(n, np.nan)

            for i in range(n):
                start = max(0, i - W + 1)
                window = has_signal[start: i + 1]

                counts[i] = np.nansum(window)
                ages = np.arange(len(window) - 1, -1, -1, dtype=float)
                weights = np.power(0.5, ages / h)
                recency[i] = np.nansum(window * weights)

                occ = np.where(window > 0)[0]
                tsl[i] = (len(window) - 1 - occ[-1]) if len(occ) > 0 else min(len(window), W)

            sdf[f"{prefix}_count_{W}"] = counts
            sdf[f"{prefix}_recency_{W}"] = recency
            sdf[f"{prefix}_tsl"] = tsl

        # Recency-weighted polarity EMA: gives more weight to latest sentiment
        polarity_vals = sdf["mean_polarity"].values
        ema = np.zeros(n)
        ema[0] = polarity_vals[0]
        for i in range(1, n):
            ema[i] = alpha * polarity_vals[i] + (1 - alpha) * ema[i - 1]
        sdf["polarity_ema"] = ema

        frames.append(sdf)

    return pd.concat(frames, ignore_index=True)


def _empty_text_features(daily_dates: pd.DataFrame) -> pd.DataFrame:
    """Return empty text features when no filings available."""
    result = daily_dates[["symbol", "date"]].copy()
    for col in TEXT_FEATURE_COLS:
        result[col] = 0
    result["no_filings_day"] = 1
    return result


# Category column names
from src.data.filings_loader import FILING_CATEGORIES
_CATEGORY_FEATURE_COLS = [f"cat_{cat}" for cat in FILING_CATEGORIES]

TEXT_FEATURE_COLS = [
    "doc_count", "mean_polarity", "max_polarity",
    "mean_p_pos", "mean_p_neg",
    "no_filings_day", "any_headline_only",
    "filing_count_20", "filing_recency_20", "filing_tsl",
    "polarity_count_20", "polarity_recency_20", "polarity_tsl",
    "polarity_ema",
] + _CATEGORY_FEATURE_COLS
