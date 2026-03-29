"""Evaluation metrics: discrimination, calibration, and reporting.

Includes ROC-AUC, PR-AUC, F1, Brier score, ECE, and reliability analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    brier_score_loss, precision_recall_curve, roc_curve,
)
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error.

    Uses right-inclusive last bin to ensure p=1.0 values are included.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return ece


def compute_reliability_stats(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Dict:
    """Compute reliability curve statistics.

    Returns bin midpoints, observed frequencies, and linear regression
    slope/intercept of the reliability curve.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    midpoints = []
    observed = []

    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() < 5:
            continue
        midpoints.append(probs[mask].mean())
        observed.append(labels[mask].mean())

    midpoints = np.array(midpoints)
    observed = np.array(observed)

    if len(midpoints) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(midpoints, observed)
        r_sq = float(r_value ** 2)
    else:
        slope, intercept, r_sq = np.nan, np.nan, np.nan

    return {
        "midpoints": midpoints.tolist(),
        "observed": observed.tolist(),
        "slope": float(slope) if not np.isnan(slope) else np.nan,
        "intercept": float(intercept) if not np.isnan(intercept) else np.nan,
        "r_squared": float(r_sq) if not np.isnan(r_sq) else np.nan,
    }


def compute_all_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """Compute all discrimination and calibration metrics.

    Args:
        probs: predicted probabilities.
        labels: binary ground truth.
        threshold: classification threshold for F1.

    Returns:
        Dict with all metric values.
    """
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class in labels. Metrics may be undefined.")
        return {
            "roc_auc": np.nan, "pr_auc": np.nan, "f1": np.nan,
            "brier": np.nan, "ece": np.nan,
            "reliability_slope": np.nan, "reliability_intercept": np.nan,
            "n_samples": len(labels), "pos_rate": float(labels.mean()),
        }

    preds = (probs >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "f1": float(f1_score(labels, preds)),
        "brier": float(brier_score_loss(labels, probs)),
        "ece": float(compute_ece(probs, labels)),
        "n_samples": len(labels),
        "pos_rate": float(labels.mean()),
    }

    rel = compute_reliability_stats(probs, labels)
    metrics["reliability_slope"] = rel["slope"]
    metrics["reliability_intercept"] = rel["intercept"]

    return metrics


def build_results_table(
    all_results: List[Dict],
) -> pd.DataFrame:
    """Build a tidy results table from per-fold, per-variant results."""
    rows = []
    for r in all_results:
        rows.append({
            "fold": r["fold"],
            "variant": r["variant"],
            "split": r["split"],  # "val" or "test"
            **r["metrics"],
        })
    return pd.DataFrame(rows)


def compute_delta_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute delta metrics between variants.

    Returns DataFrame with deltas:
    - (Numeric+Tokens) - (Numeric-only)
    - (TS+Text) - (Best TS-only)
    """
    test_results = results_df[results_df["split"] == "test"].copy()

    metric_cols = ["roc_auc", "pr_auc", "f1", "brier", "ece"]
    deltas = []

    for fold in test_results["fold"].unique():
        fold_data = test_results[test_results["fold"] == fold]

        numeric_only = fold_data[fold_data["variant"] == "numeric_only"]
        numeric_tokens = fold_data[fold_data["variant"] == "numeric_tokens"]
        ts_text = fold_data[fold_data["variant"] == "ts_text_fusion"]

        if len(numeric_only) > 0 and len(numeric_tokens) > 0:
            delta = {}
            delta["fold"] = fold
            delta["comparison"] = "tokens_vs_numeric"
            for m in metric_cols:
                v1 = numeric_tokens[m].values[0]
                v0 = numeric_only[m].values[0]
                delta[f"delta_{m}"] = v1 - v0
            deltas.append(delta)

        if len(numeric_tokens) > 0 and len(ts_text) > 0:
            delta = {}
            delta["fold"] = fold
            delta["comparison"] = "fusion_vs_tokens"
            for m in metric_cols:
                v1 = ts_text[m].values[0]
                v0 = numeric_tokens[m].values[0]
                delta[f"delta_{m}"] = v1 - v0
            deltas.append(delta)

    return pd.DataFrame(deltas) if deltas else pd.DataFrame()
