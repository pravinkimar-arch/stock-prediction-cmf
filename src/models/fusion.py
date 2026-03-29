"""Late fusion at probability level.

Combines calibrated probabilities from TS branch and Text branch.
"""

import logging
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class WeightedAverageFusion:
    """Learn optimal weights on validation set via grid search."""

    def __init__(self):
        self.weight_ts = 0.5

    def fit(self, p_ts: np.ndarray, p_text: np.ndarray, labels: np.ndarray):
        """Find weight that minimizes Brier score on validation."""
        best_w = 0.5
        best_brier = float("inf")

        for w in np.arange(0.0, 1.05, 0.05):
            fused = w * p_ts + (1 - w) * p_text
            brier = np.mean((fused - labels) ** 2)
            if brier < best_brier:
                best_brier = brier
                best_w = w

        self.weight_ts = best_w
        logger.info(f"  Fusion weights: TS={best_w:.2f}, Text={1-best_w:.2f} (Brier={best_brier:.4f})")

    def predict(self, p_ts: np.ndarray, p_text: np.ndarray) -> np.ndarray:
        return self.weight_ts * p_ts + (1 - self.weight_ts) * p_text


class MetaLRFusion:
    """Logistic regression meta-model on calibrated probabilities."""

    def __init__(self):
        self.model = None

    def fit(self, p_ts: np.ndarray, p_text: np.ndarray, labels: np.ndarray):
        """Fit meta-LR on validation calibrated probabilities."""
        X = np.column_stack([p_ts, p_text])
        self.model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self.model.fit(X, labels)
        logger.info("  Meta-LR fusion fitted")

    def predict(self, p_ts: np.ndarray, p_text: np.ndarray) -> np.ndarray:
        X = np.column_stack([p_ts, p_text])
        return self.model.predict_proba(X)[:, 1]


def get_fusion_model(method: str = "weighted_average"):
    """Factory for fusion models."""
    if method == "weighted_average":
        return WeightedAverageFusion()
    elif method == "meta_lr":
        return MetaLRFusion()
    else:
        raise ValueError(f"Unknown fusion method: {method}")
