"""Probability calibration: Platt scaling and temperature scaling.

Calibration is fit per walk-forward split using validation data only.
"""

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class PlattCalibrator:
    """Platt scaling: fit a logistic regression on raw probabilities."""

    def __init__(self):
        self.model = None

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """Fit Platt scaling on validation probabilities."""
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        log_odds = np.log(probs / (1 - probs)).reshape(-1, 1)

        self.model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        self.model.fit(log_odds, labels)
        logger.info("  Platt calibrator fitted")

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to raw probabilities."""
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        log_odds = np.log(probs / (1 - probs)).reshape(-1, 1)
        return self.model.predict_proba(log_odds)[:, 1]


class TemperatureCalibrator:
    """Temperature scaling: find optimal temperature on validation set."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """Find temperature that minimizes NLL on validation."""
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        log_odds = np.log(probs / (1 - probs))

        def nll(T):
            scaled = 1.0 / (1.0 + np.exp(-log_odds / T))
            scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
            return -np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = result.x
        logger.info(f"  Temperature calibrator fitted: T={self.temperature:.4f}")

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        log_odds = np.log(probs / (1 - probs))
        return 1.0 / (1.0 + np.exp(-log_odds / self.temperature))


def get_calibrator(method: str = "platt"):
    """Factory for calibrator objects."""
    if method == "platt":
        return PlattCalibrator()
    elif method == "temperature":
        return TemperatureCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
