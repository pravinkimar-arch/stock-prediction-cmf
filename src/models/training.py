"""Model training routines for Logistic Regression and LightGBM.

Scalers are fit only on training data per split (no global fitting).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Dict,
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train LR with per-split scaling.

    Returns (model, scaler) tuple.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        C=cfg.get("C", 1.0),
        max_iter=cfg.get("max_iter", 1000),
        solver=cfg.get("solver", "lbfgs"),
        penalty=cfg.get("penalty", "l2"),
        random_state=42,
    )
    model.fit(X_scaled, y_train)
    logger.info(f"  LR trained: {X_scaled.shape[1]} features, {len(y_train)} samples")
    return model, scaler


def predict_logistic_regression(
    model: LogisticRegression,
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """Predict probabilities using LR."""
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Dict,
    feature_names: Optional[List[str]] = None,
) -> "lightgbm.LGBMClassifier":
    """Train LightGBM with early stopping on validation set.

    No scaler needed for tree models.
    """
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=cfg.get("n_estimators", 300),
        max_depth=cfg.get("max_depth", 4),
        learning_rate=cfg.get("learning_rate", 0.05),
        num_leaves=cfg.get("num_leaves", 15),
        min_child_samples=cfg.get("min_child_samples", 20),
        subsample=cfg.get("subsample", 0.8),
        colsample_bytree=cfg.get("colsample_bytree", 0.8),
        reg_alpha=cfg.get("reg_alpha", 0.1),
        reg_lambda=cfg.get("reg_lambda", 0.1),
        random_state=42,
        verbose=-1,
    )

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    logger.info(
        f"  LGBM trained: {X_train.shape[1]} features, "
        f"best_iteration={model.best_iteration_}"
    )
    return model


def predict_lightgbm(model, X: np.ndarray) -> np.ndarray:
    """Predict probabilities using LightGBM."""
    return model.predict_proba(X)[:, 1]
