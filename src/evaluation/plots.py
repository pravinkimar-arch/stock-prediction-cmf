"""Plotting utilities for reliability curves and results visualization."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_reliability_curves(
    results: List[Dict],
    output_dir: str = "outputs/",
):
    """Plot reliability curves for each variant across folds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Group by variant
    variants = {}
    for r in results:
        if r["split"] != "test":
            continue
        v = r["variant"]
        if v not in variants:
            variants[v] = []
        if "reliability" in r:
            variants[v].append(r["reliability"])

    fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 5), squeeze=False)

    for idx, (variant, reliabilities) in enumerate(variants.items()):
        ax = axes[0][idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

        for i, rel in enumerate(reliabilities):
            if rel and "midpoints" in rel:
                ax.plot(rel["midpoints"], rel["observed"], "o-", alpha=0.6, label=f"Fold {i}")

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(f"{variant}")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out / "reliability_curves.png", dpi=150)
    plt.close()
    logger.info(f"Reliability curves saved to {out / 'reliability_curves.png'}")


def plot_metrics_comparison(
    results_df: pd.DataFrame,
    output_dir: str = "outputs/",
):
    """Bar chart comparing variants on key metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    test_df = results_df[results_df["split"] == "test"]
    if test_df.empty:
        return

    metrics = ["roc_auc", "pr_auc", "f1", "brier", "ece"]
    summary = test_df.groupby("variant")[metrics].agg(["mean", "std"])

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))
    variants = summary.index.tolist()
    x = np.arange(len(variants))

    for i, m in enumerate(metrics):
        means = summary[(m, "mean")].values
        stds = summary[(m, "std")].values
        axes[i].bar(x, means, yerr=stds, capsize=3, alpha=0.7)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
        axes[i].set_title(m)

    plt.tight_layout()
    plt.savefig(out / "metrics_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Metrics comparison saved to {out / 'metrics_comparison.png'}")
