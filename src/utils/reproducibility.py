"""Reproducibility: seeding, environment logging."""

import random
import platform
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info(f"Random seed set to {seed}")


def log_environment(output_dir: str = "outputs/"):
    """Log environment info for reproducibility."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {},
    }
    for pkg in ["numpy", "pandas", "sklearn", "lightgbm", "transformers", "torch", "scipy", "yaml"]:
        try:
            mod = __import__(pkg)
            info["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            info["packages"][pkg] = "not installed"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "environment.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Environment info saved to {out / 'environment.json'}")
    return info


def setup_logging(level: str = "INFO"):
    """Configure root logger."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
