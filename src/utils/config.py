"""Configuration loader and accessor."""

import yaml
import os
from pathlib import Path
from typing import Any, Dict


def load_config(path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load YAML config, resolve paths relative to project root."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_nested(cfg: Dict, dotpath: str, default=None):
    """Access nested config via dot notation: 'features.numeric.atr_window'."""
    keys = dotpath.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val
