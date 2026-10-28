"""Config loading and global seeding."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a `paths.<key>` entry relative to the project root."""
    return PROJECT_ROOT / cfg["paths"][key]


def set_seed(seed: int) -> None:
    """Seed every source of randomness we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional extra
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
