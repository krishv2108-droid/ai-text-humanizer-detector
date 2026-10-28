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


def load_env(path: str | Path | None = None) -> bool:
    """Read KEY=VALUE lines from the git-ignored `.env` file into os.environ (never overriding a
    variable that is already set). Returns True if a file was read. Used for ANTHROPIC_API_KEY."""
    import os

    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.exists():
        return False
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip("\"'"))
    return True


def set_seed(seed: int) -> None:
    """Seed every source of randomness we use so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional extra
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
