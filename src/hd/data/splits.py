"""Prompt-grouped train/val/test assignment with a guarded test set.

Every text that answers the same prompt shares one `prompt_id`, and the split is a pure function of
`(seed, prompt_id)`. Two consequences:
  * a human/AI pair can never straddle train and test (the detector can't memorize a prompt);
  * adding more data later never moves an existing prompt to a different split.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")


class TestSetLockedError(RuntimeError):
    __test__ = False  # not a pytest class
    pass


def stable_hash01(*parts: object) -> float:
    """Deterministic float in [0, 1) from the given parts (independent of Python's hash seed)."""
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def assign_split(prompt_id: str, seed: int, fractions: dict[str, float]) -> str:
    u = stable_hash01(seed, "split", prompt_id)
    edge = fractions["train"]
    if u < edge:
        return "train"
    edge += fractions["val"]
    return "val" if u < edge else "test"


def add_splits(df: pd.DataFrame, seed: int, fractions: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["split"] = [assign_split(p, seed, fractions) for p in out["prompt_id"]]
    return out


def load_split(
    corpus: pd.DataFrame,
    name: str,
    *,
    unlock_test: bool = False,
    access_log: Path | None = None,
) -> pd.DataFrame:
    """Return one split. The test split needs `unlock_test=True`, and each unlock is logged."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLITS}")
    if name == "test":
        if not unlock_test:
            raise TestSetLockedError(
                "The test split is locked. It is used once, for final evaluation "
                "(pass unlock_test=True there). Never use it for training or tuning."
            )
        if access_log is not None:
            access_log.parent.mkdir(parents=True, exist_ok=True)
            with open(access_log, "a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} test split unlocked\n")
    return corpus[corpus["split"] == name].reset_index(drop=True)
