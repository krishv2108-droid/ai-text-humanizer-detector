"""Rule-based controls: what do the detectors do when ONLY punctuation or sentence boundaries change?

If a trivial, meaning-preserving edit moves a detector, that detector is reacting to surface formatting, and beating it
says little about text quality. The same transforms serve as a no-LLM baseline for the humanizer.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from hd.eval.loop_report import wilson_ci


def control_table(
    texts: list[str],
    transforms: dict[str, Callable[[str], str]],
    score_ours: Callable[[list[str]], np.ndarray],
    score_external: Callable[[list[str]], np.ndarray],
    cut_ours: float,
    cut_external: float,
) -> pd.DataFrame:
    """One row per transform ('original' is always first): the share flagged by each detector at its frozen cutoff, with intervals."""
    rows = []
    for name, fn in {"original": lambda t: t, **transforms}.items():
        out = [fn(t) for t in texts]
        o, e, n = np.asarray(score_ours(out)), np.asarray(score_external(out)), len(out)
        ko, ke = int((o > cut_ours).sum()), int((e > cut_external).sum())
        rows.append({
            "variant": name, "n": n, "ours_flagged": ko / n, "ours_lo": wilson_ci(ko, n)[0], "ours_hi": wilson_ci(ko, n)[1],
            "desklib_flagged": ke / n, "desklib_lo": wilson_ci(ke, n)[0], "desklib_hi": wilson_ci(ke, n)[1], "desklib_mean": float(e.mean()),
        })  # fmt: skip
    return pd.DataFrame(rows)
