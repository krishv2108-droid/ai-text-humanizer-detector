"""How often does a rewrite get past every detector, once a further judge (MAGE) is added to the three in the loop logs?

Reads only existing loop logs plus the VALIDATION split (for the judge's 5%-false-positive cutoff); it never opens the test split.
The "detectors" counted here are: ours and desklib (frozen cutoffs in each log), fakespot (the judge in the log), and the extra judge.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hd.eval.loop_report import _flagged, outcome_step, wilson_ci
from hd.eval.metrics import threshold_at_fpr

LOOP_DETECTORS = ("ours", "desklib", "fakespot")


def judge_cutoff(judge, validation_human_texts: list[str], fpr: float = 0.05) -> float:
    """The extra judge's cutoff: the 5%-false-positive point on VALIDATION human texts, as for every other detector."""
    return threshold_at_fpr(judge.predict_proba(validation_human_texts), fpr)


def judge_joint(records: list[dict[str, Any]], judge, cutoff: float, policy: str = "kept") -> dict[str, Any]:
    """Raw vs delivered flag counts for the judge, how many it caught that slipped past after humanizing, and the all-detector rate.

    A text counts as past every detector only if a rewrite was kept (guardrails passed) and none of ours, desklib, fakespot or the judge flags it.
    """
    raw = np.asarray(judge.predict_proba([r["steps"][0]["text"] for r in records])) > cutoff
    delivered_steps = [outcome_step(r, policy) for r in records]
    out = np.asarray(judge.predict_proba([s["text"] for s in delivered_steps])) > cutoff
    loop_clear = [r["final"] is not None and all(_flagged(r, s, d) is False for d in LOOP_DETECTORS) for r, s in zip(records, delivered_steps)]
    wins = sum(bool(c and not o) for c, o in zip(loop_clear, out))
    lo, hi = wilson_ci(wins, len(records))
    return {"n": len(records), "judge_flagged_raw": int(raw.sum()), "judge_flagged_delivered": int(out.sum()),
            "judge_caught_then_evaded": int((raw & ~out).sum()), "loop_detectors_all_clear": int(sum(loop_clear)),
            "all_detectors_clear": wins, "rate": wins / len(records) if records else float("nan"), "lo": lo, "hi": hi}  # fmt: skip
