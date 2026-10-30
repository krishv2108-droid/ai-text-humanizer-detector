"""One humanizing pass end to end: detect -> rewrite -> re-detect -> guardrails -> comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from hd.config import resolve_path
from hd.detector.explain import Analysis, Detector
from hd.humanizer.feedback import HumanProfile
from hd.humanizer.guardrails import GuardrailChecker, GuardrailReport, Thresholds
from hd.humanizer.rewrite import Humanizer, RewriteResult

# Features shown in the before/after table: the ones the detector's drivers and the prompt talk about.
COMPARE = ["sent_len_cv", "short_sent_frac", "sent_initial_transition_frac", "transition_per100w", "hedge_per100w",
           "stock_phrase_per100w", "contraction_per100w", "first_person_per100w", "emdash_per100w", "mattr",
           "mean_surprisal", "top10_frac"]  # fmt: skip


@dataclass
class HumanizeReport:
    before: Analysis
    rewrite: RewriteResult
    after: Analysis
    guardrails: GuardrailReport
    comparison: pd.DataFrame
    external_before: float | None = None
    external_after: float | None = None


def load_thresholds(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Frozen 5%-false-positive cutoffs per detector and genre (from `hd detector eval-external`)."""
    path = resolve_path(cfg, "results") / "detector_thresholds.json"
    return json.loads(path.read_text()) if path.exists() else {}


def compare_features(before: Analysis, after: Analysis, profile: HumanProfile, genre: str) -> pd.DataFrame:
    rows = []
    for f in COMPARE:
        b, a, h = before.features[f], after.features[f], profile.q(genre, f, "p50")
        rows.append({"feature": f, "before": b, "after": a, "human_median": h,
                     "moved_toward_human": abs(a - h) < abs(b - h)})  # fmt: skip
    return pd.DataFrame(rows)


def human_fk_range(cfg: dict[str, Any], genre: str) -> tuple[float, float] | None:
    """Reading-grade span of real human text in this genre, from TRAIN-split human texts only."""
    path = resolve_path(cfg, "processed") / "corpus.parquet"
    if not path.exists():
        return None
    from hd.humanizer.guardrails import readability

    corpus = pd.read_parquet(path)
    humans = corpus[(corpus["genre"] == genre) & (corpus["label"] == 0) & (corpus["split"] == "train")]
    if humans.empty:
        return None
    lo_q, hi_q = cfg.get("guardrails", {}).get("fk_percentiles", [1, 99])
    grades = [readability(t)["fk_grade"] for t in humans["text"]]
    return float(np.percentile(grades, lo_q)), float(np.percentile(grades, hi_q))


def checker_for(cfg: dict[str, Any], embedder, scorer, profile: HumanProfile, genre: str, strength: str = "faithful") -> GuardrailChecker:
    t = Thresholds.from_config(cfg, strength)
    ceiling = profile.q(genre, "mean_surprisal", "p99")  # above nearly all real human text = looks incoherent
    return GuardrailChecker(embedder, replace(t, max_mean_surprisal=ceiling, fk_range=human_fk_range(cfg, genre)), scorer=scorer)


def humanize_once(
    detector: Detector,
    humanizer: Humanizer,
    checker: GuardrailChecker,
    text: str,
    genre: str,
    strategy: str = "full",
    external=None,
) -> HumanizeReport:
    before = detector.analyze(text)
    rewrite = humanizer.humanize(before.text, genre, strategy, analysis=before)
    after = detector.analyze(rewrite.text)
    guard = checker.check(before.text, rewrite.text)
    ext_b = ext_a = None
    if external is not None:
        ext_b, ext_a = (float(x) for x in external.predict_proba([before.text, rewrite.text]))
    return HumanizeReport(before, rewrite, after, guard, compare_features(before, after, humanizer.profile, genre), ext_b, ext_a)
