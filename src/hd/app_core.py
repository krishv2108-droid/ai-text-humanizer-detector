"""The logic behind the Streamlit app, kept out of the UI script so it can be tested without a browser.

Nothing here touches the held-out test split. The humanizing path is the same `run_loop` the CLI uses, so the
app and the experiments cannot drift apart.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from hd.detector.explain import Analysis
from hd.eval.style import AI_C
from hd.humanizer.pipeline import checker_for, load_thresholds
from hd.humanizer.prompts import STRATEGIES
from hd.humanizer.rewrite import Humanizer
from hd.loop import LoopResult, cutoff_for, run_loop

GENRES = ["story", "essay", "casual"]
KNOWN_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]

# What each feature group measures, in plain language (the detector's "reasons").
GROUP_LABELS = {
    "rhythm": "Sentence rhythm (how much sentence lengths vary)",
    "lexical": "Vocabulary (variety and word length)",
    "discourse": "Connectives (However, Moreover, ...)",
    "hedging": "Hedging (may, might, generally)",
    "stock": "Stock phrases (\"it is important to note\")",
    "contraction": "Contractions (don't, it's)",
    "voice": "Person (I / you)",
    "punctuation": "Punctuation (commas, dashes, quotes)",
    "pos": "Grammar mix (noun, verb, pronoun shares)",
    "perplexity": "Predictability (how expected each word is to GPT-2)",
}

# Rough tokens per input word for one rewrite call (input + output), measured in the 10-story pilot
# (results/ablation_prompts_*.csv). Free tiers meter tokens, so this tells users what a run will use up.
TOKENS_PER_WORD = {"naive": 3.3, "craft": 4.2, "feedback": 4.4, "full": 5.1}
MIN_WORDS = 5
LONG_TEXT_WORDS = 700  # beyond this GPT-2's window (1,024 tokens) no longer covers the whole text


@dataclass
class Resources:
    cfg: dict[str, Any]
    detector: Any
    profile: Any
    embedder: Any
    thresholds: dict[str, dict[str, float]]
    external: Any | None = None
    judge: Any | None = None  # the third detector (fakespot): scored and shown, never used to decide anything


def load_resources(cfg: dict[str, Any], with_external: bool = False, with_judge: bool = False) -> Resources:
    """Load models once (the app caches this). Raises RuntimeError with a clear message if training hasn't been run."""
    from hd.detector.explain import Detector
    from hd.humanizer.feedback import HumanProfile
    from hd.humanizer.guardrails import SentenceEmbedder

    thresholds = load_thresholds(cfg)
    if not thresholds:
        raise RuntimeError("no frozen cutoffs found: run `hd detector eval-external` first (see the README)")
    detector = Detector.load(cfg)
    profile = HumanProfile.load_or_build(cfg, detector.scorer)
    external = None
    if with_external:
        from hd.detector.external import DesklibDetector

        external = DesklibDetector()
    judge = None
    if with_judge:
        from hd.detector.external import FakespotDetector

        judge = FakespotDetector()
    return Resources(cfg, detector, profile, SentenceEmbedder(), thresholds, external, judge)


def word_count(text: str) -> int:
    return len(text.split())


def input_problem(text: str) -> str | None:
    """A blocking problem with the input, or None."""
    n = word_count(text)
    if n < MIN_WORDS:
        return f"Please enter at least {MIN_WORDS} words (you entered {n})."
    return None


def input_warning(text: str) -> str | None:
    if word_count(text) > LONG_TEXT_WORDS:
        return (f"This text is long ({word_count(text)} words). The detector reads the first ~1,000 words' worth of tokens, "
                "so the end of it is not scored.")
    return None


def estimated_tokens(text: str, strategy: str, rewrites: int) -> int:
    """Rough tokens an API run will consume, for the free-tier warning (an upper bound: the loop often stops early)."""
    return int(word_count(text) * TOKENS_PER_WORD[strategy] * rewrites)


def verdict(prob: float, cutoff: float | None) -> tuple[str, str]:
    """('flagged' | 'not flagged' | 'n/a', one-line explanation) against the frozen 5%-false-positive cutoff."""
    if cutoff is None or prob is None or (isinstance(prob, float) and math.isnan(prob)):
        return "n/a", "no cutoff available"
    flagged = prob > cutoff
    return ("flagged" if flagged else "not flagged"), (
        f"P(AI) {prob:.2f} is {'above' if flagged else 'at or below'} the cutoff {cutoff:.2f}, "
        "which flags 5% of real human text in this genre")  # fmt: skip


def cutoffs_for(res: Resources, genre: str) -> dict[str, float | None]:
    """Frozen cutoffs for the genre. A missing one is an error (never a silent 0.5); desklib's is None only if it was never calibrated."""
    out: dict[str, float | None] = {"ours": cutoff_for(res.thresholds, "ours", genre)}
    out["external"] = cutoff_for(res.thresholds, "external", genre) if "external" in res.thresholds else None
    out["judge"] = cutoff_for(res.thresholds, "third", genre) if "third" in res.thresholds else None
    return out


def sentence_html(analysis: Analysis) -> str:
    """The analyzed text with each sentence tinted by its P(AI), reasons in the hover title. All text is HTML-escaped."""
    text, out, pos = analysis.text, [], 0
    for s in sorted(analysis.sentences, key=lambda s: s.start):
        if s.start > pos:
            out.append(_escape(text[pos:s.start]))
        chunk = _escape(text[s.start:s.end])
        if math.isnan(s.prob_ai):
            out.append(f'<span title="not scored (beyond the detector\'s window)">{chunk}</span>')
        else:
            alpha = 0.6 * min(max(s.prob_ai, 0.0), 1.0)
            r, g, b = (int(AI_C[i:i + 2], 16) for i in (1, 3, 5))
            reasons = "; ".join(s.reasons) or "no particular cue"
            title = html.escape(f"P(AI) {s.prob_ai:.2f}: {reasons}", quote=True)
            out.append(f'<span style="background:rgba({r},{g},{b},{alpha:.2f});border-radius:3px;padding:1px 2px" title="{title}">{chunk}</span>')
        pos = max(pos, s.end)
    if pos < len(text):
        out.append(_escape(text[pos:]))
    return "".join(out)


def _escape(chunk: str) -> str:
    return html.escape(chunk).replace("\n", "<br>")


def group_effects_frame(analysis: Analysis) -> pd.DataFrame:
    """Per feature group: how many points of P(AI) disappear if the group looked typical. Positive = pushes toward AI."""
    rows = [{"group": GROUP_LABELS.get(g, g), "effect": e} for g, e in analysis.group_effects.items()]
    return pd.DataFrame(rows).sort_values("effect", ascending=False).reset_index(drop=True)


def group_chart_frame(analysis: Analysis) -> pd.DataFrame:
    """Short group names (the long explanations live in GROUP_LABELS) x two columns, so a bar can be coloured by direction."""
    effects = group_effects_frame(analysis)
    short = effects["group"].str.split(" (", regex=False).str[0]
    return pd.DataFrame({"pushes toward AI": effects["effect"].clip(lower=0).to_numpy(),
                         "pushes toward human": effects["effect"].clip(upper=0).to_numpy()}, index=short.to_numpy())  # fmt: skip


def top_drivers(analysis: Analysis, k: int = 6) -> pd.DataFrame:
    rows = [d for d in analysis.feature_effects if d["effect"] > 0][:k]
    return pd.DataFrame(rows, columns=["feature", "value", "typical", "effect"])


def humanize_text(
    res: Resources,
    client: Any,
    text: str,
    genre: str,
    *,
    strategy: str,
    model: str,
    max_rewrites: int,
    use_external: bool,
    effort: str | None = None,
    steer_desklib: bool = False,
    use_judge: bool = False,
    strength: str = "faithful",
) -> LoopResult:
    """Run the adversarial loop on one text, exactly as `hd loop run` does for the experiments.

    `strength` picks the operating point: faithful keeps every sentence's content, strong may compress minor detail (numbers and
    names are protected either way). `steer_desklib` lets desklib steer as well as our detector (stronger, but then desklib is no longer an independent check);
    `use_judge` also scores every attempt with the third detector, which never decides anything.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    cuts = cutoffs_for(res, genre)
    external = res.external if use_external else None
    if (use_external or steer_desklib) and external is None:
        raise RuntimeError("the independent detector is not loaded; enable it in the sidebar first")
    if use_judge and (res.judge is None or cuts["judge"] is None):
        raise RuntimeError("the third detector is not loaded (or has no cutoff); enable it in the sidebar first")
    humanizer = Humanizer(client, model, res.profile, effort, strength)
    checker = checker_for(res.cfg, res.embedder, res.detector.scorer, res.profile, genre, strength)
    return run_loop(res.detector, humanizer, checker, text, genre, strategy=strategy, our_cutoff=cuts["ours"],
                    ext_cutoff=cuts["external"] if external is not None else None, external=external,
                    max_iterations=max_rewrites, judge=res.judge if use_judge else None,
                    judge_cutoff=cuts["judge"] if use_judge else None, steer_external=steer_desklib)  # fmt: skip


def steps_frame(result: LoopResult) -> pd.DataFrame:
    """One row per step for display: scores, whether the guardrails passed, and why not."""
    rows = []
    for s in result.steps:
        rows.append({
            "rewrite": s.iteration, "our P(AI)": round(s.ours, 3),
            "desklib P(AI)": None if s.external is None else round(s.external, 3),
            "fakespot P(AI)": None if s.judge is None else round(s.judge, 3),
            "kept": "kept" if result.final == s.iteration else ("original" if s.iteration == 0 else ""),
            "guardrails": "" if s.guardrails_ok is None else ("ok" if s.guardrails_ok else "failed"),
            "why not": "; ".join(s.flags), "meaning similarity": None if s.similarity is None else round(s.similarity, 2),
        })  # fmt: skip
    return pd.DataFrame(rows)


def outcome_text(result: LoopResult) -> str:
    """What the user ends up with, in words (a failed rewrite is never presented as a success)."""
    if result.final is None:
        return "No rewrite passed the guardrails, so the original text is unchanged."
    kept = result.steps[result.final]
    verdict_ours = "not flagged" if kept.ours <= result.our_cutoff else "still flagged"
    return f"Kept rewrite {result.final} of {len(result.steps) - 1}: our detector says {verdict_ours} (P(AI) {kept.ours:.2f})."
