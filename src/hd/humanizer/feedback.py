"""Turn the detector's findings into concrete, evidence-backed rewriting instructions.

The detector says *how much* a text looks AI-written and *why* (feature groups, flagged sentences).
`diagnose` compares the text's features with what real human texts of the same genre look like (the
"human profile": percentiles over human TRAIN-split texts) and, wherever the text sits outside the
typical human range, produces an instruction with the numbers attached. Weaknesses are ordered by how
much the detector's score depended on that feature group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from hd.config import resolve_path
from hd.features.stylometric import find_stock_phrases, find_transitions

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
FALLBACK_GENRE = "all"


@dataclass
class Weakness:
    group: str  # feature group, matches the detector's group names
    evidence: str  # what the numbers show
    fix: str  # what to do about it
    severity: float = 0.0  # detector group effect (P(AI) points), used for ordering


class HumanProfile:
    """Percentiles of each feature over human texts, per genre (plus pooled as 'all')."""

    def __init__(self, table: pd.DataFrame):
        self.table = table.set_index(["genre", "feature"])

    @classmethod
    def build(cls, cfg: dict[str, Any], scorer=None) -> "HumanProfile":
        """Human TRAIN texts of every genre, including the stories that have no AI pair yet."""
        from hd.data.splits import load_split
        from hd.features.pipeline import compute_features
        from hd.features.stylometric import feature_names

        processed = resolve_path(cfg, "processed")
        corpus = pd.read_parquet(processed / "corpus.parquet")
        human = load_split(corpus, "train")  # the test split stays locked
        human = human[human["label"] == 0]
        feats = compute_features(human, processed / "features.parquet", scorer)
        df = human[["id", "genre"]].merge(feats, on="id")
        names = feature_names()
        rows = []
        for genre, g in [(FALLBACK_GENRE, df)] + list(df.groupby("genre")):
            for f in names:
                if f in g and g[f].notna().any():
                    rows.append({"genre": genre, "feature": f, "n": int(g[f].notna().sum()),
                                 **{f"p{int(q * 100):02d}": float(g[f].quantile(q)) for q in QUANTILES}})  # fmt: skip
        return cls(pd.DataFrame(rows))

    @classmethod
    def load_or_build(cls, cfg: dict[str, Any], scorer=None) -> "HumanProfile":
        path = resolve_path(cfg, "results") / "human_profile.csv"
        if path.exists():
            return cls(pd.read_csv(path))
        profile = cls.build(cfg, scorer)
        profile.table.reset_index().round(5).to_csv(path, index=False)
        return profile

    def q(self, genre: str, feature: str, level: str) -> float:
        key = (genre if (genre, feature) in self.table.index else FALLBACK_GENRE, feature)
        return float(self.table.loc[key, level])

    def genres(self) -> list[str]:
        return sorted({g for g, _ in self.table.index})


def _gap(value: float, profile: HumanProfile, genre: str, feature: str, side: str) -> tuple[bool, float]:
    """Is `value` outside the typical human range on `side` ('low' = below p25, 'high' = above p75)?"""
    if value is None or np.isnan(value):
        return False, np.nan
    median = profile.q(genre, feature, "p50")
    if side == "low":
        return value < profile.q(genre, feature, "p25"), median
    return value > profile.q(genre, feature, "p75"), median


def _beyond(value: float, profile: HumanProfile, genre: str, feature: str, level: str = "p90") -> tuple[bool, float, float]:
    """Is `value` above the human `level` (overshoot)? Returns (flag, human median, the level's value)."""
    if value is None or np.isnan(value):
        return False, np.nan, np.nan
    cap = profile.q(genre, feature, level)
    return value > cap, profile.q(genre, feature, "p50"), cap


OVERSHOOT_SEVERITY = 2.0  # corrections of an over-correction outrank everything else (detector effects are <= 1)


def diagnose(analysis, profile: HumanProfile, genre: str) -> list[Weakness]:
    """Weaknesses of a text, most important first. `analysis` is an `explain.Analysis`.

    Two-sided: besides "too machine-like" it reports "overshot": a rewrite pushed past the range of real human text
    (too choppy, too erratic, odd word choices) is told to come back, so the loop converges on human text instead of
    pushing one way forever.
    """
    f, out = analysis.features, []

    def add(group: str, evidence: str, fix: str, severity: float | None = None) -> None:
        out.append(Weakness(group, evidence, fix, float(analysis.group_effects.get(group, 0.0)) if severity is None else severity))

    over, med, cap = _beyond(f["short_sent_frac"], profile, genre, "short_sent_frac")
    if over:
        add("rhythm", f"Too choppy: {f['short_sent_frac']:.0%} of sentences are 5 words or fewer (human {genre} text typically {med:.0%}, rarely above {cap:.0%}).",
            "Merge some fragments into full sentences. Keep a few short ones for emphasis, but most sentences should carry a subject and a verb.",
            OVERSHOOT_SEVERITY)  # fmt: skip
    over, med, cap = _beyond(f["sent_len_cv"], profile, genre, "sent_len_cv")
    if over:
        add("rhythm", f"Sentence lengths swing too wildly (variation {f['sent_len_cv']:.2f}; human typically {med:.2f}, rarely above {cap:.2f}).",
            "Even it out a little: fewer extreme one-word lines next to very long sentences.", OVERSHOOT_SEVERITY)  # fmt: skip
    over, med, cap = _beyond(f["mean_surprisal"], profile, genre, "mean_surprisal")
    if over:
        add("perplexity", f"Some word choices are odd or showy for this kind of text (a language model finds the wording more surprising than {cap:.2f} bits; human typically {med:.2f}).",
            "Swap the strangest word choices for plainer, exact ones. Unusual is good; obscure or forced is not.", OVERSHOOT_SEVERITY)  # fmt: skip

    low, med = _gap(f["sent_len_cv"], profile, genre, "sent_len_cv", "low")
    if low:
        add("rhythm", f"Sentence lengths are too uniform (variation {f['sent_len_cv']:.2f}; human {genre} texts typically {med:.2f}).",
            "Break the rhythm: put a few very short sentences or fragments (2-6 words) next to long, winding ones. Do not keep a steady medium length.")  # fmt: skip
    else:
        low, med = _gap(f["short_sent_frac"], profile, genre, "short_sent_frac", "low")
        if low:
            add("rhythm", f"Almost no short sentences ({f['short_sent_frac']:.0%} are 5 words or fewer; human typically {med:.0%}).",
                "Add some very short sentences and fragments.")  # fmt: skip

    openers = [s for s in analysis.sentences if "stock transition opener" in s.reasons]
    high, med = _gap(f["sent_initial_transition_frac"], profile, genre, "sent_initial_transition_frac", "high")
    if high or _gap(f["transition_per100w"], profile, genre, "transition_per100w", "high")[0]:
        found = ", ".join(f"'{w}'" for w in find_transitions(analysis.text)[:5])
        add("discourse", f"Stock connectives are overused ({len(openers)} of {len(analysis.sentences)} sentences open with one; found: {found}).",
            "Delete these connectives or replace them with plain flow; let paragraphs follow on without signposting.")  # fmt: skip

    high, med = _gap(f["hedge_per100w"], profile, genre, "hedge_per100w", "high")
    if high:
        add("hedging", f"Heavy hedging ({f['hedge_per100w']:.1f} per 100 words; human typically {med:.1f}).",
            "Commit. Remove 'may', 'might', 'generally', 'often' and similar softeners and state things directly.")  # fmt: skip

    stock = find_stock_phrases(analysis.text)
    if stock and _gap(f["stock_phrase_per100w"], profile, genre, "stock_phrase_per100w", "high")[0]:
        add("stock", f"Cliche phrases found: {', '.join(repr(p) for p in stock[:6])}.",
            "Replace each with a specific, concrete image or a plain statement of your own.")  # fmt: skip

    low, med = _gap(f["contraction_per100w"], profile, genre, "contraction_per100w", "low")
    if low:
        add("contraction", f"Few contractions ({f['contraction_per100w']:.1f} per 100 words; human typically {med:.1f}).",
            "Use contractions (don't, it's, they're) wherever speech or narration would naturally use them, unless the register is deliberately formal.")  # fmt: skip

    if genre == "casual":
        low, med = _gap(f["first_person_per100w"], profile, genre, "first_person_per100w", "low")
        if low:
            add("voice", f"Impersonal voice (first-person words {f['first_person_per100w']:.1f} per 100; human typically {med:.1f}).",
                "Speak from a person: use 'I', 'my', 'we' and a point of view.")  # fmt: skip

    if _gap(f["emdash_per100w"], profile, genre, "emdash_per100w", "high")[0] and f["emdash_per100w"] > 0:
        add("punctuation", f"Heavy em-dash use ({f['emdash_per100w']:.1f} per 100 words).", "Cut most em-dashes; use commas, periods or parentheses.")

    low, med = _gap(f["mattr"], profile, genre, "mattr", "low")
    if low:
        add("lexical", f"Repetitive vocabulary (diversity {f['mattr']:.2f}; human typically {med:.2f}).", "Vary word choice; avoid repeating the same words and sentence frames.")  # fmt: skip

    unpredictable = _gap(f["top10_frac"], profile, genre, "top10_frac", "high")
    if unpredictable[0] or _gap(f["mean_surprisal"], profile, genre, "mean_surprisal", "low")[0]:
        add("perplexity", f"Word choice is very predictable (a language model ranks {f['top10_frac']:.0%} of the words among its top-10 guesses; human {genre} text typically {profile.q(genre, 'top10_frac', 'p50'):.0%}).",
            "Choose less expected words: specific nouns, unusual but exact verbs, concrete details a generic writer would not reach for. Avoid the first phrase that comes to mind.")  # fmt: skip

    return sorted(out, key=lambda w: -w.severity)


def render_feedback(weaknesses: list[Weakness], analysis, max_items: int = 6, top_sentences: int = 3) -> str:
    """The detector-derived part of the prompt."""
    lines = [f"An automated detector scored this draft {analysis.prob_ai:.0%} likely to be AI-written. What it found:"]
    for i, w in enumerate(weaknesses[:max_items], 1):
        lines.append(f"{i}. {w.evidence} {w.fix}")
    flagged = [s for s in analysis.top_sentences(top_sentences) if s.reasons]
    if flagged:
        lines.append("\nThe sentences that read most machine-like (rework these especially):")
        for s in flagged:
            lines.append(f'- "{s.text}"  [{"; ".join(s.reasons)}]')
    if not weaknesses and not flagged:
        lines.append("It found no specific weaknesses to fix; keep the voice natural.")
    return "\n".join(lines)
