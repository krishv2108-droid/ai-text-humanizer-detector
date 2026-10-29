"""Quality guardrails for rewrites: did the meaning survive, and is it still good, readable prose?

A rewrite that fools a detector by changing the story, dropping half of it, or sprinkling typos is
not a success, so every rewrite is checked on:
  * meaning: sentence-embedding similarity to the original (overall, per-sentence coverage of the
    original, and how much of the rewrite has no counterpart in the original = invented content);
  * length: word-count ratio;
  * readability: Flesch-Kincaid grade and Flesch reading ease, before vs after;
  * no typos / gibberish: rare "novel" words (not in the original, split into 3+ GPT-2 word pieces) and
    an optional ceiling on GPT-2's average surprise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from hd.features.text import analyze, sentences
from hd.humanizer.facts import check_facts

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """L2-normalized embeddings, one row per text."""
        ...


class SentenceEmbedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device or "cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64)


@dataclass(frozen=True)
class Thresholds:
    """Provisional defaults; `hd humanizer calibrate-guardrails` reports how they behave on real data."""

    min_doc_similarity: float = 0.80  # whole-text embedding similarity to the original
    dropped_sim: float = 0.50  # an original sentence below this similarity to every rewrite sentence is "dropped"
    max_dropped_frac: float = 0.20  # share of original sentences allowed to be missing from the rewrite
    max_invented_frac: float = 0.35  # share of rewrite sentences with no counterpart in the original
    invented_sim: float = 0.35  # a rewrite sentence below this similarity to every original sentence is "invented"
    length_ratio: tuple[float, float] = (0.7, 1.4)
    max_fk_delta: float = 3.0  # allowed change in Flesch-Kincaid grade level (only when fk_range is None)
    # Allowed Flesch-Kincaid grade for the rewrite: the span of real human text in this genre. Preferred over the
    # delta rule because humanizing LEGITIMATELY lowers reading level (AI stories read ~8 grades, human ~5).
    fk_range: tuple[float, float] | None = None
    max_novel_rare_rate: float = 0.01  # share of words that are new AND rare (typo/gibberish signal)
    max_missing_name_frac: float = 0.25  # share of the original's names that may vanish (any lost number is flagged)
    max_mean_surprisal: float | None = None  # GPT-2 ceiling for "incoherent"; set from human text

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Thresholds":
        g = cfg.get("guardrails", {})
        fields = {k: g[k] for k in cls.__dataclass_fields__ if k in g}
        if "length_ratio" in fields:
            fields["length_ratio"] = tuple(fields["length_ratio"])
        return cls(**fields)


@dataclass
class GuardrailReport:
    similarity: float
    dropped_frac: float
    invented_frac: float
    length_ratio: float
    fk_grade_before: float
    fk_grade_after: float
    reading_ease_before: float
    reading_ease_after: float
    novel_rare_rate: float
    mean_surprisal: float | None = None
    flags: list[str] = field(default_factory=list)
    # details that make a repair request specific ("you dropped THIS sentence"), not just "meaning drift"
    missing_sentences: list[str] = field(default_factory=list)  # original sentences with no counterpart in the rewrite
    invented_sentences: list[str] = field(default_factory=list)  # rewrite sentences with no counterpart in the original
    novel_words: list[str] = field(default_factory=list)  # new, rare words (typo/gibberish candidates)
    missing_numbers: list[int] = field(default_factory=list)
    missing_names: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.flags

    @property
    def fk_delta(self) -> float:
        return self.fk_grade_after - self.fk_grade_before


def split_sentences(text: str) -> list[str]:
    return [s.text.strip() for s in sentences(analyze(text)) if s.text.strip()]


def readability(text: str) -> dict[str, float]:
    import textstat

    return {
        "fk_grade": float(textstat.flesch_kincaid_grade(text)),
        "reading_ease": float(textstat.flesch_reading_ease(text)),
    }


_WORD = re.compile(r"[A-Za-z][A-Za-z']*")


class RareWordCounter:
    """Counts words that are NEW (absent from the original) and RARE (3+ GPT-2 word pieces).

    A typo like "teh" or "recieve" and invented gibberish both fragment into several pieces, while
    ordinary words are one or two. Names newly invented by a rewrite also count, which is fine: a
    rewrite should not need many of those.
    """

    def __init__(self, tokenizer=None):
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer = tokenizer

    def rare(self, original: str, rewrite: str) -> set[str]:
        known = {w.lower() for w in _WORD.findall(original)}
        novel = {w.lower() for w in _WORD.findall(rewrite) if w.lower() not in known}
        return {w for w in novel if len(self.tokenizer(" " + w)["input_ids"]) >= 3}

    def rate(self, original: str, rewrite: str) -> float:
        words = _WORD.findall(rewrite)
        if not words:
            return 0.0
        rare = self.rare(original, rewrite)
        return sum(1 for w in words if w.lower() in rare) / len(words)


class GuardrailChecker:
    def __init__(self, embedder: Embedder, thresholds: Thresholds | None = None, rare_words: RareWordCounter | None = None, scorer=None):
        self.embedder = embedder
        self.t = thresholds or Thresholds()
        self.rare_words = rare_words or RareWordCounter()
        self.scorer = scorer  # optional GPT2Scorer for the incoherence ceiling

    def semantic(self, original: str, rewrite: str) -> dict[str, Any]:
        so, sr = split_sentences(original), split_sentences(rewrite)
        if not so or not sr:
            return {"similarity": 0.0, "dropped_frac": 1.0, "invented_frac": 1.0, "missing": so, "invented": sr}
        eo, er = self.embedder.encode(so), self.embedder.encode(sr)
        mo, mr = eo.mean(axis=0), er.mean(axis=0)
        sim = eo @ er.T
        # Sentence-to-sentence matching. Measured (results/guardrail_restructuring.csv): splitting a sentence into
        # fragments or merging two is NOT flagged, while deleting or swapping sentences is (AUC 0.94-0.96). Matching
        # runs of 2-3 sentences instead was tried and is useless (AUC ~0.5: every run matches on a single topic).
        dropped = sim.max(axis=1) < self.t.dropped_sim  # original sentences matching nothing
        invented = sim.max(axis=0) < self.t.invented_sim  # rewrite sentences matching nothing
        return {
            "similarity": float(mo @ mr / (np.linalg.norm(mo) * np.linalg.norm(mr))),
            "dropped_frac": float(dropped.mean()), "invented_frac": float(invented.mean()),
            "missing": [x for x, d in zip(so, dropped) if d], "invented": [x for x, d in zip(sr, invented) if d],
        }  # fmt: skip

    def check(self, original: str, rewrite: str) -> GuardrailReport:
        sem = self.semantic(original, rewrite)
        before, after = readability(original), readability(rewrite)
        n_orig, n_new = len(original.split()), len(rewrite.split())
        ratio = n_new / max(n_orig, 1)
        rare = self.rare_words.rate(original, rewrite)
        facts = check_facts(original, rewrite)
        surprisal = None
        if self.scorer is not None:
            from hd.features.perplexity import perplexity_features

            surprisal = perplexity_features(self.scorer.token_stats([rewrite])[0])["mean_surprisal"]
        r = GuardrailReport(
            similarity=sem["similarity"], dropped_frac=sem["dropped_frac"], invented_frac=sem["invented_frac"], length_ratio=ratio,
            fk_grade_before=before["fk_grade"], fk_grade_after=after["fk_grade"],
            reading_ease_before=before["reading_ease"], reading_ease_after=after["reading_ease"],
            novel_rare_rate=rare, mean_surprisal=surprisal,
            missing_sentences=sem["missing"], invented_sentences=sem["invented"],
            novel_words=sorted(self.rare_words.rare(original, rewrite)) if rare else [],
            missing_numbers=facts.missing_numbers, missing_names=facts.missing_names,
        )  # fmt: skip
        t = self.t
        if r.similarity < t.min_doc_similarity:
            r.flags.append(f"meaning drift (similarity {r.similarity:.2f} < {t.min_doc_similarity})")
        if r.dropped_frac > t.max_dropped_frac:
            r.flags.append(f"dropped content ({r.dropped_frac:.0%} of original sentences have no counterpart)")
        if r.invented_frac > t.max_invented_frac:
            r.flags.append(f"invented content ({r.invented_frac:.0%} of sentences match nothing in the original)")
        if not t.length_ratio[0] <= ratio <= t.length_ratio[1]:
            r.flags.append(f"length changed too much ({ratio:.2f}x)")
        if t.fk_range is not None:
            if not t.fk_range[0] <= r.fk_grade_after <= t.fk_range[1]:
                r.flags.append(f"reading level {r.fk_grade_after:.1f} outside the human range {t.fk_range[0]:.1f} to {t.fk_range[1]:.1f}")
        elif abs(r.fk_delta) > t.max_fk_delta:
            r.flags.append(f"readability shifted ({r.fk_delta:+.1f} grade levels)")
        if facts.missing_numbers:
            r.flags.append(f"numbers changed or lost ({', '.join(map(str, facts.missing_numbers))})")
        if facts.missing_name_frac > t.max_missing_name_frac:
            r.flags.append(f"names missing ({', '.join(facts.missing_names)})")
        if rare > t.max_novel_rare_rate:
            r.flags.append(f"possible typos or gibberish ({rare:.1%} new rare words)")
        if t.max_mean_surprisal is not None and surprisal is not None and surprisal > t.max_mean_surprisal:
            r.flags.append(f"incoherent-looking text (GPT-2 surprisal {surprisal:.2f} > {t.max_mean_surprisal:.2f})")
        return r
