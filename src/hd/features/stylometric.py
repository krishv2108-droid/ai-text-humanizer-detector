"""Interpretable stylometric features (no neural model needed).

Every feature is a *rate or ratio*, never a raw count, so it does not simply track how long a text is
(length is a known confound in our data; see data/DATA_SOURCES.md). Rates are per 100 words.

Features are grouped so later ablations can drop a whole group and measure what it contributed.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from spacy.tokens import Doc

from hd.features.lexicons import (
    EXPANDABLE_PATTERN,
    FIRST_PERSON,
    HEDGES,
    SECOND_PERSON,
    STOCK_PHRASES,
    TRANSITIONS,
)
from hd.features.text import is_word, sentences, words

UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON",
        "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]  # fmt: skip

FEATURE_GROUPS: dict[str, list[str]] = {
    # How uniform or varied the sentence lengths are. "Burstiness" = human writing swings between
    # short and long sentences; LLM prose tends to be more even.
    "rhythm": ["sent_len_mean", "sent_len_std", "sent_len_cv", "burstiness", "sent_len_adjdiff",
               "short_sent_frac", "long_sent_frac"],  # fmt: skip
    "lexical": ["mattr", "mean_word_len", "long_word_frac", "stopword_frac"],
    "discourse": ["transition_per100w", "sent_initial_transition_frac"],
    "hedging": ["hedge_per100w"],
    "stock": ["stock_phrase_per100w"],
    "contraction": ["contraction_per100w", "contraction_ratio"],
    "voice": ["first_person_per100w", "second_person_per100w"],
    "punctuation": ["comma_per100w", "semicolon_per100w", "colon_per100w", "emdash_per100w",
                    "exclaim_per100w", "question_per100w", "ellipsis_per100w", "dquote_per100w",
                    "paren_per100w", "comma_per_sentence"],  # fmt: skip
    "pos": [f"pos_{t}" for t in UPOS],
    # Layout, not language: partly an artifact of where each side was collected. Ablate it.
    "format": ["paragraph_breaks_per100w"],
    # Filled in by hd.features.perplexity (needs GPT-2).
    "perplexity": ["mean_surprisal", "surprisal_std", "top1_frac", "top10_frac"],
}
STYLOMETRIC_GROUPS = [g for g in FEATURE_GROUPS if g != "perplexity"]


def feature_names(groups: list[str] | None = None) -> list[str]:
    return [f for g in (groups or list(FEATURE_GROUPS)) for f in FEATURE_GROUPS[g]]


def _phrase_regex(phrases: list[str]) -> re.Pattern:
    alternatives = "|".join(re.escape(p) for p in sorted(set(phrases), key=len, reverse=True))
    return re.compile(rf"(?<![\w'])(?:{alternatives})(?![\w'])", re.I)


_TRANSITION_RE = _phrase_regex(TRANSITIONS)
_TRANSITION_START_RE = re.compile(
    rf"^[\W_]*(?:{'|'.join(re.escape(p) for p in sorted(set(TRANSITIONS), key=len, reverse=True))})(?![\w'])",
    re.I,
)
_HEDGE_RE = _phrase_regex(HEDGES)
_STOCK_RE = _phrase_regex(STOCK_PHRASES)
_EXPANDABLE_RE = re.compile(EXPANDABLE_PATTERN, re.I)
_CONTRACTION_PIECES = {"n't", "'re", "'ve", "'ll", "'d", "'m"}
_EMDASH_RE = re.compile(r"—|\s--\s")
_ELLIPSIS_RE = re.compile(r"\.\.\.|…")


def count_hedges(text: str) -> int:
    return len(_HEDGE_RE.findall(text))


def find_stock_phrases(text: str) -> list[str]:
    """The stock phrases present in `text` (lowercased, in order, no duplicates)."""
    return list(dict.fromkeys(m.group(0).lower() for m in _STOCK_RE.finditer(text)))


def find_transitions(text: str) -> list[str]:
    """Stock transition words/phrases present in `text` (lowercased, in order, no duplicates)."""
    return list(dict.fromkeys(m.group(0).lower() for m in _TRANSITION_RE.finditer(text)))


def count_stock_phrases(text: str) -> int:
    return len(_STOCK_RE.findall(text))


def starts_with_transition(sentence: str) -> bool:
    return bool(_TRANSITION_START_RE.match(sentence.strip()))


def count_contractions(doc_or_span) -> int:
    return sum(
        1
        for t in doc_or_span
        if t.text.lower() in _CONTRACTION_PIECES or (t.text.lower() == "'s" and t.tag_ != "POS")
    )


def mattr(tokens: list[str], window: int = 25) -> float:
    """Moving-average type-token ratio: lexical diversity that doesn't shrink as texts get longer.

    Plain type-token ratio (unique words / total words) falls mechanically with length, which would
    turn "AI answers are longer" into "AI answers are less diverse". MATTR averages the ratio over a
    sliding window of fixed size, removing that length effect.
    """
    n = len(tokens)
    if n == 0:
        return math.nan
    if n <= window:
        return len(set(tokens)) / n
    counts = Counter(tokens[:window])
    total = len(counts) / window
    for i in range(window, n):
        counts[tokens[i]] += 1
        out = tokens[i - window]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        total += len(counts) / window
    return total / (n - window + 1)


def _rhythm(sent_lens: list[int]) -> dict[str, float]:
    arr = np.asarray(sent_lens, dtype=float)
    mean, std = float(arr.mean()), float(arr.std())
    return {
        "sent_len_mean": mean,
        "sent_len_std": std,
        "sent_len_cv": std / mean,  # coefficient of variation: spread relative to the average
        "burstiness": (std - mean) / (std + mean),  # Goh & Barabasi: -1 regular ... +1 bursty
        "sent_len_adjdiff": float(np.abs(np.diff(arr)).mean() / mean) if len(arr) > 1 else 0.0,
        "short_sent_frac": float((arr <= 5).mean()),
        "long_sent_frac": float((arr >= 30).mean()),
    }


def extract_features(doc: Doc, text: str) -> dict[str, float]:
    """All non-neural features for one analyzed text. Keys match `FEATURE_GROUPS` (minus perplexity)."""
    wtoks = words(doc)
    n = len(wtoks)
    if n == 0:
        return {f: math.nan for f in feature_names(STYLOMETRIC_GROUPS)}
    per100 = 100.0 / n
    lower = text.lower()

    sents = sentences(doc)
    sent_lens = [sum(1 for t in s if is_word(t)) for s in sents]
    feats: dict[str, float] = _rhythm(sent_lens)

    alpha = [t.text.lower() for t in wtoks if t.is_alpha]
    feats |= {
        "mattr": mattr(alpha),
        "mean_word_len": float(np.mean([len(t.text) for t in wtoks])),
        "long_word_frac": float(np.mean([len(t.text) >= 7 for t in wtoks])),
        "stopword_frac": float(np.mean([t.is_stop for t in wtoks])),
    }

    n_start = sum(bool(_TRANSITION_START_RE.match(s.text.strip())) for s in sents)
    feats |= {
        "transition_per100w": len(_TRANSITION_RE.findall(text)) * per100,
        "sent_initial_transition_frac": n_start / len(sents),
        "hedge_per100w": len(_HEDGE_RE.findall(text)) * per100,
        "stock_phrase_per100w": len(_STOCK_RE.findall(text)) * per100,
    }

    contracted = sum(
        1
        for t in doc
        if t.text.lower() in _CONTRACTION_PIECES or (t.text.lower() == "'s" and t.tag_ != "POS")
    )
    expandable = len(_EXPANDABLE_RE.findall(text))
    feats |= {
        "contraction_per100w": contracted * per100,
        # Of the places a writer could contract, how often did they? Undefined (NaN) if neither occurs.
        "contraction_ratio": contracted / (contracted + expandable) if contracted + expandable else math.nan,
        "first_person_per100w": sum(t.text.lower() in FIRST_PERSON for t in wtoks) * per100,
        "second_person_per100w": sum(t.text.lower() in SECOND_PERSON for t in wtoks) * per100,
    }

    feats |= {
        "comma_per100w": text.count(",") * per100,
        "semicolon_per100w": text.count(";") * per100,
        "colon_per100w": text.count(":") * per100,
        "emdash_per100w": len(_EMDASH_RE.findall(text)) * per100,
        "exclaim_per100w": text.count("!") * per100,
        "question_per100w": text.count("?") * per100,
        "ellipsis_per100w": len(_ELLIPSIS_RE.findall(text)) * per100,
        "dquote_per100w": text.count('"') * per100,
        "paren_per100w": text.count("(") * per100,
        "comma_per_sentence": text.count(",") / len(sents),
        "paragraph_breaks_per100w": lower.count("\n") * per100,
    }

    non_space = [t for t in doc if not t.is_space]
    pos_counts = Counter(t.pos_ for t in non_space)
    for tag in UPOS:
        feats[f"pos_{tag}"] = pos_counts.get(tag, 0) / len(non_space)
    return feats


def pos_sequence(doc: Doc) -> str:
    """Space-joined universal POS tags, e.g. 'PRON AUX ADJ PUNCT'. Input for POS n-gram models.

    Kept as a string here; n-gram vocabularies are fitted later on the *training split only*.
    """
    return " ".join(t.pos_ for t in doc if not t.is_space)
