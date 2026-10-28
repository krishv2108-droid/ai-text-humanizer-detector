"""The user-facing detector: P(AI) for a text, a per-sentence breakdown, and what drove the score.

* Document score: the boosted-tree baseline (`gbm_full`) on stylometric + GPT-2 features.
* Drivers: **neutralization**. We replace a feature (or a whole feature group) with its typical
  training value and see how much P(AI) drops. A positive number means "this pushes the text toward AI".
  This is model-agnostic and honest about correlated features: GPT-2's four features cover for each
  other, so the *group* effect is more meaningful than any single feature's.
* Sentences: the sentence-level model in `hd.detector.sentence` (weakly supervised, see its caveat).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hd.config import PROJECT_ROOT
from hd.data.clean import normalize_text
from hd.detector.baseline import VARIANTS, load_split_features, model_features
from hd.detector.sentence import SENTENCE_FEATURES, sentence_rows
from hd.features.perplexity import perplexity_features
from hd.features.stylometric import FEATURE_GROUPS, extract_features, pos_sequence
from hd.features.text import analyze, words

MODELS_DIR = PROJECT_ROOT / "models"
MIN_WORDS = 5

# Plain-language cues attached to a sentence (thresholds are simple rules of thumb, not fitted).
PREDICTABLE_TOP10 = 0.85
UNIFORM_LENGTH_BAND = 0.15


@dataclass
class SentenceResult:
    text: str
    start: int
    end: int
    prob_ai: float  # NaN if the sentence is past GPT-2's 512-token window
    reasons: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    text: str  # the normalized text that was scored (offsets refer to this string)
    prob_ai: float
    n_words: int
    sentences: list[SentenceResult]
    group_effects: dict[str, float]  # P(AI) points removed by neutralizing each feature group
    feature_effects: list[dict[str, Any]]  # same, per feature: name, value, typical, effect
    features: dict[str, float]

    def top_sentences(self, k: int = 3) -> list[SentenceResult]:
        scored = [s for s in self.sentences if not math.isnan(s.prob_ai)]
        return sorted(scored, key=lambda s: s.prob_ai, reverse=True)[:k]


class Detector:
    def __init__(self, doc_model, variant: str, reference: pd.Series, sentence_model, scorer, nlp=None):
        v = VARIANTS[variant]
        if v.pos_ngrams:
            raise ValueError("explanations support the non-POS-n-gram variants")
        self.doc_model, self.variant = doc_model, variant
        self.features = model_features(v.groups)
        self.groups = {g: [f for f in FEATURE_GROUPS[g] if f in self.features] for g in v.groups}
        self.reference = reference[self.features]
        self.sentence_model = sentence_model
        self.scorer = scorer

    @classmethod
    def load(cls, cfg: dict[str, Any], scorer=None, variant: str = "gbm_full") -> "Detector":
        """Load saved models (`hd detector train-baseline` and `train-sentence` create them)."""
        for name in (f"{variant}.joblib", "sentence_lr.joblib"):
            if not (MODELS_DIR / name).exists():
                raise RuntimeError(f"missing models/{name}: run `hd detector train-baseline` and `hd detector train-sentence`")
        if scorer is None:
            from hd.features.perplexity import GPT2Scorer

            scorer = GPT2Scorer()
        doc = joblib.load(MODELS_DIR / f"{variant}.joblib")["model"]
        sent = joblib.load(MODELS_DIR / "sentence_lr.joblib")["model"]
        train = load_split_features(cfg, "train")
        return cls(doc, variant, train[model_features(VARIANTS[variant].groups)].median(), sent, scorer)

    def _featurize(self, text: str):
        doc = analyze(text)
        stats = self.scorer.token_stats([text])[0]
        feats = {**extract_features(doc, text), **perplexity_features(stats)}
        return doc, stats, feats

    def _p(self, rows: pd.DataFrame) -> np.ndarray:
        return self.doc_model.predict_proba(rows[self.features])[:, 1]

    def analyze(self, text: str) -> Analysis:
        text = normalize_text(text)
        doc, stats, feats = self._featurize(text)
        n_words = len(words(doc))
        if n_words < MIN_WORDS:
            raise ValueError(f"text too short to analyze ({n_words} words; need at least {MIN_WORDS})")

        row = pd.DataFrame([feats])[self.features]
        prob = float(self._p(row)[0])

        # Neutralization: swap in typical values, one group (and one feature) at a time.
        variants = {}
        for g, cols in self.groups.items():
            r = row.copy()
            r[cols] = self.reference[cols].to_numpy()
            variants[("group", g)] = r
        for f in self.features:
            r = row.copy()
            r[f] = self.reference[f]
            variants[("feature", f)] = r
        keys = list(variants)
        probs = self._p(pd.concat([variants[k] for k in keys], ignore_index=True))
        effects = {k: prob - p for k, p in zip(keys, probs)}
        group_effects = dict(sorted(((g, effects[("group", g)]) for g in self.groups), key=lambda kv: -kv[1]))
        feature_effects = sorted(
            ({"feature": f, "value": float(row[f].iloc[0]), "typical": float(self.reference[f]), "effect": effects[("feature", f)]}
             for f in self.features),
            key=lambda d: -d["effect"],
        )  # fmt: skip

        return Analysis(text, prob, n_words, self._sentences(text, doc, stats), group_effects, feature_effects, feats)

    def _sentences(self, text: str, doc, stats) -> list[SentenceResult]:
        rows = sentence_rows(text, doc, stats)
        if not rows:
            return []
        table = pd.DataFrame(rows)
        ok = table["s_mean_surprisal"].notna().to_numpy()
        probs = np.full(len(table), np.nan)
        if ok.any():
            probs[ok] = self.sentence_model.predict_proba(table.loc[ok, SENTENCE_FEATURES])[:, 1]
        out = []
        for (_, r), p in zip(table.iterrows(), probs):
            reasons = []
            if math.isnan(p):
                reasons.append("not scored (beyond GPT-2's 512-token window)")
            else:
                if r["s_top10_frac"] >= PREDICTABLE_TOP10:
                    reasons.append(f"very predictable wording ({r['s_top10_frac']:.0%} of words in GPT-2's top-10 guesses)")
                if r["s_starts_transition"]:
                    reasons.append("stock transition opener")
                if r["s_stock"]:
                    reasons.append("stock phrase")
                if r["s_hedges"]:
                    reasons.append("hedging")
                if len(table) >= 4 and abs(r["s_len_rel"] - 1) <= UNIFORM_LENGTH_BAND:
                    reasons.append("length right at the text's average (uniform rhythm)")
            out.append(SentenceResult(r["text"], int(r["start"]), int(r["end"]), float(p), reasons))
        return out
