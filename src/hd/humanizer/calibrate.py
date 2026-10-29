"""How do the guardrails behave? Simulate known failure modes on real validation texts and measure.

No LLM needed. For each AI text from the validation split we build:
  identical      the text unchanged (the best possible rewrite as far as meaning goes)
  drop30         30% of sentences removed        -> should be flagged as dropped content
  swap30         30% of sentences replaced by sentences of unrelated texts -> invented content/drift
  unrelated      a different text of the same genre -> meaning drift
  same_topic     the human answer to the same prompt -> different words AND different content, same topic
  typos5         5% of long words get two adjacent letters swapped -> typo flag
and report each guardrail metric, so thresholds are set from evidence rather than guesses.
"""

from __future__ import annotations

import random
import re
from typing import Any

import numpy as np
import pandas as pd

from hd.config import resolve_path
from hd.data.splits import load_split
from hd.humanizer.guardrails import GuardrailChecker, SentenceEmbedder, Thresholds, split_sentences

_WORD = re.compile(r"[A-Za-z]{5,}")


def inject_typos(text: str, rate: float, rng: random.Random) -> str:
    """Swap two adjacent inner letters in a `rate` share of the long words (e.g. 'receive' -> 'recieve')."""

    def swap(m: re.Match) -> str:
        w = m.group(0)
        if rng.random() >= rate:
            return w
        i = rng.randrange(1, len(w) - 2)
        return w[:i] + w[i + 1] + w[i] + w[i + 2 :]

    return _WORD.sub(swap, text)


def make_scenarios(row: pd.Series, same_topic: str | None, other_texts: list[str], rng: random.Random) -> dict[str, str]:
    sents = split_sentences(row["text"])
    keep = [s for s in sents if rng.random() > 0.30] or sents[:1]
    donors = [s for t in rng.sample(other_texts, 3) for s in split_sentences(t)]
    swapped = [rng.choice(donors) if rng.random() < 0.30 else s for s in sents]
    out = {
        "identical": row["text"],
        "drop30": " ".join(keep),
        "swap30": " ".join(swapped),
        "unrelated": rng.choice(other_texts),
        "typos5": inject_typos(row["text"], 0.05, rng),
    }
    if same_topic:
        out["same_topic"] = same_topic
    return out


def run_calibration(cfg: dict[str, Any], n: int = 150) -> pd.DataFrame:
    processed, results = resolve_path(cfg, "processed"), resolve_path(cfg, "results")
    corpus = pd.read_parquet(processed / "corpus.parquet")
    val = load_split(corpus[corpus["has_pair"]], "val")  # test split stays locked
    rng = random.Random(cfg["seed"])

    ai = val[(val["label"] == 1)]
    ai = ai[ai["text"].map(lambda t: len(split_sentences(t)) >= 4)]
    sample = ai.sample(min(n, len(ai)), random_state=cfg["seed"])
    human_by_prompt = val[val["label"] == 0].set_index("prompt_id")["text"].to_dict()

    checker = GuardrailChecker(SentenceEmbedder(), Thresholds.from_config(cfg))
    rows = []
    for _, row in sample.iterrows():
        others = val[(val["genre"] == row["genre"]) & (val["prompt_id"] != row["prompt_id"])]["text"].tolist()
        for name, rewrite in make_scenarios(row, human_by_prompt.get(row["prompt_id"]), others, rng).items():
            r = checker.check(row["text"], rewrite)
            rows.append({"scenario": name, "genre": row["genre"], "similarity": r.similarity, "dropped_frac": r.dropped_frac,
                         "invented_frac": r.invented_frac, "length_ratio": r.length_ratio, "fk_delta": r.fk_delta,
                         "novel_rare_rate": r.novel_rare_rate, "flagged": not r.ok,
                         "flags": [f.split(" (")[0] for f in r.flags]})  # fmt: skip
    raw = pd.DataFrame(rows)
    metrics = ["similarity", "dropped_frac", "invented_frac", "length_ratio", "fk_delta", "novel_rare_rate"]
    summary = raw.groupby("scenario")[metrics].quantile([0.05, 0.5, 0.95]).unstack().round(3)
    summary.columns = [f"{m}_p{int(q * 100):02d}" for m, q in summary.columns]
    summary["flagged_rate"] = raw.groupby("scenario")["flagged"].mean().round(3)
    for flag in ("meaning drift", "dropped content", "invented content", "length changed too much", "readability shifted", "possible typos or gibberish"):
        summary[f"caught_by: {flag}"] = raw.groupby("scenario")["flags"].apply(lambda s, f=flag: s.map(lambda x: f in x).mean()).round(3)
    summary["n"] = raw.groupby("scenario").size()
    summary.to_csv(results / "guardrail_calibration.csv")
    return summary


# ---- do the content checks tell restructuring from damage? ----------------------------------------------------------

_BREAK = re.compile(r"(?:,\s+(?:and|but|so|yet|while|as|because)\s+|;\s+|\s*[—–]\s*|,\s+)")


def split_into_fragments(text: str) -> str:
    """Cut every sentence at its commas, dashes, semicolons and conjunctions (parts of 3+ words), each part a sentence.

    Content-preserving by construction: no content word is added or removed; only punctuation, capitals and the
    linking conjunction at each cut (", and" / ", but" ...) change. This is the kind of restructuring a 'vary your
    rhythm, use fragments' prompt produces.
    """
    out = []
    for sent in split_sentences(text):
        end = sent[-1] if sent[-1] in ".!?" else "."
        body = sent.rstrip(".!?\"' ")
        parts, buf = [], ""
        for piece in re.split(f"({_BREAK.pattern})", body):
            if _BREAK.fullmatch(piece):
                if len(buf.split()) >= 3:
                    parts.append(buf)
                    buf = ""
                else:
                    buf += piece
            else:
                buf += piece
        parts.append(buf)
        parts = [p.strip() for p in parts if p.strip()]
        out.extend(f"{p[0].upper()}{p[1:]}{'.' if i < len(parts) - 1 else end}" for i, p in enumerate(parts))
    return " ".join(out)


def merge_pairs(text: str) -> str:
    """Join sentences two at a time with ', and' (content-preserving)."""
    sents = split_sentences(text)
    merged = []
    for i in range(0, len(sents), 2):
        a = sents[i]
        if i + 1 < len(sents):
            b = sents[i + 1]
            merged.append(f"{a.rstrip('.!?')}, and {b[0].lower()}{b[1:]}")
        else:
            merged.append(a)
    return " ".join(merged)


PRESERVING = ("identical", "split", "merge")
DAMAGING = ("drop30", "swap30")


def coverage_scenarios(text: str, other_texts: list[str], rng: random.Random) -> dict[str, str]:
    sents = split_sentences(text)
    keep = [s for s in sents if rng.random() > 0.30] or sents[:1]
    donors = [s for t in rng.sample(other_texts, 3) for s in split_sentences(t)]
    swapped = [rng.choice(donors) if rng.random() < 0.30 else s for s in sents]
    return {"identical": text, "split": split_into_fragments(text), "merge": merge_pairs(text), "drop30": " ".join(keep), "swap30": " ".join(swapped)}


def run_restructuring_check(cfg: dict[str, Any], split: str = "train", genre: str = "story", n: int = 60) -> pd.DataFrame:
    """Do the content checks tell restructuring from damage? Per text and scenario: dropped/invented fractions.

    Uses TRAIN texts by default (this is design/validation work on the guardrail, not a result about rewrites).
    """
    corpus = pd.read_parquet(resolve_path(cfg, "processed") / "corpus.parquet")
    part = load_split(corpus[corpus["has_pair"]], split)  # the test split stays locked
    ai = part[(part["label"] == 1) & (part["genre"] == genre)]
    ai = ai[ai["text"].map(lambda t: len(split_sentences(t)) >= 6)]
    sample = ai.sample(min(n, len(ai)), random_state=cfg["seed"])
    rng = random.Random(cfg["seed"])
    checker = GuardrailChecker(SentenceEmbedder(), Thresholds.from_config(cfg))
    rows = []
    for _, row in sample.iterrows():
        others = ai[ai["prompt_id"] != row["prompt_id"]]["text"].tolist()
        for name, rewrite in coverage_scenarios(row["text"], others, rng).items():
            sem = checker.semantic(row["text"], rewrite)
            rows.append({"scenario": name, "id": row["id"], "dropped_frac": sem["dropped_frac"], "invented_frac": sem["invented_frac"], "similarity": sem["similarity"]})  # fmt: skip
    return pd.DataFrame(rows)


def summarize_restructuring(raw: pd.DataFrame, max_dropped: float, max_invented: float) -> pd.DataFrame:
    """How well the content checks separate damage from restructuring: AUC, and flag rates at the configured thresholds."""
    from sklearn.metrics import roc_auc_score

    keep = raw[raw["scenario"].isin(PRESERVING)]
    rows = []
    for damage, metric, limit in (("drop30", "dropped_frac", max_dropped), ("swap30", "invented_frac", max_invented)):
        bad = raw[raw["scenario"] == damage]
        both = pd.concat([keep.assign(y=0), bad.assign(y=1)])
        rows.append({
            "damage": damage, "metric": metric, "threshold": limit, "n_texts": bad["id"].nunique(),
            "auc_vs_restructuring": roc_auc_score(both["y"], both[metric]), "flagged_when_damaged": float((bad[metric] > limit).mean()),
            **{f"flagged_when_{s}": float((raw[raw["scenario"] == s][metric] > limit).mean()) for s in PRESERVING},
            "restructuring_p95": float(keep[metric].quantile(0.95)),
        })  # fmt: skip
    return pd.DataFrame(rows)
