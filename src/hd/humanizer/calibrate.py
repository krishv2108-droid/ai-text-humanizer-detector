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
