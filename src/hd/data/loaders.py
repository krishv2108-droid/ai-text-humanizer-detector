"""Turn the raw downloads into clean, paired records (one row per text)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from hd.data.clean import (
    count_words,
    has_residual_tokenization,
    has_unusable_content,
    normalize_text,
)
from hd.data.splits import stable_hash01

# genre of each HC3 file. HC3 is question answering, so "essay" really means explanatory prose
# answering a question, and "casual" is Reddit ELI5 (conversational explanations).
HC3_GENRE = {
    "open_qa": "essay",
    "finance": "essay",
    "medicine": "essay",
    "wiki_csai": "essay",
    "reddit_eli5": "casual",
}

_PROMPT_TAG = re.compile(r"^\s*\[\s*[A-Za-z]{2,3}\s*\]\s*")  # "[ WP ]", "[ EU ]", ...


def make_id(*parts: str) -> str:
    return hashlib.sha1("␟".join(parts).encode()).hexdigest()[:12]


def _record(**kw: Any) -> dict[str, Any]:
    kw["n_words"] = count_words(kw["text"])
    return kw


def load_writingprompts_candidates(raw_dir: Path, seed: int, word_range: tuple[int, int]) -> pd.DataFrame:
    """One eligible human story per distinct prompt, ordered by a stable hash of the prompt."""
    df = pd.read_parquet(raw_dir / "writingprompts_validation.parquet")
    lo, hi = word_range
    rows: dict[str, dict[str, Any]] = {}
    for prompt, story in zip(df["prompt"], df["story"]):
        prompt_clean = normalize_text(_PROMPT_TAG.sub("", prompt))
        if count_words(prompt_clean) < 8 or has_unusable_content(prompt_clean):
            continue
        text = normalize_text(story)
        if not lo <= count_words(text) <= hi:
            continue
        if has_unusable_content(text) or has_residual_tokenization(text):
            continue
        pid = make_id("wp", prompt_clean)
        # keep the story with the smallest hash: deterministic, independent of row order
        rank = stable_hash01(seed, "story", pid, text)
        if pid not in rows or rank < rows[pid]["_rank"]:
            rows[pid] = {
                "_rank": rank,
                **_record(
                    id=make_id("wp-human", pid),
                    prompt_id=pid,
                    genre="story",
                    source="writingprompts",
                    label=0,
                    generator="human",
                    prompt=prompt_clean,
                    text=text,
                ),
            }
    out = pd.DataFrame(rows.values())
    out["_order"] = [stable_hash01(seed, "story-order", p) for p in out["prompt_id"]]
    return out.sort_values("_order").drop(columns=["_rank", "_order"]).reset_index(drop=True)


def load_hc3_pairs(raw_dir: Path, seed: int, pairs_per_source: dict[str, int], min_words: int) -> pd.DataFrame:
    """One human + one ChatGPT answer per question, filtered and normalized identically."""
    records: list[dict[str, Any]] = []
    for source, cap in pairs_per_source.items():
        path = raw_dir / f"hc3_{source}.jsonl"
        by_question: dict[str, tuple] = {}
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                question = normalize_text(item["question"])
                if not question or has_unusable_content(question):
                    continue
                pid = make_id("hc3", source, question)
                human = _pick(item["human_answers"], seed, pid, "human", min_words)
                ai = _pick(item["chatgpt_answers"], seed, pid, "ai", min_words)
                if human is None or ai is None:
                    continue
                # Some questions repeat with different answers: keep one pair per question so
                # ids stay unique and the human:AI ratio stays exactly 1:1.
                candidate = (stable_hash01(seed, "pair-order", pid), pid, question, human, ai)
                if pid not in by_question or stable_hash01(seed, "dup", human) < stable_hash01(
                    seed, "dup", by_question[pid][3]
                ):
                    by_question[pid] = candidate
        pairs = sorted(by_question.values())
        for _, pid, question, human, ai in pairs[:cap]:
            for label, generator, text in ((0, "human", human), (1, "chatgpt", ai)):
                records.append(
                    _record(
                        id=make_id("hc3", source, pid, generator),
                        prompt_id=pid,
                        genre=HC3_GENRE[source],
                        source=f"hc3_{source}",
                        label=label,
                        generator=generator,
                        prompt=question,
                        text=text,
                    )
                )
    return pd.DataFrame(records)


def _pick(answers: list[str], seed: int, pid: str, side: str, min_words: int) -> str | None:
    candidates = []
    for a in answers:
        t = normalize_text(a)
        if count_words(t) < min_words or has_unusable_content(t) or has_residual_tokenization(t):
            continue
        candidates.append((stable_hash01(seed, side, pid, t), t))
    return min(candidates)[1] if candidates else None
