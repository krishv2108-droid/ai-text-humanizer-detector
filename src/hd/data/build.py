"""Assemble data/processed/corpus.parquet and the story-generation prompt list."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from hd.config import load_config, resolve_path
from hd.data.loaders import load_hc3_pairs, load_writingprompts_candidates, make_id
from hd.data.splits import add_splits


def build_corpus(cfg: dict[str, Any]) -> pd.DataFrame:
    raw, processed = resolve_path(cfg, "raw"), resolve_path(cfg, "processed")
    processed.mkdir(parents=True, exist_ok=True)
    seed, dcfg = cfg["seed"], cfg["data"]

    stories = load_writingprompts_candidates(raw, seed, tuple(dcfg["story_words"]))
    stories = stories.head(dcfg["story_pairs"]).reset_index(drop=True)

    # Prompts for the generator; target length mirrors the human story, rounded to 10 words.
    with open(processed / "story_prompts.jsonl", "w") as f:
        for row in stories.itertuples():
            f.write(
                json.dumps(
                    {
                        "prompt_id": row.prompt_id,
                        "prompt": row.prompt,
                        "target_words": int(round(row.n_words / 10.0) * 10),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    parts = [stories, load_hc3_pairs(raw, seed, dcfg["pairs_per_source"], dcfg["min_words"])]

    ai_path = processed / "ai_stories.jsonl"
    if ai_path.exists():
        prompts = dict(zip(stories.prompt_id, stories.prompt))
        ai_rows = []
        for line in open(ai_path):
            item = json.loads(line)
            if item["prompt_id"] not in prompts:  # prompt no longer among the first story_pairs
                continue
            ai_rows.append(
                {
                    "id": make_id("ai-story", item["prompt_id"], item["generator"]),
                    "prompt_id": item["prompt_id"],
                    "genre": "story",
                    "source": "claude_generated",
                    "label": 1,
                    "generator": item["generator"],
                    "prompt": prompts[item["prompt_id"]],
                    "text": item["text"],
                    "n_words": len(item["text"].split()),
                }
            )
        parts.append(pd.DataFrame(ai_rows))

    corpus = pd.concat([p for p in parts if len(p)], ignore_index=True)
    corpus = add_splits(corpus, seed, cfg["splits"])
    # Only prompts that have both a human and an AI text can train/evaluate a detector.
    labels_per_prompt = corpus.groupby("prompt_id")["label"].nunique()
    corpus["has_pair"] = corpus["prompt_id"].map(labels_per_prompt).eq(2)
    corpus.to_parquet(processed / "corpus.parquet", index=False)
    return corpus


def summarize(corpus: pd.DataFrame) -> pd.DataFrame:
    """Counts by genre/label/split, plus median length, for the README and DATA_SOURCES.md."""
    paired = corpus[corpus["has_pair"]]
    table = (
        paired.groupby(["genre", "split", "label"])
        .agg(texts=("id", "count"), median_words=("n_words", "median"))
        .reset_index()
    )
    return table


if __name__ == "__main__":
    cfg = load_config()
    corpus = build_corpus(cfg)
    results = resolve_path(cfg, "results")
    results.mkdir(exist_ok=True)
    summary = summarize(corpus)
    summary.to_csv(results / "corpus_summary.csv", index=False)
    print(f"{len(corpus)} texts, {int(corpus['has_pair'].sum())} in complete human/AI pairs")
    print(corpus.groupby(["genre", "label"]).size().unstack(fill_value=0))
    print(summary.to_string(index=False))
