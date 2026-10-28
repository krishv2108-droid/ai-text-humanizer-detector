"""Generate the AI side of the story pairs: one LLM-written story per WritingPrompts prompt.

The prompt is deliberately plain (no craft tips): this is the "default AI voice" the detector must
learn to spot and the humanizer must later fix. We ask for the human story's approximate length so
length can't be used as a shortcut, and for no title/markdown so formatting can't either.
Resumable: prompts already present in the output file are skipped. On a free tier the run may hit a
per-minute limit (the client waits) or a per-day limit (the run stops cleanly; rerun later to continue).
"""

from __future__ import annotations

import json
from pathlib import Path

from typing import Any

from hd.data.clean import count_words, has_markdown
from hd.humanizer.rewrite import clean_output
from hd.llm import DailyLimitReached, LLMClient, RateLimited, RefusalError

SYSTEM = (
    "You are a fiction writer. Respond with the story text only: plain prose paragraphs, "
    "no title, no markdown, no preamble, and no closing remarks."
)


def story_request(prompt: str, target_words: int) -> str:
    return f"Writing prompt: {prompt}\n\nWrite a short story in response, about {target_words} words."


def generate_stories(
    client: LLMClient,
    prompts_path: Path,
    out_path: Path,
    *,
    model: str,
    max_tokens: int,
    effort: str | None,
    limit: int | None = None,
) -> dict[str, Any]:
    done: set[str] = set()
    if out_path.exists():
        done = {json.loads(line)["prompt_id"] for line in open(out_path)}
    stats: dict[str, Any] = {"written": 0, "skipped_done": 0, "rejected": 0, "refused": 0, "stopped_by_limit": False, "message": ""}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prompts_path) as f, open(out_path, "a") as out:
        for line in f:
            if limit is not None and stats["written"] >= limit:
                break
            item = json.loads(line)
            if item["prompt_id"] in done:
                stats["skipped_done"] += 1
                continue
            try:
                resp = client.complete(
                    system=SYSTEM,
                    prompt=story_request(item["prompt"], item["target_words"]),
                    model=model,
                    # budget from the target length (about 1.3 tokens per word, with headroom), capped:
                    # free tiers meter tokens, so an oversized budget wastes the daily allowance
                    max_tokens=min(max_tokens, int(item["target_words"] * 1.8) + 150),
                    effort=effort,
                )
            except RefusalError:
                stats["refused"] += 1
                continue
            except RateLimited as err:  # includes DailyLimitReached: stop, keep what we have, resume later
                stats["stopped_by_limit"] = True
                kind = "daily limit" if isinstance(err, DailyLimitReached) else "rate limit"
                wait = f" (retry in about {err.retry_in_s / 60:.0f} min)" if err.retry_in_s else ""
                stats["message"] = f"stopped at the {kind}{wait}: {str(err)[:200]}"
                break
            text, _ = clean_output(resp.text)  # strips a preface/title the model added, normalizes
            # Truncated, empty, or formatted-like-a-chatbot outputs would leak formatting cues.
            if resp.stop_reason == "max_tokens" or count_words(text) < 50 or has_markdown(text):
                stats["rejected"] += 1
                continue
            out.write(
                json.dumps(
                    {
                        "prompt_id": item["prompt_id"],
                        "text": text,
                        "generator": model,
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
            stats["written"] += 1
    return stats
