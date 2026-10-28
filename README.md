# Humanizer vs. Detector: an adversarial study of AI-written text

**Research question:** how well can AI-generated writing (especially stories) be rewritten to read as
human, and which linguistic features give AI text away?

## Results summary

_Pending. Filled in at Milestone 7 once the evaluation has run._

| Text | Own detector (flagged as AI) | Independent detector (flagged as AI) |
|------|------------------------------|--------------------------------------|
| Raw AI text | - | - |
| Humanized text | - | - |

## How it works

1. **Detector:** interpretable linguistic features + a simple classifier, with per-sentence explanations.
2. **Humanizer:** an LLM rewrites AI text, steered by the weaknesses the detector found.
3. **Adversarial loop:** rewrite, score, feed feedback back, repeat (capped).
4. **Honest evaluation:** rewrites are also scored by an independent open-source detector the
   humanizer was never optimized against.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+ (uv fetches it for you).

```bash
uv sync                  # core deps
uv run pytest -q         # run tests
uv run hd --version
```

Optional extras are added as milestones need them: `uv sync --extra nlp`, `--extra ml`,
`--extra llm`, `--extra app`.

The Anthropic API key is read from the `ANTHROPIC_API_KEY` environment variable (never committed).

## Project status

Built in small milestones, one commit each (see `git log`); per-milestone notes are in
`LEARNING.md`. Current: **M0 - scaffold**.

## Layout

`src/hd/` holds the code (data, features, detector, humanizer, loop, eval, cli), `app/` the
Streamlit UI, `tests/` the tests, `configs/default.yaml` all settings including the random seed (42).
