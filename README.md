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

Optional extras are added as milestones need them: `--extra data`, `--extra nlp` (spaCy + model),
`--extra ml` (PyTorch + transformers), `--extra llm`, `--extra embed`, `--extra app`. Everything so far:

```bash
uv sync --extra data --extra llm --extra nlp --extra ml
```

The Anthropic API key is read from the `ANTHROPIC_API_KEY` environment variable (never committed).

## Project status

Built in small milestones, one commit each (see `git log`); per-milestone notes are in
`LEARNING.md`. Current: **M2 - features** (done). AI stories still wait for an API key.

## What gives AI text away? (preliminary, M2)

![Strongest single features, human vs AI](results/figures/feature_auc.png)

Univariate, **length-matched** AUC on the **training split** of the two genres with data so far
(essay and casual, human vs ChatGPT; stories arrive once the Claude stories are generated). Full table:
[results/feature_effects.csv](results/feature_effects.csv). Distributions:
[results/figures/feature_distributions.png](results/figures/feature_distributions.png).

- **GPT-2 predictability separates the classes almost perfectly** (AUC ~0.99 for the share of tokens in
  GPT-2's top-10 guesses), and it survives length matching. ChatGPT text is far more predictable to GPT-2
  than human text (mean surprisal 2.2 vs 3.6 nats).
- **Sentence rhythm is the strongest *interpretable* signal:** AI sentences are more uniform in length
  (burstiness AUC ~0.22, i.e. human text is burstier), in both genres.
- **AI uses fewer contractions and first-person words** (casual: contraction AUC 0.27) and **more
  sentence-initial transitions** (casual 0.71, essay 0.59).
- **Two hypotheses did not hold on this data:** em-dash rate (0.48) and my stock-phrase list (0.51) carry
  no signal. The AI side here is ChatGPT from December 2022, so this says little about today's models.
- **Caveat:** these are single-feature, training-split numbers. Length-banding reduces but does not fully
  remove the length confound (see `stratified_auc`). Held-out detector results arrive in M3.

## Data

```bash
uv sync --extra data --extra llm
uv run hd data download            # ~104 MB from Hugging Face into data/raw/
uv run hd data build               # clean, pair, split -> data/processed/corpus.parquet
export ANTHROPIC_API_KEY=...       # your key, never committed
uv run hd data generate-stories    # Claude stories for the 300 prompts (resumable, cached)
uv run hd data build               # rebuild so the new stories join the corpus
```

Features (GPT-2 weights, ~0.55 GB, download on first use into the Hugging Face cache):

```bash
uv run hd features build      # ~3 minutes on an Apple-silicon Mac; cached by text id
uv run hd features analyze    # effect sizes + figures, training split only
```

Sources, licenses, cleaning and known confounds: [data/DATA_SOURCES.md](data/DATA_SOURCES.md).
Genres: story, essay, casual (email deferred). The test split is locked in code until final evaluation.

## Layout

`src/hd/` holds the code (data, features, detector, humanizer, loop, eval, cli), `app/` the
Streamlit UI, `tests/` the tests, `configs/default.yaml` all settings including the random seed (42).
