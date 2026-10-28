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
`LEARNING.md`. Current: **M4 - explanations + independent detector** (done). AI stories still wait for an API key.

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

## Baseline detector (M3)

Logistic regression and boosted trees on interpretable features. Hyperparameters are tuned by
prompt-grouped cross-validation on the train split; the numbers below are on the **validation split**
(730 texts, held out from training and tuning). The **test split is still locked** and is used once, in
the final evaluation. Positive class = AI; 95% intervals resample whole prompts.

| Detector | Accuracy | Precision | Recall | ROC-AUC (95% CI) | Length-matched AUC |
|---|---|---|---|---|---|
| Stylometric features, logistic | 0.860 | 0.842 | 0.888 | 0.915 (0.895-0.936) | 0.914 |
| + POS n-grams, logistic | 0.868 | 0.851 | 0.893 | 0.935 (0.918-0.952) | 0.937 |
| + GPT-2 predictability, logistic | 0.971 | 0.960 | 0.984 | 0.996 (0.992-0.998) | 0.999 |
| + GPT-2 predictability, boosted trees | 0.982 | 0.978 | 0.986 | 0.998 (0.996-0.999) | 0.998 |

![ROC curves on the validation split](results/figures/baseline_roc.png)

Per-genre rows, confusion counts and train-vs-val gaps: [results/baseline_metrics.csv](results/baseline_metrics.csv);
confusion matrices: [results/figures/baseline_confusion.png](results/figures/baseline_confusion.png).

How to read this honestly:
- **The interpretable features alone reach ~0.92 AUC; GPT-2 predictability adds the rest (~0.996).**
  Burstiness-type, punctuation and transition features carry real signal without any neural model.
- **This is a detector for December-2022 ChatGPT (essay, casual), not for modern models.** Near-perfect
  scores here say the task is easy for *that* AI text. Whether it holds on Claude-written stories, and on
  rewrites designed to evade it, is exactly what M5-M7 measure.
- Boosted trees fit the training set perfectly (train accuracy 1.00) yet validate at 0.982, so the
  validation number, not the training number, is the one to quote.

## Using the detector, and the independent check (M4)

```bash
uv run hd detector train-sentence                    # once: fits the per-sentence model (~3 min)
uv run hd detect "paste a text here" --external      # or --file story.txt, or pipe via stdin
```

You get P(AI), which feature groups drove the score, and the most AI-like sentences with plain-language
reasons (predictable wording, stock transition opener, hedging, uniform length). Drivers use
*neutralization*: swap a feature group's values for typical ones and see how much P(AI) moves
(`+` = pushes toward AI). Sentence scores come from a small model on GPT-2 surprisal-in-context plus local
cues. Its labels are inherited from the document, so it ranks sentences within a text; it does not certify
any single sentence (validation sentence AUC 0.90, [results/sentence_metrics.csv](results/sentence_metrics.csv)).

**Independent detector:** [`desklib/ai-text-detector-v1.01`](https://huggingface.co/desklib/ai-text-detector-v1.01)
(DeBERTa-v3-large, MIT, trained on the RAID benchmark, not on our data). It is used only to *evaluate*;
nothing in this project is trained or tuned against it.

![Independent detector scores by true label](results/figures/external_scores.png)

Validation split, raw human vs raw AI text ([results/external_metrics.csv](results/external_metrics.csv)):

| Detector | ROC-AUC | AI caught @0.5 | Human flagged @0.5 | AI caught @ 5% human flagged |
|---|---|---|---|---|
| Ours (boosted trees) | 0.998 | 98.6% | 2.2% | 98.9% |
| desklib (independent) | 0.996 | 99.7% | **25.8%** | 98.4% |

- **The independent detector ranks texts about as well as ours, but its default 0.5 cutoff is badly
  miscalibrated for this text:** it flags a quarter of human answers (a third of the essays) as AI.
- **So every later comparison uses a matched false-positive rate.** Each detector's cutoff is chosen on
  validation *human* texts to flag 5% of them ([results/detector_thresholds.json](results/detector_thresholds.json)),
  then frozen. Otherwise "humanized text is flagged less" could just mean "the detector over-flags everything".
- **Both detectors find raw ChatGPT-2022 text easy.** The meaningful test is rewritten text (M6-M7).
- **Anecdote, not a result:** on one hand-written paragraph mixing casual and stock-AI sentences, our
  detector said 0.002 and desklib 1.000. One example proves nothing, but it fits the concern that ours leans
  on GPT-2 predictability, which a few quirky specifics can move.

## Data

```bash
uv sync --extra data --extra llm
uv run hd data download            # ~104 MB from Hugging Face into data/raw/
uv run hd data build               # clean, pair, split -> data/processed/corpus.parquet
export ANTHROPIC_API_KEY=...       # your key, never committed
uv run hd data generate-stories    # Claude stories for the 300 prompts (resumable, cached)
uv run hd data build               # rebuild so the new stories join the corpus
```

Detector (fits 5 variants, ~1 minute):

```bash
uv run hd detector train-baseline      # writes models/ (git-ignored) and results/baseline_*.csv + figures
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
