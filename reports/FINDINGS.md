# Findings

**Status: the final held-out test evaluation has NOT been run.** Everything below comes from the training and
validation splits, or from a 10-story pilot on validation text. Treat the numbers as development results. The
test-set section at the end says exactly what will be run and how its use is logged.

## Question

How well can AI-written text, especially stories, be rewritten to read as human, and which linguistic features give
AI text away? Scope: stories first (293 prompt-matched pairs), explanatory answers ("essay", HC3) and casual Reddit
answers (HC3 ELI5) at lighter depth. Email is not covered (no clean paired dataset; see `data/DATA_SOURCES.md`).

## What was built

A feature-based detector (sentence rhythm, vocabulary, punctuation, part-of-speech shares, GPT-2 predictability) with
per-sentence explanations; an independent open-source detector (`desklib`) that no part of our pipeline trains or
steers; an LLM humanizer whose prompt can use the detector's findings; guardrails (meaning similarity, dropped/invented
content, length, reading level, typo/gibberish rate); and an adversarial loop that rewrites, re-scores and repeats.
Both detectors are always reported separately, at a cutoff that flags 5% of real human text (chosen once on
validation human text, then frozen).

## 1. What gives AI text away (detector, validation split)

Detection quality on validation text (cutoff = 5% of human text flagged; stories n=100 texts, essay n=454, casual n=276):

| Genre | Our detector AUC | Desklib AUC | AI caught (ours / desklib) |
|---|---|---|---|
| story | 0.998 | 0.998 | 98% / 100% |
| essay | 0.998 | 0.995 | 99% / 98% |
| casual | 0.999 | 0.999 | 99% / 99% |

These are very high because the task is easy in this setup (see Limitations): one AI generator per genre, one human
source per genre.

**Feature-group ablation** (`results/ablation_feature_groups.csv`, prompt-grouped 5-fold CV on train+val, a model per genre).
"Drop" removes a group; "only" uses just that group. Many groups carry overlapping information, so a small drop does
not mean a group is useless.

| Genre | AUC, all features | Biggest loss when a group is dropped | Strongest group on its own |
|---|---|---|---|
| essay (ChatGPT, 2022) | 0.997 | GPT-2 predictability: -0.020 | GPT-2 predictability: 0.989 |
| casual (ChatGPT, 2022) | 0.999 | GPT-2 predictability: -0.022 | GPT-2 predictability: 0.997 |
| story (gpt-oss-120b) | 0.997 | punctuation: -0.004 | punctuation 0.989, part-of-speech shares 0.989 |

Reading: for 2022 ChatGPT answers, how *predictable* the words are to GPT-2 gives it away. For the newer open model's
stories, predictability adds nothing; the giveaway is grammatical mix (more nouns and determiners, fewer verbs and
pronouns, long comma-heavy sentences) and punctuation. A cue that exposes one generator does not transfer to another.
The story signal is not an em-dash artifact: dropping the em dash or all punctuation features leaves AUC at 0.99+.

## 2. Can the text be rewritten to look human? (pilot: 10 validation stories, 400 words or fewer)

Humanizer `qwen3.8-27b`, `full` prompt, up to 4 rewrites, guardrails on, cutoffs frozen at 5% false positives.
Rates are shown with 95% Wilson intervals, because n=10 makes them wide (`results/loop_reports/`).

| | Flagged before | Flagged after the loop |
|---|---|---|
| Our detector (steers the loop) | 90% (60-98%) | 0% (0-28%) |
| Independent detector, desklib | 100% (72-100%) | 60% (31-83%) |

Prompt strategies, first rewrite of each story (`results/ablation_prompts_story_val_qwen3.8-27b_le400w.csv`):

| Strategy | Our detector flags | Desklib flags | Kept the meaning (guardrails) |
|---|---|---|---|
| naive ("sound human") | 100% | 100% | 100% |
| feedback (detector's findings) | 20% | 90% | 100% |
| craft (technique list) | 0% | 0% | 10% |
| full (both) | 0% | 30% | 40% |

What this suggests, with the strength of each claim:

1. *Steering by one detector transfers poorly to another.* Feedback alone fools ours (20% flagged) but not desklib (90%).
   Consistent across the pilot; would need the larger evaluation to quantify.
2. *Evasion trades against fidelity.* The technique list fooled both detectors but only 1 in 10 of those rewrites passed
   the guardrails. The loop's accepted rewrites have a mean meaning-similarity of 0.86.
3. *The rewrites overshoot human style.* Compared with human stories, kept rewrites have more short sentences (+0.9 human
   standard deviations; raw AI is -1.4) and more unpredictable wording than humans (`results/loop_reports/figures/feature_shift_*.png`).
   They read as staccato ("Honking. Drone hums."). A detector trained on such rewrites would probably catch them; that
   retraining defense is not built.
4. *The humanizer model matters.* `gpt-oss-20b` (same family as the story generator) left detection almost unchanged
   (ours 90% -> 90%, desklib 100% -> 90%) and stalled on 7 of 10 stories.

**A guardrail changed after seeing real rewrites.** The original reading-level rule (grade may not move more than 3
levels) rejected 8 of 10 qwen rewrites even though AI stories read at grade 8.3 and human stories at 5.0. It now requires
the rewrite's grade to lie within the range of real human text for the genre (1st-99th percentile of train-split human
texts). With the old rule the same loop ended with ours 80% and desklib 100% flagged, 20% of rewrites accepted
(`data/processed/..._fkdelta-rule.jsonl`, summary in `results/loop_pilot_story_val_qwen.csv`). Because the change was
motivated by this pilot, the final evaluation must report both rules.

## 3. Limitations

- **One generator per genre, one human source per genre.** HC3's AI side is ChatGPT from December 2022; the story AI side is
  one open-weights model with a plain prompt. High AUCs show that these specific generators are easy to separate from these
  specific human sources, not that AI text in general is. No out-of-distribution test (other LLMs, other prompts, other
  human writers) was run.
- **"Independent" means not trained or steered by us.** Desklib's own training data may overlap the sources we use; I did not
  verify that. Its calibration is poor at 0.5 (it flags ~24% of human validation text), so all comparisons use matched 5%
  false-positive cutoffs.
- **The cutoffs are noisy.** The story cutoff comes from 50 human validation stories; 5% of 50 is 2-3 texts.
- **Small samples.** The humanizer pilot is 10 stories of at most 400 words, one humanizer model, one run per setting. The
  Wilson intervals are about +/-25 points. No result in section 2 is established beyond "worth testing at scale".
- **The length confound is only partly removed.** HC3 ChatGPT answers are about twice as long as human ones. Length-stratified AUC
  reduces but does not eliminate this (a worst-case simulation still scores 0.50-0.60). Story lengths were matched at generation.
- **"Essay" is explanatory Q&A, not essays,** and medicine human answers carry a doctor-forum greeting, a genre-specific shortcut.
- **Guardrails are imperfect.** Embedding similarity catches wholesale drift and typos but only partly catches modest content
  loss, and it does not check facts: a kept story rewrite changed "seventy-eight" to "Seven-eight". Rewrite quality was judged by
  reading a handful of examples, not by a human study.
- **Rewrites may be unnatural to a human reader** even when detectors miss them (staccato fragments). Fooling a classifier is not
  the same as passing for human.
- **Compute shaped the design.** The free Groq tier allows about 65 rewrites per model per day, so sample sizes are small and
  two humanizer models (with different results) were used.
- **Not tested:** commercial detectors (see `reports/manual_detector_log.md`), adversarial retraining of the detector on fooling
  rewrites, essay and casual humanizing at all.
- **Dual use.** A tool that makes AI text harder to detect can be used to misrepresent authorship. This is a measurement study;
  note also that even a good detector flags some human writing (5% here by construction), which matters when detectors are used
  to accuse people.

## 4. Final held-out evaluation (not yet run)

The test split (15% of prompts, never used for training, tuning or threshold selection) is unlocked only by
`uv run hd eval final --unlock-test`, which:

- logs every unlock in `results/test_access_log.txt` and refuses to run a second time without `--again`;
- uses the cutoffs frozen on validation (`results/detector_thresholds.json`), retuning nothing;
- reports per genre, per detector: AUC, AI caught, human flagged (each with intervals), plus the loop outcome tables for any
  test-split loop logs (`hd loop run --split test --unlock-test`, also logged).

Decisions already fixed before looking at test data: which humanizer settings to run (the `full` strategy, and the four-strategy
ablation), the guardrail rule (reported under both reading-level rules), and that results from different humanizer models are
reported separately and never pooled. When it runs, this section gets replaced with the actual numbers, including any
that disappoint.

## Reproduce

```bash
uv run hd data build && uv run hd features build
uv run hd detector train-baseline && uv run hd detector train-sentence && uv run hd detector eval-external
uv run hd loop run --genre story --n 10 --max-words 400 --strategy full --model qwen/qwen3.8-27b
uv run hd eval report --log data/processed/loop_val_story_full_qwen3.8-27b_le400w.jsonl --features
uv run hd eval ablate-prompts --genre story --model qwen/qwen3.8-27b --max-words 400
uv run hd eval ablate-features
```
