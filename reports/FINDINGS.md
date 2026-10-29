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
per-sentence explanations; two independent open-source detectors that no part of our pipeline trains: `desklib` (can optionally
steer the loop) and `fakespot` (a RoBERTa-base classifier that only ever judges, and decides nothing); an LLM humanizer whose prompt can use the detector's findings; guardrails (meaning similarity, dropped/invented
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

**Third detector (`fakespot-ai/roberta-base-ai-text-detection-v1`, Apache-2.0, 503 MB).** Added as a judge so that desklib can steer
without losing every independent check. Validation split, frozen 5%-false-positive cutoffs: AUC story 0.995 (n=100), essay 0.999,
casual 0.999; AI caught 98% / 99.6% / 100%. Rank correlation between detectors on validation texts: ours-desklib 0.82, ours-fakespot
0.85, desklib-fakespot 0.89, so the three are related, not three independent votes. Its scores are saturated (the story cutoff is
0.996) and its training data is documented only in a separate technical report, so overlap with our sources is not ruled out.

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

## 2. Can the text be rewritten to look human?

Headline measure: a rewrite **counts** only if it passes every guardrail (meaning similarity, dropped and invented content,
length, reading level, typos, and since v2 that numbers and names survive) **and** neither detector flags it. A failed rewrite
means the user keeps the original, which is flagged, so it never counts as a success.

**Baseline (prompt v1).** 34 validation stories of at most 400 words, humanizer `qwen3.8-27b`, `full` prompt, up to 4 rewrites,
cutoffs frozen at 5% false positives, 95% Wilson intervals (`results/loop_reports/outcomes_story_val_full_qwen_v1_n34.*`):

| | Flagged before | Flagged after the loop |
|---|---|---|
| Our detector (steers the loop) | 97% (85-100%) | 6% (2-19%) |
| Independent detector, desklib | 100% (90-100%) | 68% (51-81%) |
| Kept rewrite fools both | | 11 of 34 = 32% (19-49%) |

Replaying the v2 fact guard over those kept rewrites: 10 of 34 (29%, 17-46%), and 4 of the 32 kept rewrites had lost a number or
a name that the old guardrails did not check. (The earlier 10-story pilot reported desklib 60% flagged; the larger sample says 68%.)

**Third detector on the same 34 validation stories (scored after the fact; it steered nothing).** The kept v1 rewrites fool
fakespot on **44% (29-61%)** of stories, versus 32% for desklib; all three detectors at once: **8 of 34 = 24% (12-40%)**. Evasion
transfers: of 40 rewrites that fool desklib, 80% also fool fakespot, while of 59 that desklib still flags, 36% fool fakespot
(Spearman 0.70 between the two on 99 rewrites). That transfer is the reason steering with desklib is a sound design, judged by
fakespot. `--steer-desklib` is implemented and tested (the loop stops only when both ours and desklib pass, keeps the rewrite with the
lowest worst-case score, and tells a rewrite desklib still flags to be bolder; fakespot never decides anything), but **it has not been
measured**: the free qwen quota ran out after 3 steered stories, which is not evidence either way.

**Why it plateaus: fidelity versus evasion** (`results/frontier_story_replay_v1_v2dev.csv`; 100 rewrites with desklib scores):

| Meaning similarity of the rewrite | Fooled desklib | Passed the guardrails |
|---|---|---|
| over 0.90 | 9% | 97% |
| 0.85 to 0.90 | 35% | 85% |
| 0.80 to 0.85 | 61% | 57% |
| under 0.80 | 91-100% | 0% |

Desklib's score has a Spearman correlation of 0.73 with meaning similarity (0.68 with bigram overlap; the two overlap, r = 0.83).
Replaying different similarity floors over the stored attempts (an approximation: only attempts the loop actually made exist),
"fools both" is about 30% at a floor of 0.80, 15% at 0.85 and 6% at 0.88 or above. Faithful rewrites keep what desklib
recognizes; the guardrail floor sits exactly where evasion becomes common. This is a trade-off, not a prompt bug.

**Our own detector is gameable; desklib is not (by the same trick).** Control on 40 train stories (`hd eval control`;
`results/control_rule_based_story_train.csv`), changing only punctuation and sentence boundaries, no content word altered:

| Variant | Ours flagged | Desklib flagged |
|---|---|---|
| original | 100% (91-100%) | 100% (91-100%) |
| split into fragments | 18% (9-32%) | 90% (77-96%) |
| merge sentence pairs | 100% | 95% (84-99%) |

So most of "ours 6% flagged" reflects surface rhythm and punctuation features, not better text. This is also a no-LLM
baseline for the humanizer: it fools our detector 82% of the time and desklib 10%, versus the LLM loop's 94% and 32% (different story sets: 40 train versus 34 validation stories).

**What else was tried (logs in `data/processed/loop_train_*`, 12 train stories each, paired where noted):**

| Change | Result |
|---|---|
| Prompt v2: repair mode (fix the rejected rewrite, name the missing sentences/numbers), two-sided feedback (tell an overshot rewrite to come back), number-and-name guard, moderated rhythm | Kept 12/12, meaning similarity 0.90, no lost numbers, no caricature (short sentences +0.05 human standard deviations, v1 +1.06). Fooled both on 2 of 12 (5-45%): no measurable gain over v1's 32%; n too small to rule out a difference either way |
| Prompt v3: plus "say it in different, plainer words" | **No effect**: desklib on the first rewrite 0.76 (v2) vs 0.92 (v3), paired p = 0.68; fooled both 3 of 12. Reverted |
| Humanizer `gpt-oss-120b` (same family as the story generator) | Tiny edits (similarity 0.97), fooled both on 0 of 9 (0-30%); hit its daily limit before 12 |
| Humanizer `gpt-oss-20b` (pilot, 10 stories) | Ours 90% to 90%, desklib 100% to 90%, stalled on 7 of 10 |
| Span-based content check (match runs of 2-3 sentences) | Useless (AUC about 0.5); removed. The existing sentence-level check already separates deleting or swapping sentences from merely splitting or merging them (AUC 0.96 / 0.94, `hd humanizer check-restructuring`), which refuted my guess that it punished fragments |

**Prompt strategies, first rewrite, pilot (10 validation stories, v1, `results/ablation_prompts_story_val_qwen3.8-27b_le400w.csv`):**
naive 100% / 100% still flagged (ours / desklib); feedback 20% / 90%; craft 0% / 0% but only 10% pass the guardrails; full 0% / 30%
and 40% pass. Feedback steered by our detector transfers badly to desklib; the technique list fools both but breaks the meaning.

**A guardrail changed after seeing real rewrites.** The original reading-level rule (grade may not move more than 3
levels) rejected 8 of 10 qwen rewrites even though AI stories read at grade 8.3 and human stories at 5.0. It now requires
the rewrite's grade to lie within the range of real human text for the genre (1st-99th percentile of train-split human
texts). With the old rule the same 10-story loop ended with ours 80% and desklib 100% flagged, 20% of rewrites accepted
(`data/processed/..._fkdelta-rule.jsonl`, `results/loop_pilot_story_val_qwen.csv`). Because the change was
motivated by this pilot, the final evaluation must report both rules.

## 3. Limitations

- **One generator per genre, one human source per genre.** HC3's AI side is ChatGPT from December 2022; the story AI side is
  one open-weights model with a plain prompt. High AUCs show that these specific generators are easy to separate from these
  specific human sources, not that AI text in general is. No out-of-distribution test (other LLMs, other prompts, other
  human writers) was run.
- **Three detectors are not three independent votes.** They agree on rank order (r = 0.82-0.89) and share the same blind spot of
  being trained on text of known LLMs. If desklib steers, fakespot is the only detector left that is both independent and unsteered.
- **"Independent" means not trained or steered by us.** Desklib's own training data may overlap the sources we use; I did not
  verify that. Its calibration is poor at 0.5 (it flags ~24% of human validation text), so all comparisons use matched 5%
  false-positive cutoffs.
- **The cutoffs are noisy.** The story cutoff comes from 50 human validation stories; 5% of 50 is 2-3 texts.
- **Small samples.** The main humanizer result is 34 stories of at most 400 words (intervals about +/-15 points), the v2 and v3
  comparisons are 12 stories (about +/-20-25 points), the strategy ablation is 10. One humanizer model works at all. Treat v2 versus v1 as undecided.
- **The humanizer is limited by what free models can do.** The two models of the generator's own family make near-copies; only
  `qwen3.8-27b` rewrites enough, and it fools desklib on about a third of stories at the meaning floor used.
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
- **Not tested:** desklib steering at scale, commercial detectors (see `reports/manual_detector_log.md`), adversarial retraining of the detector on fooling
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

Decisions already fixed before looking at test data: both steering modes are reported (ours only; ours + desklib) with fakespot as the
judge in both; the humanizer model (`qwen3.8-27b`, the only one that rewrites enough), the
`full` strategy with prompt v2 and repair on (prompt v1 is reported as the baseline), the guardrail rule (reported under both
reading-level rules), and that results from different humanizer models or prompt versions are reported separately and never pooled. When it runs, this section gets replaced with the actual numbers, including any
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
