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
lowest worst-case score, and tells a rewrite desklib still flags to be bolder; fakespot never decides anything).

**Desklib steering: no demonstrated gain on the judge.** Development run, 12 train stories (qwen, prompt v2): desklib fooled on 5 of 12
instead of 2, fakespot on 4 instead of 3. Validation run (settings fixed before it started: v2 prompt, repair on, desklib steering, up to
4 rewrites, fakespot as judge): the daily quota ended it at **21 of the 34 stories** (the first 21 in hash order; the other 13 were not
run). Paired with the v1 baseline on exactly those 21 stories:

| Same 21 validation stories | v1 baseline (ours steers) | v2 + desklib steers |
|---|---|---|
| Fools our detector | 21 | 18 |
| Fools desklib (steered in the v2 run) | 6 (14-50%) | 9 (24-63%) |
| **Fools fakespot (the judge; never steers)** | **11 (32-72%)** | **7 (17-55%)** |
| Fools all three | 5 (11-45%) | 4 (8-40%) |
| Mean meaning similarity of the kept rewrite | 0.87 | 0.90 |
| Tokens per story | 2,656 | 5,822 |

Per story on the judge: fooled in both runs 4, only by the steered run 3, only by the baseline 7 (sign test p = 0.34). So steering raised the
steered detector's evasion by about 14 points (not significant) without raising the judge's, at 2.2 times the cost. The two runs differ in the
prompt version and repair mode as well as in steering, so this is a comparison of two system versions, not a clean test of steering alone;
the cleaner paired test (same train stories, only steering changed) gave desklib 2 -> 5 and the judge 3 -> 4. The most likely reading is
the fidelity frontier below: the v2 runs keep more meaning (0.90 vs 0.87), and a rewrite that keeps more of the story is flagged more
often by every detector. Desklib steering is therefore not recommended as a default.

**Rewrite strength (development, 12 train stories, ours steers only).** `faithful` keeps every sentence's content (similarity at least
0.80, dropped sentences at most 20%); `strong` may compress or drop minor descriptive detail (at least 0.72, at most 30%); numbers, names,
typos, reading level and length are checked identically in both. The first strong prompt asked for "plainer, everyday words" and sentences
"under 6 words": 13 of its 15 rejected attempts then fell below the human reading-level floor (grade 0.9-2.2, childlike prose) and were
discarded, so it fooled all three detectors on 1 of 12 stories (faithful: 2 of 12). I did not loosen the guardrail (that would have
accepted rewrites that win by sounding childish); I rewrote the strong prompt to ask for plain but not childlike prose. With it, all three
detectors were fooled on 4 of 12 (14-61%), desklib on 6 of 12 and fakespot on 4 of 12 (25-75%, 14-61%), at meaning similarity 0.89. n = 12 and the
intervals overlap, so this selects a candidate; it is not a result. This was the only strong-prompt revision made after seeing results.

**Validation protocol fixed before the validation runs** (no further prompt or guardrail changes): the 34 validation stories of at most
400 words, qwen3.8-27b, `full` strategy, our detector steers only (desklib steering did not help), fakespot judges, up to 4 rewrites,
once in `faithful` mode and once in `strong` mode (the revised "s2" prompt); results are reported separately and paired on the stories
both runs completed. Logs of the first strong prompt are kept as `*_strong_s1.jsonl`.

**Validation of the frozen modes (protocol above; qwen3.8-27b, `full` strategy, our detector steers only, fakespot judges).**
Faithful completed all 34 stories; strong completed 24 before both API keys hit their daily limits (the other 10 were not run).

| 34 validation stories | Fools our detector | Fools desklib | Fools fakespot | Fools all three | Meaning kept |
|---|---|---|---|---|---|
| v1 baseline (old prompt, no fact guard) | 32 | 11 (19-49%) | 15 (29-61%) | 8 (12-40%) | 0.87 |
| v2 faithful | 29 | 7 (10-37%) | 12 (21-52%) | 5 (6-30%) | 0.91 |

| Paired, the 24 stories both modes completed | Fools desklib | Fools fakespot | Fools all three | Meaning kept | Tokens per story |
|---|---|---|---|---|---|
| faithful | 6 (12-45%) | 8 (18-53%) | 4 (7-36%) | 0.91 | 2,842 |
| strong | 5 (9-40%) | 10 (24-61%) | 4 (7-36%) | 0.89 | 2,834 |

Per story, strong versus faithful: on fakespot 6 only strong, 4 only faithful (sign test p = 0.75); on desklib 3 versus 4 (p = 1.0); on all
three 3 versus 3 (p = 1.0). **Strong mode shows no measurable gain, and the 4-of-12 versus 2-of-12 on the development stories did not
replicate**: that was noise in a 12-story sample, which is why the validation protocol was fixed in advance. v2 faithful keeps more meaning than
v1 (0.91 vs 0.87) and cannot lose a number or name, and fools somewhat fewer detectors (5 vs 8 of 34 on all three), but the intervals overlap
heavily, so v1 versus v2 is not settled either. Across every variant and sample the system ends up fooling all three detectors on roughly
15-25% of short stories, which is the fidelity-versus-evasion limit described next, not a gap that a prompt, steering or strength setting closed.

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

## 4. Final held-out evaluation

Run once on 2026-10-30 (`hd loop run --split test --unlock-test ...`, then `hd eval final --unlock-test`; every unlock is in
`results/test_access_log.txt`). Settings were fixed on the validation split first: cutoffs frozen in
`results/detector_thresholds.json`, humanizer `qwen3.8-27b`, `full` strategy, prompt v2, repair on, faithful strength, our detector
steering and fakespot judging. Nothing was retuned after seeing test data. Tables: `results/final/`.

**4a. The detectors on unseen prompts** (`results/final/detector_test_metrics.csv`; cutoff = the frozen 5%-false-positive point):

| Genre | Detector | AUC | AI caught | Human flagged |
|---|---|---|---|---|
| story (39 AI / 39 human) | ours | 0.995 (0.983-1.000) | 97% (87-100%) | 2.6% (0.4-13%) |
| story | desklib | 1.000 | 100% (91-100%) | 2.6% (0.4-13%) |
| essay (264 / 264) | ours | 0.998 | 98.5% | 4.2% |
| essay | desklib | 0.996 | 97.0% | 3.4% |
| casual (150 / 150) | ours | 0.999 | 99.3% | 3.3% |
| casual | desklib | 1.000 | 100% | 3.3% |

Both detectors keep their promised false-positive rate on unseen prompts (2.6-4.2% against the 5% target) and catch 97-100% of
unedited AI text. The story test set is small (39 pairs), so its intervals are wide. The fakespot judge was scored on rewrites only,
not on this table.

**4b. Does the humanizer work?** 26 test stories of at most 400 words (every test story that short; 39 test stories in all),
faithful strength, policy "kept" (a rewrite that fails a guardrail means the user keeps the flagged original):

| | Flagged before | Flagged after | Fraction of flagged stories fixed |
|---|---|---|---|
| Our detector (steers) | 100% (87-100%) | 12% (4-29%) | 88% |
| desklib (never steers) | 100% (87-100%) | **88% (71-96%)** | 12% |
| fakespot (judge, never steers) | 88% (71-96%) | **65% (46-81%)** | 30% |

All 26 rewrites passed the guardrails; mean meaning similarity of the kept rewrite was 0.90.

Headline, the share of stories where the user's final text passes the guardrails AND the detectors do not flag it:

| Must fool | Kept-rewrite policy | Unguarded (ignores guardrails) |
|---|---|---|
| ours + desklib | 3 of 26: 12% (4-29%) | 6 of 26: 23% (11-42%) |
| ours + fakespot | 8 of 26: 31% (17-50%) | 12 of 26: 46% (29-65%) |
| **all three** | **2 of 26: 8% (2-24%)** | 5 of 26: 19% (9-38%) |

**What this says, plainly.** The humanizer reliably beats the detector that steers it (88% of its flags removed) and mostly fails
against independent detectors: desklib still flags 88% of the rewrites. Only 2 of 26 stories (8%) end up fooling all three
detectors with the meaning intact. That is lower than the 15% found on validation (5 of 34), but the intervals overlap heavily
(2-24% against 6-30%), so the two runs agree to within noise. The test result also confirms the fidelity/evasion limit from section 2:
dropping the guardrails (the "unguarded" column) raises the all-three rate from 8% to 19%, which is consistent with
the limit: some of the rewrites that evade the detectors are ones the guardrails reject for changing the story too much.

**4c. What was NOT run on the test split.** The plan fixed more comparisons than the test set got, so none of the following has a
test-split number:

- **Strong mode** (pre-registered). Dropped on the test split for lack of time and quota; on validation it showed no gain
  (4 of 24 against 4 of 24 for faithful on the same stories), so it is reported from validation only (section 2).
- **Desklib steering** and the **prompt-v1 baseline**: validation only (section 2).
- The **old reading-level delta rule**: the human-range rule is the only one applied to the test run.
- **Essay and casual humanizing**: not run at any split.
- **The manual commercial-detector check** (`reports/manual_detector_log.md`) is a separate small hand test; its results are
  filled in below once done.

**4d. Caveats specific to this run.** One humanizer model, one prompt, short stories (at most 400 words; the longer test stories were
excluded because the humanizer model's output cap makes them infeasible), n = 26 so each story moves a rate by about 4 points. The
run was resumed several times across rate limits, which is why `results/test_access_log.txt` has several loop entries; the items and
settings were identical each time (a larger run extends the same deterministic sample). Fakespot was used as a judge on both
validation and test, but its cutoff was set on validation only.


## Reproduce

```bash
uv run hd data build && uv run hd features build
uv run hd detector train-baseline && uv run hd detector train-sentence && uv run hd detector eval-external
uv run hd loop run --genre story --n 10 --max-words 400 --strategy full --model qwen/qwen3.8-27b
uv run hd eval report --log data/processed/loop_val_story_full_qwen3.8-27b_le400w.jsonl --features
uv run hd eval ablate-prompts --genre story --model qwen/qwen3.8-27b --max-words 400
uv run hd eval ablate-features
```
