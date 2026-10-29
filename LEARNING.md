# LEARNING.md: glossary and milestone notes

A running glossary. Terms are added as we meet them.

## Glossary

- **Adversarial loop:** two systems pushing against each other. Here, a humanizer tries to make AI
  text look human while a detector tries to catch it. Each round, both can learn from the other.
- **Goodhart's law (here: overfitting to the steering detector):** when a measure becomes the target, it stops
  being a good measure. If the humanizer is steered by our detector's feedback, beating *our* detector says little;
  the independent detector, never used to steer, is the honest check.
- **Confidence interval (Wilson):** a range of plausible true rates given a small sample. 6 of 10 texts flagged is a
  60% *measured* rate, but the true rate could plausibly be anywhere from 31% to 83%. We report intervals so a small pilot
  is not mistaken for a result.
- **Ablation:** removing or isolating one part of a system to see what it contributes. "Drop" asks what is lost without it;
  "only" asks what it can do alone. They differ when parts are redundant.
- **Seed:** a number that fixes the "random" choices a program makes (shuffling data, initial
  weights) so rerunning gives the same results. Ours is 42 in `configs/default.yaml`.
- **Held-out test set:** data the model never sees while training or tuning. It is the only honest
  measure of how the detector does on new text, so we lock it and touch it once at the end.
- **Overfitting:** a model that memorizes quirks of its training data and then does worse on new
  data. In this project it also applies to the humanizer: if it only learns to fool *our* detector,
  that is not real success. Hence the independent detector.
- **Corpus:** a collection of texts we study. Ours has a human side and an AI side.
- **Label:** the answer we want the detector to predict. Here 0 = human, 1 = AI.
- **Genre:** the kind of writing (story, essay, casual). Detectors often work well on one genre and
  poorly on another, so we keep them separate and report each.
- **Tokenization:** splitting text into pieces (words/punctuation) for a model. Some datasets store
  the pieces re-joined with spaces (`do n't`, `word ,`). That is a storage artifact, not writing style.
- **Data leakage / shortcut learning:** when a model scores well by exploiting something that
  isn't the real signal (like tokenizer spacing, or text length), and then fails on new data.
- **Confound:** a difference between the two classes that isn't the thing you want to study
  (e.g. AI answers being twice as long). It can fake a result.
- **Train / validation / test split:** train = the model learns from it; validation = we tune choices on
  it; test = untouched until the final score. Splitting **by prompt** (not by text) stops a human/AI
  pair for the same prompt from landing on both sides.
- **Hash-based split:** deciding the split from a fingerprint of the prompt's id. Same prompt, same
  split, forever, even as data grows.
- **Cache (LLM):** saving each API response on disk keyed by its inputs, so rerunning is free.
- **Feature:** a number computed from a text that describes some property of it (e.g. average
  sentence length). A classifier learns from features instead of reading the raw words.
- **Burstiness:** how much sentence length (or word surprise) swings up and down. People mix short
  and long sentences; LLM prose tends to be more even.
- **Perplexity / surprisal:** how unexpected a text is to a language model. Surprisal is the
  per-word version, -log(probability), measured in nats. LLM-written text is more predictable, so it
  scores lower.
- **GLTR-style features:** the share of words that were the model's #1 guess (`top1_frac`) or in its top
  10 (`top10_frac`).
- **Lexical diversity / MATTR:** how varied the vocabulary is. Plain type-token ratio shrinks as text
  gets longer, so we use a moving-window version (MATTR) that doesn't.
- **POS tag (part of speech):** noun, verb, adjective... POS *n-grams* are short runs of tags
  (`PRON AUX ADJ`) that capture grammar habits without depending on specific words.
- **ROC-AUC:** the chance that a randomly chosen AI text scores higher than a randomly chosen human text
  on some measure. 0.5 is a coin flip, 1.0 is perfect, 0.0 is perfectly backwards (human always higher).
- **Length-matched (stratified) AUC:** AUC computed inside bands of similar text length and then
  averaged, so a length difference between classes can't create the result.
- **Univariate:** looking at one feature at a time. Useful for explaining, but the classifier uses
  many together.
- **Logistic regression:** a linear classifier. It multiplies each (standardized) feature by a weight,
  adds them, and squashes the result into a 0-1 probability. The weights are the explanation: positive
  pushes toward "AI".
- **Gradient-boosted trees:** many small decision trees added one after another, each fixing the last
  one's mistakes. More flexible than linear (captures curves and interactions), harder to explain.
- **Regularization (C):** a penalty that keeps weights small so the model doesn't memorize noise. In
  scikit-learn's logistic regression, *smaller C = stronger penalty*.
- **Cross-validation (grouped):** split the training data into folds, train on some and score on the
  rest, rotate. "Grouped by prompt" keeps a human/AI pair together so the score isn't inflated.
- **Precision / recall:** of the texts we call AI, how many are (precision); of the real AI texts, how
  many we catch (recall). Here the positive class is AI.
- **Confusion matrix:** the 2x2 table of right and wrong calls (human called human, human called AI...).
- **Bootstrap confidence interval:** re-draw the evaluation set many times (here, whole prompts) with
  replacement to see how much a metric would wobble by chance.
- **Multicollinearity:** two features carrying nearly the same information. A linear model can give them
  large opposite weights, so individual coefficients stop meaning anything.
- **Permutation importance:** shuffle one feature and see how much the score drops. Misleading when
  other features carry the same information (they cover for it).
- **Weak supervision:** training with labels that are only approximately right. Our sentence model labels
  every sentence of an AI text "AI", though some are not. Fine for ranking, wrong for certifying.
- **Calibration:** whether a detector's probabilities mean what they say. A detector can rank perfectly
  (high AUC) yet call 26% of human texts "AI" at the 0.5 cutoff, because its scale is shifted.
- **False-positive rate (FPR) / matched-FPR comparison:** the share of human texts wrongly flagged. To
  compare two detectors fairly, set each cutoff so they flag the same share of humans, then compare how
  much AI text each catches.
- **Neutralization (ablation) attribution:** replace a feature's value by a typical one and see how much
  the score moves. Model-agnostic, and group-level versions are more reliable when features are correlated.
- **Independent / out-of-distribution evaluation:** scoring with a detector that never saw our data, to
  avoid mistaking "fooled my own detector" for "reads as human".
- **Sentence embedding:** a list of numbers (here 384) that represents a sentence's meaning, so similar
  meanings are close together. *Cosine similarity* measures how close two embeddings are (1 = same
  direction, ~0 = unrelated).
- **Meaning drift / dropped / invented content:** a rewrite whose overall meaning moved; original sentences
  that have no counterpart in the rewrite; rewrite sentences that match nothing in the original.
- **Readability (Flesch-Kincaid grade):** a formula from sentence length and syllables that estimates the
  US school grade needed to read a text. Used to show a rewrite did not become much harder or easier.
- **Prompt ablation:** running variants of a prompt that differ by one block (techniques, detector feedback)
  to measure what each block contributes.
- **Dry run:** running a command that shows what it would do (here, the exact prompt) without doing it.
- **Rate limit (RPM / TPM / RPD / TPD):** a provider's cap on requests or tokens per minute or per day. Free
  tiers are small, so the real constraint on this project is the *daily token budget*, not the request count.
- **Token:** the unit language models read and write, roughly three quarters of an English word. Limits and
  prices are measured in tokens; a 500-word story is about 650 tokens out.
- **Reasoning model:** a model that "thinks" in hidden tokens before answering (e.g. gpt-oss). Those tokens
  count against output and rate limits even though you never see them.
- **Provider-agnostic client:** code that talks to the LLM through one small interface, so swapping Anthropic
  for Groq changed one module and a config line, not the pipeline.
- **Virtual environment (venv):** an isolated folder of Python packages for one project, so
  projects don't clash. `uv sync` builds ours in `.venv/`.

---

## Milestone 0: scaffold

**3 takeaways**
1. Reproducibility is built in from the start: one config file, one seed, a locked environment.
2. Heavy dependencies (PyTorch, spaCy models) are optional extras, so we only download them when a
   milestone needs them, and we ask first.
3. Tests exist from day one, even if tiny, so later changes can't silently break the basics.

**1 experiment to try:** in a Python shell (`uv run python`), call `random.random()` three times,
then call `from hd.config import set_seed; set_seed(42)` and `random.random()` again. Repeat the
`set_seed(42)` call and you get the identical number. Now skip `set_seed` and see the number change.
That is the whole idea behind reproducible runs.

---

## Milestone 1: data

**3 takeaways**
1. **The first result was a bug, not a finding.** Before cleaning, 98.7% of human Reddit answers had a
   space before punctuation versus 0.1% of ChatGPT's, so a "detector" would have learned the tokenizer.
   Always look for how the two classes were *stored* before trusting how they *read*.
2. **Split by the thing that must not leak.** Grouping by prompt (via a hash) keeps each human/AI pair
   together and keeps the test set stable as data grows. A test also caught duplicate Reddit questions
   that would have produced colliding ids.
3. **Real confounds remain and must be reported, not hidden:** length (AI answers ~2x longer) and line
   breaks. Later milestones will exclude raw length and add a length-matched evaluation.

**1 experiment to try:** run `uv run python -I -c "from hd.data.clean import normalize_text as n; print(n('I do n\'t know , but she ca n\'t go .'))"`.
Then, in a notebook, load `data/raw/hc3_reddit_eli5.jsonl`, compare the share of human vs ChatGPT answers
that contain `" ,"`, and then ask: if you trained a classifier on the *raw* text, what would it learn?


---

## Milestone 2: features

**3 takeaways**
1. **A near-perfect result needs a hostile review.** GPT-2 features hit AUC ~0.99. Before believing it
   I checked that it survives within length bands (it does: 0.98-1.00), because short texts get higher
   perplexity just from having less context. It was real: ChatGPT text is very predictable to GPT-2.
2. **The tools you trust have limits too.** My length-banding helper turned out to under-correct with 8
   bands (a pure-length feature scored 0.60), so I raised it to 16 and documented the residual
   0.50-0.60 in the worst case. Test your fixes on a case where you know the answer.
3. **Some hypotheses fail, and that is a result.** Em-dashes and my stock-phrase list showed no signal
   on 2022-era ChatGPT. The strongest *interpretable* tell is sentence rhythm (burstiness), which is
   also something the humanizer can act on.

**1 experiment to try:** in `uv run python`, score two texts with the scorer from `hd.features.perplexity`:
a plain, textbook-style sentence and a quirky, unusual one. Compare `mean_surprisal`. Then rewrite the
plain sentence by swapping in rarer words and see how far the number moves. This is exactly the lever the
humanizer will pull, and the one an independent detector may not care about.


---

## Milestone 3: baseline detector

**3 takeaways**
1. **Interpretable features get you most of the way; a language model gets you the rest.** Stylometric
   features alone reach val AUC ~0.92 (accuracy 0.86); adding GPT-2 predictability reaches ~0.996. That gap
   matters for the humanizer: it can change rhythm and word choice, but GPT-2 predictability is much
   harder to move on purpose.
2. **Be skeptical of your own justifications.** I dropped `burstiness` as "redundant" with
   `sent_len_cv`, and AUC fell by ~0.02: a monotone *transform* gives a linear model a curve, so it
   wasn't redundant. I kept the drop for interpretability (the pair had +6/-6 coefficients) and said so.
3. **Importance numbers have blind spots.** Permutation importance showed tiny values for every GPT-2
   feature, because the four of them cover for each other when one is shuffled. Drop-a-whole-group
   ablations (M7) answer the real question.

**1 experiment to try:** open `results/coefficients_lr_style.csv` and `results/coefficients_lr_full.csv`.
Notice how the coefficients move once the GPT-2 features are available. Then edit `VARIANTS` in
`hd/detector/baseline.py` to add a variant with only the `rhythm` group and see how little AUC one
feature family buys on its own.


---

## Milestone 4: explanations and the independent detector

**3 takeaways**
1. **A great ranker can be a terrible classifier at its default threshold.** desklib's AUC is 0.996, yet at
   0.5 it flags 26% of human text. If I had reported "detection rate at 0.5" for rewritten text, the result
   would have mostly measured that miscalibration. Matched-false-positive-rate comparisons fix it.
2. **Verify that a borrowed model really loaded.** The custom model class could silently leave the
   classifier head randomly initialised and still return numbers. A test now checks the loading report,
   and that scores do not depend on batch padding.
3. **Explanations need a sign convention and honest limits.** My first output printed `perplexity -0.992`
   under a header that implied "toward AI"; the group in fact pushed toward *human*. And because
   GPT-2's four features cover for each other, group-level attributions beat per-feature ones.

**1 experiment to try:** run `uv run hd detect --external` on (a) a paragraph you wrote, (b) the same
paragraph pasted into a chatbot with "rewrite this more formally", and (c) a mix of both. Compare our
detector with desklib, and look at which feature group drives each score. Where do they disagree, and why?


---

## Milestone 5: the humanizer

**3 takeaways**
1. **Test a safety check with the failures it is meant to catch.** I simulated bad rewrites on real text and
   measured the guardrails. That showed whole-text similarity barely moves when 30% of the sentences are
   deleted (median 0.98), so my first "coverage" check was weak; and my typo threshold caught only 31% of
   typo-laden text until I tightened it. A guardrail you never tested is a hope, not a guardrail.
2. **Feedback is only useful if it is specific and grounded.** "Vary your sentences" is easy to ignore;
   "your variation is 0.39, human stories average 0.62, and 4 of 15 sentences open with 'However'" is
   something a rewriter can act on. The human profile (percentiles of human text per genre) is what makes
   the numbers meaningful.
3. **Say what is untested.** The humanizer has not met a real LLM yet. The README says so, and the guardrail
   thresholds are marked provisional until false-positive rates can be measured on genuine rewrites.

**1 experiment to try:** run `uv run hd humanize --file <an AI-written text of your own> --dry-run` and read
the prompt. Is any instruction something you would ignore or misread? Edit `TECHNIQUES` in
`hd/humanizer/prompts.py`, rerun, and see how the prompt changes. When you have a key, compare `--strategy
naive` against `--strategy full` on the same text.


---

## Provider change: Groq free tier instead of Anthropic

**3 takeaways**
1. **A small interface paid for itself.** Because every LLM call went through `complete(system, prompt, ...)`,
   adding Groq meant a new client class, a config switch and new tests; the generator, humanizer and
   guardrails did not change.
2. **On a free tier, plan around the scarcest limit.** A story costs ~900 tokens and a humanizer call ~2,800,
   so the daily *token* cap decides how many experiments are possible, not the per-minute request cap. The
   client therefore paces from response headers, stops cleanly at the daily limit, and every call is cached so
   a stopped run loses nothing.
3. **The generator is part of the experiment.** AI text from Llama-3.3-70B is a different thing from HC3's
   2022 ChatGPT. Results on stories and on essay/casual are not directly comparable, and the README says so.

**1 experiment to try:** run `uv run hd llm check` and compare your account's per-minute token limit with the
costs above. How many stories fit in a minute? In a day (check console.groq.com/settings/limits)? Then change
`generation_model` to `llama-3.1-8b-instant` and see how the limits differ.


---

## Milestone 6: the adversarial loop

**3 takeaways**
1. **The detector you steer with is not the one you can believe.** With only our detector's feedback, one rewrite
   left 20% of stories flagged by ours but 90% by the independent detector. The loop therefore never lets desklib
   influence a decision; it is only recorded. (Pilot, n=10: wide uncertainty.)
2. **A guardrail is a hypothesis too.** My reading-level rule rejected 8 of 10 real rewrites because I had assumed
   "big change in reading level = damage", while human stories really are written at a much lower grade than AI
   stories. Check a rule against the data it will judge, not only against synthetic failures; then say you changed it.
3. **Evasion and fidelity pull against each other.** The technique list fooled both detectors but 90% of those
   rewrites broke the guardrails; "sound human" alone kept the meaning and changed nothing. The useful number is
   detection rate *among rewrites that keep the meaning*.

**Also learned the hard way:** free tiers have several different caps (tokens per minute, tokens per day, and for
some models output tokens per minute where the *requested maximum* counts, not the actual output). The daily
token cap (200,000 per model) limits us to roughly 65 full-strategy rewrites a day, so experiment sizes follow the
quota. Every result file records which model wrote the rewrites, and models are never pooled.

**1 experiment to try:** run `uv run hd loop run --genre story --n 10 --max-words 400 --model qwen/qwen3.8-27b
--strategy full`, then open the log (`data/processed/loop_val_story_full_*.jsonl`) and read the last kept rewrite
of three stories next to their originals. Would you have guessed a human wrote them? What does your answer say about
what our detector and desklib each pay attention to?


---

## Milestone 7: evaluation tooling and the held-out set

**3 takeaways**
1. **The interval is part of the result.** With 10 texts, "60% flagged" means "somewhere between 31% and 83%". Every rate in
   the reports carries a Wilson interval, and FINDINGS.md says which claims survive that and which are only "worth testing".
2. **A held-out set is a budget, not a resource.** Each time you look at it and react, it leaks into your decisions. So the code
   opens it from one command, logs every access, refuses a second final run, and the report must disclose any repeats. Decisions
   (which settings, which guardrail rule) are written down before looking.
3. **Cues are specific to a generator.** Dropping GPT-2 predictability costs 0.02 AUC on 2022-ChatGPT essays but nothing on the newer
   model's stories, where grammar and punctuation carry the signal. A detector built on one generator's tells is not a detector of AI.

**Also learned:** two numbers can look alike but answer different questions. "Fooled the detector" counts the rewrite our detector
likes best; "what a user gets" only counts rewrites that kept the meaning. Reporting both, side by side, shows what fidelity costs.

**1 experiment to try:** before running `uv run hd eval ablate-features`, write down which feature group you expect to matter most
for stories and for casual answers. Then compare "drop" with "only" in `results/ablation_feature_groups.csv`. Where were you wrong,
and which groups are redundant with each other?
