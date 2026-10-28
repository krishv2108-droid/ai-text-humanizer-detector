# LEARNING.md: glossary and milestone notes

A running glossary. Terms are added as we meet them.

## Glossary

- **Adversarial loop:** two systems pushing against each other. Here, a humanizer tries to make AI
  text look human while a detector tries to catch it. Each round, both can learn from the other.
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
