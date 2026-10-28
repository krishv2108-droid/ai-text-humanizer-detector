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
