"""Evaluation metrics shared by the detectors, the loop, and the final report."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def stratified_auc(x: np.ndarray, y: np.ndarray, n_words: np.ndarray, bins: int = 16, min_class: int = 10) -> float:
    """ROC-AUC computed *within* text-length bands, then averaged (weighted by human*AI pair count).

    Human and AI texts differ in length in our data, and many features (like language-model
    perplexity) also drift with length. Comparing only texts of similar length means length can't be
    what drives the number. Bins where either class has < `min_class` texts are skipped.

    This reduces the length confound but does not eliminate it: texts inside one band still differ a
    little in length by label. In a worst-case simulation (a feature that *is* length, 2-SD length gap)
    it returns 0.50-0.60 instead of the raw 0.85+. Treat small values near 0.5 as "no signal", not as
    exact zeros, and prefer features whose stratified AUC is far from 0.5.
    """
    edges = np.unique(np.quantile(n_words, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, n_words, side="right") - 1, 0, len(edges) - 2)
    num = den = 0.0
    for b in range(len(edges) - 1):
        m = which == b
        n0, n1 = int((y[m] == 0).sum()), int((y[m] == 1).sum())
        if n0 < min_class or n1 < min_class:
            continue
        num += n0 * n1 * roc_auc_score(y[m], x[m])
        den += n0 * n1
    return num / den if den else np.nan


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Positive class = AI. `p` is the predicted probability that a text is AI-written."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) == 2 else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }  # fmt: skip


def grouped_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% interval for `metric`, resampling whole prompts (a human/AI pair moves together)."""
    rng = np.random.default_rng(seed)
    ids, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(ids))]
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[i] for i in pick])
        if len(set(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
