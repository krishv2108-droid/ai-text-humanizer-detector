import numpy as np
import pandas as pd
import pytest

from hd.data.splits import TestSetLockedError, add_splits
from hd.detector import baseline
from hd.detector.baseline import (
    VARIANTS,
    evaluate,
    fit_variant,
    load_split_features,
    model_features,
    predict_proba,
    run_baseline,
)
from hd.eval.metrics import classification_metrics, grouped_bootstrap_ci
from hd.features.stylometric import feature_names

FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
TAGS = ["NOUN", "VERB", "ADJ", "PRON", "ADP", "DET", "AUX", "PUNCT"]


def synthetic(n_prompts=500, seed=0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A human/AI pair per prompt, with a planted signal in a few features and in POS sequences."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_prompts):
        genre = "essay" if i % 2 else "casual"
        for label in (0, 1):
            r = {f: rng.normal() for f in feature_names()}
            # planted: perplexity separates strongly (single-feature AUC ~0.98), rhythm clearly (~0.92)
            r["mean_surprisal"] += -3.0 * label
            r["sent_len_cv"] += -2.0 * label
            r["contraction_ratio"] = np.nan if rng.random() < 0.2 else r["contraction_ratio"]
            # AI texts lean on "NOUN ADP" runs, human texts on "PRON VERB" runs
            lean = ["NOUN", "ADP"] if label else ["PRON", "VERB"]
            seq = [rng.choice(lean) if rng.random() < 0.5 else rng.choice(TAGS) for _ in range(60)]
            rows.append({"id": f"p{i}-{label}", "prompt_id": f"p{i}", "genre": genre, "label": label,
                         "n_words": int(rng.normal(150 + 30 * label, 40)), "has_pair": True,
                         "pos_seq": " ".join(seq), **r})  # fmt: skip
    df = add_splits(pd.DataFrame(rows), 42, FRACTIONS)
    corpus = df[["id", "prompt_id", "genre", "label", "n_words", "has_pair", "split"]]
    feats = df[["id", "pos_seq", *feature_names()]]
    return corpus, feats


@pytest.fixture(scope="module")
def data():
    corpus, feats = synthetic()
    merged = corpus.merge(feats, on="id")
    return merged[merged["split"] == "train"], merged[merged["split"] == "val"]


def test_redundant_feature_not_modeled_but_still_extracted():
    assert "burstiness" in feature_names()
    for v in VARIANTS.values():
        assert "burstiness" not in model_features(v.groups)
        assert "paragraph_breaks_per100w" not in model_features(v.groups)  # format group is off


# Style variants cannot see the (strongest) perplexity signal, only the rhythm one, so they get a lower bar.
@pytest.mark.parametrize("name,floor", [("lr_style", 0.85), ("lr_style_pos", 0.85), ("lr_full", 0.95), ("gbm_full", 0.95)])
def test_variants_learn_the_planted_signal(name, floor, data):
    train, val = data
    model, info = fit_variant(name, train, seed=42, cv_folds=3)
    p = predict_proba(model, val)
    assert p.shape == (len(val),) and ((p >= 0) & (p <= 1)).all()
    assert classification_metrics(val["label"].to_numpy(), p)["roc_auc"] > floor
    assert info["cv_auc"] > floor


def test_pos_ngrams_need_a_linear_variant():
    with pytest.raises(ValueError):
        baseline.build_pipeline(baseline.Variant("x", "gbm", ("rhythm",), pos_ngrams=True), 0)


def test_evaluate_rows_are_consistent(data):
    train, val = data
    model, _ = fit_variant("lr_full", train, seed=42, cv_folds=3)
    rows = evaluate(model, val, n_boot=50)
    assert [r["genre"] for r in rows] == ["all", "casual", "essay"]
    allrow = rows[0]
    assert allrow["tn"] + allrow["fp"] + allrow["fn"] + allrow["tp"] == len(val) == allrow["n"]
    assert allrow["roc_auc_ci_lo"] <= allrow["roc_auc"] <= allrow["roc_auc_ci_hi"]
    assert 0 <= allrow["accuracy"] <= 1
    assert not np.isnan(allrow["auc_length_matched"])
    no_ci = evaluate(model, val, n_boot=0)
    assert np.isnan(no_ci[0]["roc_auc_ci_lo"])


def test_classification_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.4, 0.9])  # one false positive, one false negative at 0.5
    m = classification_metrics(y, p)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == m["precision"] == m["recall"] == 0.5
    assert m["roc_auc"] == pytest.approx(0.75)


def test_bootstrap_resamples_whole_prompts():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(100), 2)
    y = np.tile([0, 1], 100)
    p = np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.3, 200)
    from sklearn.metrics import roc_auc_score

    lo, hi = grouped_bootstrap_ci(y, p, groups, roc_auc_score, n_boot=200)
    assert lo < roc_auc_score(y, p) < hi and hi - lo < 0.2


# ---- the test split must stay locked ------------------------------------------------------


def _cfg(tmp_path, corpus, feats):
    processed = tmp_path / "processed"
    processed.mkdir()
    corpus.to_parquet(processed / "corpus.parquet")
    feats.to_parquet(processed / "features.parquet")
    return {"seed": 42, "paths": {"processed": str(processed), "results": str(tmp_path / "results")}}


def test_loading_the_test_split_requires_unlocking(tmp_path):
    cfg = _cfg(tmp_path, *synthetic(60))
    assert len(load_split_features(cfg, "val")) > 0
    with pytest.raises(TestSetLockedError):
        load_split_features(cfg, "test")


def test_run_baseline_never_requests_the_test_split(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, *synthetic(300))
    (tmp_path / "results").mkdir()
    requested = []
    real = baseline.load_split
    monkeypatch.setattr(baseline, "load_split",
                        lambda corpus, name, **kw: (requested.append((name, kw.get("unlock_test", False))), real(corpus, name, **kw))[1])  # fmt: skip
    table = run_baseline(cfg, variants=["lr_style", "gbm_full"])
    assert {name for name, _ in requested} == {"train", "val"}
    assert not any(unlocked for _, unlocked in requested)
    assert set(table["split"]) == {"train", "val"}
    results = tmp_path / "results"
    for f in ("baseline_metrics.csv", "coefficients_lr_style.csv", "importance_gbm_full.csv"):
        assert (results / f).exists(), f
    assert (results / "figures" / "baseline_roc.png").exists()
    assert (tmp_path / "models" / "lr_style.joblib").exists()
