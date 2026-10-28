import json
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from test_detector import synthetic

from hd.detector import explain
from hd.detector.baseline import VARIANTS, fit_variant, model_features
from hd.detector.explain import MODELS_DIR, Detector
from hd.detector.sentence import SENTENCE_FEATURES, fit_sentence_model, sentence_rows
from hd.eval.external_eval import detection_summary
from hd.eval.metrics import threshold_at_fpr
from hd.features.perplexity import perplexity_features
from hd.features.stylometric import feature_names
from hd.features.text import analyze


def fake_stats(text: str, surprisal_of_sentence=lambda k: float(k)):
    """One fake 'token' per word, with surprisal = f(sentence index), so sentence means are known."""
    doc = analyze(text)
    spans = [s for s in doc.sents if s.text.strip()]
    offsets, surprisal = [], []
    for m in re.finditer(r"\S+", text):
        k = next(i for i, s in enumerate(spans) if s.start_char <= m.start() < s.end_char + 1)
        offsets.append((m.start(), m.end()))
        surprisal.append(surprisal_of_sentence(k))
    n = len(offsets)
    return doc, {"surprisal": np.array(surprisal), "rank": np.full(n, 3), "offsets": np.array(offsets)}


# ---- sentence features -------------------------------------------------------------------


def test_sentence_rows_align_tokens_to_the_right_sentence():
    text = "The cat sat down. However, the dog barked loudly at the mailman. It's late."
    doc, stats = fake_stats(text)
    rows = sentence_rows(text, doc, stats)
    assert [r["s_mean_surprisal"] for r in rows] == [0.0, 1.0, 2.0]  # each sentence got its own tokens
    assert [r["s_starts_transition"] for r in rows] == [0.0, 1.0, 0.0]
    assert [r["s_has_contraction"] for r in rows] == [0.0, 0.0, 1.0]
    assert rows[1]["s_len_rel"] > 1 > rows[0]["s_len_rel"]  # relative to the text's average length
    assert text[rows[1]["start"] : rows[1]["end"]].startswith("However")


def test_sentences_past_the_scored_window_get_nan():
    text = "First sentence here. Second sentence there."
    doc, stats = fake_stats(text)
    keep = stats["offsets"][:, 1] <= len("First sentence here.")  # pretend GPT-2 stopped after sentence 1
    truncated = {k: v[keep] for k, v in stats.items()}
    rows = sentence_rows(text, doc, truncated)
    assert not np.isnan(rows[0]["s_mean_surprisal"]) and np.isnan(rows[1]["s_mean_surprisal"])


def test_sentence_model_learns_a_planted_signal():
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({f: rng.normal(size=n) for f in SENTENCE_FEATURES})
    df["label"] = rng.integers(0, 2, n)
    df["prompt_id"] = rng.integers(0, 300, n).astype(str)
    df["s_mean_surprisal"] -= 1.5 * df["label"]  # AI sentences are more predictable
    model, info = fit_sentence_model(df)
    assert info["cv_auc"] > 0.8
    coef = dict(zip(SENTENCE_FEATURES, model.named_steps["clf"].coef_[0]))
    assert coef["s_mean_surprisal"] < -0.5 and abs(coef["s_commas"]) < 0.3


# ---- thresholds at a fixed false-positive rate -------------------------------------------


def test_threshold_at_fpr_bounds_the_false_positive_rate():
    human = np.linspace(0, 1, 1000)
    thr = threshold_at_fpr(human, 0.05)
    assert (human > thr).mean() <= 0.05 + 1e-9 and (human > thr).mean() > 0.04


def test_detection_summary_uses_the_calibrated_threshold():
    df = pd.DataFrame({"label": [0] * 100 + [1] * 100, "s": list(np.linspace(0.5, 1, 100)) + [0.99] * 100})
    thr = threshold_at_fpr(df.loc[df.label == 0, "s"].to_numpy(), 0.05)
    out = detection_summary(df, "s", thr)
    assert out["human_flagged_at_0.5"] == 1.0  # a miscalibrated detector flags every human at 0.5...
    assert out["human_flagged_at_fpr5"] <= 0.05 + 1e-9  # ...but not at the calibrated cutoff
    assert out["ai_detected_at_fpr5"] == 1.0


# ---- Detector.analyze (stubbed featurization, so no model downloads) ---------------------


class StubScorer:
    def token_stats(self, texts):
        return [fake_stats(t)[1] for t in texts]


@pytest.fixture(scope="module")
def planted_detector():
    corpus, feats = synthetic(400)
    train = corpus.merge(feats, on="id")
    train = train[train["split"] == "train"]
    doc_model, _ = fit_variant("gbm_full", train, seed=42, cv_folds=3)
    rng = np.random.default_rng(1)
    sent = pd.DataFrame(rng.normal(size=(300, len(SENTENCE_FEATURES))), columns=SENTENCE_FEATURES)
    sent_model = Pipeline([("impute", SimpleImputer()), ("clf", LogisticRegression())]).fit(sent, rng.integers(0, 2, 300))
    reference = train[feature_names()].median()
    return Detector(doc_model, "gbm_full", reference, sent_model, StubScorer()), train


def test_analyze_attributes_the_planted_group(planted_detector, monkeypatch):
    det, train = planted_detector
    text = "The cat sat down. However, the dog barked loudly at the mailman. It's late."
    ai_row = train[train["label"] == 1][feature_names()].median().to_dict()  # AI-typical feature values
    doc, stats = fake_stats(text)
    monkeypatch.setattr(det, "_featurize", lambda t: (doc, stats, ai_row))

    a = det.analyze(text)
    assert a.prob_ai > 0.9
    groups = list(a.group_effects)
    assert groups[0] in {"perplexity", "rhythm"}  # the two planted signals
    assert a.group_effects["perplexity"] > 0.1
    assert list(a.group_effects.values()) == sorted(a.group_effects.values(), reverse=True)
    assert set(a.group_effects) == set(VARIANTS["gbm_full"].groups)
    assert all(set(f) == {"feature", "value", "typical", "effect"} for f in a.feature_effects)
    assert len(a.sentences) == 3 and all(0 <= s.prob_ai <= 1 for s in a.sentences)
    assert "stock transition opener" in a.sentences[1].reasons
    assert a.top_sentences(1)[0].prob_ai == max(s.prob_ai for s in a.sentences)


def test_analyze_rejects_tiny_texts(planted_detector):
    det, _ = planted_detector
    with pytest.raises(ValueError, match="too short"):
        det.analyze("Hi there.")


def test_detector_refuses_pos_ngram_variants():
    with pytest.raises(ValueError):
        Detector(None, "lr_full_pos", pd.Series(dtype=float), None, None)


# ---- real models (skipped when not downloaded/trained) -----------------------------------


@pytest.fixture(scope="module")
def gpt2():
    pytest.importorskip("transformers")
    from hd.features.perplexity import GPT2Scorer

    try:
        return GPT2Scorer(batch_size=4, device="cpu")
    except OSError:
        pytest.skip("GPT-2 weights not downloaded")


def test_gpt2_offsets_map_tokens_back_to_the_text(gpt2):
    text = "The quick brown fox jumps over the lazy dog. It was a sunny day."
    stats = gpt2.token_stats([text])[0]
    off = stats["offsets"]
    assert len(off) == len(stats["surprisal"]) == len(stats["rank"])
    assert (off[:, 0] <= off[:, 1]).all() and off[-1, 1] == len(text) and (np.diff(off[:, 0]) >= 0).all()
    assert "".join(text[a:b] for a, b in off) == text
    assert set(perplexity_features(stats)) == {"mean_surprisal", "surprisal_std", "top1_frac", "top10_frac"}


@pytest.fixture(scope="module")
def desklib():
    pytest.importorskip("transformers")
    from hd.detector.external import DESKLIB, DesklibDetector, DesklibAIDetectionModel

    try:
        model, info = DesklibAIDetectionModel.from_pretrained(DESKLIB, output_loading_info=True)
    except OSError:
        pytest.skip("desklib weights not downloaded")
    del model
    return DesklibDetector(device="cpu"), info


def test_desklib_head_loaded_and_scores_are_probabilities(desklib):
    det, info = desklib
    assert not any("classifier" in k for k in info["missing_keys"]), "classifier head was randomly initialised"
    p = det.predict_proba(["A short plain text about nothing in particular.", "Another one, a bit longer than the first one is."])
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()


def test_desklib_score_does_not_depend_on_batch_padding(desklib):
    det, _ = desklib
    short = "The cat sat."
    long_ = "The quick brown fox jumps over the lazy dog while the sun slowly sets behind the hills and the wind drops."
    alone = det.predict_proba([short])[0]
    batched = det.predict_proba([long_, short])[1]
    assert alone == pytest.approx(batched, abs=1e-3)


@pytest.mark.skipif(not (MODELS_DIR / "gbm_full.joblib").exists(), reason="run `hd detector train-baseline` first")
def test_detect_cli_json_end_to_end(capsys):
    from hd.cli import main

    text = "I went to the shop on Saturday. It was quiet, and nobody was there. I bought bread and left. Then I walked home."
    assert main(["detect", text, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert 0 <= out["prob_ai"] <= 1 and out["n_words"] > 10 and len(out["sentences"]) == 4
    assert "perplexity" in out["group_effects"]
