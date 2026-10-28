import math

import numpy as np
import pandas as pd
import pytest

from hd.features import pipeline
from hd.features.pipeline import compute_features
from hd.features.stylometric import (
    FEATURE_GROUPS,
    STYLOMETRIC_GROUPS,
    extract_features,
    feature_names,
    mattr,
    pos_sequence,
)
from hd.features.text import analyze, sentences


def feats(text: str) -> dict[str, float]:
    return extract_features(analyze(text), text)


# ---- structure ---------------------------------------------------------------------------


def test_feature_keys_match_declared_groups():
    f = feats("She walked home. It was late, and she was tired.")
    assert list(f) and set(f) == set(feature_names(STYLOMETRIC_GROUPS))
    assert len(feature_names()) == len(set(feature_names()))  # no name used twice


def test_no_raw_length_features():
    banned = {"n_words", "n_chars", "n_sentences", "n_tokens", "length"}
    assert banned.isdisjoint(feature_names())


def test_pos_proportions_sum_to_one():
    f = feats("The old dog barked loudly at the mailman, and he ran away.")
    assert sum(f[k] for k in FEATURE_GROUPS["pos"]) == pytest.approx(1.0)


# ---- rhythm / burstiness -----------------------------------------------------------------


def test_rhythm_known_values():
    f = feats("One two three. Four five six seven eight nine.")  # sentence lengths 3 and 6
    assert f["sent_len_mean"] == pytest.approx(4.5)
    assert f["sent_len_std"] == pytest.approx(1.5)
    assert f["sent_len_cv"] == pytest.approx(1 / 3)
    assert f["burstiness"] == pytest.approx(-0.5)


def test_uniform_sentences_are_less_bursty_than_varied():
    uniform = feats("I like red cars. You like old bikes. We like big dogs. They like fast boats.")
    varied = feats("Stop. I never expected the long grey winter to settle over the harbour town so "
                   "completely that year. No. Why? Because the ships stayed in port all season.")  # fmt: skip
    assert uniform["sent_len_cv"] == pytest.approx(0.0)
    assert varied["sent_len_cv"] > 0.5 and varied["burstiness"] > uniform["burstiness"]
    assert uniform["sent_len_adjdiff"] == pytest.approx(0.0)
    assert varied["sent_len_adjdiff"] > 0.5  # consecutive sentences jump in length


def test_line_breaks_split_sentences():
    doc = analyze("He ran\nShe hid\nThey left")
    assert len(sentences(doc)) == 3


# ---- lexical diversity -------------------------------------------------------------------


def test_mattr_extremes_and_length_invariance():
    assert mattr(["a"] * 60) == pytest.approx(1 / 25)
    assert mattr([f"w{i}" for i in range(60)]) == pytest.approx(1.0)
    pattern = [f"w{i % 10}" for i in range(100)]
    assert mattr(pattern) == pytest.approx(mattr(pattern * 4))  # same style, 4x longer -> same score


# ---- discourse, hedging, stock phrases ---------------------------------------------------


def test_transitions():
    f = feats("However, it works. The sun rose.")  # 6 words, one sentence-initial transition
    assert f["sent_initial_transition_frac"] == pytest.approx(0.5)
    assert f["transition_per100w"] == pytest.approx(100 / 6)
    assert feats("The weather was cold. Nobody went out.")["transition_per100w"] == 0


def test_hedges_and_stock_phrases():
    assert feats("It may rain, and perhaps it might snow.")["hedge_per100w"] > 0
    assert feats("The sun rose over the town.")["hedge_per100w"] == 0
    assert feats("She took a deep breath, a testament to her resolve.")["stock_phrase_per100w"] > 0
    # word boundaries: "resolved" must not match the phrase "resolve"
    assert feats("He resolved the issue quickly today.")["stock_phrase_per100w"] == 0


# ---- contractions, voice, punctuation ----------------------------------------------------


def test_contraction_ratio():
    assert feats("I don't think it's fine. She can't go.")["contraction_ratio"] == pytest.approx(1.0)
    assert feats("I do not think it is fine. She cannot go.")["contraction_ratio"] == pytest.approx(0.0)
    assert math.isnan(feats("The river runs south.")["contraction_ratio"])


def test_possessive_s_is_not_a_contraction():
    assert feats("The dog's bone was buried in the garden.")["contraction_per100w"] == 0


def test_voice():
    f = feats("I told you my plan, and you said your piece.")
    assert f["first_person_per100w"] > 0 and f["second_person_per100w"] > 0


def test_punctuation_rates():
    text = "Wait—what? Yes! No... Maybe (just maybe); \"fine\": done, finally, ok."
    f = feats(text)
    for key in ["emdash", "question", "exclaim", "ellipsis", "paren", "semicolon", "colon", "dquote", "comma"]:
        assert f[f"{key}_per100w"] > 0, key
    assert feats("Plain words only here.")["emdash_per100w"] == 0


def test_pos_sequence():
    assert pos_sequence(analyze("She did not go.")) == "PRON AUX PART VERB PUNCT"


# ---- pipeline: caching and perplexity plumbing -------------------------------------------


class FakeScorer:
    def __init__(self):
        self.calls = 0

    def features(self, texts):
        self.calls += len(texts)
        return [{"mean_surprisal": 3.0, "surprisal_std": 1.0, "top1_frac": 0.2, "top10_frac": 0.6}] * len(texts)


def _df(n):
    return pd.DataFrame({"id": [f"id{i}" for i in range(n)], "text": [f"Sentence number {i} is here. Another one follows." for i in range(n)]})


def test_pipeline_columns_and_caching(tmp_path, monkeypatch):
    cache, scorer = tmp_path / "features.parquet", FakeScorer()
    first = compute_features(_df(3), cache, scorer)
    assert set(feature_names()) <= set(first.columns) and "pos_seq" in first
    assert (first["mean_surprisal"] == 3.0).all() and scorer.calls == 3

    calls = []
    real = pipeline.stylometric_table
    monkeypatch.setattr(pipeline, "stylometric_table", lambda d, *a, **k: (calls.append(len(d)), real(d, *a, **k))[1])
    second = compute_features(_df(5), cache, scorer)  # 2 new rows only
    assert calls == [2] and scorer.calls == 5 and len(second) == 5
    assert list(second["id"]) == [f"id{i}" for i in range(5)]


# ---- real GPT-2 (skipped if the weights are not available locally) -----------------------


@pytest.fixture(scope="module")
def gpt2():
    pytest.importorskip("transformers")
    from hd.features.perplexity import GPT2Scorer

    try:
        return GPT2Scorer(batch_size=4, device="cpu")
    except OSError:
        pytest.skip("GPT-2 weights not downloaded")


def test_gpt2_predictable_text_is_less_surprising(gpt2):
    plain, scrambled = gpt2.features(["The capital of France is Paris, and it is a large city.",
                                      "Purple seventeen quickly of ostrich the under because lamp."])  # fmt: skip
    assert plain["mean_surprisal"] < scrambled["mean_surprisal"]
    assert 0 <= plain["top10_frac"] <= 1 and plain["top1_frac"] <= plain["top10_frac"]


def test_gpt2_score_does_not_depend_on_batch_padding(gpt2):
    short, long_ = "The cat sat.", "The quick brown fox jumps over the lazy dog while the sun slowly sets behind the hills."
    alone = gpt2.features([short])[0]
    batched = gpt2.features([long_, short])[1]  # `short` is padded when batched with `long_`
    for k in alone:
        assert alone[k] == pytest.approx(batched[k], abs=1e-3), k
    assert np.isfinite(list(alone.values())).all()


# ---- length-matched AUC ------------------------------------------------------------------


def test_stratified_auc_removes_a_pure_length_effect():
    from sklearn.metrics import roc_auc_score

    from hd.eval.feature_analysis import stratified_auc

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 1000)
    # AI texts are longer on average, but within any length band the labels are indistinguishable
    n_words = np.where(y == 1, rng.normal(180, 40, y.size), rng.normal(90, 40, y.size)).clip(20)
    length_only = n_words + rng.normal(0, 1, y.size)  # a "feature" that is just length
    assert roc_auc_score(y, length_only) > 0.85  # raw AUC is fooled
    # Banding shrinks the length effect but cannot remove it entirely: texts inside one band still
    # differ a little in length by label (0.50-0.60 across seeds in this worst case, a 2-SD length gap).
    assert stratified_auc(length_only, y, n_words) < 0.65

    real_signal = y + rng.normal(0, 0.3, y.size)  # genuinely separates labels at any length
    assert stratified_auc(real_signal, y, n_words) > 0.95
