import re

import pandas as pd
import pytest

from hd import app_core
from hd.app_core import (Resources, cutoffs_for, estimated_tokens, group_effects_frame, humanize_text, input_problem, input_warning,
                         outcome_text, sentence_html, steps_frame, top_drivers, verdict)  # fmt: skip
from hd.detector.explain import Analysis, SentenceResult
from hd.humanizer.guardrails import GuardrailReport
from hd.humanizer.rewrite import RewriteResult

THRESHOLDS = {"ours": {"story": 0.2, "essay": 0.1, "all": 0.15}, "external": {"story": 0.7, "all": 0.75}}


def analysis(text="Hello there. Second one! Third.", probs=(0.9, 0.2, float("nan")), reasons=None) -> Analysis:
    spans, pos = [], 0
    for sentence, p in zip(("Hello there.", "Second one!", "Third."), probs):
        start = text.index(sentence, pos)
        spans.append(SentenceResult(sentence, start, start + len(sentence), p, reasons or []))
        pos = start + len(sentence)
    return Analysis(text, 0.8, 5, spans, {"rhythm": 0.3, "perplexity": 0.1, "pos": -0.05},
                    [{"feature": "a", "value": 1.0, "typical": 0.5, "effect": 0.2}, {"feature": "b", "value": 2.0, "typical": 2.0, "effect": -0.1}], {})  # fmt: skip


# ---- verdicts, cutoffs, input checks -----------------------------------------------------------------------------


def test_verdict_is_strictly_above_the_cutoff_and_handles_missing_values():
    assert verdict(0.31, 0.30)[0] == "flagged" and verdict(0.30, 0.30)[0] == "not flagged"
    assert "5% of real human text" in verdict(0.9, 0.3)[1]
    assert verdict(0.5, None)[0] == "n/a" and verdict(float("nan"), 0.3)[0] == "n/a"


def test_cutoffs_come_from_the_frozen_file_and_never_default():
    res = Resources({}, None, None, None, THRESHOLDS)
    assert cutoffs_for(res, "story") == {"ours": 0.2, "external": 0.7, "judge": None}  # no third detector calibrated yet
    with_third = Resources({}, None, None, None, THRESHOLDS | {"third": {"story": 0.9, "all": 0.9}})
    assert cutoffs_for(with_third, "story")["judge"] == 0.9
    with pytest.raises(RuntimeError, match="casual"):
        cutoffs_for(res, "casual")  # no silent 0.5 or borrowed genre
    assert cutoffs_for(Resources({}, None, None, None, {"ours": {"story": 0.2}}), "story")["external"] is None


def test_input_checks_and_cost_estimate():
    assert input_problem("one two three") and "at least 5" in input_problem("one two three")
    assert input_problem("one two three four five") is None
    assert input_warning("word " * 701) and input_warning("word " * 100) is None
    assert estimated_tokens("word " * 100, "full", 2) == int(100 * 5.1 * 2)
    assert estimated_tokens("word " * 100, "naive", 1) < estimated_tokens("word " * 100, "full", 1)  # richer prompts cost more


# ---- the HTML that reaches the browser must never carry the user's markup ---------------------------------------------


def test_sentence_html_escapes_user_text_and_reasons():
    nasty = "<script>alert(1)</script> Hello there. Second one! Third."
    a = analysis(text=nasty, reasons=['"><img src=x onerror=alert(2)>'])
    out = sentence_html(a)
    assert "&lt;script&gt;" in out  # the user's text arrives as visible text
    assert set(re.findall(r"<(/?\w+)", out)) <= {"span", "/span", "br"}  # and no tag but ours exists
    assert 'title="P(AI) 0.90: &quot;&gt;&lt;img src=x onerror=alert(2)&gt;"' in out  # the hover reason cannot break out of its attribute


def test_sentence_html_shades_by_probability_and_keeps_every_character():
    a = analysis(text="Hello there.\nSecond one! Third.", probs=(0.9, 0.1, float("nan")))
    out = sentence_html(a)
    alphas = [float(x) for x in re.findall(r"rgba\(\d+,\d+,\d+,([0-9.]+)\)", out)]
    assert alphas == [0.54, 0.06]  # 0.6 * P(AI); the unscored sentence has no tint
    assert "<br>" in out and "not scored" in out
    plain = re.sub(r"<[^>]+>", "", out).replace("&#x27;", "'")
    assert plain == "Hello there.Second one! Third."  # newline became <br>, no sentence text lost
    assert re.sub(r"<[^>]+>", "", sentence_html(analysis())) == "Hello there. Second one! Third."


def test_chart_frame_has_short_names_and_splits_by_direction():
    chart = app_core.group_chart_frame(analysis())
    assert list(chart.index) == ["Sentence rhythm", "Predictability", "Grammar mix"]
    assert chart.loc["Sentence rhythm"].tolist() == [0.3, 0.0] and chart.loc["Grammar mix"].tolist() == [0.0, -0.05]
    assert (chart["pushes toward AI"] >= 0).all() and (chart["pushes toward human"] <= 0).all()


def test_effect_frames_are_sorted_and_labelled():
    g = group_effects_frame(analysis())
    assert list(g["effect"]) == sorted(g["effect"], reverse=True) and g["group"][0].startswith("Sentence rhythm")
    d = top_drivers(analysis())
    assert list(d["feature"]) == ["a"]  # only cues pushing toward AI


# ---- humanize_text drives the same loop as the experiments ---------------------------------------------------------


class FakeDetector:
    scorer = None

    def analyze(self, text):
        return Analysis(text, {"orig": 0.9, "r1": 0.1}[text], 3, [], {}, [], {})


class FakeHumanizer:
    instances = []

    def __init__(self, client, model, profile=None, effort=None, strength="faithful"):
        self.model, self.effort, self.strength = model, effort, strength
        FakeHumanizer.instances.append(self)

    def humanize(self, draft, genre, strategy, analysis=None, original=None, retry_note=None):
        return RewriteResult("r1", strategy, "", "", 10, 20)


class FakeChecker:
    def check(self, original, rewrite):
        return GuardrailReport(0.9, 0.0, 0.0, 1.0, 5.0, 5.0, 70.0, 70.0, 0.0, None, [])


class FakeExternal:
    def predict_proba(self, texts):
        return [0.99 if t == "orig" else 0.4 for t in texts]


seen_strengths: list[str] = []


@pytest.fixture
def patched(monkeypatch):
    FakeHumanizer.instances.clear()
    monkeypatch.setattr(app_core, "Humanizer", FakeHumanizer)
    monkeypatch.setattr(app_core, "checker_for", lambda cfg, embedder, scorer, profile, genre, strength="faithful": seen_strengths.append(strength) or FakeChecker())
    seen_strengths.clear()


def test_humanize_text_uses_frozen_cutoffs_and_the_chosen_model(patched):
    res = Resources({"x": 1}, FakeDetector(), object(), object(), THRESHOLDS, FakeExternal())
    result = humanize_text(res, object(), "orig", "story", strategy="full", model="qwen/qwen3.8-27b", max_rewrites=3, use_external=True, effort="low")
    assert result.our_cutoff == 0.2 and result.ext_cutoff == 0.7 and result.final == 1 and result.stop_reason == "success"
    assert result.humanizer_model == "qwen/qwen3.8-27b" and FakeHumanizer.instances[0].effort == "low"
    assert [s.external for s in result.steps] == [0.99, 0.4]
    assert humanize_text(res, object(), "orig", "story", strategy="full", model="m", max_rewrites=1, use_external=False).steps[0].external is None


def test_humanize_text_refuses_bad_requests(patched):
    res = Resources({}, FakeDetector(), object(), object(), THRESHOLDS, None)
    with pytest.raises(ValueError, match="unknown strategy"):
        humanize_text(res, object(), "orig", "story", strategy="magic", model="m", max_rewrites=1, use_external=False)
    with pytest.raises(RuntimeError, match="not loaded"):
        humanize_text(res, object(), "orig", "story", strategy="full", model="m", max_rewrites=1, use_external=True)


def test_outcome_text_never_presents_a_failed_rewrite_as_a_success(patched):
    res = Resources({}, FakeDetector(), object(), object(), THRESHOLDS, FakeExternal())
    ok = humanize_text(res, object(), "orig", "story", strategy="naive", model="m", max_rewrites=1, use_external=True)
    assert "Kept rewrite 1" in outcome_text(ok) and "not flagged" in outcome_text(ok)
    frame = steps_frame(ok)
    assert list(frame["kept"]) == ["original", "kept"] and list(frame["guardrails"]) == ["", "ok"]
    ok.final, ok.steps[1].guardrails_ok, ok.steps[1].flags = None, False, ["meaning drift (similarity 0.50 < 0.8)"]
    assert "unchanged" in outcome_text(ok) and "success" not in outcome_text(ok).lower()
    assert "meaning drift" in steps_frame(ok)["why not"][1]


def test_humanize_text_can_let_desklib_steer_and_records_the_judge_without_letting_it_decide(patched):
    class Judge:
        def predict_proba(self, texts):
            return [0.99 if t == "orig" else 0.05 for t in texts]

    thresholds = THRESHOLDS | {"third": {"story": 0.5}}
    res = Resources({}, FakeDetector(), object(), object(), thresholds, FakeExternal(), Judge())
    steered = humanize_text(res, object(), "orig", "story", strategy="full", model="m", max_rewrites=2, use_external=True, steer_desklib=True, use_judge=True)
    assert steered.steer == ["ours", "external"] and steered.judge_cutoff == 0.5 and [s.judge for s in steered.steps] == [0.99, 0.05]
    plain = humanize_text(res, object(), "orig", "story", strategy="full", model="m", max_rewrites=2, use_external=True)
    assert plain.steer == ["ours"] and all(s.judge is None for s in plain.steps)  # off unless asked for
    assert steps_frame(steered)["fakespot P(AI)"].tolist() == [0.99, 0.05]
    assert steps_frame(plain)["fakespot P(AI)"].isna().all()  # the column exists but is empty when no judge ran


def test_humanize_text_refuses_steering_or_judging_without_the_detector_loaded(patched):
    res = Resources({}, FakeDetector(), object(), object(), THRESHOLDS | {"third": {"story": 0.5}}, None, None)
    with pytest.raises(RuntimeError, match="not loaded"):
        humanize_text(res, object(), "orig", "story", strategy="full", model="m", max_rewrites=1, use_external=False, steer_desklib=True)
    with pytest.raises(RuntimeError, match="third detector"):
        humanize_text(res, object(), "orig", "story", strategy="full", model="m", max_rewrites=1, use_external=False, use_judge=True)


def test_humanize_text_forwards_the_strength_to_both_the_prompt_and_the_guardrails(patched):
    res = Resources({}, FakeDetector(), object(), object(), THRESHOLDS, None, None)
    humanize_text(res, object(), "orig", "story", strategy="naive", model="m", max_rewrites=1, use_external=False, strength="strong")
    assert FakeHumanizer.instances[-1].strength == "strong" and seen_strengths[-1] == "strong"  # prompt and limits move together
    humanize_text(res, object(), "orig", "story", strategy="naive", model="m", max_rewrites=1, use_external=False)
    assert FakeHumanizer.instances[-1].strength == "faithful" and seen_strengths[-1] == "faithful"
