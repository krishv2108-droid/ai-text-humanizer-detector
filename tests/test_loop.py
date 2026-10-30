import json

import pandas as pd
import pytest

from hd.data.splits import TestSetLockedError
from hd.detector.explain import Analysis
from hd.humanizer.guardrails import GuardrailReport
from hd.humanizer.prompts import build_prompt
from hd.humanizer.rewrite import RewriteResult
from hd.llm import DailyLimitReached
from hd.loop import Step, choose_final, cutoff_for, iteration_curve, load_results, run_batch, run_loop, select_items, summarize

CUT = 0.30  # our detector's frozen cutoff in these tests; "flagged" means strictly above


class FakeDetector:
    def __init__(self, probs: dict[str, float]):
        self.probs = probs

    def analyze(self, text: str) -> Analysis:
        return Analysis(text, self.probs[text], len(text.split()), [], {}, [], {})


class FakeHumanizer:
    """Returns scripted rewrites in order (an Exception in the script is raised) and records every call."""

    model = "fake/model-1"

    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []

    def humanize(self, draft, genre, strategy, analysis=None, original=None, retry_note=None):
        self.calls.append({"draft": draft, "original": original, "retry_note": retry_note, "analysis": analysis})
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return RewriteResult(out, strategy, "", "", 10, 20)


class FakeChecker:
    """`bad` is a set of rewrites that fail, or a dict {rewrite: extra report fields} to give the failure details."""

    def __init__(self, bad=()):
        self.bad, self.compared_to = bad, []

    def check(self, original, rewrite):
        self.compared_to.append(original)
        if rewrite not in self.bad:
            return GuardrailReport(0.9, 0.0, 0.0, 1.0, 5.0, 5.0, 70.0, 70.0, 0.0, None, [])
        details = self.bad[rewrite] if isinstance(self.bad, dict) else {}
        return GuardrailReport(0.9, 0.0, 0.0, 1.0, 5.0, 5.0, 70.0, 70.0, 0.0, None, ["meaning drifted from the original"], **details)


class FakeExternal:
    def __init__(self, probs: dict[str, float]):
        self.probs = probs

    def predict_proba(self, texts):
        return [self.probs[t] for t in texts]


def go(probs, outputs, bad=(), external=None, max_iterations=4, repair=True):
    humanizer, checker = FakeHumanizer(outputs), FakeChecker(bad)
    result = run_loop(FakeDetector(probs), humanizer, checker, "orig", "story", strategy="full", our_cutoff=CUT,
                      ext_cutoff=0.9 if external else None, external=external, max_iterations=max_iterations, item_id="x",
                      repair=repair)  # fmt: skip
    return result, humanizer, checker


def test_stops_as_soon_as_our_detector_passes():
    result, humanizer, _ = go({"orig": 0.95, "r1": 0.60, "r2": 0.20, "r3": 0.10}, ["r1", "r2", "r3"])
    assert [s.iteration for s in result.steps] == [0, 1, 2]
    assert result.stop_reason == "success" and result.final == 2
    assert len(humanizer.calls) == 2  # never asked for the third rewrite


def test_every_draft_gets_one_rewrite_even_if_already_passing():
    result, humanizer, _ = go({"orig": 0.05, "r1": 0.04}, ["r1"])
    assert len(humanizer.calls) == 1 and result.final == 1 and result.stop_reason == "success"


def test_flagged_means_strictly_above_the_cutoff():
    result, _, _ = go({"orig": 0.9, "r1": CUT}, ["r1"])  # exactly at the cutoff is not flagged
    assert result.stop_reason == "success"


def test_runs_out_of_iterations_and_keeps_the_best_accepted_rewrite():
    result, _, _ = go({"orig": 0.9, "r1": 0.7, "r2": 0.5, "r3": 0.6}, ["r1", "r2", "r3"], max_iterations=3)
    assert result.stop_reason == "max_iterations" and len(result.steps) == 4
    assert result.final == 2  # lowest score, not the last one


def test_without_repair_a_failed_rewrite_is_discarded_and_the_retry_restarts_from_the_last_accepted_draft():
    result, humanizer, checker = go({"orig": 0.9, "r1": 0.05, "r2": 0.80, "r3": 0.20}, ["r1", "r2", "r3"], bad={"r1", "r2"}, repair=False)
    # r1 fooled the detector but broke the guardrails, so it can't be kept or end the loop
    assert [s.guardrails_ok for s in result.steps[1:]] == [False, False, True]
    assert result.final == 3 and result.stop_reason == "success" and result.repair is False
    assert humanizer.calls[1]["retry_note"].startswith("Your previous attempt was rejected")
    assert "meaning drifted" in humanizer.calls[1]["retry_note"]
    assert all(c["draft"] == "orig" for c in humanizer.calls)  # restarted from the original every time
    assert set(checker.compared_to) == {"orig"}  # always judged against the original, never the previous rewrite


def test_repair_mode_fixes_the_rejected_rewrite_and_names_exactly_what_is_wrong():
    details = {"r1": {"missing_sentences": ["Marta stared at the cup."], "invented_sentences": ["A dragon appeared."],
                      "missing_numbers": [78], "missing_names": ["spokane"], "novel_words": ["glorbish"]}}  # fmt: skip
    result, humanizer, checker = go({"orig": 0.9, "r1": 0.05, "r2": 0.10}, ["r1", "r2"], bad=details)
    assert [s.guardrails_ok for s in result.steps[1:]] == [False, True] and result.final == 2 and result.stop_reason == "success"
    assert humanizer.calls[1]["draft"] == "r1" and humanizer.calls[1]["original"] == "orig"  # the rejected rewrite is the draft now
    assert humanizer.calls[1]["analysis"].text == "r1"  # feedback is about the draft being repaired
    note = humanizer.calls[1]["retry_note"]
    for expected in ("Marta stared at the cup.", "A dragon appeared.", "78", "spokane", "glorbish", "keep its new voice"):
        assert expected in note
    assert set(checker.compared_to) == {"orig"}  # still judged against the original, so repairs cannot drift
    assert result.steps[2].retry_note == note and result.repair is True


def test_repair_failure_is_never_kept_even_if_it_fooled_the_detector():
    result, _, _ = go({"orig": 0.9, "r1": 0.0, "r2": 0.0}, ["r1", "r2"], bad={"r1", "r2"}, max_iterations=2)
    assert result.final is None  # two rejected repairs: nothing is kept, however well they scored


def test_repair_note_only_mentions_what_failed():
    from hd.loop import repair_note

    g = GuardrailReport(0.7, 0.3, 0.0, 1.6, 5.0, 2.0, 70.0, 80.0, 0.0, None,
                        ["length changed too much (1.60x)", "reading level 2.0 outside the human range 3.0 to 11.0"], missing_sentences=["a", "b", "c", "d", "e", "f"])  # fmt: skip
    note = repair_note(g)
    assert "(and 2 more)" in note and "1.60x" in note and "reading level" in note.lower()
    assert "numbers" not in note and "names" not in note and "typos" not in note
    assert "Stay closer" in repair_note(GuardrailReport(0.7, 0, 0, 1.0, 5.0, 5.0, 70.0, 70.0, 0.0))  # generic fallback


def test_results_record_the_prompt_version_and_repair_setting():
    from hd.humanizer.prompts import PROMPT_VERSION

    result, _, _ = go({"orig": 0.9, "r1": 0.1}, ["r1"])
    assert result.to_dict()["prompt_version"] == PROMPT_VERSION == "v2" and result.to_dict()["repair"] is True


def test_next_attempt_builds_on_the_latest_accepted_rewrite():
    _, humanizer, _ = go({"orig": 0.9, "r1": 0.6, "r2": 0.1}, ["r1", "r2"])
    assert humanizer.calls[1]["draft"] == "r1" and humanizer.calls[1]["original"] == "orig"
    assert humanizer.calls[1]["analysis"].text == "r1"  # feedback comes from the draft being rewritten
    assert humanizer.calls[1]["retry_note"] is None


def test_no_acceptable_rewrite_means_no_final():
    result, _, _ = go({"orig": 0.9, "r1": 0.05, "r2": 0.04}, ["r1", "r2"], bad={"r1", "r2"}, max_iterations=2)
    assert result.final is None and result.stop_reason == "max_iterations"


def test_repeated_output_stops_the_loop_instead_of_wasting_calls():
    result, humanizer, _ = go({"orig": 0.9, "r1": 0.6}, ["r1", "r1", "r1"])
    assert result.stop_reason == "stalled" and len(result.steps) == 2 and len(humanizer.calls) == 2


def test_model_errors_end_that_text_but_rate_limits_propagate():
    result, _, _ = go({"orig": 0.9, "r1": 0.6}, ["r1", RuntimeError("the model returned an empty rewrite")])
    assert result.stop_reason.startswith("error:") and result.final == 1
    with pytest.raises(DailyLimitReached):
        go({"orig": 0.9}, [DailyLimitReached("daily tokens used up")])


def test_independent_detector_is_recorded_but_never_decides_anything():
    # external still flags every step, yet the loop stops and keeps the rewrite our detector prefers
    ext = FakeExternal({"orig": 0.99, "r1": 0.98, "r2": 0.97, "r3": 0.99})
    result, _, _ = go({"orig": 0.9, "r1": 0.25, "r2": 0.20, "r3": 0.1}, ["r1", "r2", "r3"], external=ext)
    assert result.stop_reason == "success" and result.final == 1 and len(result.steps) == 2
    assert [s.external for s in result.steps] == [0.99, 0.98]
    no_ext, _, _ = go({"orig": 0.9, "r1": 0.25}, ["r1"])
    assert no_ext.steps[0].external is None


def test_choose_final_ties_go_to_the_earlier_iteration():
    steps = [Step(0, "o", 0.9, None), Step(1, "a", 0.2, None, True), Step(2, "b", 0.2, None, True), Step(3, "c", 0.1, None, False)]
    assert choose_final(steps) == 1


def test_cutoff_for_requires_the_exact_genre():
    table = {"ours": {"casual": 0.02, "all": 0.04}}
    assert cutoff_for(table, "ours", "casual") == 0.02
    with pytest.raises(RuntimeError, match="story"):
        cutoff_for(table, "ours", "story")  # no silent fallback to "all"
    with pytest.raises(RuntimeError, match="external"):
        cutoff_for(table, "external", "casual")


def test_retry_note_reaches_the_prompt_only_when_given():
    _, plain = build_prompt("the draft", "story", "craft")
    _, retried = build_prompt("the draft", "story", "craft", retry_note="It dropped a scene.")
    assert "NOTE" not in plain and "NOTE: It dropped a scene." in retried
    assert retried.rstrip().endswith('"""')  # the draft is still last


# ---- batches, sampling, summaries -------------------------------------------------------------

def make_corpus():
    rows = []
    for i in range(40):
        for label in (0, 1):
            rows.append({"id": f"{'ai' if label else 'h'}{i}", "genre": "story", "label": label, "has_pair": i % 10 != 0,
                         "split": "val" if i < 30 else "test", "text": f"t{i}", "n_words": 5})  # fmt: skip
    rows.append({"id": "casual-ai", "genre": "casual", "label": 1, "has_pair": True, "split": "val", "text": "c", "n_words": 5})
    return pd.DataFrame(rows)


def test_select_items_is_deterministic_extendable_and_ai_only():
    corpus = make_corpus()
    five = select_items(corpus, "val", "story", 5, seed=1)
    ten = select_items(corpus, "val", "story", 10, seed=1)
    assert list(ten["id"][:5]) == list(five["id"])  # a bigger n extends the same sample
    assert list(select_items(corpus, "val", "story", 5, seed=2)["id"]) != list(five["id"])
    assert (ten["label"] == 1).all() and (ten["genre"] == "story").all() and ten["has_pair"].all()
    assert not ten["id"].isin(["ai0", "ai10", "ai20"]).any()  # no partner, so excluded


def test_select_items_cannot_touch_the_locked_test_split():
    with pytest.raises(TestSetLockedError):
        select_items(make_corpus(), "test", "story", 5, seed=1)


def result_for(item_id, probs, outputs, max_iterations=None):
    item_probs = {"orig": probs[0]} | {f"r{i}": p for i, p in enumerate(probs[1:], 1)}
    humanizer, checker = FakeHumanizer(outputs), FakeChecker()
    return run_loop(FakeDetector(item_probs), humanizer, checker, "orig", "story", our_cutoff=CUT, item_id=item_id,
                    max_iterations=max_iterations or len(probs) - 1)  # fmt: skip


def test_run_batch_writes_resumes_and_stops_cleanly_at_the_daily_limit(tmp_path):
    out = tmp_path / "log.jsonl"
    items = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    def first_run(item):
        if item["id"] == "c":
            raise DailyLimitReached("tokens per day exhausted", 3600)
        return result_for(item["id"], [0.9, 0.1], ["r1"])

    stats = run_batch(items, out, first_run)
    assert stats["written"] == 2 and stats["stopped_by_limit"]
    assert "60 min" in stats["message"]
    again = run_batch(items, out, lambda item: result_for(item["id"], [0.9, 0.1], ["r1"]))
    assert again["skipped_done"] == 2 and again["written"] == 1
    assert [r["id"] for r in load_results(out)] == ["a", "b", "c"]
    assert all(json.loads(line)["steps"][0]["text"] == "orig" for line in open(out))


def test_summary_counts_an_unaccepted_text_as_unchanged():
    ok = result_for("a", [0.9, 0.1], ["r1"])  # flagged -> fixed
    kept_flagged = result_for("b", [0.9, 0.8, 0.7, 0.6, 0.5], ["r1", "r2", "r3", "r4"])  # still flagged after 4
    ok.ext_cutoff = kept_flagged.ext_cutoff = 0.5
    for r, ext in ((ok, [0.9, 0.2]), (kept_flagged, [0.9, 0.9, 0.9, 0.9, 0.9])):
        for s, e in zip(r.steps, ext):
            s.external = e
    unaccepted = result_for("c", [0.9, 0.1], ["r1"])
    unaccepted.steps[1].guardrails_ok, unaccepted.final = False, None
    unaccepted.ext_cutoff, unaccepted.steps[0].external, unaccepted.steps[1].external = 0.5, 0.9, 0.1
    row = summarize([r.to_dict() for r in (ok, kept_flagged, unaccepted)]).iloc[0]
    assert row["n"] == 3 and row["ours_flagged_before"] == 1.0
    assert row["ours_flagged_after"] == pytest.approx(2 / 3)  # b still flagged; c unchanged => still flagged
    assert row["external_flagged_after"] == pytest.approx(2 / 3)  # c's low external score is not credited
    assert row["rewrite_accepted"] == pytest.approx(2 / 3)


def test_iteration_curve_carries_finished_texts_forward():
    a = result_for("a", [0.9, 0.1], ["r1"]).to_dict()  # stops after 1 rewrite
    b = result_for("b", [0.9, 0.8, 0.7], ["r1", "r2"]).to_dict()
    curve = iteration_curve([a, b])
    assert list(curve["rewrites"]) == [0, 1, 2]
    assert curve["mean_ours"].round(3).tolist() == [0.9, 0.45, 0.4]


def test_results_record_the_humanizer_model_and_summaries_keep_models_apart():
    a = result_for("a", [0.9, 0.1], ["r1"]).to_dict()
    b = result_for("b", [0.9, 0.1], ["r1"]).to_dict()
    assert a["humanizer_model"] == "fake/model-1"
    b["humanizer_model"] = "other/model-2"
    table = summarize([a, b])
    assert sorted(table["model"]) == ["fake/model-1", "other/model-2"] and (table["n"] == 1).all()


def test_select_items_can_be_limited_to_short_texts():
    corpus = make_corpus()
    corpus.loc[corpus["id"].isin(["ai1", "ai2", "ai3"]), "n_words"] = 900
    short = select_items(corpus, "val", "story", 100, seed=1, max_words=400)
    assert (short["n_words"] <= 400).all() and not short["id"].isin(["ai1", "ai2", "ai3"]).any()
    assert len(select_items(corpus, "val", "story", 100, seed=1)) == len(short) + 3


# ---- the judge (third detector) and optional desklib steering ----------------------------------------------------


def go3(probs, outputs, ext, judge=None, steer=False, max_iterations=4):
    humanizer, checker = FakeHumanizer(outputs), FakeChecker()
    result = run_loop(FakeDetector(probs), humanizer, checker, "orig", "story", strategy="full", our_cutoff=CUT, ext_cutoff=0.7,
                      external=FakeExternal(ext), max_iterations=max_iterations, item_id="x", judge=FakeExternal(judge) if judge else None,
                      judge_cutoff=0.5 if judge else None, steer_external=steer)  # fmt: skip
    return result, humanizer


EXT = {"orig": 0.99, "r1": 0.95, "r2": 0.40, "r3": 0.10}


def test_the_judge_is_recorded_at_every_step_and_never_decides_anything():
    probs = {"orig": 0.9, "r1": 0.1, "r2": 0.05}
    low = {"orig": 0.9, "r1": 0.1, "r2": 0.1}
    high = {"orig": 0.1, "r1": 0.99, "r2": 0.99}  # opposite stories about the judge
    a, _ = go3(probs, ["r1", "r2"], EXT, judge=low)
    b, _ = go3(probs, ["r1", "r2"], EXT, judge=high)
    assert [s.judge for s in a.steps] == [0.9, 0.1] and [s.judge for s in b.steps] == [0.1, 0.99]
    assert (a.final, a.stop_reason, len(a.steps)) == (b.final, b.stop_reason, len(b.steps))  # decisions identical whatever the judge says
    assert a.judge_cutoff == 0.5 and a.steer == ["ours"]
    none, _ = go3(probs, ["r1", "r2"], EXT)
    assert all(s.judge is None for s in none.steps) and none.judge_cutoff is None


def test_steering_stops_at_the_first_rewrite_that_satisfies_both_and_pushes_when_desklib_still_flags():
    from hd.loop import STEER_NOTE

    ext = {"orig": 0.99, "r1": 0.95, "r2": 0.30}
    result, humanizer = go3({"orig": 0.9, "r1": 0.1, "r2": 0.1}, ["r1", "r2"], ext, steer=True)
    assert result.stop_reason == "success" and result.final == 2 and [s.iteration for s in result.steps] == [0, 1, 2]
    assert humanizer.calls[1]["retry_note"] == STEER_NOTE  # r1 was faithful and fooled ours, but desklib still flagged it
    assert humanizer.calls[1]["draft"] == "r1"  # push further FROM the accepted rewrite
    assert result.steer == ["ours", "external"]
    plain, plain_h = go3({"orig": 0.9, "r1": 0.1, "r2": 0.1}, ["r1", "r2"], ext, steer=False)
    assert plain.final == 1 and plain.stop_reason == "success" and len(plain_h.calls) == 1  # without steering r1 would have ended it


def test_steering_keeps_the_rewrite_with_the_lowest_worst_case_score():
    ext = {"orig": 0.99, "r1": 0.95, "r2": 0.90, "r3": 0.85}  # desklib never relents
    probs = {"orig": 0.9, "r1": 0.05, "r2": 0.25, "r3": 0.28}
    result, _ = go3(probs, ["r1", "r2", "r3"], ext, steer=True, max_iterations=3)
    assert result.stop_reason == "max_iterations"
    ratios = [max(p / CUT, e / 0.7) for p, e in ((0.05, 0.95), (0.25, 0.90), (0.28, 0.85))]
    assert result.final == 1 + ratios.index(min(ratios))  # chosen by the worse of the two detectors, relative to each cutoff
    assert result.final == 3 and go3(probs, ["r1", "r2", "r3"], ext, steer=False, max_iterations=3)[0].final == 1  # ours alone prefers r1


def test_steering_needs_the_independent_detector_and_its_cutoff():
    with pytest.raises(ValueError, match="independent detector"):
        run_loop(FakeDetector({"orig": 0.9}), FakeHumanizer(["r1"]), FakeChecker(), "orig", "story", our_cutoff=CUT, steer_external=True)
    steps = [Step(0, "o", 0.9, 0.9), Step(1, "a", 0.1, None, True)]
    with pytest.raises(ValueError, match="both cutoffs"):
        choose_final(steps, CUT, 0.7, steer_external=True)
    assert choose_final(steps) == 1  # without steering a missing desklib score is irrelevant


def test_loop_records_the_strength_and_logs_of_different_strengths_never_share_a_file():
    from pathlib import Path as _P

    from hd.loop import log_path

    class Strong(FakeHumanizer):
        strength = "strong"

    result = run_loop(FakeDetector({"orig": 0.9, "r1": 0.1}), Strong(["r1"]), FakeChecker(), "orig", "story", our_cutoff=CUT)
    assert result.strength == "strong" and result.to_dict()["strength"] == "strong"
    assert go({"orig": 0.9, "r1": 0.1}, ["r1"])[0].strength == "faithful"  # the default, and what old logs mean
    base = log_path(_P("d"), "val", "story", "full", "q/m", 400)
    assert log_path(_P("d"), "val", "story", "full", "q/m", 400, "", "faithful") == base  # existing filenames are unchanged
    assert log_path(_P("d"), "val", "story", "full", "q/m", 400, "", "strong") != base and "_strong" in log_path(_P("d"), "val", "story", "full", "q/m", 400, "", "strong").name
