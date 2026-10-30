from types import SimpleNamespace

import numpy as np
import pytest

from hd.detector.explain import Analysis
from hd.eval.loop_report import below_target
from hd.humanizer.guardrails import GuardrailReport
from hd.humanizer.surgery import contributions, mark, parse_replies, pick_targets, run_surgery, sentence_spans, splice
from hd.llm import LLMResponse, RateLimited

STORY = "A robotic one. A robotic two. A robotic three. A robotic four. Plain five."


class FakeDetector:
    """Our detector: flags any text that still contains the word 'robotic'."""

    def analyze(self, text):
        return Analysis(text, 0.9 if "robotic" in text else 0.1, len(text.split()), [], {}, [], {})


class FakeExternal:
    """Desklib stand-in: its logit is 3 per 'robotic' minus 2, so every 'robotic' sentence is a measurable contributor."""

    def logits(self, texts):
        return np.array([3.0 * t.count("robotic") - 2.0 for t in texts])

    def predict_proba(self, texts):
        return 1.0 / (1.0 + np.exp(-self.logits(texts)))


class FakeJudge:
    def predict_proba(self, texts):
        return [0.5 for _ in texts]


class FakeChecker:
    def __init__(self, bad=()):
        self.bad, self.compared_to = set(bad), []

    def check(self, original, rewrite):
        self.compared_to.append(original)
        flags = ["meaning drift"] if rewrite in self.bad else []
        return GuardrailReport(0.9, 0.0, 0.0, 1.0, 5.0, 5.0, 70.0, 70.0, 0.0, None, flags)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.prompts, self.kwargs = list(replies), [], []

    def complete(self, **kw):
        self.prompts.append(kw["prompt"])
        self.kwargs.append(kw)
        out = self.replies.pop(0)
        if isinstance(out, Exception):
            raise out
        return LLMResponse(out, 100, 20)


def go(replies, text=STORY, bad=(), **kw):
    client = FakeClient(replies)
    humanizer = SimpleNamespace(client=client, model="fake/m", effort=None, strength="faithful")
    checker = FakeChecker(bad)
    result = run_surgery(FakeDetector(), humanizer, checker, text, "story", our_cutoff=0.5, ext_cutoff=0.8, external=FakeExternal(),
                         item_id="x", **kw)  # fmt: skip
    return result, client, checker


# ---- the pieces ------------------------------------------------------------------------------------------------------

def test_spans_cover_each_sentence_exactly():
    text = "One two.\n\nThree four. Five!"
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == ["One two.", "Three four.", "Five!"]


def test_contributions_find_the_sentences_that_carry_the_signal():
    spans = sentence_spans(STORY)
    c = contributions(STORY, spans, FakeExternal().logits)
    assert c[:4] == pytest.approx([3.0] * 4) and c[4] == pytest.approx(0.0)  # removing 'Plain five.' changes nothing


def test_pick_targets_takes_the_biggest_eligible_and_never_fewer_than_the_minimum():
    contribs = [1.0, 5.0, 3.0, 4.0, 0.0, 2.0]
    assert pick_targets(contribs, [True] * 6, frac=0.3) == [1, 3]  # ceil(0.3 * 6) = 2
    assert pick_targets(contribs, [True, False, True, True, True, True], frac=0.3) == [2, 3]  # 1 is spent
    assert pick_targets(contribs, [True] + [False] * 5, frac=0.3) == [0]  # fewer eligible than the minimum
    assert pick_targets(contribs, [False] * 6) == []


def test_mark_and_splice_keep_everything_outside_the_targets():
    text = "One two.\n\nThree four. Five six."
    spans = sentence_spans(text)
    assert mark(text, spans, [1]) == "One two.\n\n[#2]Three four.[/#2] Five six."
    assert splice(text, spans, {0: "Uno.", 2: "Cinco seis siete."}) == "Uno.\n\nThree four. Cinco seis siete."
    assert splice(text, spans, {}) == text


def test_parse_replies_ignores_chatter_unknown_numbers_and_thinking():
    raw = "<think>hmm [#9] nope</think>\nSure!\n[#1] The cat sat.\n[#3] It said \"hi\"\nand left.\n[#9] not asked for\n"
    assert parse_replies(raw, [0, 2]) == {0: "The cat sat.", 2: 'It said "hi" and left.'}
    assert parse_replies("nothing useful", [0]) == {}


# ---- the loop --------------------------------------------------------------------------------------------------------

def test_success_when_both_steering_detectors_get_under_their_goals():
    result, client, checker = go(["[#1] One.\n[#2] Two."], text="A robotic one. A robotic two. Plain three.")
    assert result.stop_reason == "success" and result.final == 1
    assert result.steps[1].text == "One. Two. Plain three."
    assert checker.compared_to == ["A robotic one. A robotic two. Plain three."]  # always compared with the ORIGINAL
    assert result.strategy == "surgery" and result.steer == ["ours", "external"] and result.target == 0.30
    assert result.prompt_version == "surgery1" and result.judge_cutoff is None
    assert result.steps[1].judge is None


def test_only_the_contributing_sentences_are_marked_and_the_rest_is_untouched():
    result, client, _ = go(["[#1] One.\n[#2] Two."], text="A robotic one. A robotic two. Plain three.")
    prompt = client.prompts[0]
    assert "[#1]A robotic one.[/#1] [#2]A robotic two.[/#2] Plain three." in prompt
    assert "[#3]Plain three." not in prompt  # the unmarked sentence stays plain


def test_judge_is_scored_but_never_decides():
    client = FakeClient(["[#1] One.\n[#2] Two."])
    humanizer = SimpleNamespace(client=client, model="fake/m", effort=None, strength="faithful")
    result = run_surgery(FakeDetector(), humanizer, FakeChecker(), "A robotic one. A robotic two. Plain three.", "story", our_cutoff=0.5,
                         ext_cutoff=0.8, external=FakeExternal(), judge=FakeJudge(), judge_cutoff=0.4)  # fmt: skip
    assert [s.judge for s in result.steps] == [0.5, 0.5] and result.judge_cutoff == 0.4 and result.stop_reason == "success"


def test_target_lower_than_the_cutoff_keeps_the_loop_going():
    # desklib's frozen cutoff is 0.8 but the user asked for under 0.30: a score of 0.5 passes the cutoff and still is not enough
    class Middling(FakeExternal):
        def logits(self, texts):
            return np.array([3.0 * t.count("robotic") + 0.0 for t in texts])  # no 'robotic' -> logit 0 -> p = 0.5

    client = FakeClient(["[#1] One.\n[#2] Two."])
    humanizer = SimpleNamespace(client=client, model="fake/m", effort=None, strength="faithful")
    result = run_surgery(FakeDetector(), humanizer, FakeChecker(), "A robotic one. A robotic two. Plain three.", "story", our_cutoff=0.5,
                         ext_cutoff=0.8, external=Middling(), target=0.30, max_iterations=1)  # fmt: skip
    assert result.stop_reason == "max_iterations"  # 0.5 > 0.30
    loose = run_surgery(FakeDetector(), SimpleNamespace(client=FakeClient(["[#1] One.\n[#2] Two."]), model="m", effort=None),
                        FakeChecker(), "A robotic one. A robotic two. Plain three.", "story", our_cutoff=0.5, ext_cutoff=0.8,
                        external=Middling(), target=None, max_iterations=1)  # fmt: skip
    assert loose.stop_reason == "success"  # with only the frozen cutoffs 0.5 <= 0.8 is enough


def test_a_rejected_edit_is_dropped_and_the_next_pass_attacks_other_sentences():
    first = "[#1] One.\n[#2] Two."
    rejected = "One. Two. A robotic three. A robotic four. Plain five."
    result, client, _ = go([first, "[#3] Three.\n[#4] Four."], bad={rejected}, max_iterations=2)
    assert result.steps[1].guardrails_ok is False and result.steps[1].flags == ["meaning drift"]
    assert "[#3]A robotic three.[/#3] [#4]A robotic four.[/#4]" in client.prompts[1]
    assert "[#1]" not in client.prompts[1]  # the rejected targets are not attacked again
    # pass 2 starts from the ORIGINAL text, so the rejected rewrite did not leak into it
    assert result.steps[2].text == "A robotic one. A robotic two. Three. Four. Plain five."
    assert result.final == 2  # the only guardrail-passing step


def test_no_accepted_rewrite_means_no_final():
    result, _, _ = go(["[#1] One.\n[#2] Two."], text="A robotic one. A robotic two. Plain three.", bad={"One. Two. Plain three."}, max_iterations=1)
    assert result.final is None


def test_stops_cleanly_when_every_sentence_was_already_attacked():
    result, _, _ = go(["[#1] One.\n[#2] Two.", "[#3] Three.\n[#4] Four.", "[#5] Five."], max_iterations=4)
    assert result.stop_reason in ("success", "exhausted")
    assert len(result.steps) <= 4


def test_unusable_reply_ends_the_text_without_a_step():
    result, _, _ = go(["I cannot help with that."])
    assert result.stop_reason.startswith("error") and len(result.steps) == 1 and result.final is None


def test_reply_identical_to_the_text_is_a_stall():
    result, _, _ = go(["[#1] A robotic one.\n[#2] A robotic two."], text="A robotic one. A robotic two. Plain three.")
    assert result.stop_reason == "stalled" and len(result.steps) == 1


def test_rate_limits_propagate_so_the_batch_runner_can_stop():
    with pytest.raises(RateLimited):
        go([RateLimited("slow down", None)])


def test_output_budget_scales_with_the_number_of_targets_and_stays_small():
    _, client, _ = go(["[#1] One.\n[#2] Two."], text="A robotic one. A robotic two. Plain three.")
    assert 0 < client.kwargs[0]["max_tokens"] <= 900


# ---- the "under 30%" measure -----------------------------------------------------------------------------------------

def _rec(final, before, after):
    step = lambda i, ours, ext, judge: {"iteration": i, "ours": ours, "external": ext, "judge": judge}  # noqa: E731
    return {"final": final, "our_cutoff": 0.2, "ext_cutoff": 0.7, "judge_cutoff": 0.5, "steps": [step(0, *before), step(1, *after)]}


def test_below_target_counts_only_texts_under_the_flat_bar_on_every_detector():
    recs = [
        _rec(1, (0.9, 0.99, 0.9), (0.1, 0.2, 0.25)),  # all under 0.30
        _rec(1, (0.9, 0.99, 0.9), (0.1, 0.2, 0.30)),  # judge exactly at 0.30: not strictly under
        _rec(1, (0.9, 0.99, 0.9), (0.1, 0.5, 0.1)),  # desklib above the bar (though under its frozen cutoff of 0.7)
        _rec(None, (0.9, 0.99, 0.9), (0.0, 0.0, 0.0)),  # rewrite failed the guardrails: the user keeps the original
    ]
    out = below_target(recs, 0.30)
    assert (out["n"], out["under_target"]) == (4, 1) and out["rate"] == 0.25
    assert below_target(recs, 0.30, detectors=("ours", "desklib"))["under_target"] == 2
    assert below_target(recs, 0.30, policy="unguarded", detectors=("ours",))["under_target"] == 4
