import numpy as np

from hd.eval.judge_joint import judge_cutoff, judge_joint


class FakeJudge:
    """P(AI) = 0.9 if the text contains 'robot', else 0.1."""

    def predict_proba(self, texts):
        return np.array([0.9 if "robot" in t else 0.1 for t in texts])


def _rec(raw, delivered, final, flagged_by):
    """A loop record whose delivered step is flagged by the detectors named in `flagged_by` (frozen cutoffs 0.5)."""
    def step(i, text, flagged):
        hi, lo = 0.9, 0.1
        return {"iteration": i, "text": text, "ours": hi if "ours" in flagged else lo, "external": hi if "desklib" in flagged else lo,
                "judge": hi if "fakespot" in flagged else lo}
    return {"final": final, "our_cutoff": 0.5, "ext_cutoff": 0.5, "judge_cutoff": 0.5,
            "steps": [step(0, raw, {"ours", "desklib", "fakespot"}), step(1, delivered, flagged_by)]}


def test_cutoff_is_the_five_percent_false_positive_point_on_human_texts():
    humans = ["plain"] * 19 + ["robot"]  # one of twenty human texts looks AI: the 95th percentile is the lone high score
    assert judge_cutoff(FakeJudge(), humans, 0.05) > 0.1


def test_all_detectors_clear_needs_a_kept_rewrite_and_every_detector_quiet():
    recs = [
        _rec("a robot", "a person", 1, set()),  # kept, nobody flags, judge quiet -> counts
        _rec("a robot", "a robot again", 1, set()),  # loop detectors quiet but the judge flags the rewrite
        _rec("a robot", "a person", 1, {"desklib"}),  # desklib flags
        _rec("a robot", "a person", None, set()),  # no rewrite kept: the user keeps the flagged original
    ]
    out = judge_joint(recs, FakeJudge(), cutoff=0.5)
    assert (out["n"], out["all_detectors_clear"], out["loop_detectors_all_clear"]) == (4, 1, 2)
    assert out["judge_flagged_raw"] == 4 and out["judge_flagged_delivered"] == 2  # the second rewrite and the unchanged fourth original
    assert out["judge_caught_then_evaded"] == 2  # records 1 and 3: the judge caught the raw text and not the delivered one
