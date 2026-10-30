import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hd.eval import final as final_mod
from hd.detector.baseline import model_features
from hd.eval.ablation import group_ablation
from hd.eval.final import TestSetAlreadyUsed, detector_test_table, final_eval_runs, log_test_access, unlock_for_final
from hd.eval.loop_report import (joint_success, feature_shift, first_pass_table, human_spread, outcome_step, outcome_table, plot_before_after,
                                 plot_feature_shift, plot_iteration_curves, plot_strategy_ablation, to_markdown, wilson_ci,
                                 write_loop_report)  # fmt: skip
from hd.eval.manual_kit import build_manual_kit, write_log_template
from hd.humanizer.feedback import HumanProfile

# ---- records shaped like the loop's JSONL ------------------------------------------------------------------------


def step(i, text, ours, ext=None, ok=None, sim=None, fk=None, judge=None):
    d = {"iteration": i, "text": text, "ours": ours, "external": ext, "guardrails_ok": ok, "flags": [], "similarity": sim,
         "length_ratio": 1.0, "fk_grade": fk, "input_tokens": 100, "output_tokens": 50, "cached": False, "retry_note": ""}  # fmt: skip
    if judge is not None:
        d["judge"] = judge
    return d


def record(rid, steps, final, strategy="full", model="m/x", genre="story", our_cut=0.3, ext_cut=0.7, judge_cut=None, steer=None):
    r = {"id": rid, "genre": genre, "strategy": strategy, "our_cutoff": our_cut, "ext_cutoff": ext_cut, "steps": steps,
         "final": final, "stop_reason": "success", "humanizer_model": model}  # fmt: skip
    if judge_cut is not None:
        r["judge_cutoff"] = judge_cut
    if steer is not None:
        r["steer"] = steer
    return r


def sample_records():
    return [
        # fixed for both detectors, rewrite kept
        record("a", [step(0, "a0", 0.9, 0.95), step(1, "a1", 0.1, 0.2, True, 0.9, 4.0)], 1),
        # fixes ours only: desklib still flags
        record("b", [step(0, "b0", 0.9, 0.95), step(1, "b1", 0.1, 0.9, True, 0.85, 4.0)], 1),
        # rewrite fooled both but failed the guardrails -> unchanged under "kept", success under "unguarded"
        record("c", [step(0, "c0", 0.9, 0.95), step(1, "c1", 0.0, 0.1, False, 0.6, 2.0)], None),
        # never flagged by ours to begin with
        record("d", [step(0, "d0", 0.1, 0.95), step(1, "d1", 0.1, 0.9, True, 0.9, 5.0)], 1),
    ]


def test_wilson_interval_known_values_and_edges():
    lo, hi = wilson_ci(6, 10)
    assert lo == pytest.approx(0.3127, abs=1e-3) and hi == pytest.approx(0.8318, abs=1e-3)
    assert wilson_ci(0, 10)[0] == 0.0 and 0 < wilson_ci(0, 10)[1] < 0.35  # 0 successes still leaves real uncertainty
    assert wilson_ci(10, 10)[1] == 1.0 and wilson_ci(10, 10)[0] > 0.65
    assert math.isnan(wilson_ci(0, 0)[0])
    for k in range(0, 21):
        lo, hi = wilson_ci(k, 20)
        assert 0 <= lo <= k / 20 <= hi <= 1


def test_outcome_policies_do_not_credit_failed_rewrites():
    recs = sample_records()
    assert outcome_step(recs[2], "kept")["text"] == "c0"  # no accepted rewrite -> the original
    assert outcome_step(recs[2], "unguarded")["text"] == "c1"
    kept = outcome_table(recs, "kept").set_index("detector")
    assert kept.loc["ours", "flagged_before"] == 0.75 and kept.loc["ours", "flagged_after"] == 0.25  # only c still flagged
    assert kept.loc["desklib", "flagged_before"] == 1.0 and kept.loc["desklib", "flagged_after"] == 0.75  # b, c, d
    assert kept.loc["ours", "rewrite_accepted"] == 0.75
    assert kept.loc["ours", "evaded_of_detected"] == pytest.approx(2 / 3)  # of a, b, c (flagged before): a and b fixed
    unguarded = outcome_table(recs, "unguarded").set_index("detector")
    assert unguarded.loc["ours", "flagged_after"] == 0.0  # ignoring guardrails every rewrite "worked" for ours
    assert (kept[["before_lo", "after_lo"]] <= kept[["flagged_before", "flagged_after"]].to_numpy()).all().all()


def test_outcome_table_keeps_models_and_strategies_apart_and_skips_missing_detectors():
    recs = sample_records() + [record("e", [step(0, "e0", 0.9), step(1, "e1", 0.1, None, True, 0.9, 4.0)], 1, model="m/y")]
    for r in recs[-1:]:
        r["ext_cutoff"] = None
    table = outcome_table(recs)
    assert set(zip(table.model, table.detector)) == {("m/x", "ours"), ("m/x", "desklib"), ("m/y", "ours")}  # no desklib for m/y
    with pytest.raises(ValueError):
        outcome_step(recs[0], "optimistic")


def test_first_pass_table_compares_at_equal_budget():
    naive = [record("a", [step(0, "a0", 0.9, 0.9), step(1, "a1", 0.9, 0.9, True, 0.99, 8.0)], 1, "naive")]
    full = [record("a", [step(0, "a0", 0.9, 0.9), step(1, "a1", 0.1, 0.2, False, 0.7, 3.0), step(2, "a2", 0.1, 0.8, True, 0.9, 5.0)], 2, "full")]
    t = first_pass_table({"naive": naive, "full": full}).set_index("strategy")
    assert t.loc["naive", "ours_flagged"] == 1.0 and t.loc["full", "ours_flagged"] == 0.0  # the first rewrite, not the kept one
    assert t.loc["full", "guardrails_ok"] == 0.0 and t.loc["full", "mean_reading_grade"] == 3.0
    assert t.loc["full", "desklib_flagged"] == 0.0 and t.loc["naive", "mean_tokens_per_call"] == 150


def test_markdown_renders_percentages_and_missing_values():
    md = to_markdown(pd.DataFrame({"name": ["x"], "rate": [0.256], "other": [0.123456], "gap": [float("nan")]}), percent=("rate",))
    assert "| 26% |" in md and "| 0.12 |" in md and "| - |" in md and md.splitlines()[1].count("---") == 4


# ---- feature shift -----------------------------------------------------------------------------------------------


def profile_with(rows: dict[str, tuple[float, ...]]) -> HumanProfile:
    keys = ("p10", "p25", "p50", "p75", "p90", "p99")
    return HumanProfile(pd.DataFrame([{"genre": "story", "feature": f, "n": 100, **dict(zip(keys, v))} for f, v in rows.items()]))


def test_human_spread_survives_features_that_are_zero_for_most_humans():
    profile = profile_with({"emdash": (0, 0, 0, 0, 0, 4.6), "cv": (0.44, 0.53, 0.62, 0.72, 0.81, 1.07)})
    assert human_spread(profile, "story", "emdash") == pytest.approx(4.6 / 2.326)  # IQR is 0; the 99th percentile saves it
    assert human_spread(profile, "story", "cv") == pytest.approx(max(0.19 / 1.349, 0.37 / 2.563, 0.45 / 2.326))
    assert human_spread(profile_with({"never": (0, 0, 0, 0, 0, 0)}), "story", "never") == 1e-9  # degenerate: tiny, not zero


def test_feature_shift_uses_kept_rewrites_and_human_units():
    profile = profile_with({"cv": (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)})
    spread = human_spread(profile, "story", "cv")
    recs = [record("a", [step(0, "raw", 0.9), step(1, "new", 0.1, None, True)], 1), record("b", [step(0, "raw2", 0.9), step(1, "x", 0.1, None, False)], None)]
    values = {"raw": 0.3, "new": 0.6, "raw2": 99.0, "x": 99.0}
    shift = feature_shift(recs, lambda t: {"cv": values[t]}, profile, "story", ["cv"]).iloc[0]
    assert shift["n"] == 1  # record b had no accepted rewrite, so it is not part of "what users get"
    assert shift["raw_ai_gap"] == pytest.approx((0.3 - 0.6) / spread) and shift["rewrite_gap"] == pytest.approx(0.0)


# ---- figures and the report writer: smoke tests that the files exist and the plotted numbers are the table's ------


def test_report_writer_creates_tables_and_figures(tmp_path):
    recs = sample_records()
    profile = profile_with({"cv": (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)})
    shift = feature_shift(recs, lambda t: {"cv": 0.4}, profile, "story", ["cv"])
    written = write_loop_report(recs, tmp_path, "t", "4 texts", shift)
    for key in ("outcomes", "outcomes_md", "before_after", "iterations", "feature_shift"):
        assert written[key].exists() and written[key].stat().st_size > 0
    table = pd.read_csv(written["outcomes"])
    assert set(table["policy"]) == {"kept", "unguarded"}
    import re

    assert re.search(r"\d+% \(\d+%-\d+%\)", written["outcomes_md"].read_text())  # every rate is shown with its interval


def test_plots_run_with_a_single_group_and_a_single_detector(tmp_path):
    recs = [r for r in sample_records()]
    for r in recs:
        for s in r["steps"]:
            s["external"] = None
        r["ext_cutoff"] = None
    plot_before_after(outcome_table(recs), tmp_path / "a.png", "t")
    plot_iteration_curves(recs, tmp_path / "b.png", "t")
    plot_strategy_ablation(first_pass_table({"naive": recs, "full": recs}), tmp_path / "c.png", "t")
    plot_feature_shift(pd.DataFrame({"feature": ["cv", "mattr"], "raw_ai_gap": [-1.0, 0.5], "rewrite_gap": [0.1, 0.2]}), tmp_path / "d.png", "t")
    assert all((tmp_path / f"{n}.png").stat().st_size > 1000 for n in "abcd")


# ---- feature-group ablation --------------------------------------------------------------------------------------


def test_group_ablation_finds_the_planted_signal():
    rng = np.random.default_rng(0)
    n = 160
    y = np.tile([0, 1], n // 2)
    df = pd.DataFrame({"genre": "story", "label": y, "prompt_id": np.repeat(np.arange(n // 2), 2)})
    groups = ["rhythm", "lexical", "hedging", "perplexity"]
    for g in groups:  # every feature of every group is pure noise ...
        for col in model_features([g]):
            df[col] = rng.normal(size=n)
    df["sent_len_cv"] = rng.normal(size=n) + 3.0 * y  # ... except this one, in the 'rhythm' group
    table = group_ablation(df, groups=groups, folds=4, n_boot=0, seed=1)
    change = table.set_index("condition")["auc_change_vs_all"]
    auc = table.set_index("condition")["auc"]
    assert auc["all"] > 0.9 and auc["only:rhythm"] > 0.9 and auc["only:lexical"] < 0.7
    assert change["drop:rhythm"] < -0.2  # losing the informative group hurts a lot
    assert abs(change["drop:lexical"]) < 0.1 and abs(change["drop:hedging"]) < 0.1  # losing noise costs about nothing


# ---- the held-out test set: used once, logged, frozen cutoffs -----------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("hd.config.PROJECT_ROOT", tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "results").mkdir()
    rows = [{"id": f"{lab}{i}", "prompt_id": f"p{i}", "genre": "story", "label": lab, "split": split, "has_pair": True, "text": f"t{lab}{i}", "n_words": 5}
            for i in range(6) for lab, split in ((0, "test" if i < 3 else "val"), (1, "test" if i < 3 else "val"))]
    pd.DataFrame(rows).to_parquet(tmp_path / "data" / "processed" / "corpus.parquet")
    pd.DataFrame({"id": [r["id"] for r in rows], "sent_len_cv": 0.5}).to_parquet(tmp_path / "data" / "processed" / "features.parquet")
    return {"paths": {"processed": "data/processed", "results": "results"}}


def test_final_evaluation_unlocks_once_and_logs_every_access(project):
    assert final_eval_runs(project) == 0
    log_test_access(project, "loop run genre=story")  # a loop run on test items is logged but is not the final evaluation
    assert final_eval_runs(project) == 0
    test = unlock_for_final(project)
    assert set(test["split"]) == {"test"} and len(test) == 6
    assert final_eval_runs(project) == 1
    with pytest.raises(TestSetAlreadyUsed, match="second validation set"):
        unlock_for_final(project)  # no quiet re-runs
    assert len(unlock_for_final(project, again=True)) == 6  # a deliberate repeat is allowed...
    lines = final_mod.access_log_path(project).read_text().splitlines()
    assert len(lines) == 3 and "repeat" in lines[-1] and "loop run" in lines[0]  # ...and recorded


def test_detector_test_table_uses_the_frozen_cutoffs_and_reports_intervals():
    rng = np.random.default_rng(0)
    n = 60
    y = np.tile([0, 1], n)
    df = pd.DataFrame({"genre": "story", "label": y, "prompt_id": np.repeat(np.arange(n), 2)})
    df["ours"] = np.where(y == 1, rng.uniform(0.3, 1, 2 * n), rng.uniform(0, 0.7, 2 * n))  # overlapping, so not a perfect AUC
    df["external"] = np.where(y == 1, rng.uniform(0.4, 1, 2 * n), rng.uniform(0, 0.8, 2 * n))
    cuts = {"ours": {"story": 0.45, "all": 0.45}, "external": {"story": 0.5, "all": 0.5}}
    t = detector_test_table(df, cuts, n_boot=50).set_index(["genre", "detector"])
    row = t.loc[("story", "ours (gbm_full)")]
    assert row["cutoff"] == 0.45  # taken from the frozen file, not recomputed from this data
    assert row["human_flagged"] == pytest.approx((df.ours[df.label == 0] > 0.45).mean())
    eps = 1e-9
    assert row["caught_lo"] - eps <= row["ai_caught"] <= row["caught_hi"] + eps
    assert row["auc_lo"] - eps <= row["roc_auc"] <= row["auc_hi"] + eps and 0.7 < row["roc_auc"] < 0.95
    assert set(t.index.get_level_values("genre")) == {"story", "all"}


# ---- manual kit --------------------------------------------------------------------------------------------------


def test_manual_kit_is_blind_exports_only_ai_text_and_has_an_answer_key(tmp_path):
    recs = sample_records()
    key = build_manual_kit(recs, tmp_path / "kit", n=2, seed=1)
    assert len(key) == 4  # 2 records x (raw + rewrite); the record without an accepted rewrite is never exported
    assert {k["kind"] for k in key} == {"raw_ai", "humanized"} and all(k["record_id"] != "c" for k in key)
    names = sorted(p.name for p in (tmp_path / "kit").glob("sample-*.txt"))
    assert names == ["sample-01.txt", "sample-02.txt", "sample-03.txt", "sample-04.txt"]  # neutral names: nothing says which is which
    exported = {(tmp_path / "kit" / f"{k['sample']}.txt").read_text().strip(): k["kind"] for k in key}
    assert set(exported) <= {"a0", "a1", "b0", "b1", "d0", "d1"} and len(exported) == 4
    assert (tmp_path / "kit" / "answer_key.csv").exists()
    write_log_template(tmp_path / "log.csv", [k["sample"] for k in key], own_controls=2)
    rows = list(csv.DictReader(open(tmp_path / "log.csv")))
    assert [r["sample"] for r in rows][-2:] == ["sample-own-1", "sample-own-2"] and len(rows) == 6 and rows[0]["tool"] == ""


def test_joint_success_needs_a_kept_rewrite_below_both_cutoffs():
    recs = sample_records()  # a: both fixed, b: desklib still flags, c: failed guardrails (unchanged), d: fixed-looking but never flagged by ours
    kept = joint_success(recs, "kept")
    assert kept["n"] == 4 and kept["fooled_both"] == 1 and kept["rate"] == 0.25  # only 'a'; 'd' keeps a desklib-flagged rewrite
    assert kept["lo"] <= 0.25 <= kept["hi"]
    unguarded = joint_success(recs, "unguarded")
    assert unguarded["fooled_both"] == 2  # ignoring guardrails, c's rewrite fooled both detectors as well
    only_long = joint_success(recs, "kept", extra_ok=lambda r, s: r["id"] != "a")  # an extra requirement can veto a win
    assert only_long["fooled_both"] == 0


def test_joint_success_skips_records_without_the_independent_detector():
    recs = sample_records()
    for r in recs:
        r["ext_cutoff"] = None
    assert joint_success(recs)["n"] == 0 and math.isnan(joint_success(recs)["rate"])


def test_control_table_flags_at_the_frozen_cutoffs_and_includes_the_original():
    from hd.eval.control import control_table

    texts = ["aa bb", "cc dd", "ee ff", "gg hh"]
    ours = lambda ts: np.array([0.9 if "," in t else 0.1 for t in ts])  # a detector that only reacts to commas  # noqa: E731
    ext = lambda ts: np.array([0.95] * len(ts))  # noqa: E731
    table = control_table(texts, {"drop comma": lambda t: t.replace(",", "")}, ours, ext, 0.3, 0.7).set_index("variant")
    assert list(table.index) == ["original", "drop comma"] and (table["n"] == 4).all()
    assert table.loc["original", "ours_flagged"] == 0.0 and table.loc["original", "desklib_flagged"] == 1.0
    assert table.loc["drop comma", "desklib_lo"] < 1.0 <= table.loc["drop comma", "desklib_hi"] + 1e-9  # interval, not certainty
    with_commas = control_table([t.replace(" ", ", ") for t in texts], {"drop comma": lambda t: t.replace(",", "")}, ours, ext, 0.3, 0.7).set_index("variant")
    assert with_commas.loc["original", "ours_flagged"] == 1.0 and with_commas.loc["drop comma", "ours_flagged"] == 0.0  # the surface-only detector is fooled


def judged_records(steer=None):
    # judge cutoff 0.5. a: all three fooled. b: desklib+judge fooled, ours not. c: judge still flags the kept rewrite.
    return [
        record("a", [step(0, "a0", 0.9, 0.95, judge=0.99), step(1, "a1", 0.1, 0.2, True, 0.9, 4.0, judge=0.1)], 1, judge_cut=0.5, steer=steer),
        record("b", [step(0, "b0", 0.9, 0.95, judge=0.99), step(1, "b1", 0.1, 0.2, True, 0.9, 4.0, judge=0.9)], 1, judge_cut=0.5, steer=steer),
        record("c", [step(0, "c0", 0.9, 0.95, judge=0.99), step(1, "c1", 0.9, 0.2, True, 0.9, 4.0, judge=0.1)], 1, judge_cut=0.5, steer=steer),
    ]


def test_three_detector_tables_and_joint_success():
    recs = judged_records()
    t = outcome_table(recs).set_index("detector")
    assert set(t.index) == {"ours", "desklib", "fakespot"}
    assert t.loc["fakespot", "flagged_before"] == 1.0 and t.loc["fakespot", "flagged_after"] == pytest.approx(1 / 3)  # only b is still flagged
    assert t.loc["ours", "flagged_after"] == pytest.approx(1 / 3) and t.loc["desklib", "flagged_after"] == 0.0
    assert joint_success(recs, detectors=("ours", "desklib"))["fooled_both"] == 2  # a and b
    assert joint_success(recs, detectors=("ours", "desklib", "fakespot"))["fooled_both"] == 1  # only a satisfies all three
    assert joint_success(recs, detectors=("ours", "fakespot"))["fooled_both"] == 1
    assert joint_success(sample_records(), detectors=("ours", "fakespot"))["n"] == 0  # records without a judge are skipped


def test_the_table_says_which_detectors_steered_and_never_pools_steering_modes():
    plain, steered = judged_records(), judged_records(steer=["ours", "external"])
    t = outcome_table(plain + steered).set_index(["steer", "detector"])
    assert t.loc[("ours", "ours"), "steers"] and not t.loc[("ours", "desklib"), "steers"] and not t.loc[("ours", "fakespot"), "steers"]
    assert t.loc[("ours+external", "desklib"), "steers"] and not t.loc[("ours+external", "fakespot"), "steers"]  # the judge never steers
    assert (t["n"] == 3).all()  # two separate groups of three, not one group of six


def test_report_writer_handles_the_judge_and_records_the_three_detector_claim(tmp_path):
    import json as _json

    written = write_loop_report(judged_records(steer=["ours", "external"]), tmp_path, "j", "3 texts")
    joint = _json.loads((tmp_path / "joint_j.json").read_text())
    assert joint["kept"]["ours+desklib+fakespot"]["fooled_both"] == 1 and joint["kept"]["ours+desklib"]["fooled_both"] == 2
    assert written["before_after"].stat().st_size > 1000 and written["iterations"].stat().st_size > 1000
    old_style = write_loop_report(sample_records(), tmp_path, "old", "4 texts")  # logs from before the judge existed still report
    assert old_style["before_after"].exists()


def test_outcome_table_never_pools_strengths():
    faithful, strong = judged_records(), judged_records()
    for r in strong:
        r["strength"] = "strong"
    t = outcome_table(faithful + strong).set_index(["strength", "detector"])
    assert set(t.index.get_level_values("strength")) == {"faithful", "strong"} and (t["n"] == 3).all()
    assert outcome_table(sample_records())["strength"].eq("faithful").all()  # logs from before strengths existed count as faithful
