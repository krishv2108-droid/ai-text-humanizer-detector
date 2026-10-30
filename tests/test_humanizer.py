import hashlib
import random
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from hd.detector.explain import Analysis, SentenceResult
from hd.humanizer.calibrate import inject_typos
from hd.humanizer.feedback import HumanProfile, diagnose, render_feedback
from hd.humanizer.guardrails import GuardrailChecker, RareWordCounter, Thresholds, readability
from hd.humanizer.pipeline import COMPARE, compare_features, humanize_once
from hd.humanizer.prompts import STRATEGIES, build_prompt
from hd.humanizer.rewrite import Humanizer, clean_output
from hd.llm import LLMResponse, MockClient

# ---- fixtures: a synthetic human profile, a fake analysis, a fake embedder ---------------

FEATURES = ["sent_len_cv", "short_sent_frac", "sent_initial_transition_frac", "transition_per100w", "hedge_per100w",
            "stock_phrase_per100w", "contraction_per100w", "first_person_per100w", "emdash_per100w", "mattr",
            "mean_surprisal", "top10_frac"]  # fmt: skip


def make_profile() -> HumanProfile:
    rows = []
    for genre in ("all", "story", "casual"):
        for f in FEATURES:  # every human percentile is a simple multiple of 1.0
            rows.append({"genre": genre, "feature": f, "n": 100, "p10": 0.6, "p25": 0.8, "p50": 1.0, "p75": 1.2, "p90": 1.4, "p99": 2.0})
    return HumanProfile(pd.DataFrame(rows))


def make_analysis(text="The cat sat. However, it left. It was late.", prob=0.9, **overrides) -> Analysis:
    feats = {f: 1.0 for f in FEATURES} | overrides
    sents = [SentenceResult("The cat sat.", 0, 12, 0.3), SentenceResult("However, it left.", 13, 30, 0.8, ["stock transition opener"]),
             SentenceResult("It was late.", 31, 43, 0.4)]  # fmt: skip
    return Analysis(text, prob, 9, sents, {"rhythm": 0.3, "discourse": 0.1, "perplexity": 0.5, "contraction": 0.0}, [], feats)


class FakeEmbedder:
    """Bag-of-words hashing embeddings: identical words -> identical vectors, no model needed."""

    def encode(self, texts):
        out = np.zeros((len(texts), 128))
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z']+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class FakeTokenizer:
    """Known words are one piece; anything else fragments into 3+ pieces (like a real BPE vocabulary)."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        return {"input_ids": [0] if text.strip().lower() in self.vocab else [0, 1, 2]}


STORY = ("The lighthouse stood on the cliff. Maya climbed the stairs slowly. The storm was coming from the west. "
         "Her grandfather had kept the light for forty years. She lit the lamp and waited. The waves hit the rocks all night. "
         "At dawn the sea was calm again.")  # fmt: skip


COMMON = set(re.findall(r"[a-z']+", STORY.lower())) | {"carefully"}


def checker(**kw) -> GuardrailChecker:
    return GuardrailChecker(FakeEmbedder(), Thresholds(**kw), RareWordCounter(FakeTokenizer(COMMON)))


# ---- prompts -----------------------------------------------------------------------------


def test_strategies_include_exactly_their_blocks():
    fb = "An automated detector scored this draft 90% likely to be AI-written."
    sys_, naive = build_prompt("Some draft.", "story", "naive")
    assert "Hard requirements" in naive and "Writing techniques" not in naive and "detector" not in naive
    assert "Writing techniques" in build_prompt("Some draft.", "story", "craft")[1]
    feedback = build_prompt("Some draft.", "story", "feedback", fb)[1]
    assert "detector" in feedback and "Writing techniques" not in feedback
    full = build_prompt("Some draft.", "story", "full", fb)[1]
    assert "detector" in full and "Writing techniques" in full
    assert "Do NOT add typos" in full  # constraints are in every strategy
    assert STRATEGIES == ("naive", "craft", "feedback", "full")


def test_prompt_errors_and_original_anchor():
    with pytest.raises(ValueError):
        build_prompt("x", "story", "feedback")  # needs feedback text
    with pytest.raises(ValueError):
        build_prompt("x", "story", "bogus")
    same = build_prompt("Draft one.", "story", "craft", original="Draft one.")[1]
    assert "ORIGINAL" not in same
    anchored = build_prompt("Draft two.", "story", "craft", original="Draft one.")[1]
    assert "ORIGINAL" in anchored and "Draft one." in anchored and anchored.rstrip().endswith('"""')


def test_genre_selects_different_techniques():
    assert "dialogue" in build_prompt("d", "story", "craft")[1]
    assert "connectives" in build_prompt("d", "essay", "craft")[1]
    assert "someone talks" in build_prompt("d", "casual", "craft")[1]


# ---- output cleaning ---------------------------------------------------------------------


def test_clean_output():
    text, warn = clean_output("Here is the rewritten text:\n\n\"He said “hi”.\"\n")
    assert text == 'He said "hi".'  # preface, outer quotes and curly quotes all handled
    assert "removed a chatty preface" in warn
    text, _ = clean_output("```\nPlain story text here.\n```")
    assert text == "Plain story text here."
    _, warn = clean_output("# Title\nSome **bold** words.")
    assert "output contains markdown" in warn
    assert clean_output("It’s “fine”")[0] == 'It\'s "fine"'  # normalized like every other text


# ---- Humanizer (mock LLM) ----------------------------------------------------------------


def test_humanize_sends_the_prompt_and_cleans_the_reply():
    seen = {}

    def responder(prompt):
        seen["prompt"] = prompt
        return "Sure, here's the rewrite:\n\nShe lit the lamp. Waited."

    client = MockClient(responder)
    h = Humanizer(client, "m", make_profile(), effort="medium")
    out = h.humanize(STORY, "story", "full", analysis=make_analysis())
    assert out.text == "She lit the lamp. Waited." and "removed a chatty preface" in out.warnings
    assert "DRAFT TO REWRITE" in seen["prompt"] and "the detector scored" not in seen["prompt"]
    assert "detector scored this draft 90%" in seen["prompt"]  # feedback came from the analysis
    assert out.strategy == "full" and client.calls == 1


def test_feedback_strategies_need_analysis_and_profile():
    with pytest.raises(ValueError, match="needs a detector analysis"):
        Humanizer(MockClient(), "m", make_profile()).humanize(STORY, "story", "full")
    naive = Humanizer(MockClient(lambda p: "ok text here"), "m").humanize(STORY, "story", "naive")  # no analysis needed
    assert naive.text == "ok text here"


def test_truncated_or_empty_rewrites_are_rejected():
    class Truncated:
        def complete(self, **kw):
            return LLMResponse(text="cut off mid", stop_reason="max_tokens")

    with pytest.raises(RuntimeError, match="cut off"):
        Humanizer(Truncated(), "m").humanize(STORY, "story", "naive")
    with pytest.raises(RuntimeError, match="empty"):
        Humanizer(MockClient(lambda p: "   "), "m").humanize(STORY, "story", "naive")


# ---- diagnosis ---------------------------------------------------------------------------


def test_diagnose_flags_only_what_is_outside_the_human_range():
    profile = make_profile()
    assert diagnose(make_analysis(), profile, "story") == []  # everything typical -> nothing to fix
    a = make_analysis(sent_len_cv=0.3, contraction_per100w=0.0, top10_frac=3.0, hedge_per100w=5.0)
    groups = [w.group for w in diagnose(a, profile, "story")]
    assert set(groups) == {"rhythm", "contraction", "perplexity", "hedging"}
    assert groups[0] == "perplexity" and groups[1] == "rhythm"  # ordered by the detector's group effect (0.5, 0.3)
    w = next(w for w in diagnose(a, profile, "story") if w.group == "rhythm")
    assert "0.30" in w.evidence and "1.00" in w.evidence  # the numbers are in the instruction


def test_render_feedback_includes_flagged_sentences():
    a = make_analysis(sent_len_cv=0.3)
    text = render_feedback(diagnose(a, make_profile(), "story"), a)
    assert text.startswith("An automated detector scored this draft 90%")
    assert '"However, it left."' in text and "stock transition opener" in text
    quiet = replace(make_analysis(prob=0.1), sentences=[])
    assert "no specific weaknesses" in render_feedback([], quiet)


def test_profile_falls_back_to_pooled_genre():
    p = make_profile()
    assert p.q("essay", "mattr", "p50") == 1.0  # 'essay' not in the table -> pooled 'all'
    assert p.q("story", "mattr", "p99") == 2.0


# ---- guardrails --------------------------------------------------------------------------


def test_identical_text_passes_everything():
    r = checker().check(STORY, STORY)
    assert r.ok and r.similarity == pytest.approx(1.0) and r.length_ratio == 1.0 and r.novel_rare_rate == 0


def test_unrelated_text_is_flagged_as_drift():
    other = "Quarterly revenue increased nine percent while operating costs fell across every region of the company."
    r = checker().check(STORY, other)
    assert not r.ok and any(f.startswith("meaning drift") for f in r.flags)
    assert any(f.startswith("invented content") for f in r.flags)


def test_dropped_sentences_and_length_are_flagged():
    half = " ".join(STORY.split(". ")[:3]) + "."
    r = checker().check(STORY, half)
    assert any(f.startswith("length changed") for f in r.flags) or any(f.startswith("dropped content") for f in r.flags)
    assert any(f.startswith("dropped content") for f in r.flags)


def test_typos_are_flagged_clean_paraphrase_is_not():
    typo = inject_typos(STORY, 1.0, random.Random(0))
    assert typo != STORY
    assert any(f.startswith("possible typos") for f in checker().check(STORY, typo).flags)
    assert checker().check(STORY, STORY.replace("slowly", "carefully")).ok


def test_readability_shift_is_flagged():
    plain = "The cat sat on the mat. The dog ran. It was fun. We all went home. Then we ate."
    dense = ("Notwithstanding considerable epistemological uncertainty, the aforementioned quadruped relinquished "
             "its horizontal position, whereupon institutional stakeholders deliberated extensively regarding ramifications.")  # fmt: skip
    assert readability(dense)["fk_grade"] > readability(plain)["fk_grade"] + 5
    r = checker(min_doc_similarity=-1, max_dropped_frac=1, max_invented_frac=1, length_ratio=(0, 99)).check(plain, dense)
    assert any(f.startswith("readability shifted") for f in r.flags)


def test_reading_level_rule_uses_the_human_range_not_the_change():
    plain = "The cat sat on the mat. The dog ran. It was fun. We all went home. Then we ate."
    dense = ("Notwithstanding considerable epistemological uncertainty, the aforementioned quadruped relinquished "
             "its horizontal position, whereupon institutional stakeholders deliberated extensively regarding ramifications.")  # fmt: skip
    loose = dict(min_doc_similarity=-1, max_dropped_frac=1, max_invented_frac=1, length_ratio=(0, 99))
    g_plain, g_dense = readability(plain)["fk_grade"], readability(dense)["fk_grade"]
    # Dense -> plain is a huge drop, but plain text lies inside the human range, so it is NOT flagged...
    inside = checker(**loose, fk_range=(g_plain - 1, g_dense + 1)).check(dense, plain)
    assert not any("reading" in f or "readability" in f for f in inside.flags)
    # ...while a rewrite that leaves the human range is flagged, even with a tiny change.
    outside = checker(**loose, fk_range=(g_plain - 1, g_plain + 1)).check(plain, dense)
    assert any(f.startswith("reading level") and "outside the human range" in f for f in outside.flags)


def test_human_fk_range_comes_from_train_split_human_text_only(tmp_path, monkeypatch):
    from hd.humanizer.pipeline import human_fk_range

    easy, hard = "The cat sat. The dog ran. It was fun.", ("Notwithstanding considerable epistemological uncertainty, "
                  "institutional stakeholders deliberated extensively regarding ramifications.")  # fmt: skip
    rows = [{"genre": "story", "label": 0, "split": "train", "text": easy}] * 5
    rows += [{"genre": "story", "label": 1, "split": "train", "text": hard}] * 5  # AI side must not widen the range
    rows += [{"genre": "story", "label": 0, "split": "val", "text": hard}] * 5  # neither may validation text
    rows += [{"genre": "essay", "label": 0, "split": "train", "text": hard}] * 5  # nor another genre
    (tmp_path / "data" / "processed").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(tmp_path / "data" / "processed" / "corpus.parquet")
    monkeypatch.setattr("hd.config.PROJECT_ROOT", tmp_path)  # resolve_path is relative to the project root
    cfg = {"paths": {"processed": "data/processed"}}
    lo, hi = human_fk_range(cfg, "story")
    assert lo == pytest.approx(readability(easy)["fk_grade"]) and hi == pytest.approx(lo)
    assert human_fk_range(cfg, "casual") is None


def test_guardrails_catch_changed_numbers_and_say_which():
    loose = dict(min_doc_similarity=-1, max_dropped_frac=1, max_invented_frac=1, length_ratio=(0, 99), fk_range=(-99, 99))
    changed = checker(**loose).check(STORY, STORY.replace("forty", "fourteen"))
    assert changed.missing_numbers == [40] and any(f.startswith("numbers changed or lost (40)") for f in changed.flags)
    assert checker(**loose).check(STORY, STORY.replace("forty", "40")).ok  # same value spelled as digits is fine


def test_guardrails_report_exactly_which_sentences_were_dropped_or_invented():
    dropped = STORY.replace("She lit the lamp and waited. ", "")
    r = checker().check(STORY, dropped)
    assert r.missing_sentences == ["She lit the lamp and waited."] and r.invented_sentences == []
    invented = STORY + " Quantum cheese negotiations continued indefinitely."  # no vocabulary in common with the story
    r = checker().check(STORY, invented)
    assert r.invented_sentences == ["Quantum cheese negotiations continued indefinitely."] and r.missing_sentences == []
    typo = inject_typos(STORY, 1.0, random.Random(0))
    assert checker().check(STORY, typo).novel_words  # the report lists the suspect words, not just a rate


def test_diagnose_tells_an_overshot_rewrite_to_come_back():
    profile = make_profile()  # human p90 is 1.4 for every feature
    choppy = diagnose(make_analysis(short_sent_frac=1.6), profile, "story")
    assert choppy[0].evidence.startswith("Too choppy") and "Merge some fragments" in choppy[0].fix
    assert choppy[0].severity == 2.0  # a correction outranks every ordinary weakness
    assert diagnose(make_analysis(short_sent_frac=1.39), profile, "story") == []  # inside the human range: leave it alone
    odd = diagnose(make_analysis(mean_surprisal=1.8), profile, "story")
    assert any("odd or showy" in w.evidence for w in odd)
    wild = diagnose(make_analysis(sent_len_cv=1.9), profile, "story")
    assert any("swing too wildly" in w.evidence for w in wild)
    both = diagnose(make_analysis(short_sent_frac=1.6, sent_len_cv=0.5), profile, "story")  # choppy AND otherwise AI-like cues
    assert both[0].evidence.startswith("Too choppy")


def test_every_strategy_demands_full_coverage_and_exact_facts_and_the_version_is_recorded():
    from hd.humanizer.prompts import PROMPT_VERSION

    fb = "An automated detector scored this draft 90% likely to be AI-written."
    for strategy in STRATEGIES:
        user = build_prompt("Some draft.", "story", strategy, fb if strategy in ("feedback", "full") else None)[1]
        assert "sentence by sentence" in user and "exactly as written" in user
    assert "do not turn every sentence into a fragment" in build_prompt("Some draft.", "story", "craft")[1]
    assert "Say it in your own words" not in build_prompt("Some draft.", "story", "craft")[1]  # the reverted v3 lever stays out
    assert PROMPT_VERSION == "v2"


def test_restructuring_transforms_preserve_every_word_and_damage_transforms_do_not():
    from hd.humanizer.calibrate import coverage_scenarios, merge_pairs, split_into_fragments

    text = ("The old lighthouse stood at the edge of the cliff, its weathered stones bearing silent witness to countless storms. "
            "Maria climbed the winding staircase, and her heart was heavy with loss; however, she kept going. "
            "At the top she stopped. The sea was wide and grey. Nobody was waiting for her there.")  # fmt: skip
    links = {"and", "but", "so", "yet", "while", "as", "because"}
    words = lambda t: sorted(w for w in re.findall(r"[a-z']+", t.lower()) if w not in links)  # noqa: E731
    split, merged = split_into_fragments(text), merge_pairs(text)
    assert words(split) == words(text)  # no content word lost: only punctuation, capitals and linking conjunctions change
    assert len(re.findall(r"[.!?]", split)) > len(re.findall(r"[.!?]", text))  # more, shorter sentences
    assert words(merged) == words(text) and len(re.findall(r"[.!?]", merged)) < len(re.findall(r"[.!?]", text))
    scenarios = coverage_scenarios(text, [text.replace("lighthouse", "harbour")] * 4, random.Random(1))
    assert scenarios["identical"] == text and len(scenarios["drop30"].split()) < len(text.split())


def test_restructuring_summary_reports_separation_and_flag_rates():
    from hd.humanizer.calibrate import summarize_restructuring

    rows = []
    for i in range(20):
        for s in ("identical", "split", "merge"):
            rows.append({"scenario": s, "id": i, "dropped_frac": 0.0, "invented_frac": 0.02 * (i % 3), "similarity": 0.99})
        rows.append({"scenario": "drop30", "id": i, "dropped_frac": 0.3, "invented_frac": 0.0, "similarity": 0.9})
        rows.append({"scenario": "swap30", "id": i, "dropped_frac": 0.2, "invented_frac": 0.3, "similarity": 0.8})
    t = summarize_restructuring(pd.DataFrame(rows), max_dropped=0.2, max_invented=0.25).set_index("damage")
    assert t.loc["drop30", "auc_vs_restructuring"] == 1.0 and t.loc["drop30", "flagged_when_damaged"] == 1.0
    assert t.loc["drop30", "flagged_when_split"] == 0.0 and t.loc["swap30", "flagged_when_merge"] == 0.0  # restructuring is not punished
    assert t.loc["swap30", "flagged_when_damaged"] == 1.0 and t.loc["swap30", "restructuring_p95"] == pytest.approx(0.04)


def test_strong_strength_changes_the_prompt_but_keeps_facts_protected():
    fb = "An automated detector scored this draft 90% likely to be AI-written."
    faithful = build_prompt("Some draft.", "story", "full", fb)[1]
    strong = build_prompt("Some draft.", "story", "full", fb, strength="strong")[1]
    assert "sentence by sentence" in faithful and "sentence by sentence" not in strong  # strong may drop minor detail
    assert "compress, merge or drop minor descriptive details" in strong and "compress" not in faithful
    assert "one sentence in three" in strong and "one sentence in four" in faithful  # the rhythm technique is the bolder one
    for text in (faithful, strong):  # numbers and names are never negotiable
        assert "every number" in text and "exactly as written" in text and "Do NOT add typos" in text
    assert "Do not invent new plot points" in strong
    essay_f, essay_s = build_prompt("d.", "essay", "craft")[1], build_prompt("d.", "essay", "craft", strength="strong")[1]
    assert essay_f.split("Writing techniques")[1] == essay_s.split("Writing techniques")[1]  # the technique list only differs for stories
    with pytest.raises(ValueError, match="unknown strength"):
        build_prompt("d.", "story", "naive", strength="reckless")


def test_humanizer_passes_its_strength_into_the_request():
    prompt_for = lambda strength: Humanizer(None, "m", None, None, strength).build_request("A draft.", "story", "naive")[1]  # noqa: E731
    assert "compress, merge or drop" in prompt_for("strong") and "compress" not in prompt_for("faithful")


def test_guardrail_limits_follow_the_strength_but_only_the_semantic_ones():
    cfg = {"guardrails": {"min_doc_similarity": 0.5, "max_dropped_frac": 0.5, "length_ratio": [0.7, 1.4], "max_novel_rare_rate": 0.01,
                          "strengths": {"faithful": {"min_doc_similarity": 0.80, "max_dropped_frac": 0.20},
                                        "strong": {"min_doc_similarity": 0.72, "max_dropped_frac": 0.30}}}}  # fmt: skip
    faithful, strong = Thresholds.from_config(cfg), Thresholds.from_config(cfg, "strong")
    assert (faithful.min_doc_similarity, faithful.max_dropped_frac) == (0.80, 0.20)
    assert (strong.min_doc_similarity, strong.max_dropped_frac) == (0.72, 0.30)
    assert faithful.length_ratio == strong.length_ratio == (0.7, 1.4) and faithful.max_novel_rare_rate == strong.max_novel_rare_rate == 0.01
    assert faithful.max_missing_name_frac == strong.max_missing_name_frac  # facts are checked the same way in every mode
    with pytest.raises(ValueError, match="unknown strength"):
        Thresholds.from_config(cfg, "reckless")
    assert Thresholds.from_config({"guardrails": {"min_doc_similarity": 0.6}}, "strong").min_doc_similarity == 0.6  # old configs still load


def test_strong_mode_accepts_a_looser_rewrite_but_still_rejects_a_changed_number():
    cfg = {"guardrails": {"strengths": {"faithful": {"min_doc_similarity": 0.99, "max_dropped_frac": 0.0},
                                        "strong": {"min_doc_similarity": 0.20, "max_dropped_frac": 1.0}}}}  # fmt: skip
    rewrite = STORY.replace("She lit the lamp and waited. ", "")  # a sentence is gone
    make = lambda strength: GuardrailChecker(FakeEmbedder(), replace(Thresholds.from_config(cfg, strength), length_ratio=(0, 99), fk_range=(-99, 99)),  # noqa: E731
                                              RareWordCounter(FakeTokenizer(COMMON)))
    assert not make("faithful").check(STORY, rewrite).ok  # dropped content is flagged
    assert make("strong").check(STORY, rewrite).ok  # allowed to drop a minor detail
    assert any("numbers changed" in f for f in make("strong").check(STORY, STORY.replace("forty", "fourteen")).flags)  # but never a fact


def test_inject_typos_is_deterministic_and_respects_rate():
    a, b = inject_typos(STORY, 0.5, random.Random(1)), inject_typos(STORY, 0.5, random.Random(1))
    assert a == b and inject_typos(STORY, 0.0, random.Random(1)) == STORY
    assert len(a) == len(STORY)  # swaps letters, never adds or removes any


def test_thresholds_from_config():
    t = Thresholds.from_config({"guardrails": {"min_doc_similarity": 0.5, "length_ratio": [0.5, 2.0], "unknown_key": 1}})
    assert t.min_doc_similarity == 0.5 and t.length_ratio == (0.5, 2.0) and t.max_fk_delta == 3.0


# ---- one pass end to end (all stubs) -----------------------------------------------------


class StubDetector:
    def analyze(self, text):
        n_short = sum(len(s) < 20 for s in re.split(r"(?<=[.!?])\s+", text))
        feats = {f: 1.0 for f in FEATURES} | {"sent_len_cv": 0.3 + 0.2 * n_short}
        return make_analysis(text, prob=0.95 if n_short == 0 else 0.3, **feats)


class StubExternal:
    def predict_proba(self, texts):
        return np.array([0.9, 0.4])


def test_humanize_once_builds_the_full_report():
    rewrite = "The lighthouse stood on the cliff. Maya climbed the stairs. A storm. Her grandfather had kept the light for forty years. She lit the lamp and waited. The waves hit the rocks all night. At dawn the sea was calm."  # fmt: skip
    h = Humanizer(MockClient(lambda p: rewrite), "m", make_profile())
    rep = humanize_once(StubDetector(), h, checker(), STORY, "story", "full", StubExternal())
    assert rep.before.prob_ai > rep.after.prob_ai
    assert rep.rewrite.text == rewrite and rep.guardrails.similarity > 0.8
    assert (rep.external_before, rep.external_after) == (0.9, 0.4)
    assert set(rep.comparison["feature"]) == set(COMPARE)
    row = rep.comparison.set_index("feature").loc["sent_len_cv"]
    assert row["after"] > row["before"] and set(rep.comparison.columns) >= {"before", "after", "human_median", "moved_toward_human"}
    assert compare_features(rep.before, rep.after, make_profile(), "story").equals(rep.comparison)


# ---- real embedder / tokenizer (skipped when not cached) ---------------------------------


@pytest.fixture(scope="module")
def real_checker():
    pytest.importorskip("sentence_transformers")
    from hd.humanizer.guardrails import SentenceEmbedder

    try:
        return GuardrailChecker(SentenceEmbedder(), Thresholds(), RareWordCounter())
    except OSError:
        pytest.skip("embedding model or GPT-2 tokenizer not cached")


def test_real_models_tell_paraphrase_from_drift_and_catch_typos(real_checker):
    para = ("A lighthouse stood at the cliff's edge. Maya slowly went up the stairs. A storm was rolling in from the west. "
            "Her grandfather had tended the light for forty years. She lit the lamp, then waited. All night the waves pounded the rocks. "
            "By dawn the sea had settled.")  # fmt: skip
    other = "The committee approved the annual budget after a long debate about funding for public transport and school repairs. Several members objected."
    assert real_checker.check(STORY, para).similarity > 0.85
    assert real_checker.check(STORY, para).ok
    assert any("drift" in f for f in real_checker.check(STORY, other).flags)
    typo = inject_typos(STORY, 0.5, random.Random(0))
    assert any(f.startswith("possible typos") for f in real_checker.check(STORY, typo).flags)
