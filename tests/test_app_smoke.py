"""Run the real Streamlit script headlessly with fake models (no GPU, no network, no API quota)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from hd import app_core  # noqa: E402
from hd.app_core import Resources  # noqa: E402
from hd.config import PROJECT_ROOT, load_config  # noqa: E402
from test_humanizer import make_analysis, make_profile  # noqa: E402

THRESHOLDS = {d: {g: 0.3 for g in ("story", "essay", "casual", "all")} for d in ("ours", "external")}


class FakeDetector:
    scorer = None

    def analyze(self, text):
        return make_analysis(prob=0.9)


@pytest.fixture
def app(monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setattr("hd.config.load_env", lambda *a, **k: False)  # never read the real .env in tests
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    fake = Resources(load_config(), FakeDetector(), make_profile(), None, THRESHOLDS, None)
    monkeypatch.setattr(app_core, "load_resources", lambda cfg, with_external=False, with_judge=False: fake)
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_app_starts_with_two_tabs_and_the_honesty_banner(app):
    assert [t.label for t in app.tabs] == ["Detect", "Humanize"]
    assert any("not a lie detector" in i.value for i in app.info)
    assert app.title[0].value == "Humanizer vs. Detector"


def test_detect_shows_the_score_a_verdict_and_shaded_sentences(app):
    app.text_area(key="detect_text").set_value("The cat sat. However, it left. It was late.").run()
    app.button(key="detect_go").click().run()
    assert not app.exception
    metric = app.metric[0]
    assert metric.label == "Our detector: P(AI)" and metric.value == "0.90" and metric.delta == "flagged"
    assert any("<span style=" in m.value and "However, it left." in m.value for m in app.markdown)


def test_short_input_is_rejected_with_a_message_not_a_crash(app):
    app.text_area(key="detect_text").set_value("too short").run()
    app.button(key="detect_go").click().run()
    assert not app.exception and any("at least 5 words" in w.value for w in app.warning)
    assert len(app.metric) == 0


def test_humanize_dry_run_shows_the_prompt_and_calls_no_api(app):
    app.text_area(key="hum_text").set_value("The cat sat. However, it left. It was late.").run()
    app.checkbox(key="hum_dry").check().run()
    app.button(key="hum_go").click().run()
    assert not app.exception
    prompt = app.code[0].value
    assert "--- SYSTEM ---" in prompt and "DRAFT TO REWRITE" in prompt and "Writing techniques" in prompt  # the 'full' strategy


def test_humanize_without_an_api_key_gives_a_clear_error(app):
    app.text_area(key="hum_text").set_value("The cat sat. However, it left. It was late.").run()
    app.button(key="hum_go").click().run()
    assert not app.exception
    assert any("GROQ_API_KEY" in e.value for e in app.error)  # names the variable the user has to set


def test_example_button_fills_the_text_box_without_a_widget_state_error(app):
    app.button(key="detect_example").click().run()
    assert not app.exception  # regression: setting a widget's value after it exists raises in Streamlit
    assert "lighthouse" in app.text_area(key="detect_text").value
    app.button(key="detect_go").click().run()
    assert not app.exception and app.metric[0].value == "0.90"


def _result(kept: bool):
    from hd.loop import LoopResult, Step

    orig = Step(0, "The cat sat. However, it left. It was late.", 0.9, 0.95)
    rewrite = Step(1, "Cat sat. It left. Late.", 0.1, 0.8, kept, [] if kept else ["meaning drift (similarity 0.50 < 0.8)"], 0.9, 1.0, 4.0, 100, 50)
    return LoopResult("x", "story", "full", 0.3, 0.7, [orig, rewrite], 1 if kept else None, "success" if kept else "max_iterations", "m/x")


def _humanize(app, monkeypatch, result):
    monkeypatch.setattr("hd.llm.make_client", lambda cfg, cache_dir: object())
    monkeypatch.setattr(app_core, "humanize_text", lambda *a, **k: result)
    app.text_area(key="hum_text").set_value("The cat sat. However, it left. It was late.").run()
    app.button(key="hum_go").click().run()
    assert not app.exception


def test_humanize_result_shows_the_kept_rewrite_and_both_detectors(app, monkeypatch):
    _humanize(app, monkeypatch, _result(kept=True))
    assert any("Kept rewrite 1" in m.value for m in app.markdown)
    assert [m.label for m in app.metric] == ["Our detector: P(AI)", "desklib: P(AI)"]
    assert app.metric[0].value == "0.10" and app.text_area[-1].value == "Cat sat. It left. Late."
    assert app.dataframe  # the table of every attempt


def test_humanize_failure_never_shows_a_rejected_rewrite_as_the_result(app, monkeypatch):
    _humanize(app, monkeypatch, _result(kept=False))
    assert any("unchanged" in m.value for m in app.markdown)
    assert len(app.metric) == 0  # no before/after scores presented for a rewrite the user doesn't get
    assert all(t.value != "Cat sat. It left. Late." for t in app.text_area)  # rejected text is only inside the warned expander
    assert any("failed the guardrails" in e.label for e in app.expander)


def test_humanize_reports_a_used_up_daily_limit_clearly(app, monkeypatch):
    from hd.llm import DailyLimitReached

    monkeypatch.setattr("hd.llm.make_client", lambda cfg, cache_dir: object())

    def boom(*a, **k):
        raise DailyLimitReached("tokens per day", 3600)

    monkeypatch.setattr(app_core, "humanize_text", boom)
    app.text_area(key="hum_text").set_value("The cat sat. However, it left. It was late.").run()
    app.button(key="hum_go").click().run()
    assert not app.exception and any("daily limit" in e.value for e in app.error)


def test_dry_run_shows_the_strong_prompt_when_strong_is_chosen(app):
    app.text_area(key="hum_text").set_value("The cat sat. However, it left. It was late.").run()
    app.checkbox(key="hum_dry").check().run()
    app.radio(key="hum_strength").set_value("strong").run()
    app.button(key="hum_go").click().run()
    assert not app.exception
    prompt = app.code[0].value
    assert "compress, merge or drop minor descriptive details" in prompt and "sentence by sentence" not in prompt
