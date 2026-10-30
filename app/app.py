"""Streamlit app: Detect and Humanize tabs. Run with `uv run streamlit run app/app.py` (or `uv run hd app`)."""

from __future__ import annotations

import streamlit as st

from hd.app_core import (GENRES, KNOWN_MODELS, estimated_tokens, GROUP_LABELS, group_chart_frame, humanize_text, input_problem, input_warning,
                         load_resources, outcome_text, sentence_html, steps_frame, top_drivers, cutoffs_for, verdict)  # fmt: skip
from hd.config import load_config, load_env, resolve_path
from hd.eval.style import AI_C, HUMAN_C
from hd.humanizer.prompts import STRATEGIES
from hd.llm import DailyLimitReached, RateLimited, make_client

EXAMPLE = (
    "The old lighthouse stood at the edge of the cliff, its weathered stones bearing silent witness to countless storms. "
    "Maria climbed the winding staircase, her heart heavy with a profound sense of loss. However, as she reached the top, "
    "she realized that the light had always been a symbol of hope. It was a moment of quiet transformation, a reminder "
    "that even in the darkest times, there is always a path forward."
)

st.set_page_config(page_title="Humanizer vs. Detector", page_icon=None, layout="wide")


@st.cache_resource(show_spinner="Loading the detector (first time only)...")
def resources(with_external: bool, with_judge: bool):
    load_env()
    return load_resources(load_config(), with_external, with_judge)


def _load_example(key: str) -> None:
    """Button callback: callbacks run before the script reruns, the only time a widget's value may be set from code."""
    st.session_state[key] = EXAMPLE


def banner() -> None:
    st.info(
        "A research tool, not a lie detector. Both detectors are set to flag 5% of real human text, so \"flagged\" is a "
        "statistical signal about *this kind of text*, never proof of who wrote it, and \"not flagged\" is not proof of human "
        "authorship. The evidence about rewriting is a small pilot (see `reports/FINDINGS.md`)."
    )


def detect_tab(res, genre: str, use_external: bool, use_judge: bool) -> None:
    text = st.text_area("Text to analyze", key="detect_text", height=220, placeholder="Paste a story, an answer or a short essay...")
    st.button("Load an example paragraph", key="detect_example", on_click=_load_example, args=("detect_text",))
    if st.button("Detect", type="primary", key="detect_go"):
        problem = input_problem(text)
        if problem:
            st.warning(problem)
            return
        if (warn := input_warning(text)):
            st.warning(warn)
        analysis = res.detector.analyze(text)
        cuts = cutoffs_for(res, genre)
        c1, c2 = st.columns(2)
        label, why = verdict(analysis.prob_ai, cuts["ours"])
        c1.metric("Our detector: P(AI)", f"{analysis.prob_ai:.2f}", label)
        c1.caption(why)
        if use_external and res.external is not None and cuts["external"] is not None:
            ext = float(res.external.predict_proba([analysis.text])[0])
            label2, why2 = verdict(ext, cuts["external"])
            c2.metric("Independent detector (desklib): P(AI)", f"{ext:.2f}", label2)
            c2.caption(why2)
        if use_judge and res.judge is not None and cuts["judge"] is not None:
            third = float(res.judge.predict_proba([analysis.text])[0])
            label3, why3 = verdict(third, cuts["judge"])
            st.metric("Third detector (fakespot): P(AI)", f"{third:.2f}", label3)
            st.caption(why3)
        st.subheader("Sentence by sentence")
        st.caption("Darker = the sentence looks more AI-written to our detector. Hover a sentence for the reasons. "
                   "Sentence scores are a weakly supervised estimate: read them as hints, not verdicts.")  # fmt: skip
        st.markdown(f"<div style='line-height:1.8'>{sentence_html(analysis)}</div>", unsafe_allow_html=True)
        st.subheader("What drives the score")
        st.caption("Points of P(AI) that disappear if this group of cues looked typical. Orange pushes toward AI, blue toward human.")
        st.bar_chart(group_chart_frame(analysis), horizontal=True, color=[AI_C, HUMAN_C], stack=True)
        with st.expander("What do these groups measure?"):
            st.markdown("\n".join(f"- {label}" for label in GROUP_LABELS.values()))
        st.subheader("Strongest individual cues")
        drivers = top_drivers(analysis)
        st.dataframe(drivers.round(3) if len(drivers) else drivers, hide_index=True, use_container_width=True)
        st.caption("'typical' is the median over the detector's training texts. Cues that overlap (such as the four GPT-2 "
                   "predictability features) cover for each other, so the group chart is more trustworthy.")  # fmt: skip


def humanize_tab(res, genre: str, use_external: bool, use_judge: bool) -> None:
    cfg = res.cfg
    text = st.text_area("AI-written text to rewrite", key="hum_text", height=220, placeholder="Paste AI-written text...")
    c1, c2, c3 = st.columns(3)
    strategy = c1.selectbox("Prompt strategy", STRATEGIES, index=STRATEGIES.index("full"),
                            help="naive: just 'sound human'. craft: + writing techniques. feedback: + what the detector found. full: all of it.")  # fmt: skip
    models = list(dict.fromkeys([cfg["llm"]["humanizer_model"]] + KNOWN_MODELS))
    model = c2.selectbox("Humanizer model (Groq)", models, help="Each Groq model has its own daily token allowance on the free tier.")
    rewrites = c3.slider("Max rewrites", 1, 4, 2, help="The loop stops early once our detector stops flagging a rewrite that keeps the meaning.")
    strength = st.radio("Rewrite strength", ["faithful", "strong"], horizontal=True, key="hum_strength",
                        help="Faithful keeps every sentence's content (meaning similarity of at least 0.80). Strong may compress or drop minor descriptive "
                             "detail (at least 0.72): more likely to fool detectors, but it changes more of your text. Numbers and names are protected in both.")  # fmt: skip
    steer = st.checkbox("Let desklib steer the rewrite too (stronger, uses more quota; desklib is then no longer an independent check)",
                        value=False, key="hum_steer", disabled=not use_external,
                        help="Needs the desklib option in the sidebar. The third detector (fakespot) never steers, so it stays a fair check.")  # fmt: skip
    dry = st.checkbox("Dry run: show the prompt, call no API", value=False, key="hum_dry")
    if text.strip() and not input_problem(text):
        st.caption(f"Rough cost if every rewrite runs: about {estimated_tokens(text, strategy, rewrites):,} tokens "
                   f"({len(text.split())} words). The free tier allows 200,000 tokens per model per day.")  # fmt: skip
    if not st.button("Humanize", type="primary", key="hum_go"):
        return
    problem = input_problem(text)
    if problem:
        st.warning(problem)
        return
    if (warn := input_warning(text)):
        st.warning(warn)
    if dry:
        from hd.humanizer.rewrite import Humanizer

        analysis = res.detector.analyze(text)
        system, user = Humanizer(None, model, res.profile, strength=strength).build_request(analysis.text, genre, strategy, analysis)  # type: ignore[arg-type]
        st.code(f"--- SYSTEM ---\n{system}\n\n--- USER ---\n{user}", language="text", wrap_lines=True)
        return
    try:
        client = make_client(cfg, resolve_path(cfg, "cache"))
        with st.spinner("Rewriting and scoring (each rewrite is one API call plus local scoring)..."):
            result = humanize_text(res, client, text, genre, strategy=strategy, model=model, max_rewrites=rewrites, strength=strength,
                                   use_external=use_external and res.external is not None, effort=cfg["llm"].get("humanizer_effort"),
                                   steer_desklib=steer and use_external and res.external is not None, use_judge=use_judge and res.judge is not None)  # fmt: skip
    except DailyLimitReached as err:
        st.error(f"The free-tier daily limit for this model is used up ({err}). Try another model or wait.")
        return
    except RateLimited as err:
        st.error(f"Rate limited: {err}. Wait a minute and try again.")
        return
    except RuntimeError as err:  # e.g. missing GROQ_API_KEY
        st.error(str(err))
        return

    st.subheader("Result")
    st.write(outcome_text(result))
    kept = result.steps[result.final] if result.final is not None else None
    if kept is not None:
        st.text_area("Rewritten text (kept)", kept.text, height=260)
        before, after = result.steps[0], kept
        m1, m2 = st.columns(2)
        m1.metric("Our detector: P(AI)", f"{after.ours:.2f}", f"{after.ours - before.ours:+.2f} vs original", delta_color="inverse")
        if after.external is not None:
            m2.metric("desklib: P(AI)", f"{after.external:.2f}", f"{after.external - before.external:+.2f} vs original", delta_color="inverse")
        if after.judge is not None:
            st.metric("fakespot (third detector, never steers): P(AI)", f"{after.judge:.2f}", f"{after.judge - before.judge:+.2f} vs original", delta_color="inverse")
        st.caption("Our detector steers the loop" + (" and so does desklib" if result.steer == ["ours", "external"] else "; desklib never does")
                   + ". Detectors that never steer are the fairer checks. A rewrite can fool one and not another.")  # fmt: skip
    st.subheader("Every attempt")
    st.dataframe(steps_frame(result), hide_index=True, use_container_width=True)
    if kept is None and len(result.steps) > 1:
        with st.expander("Show the rejected attempts (not recommended: they failed the guardrails)"):
            for s in result.steps[1:]:
                st.markdown(f"**Rewrite {s.iteration}** rejected: {'; '.join(s.flags)}")
                st.text(s.text)
    st.caption(f"Strength: {result.strength}. Stopped because: {result.stop_reason}. Tokens used this run: "
               f"{sum(s.input_tokens + s.output_tokens for s in result.steps):,}.")  # fmt: skip


def main() -> None:
    st.title("Humanizer vs. Detector")
    banner()
    st.sidebar.header("Settings")
    genre = st.sidebar.selectbox("Genre of the text", GENRES, help="Sets the human reference and the detector cutoff.")
    use_external = st.sidebar.checkbox("Also score with the independent detector (desklib)", value=False,
                                       help="Loads a ~1.7 GB model the first time; scoring is slower.", key="use_external")  # fmt: skip
    use_judge = st.sidebar.checkbox("Also score with the third detector (fakespot)", value=False, key="use_judge",
                                    help="Loads a ~0.5 GB model the first time. It only ever judges: it never steers a rewrite.")  # fmt: skip
    try:
        res = resources(use_external, use_judge)
    except RuntimeError as err:
        st.error(str(err))
        st.stop()
    tab1, tab2 = st.tabs(["Detect", "Humanize"])
    with tab1:
        detect_tab(res, genre, use_external, use_judge)
    with tab2:
        humanize_tab(res, genre, use_external, use_judge)
    with st.expander("Limitations"):
        st.markdown(
            "- Trained and tested on one AI generator per genre (2022 ChatGPT for essays/casual, one open model for stories): "
            "high scores here do not mean AI text in general is easy to spot.\n"
            "- Rewriting results come from a 10-story pilot and are not established.\n"
            "- Rewrites that fool a detector can read as unnatural (choppy fragments); read the output yourself.\n"
            "- The free Groq tier limits how many rewrites you can run per day."
        )


main()
