"""Sentence-level surgery: rewrite only the sentences that make a text look machine-written, and leave the rest alone.

Whole-text rewriting trades meaning for evasion (the frontier in reports/FINDINGS.md section 2). Surgery tries to break that
link: most of the text is the original, byte for byte, so meaning can only move where a sentence was replaced.

Each pass:
  1. For every sentence, measure how much the independent detector's raw score (its logit, which does not saturate) falls
     when that sentence is removed. Large drop = that sentence is carrying the "AI" signal.
  2. Ask the LLM to rewrite only the top few of them, shown in context, one replacement per marked sentence.
  3. Splice the replacements back at their exact character positions (paragraph breaks and every other sentence untouched),
     then run the usual guardrails against the ORIGINAL text and score every detector.
Sentences already rewritten (or whose rewrite was rejected) are not picked again, so each pass attacks new material.

Here the independent detector (desklib) STEERS, which is why it is no longer an independent check; the third detector
(fakespot) only judges, exactly as with `hd loop run --steer-desklib`. Logs record steer = ["ours", "external"].
"""

from __future__ import annotations

import math
import re
from typing import Any

from hd.data.clean import normalize_text
from hd.features.text import analyze, sentences
from hd.llm import RateLimited
from hd.loop import LoopResult, Step, choose_final

SURGERY_VERSION = "surgery1"

SYSTEM = "You are a line editor who makes machine-written prose read like ordinary human writing. You never change what a text says."

USER = """Below is a {genre}. Some sentences are marked like [#3]...[/#3]. They read as machine-written. Rewrite ONLY the marked sentences.

Rules for each rewritten sentence:
- Say exactly what the original says: the same events, facts, names and numbers. Add nothing and drop nothing.
- It may become one or two sentences, and it must fit smoothly between its neighbours.
- Write it the way an ordinary person would say it: plain words, a different sentence shape from the original (shorter, or a loose run-on, or opening with something other than the subject), contractions where a person would use them.
- Avoid polished flourishes: no stacked adjectives, no lists of three, no em dashes, no tidy closing line.
- Do not make every rewrite short, and do not reuse the original's opening words.

Answer with one line per marked sentence, exactly like this and nothing else:
[#3] <your rewrite>
[#7] <your rewrite>

TEXT:
{marked}"""

_REPLY = re.compile(r"^[ \t]*\[#(\d+)\][ \t]*(.+?)\s*(?=^[ \t]*\[#\d+\]|\Z)", re.M | re.S)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of each sentence."""
    return [(s.start_char, s.end_char) for s in sentences(analyze(text))]


def contributions(text: str, spans: list[tuple[int, int]], logit_fn) -> list[float]:
    """Per sentence: how many logit points the detector's score falls when the sentence is deleted (positive = pushes toward AI)."""
    variants = [(text[:a].rstrip() + " " + text[b:].lstrip()).strip() for a, b in spans]
    scores = logit_fn([text] + variants)
    return [float(scores[0] - s) for s in scores[1:]]


def pick_targets(contribs: list[float], eligible: list[bool], frac: float = 0.3, minimum: int = 2) -> list[int]:
    """Indices of the sentences to rewrite this pass: the biggest contributors among the eligible ones."""
    pool = [i for i, ok in enumerate(eligible) if ok]
    k = min(len(pool), max(minimum, math.ceil(frac * len(contribs))))
    return sorted(sorted(pool, key=lambda i: (-contribs[i], i))[:k])


def mark(text: str, spans: list[tuple[int, int]], targets: list[int]) -> str:
    """The text with each target sentence wrapped as [#n]...[/#n] (n counts from 1)."""
    out, pos = [], 0
    for i in targets:
        a, b = spans[i]
        out += [text[pos:a], f"[#{i + 1}]", text[a:b], f"[/#{i + 1}]"]
        pos = b
    return "".join(out) + text[pos:]


def parse_replies(raw: str, targets: list[int]) -> dict[int, str]:
    """{sentence index: replacement} for the targets the model answered; anything else in the reply is ignored."""
    raw = raw.split("</think>")[-1]
    wanted, found = {t + 1 for t in targets}, {}
    for num, body in _REPLY.findall(raw):
        text = normalize_text(re.sub(r"\s+", " ", body)).strip()
        text = re.sub(r"\[/?#\d+\]", "", text).strip()
        if int(num) in wanted and text:
            found[int(num) - 1] = text
    return found


def splice(text: str, spans: list[tuple[int, int]], replacements: dict[int, str]) -> str:
    """Put each replacement at its sentence's exact position; everything between sentences is kept as it was."""
    for i in sorted(replacements, reverse=True):
        a, b = spans[i]
        text = text[:a] + replacements[i] + text[b:]
    return text


def run_surgery(
    detector,
    humanizer,
    checker,
    text: str,
    genre: str,
    *,
    our_cutoff: float,
    ext_cutoff: float,
    external,
    target: float | None = 0.30,
    judge=None,
    judge_cutoff: float | None = None,
    max_iterations: int = 4,
    frac: float = 0.3,
    item_id: str = "",
) -> LoopResult:
    """Run surgery passes on one text. `target` is the P(AI) both steering detectors must get under (or their own frozen cutoff if lower)."""
    ours_goal = min(our_cutoff, target) if target is not None else our_cutoff
    ext_goal = min(ext_cutoff, target) if target is not None else ext_cutoff

    def ext_score(t: str) -> float:
        return float(external.predict_proba([t])[0])

    def judge_score(t: str) -> float | None:
        return None if judge is None else float(judge.predict_proba([t])[0])

    base = detector.analyze(text)
    original = base.text
    steps = [Step(0, original, base.prob_ai, ext_score(original), judge=judge_score(original))]
    current, spent, stop = original, [], "max_iterations"  # `spent`: texts of sentences already attacked (or whose attack was rejected)

    for i in range(1, max_iterations + 1):
        spans = sentence_spans(current)
        eligible = [not any(current[a:b].strip() in s for s in spent) for a, b in spans]
        if not any(eligible):
            stop = "exhausted"
            break
        targets = pick_targets(contributions(current, spans, external.logits), eligible, frac)
        prompt = USER.format(genre=genre, marked=mark(current, spans, targets))
        try:
            resp = humanizer.client.complete(system=SYSTEM, prompt=prompt, model=humanizer.model,
                                             max_tokens=min(900, 60 * len(targets) + 120), effort=humanizer.effort)  # fmt: skip
        except RateLimited:
            raise
        except RuntimeError as err:
            stop = f"error: {str(err)[:120]}"
            break
        replies = parse_replies(resp.text, targets)
        if not replies:
            stop = "error: no usable replacement in the reply"
            break
        candidate = splice(current, spans, replies)
        spent += [current[spans[t][0]:spans[t][1]].strip() for t in targets] + list(replies.values())
        if candidate == current:
            stop = "stalled"
            break
        analysis = detector.analyze(candidate)
        guard = checker.check(original, analysis.text)
        steps.append(Step(i, analysis.text, analysis.prob_ai, ext_score(analysis.text), guard.ok, list(guard.flags), guard.similarity,
                          guard.length_ratio, guard.fk_grade_after, resp.input_tokens, resp.output_tokens, resp.cached, "",
                          judge_score(analysis.text)))  # fmt: skip
        steps[0].fk_grade = steps[0].fk_grade if steps[0].fk_grade is not None else guard.fk_grade_before
        if guard.ok:
            current = analysis.text
            if analysis.prob_ai <= ours_goal and steps[-1].external <= ext_goal:
                stop = "success"
                break
        # a rejected edit is simply dropped: the next pass starts again from the last accepted text, on different sentences

    return LoopResult(item_id, genre, "surgery", our_cutoff, ext_cutoff, steps,
                      choose_final(steps, ours_goal, ext_goal, steer_external=True), stop, getattr(humanizer, "model", ""),
                      SURGERY_VERSION, False, judge_cutoff, ["ours", "external"], getattr(humanizer, "strength", "faithful"), target)  # fmt: skip
