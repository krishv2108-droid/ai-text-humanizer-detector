"""The humanizer: ask an LLM to rewrite AI text, steered by what the detector found."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from hd.data.clean import has_markdown, normalize_text
from hd.humanizer.feedback import HumanProfile, diagnose, render_feedback
from hd.humanizer.prompts import STRATEGIES, build_prompt
from hd.llm import LLMClient


@dataclass
class RewriteResult:
    text: str
    strategy: str
    system: str
    user: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    warnings: list[str] = field(default_factory=list)


_PREFACE = re.compile(r"^\s*(here(?:'s| is)|sure|certainly|rewritten|revised|below is)[^\n]{0,80}:\s*\n+", re.I)
_FENCE = re.compile(r"^```[a-z]*\n|\n```\s*$")


def clean_output(raw: str) -> tuple[str, list[str]]:
    """Strip chatty wrapping the model sometimes adds, then normalize like every other text."""
    warnings, text = [], raw.strip()
    text = _FENCE.sub("", text)
    if _PREFACE.match(text):
        text = _PREFACE.sub("", text, count=1)
        warnings.append("removed a chatty preface")
    if len(text) > 1 and text[0] == text[-1] == '"' and text.count('"') == 2:
        text = text[1:-1]
    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3]
    text = normalize_text(text)
    if has_markdown(text):
        warnings.append("output contains markdown")
    return text, warnings


class Humanizer:
    def __init__(self, client: LLMClient, model: str, profile: HumanProfile | None = None, effort: str | None = None):
        self.client, self.model, self.profile, self.effort = client, model, profile, effort

    def build_request(self, draft: str, genre: str, strategy: str = "full", analysis=None, original: str | None = None) -> tuple[str, str]:
        """The exact (system, user) prompt that would be sent. Costs nothing: used by --dry-run and tests."""
        feedback_text = None
        if strategy in ("feedback", "full"):
            if analysis is None or self.profile is None:
                raise ValueError(f"strategy {strategy!r} needs a detector analysis and a human profile")
            feedback_text = render_feedback(diagnose(analysis, self.profile, genre), analysis)
        return build_prompt(draft, genre, strategy, feedback_text, original)

    def humanize(self, draft: str, genre: str, strategy: str = "full", analysis=None, original: str | None = None) -> RewriteResult:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {strategy!r}")
        system, user = self.build_request(draft, genre, strategy, analysis, original)
        words = len(draft.split())
        resp = self.client.complete(system=system, prompt=user, model=self.model,
                                    max_tokens=min(6000, int(words * 2.5) + 400), effort=self.effort)  # fmt: skip
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("the rewrite was cut off at the token limit; try again or shorten the text")
        text, warnings = clean_output(resp.text)
        if not text:
            raise RuntimeError("the model returned an empty rewrite")
        return RewriteResult(text, strategy, system, user, resp.input_tokens, resp.output_tokens, resp.cached, warnings)
