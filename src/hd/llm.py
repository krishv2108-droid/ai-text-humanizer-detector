"""Thin LLM client layer: real Anthropic client, on-disk cache, and a mock for offline tests.

Every call is cached by its full parameters, so re-running a pipeline costs nothing and a crash
halfway through a batch loses nothing. Note: current Claude models reject non-default sampling
parameters (temperature/top_p/top_k), so none are exposed here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol


class RefusalError(RuntimeError):
    """The model declined the request (stop_reason == "refusal")."""


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = "end_turn"
    cached: bool = False


class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        prompt: str,
        model: str,
        max_tokens: int,
        effort: str | None = None,
    ) -> LLMResponse: ...


class AnthropicClient:
    def __init__(self, max_retries: int = 5):
        import anthropic

        self._anthropic = anthropic
        # The SDK retries connection errors, 408/409/429 and 5xx with exponential backoff.
        self._client = anthropic.Anthropic(max_retries=max_retries)

    def complete(self, *, system, prompt, model, max_tokens, effort=None) -> LLMResponse:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            # Plain creative writing needs no reasoning pass; leaving it off keeps output cheap.
            "thinking": {"type": "disabled"},
        }
        if effort:
            kwargs["output_config"] = {"effort": effort}
        response = self._client.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            raise RefusalError(str(getattr(response, "stop_details", "")))
        text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )


class CachedClient:
    def __init__(self, inner: LLMClient, cache_dir: Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, *, system, prompt, model, max_tokens, effort=None) -> LLMResponse:
        key_src = json.dumps(
            [system, prompt, model, max_tokens, effort], ensure_ascii=False, sort_keys=True
        )
        path = self.cache_dir / f"{hashlib.sha256(key_src.encode()).hexdigest()}.json"
        if path.exists():
            data = json.loads(path.read_text())
            data["cached"] = True
            return LLMResponse(**data)
        response = self.inner.complete(
            system=system, prompt=prompt, model=model, max_tokens=max_tokens, effort=effort
        )
        path.write_text(json.dumps(asdict(response), ensure_ascii=False))
        return response


class MockClient:
    """Deterministic offline stand-in. `responder(prompt) -> text`, default echoes the prompt."""

    def __init__(self, responder: Callable[[str], str] | None = None):
        self.responder = responder or (lambda p: f"[mock] {p}")
        self.calls = 0

    def complete(self, *, system, prompt, model, max_tokens, effort=None) -> LLMResponse:
        self.calls += 1
        text = self.responder(prompt)
        return LLMResponse(text=text, input_tokens=len(prompt.split()), output_tokens=len(text.split()))


def make_client(cfg: dict, cache_dir: Path) -> LLMClient:
    """Real client wrapped in the disk cache. Fails early with a clear message if no key is set."""
    import os

    from hd.config import load_env

    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Either export it in your shell, or put this line in a file "
            "named .env in the project root (git-ignored, never committed):\n"
            "  ANTHROPIC_API_KEY=your-key-here"
        )
    inner = AnthropicClient(max_retries=cfg["llm"]["max_retries"])
    return CachedClient(inner, cache_dir) if cfg["llm"]["cache"] else inner
