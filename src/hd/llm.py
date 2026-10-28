"""Thin LLM client layer: Groq (free tier) or Anthropic, an on-disk cache, and a mock for offline tests.

Every call is cached by its full parameters, so re-running a pipeline costs nothing and a crash or a
rate-limit stop halfway through a batch loses nothing. No sampling parameters (temperature, top_p) are
exposed: some providers' current models reject non-default values, and the cache key should not depend
on them. Groq's `seed` is passed for best-effort reproducibility of uncached calls.

Free tiers have small per-minute and per-day limits. `OpenAICompatClient` therefore (1) paces itself from
the provider's `x-ratelimit-*` response headers so it rarely hits a per-minute limit, and (2) turns a daily
limit into `DailyLimitReached`, which batch jobs catch to stop cleanly and resume later.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

PROVIDERS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "anthropic": {"base_url": None, "key_env": "ANTHROPIC_API_KEY"},
}


class RefusalError(RuntimeError):
    """The model declined the request (stop_reason == "refusal")."""


class RateLimited(RuntimeError):
    """A rate limit was hit and the client's own retries did not clear it."""

    def __init__(self, message: str, retry_in_s: float | None = None):
        super().__init__(message)
        self.retry_in_s = retry_in_s


class DailyLimitReached(RateLimited):
    """A per-day request or token limit is exhausted: stop now, resume later (work is cached)."""


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


# ---------------------------------------------------------------------------------------------
# Rate-limit helpers
# ---------------------------------------------------------------------------------------------

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h|d)")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(text: str | None) -> float | None:
    """'2m59.56s' -> 179.56, '7.66s' -> 7.66, '250ms' -> 0.25; None if nothing parses."""
    if not text:
        return None
    parts = _DURATION.findall(text.strip())
    if not parts:
        return None
    return sum(float(v) * _UNIT_S[u] for v, u in parts)


def estimate_tokens(text: str) -> int:
    """Rough token count (about 3.5 characters per token for English)."""
    return int(len(text) / 3.5) + 1


class HeaderPacer:
    """Sleep only when needed, using the provider's rate-limit headers from the previous response.

    `x-ratelimit-remaining-tokens` / `x-ratelimit-reset-tokens` describe the per-minute token budget.
    If the next request would not fit in what is left, wait for the reset instead of earning a 429.
    `min_interval_s` additionally spaces requests (a guard for the per-minute request limit).
    """

    def __init__(self, min_interval_s: float = 2.0, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):  # fmt: skip
        self.min_interval_s, self._sleep, self._clock = min_interval_s, sleep, clock
        self._last_request = -1e9
        self._remaining_tokens: float | None = None
        self._limit_tokens: float | None = None
        self._reset_at = 0.0
        self.slept_s = 0.0

    def before(self, est_tokens: int) -> float:
        now = self._clock()
        wait = max(0.0, self._last_request + self.min_interval_s - now)
        known = self._remaining_tokens is not None and self._limit_tokens is not None
        # Only wait if waiting can help: a request bigger than the whole per-minute budget never fits.
        if known and est_tokens <= self._limit_tokens and self._remaining_tokens < est_tokens:  # type: ignore[operator]
            wait = max(wait, self._reset_at - now)
        if wait > 0:
            self._sleep(wait)
            self.slept_s += wait
        self._last_request = self._clock()
        return wait

    def after(self, headers: Any) -> None:
        def get(name: str) -> str | None:
            value = headers.get(name) if headers is not None else None
            return None if value is None else str(value)

        try:
            remaining, limit = get("x-ratelimit-remaining-tokens"), get("x-ratelimit-limit-tokens")
            self._remaining_tokens = float(remaining) if remaining is not None else None
            self._limit_tokens = float(limit) if limit is not None else None
        except ValueError:
            self._remaining_tokens = self._limit_tokens = None
        reset = parse_duration(get("x-ratelimit-reset-tokens"))
        self._reset_at = self._clock() + (reset if reset is not None else 0.0)


def classify_rate_limit(message: str) -> RateLimited:
    """Groq's 429 text names the exhausted limit, e.g. '... on tokens per day (TPD) ... try again in 13m54s'."""
    retry = re.search(r"try again in ([0-9hmsd.]+)", message)
    retry_in = parse_duration(retry.group(1)) if retry else None
    lowered = message.lower()
    if "per day" in lowered or "(tpd)" in lowered or "(rpd)" in lowered:
        return DailyLimitReached(message, retry_in)
    return RateLimited(message, retry_in)


# ---------------------------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------------------------


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


_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens"}
_REASONING_EFFORTS = {"low", "medium", "high"}


class OpenAICompatClient:
    """Chat-completions client for OpenAI-compatible endpoints (used for Groq's free tier)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        max_retries: int = 6,
        seed: int | None = None,
        pacer: HeaderPacer | None = None,
        reasoning_headroom: int = 1500,
        _client: Any = None,
    ):
        import openai

        self._openai = openai
        # The SDK already retries 429/5xx with `retry-after`; the pacer makes 429s rare to begin with.
        self._client = _client or openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.seed, self.pacer, self.reasoning_headroom = seed, pacer or HeaderPacer(), reasoning_headroom

    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        return model.startswith("openai/gpt-oss")

    def _request(self, system: str, prompt: str, model: str, max_tokens: int, effort: str | None) -> dict[str, Any]:
        reasoning = self.is_reasoning_model(model)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            # reasoning models spend hidden tokens that count toward the completion budget
            "max_completion_tokens": max_tokens + (self.reasoning_headroom if reasoning else 0),
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if reasoning:
            kwargs["extra_body"] = {
                "reasoning_effort": effort if effort in _REASONING_EFFORTS else "low",
                "include_reasoning": False,
            }
        return kwargs

    def _call(self, kwargs: dict[str, Any], est_tokens: int):
        self.pacer.before(est_tokens)
        openai = self._openai
        try:
            raw = self._client.chat.completions.with_raw_response.create(**kwargs)
        except openai.RateLimitError as err:
            raise classify_rate_limit(getattr(err, "message", None) or str(err)) from err
        except openai.AuthenticationError as err:
            raise RuntimeError("the provider rejected the API key (check it in .env and that it is active)") from err
        except (openai.NotFoundError, openai.PermissionDeniedError) as err:
            raise RuntimeError(f"model '{kwargs['model']}' is not available to this key: {getattr(err, 'message', err)}") from err
        except openai.APIConnectionError as err:
            raise RuntimeError(f"could not reach the provider: {err}") from err
        self.pacer.after(raw.headers)
        return raw

    def complete(self, *, system, prompt, model, max_tokens, effort=None) -> LLMResponse:
        kwargs = self._request(system, prompt, model, max_tokens, effort)
        est = estimate_tokens(system + prompt) + kwargs["max_completion_tokens"]
        resp = self._call(kwargs, est).parse()
        choice = resp.choices[0]
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            stop_reason=_STOP_REASONS.get(choice.finish_reason, choice.finish_reason or "end_turn"),
        )

    def probe(self, model: str) -> dict[str, Any]:
        """One tiny request that reports the account's rate-limit headers for `model` (for `hd llm check`)."""
        kwargs = self._request("Reply with the single word OK.", "Say OK.", model, 8, "low")
        started = time.monotonic()
        raw = self._call(kwargs, 40)
        resp = raw.parse()
        h = raw.headers
        return {
            "model": getattr(resp, "model", model),
            "seconds": round(time.monotonic() - started, 2),
            "limit_requests_per_day": h.get("x-ratelimit-limit-requests"),
            "remaining_requests": h.get("x-ratelimit-remaining-requests"),
            "limit_tokens_per_minute": h.get("x-ratelimit-limit-tokens"),
            "remaining_tokens": h.get("x-ratelimit-remaining-tokens"),
            "reply": (resp.choices[0].message.content or "").strip()[:40],
        }


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


def api_key(provider: str) -> str:
    """The provider's key from the environment or the git-ignored `.env` file; clear error if missing."""
    import os

    from hd.config import load_env

    if provider not in PROVIDERS:
        raise RuntimeError(f"unknown llm.provider {provider!r}; choose one of {sorted(PROVIDERS)}")
    load_env()
    env = PROVIDERS[provider]["key_env"]
    key = os.environ.get(env, "").strip()
    if not key:
        raise RuntimeError(
            f"{env} is not set. Either export it in your shell, or put this line in a file named .env in "
            f"the project root (git-ignored, never committed):\n  {env}=your-key-here"
        )
    return key


def make_client(cfg: dict, cache_dir: Path) -> LLMClient:
    """The configured provider's client, wrapped in the disk cache. Fails early if no key is set."""
    llm = cfg["llm"]
    provider = llm.get("provider", "anthropic")
    key = api_key(provider)  # validates provider + key before anything is built
    if provider == "anthropic":
        inner: LLMClient = AnthropicClient(max_retries=llm["max_retries"])
    else:
        inner = OpenAICompatClient(
            PROVIDERS[provider]["base_url"], key,
            max_retries=llm["max_retries"], seed=cfg.get("seed"),
            pacer=HeaderPacer(llm.get("min_interval_s", 2.0)),
            reasoning_headroom=llm.get("reasoning_headroom", 1500),
        )  # fmt: skip
    return CachedClient(inner, cache_dir) if llm["cache"] else inner
