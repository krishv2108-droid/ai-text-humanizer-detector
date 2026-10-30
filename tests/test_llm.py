import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from hd.data.generate import generate_stories
from hd.humanizer.rewrite import clean_output
from hd.llm import (
    DailyLimitReached,
    HeaderPacer,
    LLMResponse,
    MockClient,
    OpenAICompatClient,
    RateLimited,
    api_key,
    classify_rate_limit,
    estimate_tokens,
    make_client,
    parse_duration,
)

# ---- durations and rate-limit messages ---------------------------------------------------


@pytest.mark.parametrize("text,seconds", [("7.66s", 7.66), ("2m59.56s", 179.56), ("250ms", 0.25), ("1h2m3s", 3723.0), ("45", None), ("", None), (None, None)])
def test_parse_duration(text, seconds):
    got = parse_duration(text)
    assert (got is None) if seconds is None else got == pytest.approx(seconds)


def test_classify_daily_vs_minute_limits():
    daily = classify_rate_limit("Rate limit reached for model `x` on tokens per day (TPD): Limit 100000. Please try again in 13m54.2s.")
    minute = classify_rate_limit("Rate limit reached for model `x` on tokens per minute (TPM): Limit 6000. Please try again in 7.5s.")
    assert isinstance(daily, DailyLimitReached) and daily.retry_in_s == pytest.approx(834.2)
    assert isinstance(minute, RateLimited) and not isinstance(minute, DailyLimitReached) and minute.retry_in_s == pytest.approx(7.5)
    assert isinstance(classify_rate_limit("requests per day (RPD) exceeded"), DailyLimitReached)


def test_estimate_tokens_is_roughly_chars_over_three_and_a_half():
    assert 280 <= estimate_tokens("x" * 1000) <= 290


# ---- pacing from headers -----------------------------------------------------------------


class Clock:
    def __init__(self):
        self.now, self.slept = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.slept.append(round(s, 3))
        self.now += s


def pacer(min_interval=2.0):
    c = Clock()
    return HeaderPacer(min_interval, sleep=c.sleep, clock=c), c


def test_pacer_spaces_requests_but_not_the_first():
    p, c = pacer(2.0)
    assert p.before(100) == 0 and c.slept == []
    c.now += 0.5
    assert p.before(100) == pytest.approx(1.5)  # only the remainder of the 2 s interval
    c.now += 5
    assert p.before(100) == 0


def test_pacer_waits_for_the_token_reset_only_when_the_request_will_not_fit():
    p, c = pacer(0.0)
    p.before(10)
    p.after({"x-ratelimit-remaining-tokens": "500", "x-ratelimit-limit-tokens": "6000", "x-ratelimit-reset-tokens": "12.5s"})
    assert p.before(300) == 0  # fits in what is left: no wait
    p.after({"x-ratelimit-remaining-tokens": "500", "x-ratelimit-limit-tokens": "6000", "x-ratelimit-reset-tokens": "12.5s"})
    waited = p.before(2000)  # does not fit: wait out the reset
    assert waited == pytest.approx(12.5)
    p.after({"x-ratelimit-remaining-tokens": "5000", "x-ratelimit-limit-tokens": "6000", "x-ratelimit-reset-tokens": "1s"})
    assert p.before(2000) == 0


def test_pacer_never_waits_for_a_request_bigger_than_the_whole_budget():
    p, c = pacer(0.0)
    p.after({"x-ratelimit-remaining-tokens": "10", "x-ratelimit-limit-tokens": "6000", "x-ratelimit-reset-tokens": "30s"})
    assert p.before(9000) == 0  # waiting could never help; let the API answer


def test_pacer_tolerates_missing_or_odd_headers():
    p, c = pacer(0.0)
    p.after(None)
    p.after({"x-ratelimit-remaining-tokens": "not a number"})
    assert p.before(100) == 0


# ---- the Groq client with a fake transport -----------------------------------------------


def fake_response(text="Once upon a time.", finish="stop", headers=None):
    msg = SimpleNamespace(content=text)
    parsed = SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)], usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7), model="m")
    return SimpleNamespace(headers=headers or {}, parse=lambda: parsed)


class FakeTransport:
    def __init__(self, result):
        self.result, self.calls = result, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=self._create)))

    def _create(self, **kw):
        self.calls.append(kw)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def groq(result, **kw):
    t = FakeTransport(result)
    return OpenAICompatClient("http://x", "k", seed=42, pacer=HeaderPacer(0.0), _client=t, **kw), t


def test_plain_model_request_shape():
    client, t = groq(fake_response())
    out = client.complete(system="SYS", prompt="P", model="llama-3.3-70b-versatile", max_tokens=300, effort="medium")
    sent = t.calls[0]
    assert sent["model"] == "llama-3.3-70b-versatile" and sent["max_completion_tokens"] == 300 and sent["seed"] == 42
    assert sent["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "P"}]
    assert "extra_body" not in sent and "temperature" not in sent and "max_tokens" not in sent  # effort ignored off reasoning models
    assert out == LLMResponse("Once upon a time.", 11, 7, "end_turn")


def test_reasoning_model_gets_effort_hidden_reasoning_and_token_headroom():
    client, t = groq(fake_response(), reasoning_headroom=1500)
    client.complete(system="s", prompt="p", model="openai/gpt-oss-120b", max_tokens=500, effort="high")
    sent = t.calls[0]
    assert sent["max_completion_tokens"] == 2000
    assert sent["extra_body"] == {"reasoning_effort": "high", "include_reasoning": False}
    client.complete(system="s", prompt="p", model="openai/gpt-oss-120b", max_tokens=500, effort="xhigh")  # not valid on Groq
    assert t.calls[1]["extra_body"]["reasoning_effort"] == "low"


def test_finish_reason_length_means_truncated():
    client, _ = groq(fake_response(finish="length"))
    assert client.complete(system="s", prompt="p", model="m", max_tokens=5).stop_reason == "max_tokens"
    client, _ = groq(fake_response(text=None))
    assert client.complete(system="s", prompt="p", model="m", max_tokens=5).text == ""


def _status_error(cls, status, message):
    resp = httpx.Response(status, request=httpx.Request("POST", "http://x"), json={"error": {"message": message}})
    return cls(message, response=resp, body={"error": {"message": message}})


def test_errors_become_clear_exceptions():
    client, _ = groq(_status_error(openai.RateLimitError, 429, "Rate limit reached ... on tokens per day (TPD) ... try again in 2h"))
    with pytest.raises(DailyLimitReached) as e:
        client.complete(system="s", prompt="p", model="m", max_tokens=5)
    assert e.value.retry_in_s == pytest.approx(7200)
    client, _ = groq(_status_error(openai.AuthenticationError, 401, "Invalid API Key"))
    with pytest.raises(RuntimeError, match="rejected the API key"):
        client.complete(system="s", prompt="p", model="m", max_tokens=5)
    client, _ = groq(_status_error(openai.NotFoundError, 404, "model does not exist"))
    with pytest.raises(RuntimeError, match="not available to this key"):
        client.complete(system="s", prompt="p", model="nope", max_tokens=5)


def test_pacer_sees_each_responses_headers():
    hdr = {"x-ratelimit-remaining-tokens": "100", "x-ratelimit-limit-tokens": "6000", "x-ratelimit-reset-tokens": "9s"}
    c = Clock()
    t = FakeTransport(fake_response(headers=hdr))
    client = OpenAICompatClient("http://x", "k", pacer=HeaderPacer(0.0, sleep=c.sleep, clock=c), _client=t)
    client.complete(system="s", prompt="p", model="m", max_tokens=50)  # leaves 100 tokens in the minute
    client.complete(system="s", prompt="p" * 4000, model="m", max_tokens=800)  # needs ~2000: must wait
    assert c.slept and c.slept[0] == pytest.approx(9.0)


def test_probe_reports_account_limits():
    hdr = {"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "999", "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "7990"}
    client, _ = groq(fake_response(text="OK", headers=hdr))
    info = client.probe("llama-3.3-70b-versatile")
    assert info["limit_requests_per_day"] == "1000" and info["limit_tokens_per_minute"] == "8000" and info["reply"] == "OK"


# ---- keys and provider selection ---------------------------------------------------------


def test_missing_key_message_names_the_right_variable(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("hd.config.PROJECT_ROOT", tmp_path)  # no .env here
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        api_key("groq")
    with pytest.raises(RuntimeError, match="unknown llm.provider"):
        api_key("nope")


def test_make_client_builds_cached_groq_client(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    cfg = {"seed": 1, "llm": {"provider": "groq", "max_retries": 2, "cache": True, "min_interval_s": 0.0}}
    client = make_client(cfg, tmp_path)
    assert type(client).__name__ == "CachedClient" and isinstance(client.inner, OpenAICompatClient)
    cfg["llm"]["cache"] = False
    assert isinstance(make_client(cfg, tmp_path), OpenAICompatClient)


# ---- cleaning titles/prefaces, and a generator that stops cleanly ------------------------


def test_title_lines_are_stripped_but_real_openings_are_not():
    body = "She walked home in the rain. It was cold and late."
    assert clean_output(f"The Lighthouse Keeper\n\n{body}")[0] == body
    assert clean_output(f'"The Lighthouse Keeper"\n\n{body}')[0] == body
    assert "removed a title line" in clean_output(f"**Rain**\n\n{body}")[1]
    kept = clean_output(f"Nothing\n\n{body}", draft=f"Nothing\n\n{body}")[0]  # the draft itself opened with that line
    assert kept.startswith("Nothing")
    assert clean_output(f"She said no, twice.\n\n{body}")[0].startswith("She said no")  # has punctuation: a sentence
    assert clean_output(body)[0] == body


def _prompts(path, n):
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"prompt_id": f"p{i}", "prompt": f"prompt {i}", "target_words": 100}) + "\n")


class FlakyClient:
    """Succeeds `ok` times, then hits a daily limit."""

    def __init__(self, ok):
        self.ok, self.calls, self.max_tokens_seen = ok, 0, []

    def complete(self, **kw):
        self.calls += 1
        self.max_tokens_seen.append(kw["max_tokens"])
        if self.calls > self.ok:
            raise DailyLimitReached("on tokens per day (TPD) ... try again in 30m", 1800)
        return LLMResponse("Title Here\n\n" + "She walked home slowly in the rain. " * 12)


def test_generator_stops_at_the_daily_limit_and_resumes(tmp_path):
    prompts, out = tmp_path / "p.jsonl", tmp_path / "ai.jsonl"
    _prompts(prompts, 5)
    kw = dict(model="m", max_tokens=2000, effort=None)
    client = FlakyClient(ok=2)
    stats = generate_stories(client, prompts, out, **kw)
    assert stats["written"] == 2 and stats["stopped_by_limit"] and "daily limit" in stats["message"] and "30 min" in stats["message"]
    assert client.max_tokens_seen[0] == int(100 * 1.8) + 150  # budget derived from the target length
    rows = [json.loads(line) for line in open(out)]
    assert len(rows) == 2 and not rows[0]["text"].startswith("Title Here")  # the title line was stripped

    later = generate_stories(MockClient(lambda p: "She walked home slowly in the rain. " * 12), prompts, out, **kw)
    assert later["skipped_done"] == 2 and later["written"] == 3 and not later["stopped_by_limit"]
    assert len({json.loads(line)["prompt_id"] for line in open(out)}) == 5


# ---- several keys for the same provider and model --------------------------------------------------------------------


class Scripted:
    """A fake per-key client: replays a script of responses/exceptions (the last item repeats) and counts its calls."""

    def __init__(self, *script):
        self.script, self.calls = list(script), 0

    def complete(self, **kwargs):
        self.calls += 1
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item)


class TickClock:
    now = 1000.0

    def __call__(self):
        return self.now


CALL = dict(system="s", prompt="p", model="m", max_tokens=10)


def test_rotation_moves_to_the_next_key_when_one_runs_out_and_returns_after_the_retry_time():
    from hd.llm import DailyLimitReached, RotatingClient

    clock = TickClock()
    a = Scripted(DailyLimitReached("tokens per day", 600.0), "from-a")  # runs out once, then works again
    b = Scripted("from-b")
    client = RotatingClient([a, b], clock=clock)
    assert client.complete(**CALL).text == "from-b" and (a.calls, b.calls) == (1, 1)  # a failed, b answered
    assert client.complete(**CALL).text == "from-b" and (a.calls, b.calls) == (1, 2)  # a is skipped, not hammered
    clock.now += 601  # a's retry time has passed
    assert client.complete(**CALL).text == "from-b" and a.calls == 1  # b stays current until it fails
    b.script = [DailyLimitReached("tokens per day", 600.0)]
    assert client.complete(**CALL).text == "from-a" and a.calls == 2  # a is usable again


def test_when_every_key_is_limited_the_error_carries_the_soonest_retry_time():
    from hd.llm import DailyLimitReached, RateLimited, RotatingClient

    clock = TickClock()
    client = RotatingClient([Scripted(DailyLimitReached("tokens per day", 900.0)), Scripted(DailyLimitReached("tokens per day", 300.0))], clock=clock)
    with pytest.raises(DailyLimitReached) as err:
        client.complete(**CALL)
    assert err.value.retry_in_s == 300.0 and "2 API key" in str(err.value)
    mixed = RotatingClient([Scripted(DailyLimitReached("per day", 900.0)), Scripted(RateLimited("per minute", 20.0))], clock=clock)
    with pytest.raises(RateLimited) as err2:
        mixed.complete(**CALL)
    assert not isinstance(err2.value, DailyLimitReached) and err2.value.retry_in_s == 20.0  # a per-minute limit is not reported as a daily one


def test_a_per_minute_limit_also_rotates():
    from hd.llm import RateLimited, RotatingClient

    a, b = Scripted(RateLimited("per minute", 20.0)), Scripted("from-b")
    assert RotatingClient([a, b], clock=TickClock()).complete(**CALL).text == "from-b"


def test_an_invalid_key_is_reported_by_position_not_skipped_and_refusals_are_not_retried():
    from hd.llm import RefusalError, RotatingClient

    bad, good = Scripted(RuntimeError("the provider rejected the API key")), Scripted("ok")
    with pytest.raises(RuntimeError, match="key 1 of 2: the provider rejected"):
        RotatingClient([bad, good], clock=TickClock()).complete(**CALL)
    assert good.calls == 0  # silently using the other key would hide a broken one
    refuser, other = Scripted(RefusalError("declined")), Scripted("ok")
    with pytest.raises(RefusalError):
        RotatingClient([refuser, other], clock=TickClock()).complete(**CALL)
    assert other.calls == 0  # a refusal is about the request, not the key


def test_api_keys_are_collected_in_order_without_blanks_or_duplicates(monkeypatch):
    from hd.llm import api_keys

    monkeypatch.setattr("hd.config.load_env", lambda *a, **k: False)  # never read the real .env in tests
    for name in ("GROQ_API_KEY",) + tuple(f"GROQ_API_KEY_{n}" for n in range(2, 10)):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "aaa")
    monkeypatch.setenv("GROQ_API_KEY_2", "bbb")
    monkeypatch.setenv("GROQ_API_KEY_3", "   ")
    monkeypatch.setenv("GROQ_API_KEY_4", "aaa")
    monkeypatch.setenv("GROQ_API_KEY_5", " ccc ")
    assert api_keys("groq") == ["aaa", "bbb", "ccc"]
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        api_keys("groq")  # a second key alone is not enough: the primary one is required


def test_make_client_rotates_only_when_there_is_more_than_one_key_and_caches_independently_of_the_key(monkeypatch, tmp_path):
    from hd.llm import CachedClient, OpenAICompatClient, RotatingClient, make_client

    monkeypatch.setattr("hd.config.load_env", lambda *a, **k: False)
    monkeypatch.setenv("GROQ_API_KEY", "key-one")
    monkeypatch.delenv("GROQ_API_KEY_2", raising=False)
    cfg = {"seed": 1, "llm": {"provider": "groq", "max_retries": 1, "min_interval_s": 0.0, "reasoning_headroom": 10, "cache": False}}
    assert isinstance(make_client(cfg, tmp_path), OpenAICompatClient)
    monkeypatch.setenv("GROQ_API_KEY_2", "key-two")
    rotating = make_client(cfg, tmp_path)
    assert isinstance(rotating, RotatingClient) and len(rotating.clients) == 2
    assert len({id(c.pacer) for c in rotating.clients}) == 2  # each account is paced against its own limits
    cached = make_client({**cfg, "llm": {**cfg["llm"], "cache": True}}, tmp_path)
    assert isinstance(cached, CachedClient)


def test_a_cached_answer_is_reused_whichever_key_produced_it(tmp_path):
    from hd.llm import CachedClient, DailyLimitReached, RotatingClient

    a, b = Scripted(DailyLimitReached("tokens per day", 600.0)), Scripted("story")
    client = CachedClient(RotatingClient([a, b], clock=TickClock()), tmp_path)
    assert client.complete(**CALL).text == "story" and b.calls == 1
    assert client.complete(**CALL).text == "story" and (a.calls, b.calls) == (1, 1)  # served from the cache: no key was touched
