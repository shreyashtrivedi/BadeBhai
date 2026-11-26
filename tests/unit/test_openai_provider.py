"""OpenAI-compatible provider (Gemini / Groq / OpenRouter / LM Studio / Ollama /v1), against a mock server."""

import json

import httpx
import pytest
from pydantic import BaseModel

from badebhai.config import Settings
from badebhai.llm import LLMUnavailable, track_usage
from badebhai.llm.openai_provider import OpenAICompatProvider


class Verdict(BaseModel):
    answer: str
    confident: bool


def _provider(handler, **kw) -> OpenAICompatProvider:
    s = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_base_url="https://api.example.test/v1",
        openai_api_key="sk-test-key",
        **{"llm_model": "demo-model", **kw},
    )
    return OpenAICompatProvider(s, transport=httpx.MockTransport(handler))


def _reply(content: str, usage=None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": usage or {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )


async def test_complete_sends_key_and_records_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return _reply("<think>hmm</think>GET")

    with track_usage() as u:
        text = await _provider(handler, openai_free=True).complete(
            purpose="t", system="sys", messages=[{"role": "user", "content": "hi"}]
        )
    assert text == "GET" and seen["auth"] == "Bearer sk-test-key"
    assert seen["url"] == "https://api.example.test/v1/chat/completions"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert (
        u.summary()["input_tokens"] == 7 and u.summary()["cost_usd"] == 0.0 and u.summary()["cost_complete"]
    )


async def test_json_schema_then_fallback_to_json_object():
    formats = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        formats.append(body["response_format"]["type"])
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(
                400, json={"error": {"message": "response_format json_schema is not supported"}}
            )
        return _reply('```json\n{"answer": "yes", "confident": true}\n```')

    p = _provider(handler)
    out = await p.complete_json(
        purpose="j", system="s", messages=[{"role": "user", "content": "q"}], schema=Verdict
    )
    assert out == Verdict(answer="yes", confident=True) and formats == ["json_schema", "json_object"]
    await p.complete_json(
        purpose="j", system="s", messages=[{"role": "user", "content": "q"}], schema=Verdict
    )
    assert formats[-1] == "json_object"  # remembered


async def test_rate_limits_wait_for_retry_after(monkeypatch):
    calls, slept = [], []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("badebhai.llm.openai_provider.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(429, headers={"retry-after": "3"}, json={"error": "slow down"})
        return _reply("ok")

    assert await _provider(handler).complete(purpose="t", system="s", messages=[]) == "ok"
    assert len(calls) == 3 and slept == [3.0, 3.0]


async def test_streaming_parses_server_sent_events():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        chunks = [{"choices": [{"delta": {"content": c}}]} for c in ["Hel", "lo", "!"]]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    parts = [p async for p in _provider(handler).stream(purpose="s", system="s", messages=[])]
    assert "".join(parts) == "Hello!"


async def test_bad_key_and_missing_config_are_clear():
    with pytest.raises(LLMUnavailable, match="OPENAI_API_KEY"):
        await _provider(lambda r: httpx.Response(401, json={})).complete(purpose="t", system="s", messages=[])
    with pytest.raises(LLMUnavailable, match="LLM_MODEL"):
        OpenAICompatProvider(Settings(_env_file=None, llm_provider="openai", openai_base_url="http://x/v1"))


def test_openai_compatible_embedder_batches_normalizes_and_retries(monkeypatch):
    import numpy as np

    from badebhai.embeddings import OpenAICompatEmbedder

    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(len(body["input"]))
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "1"})
        assert body["dimensions"] == 3 and request.headers["authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [3.0, 4.0, 0.0]} for i in range(len(body["input"]))]},
        )

    emb = OpenAICompatEmbedder(
        "m", 3, "https://api.example.test/v1", "k", batch=2, transport=httpx.MockTransport(handler)
    )
    vecs = emb.embed_documents(["a", "b", "c"])
    assert vecs.shape == (3, 3) and np.allclose(vecs[0], [0.6, 0.8, 0.0])  # unit length
    assert calls == [2, 2, 1]  # first batch retried after a 429, then the last partial batch


async def test_overloaded_model_falls_back_to_another_flash_model(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("badebhai.llm.openai_provider.asyncio.sleep", no_sleep)
    used = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            ids = ["models/gemini-2.5-flash", "models/gemini-2.5-flash-lite", "models/gemini-3-flash-preview"]
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})
        model = json.loads(request.content)["model"]
        used.append(model)
        if model == "gemini-2.5-flash":
            return httpx.Response(503, json={"error": {"message": "high demand", "status": "UNAVAILABLE"}})
        return _reply("ok")

    p = _provider(handler, llm_model="gemini-2.5-flash")
    assert await p.complete(purpose="t", system="s", messages=[]) == "ok"
    assert used == ["gemini-2.5-flash", "gemini-3-flash-preview"]  # no slow retries of the busy model
    await p.complete(purpose="t", system="s", messages=[])
    assert used[-1] == "gemini-3-flash-preview"  # stays on the stand-in while the busy model is skipped


async def test_daily_quota_switches_model_but_per_minute_limit_waits(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("badebhai.llm.openai_provider.asyncio.sleep", no_sleep)
    used = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        used.append(model)
        if model == "main":
            msg = "Quota exceeded for metric: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
            return httpx.Response(429, json={"error": {"message": msg}})
        return _reply("ok")

    p = _provider(handler, llm_model="main", openai_fallback_models="backup-a, backup-b")
    assert await p.complete(purpose="t", system="s", messages=[]) == "ok"
    assert used == ["main", "backup-a"]


async def test_every_model_busy_reports_what_was_tried(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("badebhai.llm.openai_provider.asyncio.sleep", no_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(500, json={})
        return httpx.Response(503, json={"error": {"message": "high demand"}})

    p = _provider(handler, llm_model="main", openai_fallback_models="backup")
    with pytest.raises(Exception, match=r"kept refusing t \(main: 503; backup: 503; main: 503"):
        await p.complete(purpose="t", system="s", messages=[])


async def test_standin_without_free_quota_or_gone_is_skipped(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("badebhai.llm.openai_provider.asyncio.sleep", no_sleep)
    used = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        used.append(model)
        if model == "main":
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        if model == "paid-only":
            msg = "Quota exceeded for metric: generate_content_free_tier_requests, limit: 0, model: paid-only"
            return httpx.Response(429, json={"error": {"message": msg}})
        if model == "retired":
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return _reply("ok")

    p = _provider(handler, llm_model="main", openai_fallback_models="paid-only,retired,lite")
    assert await p.complete(purpose="t", system="s", messages=[]) == "ok"
    assert used == ["main", "paid-only", "retired", "lite"]


async def test_stream_on_overloaded_model_answers_from_a_standin(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("badebhai.llm.openai_provider.asyncio.sleep", no_sleep)
    used = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        used.append(model)
        if model == "main":
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return _reply("from backup")

    p = _provider(handler, llm_model="main", openai_fallback_models="backup")
    out = "".join([piece async for piece in p.stream(purpose="t", system="s", messages=[])])
    assert out == "from backup"
    assert used == ["main", "backup"]  # the busy model is not asked twice


async def test_hung_model_falls_back_to_a_standin(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("badebhai.llm.openai_provider.asyncio.sleep", no_sleep)
    used = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        used.append(model)
        if model == "main":
            raise httpx.ReadTimeout("hung", request=request)
        return _reply("from backup")

    p = _provider(handler, llm_model="main", openai_fallback_models="backup")
    assert await p.complete(purpose="t", system="s", messages=[]) == "from backup"
    out = "".join(
        [
            x
            async for x in _provider(handler, llm_model="main", openai_fallback_models="backup").stream(
                purpose="t", system="s", messages=[]
            )
        ]
    )
    assert out == "from backup"
    assert used == ["main", "backup", "main", "backup"]


async def test_brief_blips_on_every_model_still_get_an_answer(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr("badebhai.llm.openai_provider.asyncio.sleep", fake_sleep)
    used = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":  # the API also lists models that cannot serve: never use them
            return httpx.Response(200, json={"data": [{"id": "models/listed-but-overloaded"}]})
        model = json.loads(request.content)["model"]
        used.append(model)
        if len(used) <= 2:  # both models blip once
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return _reply("ok")

    p = _provider(handler, llm_model="main", openai_fallback_models="backup")
    assert await p.complete(purpose="t", system="s", messages=[]) == "ok"
    assert used == ["main", "backup", "main"]  # waited for main to come back instead of giving up
    assert "listed-but-overloaded" not in used and slept and max(slept) <= 8.0


async def test_retired_model_is_replaced_by_a_current_flash_model():
    used = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            ids = [
                "models/gemini-3.0-flash",
                "models/gemini-3.0-flash-lite",
                "models/gemini-embedding-001",
                "models/gemini-2.9-pro",
            ]
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})
        model = json.loads(request.content)["model"]
        used.append(model)
        if model == "gemini-2.5-flash":
            return httpx.Response(404, json={"error": {"message": "model not found"}})
        return _reply("ok")

    p = _provider(handler, llm_model="gemini-2.5-flash")
    assert await p.complete(purpose="t", system="s", messages=[]) == "ok"
    assert used == ["gemini-2.5-flash", "gemini-3.0-flash"]
    await p.complete(purpose="t", system="s", messages=[])
    assert used[-1] == "gemini-3.0-flash"  # remembered
