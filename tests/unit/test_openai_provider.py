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
        llm_model="demo-model",
        **kw,
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
