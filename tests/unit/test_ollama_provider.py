"""Ollama provider against a mocked Ollama server (no model needed)."""

import json

import httpx
import pytest
from pydantic import BaseModel

from repomind.config import Settings
from repomind.llm import LLMUnavailable, get_llm, track_usage
from repomind.llm.ollama_provider import OllamaProvider, _ThinkFilter, extract_json, strip_think

S = Settings(_env_file=None, llm_provider="ollama", ollama_model="qwen3:4b-instruct", ollama_num_ctx=8192)


class Plan(BaseModel):
    steps: list[str]
    done: bool


def provider(handler, settings=S):
    return OllamaProvider(settings, transport=httpx.MockTransport(handler))


def chat_reply(content, **extra):
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "done": True,
            "prompt_eval_count": 120,
            "eval_count": 30,
            **extra,
        },
    )


async def test_complete_sends_explicit_context_and_strips_thinking():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return chat_reply("<think>internal musing</think>The answer is 42.")

    with track_usage() as usage:
        out = await provider(handler).complete(
            purpose="naive", system="SYS", messages=[{"role": "user", "content": "q"}]
        )
    assert out == "The answer is 42."
    assert seen["model"] == "qwen3:4b-instruct" and seen["stream"] is False
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["options"]["num_ctx"] == 8192 and seen["think"] is False
    summary = usage.summary()
    assert summary["input_tokens"] == 120 and summary["cost_usd"] == 0.0 and summary["cost_complete"]


async def test_complete_json_uses_schema_format_and_retries_once():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            return chat_reply('{"steps": ["a"]}')  # missing "done"
        return chat_reply('Sure! {"steps": ["a", "b"], "done": true}')

    out = await provider(handler).complete_json(
        purpose="decompose", system="plan", messages=[{"role": "user", "content": "q"}], schema=Plan
    )
    assert out == Plan(steps=["a", "b"], done=True)
    assert calls[0]["format"]["required"] == ["steps", "done"]
    assert calls[0]["format"]["additionalProperties"] is False
    assert "JSON schema" in calls[0]["messages"][0]["content"]
    assert "did not match" in calls[1]["messages"][-1]["content"]


async def test_stream_yields_text_and_filters_split_think_tags():
    lines = [
        {"message": {"content": "<thi"}, "done": False},
        {"message": {"content": "nk>hidden</th"}, "done": False},
        {"message": {"content": "ink>Hello "}, "done": False},
        {"message": {"content": "world [1]."}, "done": False},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 50, "eval_count": 5},
    ]

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    with track_usage() as usage:
        chunks = [c async for c in provider(handler).stream(purpose="synthesize", system="s", messages=[])]
    assert "".join(chunks) == "Hello world [1]."
    assert usage.summary()["output_tokens"] == 5


async def test_stream_retries_when_the_runner_crashes_before_any_output():
    streamed = []

    def handler(request):
        body = json.loads(request.content)
        streamed.append(body["stream"])
        if body["stream"]:  # the model runner crashed (seen in practice as a CUDA error) and reloads
            return httpx.Response(500, json={"error": "CUDA error: unknown error"})
        return chat_reply("Recovered answer [1].")

    chunks = [c async for c in provider(handler).stream(purpose="synthesize", system="s", messages=[])]
    assert "".join(chunks) == "Recovered answer [1]." and streamed == [True, False]


async def test_missing_model_and_unreachable_server_give_actionable_errors():
    def missing(request):
        return httpx.Response(
            404, json={"error": "model 'qwen3:4b-instruct' not found, try pulling it first"}
        )

    with pytest.raises(LLMUnavailable, match="ollama pull qwen3:4b-instruct"):
        await provider(missing).complete(purpose="x", system="s", messages=[])

    def refused(request):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailable, match="docker compose up -d ollama"):
        await provider(refused).complete(purpose="x", system="s", messages=[])


async def test_retries_without_think_when_model_does_not_support_it():
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": '"llama3.2" does not support thinking'})
        return chat_reply("ok")

    p = provider(handler)
    assert await p.complete(purpose="x", system="s", messages=[]) == "ok"
    assert "think" in bodies[0] and "think" not in bodies[1]
    await p.complete(purpose="x", system="s", messages=[])
    assert "think" not in bodies[2]  # remembered


def test_helpers():
    assert strip_think("<think>a</think>b") == "b"
    assert strip_think("ok <think>never closed") == "ok"
    assert strip_think("I reason [1] here.\n</think>\n\nAnswer [2].") == "Answer [2]."
    assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'
    f = _ThinkFilter()
    assert f.feed("x <th") == "x " and f.feed("ink>y</think>z") == "z" and f.flush() == ""


def test_settings_defaults_are_free_and_budget_scales_with_context():
    s = Settings(_env_file=None)
    assert (
        s.llm_provider == "ollama"
        and s.main_model == "qwen3:4b-instruct"
        and s.fast_model == "qwen3:4b-instruct"
    )
    assert Settings(_env_file=None, ollama_fast_model="qwen3:1.7b").fast_model == "qwen3:1.7b"
    assert Settings(_env_file=None, llm_provider="anthropic").main_model == "claude-opus-5-5"
    small = Settings(_env_file=None, ollama_num_ctx=4096).evidence_char_budget
    big = Settings(_env_file=None, ollama_num_ctx=32768).evidence_char_budget
    assert small < big and small < 8000


def test_anthropic_is_an_optional_extra():
    pytest.importorskip("pytest")
    try:
        import anthropic  # noqa: F401
    except ImportError:
        with pytest.raises(LLMUnavailable, match="--extra anthropic"):
            get_llm(Settings(_env_file=None, llm_provider="anthropic"))
