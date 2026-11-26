"""Any OpenAI-compatible chat API: Google Gemini (free tier), Groq, OpenRouter, Together, a local LM Studio
or vLLM server, or Ollama's own /v1 endpoint. Used for hosted deployments where the machine is too small
to run a model, e.g. the public demo on a free CPU host.

    LLM_PROVIDER=openai
    OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    OPENAI_API_KEY=...            # never logged, never sent anywhere but OPENAI_BASE_URL
    LLM_MODEL=gemini-2.5-flash

Structured output uses `response_format: json_schema` and falls back to `json_object` (schema in the
prompt, validated with pydantic) on servers that do not support schemas. 429s wait for Retry-After;
OPENAI_MAX_RPM keeps a free tier's requests-per-minute limit from being hit in the first place.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections import deque
from collections.abc import AsyncIterator
from typing import Any

import httpx
from langsmith import traceable
from pydantic import ValidationError

from badebhai.config import Settings
from badebhai.llm.base import (
    FREE_MODEL_PREFIXES,
    LLMCall,
    LLMError,
    LLMProvider,
    LLMUnavailable,
    T,
    Timer,
    record_call,
    strict_json_schema,
)
from badebhai.llm.ollama_provider import _ThinkFilter, extract_json, strip_think
from badebhai.log import get_logger

log = get_logger(__name__)

_MAX_429_WAIT_S = 45.0
_STANDIN_FOR_S = 300.0  # an overloaded / out-of-quota model is skipped this long, then tried again
_NOT_CHAT = ("embed", "image", "tts", "audio", "live", "vision", "aqa", "learnlm", "imagen", "veo", "exp")
_DAILY_QUOTA = re.compile(r"per\s*day|perday|daily", re.I)


def _retry_after(r: httpx.Response) -> float:
    try:
        return float(r.headers.get("retry-after") or 0)
    except ValueError:
        return 0.0


def _version_key(model_id: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", model_id)]


def rank_standins(ids: list[str], family: str) -> list[str]:
    """Chat models to fall back on, best first: stable Flash, preview Flash, then Flash-Lite; newest first."""
    chat = [i for i in ids if family in i and not any(k in i.lower() for k in _NOT_CHAT)]
    flash = [i for i in chat if "flash" in i]

    def tier(i: str) -> int:
        return (2 if "lite" in i else 0) + (1 if "preview" in i else 0)

    return sorted(flash or chat, key=lambda i: (tier(i), [-v for v in _version_key(i)]))


class OpenAICompatProvider(LLMProvider):
    name = "openai"

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        if not settings.llm_model:
            raise LLMUnavailable("LLM_PROVIDER=openai needs LLM_MODEL (e.g. gemini-2.5-flash)")
        if not settings.openai_base_url:
            raise LLMUnavailable("LLM_PROVIDER=openai needs OPENAI_BASE_URL (see .env.example for presets)")
        self.s = settings
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._json_mode = "json_schema"  # downgraded to "json_object" if the server rejects schemas
        self._replacement: str | None = None  # set when the configured model turned out to be retired
        self._models: list[str] | None = None  # the API's model ids, fetched once when needed
        self._down: dict[str, float] = {}  # model -> monotonic time until which it is skipped
        self._standin: dict[str, str] = {}  # overloaded model -> the model answering in its place
        self._sent: deque[float] = deque()
        self._rate_lock = asyncio.Lock()
        if settings.openai_free and "openai-compat/" not in FREE_MODEL_PREFIXES:
            FREE_MODEL_PREFIXES.append("openai-compat/")

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.s.openai_api_key:
                headers["Authorization"] = f"Bearer {self.s.openai_api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.s.openai_base_url.rstrip("/") + "/",
                headers=headers,
                timeout=httpx.Timeout(self.s.openai_timeout_s, connect=10.0),
                transport=self._transport,
            )
        return self._client

    def _model(self, fast: bool) -> str:
        model = self._replacement or (self.s.fast_model if fast else self.s.main_model)
        for _ in range(5):  # follow stand-ins while the model they replace is still being skipped
            nxt = self._standin.get(model)
            if not nxt or self._down.get(model, 0) <= time.monotonic():
                break
            model = nxt
        return model

    async def _model_ids(self) -> list[str]:
        if self._models is None:
            try:
                r = await self.client.get("models")
                ids = [m.get("id", "") for m in r.json().get("data", [])] if r.status_code < 400 else []
            except (httpx.HTTPError, ValueError):
                return []
            self._models = [i.split("/", 1)[-1] for i in ids if i]
        return self._models

    async def _pick_replacement(self, missing: str) -> str | None:
        """The configured model was retired (404): choose the closest available chat model instead."""
        family = missing.split("-")[0]  # e.g. "gemini"
        ranked = [i for i in rank_standins(await self._model_ids(), family) if i != missing]
        stable = [i for i in ranked if "lite" not in i and "preview" not in i]
        choice = (stable or ranked or [None])[0]
        if choice:
            log.warning("model not available; switching", extra={"configured": missing, "using": choice})
            self._replacement = choice
        return choice

    async def _pick_standin(self, failing: str, why: str) -> str | None:
        """`failing` is overloaded or out of quota: skip it for a while and answer with another model
        (OPENAI_FALLBACK_MODELS if set, otherwise the best other chat model the API lists)."""
        now = time.monotonic()
        self._down[failing] = now + _STANDIN_FOR_S
        configured = [m.strip() for m in (self.s.openai_fallback_models or "").split(",") if m.strip()]
        pool = configured or rank_standins(await self._model_ids(), failing.split("-")[0])
        for model in pool:
            if model != failing and self._down.get(model, 0) <= now:
                log.warning(
                    "model busy; using a stand-in", extra={"busy": failing, "using": model, "why": why}
                )
                self._standin[failing] = model
                return model
        return None

    # ---------- plumbing ----------
    async def _throttle(self) -> None:
        """Stay under OPENAI_MAX_RPM (free tiers allow ~10-30 requests per minute)."""
        rpm = self.s.openai_max_rpm
        if rpm <= 0:
            return
        async with self._rate_lock:
            while True:
                now = time.monotonic()
                while self._sent and self._sent[0] <= now - 60:
                    self._sent.popleft()
                if len(self._sent) < rpm:
                    self._sent.append(now)
                    return
                await asyncio.sleep(self._sent[0] + 60 - now + 0.05)

    def _body(
        self, model: str, system: str, messages: list[dict], *, max_tokens: int, structured: bool
    ) -> dict:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_tokens": min(max_tokens, self.s.openai_max_output_tokens),
            "temperature": 0.0 if structured else self.s.openai_temperature,
        }
        if self.s.openai_reasoning_effort:
            body["reasoning_effort"] = self.s.openai_reasoning_effort
        return body

    def _record(self, purpose: str, model: str, usage: dict | None, ms: int) -> None:
        usage = usage or {}
        record_call(
            LLMCall(
                purpose=purpose,
                model=f"openai-compat/{model}",
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                latency_ms=ms,
            )
        )

    def _error(self, status: int, text: str) -> LLMError:
        if status in (401, 403):
            return LLMUnavailable(f"The model API rejected the key (HTTP {status}). Check OPENAI_API_KEY.")
        if status == 404:
            return LLMUnavailable(
                f"The model API has no model {self.s.main_model!r} (HTTP 404). Check LLM_MODEL."
            )
        return LLMError(f"Model API error {status}: {text[:300]}")

    async def _post(self, purpose: str, body: dict) -> dict:
        model = body["model"]
        for attempt in range(1, 7):
            await self._throttle()
            try:
                with Timer() as t:
                    r = await self.client.post("chat/completions", json=body)
            except httpx.ConnectError as e:
                raise LLMUnavailable(f"Cannot reach the model API at {self.s.openai_base_url}") from e
            except httpx.TimeoutException as e:
                raise LLMError(
                    f"The model API timed out after {self.s.openai_timeout_s:.0f}s on {purpose}"
                ) from e
            retry_after = _retry_after(r)
            # Overloaded (503) or out of quota for the day: waiting will not help, another model will.
            if r.status_code == 503 or (
                r.status_code == 429 and (retry_after > _MAX_429_WAIT_S or _DAILY_QUOTA.search(r.text))
            ):
                why = "overloaded" if r.status_code == 503 else "out of quota"
                if attempt < 6 and (standin := await self._pick_standin(model, why)):
                    body, model = {**body, "model": standin}, standin
                    continue
            if r.status_code == 429 and attempt < 4:
                wait = min(_MAX_429_WAIT_S, retry_after or 2**attempt)
                log.warning("model API rate limited, waiting", extra={"purpose": purpose, "wait_s": wait})
                await asyncio.sleep(wait)
                continue
            if r.status_code >= 500 and attempt < 3:
                await asyncio.sleep(attempt)
                continue
            if r.status_code == 404 and not self._replacement and await self._pick_replacement(model):
                body = {**body, "model": self._replacement}
                model = self._replacement
                continue
            if r.status_code >= 400:
                raise self._error(r.status_code, r.text)
            data = r.json()
            self._record(purpose, model, data.get("usage"), t.ms)
            return data
        raise LLMError(f"The model API kept rate limiting {purpose}; try again in a minute")

    @staticmethod
    def _content(data: dict) -> str:
        try:
            return strip_think(data["choices"][0]["message"].get("content") or "")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected model API response: {str(data)[:200]}") from e

    # ---------- interface ----------
    @traceable(run_type="llm", name="llm.complete")
    async def complete(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 16000
    ) -> str:
        body = self._body(self._model(fast), system, messages, max_tokens=max_tokens, structured=False)
        return self._content(await self._post(purpose, body))

    @traceable(run_type="llm", name="llm.complete_json")
    async def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        messages: list[dict],
        schema: type[T],
        fast: bool = True,
        max_tokens: int = 16000,
    ) -> T:
        js = strict_json_schema(schema)
        guided = (
            f"{system}\n\nRespond with one JSON object that matches this JSON schema exactly. "
            f"No prose, no markdown fences.\nJSON schema: {json.dumps(js, separators=(',', ':'))}"
        )
        convo = list(messages)
        last_error: Exception | None = None
        for _attempt in (1, 2):
            body = self._body(self._model(fast), guided, convo, max_tokens=max_tokens, structured=True)
            body["response_format"] = (
                {
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": js, "strict": True},
                }
                if self._json_mode == "json_schema"
                else {"type": "json_object"}
            )
            try:
                data = await self._post(purpose, body)
            except LLMError as e:
                if self._json_mode == "json_schema" and "400" in str(e):
                    log.info("model API rejected json_schema; using json_object", extra={"purpose": purpose})
                    self._json_mode = "json_object"
                    continue
                raise
            text = self._content(data)
            try:
                return schema.model_validate_json(extract_json(text))
            except ValidationError as e:
                last_error = e
                convo = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": f"That JSON did not match the schema ({e.errors()[:3]}). Reply with corrected JSON only.",
                    },
                ]
        raise LLMError(f"{purpose}: the model returned JSON that does not match the schema: {last_error}")

    async def stream(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 32000
    ) -> AsyncIterator[str]:
        model = self._model(fast)
        body = {
            **self._body(model, system, messages, max_tokens=max_tokens, structured=False),
            "stream": True,
        }
        filt = _ThinkFilter()
        await self._throttle()
        with Timer() as t:
            try:
                async with self.client.stream("POST", "chat/completions", json=body) as r:
                    if r.status_code == 429 or r.status_code >= 500:
                        # nothing was streamed yet: let the non-streaming path wait and retry
                        await r.aread()
                        yield await self.complete(
                            purpose=purpose,
                            system=system,
                            messages=messages,
                            fast=fast,
                            max_tokens=max_tokens,
                        )
                        return
                    if r.status_code >= 400:
                        raise self._error(r.status_code, (await r.aread()).decode(errors="replace"))
                    usage = None
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        obj = json.loads(payload)
                        usage = obj.get("usage") or usage
                        for choice in obj.get("choices") or []:
                            piece = filt.feed((choice.get("delta") or {}).get("content") or "")
                            if piece:
                                yield piece
                    tail = filt.flush()
                    if tail:
                        yield tail
            except httpx.ConnectError as e:
                raise LLMUnavailable(f"Cannot reach the model API at {self.s.openai_base_url}") from e
        self._record(purpose, model, usage, t.ms)
