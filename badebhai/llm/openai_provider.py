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
        if self._replacement:
            return self._replacement
        return self.s.fast_model if fast else self.s.main_model

    async def _pick_replacement(self, missing: str) -> str | None:
        """The configured model was retired (404): choose the closest available chat model instead."""
        try:
            r = await self.client.get("models")
            ids = [m.get("id", "") for m in r.json().get("data", [])] if r.status_code < 400 else []
        except (httpx.HTTPError, ValueError):
            return None
        ids = [i.split("/", 1)[-1] for i in ids]
        skip = ("embed", "image", "tts", "audio", "live", "vision", "aqa", "learnlm", "imagen", "veo")
        chat = [i for i in ids if not any(k in i.lower() for k in skip)]
        family = missing.split("-")[0]  # e.g. "gemini"
        flash = [
            i
            for i in chat
            if family in i and "flash" in i and "lite" not in i and "preview" not in i and "exp" not in i
        ]
        pool = flash or [i for i in chat if family in i] or chat
        if not pool:
            return None
        choice = sorted(pool, key=lambda i: [int(x) if x.isdigit() else 0 for x in re.findall(r"\d+", i)])[-1]
        log.warning("model not available; switching", extra={"configured": missing, "using": choice})
        self._replacement = choice
        return choice

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
        for attempt in range(1, 5):
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
            if r.status_code == 429 and attempt < 4:
                wait = min(_MAX_429_WAIT_S, float(r.headers.get("retry-after") or 2**attempt))
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
