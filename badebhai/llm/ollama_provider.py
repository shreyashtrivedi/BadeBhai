"""Free, local LLM provider: Ollama (open models such as Qwen, Llama, Gemma), via its native HTTP API.

* Structured steps pass a JSON schema as Ollama's `format`; Ollama turns it into a decoding grammar,
  so the output is always parseable JSON. A second attempt with the validation error is made if the
  JSON does not satisfy the Pydantic model (for example a missing semantic constraint).
* `num_ctx` is always set explicitly: Ollama's default context is small and silently truncates
  long prompts (evidence would be cut off without any error).
* Thinking is disabled by default (`OLLAMA_THINK=false`) for speed; stray <think> blocks are stripped.
* Usage is recorded with a $0 price so cost reports read "free (local)".
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx
from langsmith import traceable
from pydantic import ValidationError

from badebhai.config import Settings
from badebhai.llm.base import (
    LLMCall,
    LLMError,
    LLMProvider,
    LLMUnavailable,
    T,
    Timer,
    record_call,
    strict_json_schema,
)
from badebhai.log import get_logger

log = get_logger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think(text: str) -> str:
    """Remove model reasoning: <think>...</think> blocks, or everything before a bare </think>."""
    close, open_ = text.find("</think>"), text.find("<think>")
    if close != -1 and (open_ == -1 or close < open_):  # template opened the block, model closed it
        text = text[close + len("</think>") :]
    text = _THINK_BLOCK.sub("", text)
    if "<think>" in text:  # unterminated block: drop everything after the tag
        text = text.split("<think>", 1)[0]
    return text.strip()


def extract_json(text: str) -> str:
    """Return the outermost {...} span (models occasionally wrap JSON in prose or fences)."""
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


class _ThinkFilter:
    """Streaming filter that drops <think>...</think> spans split across chunks."""

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False

    def feed(self, piece: str) -> str:
        self.buf += piece
        out = []
        while self.buf:
            if self.inside:
                end = self.buf.find("</think>")
                if end == -1:
                    self.buf = self.buf[-8:]  # keep a tail in case the closing tag is split
                    return "".join(out)
                self.buf = self.buf[end + len("</think>") :]
                self.inside = False
            else:
                start = self.buf.find("<think>")
                if start == -1:
                    # hold back a possible partial "<think" at the end
                    keep = max(
                        (
                            i
                            for i in range(1, min(7, len(self.buf)) + 1)
                            if "<think>".startswith(self.buf[-i:])
                        ),
                        default=0,
                    )
                    out.append(self.buf[: len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep :]
                    return "".join(out)
                out.append(self.buf[:start])
                self.buf = self.buf[start + len("<think>") :]
                self.inside = True
        return "".join(out)

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return rest


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self.s = settings
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._send_think = True  # dropped automatically if the server/model rejects the field

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.s.ollama_base_url,
                timeout=httpx.Timeout(self.s.ollama_timeout_s, connect=10.0),
                transport=self._transport,
            )
        return self._client

    def _model(self, fast: bool) -> str:
        return self.s.fast_model if fast else self.s.main_model

    def _body(
        self,
        model: str,
        system: str,
        messages: list[dict],
        *,
        stream: bool,
        max_tokens: int,
        fmt: dict | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "stream": stream,
            "keep_alive": self.s.ollama_keep_alive,
            "options": {
                "num_ctx": self.s.ollama_num_ctx,
                # structured calls (plans, grades, judgements) decode greedily: same input, same verdict
                "temperature": 0.0 if fmt is not None else self.s.ollama_temperature,
                "num_predict": min(max_tokens, self.s.ollama_max_output_tokens),
            },
        }
        if fmt is not None:
            body["format"] = fmt
        if self._send_think:
            body["think"] = self.s.ollama_think
        return body

    def _error(self, model: str, status: int, text: str) -> LLMError:
        low = text.lower()
        if status == 404 or ("model" in low and "not found" in low):
            return LLMUnavailable(
                f"Ollama has no model {model!r}. Pull it with `docker compose exec ollama ollama pull {model}` "
                f"(or `ollama pull {model}` for a local install)."
            )
        return LLMError(f"Ollama error {status}: {text[:300]}")

    def _unreachable(self) -> LLMUnavailable:
        return LLMUnavailable(
            f"Cannot reach Ollama at {self.s.ollama_base_url}. Start it with `docker compose up -d ollama` "
            "(add `-f docker-compose.gpu.yml` on an NVIDIA GPU) or install Ollama from ollama.com."
        )

    async def _chat(self, purpose: str, body: dict) -> dict:
        model = body["model"]
        for attempt in (1, 2):
            try:
                with Timer() as t:
                    r = await self.client.post("/api/chat", json=body)
            except httpx.ConnectError as e:
                raise self._unreachable() from e
            except httpx.TimeoutException as e:
                raise LLMError(
                    f"Ollama timed out after {self.s.ollama_timeout_s:.0f}s on {purpose}; the model may be too "
                    "large for this machine (try a smaller OLLAMA_MODEL or the GPU override)."
                ) from e
            if r.status_code == 400 and "think" in r.text.lower() and self._send_think:
                self._send_think = False  # model/server without thinking support: retry without the field
                body = {k: v for k, v in body.items() if k != "think"}
                continue
            if r.status_code >= 500 and attempt == 1:
                log.warning("ollama server error, retrying", extra={"status": r.status_code})
                continue
            if r.status_code >= 400:
                raise self._error(model, r.status_code, r.text)
            data = r.json()
            self._record(purpose, model, data, t.ms)
            if data.get("done_reason") == "length":
                log.warning("ollama output truncated", extra={"purpose": purpose, "model": model})
            return data
        raise LLMError(f"Ollama failed twice on {purpose}")

    @staticmethod
    def _record(purpose: str, model: str, data: dict, ms: int) -> None:
        record_call(
            LLMCall(
                purpose=purpose,
                model=f"ollama/{model}",
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
                latency_ms=ms,
            )
        )

    # ---------- interface ----------
    @traceable(run_type="llm", name="llm.complete")
    async def complete(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 16000
    ) -> str:
        body = self._body(self._model(fast), system, messages, stream=False, max_tokens=max_tokens)
        data = await self._chat(purpose, body)
        return strip_think(data.get("message", {}).get("content", ""))

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
        js = strict_json_schema(schema, keep_array_bounds=True)
        system = (
            f"{system}\n\nRespond with one JSON object that matches this JSON schema exactly. "
            f"No prose, no markdown fences.\nJSON schema: {json.dumps(js, separators=(',', ':'))}"
        )
        convo = list(messages)
        last_error: Exception | None = None
        for _attempt in (1, 2):
            body = self._body(self._model(fast), system, convo, stream=False, max_tokens=max_tokens, fmt=js)
            data = await self._chat(purpose, body)
            text = strip_think(data.get("message", {}).get("content", ""))
            try:
                return schema.model_validate_json(extract_json(text))
            except ValidationError as e:
                last_error = e
                convo = [
                    *messages,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": f"That JSON did not match the schema ({e.errors()[:3]}). "
                        "Reply with corrected JSON only.",
                    },
                ]
        raise LLMError(f"{purpose}: local model returned JSON that does not match the schema: {last_error}")

    async def stream(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 32000
    ) -> AsyncIterator[str]:
        model = self._model(fast)
        body = self._body(model, system, messages, stream=True, max_tokens=max_tokens)
        filt = _ThinkFilter()
        final: dict = {}
        try:
            with Timer() as t:
                async with self.client.stream("POST", "/api/chat", json=body) as r:
                    if r.status_code == 400 and self._send_think:
                        text = (await r.aread()).decode(errors="replace")
                        if "think" in text.lower():  # retry without `think` via the non-streaming path
                            self._send_think = False
                            yield await self.complete(
                                purpose=purpose,
                                system=system,
                                messages=messages,
                                fast=fast,
                                max_tokens=max_tokens,
                            )
                            return
                        raise self._error(model, r.status_code, text)
                    if r.status_code >= 500:  # e.g. the runner crashed (CUDA error) and reloads: nothing
                        # was streamed yet, so retry through the non-streaming path (which retries once)
                        log.warning(
                            "ollama server error before streaming, retrying", extra={"status": r.status_code}
                        )
                        yield await self.complete(
                            purpose=purpose,
                            system=system,
                            messages=messages,
                            fast=fast,
                            max_tokens=max_tokens,
                        )
                        return
                    if r.status_code >= 400:
                        raise self._error(model, r.status_code, (await r.aread()).decode(errors="replace"))
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        if obj.get("error"):
                            raise LLMError(f"Ollama error while streaming: {obj['error']}")
                        piece = filt.feed(obj.get("message", {}).get("content", ""))
                        if piece:
                            yield piece
                        if obj.get("done"):
                            final = obj
                    tail = filt.flush()
                    if tail:
                        yield tail
        except httpx.ConnectError as e:
            raise self._unreachable() from e
        except httpx.TimeoutException as e:
            raise LLMError(f"Ollama timed out after {self.s.ollama_timeout_s:.0f}s while streaming") from e
        self._record(purpose, model, final, t.ms)
