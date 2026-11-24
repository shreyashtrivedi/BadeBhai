"""Claude implementation of LLMProvider (official `anthropic` SDK)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anthropic
from langsmith import traceable
from pydantic import ValidationError

from repomind.config import Settings
from repomind.llm.base import (
    LLMCall,
    LLMError,
    LLMProvider,
    LLMRefusal,
    LLMUnavailable,
    T,
    Timer,
    record_call,
    strict_json_schema,
)
from repomind.log import get_logger

log = get_logger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = ("claude-fable-5", "claude-mythos-5", "claude-opus-5", "claude-sonnet-5-5")
# Models that reject `output_config.effort`.
_NO_EFFORT = ("claude-sonnet-4-5", "claude-haiku-4-5", "claude-3")


def _supports_fallbacks(model: str) -> bool:
    return model.startswith(_FALLBACK_MODELS)


def _supports_effort(model: str) -> bool:
    return not model.startswith(_NO_EFFORT)


def _maybe_missing_credentials(e: TypeError) -> Exception:
    """The SDK raises TypeError at request time when no credentials can be resolved."""
    if "authentication" in str(e).lower():
        return LLMUnavailable(
            "No Anthropic credentials found. Set ANTHROPIC_API_KEY in .env (or run `ant auth login`)."
        )
    return e


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, settings: Settings):
        self.s = settings
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            try:
                key = self.s.anthropic_api_key.get_secret_value() if self.s.anthropic_api_key else None
                kwargs = {"api_key": key} if key else {}  # else: env var / `ant auth login` profile
                self._client = anthropic.AsyncAnthropic(
                    timeout=self.s.llm_timeout_s, max_retries=self.s.llm_max_retries, **kwargs
                )
            except Exception as e:  # missing credentials raise at construction time
                raise LLMUnavailable(
                    "Anthropic client could not be created. Set ANTHROPIC_API_KEY in .env "
                    f"(or log in with `ant auth login`). Details: {e}"
                ) from e
        return self._client

    # ---------- request building ----------
    def _params(
        self, *, fast: bool, system: str, messages: list[dict], max_tokens: int, fmt: dict | None = None
    ) -> tuple[dict[str, Any], bool]:
        model = self.s.fast_model if fast else self.s.llm_model
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            # Stable per-purpose system prompt first, so it can be prompt-cached.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
        }
        output_config: dict[str, Any] = {}
        if _supports_effort(model):
            output_config["effort"] = self.s.llm_fast_effort if fast else self.s.llm_effort
        if fmt:
            output_config["format"] = fmt
        if output_config:
            params["output_config"] = output_config
        use_beta = self.s.llm_fallbacks != "off" and _supports_fallbacks(model)
        if use_beta:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"
        return params, use_beta

    def _api(self, beta: bool):
        return self.client.beta.messages if beta else self.client.messages

    @staticmethod
    def _text(resp) -> str:
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")

    @staticmethod
    def _record(purpose: str, resp, ms: int) -> None:
        u = resp.usage
        record_call(
            LLMCall(
                purpose=purpose,
                model=resp.model,
                input_tokens=u.input_tokens or 0,
                output_tokens=u.output_tokens or 0,
                cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                latency_ms=ms,
            )
        )

    @staticmethod
    def _check_stop(resp, purpose: str) -> None:
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            cat = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"model declined the {purpose} request (category={cat})")

    async def _create(self, purpose: str, params: dict, beta: bool):
        try:
            with Timer() as t:
                resp = await self._api(beta).create(**params)
        except anthropic.AuthenticationError as e:
            raise LLMUnavailable("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMUnavailable(f"Anthropic API key lacks permission: {e.message}") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"model not found ({params['model']}): {e.message}") from e
        except anthropic.RateLimitError as e:
            raise LLMError("Anthropic rate limit hit after retries; try again shortly.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the Anthropic API (network error).") from e
        except TypeError as e:
            raise _maybe_missing_credentials(e) from e
        self._record(purpose, resp, t.ms)
        self._check_stop(resp, purpose)
        return resp

    # ---------- interface ----------
    @traceable(run_type="llm", name="llm.complete")
    async def complete(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 16000
    ) -> str:
        params, beta = self._params(fast=fast, system=system, messages=messages, max_tokens=max_tokens)
        resp = await self._create(purpose, params, beta)
        if resp.stop_reason == "max_tokens":
            log.warning("llm output truncated", extra={"purpose": purpose})
        return self._text(resp)

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
        fmt = {"type": "json_schema", "schema": strict_json_schema(schema)}
        params, beta = self._params(
            fast=fast, system=system, messages=messages, max_tokens=max_tokens, fmt=fmt
        )
        resp = await self._create(purpose, params, beta)
        if resp.stop_reason == "max_tokens":
            raise LLMError(f"{purpose}: structured output truncated at max_tokens={max_tokens}")
        text = self._text(resp)
        try:
            return schema.model_validate_json(text)
        except ValidationError as e:
            raise LLMError(f"{purpose}: model returned JSON that does not match the schema: {e}") from e

    async def stream(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 32000
    ) -> AsyncIterator[str]:
        params, beta = self._params(fast=fast, system=system, messages=messages, max_tokens=max_tokens)
        try:
            with Timer() as t:
                async with self._api(beta).stream(**params) as stream:
                    async for text in stream.text_stream:
                        yield text
                    final = await stream.get_final_message()
        except anthropic.AuthenticationError as e:
            raise LLMUnavailable("Anthropic rejected the API key. Check ANTHROPIC_API_KEY.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Lost connection to the Anthropic API while streaming.") from e
        except TypeError as e:
            raise _maybe_missing_credentials(e) from e
        self._record(purpose, final, t.ms)
        self._check_stop(final, purpose)
