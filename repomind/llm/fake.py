"""Scripted LLM for tests. Each `purpose` maps to a handler that returns the response."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from repomind.llm.base import LLMCall, LLMError, LLMProvider, T, record_call

Handler = Callable[[str, list[dict], Any], Any]


@dataclass
class FakeCall:
    purpose: str
    system: str
    messages: list[dict]
    fast: bool


class FakeLLM(LLMProvider):
    name = "fake"

    def __init__(self, handlers: dict[str, Handler | Any] | None = None):
        self.handlers = dict(handlers or {})
        self.calls: list[FakeCall] = []

    def _respond(self, purpose: str, system: str, messages: list[dict], schema: Any, fast: bool) -> Any:
        self.calls.append(FakeCall(purpose, system, messages, fast))
        record_call(
            LLMCall(purpose=purpose, model="fake", input_tokens=len(str(messages)) // 4, output_tokens=20)
        )
        if purpose not in self.handlers:
            raise LLMError(f"FakeLLM has no handler for purpose {purpose!r}")
        h = self.handlers[purpose]
        return h(system, messages, schema) if callable(h) else h

    def calls_for(self, purpose: str) -> list[FakeCall]:
        return [c for c in self.calls if c.purpose == purpose]

    async def complete(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 16000
    ) -> str:
        return str(self._respond(purpose, system, messages, None, fast))

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
        out = self._respond(purpose, system, messages, schema, fast)
        if isinstance(out, BaseModel):
            return schema.model_validate(out.model_dump())
        return schema.model_validate(out)

    async def stream(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 32000
    ) -> AsyncIterator[str]:
        text = str(self._respond(purpose, system, messages, None, fast))
        for i in range(0, len(text), 16):
            yield text[i : i + 16]
