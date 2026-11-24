"""LLM provider interface, usage/cost accounting, and JSON-schema helpers."""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class LLMUnavailable(LLMError):
    """No credentials / provider misconfigured. Surfaced to the user as-is."""


# USD per 1M tokens: (input, output, cache_read). Cache writes bill at 1.25x input.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-2": (8.0, 40.0, 0.20),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10),
    "claude-sonnet-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


def price_for(model: str) -> tuple[float, float, float] | None:
    for name, p in PRICES.items():
        if model == name or model.startswith(name + "-"):
            return p
    return None


@dataclass
class LLMCall:
    purpose: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    latency_ms: int = 0

    @property
    def cost_usd(self) -> float | None:
        p = price_for(self.model)
        if p is None:
            return None
        inp, out, cr = p
        return (
            self.input_tokens * inp
            + self.output_tokens * out
            + self.cache_read_tokens * cr
            + self.cache_write_tokens * inp * 1.25
        ) / 1_000_000


@dataclass
class UsageTracker:
    calls: list[LLMCall] = field(default_factory=list)

    def add(self, call: LLMCall) -> None:
        self.calls.append(call)

    def summary(self) -> dict[str, Any]:
        costs = [c.cost_usd for c in self.calls]
        return {
            "llm_calls": len(self.calls),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "cache_read_tokens": sum(c.cache_read_tokens for c in self.calls),
            "cost_usd": round(sum(c for c in costs if c is not None), 5) if self.calls else 0.0,
            "cost_complete": all(c is not None for c in costs),
            "by_purpose": _by_purpose(self.calls),
        }


def _by_purpose(calls: list[LLMCall]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in calls:
        d = out.setdefault(c.purpose, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_ms": 0})
        d["calls"] += 1
        d["input_tokens"] += c.input_tokens
        d["output_tokens"] += c.output_tokens
        d["latency_ms"] += c.latency_ms
    return out


_tracker: ContextVar[UsageTracker | None] = ContextVar("repomind_usage", default=None)


@contextmanager
def track_usage() -> Iterator[UsageTracker]:
    tracker = UsageTracker()
    token = _tracker.set(tracker)
    try:
        yield tracker
    finally:
        _tracker.reset(token)


def record_call(call: LLMCall) -> None:
    tracker = _tracker.get()
    if tracker is not None:
        tracker.add(call)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.perf_counter() - self.t0) * 1000)


_DROP_KEYS = {
    "title",
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
    "examples",
    "format",
}


def strict_json_schema(model: type[BaseModel]) -> dict:
    """Pydantic schema -> structured-output schema: refs inlined, closed objects, all fields required."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(x) for x in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return walk(copy.deepcopy(defs[node["$ref"].split("/")[-1]]))
        out: dict[str, Any] = {}
        for key, val in node.items():
            if key in _DROP_KEYS:
                continue
            if key == "properties":
                out[key] = {name: walk(sub) for name, sub in val.items()}
            else:
                out[key] = walk(val)
        if out.get("type") == "object" or "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out.get("properties", {}).keys())
        return out

    return walk(schema)


class LLMProvider(ABC):
    """The only surface the rest of the app uses. `fast=True` selects the cheaper/faster role model."""

    name: str = "abstract"

    @abstractmethod
    async def complete(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 16000
    ) -> str: ...

    @abstractmethod
    async def complete_json(
        self,
        *,
        purpose: str,
        system: str,
        messages: list[dict],
        schema: type[T],
        fast: bool = True,
        max_tokens: int = 16000,
    ) -> T: ...

    @abstractmethod
    def stream(
        self, *, purpose: str, system: str, messages: list[dict], fast: bool = False, max_tokens: int = 32000
    ) -> AsyncIterator[str]: ...
