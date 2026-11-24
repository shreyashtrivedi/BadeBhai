"""LLM providers. `get_llm()` returns the configured provider."""

from __future__ import annotations

from repomind.config import Settings, get_settings
from repomind.llm.base import (
    LLMError,
    LLMProvider,
    LLMRefusal,
    LLMUnavailable,
    UsageTracker,
    track_usage,
)

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMRefusal",
    "LLMUnavailable",
    "UsageTracker",
    "get_llm",
    "track_usage",
]


def get_llm(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    if s.llm_provider == "anthropic":
        from repomind.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider(s)
    if s.llm_provider == "fake":
        from repomind.llm.fake import FakeLLM

        return FakeLLM()
    raise ValueError(f"unknown LLM_PROVIDER {s.llm_provider!r}")
