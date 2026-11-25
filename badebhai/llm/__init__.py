"""LLM providers. `get_llm()` returns the configured provider."""

from __future__ import annotations

from badebhai.config import Settings, get_settings
from badebhai.llm.base import (
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
    if s.llm_provider == "ollama":  # default: free, local
        from badebhai.llm.ollama_provider import OllamaProvider

        return OllamaProvider(s)
    if s.llm_provider == "anthropic":  # optional, paid
        try:
            from badebhai.llm.anthropic_provider import AnthropicProvider
        except ImportError as e:
            raise LLMUnavailable(
                "LLM_PROVIDER=anthropic needs the optional extra: `uv sync --extra anthropic`"
            ) from e

        return AnthropicProvider(s)
    if s.llm_provider == "fake":
        from badebhai.llm.fake import FakeLLM

        return FakeLLM()
    raise ValueError(f"unknown LLM_PROVIDER {s.llm_provider!r}")
