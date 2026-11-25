"""LangGraph state and per-run dependencies.

State is checkpointed per thread (short-term memory). Per-turn fields are reset by the input of
each new question; `history` carries the last few Q/A turns of the thread forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from badebhai.app.mcp_client import McpToolbox
from badebhai.config import Settings
from badebhai.db.repos import RepoRecord
from badebhai.llm.base import LLMProvider

HISTORY_TURNS = 6


class AgentState(TypedDict, total=False):
    question: str
    repo: str
    user_id: str
    thread_id: str
    history: list[dict[str, str]]  # [{question, answer}] most recent last
    memories: list[dict[str, Any]]
    sub_questions: list[dict[str, Any]]
    evidence: dict[str, dict[str, Any]]
    synth_keys: list[str]  # evidence keys in the order numbered for the synthesizer
    answer_raw: str
    answer: str
    citations: list[dict[str, Any]]
    verification: dict[str, Any]
    warnings: list[str]
    trace: list[dict[str, Any]]
    memories_written: list[dict[str, Any]]
    round: int


def fresh_turn(question: str, repo: str, user_id: str, thread_id: str) -> AgentState:
    return AgentState(
        question=question,
        repo=repo,
        user_id=user_id,
        thread_id=thread_id,
        memories=[],
        sub_questions=[],
        evidence={},
        synth_keys=[],
        answer_raw="",
        answer="",
        citations=[],
        verification={},
        warnings=[],
        trace=[],
        memories_written=[],
        round=0,
    )


@dataclass
class RunDeps:
    """Non-serializable dependencies, passed through `config["configurable"]["deps"]`."""

    settings: Settings
    llm: LLMProvider
    pool: Any
    embedder: Any
    repo: RepoRecord
    toolbox: McpToolbox
    memory_store: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


def deps_from(config) -> RunDeps:
    return config["configurable"]["deps"]
