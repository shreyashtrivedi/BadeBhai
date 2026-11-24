"""Runs the agent graph for one question: blocking (`ask`) or as a stream of events (`stream`)."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from repomind.app.agent.graph import build_graph
from repomind.app.agent.state import RunDeps, fresh_turn
from repomind.app.mcp_client import McpToolbox
from repomind.app.memory.store import MemoryStore
from repomind.app.schemas import AskRequest, AskResponse, Citation, Evidence, SubQuestion, TraceStep
from repomind.config import Settings
from repomind.db.repos import RepoRecord
from repomind.llm.base import LLMProvider


class AgentRunner:
    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        pool,
        embedder,
        checkpointer=None,
        toolbox_factory: Callable[[], McpToolbox] | None = None,
        memory_store=None,
    ):
        self.settings = settings
        self.llm = llm
        self.pool = pool
        self.embedder = embedder
        self.memory_store = (
            memory_store if memory_store is not None else MemoryStore(pool, embedder, settings)
        )
        self.toolbox_factory = toolbox_factory or (lambda: McpToolbox.from_settings(settings))
        self.graph = build_graph().compile(checkpointer=checkpointer or InMemorySaver())

    @staticmethod
    def thread_key(req: AskRequest, thread_id: str) -> str:
        # Namespaced so one user can never load another user's thread state.
        return f"{req.user_id}::{req.repo}::{thread_id}"

    def _prepare(self, req: AskRequest, repo: RepoRecord, toolbox: McpToolbox):
        thread_id = req.thread_id or uuid.uuid4().hex[:12]
        deps = RunDeps(
            settings=self.settings,
            llm=self.llm,
            pool=self.pool,
            embedder=self.embedder,
            repo=repo,
            toolbox=toolbox,
            memory_store=self.memory_store,
        )
        config = {
            "configurable": {"thread_id": self.thread_key(req, thread_id), "deps": deps},
            "recursion_limit": 60,
            # Picked up by LangSmith when LANGSMITH_TRACING=true (LangGraph traces every node).
            "run_name": "repomind.ask",
            "tags": ["repomind", "agent"],
            "metadata": {"user_id": req.user_id, "repo": repo.name, "thread_id": thread_id},
        }
        return thread_id, fresh_turn(req.question, repo.name, req.user_id, thread_id), config

    async def ask(self, req: AskRequest, repo: RepoRecord) -> AskResponse:
        t0 = time.perf_counter()
        async with self.toolbox_factory() as toolbox:
            thread_id, inputs, config = self._prepare(req, repo, toolbox)
            async with asyncio.timeout(self.settings.request_timeout_s):
                final = await self.graph.ainvoke(inputs, config)
            return self.to_response(final, toolbox, thread_id, t0)

    async def stream(self, req: AskRequest, repo: RepoRecord) -> AsyncIterator[dict[str, Any]]:
        """Yields {"type": "step"|"tool"|"token"|"final", ...} events."""
        t0 = time.perf_counter()
        async with self.toolbox_factory() as toolbox:
            thread_id, inputs, config = self._prepare(req, repo, toolbox)
            yield {"type": "start", "thread_id": thread_id}
            final: dict[str, Any] = {}
            async with asyncio.timeout(self.settings.request_timeout_s):
                async for mode, chunk in self.graph.astream(inputs, config, stream_mode=["custom", "values"]):
                    if mode == "custom":
                        yield chunk
                    else:
                        final = chunk
            yield {"type": "final", "response": self.to_response(final, toolbox, thread_id, t0).model_dump()}

    def to_response(self, final: dict, toolbox: McpToolbox, thread_id: str, t0: float) -> AskResponse:
        evidence = final.get("evidence", {})
        subs = [SubQuestion(**x) for x in final.get("sub_questions", [])]
        return AskResponse(
            answer=final.get("answer", ""),
            citations=[Citation(**c) for c in final.get("citations", [])],
            verification=final.get("verification", {}),
            trace=[TraceStep(**t) for t in final.get("trace", [])],
            mode="agent",
            repo=final.get("repo", ""),
            thread_id=thread_id,
            sub_questions=subs,
            insufficient=[sq.text for sq in subs if sq.status == "insufficient"],
            memories_used=final.get("memories", []),
            memories_written=final.get("memories_written", []),
            evidence=[Evidence(**evidence[k]) for k in final.get("synth_keys", []) if k in evidence],
            warnings=final.get("warnings", []),
            tool_calls=[asdict(c) for c in toolbox.calls],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
