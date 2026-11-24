"""Eval runner: naive RAG baseline vs. the agentic pipeline on a golden question set."""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime
from itertools import zip_longest
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from repomind.app.agent.runner import AgentRunner
from repomind.app.evidence import evidence_from_hit
from repomind.app.memory.store import MemoryStore
from repomind.app.naive import answer_naive
from repomind.app.schemas import AskRequest, AskResponse
from repomind.config import Settings
from repomind.db import repos as repo_db
from repomind.db.pool import get_pool
from repomind.embeddings import get_embedder
from repomind.evals.golden import DEFAULT_GOLDEN, GoldenQuestion, load_golden
from repomind.evals.judge import judge_answer
from repomind.evals.metrics import Ref, fully_abstained, hits_at_k, matches, said_insufficient, summarize
from repomind.llm import LLMProvider, get_llm, track_usage
from repomind.log import get_logger
from repomind.retrieval.hybrid import hybrid_search

log = get_logger(__name__)


def _naive_refs(resp: AskResponse) -> list[Ref]:
    # The order the model sees: code and doc hits interleaved.
    return [Ref.from_evidence(e) for e in resp.evidence]


def _agent_refs(resp: AskResponse) -> list[Ref]:
    ranked = sorted(resp.evidence, key=lambda e: e.relevance or 0, reverse=True)
    return [Ref.from_evidence(e) for e in ranked]


def _agent_gathered(resp: AskResponse) -> list[Ref]:
    refs = []
    for step in resp.trace:
        if step.node != "retrieve":
            continue
        for call in step.detail.get("calls", []):
            for key in call.get("evidence", []):
                kind, _, rest = key.partition(":")
                if kind in ("code", "doc"):
                    refs.append(Ref(kind, path=rest.split("#", 1)[0]))
                elif kind == "commit":
                    refs.append(Ref("commit", sha=rest))
                elif kind in ("issue", "pr"):
                    refs.append(Ref(kind, number=int(rest)))
    return refs


class EvalRunner:
    def __init__(
        self, settings: Settings, llm: LLMProvider | None = None, judge_llm: LLMProvider | None = None
    ):
        self.s = settings
        self.pool = get_pool(settings)
        self.embedder = get_embedder(settings)
        self.llm = llm
        self.judge_llm = judge_llm
        self._agent: AgentRunner | None = None

    @property
    def agent(self) -> AgentRunner:
        if self._agent is None:
            self._agent = AgentRunner(
                self.s,
                self.llm or get_llm(self.s),
                self.pool,
                self.embedder,
                checkpointer=InMemorySaver(),
                memory_store=MemoryStore(self.pool, self.embedder, self.s),
            )
        return self._agent

    async def run_retrieval_only(self, q: GoldenQuestion, repo) -> dict:
        t0 = time.perf_counter()
        code, docs = await asyncio.gather(
            asyncio.to_thread(
                hybrid_search,
                self.pool,
                self.embedder,
                repo.id,
                q.question,
                k=6,
                kinds=("code",),
                settings=self.s,
            ),
            asyncio.to_thread(
                hybrid_search,
                self.pool,
                self.embedder,
                repo.id,
                q.question,
                k=3,
                kinds=("doc",),
                settings=self.s,
            ),
        )
        hits = [h for pair in zip_longest(code, docs) for h in pair if h is not None]  # same order as naive
        refs = [Ref.from_evidence(evidence_from_hit(h, repo)) for h in hits]
        return {
            "retrieval_only": True,
            "hits": hits_at_k(refs, q),
            "retrieved": [r.path for r in refs[:10]],
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }

    async def run_system(self, system: str, q: GoldenQuestion, repo, user_id: str) -> dict:
        llm = self.llm or get_llm(self.s)
        with track_usage() as usage:
            t0 = time.perf_counter()
            if system == "naive":
                resp = await answer_naive(
                    q.question, repo, llm=llm, pool=self.pool, embedder=self.embedder, settings=self.s
                )
                refs = _naive_refs(resp)
                gathered = refs
            else:
                req = AskRequest(
                    question=q.question, repo=repo.name, user_id=user_id, thread_id=uuid.uuid4().hex[:8]
                )
                resp = await self.agent.ask(req, repo)
                refs = _agent_refs(resp)
                gathered = _agent_gathered(resp)
            latency = int((time.perf_counter() - t0) * 1000)
        summary = usage.summary()
        cited = [Ref.from_evidence(c) for c in resp.citations]
        out = {
            "answer": resp.answer,
            "hits": hits_at_k(refs, q),
            "recall_any": any(matches(r, q) for r in gathered),
            "cited_hit": any(matches(r, q) for r in cited),
            "citations": [c.label for c in resp.citations],
            "citations_total": resp.verification.total_citations,
            "citations_valid": resp.verification.valid_citations,
            "said_insufficient": said_insufficient(resp.answer, resp.insufficient),
            "fully_abstained": fully_abstained(resp.answer),
            "latency_ms": latency,
            "cost_usd": summary["cost_usd"] if summary["cost_complete"] else None,
            "llm_calls": summary["llm_calls"],
            "tool_calls": len(resp.tool_calls),
            "warnings": resp.warnings,
        }
        if self.judge_llm is not None:
            with track_usage() as judge_usage:
                j = await judge_answer(self.judge_llm, q.question, q.reference, resp.answer)
            out.update(
                judge_score=j.score,
                judge_verdict=j.verdict,
                judge_rationale=j.rationale,
                judge_cost_usd=judge_usage.summary()["cost_usd"],
            )
        return out

    async def evaluate(
        self,
        golden_path: Path | str = DEFAULT_GOLDEN,
        systems: tuple[str, ...] = ("naive", "agent"),
        *,
        limit: int | None = None,
        ids: list[str] | None = None,
        retrieval_only: bool = False,
    ) -> dict[str, Any]:
        golden = load_golden(golden_path)
        repo = repo_db.get_repo(self.pool, golden.repo)
        if repo is None:
            raise RuntimeError(
                f"{golden.repo} is not indexed; run `repomind ingest {golden.repo}@{golden.ref}`"
            )
        if repo.last_indexed_commit != golden.commit:
            log.warning(
                "indexed commit differs from the golden set's pinned commit",
                extra={"indexed": repo.last_indexed_commit, "golden": golden.commit},
            )
        questions = [q for q in golden.questions if not ids or q.id in ids][: limit or None]
        run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        user_id = f"eval-{run_id}"
        systems = ("retrieval",) if retrieval_only else systems
        results: dict[str, list[dict]] = {s: [] for s in systems}
        for i, q in enumerate(questions, start=1):
            for system in systems:
                base = {
                    "id": q.id,
                    "category": q.category,
                    "question": q.question,
                    "answerable": q.answerable,
                    "has_expectations": q.has_expectations,
                }
                try:
                    if system == "retrieval":
                        r = await self.run_retrieval_only(q, repo)
                    else:
                        r = await self.run_system(system, q, repo, user_id)
                except Exception as e:  # recorded, never silently dropped
                    log.exception("eval question failed", extra={"id": q.id, "system": system})
                    r = {"error": f"{type(e).__name__}: {e}"[:300], "hits": {}}
                results[system].append({**base, **r})
                status = r.get("error") or " ".join(f"{k}={int(v)}" for k, v in r.get("hits", {}).items())
                log.info(f"[{i}/{len(questions)}] {system:9s} {q.id}: {status}")
        return {
            "run_id": run_id,
            "repo": golden.repo,
            "ref": golden.ref,
            "commit": repo.last_indexed_commit,
            "golden": str(Path(golden_path).name),
            "models": {"embedding": self.embedder.name}
            if retrieval_only
            else {"llm": self.s.llm_model, "fast": self.s.fast_model, "embedding": self.embedder.name},
            "settings": {
                "max_retries": self.s.max_retries,
                "max_tool_calls": self.s.max_tool_calls,
                "grade_threshold": self.s.grade_threshold,
                "mcp_transport": self.s.mcp_transport,
            },
            "summary": {s: summarize(rs) for s, rs in results.items()},
            "results": results,
        }
