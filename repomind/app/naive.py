"""Naive RAG baseline: one hybrid search, one LLM call, citation verification.

This is the Stage 1 `/ask` path and the baseline the agentic pipeline is evaluated against.
"""

from __future__ import annotations

import asyncio
import time
from itertools import zip_longest

from repomind.app.citations import verify_answer
from repomind.app.evidence import evidence_from_hit
from repomind.app.oracle import RepoOracle
from repomind.app.prompts import load_prompt
from repomind.app.schemas import AskResponse, TraceStep
from repomind.app.security import new_tag, per_item_chars, render_evidence_list
from repomind.config import Settings
from repomind.db.repos import RepoRecord
from repomind.llm.base import LLMProvider
from repomind.retrieval.hybrid import hybrid_search


async def answer_naive(
    question: str,
    repo: RepoRecord,
    *,
    llm: LLMProvider,
    pool,
    embedder,
    settings: Settings,
    k_code: int = 6,
    k_doc: int = 3,
) -> AskResponse:
    t0 = time.perf_counter()
    trace: list[TraceStep] = []

    t = time.perf_counter()
    # Separate quotas: prose docs otherwise outrank code for "where is X" questions.
    code_hits, doc_hits = await asyncio.gather(
        asyncio.to_thread(
            hybrid_search, pool, embedder, repo.id, question, k=k_code, kinds=("code",), settings=settings
        ),
        asyncio.to_thread(
            hybrid_search, pool, embedder, repo.id, question, k=k_doc, kinds=("doc",), settings=settings
        ),
    )
    # Interleave (best code, best doc, ...) so neither source buries the other.
    hits = [h for pair in zip_longest(code_hits, doc_hits) for h in pair if h is not None]
    evidence = [evidence_from_hit(h, repo) for h in hits]
    trace.append(
        TraceStep(
            node="retrieve",
            summary=f"hybrid search returned {len(hits)} chunks",
            detail={"query": question, "evidence": [e.key for e in evidence]},
            ms=int((time.perf_counter() - t) * 1000),
        )
    )

    tag = new_tag()
    item_chars = per_item_chars(settings.evidence_char_budget, len(evidence), 600, 2400)
    prompt = load_prompt("naive")
    system = prompt.render(repo=repo.name, tag=tag)
    user = f"Question: {question}\n\nEvidence ({len(evidence)} items):\n\n{render_evidence_list(evidence, tag, item_chars)}"
    t = time.perf_counter()
    raw = await llm.complete(purpose="naive", system=system, messages=[{"role": "user", "content": user}])
    trace.append(
        TraceStep(
            node="synthesize",
            summary=f"answered with prompt {prompt.id}",
            ms=int((time.perf_counter() - t) * 1000),
        )
    )

    t = time.perf_counter()
    verified = await asyncio.to_thread(verify_answer, raw, evidence, RepoOracle(pool, repo))
    rep = verified.report
    trace.append(
        TraceStep(
            node="verify_citations",
            summary=f"{rep.valid_citations}/{rep.total_citations} citations valid",
            detail=rep.model_dump(),
            ms=int((time.perf_counter() - t) * 1000),
        )
    )
    return AskResponse(
        answer=verified.answer,
        citations=verified.citations,
        verification=rep,
        trace=trace,
        mode="naive",
        repo=repo.name,
        evidence=evidence,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
