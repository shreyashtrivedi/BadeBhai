"""LangGraph nodes. Each node returns a partial state update and appends one trace step."""

from __future__ import annotations

import asyncio
import contextlib
import time
from datetime import UTC, datetime
from typing import Any

from langgraph.config import get_stream_writer

from repomind.app.agent.planning import DecomposeOutput, default_calls, validate_calls
from repomind.app.agent.state import HISTORY_TURNS, AgentState, deps_from
from repomind.app.citations import verify_answer
from repomind.app.evidence import evidence_from_tool
from repomind.app.mcp_client import ToolBudgetExceeded
from repomind.app.oracle import RepoOracle
from repomind.app.prompts import load_prompt
from repomind.app.schemas import Evidence, SubQuestion
from repomind.app.security import new_tag, render_evidence_list
from repomind.llm.base import LLMError, LLMUnavailable
from repomind.log import get_logger

log = get_logger(__name__)


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _emit(event: dict[str, Any]) -> None:
    """Send a custom stream event (no-op unless the run streams in 'custom' mode)."""
    with contextlib.suppress(Exception):  # outside a streaming context
        get_stream_writer()(event)


def _step(state: AgentState, node: str, summary: str, t0: float, **detail: Any) -> list[dict]:
    step = {"node": node, "summary": summary, "detail": detail, "ms": int((time.perf_counter() - t0) * 1000)}
    _emit({"type": "step", **step})
    return [*state.get("trace", []), step]


def _context_block(state: AgentState) -> str:
    parts = []
    history = state.get("history") or []
    if history:
        turns = "\n".join(
            f"- Q: {h['question'][:300]}\n  A (summary): {h['answer'][:400]}" for h in history[-3:]
        )
        parts.append(f"Recent conversation in this thread:\n{turns}")
    memories = state.get("memories") or []
    if memories:
        parts.append("Known facts about the user:\n" + "\n".join(f"- {m['text']}" for m in memories))
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- decompose
async def decompose(state: AgentState, config) -> dict:
    t0 = time.perf_counter()
    d = deps_from(config)
    s = d.settings
    tools = await d.toolbox.tools()
    catalog = await d.toolbox.catalog_text()
    prompt = load_prompt("decompose")
    system = prompt.render(repo=d.repo.name, today=_today(), max_subq=s.max_subquestions, catalog=catalog)
    ctx = _context_block(state)
    user = (f"{ctx}\n\n" if ctx else "") + f"Question: {state['question']}"
    warnings = list(state.get("warnings", []))
    try:
        plan = await d.llm.complete_json(
            purpose="decompose",
            system=system,
            messages=[{"role": "user", "content": user}],
            schema=DecomposeOutput,
            fast=True,
        )
        plans = plan.sub_questions[: s.max_subquestions]
    except LLMUnavailable:
        raise
    except LLMError as e:
        log.warning("decompose failed; using single-question fallback", extra={"error": str(e)})
        warnings.append("Planning step failed; fell back to a single code + docs search.")
        plans = []

    subs: list[SubQuestion] = []
    for i, p in enumerate(plans, start=1):
        calls = validate_calls(p.calls, tools, s.max_calls_per_subquestion) or default_calls(
            list(p.sources), p.query, tools
        )
        subs.append(SubQuestion(id=f"q{i}", text=p.text, sources=list(p.sources), query=p.query, calls=calls))
    if not subs:
        subs = [
            SubQuestion(
                id="q1",
                text=state["question"],
                sources=["code", "docs"],
                query=state["question"],
                calls=default_calls(["code", "docs"], state["question"], tools),
            )
        ]
    summary = f"{len(subs)} sub-question(s): " + " | ".join(sq.text for sq in subs)
    return {
        "sub_questions": [sq.model_dump() for sq in subs],
        "warnings": warnings,
        "trace": _step(
            state,
            "decompose",
            summary,
            t0,
            prompt=prompt.id,
            sub_questions=[
                {"id": sq.id, "text": sq.text, "sources": sq.sources, "calls": sq.calls} for sq in subs
            ],
        ),
    }


# --------------------------------------------------------------------------- retrieve
async def retrieve(state: AgentState, config) -> dict:
    """Run the planned MCP tool calls of every pending sub-question, all in parallel."""
    t0 = time.perf_counter()
    d = deps_from(config)
    s = d.settings
    subs = [SubQuestion(**x) for x in state["sub_questions"]]
    evidence: dict[str, dict] = {k: dict(v) for k, v in state.get("evidence", {}).items()}
    warnings = list(state.get("warnings", []))

    jobs: list[tuple[SubQuestion, dict]] = []
    for sq in subs:
        if sq.status == "pending":
            jobs.extend((sq, c) for c in sq.calls[: s.max_calls_per_subquestion])
    if len(jobs) > d.toolbox.remaining:
        warnings.append(f"Tool-call budget reached ({s.max_tool_calls}); some planned searches were skipped.")
        jobs = jobs[: d.toolbox.remaining]

    async def run(sq: SubQuestion, call: dict) -> tuple[SubQuestion, dict, dict]:
        args = {**call["args"], "repo": d.repo.name}
        _emit({"type": "tool", "sub_question": sq.id, "tool": call["tool"], "args": call["args"]})
        try:
            payload = await d.toolbox.call(call["tool"], args)
        except ToolBudgetExceeded as e:
            payload = {"error": str(e), "error_type": "budget"}
        return sq, call, payload

    results = await asyncio.gather(*(run(sq, c) for sq, c in jobs))
    calls_detail = []
    for sq, call, payload in results:
        if "error" in payload:
            msg = payload["error"]
            if payload.get("error_type") == "rate_limited":
                msg = f"GitHub issues/PRs unavailable: {msg}"
            if msg not in warnings and payload.get("error_type") in (
                "rate_limited",
                "unsupported",
                "timeout",
            ):
                warnings.append(msg)
            calls_detail.append(
                {"sub_question": sq.id, "tool": call["tool"], "args": call["args"], "error": msg}
            )
            continue
        items = evidence_from_tool(call["tool"], payload)[: s.evidence_per_call]
        keys = []
        for ev in items:
            existing = evidence.get(ev.key)
            if existing:
                if sq.id not in existing["sub_question_ids"]:
                    existing["sub_question_ids"].append(sq.id)
            else:
                ev.sub_question_ids = [sq.id]
                evidence[ev.key] = ev.model_dump()
            if ev.key not in sq.evidence_keys:
                sq.evidence_keys.append(ev.key)
            keys.append(ev.key)
        calls_detail.append(
            {"sub_question": sq.id, "tool": call["tool"], "args": call["args"], "evidence": keys}
        )

    for sq in subs:
        if sq.status == "pending":
            sq.attempts += 1
            sq.calls = []
    n_new = len(evidence) - len(state.get("evidence", {}))
    return {
        "sub_questions": [sq.model_dump() for sq in subs],
        "evidence": evidence,
        "warnings": warnings,
        "trace": _step(
            state,
            "retrieve",
            f"{len(jobs)} MCP tool call(s), {n_new} new evidence item(s)",
            t0,
            calls=calls_detail,
        ),
    }


# --------------------------------------------------------------------------- synthesize
def select_evidence(state: AgentState, limit: int, threshold: float | None) -> list[Evidence]:
    """Pick evidence for the answer: graded-relevant items if grading ran, round-robin by sub-question."""
    evidence = state.get("evidence", {})
    per_sq: list[list[str]] = []
    for sq in state.get("sub_questions", []):
        keys = [k for k in sq["evidence_keys"] if k in evidence]
        if threshold is not None:
            keys = [k for k in keys if (evidence[k].get("relevance") or 0) >= threshold]
            keys.sort(key=lambda k: evidence[k].get("relevance") or 0, reverse=True)
        per_sq.append(keys)
    chosen: list[str] = []
    while len(chosen) < limit and any(per_sq):
        for keys in per_sq:
            while keys and keys[0] in chosen:
                keys.pop(0)
            if keys and len(chosen) < limit:
                chosen.append(keys.pop(0))
    return [Evidence(**evidence[k]) for k in chosen]


async def synthesize(state: AgentState, config) -> dict:
    t0 = time.perf_counter()
    d = deps_from(config)
    s = d.settings
    graded = any(e.get("relevance") is not None for e in state.get("evidence", {}).values())
    items = select_evidence(state, s.max_evidence_for_synthesis, s.grade_threshold if graded else None)
    tag = new_tag()
    prompt = load_prompt("synthesize")
    system = prompt.render(repo=d.repo.name, tag=tag, today=_today())

    sq_lines = []
    for sq in state.get("sub_questions", []):
        status = "INSUFFICIENT" if sq["status"] == "insufficient" else sq["status"].upper()
        note = f" ({sq['note']})" if sq.get("note") else ""
        sq_lines.append(f"- [{status}] {sq['text']}{note}")
    ctx = _context_block(state)
    warnings = state.get("warnings", [])
    user = "\n\n".join(
        x
        for x in [
            ctx,
            f"Question: {state['question']}",
            "Sub-questions investigated:\n" + "\n".join(sq_lines),
            ("Notes:\n" + "\n".join(f"- {w}" for w in warnings)) if warnings else "",
            f"Evidence ({len(items)} items):\n\n" + (render_evidence_list(items, tag) if items else "(none)"),
        ]
        if x
    )
    chunks: list[str] = []
    async for text in d.llm.stream(
        purpose="synthesize", system=system, messages=[{"role": "user", "content": user}]
    ):
        chunks.append(text)
        _emit({"type": "token", "text": text})
    answer_raw = "".join(chunks)
    return {
        "answer_raw": answer_raw,
        "synth_keys": [e.key for e in items],
        "trace": _step(
            state,
            "synthesize",
            f"answer drafted from {len(items)} evidence item(s)",
            t0,
            prompt=prompt.id,
            evidence=[e.key for e in items],
        ),
    }


# --------------------------------------------------------------------------- verify
async def verify_citations(state: AgentState, config) -> dict:
    t0 = time.perf_counter()
    d = deps_from(config)
    evidence = state.get("evidence", {})
    items = [Evidence(**evidence[k]) for k in state.get("synth_keys", [])]
    verified = await asyncio.to_thread(
        verify_answer, state.get("answer_raw", ""), items, RepoOracle(d.pool, d.repo)
    )
    rep = verified.report
    summary = f"{rep.valid_citations}/{rep.total_citations} citations valid"
    if rep.invalid_citations:
        summary += f"; {len(rep.invalid_citations)} stripped"
    if rep.flagged_claims:
        summary += f"; {len(rep.flagged_claims)} claim(s) flagged"
    return {
        "answer": verified.answer,
        "citations": [c.model_dump() for c in verified.citations],
        "verification": rep.model_dump(),
        "trace": _step(state, "verify_citations", summary, t0, **rep.model_dump()),
    }


# --------------------------------------------------------------------------- finalize
async def finalize(state: AgentState, config) -> dict:
    """Append this turn to the thread's short-term history."""
    t0 = time.perf_counter()
    history = [
        *(state.get("history") or []),
        {"question": state["question"], "answer": state.get("answer", "")},
    ]
    return {
        "history": history[-HISTORY_TURNS:],
        "trace": _step(state, "finalize", f"thread history: {len(history[-HISTORY_TURNS:])} turn(s)", t0),
    }
