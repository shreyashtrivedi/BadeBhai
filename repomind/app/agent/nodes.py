"""LangGraph nodes. Each node returns a partial state update and appends one trace step."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from datetime import UTC, datetime
from typing import Any

from langgraph.config import get_stream_writer

from repomind.app.agent.planning import (
    DecomposeOutput,
    GradeOutput,
    RewriteOutput,
    default_calls,
    repair_plan,
    validate_calls,
)
from repomind.app.agent.state import HISTORY_TURNS, AgentState, deps_from
from repomind.app.citations import verify_answer
from repomind.app.evidence import evidence_from_tool
from repomind.app.mcp_client import ToolBudgetExceeded
from repomind.app.memory.policy import ExtractionOutput, apply_candidates, worth_extracting
from repomind.app.oracle import RepoOracle
from repomind.app.prompts import load_prompt
from repomind.app.schemas import Evidence, SubQuestion
from repomind.app.security import new_tag, per_item_chars, render_evidence, render_evidence_list
from repomind.llm.base import LLMError, LLMUnavailable
from repomind.llm.ollama_provider import strip_think
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
        calls = validate_calls(p.calls, tools, s.max_calls_per_subquestion)
        calls = repair_plan(
            calls,
            p.text,
            list(p.sources),
            p.query,
            tools,
            s.max_calls_per_subquestion,
            context=state["question"],
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
        sq.tried.append(call_signature(call))
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
            if sq.query not in sq.queries:
                sq.queries.append(sq.query)
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

            def rel(k: str, sq_id: str = sq["id"]) -> float:
                return float(evidence[k].get("relevance_by_sq", {}).get(sq_id, 0.0))

            keys = [k for k in keys if rel(k) >= threshold]
            keys.sort(key=rel, reverse=True)
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
    if graded and not items:
        # Nothing survived grading: answer deterministically instead of letting the model guess.
        answer = insufficient_answer(state)
        _emit({"type": "token", "text": answer})
        return {
            "answer_raw": answer,
            "synth_keys": [],
            "trace": _step(state, "synthesize", "no evidence passed grading: insufficient evidence", t0),
        }
    tag = new_tag()
    item_chars = per_item_chars(s.evidence_char_budget, len(items), 600, 2400)
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
            f"Evidence ({len(items)} items):\n\n"
            + (render_evidence_list(items, tag, item_chars) if items else "(none)"),
        ]
        if x
    )
    chunks: list[str] = []
    async for text in d.llm.stream(
        purpose="synthesize", system=system, messages=[{"role": "user", "content": user}]
    ):
        chunks.append(text)
        _emit({"type": "token", "text": text})
    answer_raw = strip_think("".join(chunks))
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


def insufficient_answer(state: AgentState) -> str:
    lines = ["Insufficient evidence: I could not find support in the repository for this question."]
    for sq in state.get("sub_questions", []):
        tried = ", ".join(f"`{q}`" for q in (sq.get("queries") or [sq["query"]])[:3])
        lines.append(f"- {sq['text']} (searched: {tried})")
    lines.append("Try naming a file, function, or time range, or check that the right repository is indexed.")
    return "\n".join(lines)


def call_signature(call: dict) -> str:
    return call["tool"] + json.dumps(call["args"], sort_keys=True, default=str)


# --------------------------------------------------------------------------- memory: load
async def load_memory(state: AgentState, config) -> dict:
    """Fetch this user's relevant long-term memories for the repo (thread history comes from the checkpointer)."""
    t0 = time.perf_counter()
    d = deps_from(config)
    memories: list[dict] = []
    if d.memory_store is not None:
        found = await asyncio.to_thread(
            d.memory_store.search, state["user_id"], d.repo.name, state["question"], d.settings.memory_top_k
        )
        await asyncio.to_thread(d.memory_store.touch, [m.id for m in found])
        memories = [
            {
                "id": m.id,
                "text": m.text,
                "category": m.category,
                "importance": m.importance,
                "weight": m.weight,
            }
            for m in found
        ]
    n_hist = len(state.get("history") or [])
    return {
        "memories": memories,
        "trace": _step(
            state,
            "load_memory",
            f"{len(memories)} long-term memorie(s), {n_hist} earlier turn(s) in this thread",
            t0,
            memories=memories,
        ),
    }


# --------------------------------------------------------------------------- grade
async def grade_evidence(state: AgentState, config) -> dict:
    """LLM grader: per-item relevance and a sufficiency verdict for each sub-question just retrieved."""
    t0 = time.perf_counter()
    d = deps_from(config)
    s = d.settings
    subs = [SubQuestion(**x) for x in state["sub_questions"]]
    evidence: dict[str, dict] = {k: dict(v) for k, v in state.get("evidence", {}).items()}
    to_grade = [sq for sq in subs if sq.status == "pending"]

    tag = new_tag()
    # New (ungraded) items per sub-question, capped and interleaved across source types.
    batches: dict[str, list[str]] = {
        sq.id: pick_for_grading(sq, evidence, s.grade_items_per_subquestion) for sq in to_grade
    }
    n_to_grade = sum(len(v) for v in batches.values())
    # The grader only needs enough of each item to judge relevance: half the prompt budget.
    item_chars = per_item_chars(s.evidence_char_budget // 2, n_to_grade, 250, 900)
    blocks = []
    for sq in to_grade:
        items = [
            render_evidence(i, Evidence(**evidence[key]), tag, max_chars=item_chars)
            for i, key in enumerate(batches[sq.id], start=1)
        ]
        graded_before = [
            f"{evidence[k]['title']} (relevance {evidence[k]['relevance_by_sq'][sq.id]:.1f})"
            for k in sq.evidence_keys
            if k in evidence and sq.id in evidence[k].get("relevance_by_sq", {})
        ]
        prior = (
            ("\nAlready graded for this sub-question:\n- " + "\n- ".join(graded_before))
            if graded_before
            else ""
        )
        blocks.append(
            f"## Sub-question {sq.id}: {sq.text}{prior}\nNew items ({len(items)}):\n"
            + ("\n\n".join(items) if items else "(none)")
        )

    verdicts: dict[str, tuple[bool, str]] = {}
    if to_grade:
        prompt = load_prompt("grade")
        try:
            out = await d.llm.complete_json(
                purpose="grade",
                system=prompt.render(repo=d.repo.name, tag=tag),
                messages=[{"role": "user", "content": "\n\n".join(blocks)}],
                schema=GradeOutput,
                fast=True,
            )
            for g in out.grades:
                keys = batches.get(g.sub_question_id)
                if keys is None:
                    continue
                for i, key in enumerate(keys):  # a missing score counts as irrelevant
                    score = max(0.0, min(1.0, float(g.scores[i]))) if i < len(g.scores) else 0.0
                    evidence[key].setdefault("relevance_by_sq", {})[g.sub_question_id] = round(score, 2)
                verdicts[g.sub_question_id] = (g.sufficient, g.missing)
        except LLMUnavailable:
            raise
        except LLMError as e:
            log.warning("grader failed; treating retrieval scores as neutral", extra={"error": str(e)})
            for sq_id, keys in batches.items():
                for key in keys:
                    evidence[key].setdefault("relevance_by_sq", {})[sq_id] = s.grade_threshold

    for ev in evidence.values():
        if ev.get("relevance_by_sq"):
            ev["relevance"] = max(ev["relevance_by_sq"].values())

    budget_left = d.toolbox.remaining > 0
    detail = []
    for sq in to_grade:
        strong = [
            k
            for k in sq.evidence_keys
            if evidence.get(k, {}).get("relevance_by_sq", {}).get(sq.id, 0) >= s.grade_threshold
        ]
        sufficient, missing = verdicts.get(sq.id, (bool(strong), ""))
        if sufficient and strong:
            sq.status, sq.note = "sufficient", None
        elif sq.attempts <= s.max_retries and budget_left:
            sq.status, sq.note = "weak", missing or "evidence too weak"
        else:
            # Out of retries: related evidence -> partial answer; nothing relevant -> insufficient.
            sq.status = "partial" if strong else "insufficient"
            sq.note = (missing or ("not fully answered" if strong else "no relevant evidence found")) + (
                "" if budget_left else " (tool budget exhausted)"
            )
        detail.append(
            {
                "id": sq.id,
                "status": sq.status,
                "attempt": sq.attempts,
                "strong_items": len(strong),
                "missing": missing,
                "scores": {
                    Evidence(**evidence[k]).label: evidence[k]["relevance_by_sq"][sq.id]
                    for k in batches[sq.id]
                },
            }
        )
    summary = ", ".join(f"{x['id']}={x['status']}" for x in detail) or "nothing to grade"
    return {
        "sub_questions": [sq.model_dump() for sq in subs],
        "evidence": evidence,
        "round": state.get("round", 0) + 1,
        "trace": _step(state, "grade_evidence", summary, t0, verdicts=detail, graded_items=n_to_grade),
    }


def pick_for_grading(sq: SubQuestion, evidence: dict[str, dict], cap: int) -> list[str]:
    """Up to `cap` ungraded items for this sub-question, round-robin across code/doc/commit/issue
    so one chatty source cannot crowd out the others. Items left out are graded in a later round."""
    buckets: dict[str, list[str]] = {}
    for key in sq.evidence_keys:
        if key in evidence and sq.id not in evidence[key].get("relevance_by_sq", {}):
            buckets.setdefault(evidence[key]["type"], []).append(key)
    picked: list[str] = []
    while len(picked) < cap and any(buckets.values()):
        for keys in buckets.values():
            if keys and len(picked) < cap:
                picked.append(keys.pop(0))
    return picked


def route_after_grade(state: AgentState) -> str:
    return "rewrite_queries" if any(sq["status"] == "weak" for sq in state["sub_questions"]) else "synthesize"


# --------------------------------------------------------------------------- rewrite
async def rewrite_queries(state: AgentState, config) -> dict:
    """Re-plan weak sub-questions: new queries and follow-up tool calls, never repeating a tried call."""
    t0 = time.perf_counter()
    d = deps_from(config)
    s = d.settings
    tools = await d.toolbox.tools()
    subs = [SubQuestion(**x) for x in state["sub_questions"]]
    evidence = state.get("evidence", {})
    weak = [sq for sq in subs if sq.status == "weak"]

    blocks = []
    for sq in weak:
        leads = sorted(
            (evidence[k] for k in sq.evidence_keys if k in evidence),
            key=lambda e: e.get("relevance_by_sq", {}).get(sq.id, 0),
            reverse=True,
        )[:6]
        lead_lines = "\n".join(
            f"  - {Evidence(**e).label} | {e['title'][:120]} | relevance {e.get('relevance_by_sq', {}).get(sq.id, 0):.1f}"
            for e in leads
        )
        tried = "\n".join(f"  - {t}" for t in sq.tried) or "  - (none)"
        blocks.append(
            f"## {sq.id}: {sq.text}\nMissing: {sq.note}\nQueries used: {sq.queries}\n"
            f"Calls already tried:\n{tried}\nEvidence found so far (leads):\n{lead_lines or '  - (none)'}"
        )
    replans: dict[str, Any] = {}
    prompt = load_prompt("rewrite")
    try:
        out = await d.llm.complete_json(
            purpose="rewrite",
            system=prompt.render(repo=d.repo.name, today=_today(), catalog=await d.toolbox.catalog_text()),
            messages=[{"role": "user", "content": "\n\n".join(blocks)}],
            schema=RewriteOutput,
            fast=True,
        )
        replans = {r.sub_question_id: r for r in out.replans}
    except LLMUnavailable:
        raise
    except LLMError as e:
        log.warning("rewrite failed; using fallback queries", extra={"error": str(e)})

    detail = []
    for sq in weak:
        r = replans.get(sq.id)
        new_query = (r.query if r and r.query.strip() else f"{sq.text}").strip()
        calls = validate_calls(r.calls, tools, s.max_calls_per_subquestion) if r else []
        calls = repair_plan(
            calls,
            sq.text,
            sq.sources,
            new_query,
            tools,
            s.max_calls_per_subquestion,
            context=state["question"],
        )
        calls = [c for c in calls if call_signature(c) not in sq.tried]
        if not calls:
            calls = [
                c for c in default_calls(sq.sources, new_query, tools) if call_signature(c) not in sq.tried
            ]
        if not calls:
            strong = any(
                evidence.get(k, {}).get("relevance_by_sq", {}).get(sq.id, 0) >= s.grade_threshold
                for k in sq.evidence_keys
            )
            sq.status = "partial" if strong else "insufficient"
            sq.note = (sq.note or "") + " (no new searches left to try)"
        else:
            sq.status, sq.query, sq.calls = "pending", new_query, calls
        detail.append({"id": sq.id, "query": new_query, "calls": calls, "status": sq.status})
    return {
        "sub_questions": [sq.model_dump() for sq in subs],
        "trace": _step(
            state, "rewrite_queries", f"re-planned {len(weak)} weak sub-question(s)", t0, replans=detail
        ),
    }


def route_after_rewrite(state: AgentState) -> str:
    return "retrieve" if any(sq["status"] == "pending" for sq in state["sub_questions"]) else "synthesize"


# --------------------------------------------------------------------------- memory: write
async def write_memory(state: AgentState, config) -> dict:
    """Propose memories from the user's own messages and apply the write policy; update thread history."""
    t0 = time.perf_counter()
    d = deps_from(config)
    history = [
        *(state.get("history") or []),
        {"question": state["question"], "answer": state.get("answer", "")},
    ][-HISTORY_TURNS:]
    decisions: list[dict] = []
    summary = "no memory store"
    if d.memory_store is not None:
        user_msgs = [h["question"] for h in history[-3:]]
        if not worth_extracting([state["question"]]):
            summary = "skipped: no statements about the user"
        else:
            existing = await asyncio.to_thread(
                d.memory_store.search, state["user_id"], d.repo.name, state["question"], 8
            )
            known = "\n".join(f"- #{m.id} [{m.category}] {m.text}" for m in existing) or "(none)"
            prompt = load_prompt("memory_extract")
            try:
                out = await d.llm.complete_json(
                    purpose="memory",
                    system=prompt.render(repo=d.repo.name),
                    messages=[
                        {
                            "role": "user",
                            "content": f"Existing memories:\n{known}\n\nUser messages (latest last):\n"
                            + "\n".join(f"- {m}" for m in user_msgs),
                        }
                    ],
                    schema=ExtractionOutput,
                    fast=True,
                )
                applied = await asyncio.to_thread(
                    apply_candidates,
                    d.memory_store,
                    state["user_id"],
                    d.repo.name,
                    out.candidates,
                    d.settings,
                    state.get("thread_id"),
                )
                decisions = [x.to_dict() for x in applied]
                summary = (
                    ", ".join(f"{x['verdict']}: {x['text'][:60]}" for x in decisions)
                    or "nothing worth remembering"
                )
            except LLMUnavailable:
                raise
            except LLMError as e:
                log.warning("memory extraction failed", extra={"error": str(e)})
                summary = "memory extraction failed"
    return {
        "history": history,
        "memories_written": decisions,
        "trace": _step(state, "write_memory", summary, t0, decisions=decisions),
    }
