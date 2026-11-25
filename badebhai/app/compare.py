"""Compare repositories: answer one question for each repo, then write a cited comparison.

Each repo gets a normal verified answer (fast path or agent, as auto mode decides). Their citations are
renumbered into one list ([1..n] for the first repo, [n+1..] for the next, each tagged with its repo),
the answers are rewritten to the new numbers, and one more LLM call compares them. The comparison may
only cite numbers from that list; anything else is marked, like the normal citation verifier does.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from datetime import date

from badebhai.app.prompts import load_prompt
from badebhai.app.schemas import AskResponse, Citation, TraceStep, VerificationReport

_MARKER = re.compile(r"\[(\d{1,3})\]")


def renumber(resp: AskResponse, repo: str, offset: int) -> tuple[str, list[Citation]]:
    """The answer with its [n] markers shifted by `offset`, and its citations tagged with `repo`."""
    mapping = {c.id: c.id + offset for c in resp.citations}
    text = _MARKER.sub(
        lambda m: f"[{mapping[int(m.group(1))]}]" if int(m.group(1)) in mapping else m.group(0), resp.answer
    )
    cites = [
        c.model_copy(update={"id": mapping[c.id], "repo": repo, "label": f"{repo}: {c.label}"})
        for c in resp.citations
    ]
    return text, cites


def check_markers(text: str, valid: set[int]) -> tuple[str, list[dict]]:
    bad: list[dict] = []

    def fix(m: re.Match) -> str:
        n = int(m.group(1))
        if n in valid:
            return m.group(0)
        bad.append({"marker": n, "problem": "not a citation from the per-repo answers"})
        return "⚠️(reference not found)"

    return _MARKER.sub(fix, text), bad


async def compare(
    question: str,
    repos: list,
    answer_one: Callable[[object], Awaitable[AskResponse]],
    llm,
) -> AskResponse:
    t0 = time.perf_counter()
    sections, citations, trace, tool_calls, insufficient = [], [], [], [], []
    for repo in repos:
        resp = await answer_one(repo)
        text, cites = renumber(resp, repo.name, len(citations))
        citations += cites
        sections.append(f"### {repo.name}\n{text}")
        trace += [s.model_copy(update={"node": f"{repo.name}: {s.node}"}) for s in resp.trace]
        tool_calls += resp.tool_calls
        insufficient += [f"{repo.name}: {x}" for x in resp.insufficient]

    prompt = load_prompt("compare")
    names = ", ".join(r.name for r in repos)
    system = prompt.render(repos=names, today=date.today().isoformat())
    user = f"Question: {question}\n\n" + "\n\n".join(sections)
    t = time.perf_counter()
    raw = await llm.complete(purpose="compare", system=system, messages=[{"role": "user", "content": user}])
    answer, bad = check_markers(raw, {c.id for c in citations})
    trace.append(
        TraceStep(
            node="compare",
            summary=f"compared {names} with prompt {prompt.id}",
            ms=int((time.perf_counter() - t) * 1000),
        )
    )

    used = {int(n) for n in _MARKER.findall(answer)}
    return AskResponse(
        answer=answer,
        citations=citations,
        verification=VerificationReport(
            total_citations=len(used) + len(bad),
            valid_citations=len(used),
            invalid_citations=bad,
        ),
        trace=trace,
        mode="compare",
        repo=" vs ".join(r.name for r in repos),
        insufficient=insufficient,
        tool_calls=tool_calls,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
