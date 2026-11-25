"""Structured outputs for the planning steps, and validation of planned tool calls."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from repomind.app.mcp_client import ToolSpec
from repomind.retrieval.hybrid import keyword_terms

Source = Literal["code", "docs", "commits", "issues"]


class PlannedCall(BaseModel):
    tool: str = Field(description="Tool name from the catalog")
    args_json: str = Field(description="JSON object with the tool arguments, excluding 'repo'")


class SubQuestionPlan(BaseModel):
    text: str = Field(description="A focused sub-question")
    sources: list[Source] = Field(description="Which sources this sub-question needs")
    query: str = Field(description="Best search query for this sub-question")
    calls: list[PlannedCall] = Field(description="1-3 tool calls to run first")


class DecomposeOutput(BaseModel):
    sub_questions: list[SubQuestionPlan]


DEFAULT_TOOL = {
    "code": "search_code",
    "docs": "search_docs",
    "commits": "search_commits",
    "issues": "search_issues",
}


def validate_calls(
    calls: list[PlannedCall] | list[dict], tools: dict[str, ToolSpec], max_calls: int
) -> list[dict]:
    """Keep calls to known tools with parseable, schema-conformant args. `repo` is injected later."""
    out: list[dict] = []
    seen: set[str] = set()
    for c in calls:
        tool = c.tool if isinstance(c, PlannedCall) else c.get("tool", "")
        raw = c.args_json if isinstance(c, PlannedCall) else c.get("args_json", c.get("args", {}))
        spec = tools.get(tool)
        if spec is None:
            continue
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(args, dict):
            continue
        args = {k: v for k, v in args.items() if k in spec.params and k != "repo" and v is not None}
        if any(r not in args for r in spec.required if r != "repo"):
            continue
        sig = tool + json.dumps(args, sort_keys=True, default=str)
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"tool": tool, "args": args})
        if len(out) >= max_calls:
            break
    return out


def default_calls(sources: list[str], query: str, tools: dict[str, ToolSpec]) -> list[dict]:
    calls = []
    for src in sources or ["code"]:
        tool = DEFAULT_TOOL.get(src)
        if tool in tools:
            calls.append({"tool": tool, "args": {"query": query}})
    return calls


# ------------------------------------------------------------------ plan repair
# Small local models often write queries such as "open issue" or skip the commit search a "why"
# question needs. These deterministic fixes keep plans useful whatever model produced them.
_GENERIC = {
    "issue", "issues", "open", "closed", "pr", "prs", "pull", "request", "requests", "commit", "commits",
    "related", "relevant", "added", "add", "change", "changes", "changed", "doc", "docs", "documentation",
    "github", "repo", "repository", "code", "implementation", "reason", "motivation", "history", "any",
    "library", "package", "feature", "features", "support", "way", "used", "logic", "behavior", "behaviour",
}  # fmt: skip
_HISTORY_Q = re.compile(r"(?i)\b(why|when|who|introduc\w*|added|history|originally|motivat\w*|changed)\b")
_SEARCH_TOOLS = {"search_code", "search_docs", "search_commits", "search_issues"}
_TIME_Q = re.compile(
    r"(?i)\b(19|20)\d{2}\b|\b(last|past|this|previous|recent)\s+(\d+\s+)?(days?|weeks?|months?|years?|quarters?|"
    r"releases?|sprints?)\b|\b(since|between|ago|yesterday|today|recently|lately)\b|"
    r"\b(january|february|march|april|june|july|august|september|october|november|december)\b|\bmay\s+\d"
)


def topic_terms(text: str) -> list[str]:
    return [t for t in keyword_terms(text) if t not in _GENERIC]


def enrich_query(query: str, sub_question: str) -> str:
    """Add the sub-question's topic words to a query that has (almost) none of its own."""
    own = topic_terms(query)
    missing = [t for t in topic_terms(sub_question) if t not in own]
    if len(own) >= 2 or not missing:
        return query
    return " ".join([query.strip(), *missing[:4]]).strip()


def repair_plan(
    calls: list[dict],
    sub_question: str,
    sources: list[str],
    query: str,
    tools: dict[str, ToolSpec],
    max_calls: int,
    context: str = "",
) -> list[dict]:
    """Enrich generic search queries, drop invented date windows, then make sure each requested
    source (and commit history for why/when questions) has a search call, within the call budget.
    `context` is the user's full question (sub-questions like "is it related..." lose the topic)."""
    topic_source = f"{sub_question} {context}".strip()
    has_time = bool(_TIME_Q.search(topic_source))
    fixed = []
    for c in calls:
        args = dict(c["args"])
        if c["tool"] in _SEARCH_TOOLS and isinstance(args.get("query"), str):
            args["query"] = enrich_query(args["query"], topic_source)
        if not has_time and c["tool"] in ("search_commits", "search_issues"):
            args.pop("since", None)  # small models invent windows that hide old history
            args.pop("until", None)
        if c["tool"] == "recent_changes" and not has_time:
            continue  # a time-window tool when the question names no time window
        fixed.append({"tool": c["tool"], "args": args})
    wanted = list(dict.fromkeys(sources or ["code"]))
    if _HISTORY_Q.search(sub_question) and "commits" not in wanted:
        wanted.append("commits")
    present = {c["tool"] for c in fixed}
    base_query = enrich_query(query or sub_question, topic_source)
    for src in wanted:
        tool = DEFAULT_TOOL.get(src)
        if (
            tool
            and tool in tools
            and tool not in present
            and not (src == "commits" and present & {"blame", "file_history", "get_commit"})
        ):
            if len(fixed) >= max_calls:
                # make room: a requested source beats a second call to an already-covered tool
                dupes = [i for i, c in enumerate(fixed) if [x["tool"] for x in fixed].count(c["tool"]) > 1]
                if not dupes:
                    break
                fixed.pop(dupes[-1])
            fixed.append({"tool": tool, "args": {"query": base_query}})
            present.add(tool)
    return fixed[:max_calls]


class SubQuestionGrade(BaseModel):
    """Compact on purpose: a bare list of scores costs ~4 output tokens per item (small local models
    are slow at generating, so verbose per-item objects made grading the slowest step by far)."""

    sub_question_id: str
    scores: list[float] = Field(description="Relevance 0.0-1.0 of each new item, in the order listed")
    sufficient: bool = Field(
        description="Do the items scoring >= 0.5 answer the sub-question with confidence?"
    )
    missing: str = Field(description="What is missing if not sufficient, else empty")


class GradeOutput(BaseModel):
    grades: list[SubQuestionGrade]


class Replan(BaseModel):
    sub_question_id: str
    query: str = Field(description="Rewritten search query")
    calls: list[PlannedCall] = Field(description="1-3 new tool calls, none already tried")


class RewriteOutput(BaseModel):
    replans: list[Replan]
