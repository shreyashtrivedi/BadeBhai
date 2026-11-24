"""Structured outputs for the planning steps, and validation of planned tool calls."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from repomind.app.mcp_client import ToolSpec

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


class ItemGrade(BaseModel):
    id: int = Field(description="Evidence item id")
    relevance: float = Field(description="0.0-1.0 relevance to the item's sub-question")


class SubQuestionVerdict(BaseModel):
    sub_question_id: str
    sufficient: bool = Field(
        description="Do the items scoring >= 0.5 answer the sub-question with confidence?"
    )
    missing: str = Field(description="What is missing if not sufficient, else empty")


class GradeOutput(BaseModel):
    grades: list[ItemGrade]
    verdicts: list[SubQuestionVerdict]


class Replan(BaseModel):
    sub_question_id: str
    query: str = Field(description="Rewritten search query")
    calls: list[PlannedCall] = Field(description="1-3 new tool calls, none already tried")


class RewriteOutput(BaseModel):
    replans: list[Replan]
