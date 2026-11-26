"""Structured outputs for the planning steps, and validation of planned tool calls."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field, create_model

from badebhai.app.mcp_client import ToolSpec
from badebhai.retrieval.hybrid import keyword_terms

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
_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], start=1)}  # fmt: skip
_MONTH_NAMES = "|".join(list(_MONTHS) + [m[:3] for m in _MONTHS if len(m) > 3] + ["sept"])
_ISO = r"\d{4}-\d{2}-\d{2}"
_CODE_CHANGE_Q = re.compile(
    r"(?i)\b((which|what) (code|files?|functions?|modules?)|chang(e|ed|es)|fix(ed|es)?|introduc\w*|commits?|added|removed)\b"
)
_REPO_PATH = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]*)")  # src/requests, src/requests/models.py


def _month(name: str) -> int:
    name = name.lower()
    return _MONTHS.get(name) or next(i for m, i in _MONTHS.items() if m.startswith(name[:3]))


def parse_window(text: str) -> tuple[str, str | None] | None:
    """A concrete time window the user named, as git dates: "May 2024", "in 2023", "between
    2024-05-01 and 2024-05-31", "since March 2024". Relative phrases ("last week") are left alone:
    they are relative to an indexed snapshot that may be old."""
    t = text.lower()
    if m := re.search(rf"(?:between|from)\s+({_ISO})\s+(?:and|to|until)\s+({_ISO})", t):
        return m.group(1), m.group(2)
    if m := re.search(rf"\b({_MONTH_NAMES})\.?\s+(\d{{4}})\b", t):
        y, mo = int(m.group(2)), _month(m.group(1))
        nxt = f"{y + (mo == 12)}-{1 if mo == 12 else mo + 1:02d}-01"
        if re.search(rf"\b(since|after)\s+({_MONTH_NAMES})", t):
            return f"{y}-{mo:02d}-01", None
        return f"{y}-{mo:02d}-01", nxt
    if m := re.search(rf"\b(since|after)\s+({_ISO})", t):
        return m.group(2), None
    if m := re.search(r"\b(?:in|during)\s+((?:19|20)\d{2})\b", t):
        return f"{m.group(1)}-01-01", f"{int(m.group(1)) + 1}-01-01"
    return None


def release_calls(question: str, tools) -> list[dict]:
    """Deterministic first calls for version and time-window questions (small models miss these)."""
    calls: list[dict] = []
    path = next((m for m in _REPO_PATH.findall(question) if not re.fullmatch(r"[\d./]+", m)), None)
    versions = [v.lstrip("vV") for v in _VERSION_Q.findall(question)]
    if versions:
        v = versions[0]
        if "release_notes" in tools:
            calls.append({"tool": "release_notes", "args": {"version": v}})
        if "changes_between_versions" in tools and _CODE_CHANGE_Q.search(question):
            calls.append(
                {
                    "tool": "changes_between_versions",
                    "args": {"version": v, **({"path": path} if path else {})},
                }
            )
    win = parse_window(question)
    if "recent_changes" in tools and (win or (not versions and _RECENT_Q.search(question))):
        # a named window as given; "recently" / "latest" = the last 90 days before the snapshot's newest
        # commit (the tool falls back to the latest commits when that window is empty)
        args = (
            {"since": win[0], "limit": 30, **({"until": win[1]} if win[1] else {})}
            if win
            else {"since": "90 days ago", "limit": 30}
        )
        if path:
            args["path_prefix"] = path.rstrip("/")
        calls.append({"tool": "recent_changes", "args": args})
    return calls


_RECENT_Q = re.compile(
    r"(?i)\b(recent(ly)?|lately|latest (changes?|commits?)|newest|last (few|couple of|several) (commits?|changes?|weeks?|months?))\b"
)


_OVERVIEW_Q = re.compile(
    r"(?i)\b(overview|architecture|main (parts|components|modules|pieces)|high[- ]level|structured|organi[sz]ed|"
    r"layout|where (do|should) i start|new to (this|the) (code|repo|project)|tour|what does (this|the) "
    r"(repo|repository|project|codebase|library|package) do|how is (this|the) (code|repo|project|codebase))"
)
_VERSION_Q = re.compile(r"(?i)\bv?\d+\.\d+(?:\.\d+)?\b")  # "2.31.0": the changelog (docs) says what changed


def topic_terms(text: str) -> list[str]:
    return [t for t in keyword_terms(text) if t not in _GENERIC]


def enrich_query(query: str, sub_question: str) -> str:
    """Add the sub-question's topic words to a query that has (almost) none of its own."""
    own = topic_terms(query)
    missing = [t for t in topic_terms(sub_question) if t not in own]
    if len(own) >= 2 or not missing:
        return query
    return " ".join([query.strip(), *missing[:4]]).strip()


def _stems(text: str) -> set[str]:
    return {t[:5] for t in topic_terms(text)}  # crude stemming: "stripped"/"stripping", "redirect(s)"


def merge_duplicate_subquestions(
    plans: list[SubQuestionPlan], threshold: float = 0.6
) -> list[SubQuestionPlan]:
    """Small models often restate one narrow question as several near-identical sub-questions, which
    multiplies grading work and the chance of a grader false negative. A sub-question is merged into
    an earlier one when their topic words mostly overlap and one's sources contain the other's, so
    "why was X added" (commits) and "is X related to open issues" (issues) stay separate."""
    kept: list[SubQuestionPlan] = []
    for p in plans:
        stems, sources = _stems(p.text), set(p.sources)
        for k in kept:
            k_stems, k_sources = _stems(k.text), set(k.sources)
            union = stems | k_stems
            similar = bool(union) and len(stems & k_stems) / len(union) >= threshold
            if similar and (sources <= k_sources or k_sources <= sources):
                k.sources = list(dict.fromkeys([*k.sources, *p.sources]))
                seen = {(c.tool, c.args_json) for c in k.calls}
                k.calls = [*k.calls, *(c for c in p.calls if (c.tool, c.args_json) not in seen)]
                break
        else:
            kept.append(p.model_copy(deep=True))
    return kept


def repair_plan(
    calls: list[dict],
    sub_question: str,
    sources: list[str],
    query: str,
    tools: dict[str, ToolSpec],
    max_calls: int,
    context: str = "",
    baseline: bool = False,
) -> list[dict]:
    """Enrich generic search queries, drop invented date windows, then make sure each requested
    source (commit history for why/when questions, the changelog for version questions) has a search
    call, within the call budget. `context` is the user's full question (sub-questions like "is it
    related..." lose the topic). Time windows must come from the user's words: a small model sometimes
    writes its own date range into the sub-question, which would otherwise license the invented window.
    `baseline` (first round) also searches code/docs with the sub-question verbatim: small models
    paraphrase away the decisive words ("case-insensitively"), which naive RAG keeps."""
    topic_source = f"{sub_question} {context}".strip()
    has_time = bool(_TIME_Q.search(context or sub_question))
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
    if (
        _VERSION_Q.search(context or sub_question)
        and _HISTORY_Q.search(topic_source)
        and "docs" not in wanted
    ):
        wanted.append("docs")
    present = {c["tool"] for c in fixed}
    first: list[dict] = []
    if baseline:  # the sub-question's own words, as naive RAG would search: the agent never retrieves less
        for src in ("code", "docs"):
            tool = DEFAULT_TOOL[src]
            if (
                src in wanted
                and tool in tools
                and not any(
                    c["tool"] == tool and _norm(c["args"].get("query")) == _norm(sub_question) for c in fixed
                )
            ):
                first.append({"tool": tool, "args": {"query": sub_question}})
    if (
        baseline
        and "repo_overview" in tools
        and "repo_overview" not in present
        and _OVERVIEW_Q.search(topic_source)
    ):
        first.insert(0, {"tool": "repo_overview", "args": {}})  # big-picture questions: start from the map
    if baseline:  # versions -> release notes (+ the release's commits); named time windows -> that window
        special = [c for c in release_calls(context or sub_question, tools) if c["tool"] not in present]
        fixed = [
            c
            for c in fixed
            if not (
                c["tool"] == "recent_changes"
                and special
                and any(x["tool"] == "recent_changes" for x in special)
            )
        ]
        first = special + first
    covered = present | {c["tool"] for c in first}
    base_query = enrich_query(query or sub_question, topic_source)
    last: list[dict] = []
    for src in wanted:
        tool = DEFAULT_TOOL.get(src)
        if (
            tool
            and tool in tools
            and tool not in covered
            and not (src == "commits" and present & {"blame", "file_history", "get_commit"})
        ):
            last.append({"tool": tool, "args": {"query": base_query}})
            covered.add(tool)
    # Over budget: drop the model's calls, second calls to an already-covered tool first.
    model = list(fixed)
    while len(first) + len(model) + len(last) > max_calls and model:
        names = [c["tool"] for c in (*first, *model, *last)]
        dupes = [i for i, c in enumerate(model) if names.count(c["tool"]) > 1]
        model.pop(dupes[-1] if dupes else -1)
    return [*first, *model, *last][:max_calls]


def _norm(text: object) -> str:
    return " ".join(re.findall(r"\w+", str(text or "").lower()))


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


def grade_schema(ids: list[str]) -> type[GradeOutput]:
    """GradeOutput constrained to the given sub-question ids, one entry each. Small models otherwise
    invent grades for sub-questions that do not exist, and every extra token is slow. The count is a
    schema hint (grammar-enforced by Ollama), not a validation rule, so a miscount never costs a retry."""
    grade = create_model(
        "SubQuestionGrade", __base__=SubQuestionGrade, sub_question_id=(Literal[tuple(ids)], ...)
    )
    bounds = {"minItems": len(ids), "maxItems": len(ids)}
    return create_model(
        "GradeOutput", __base__=GradeOutput, grades=(list[grade], Field(json_schema_extra=bounds))
    )


class Replan(BaseModel):
    sub_question_id: str
    query: str = Field(description="Rewritten search query")
    calls: list[PlannedCall] = Field(description="1-3 new tool calls, none already tried")


class RewriteOutput(BaseModel):
    replans: list[Replan]
