"""Shared data models: evidence, citations, sub-questions, trace, API payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceType = Literal["code", "doc", "commit", "issue", "pr"]


class Evidence(BaseModel):
    """One retrieved item. `content` is untrusted repository data."""

    key: str  # stable identity, e.g. code:requests/auth.py#L70-L105, commit:<sha>, issue:123
    type: EvidenceType
    source_tool: str
    title: str
    content: str
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    symbol: str | None = None
    sha: str | None = None
    number: int | None = None
    url: str | None = None
    date: str | None = None
    score: float | None = None  # retrieval score
    relevance: float | None = None  # grader score 0-1 (max over sub-questions)
    relevance_by_sq: dict[str, float] = Field(default_factory=dict)
    sub_question_ids: list[str] = Field(default_factory=list)
    suspicious: bool = False  # contains instruction-like text (prompt-injection heuristic)

    @property
    def label(self) -> str:
        if self.type in ("code", "doc") and self.path:
            if self.start_line:
                return f"{self.path}#L{self.start_line}-L{self.end_line}"
            return self.path
        if self.type == "commit" and self.sha:
            return f"commit {self.sha[:10]}"
        if self.type in ("issue", "pr") and self.number is not None:
            return f"{'PR' if self.type == 'pr' else 'issue'} #{self.number}"
        return self.key


class Citation(BaseModel):
    id: int
    evidence_key: str
    type: EvidenceType
    label: str
    title: str
    url: str | None = None
    path: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    sha: str | None = None
    number: int | None = None
    snippet: str = ""
    valid: bool = True
    problem: str | None = None


class VerificationReport(BaseModel):
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: list[dict[str, Any]] = Field(default_factory=list)
    flagged_claims: list[str] = Field(default_factory=list)
    uncited_claims: list[str] = Field(default_factory=list)
    bad_inline_refs: list[str] = Field(default_factory=list)

    @property
    def validity_rate(self) -> float | None:
        return self.valid_citations / self.total_citations if self.total_citations else None


class SubQuestion(BaseModel):
    id: str
    text: str
    sources: list[str]  # code | docs | commits | issues
    query: str  # current search query (rewritten on retry)
    status: Literal["pending", "sufficient", "weak", "insufficient"] = "pending"
    attempts: int = 0
    evidence_keys: list[str] = Field(default_factory=list)
    calls: list[dict[str, Any]] = Field(default_factory=list)  # planned tool calls not yet executed
    tried: list[str] = Field(default_factory=list)  # signatures of tool calls already executed
    queries: list[str] = Field(default_factory=list)  # every query used so far (original + rewrites)
    note: str | None = None


class TraceStep(BaseModel):
    node: str
    summary: str
    detail: dict[str, Any] = Field(default_factory=dict)
    ms: int | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    repo: str
    mode: Literal["agent", "naive"] = "agent"
    user_id: str = "anonymous"
    thread_id: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    verification: VerificationReport
    trace: list[TraceStep]
    mode: str
    repo: str
    thread_id: str | None = None
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    insufficient: list[str] = Field(default_factory=list)
    memories_used: list[dict[str, Any]] = Field(default_factory=list)
    memories_written: list[dict[str, Any]] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
