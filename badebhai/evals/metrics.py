"""Scoring helpers: retrieval hits, abstention detection, aggregation."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

from badebhai.evals.golden import GoldenQuestion

K_VALUES = (1, 3, 5, 10)

_INSUFFICIENT = re.compile(
    r"(?i)insufficient evidence|couldn'?t find|could not find|no evidence|did not find|didn'?t find|"
    r"not (?:implemented|supported|present) in|does(?:n'?t| not) (?:implement|support|contain|include|have|provide|manage|handle|offer)|"
    r"there is no (?:graphql|websocket|http/3|quic)"
)


@dataclass
class Ref:
    """One retrieved or cited item, reduced to what the golden set can match."""

    type: str
    path: str | None = None
    sha: str | None = None
    number: int | None = None

    @classmethod
    def from_evidence(cls, ev: Any) -> Ref:
        get = ev.get if isinstance(ev, dict) else lambda k, d=None: getattr(ev, k, d)
        return cls(type=get("type"), path=get("path"), sha=get("sha"), number=get("number"))


def matches(ref: Ref, q: GoldenQuestion) -> bool:
    if ref.type in ("code", "doc") and ref.path and ref.path in q.expected_files:
        return True
    if ref.type == "commit" and ref.sha and any(ref.sha.startswith(p) for p in q.expected_commits):
        return True
    return ref.type in ("issue", "pr") and q.expected_issue


def hits_at_k(refs: list[Ref], q: GoldenQuestion) -> dict[str, bool]:
    return {f"hit@{k}": any(matches(r, q) for r in refs[:k]) for k in K_VALUES}


def said_insufficient(answer: str, insufficient: list[str] | None = None) -> bool:
    return bool(insufficient) or bool(_INSUFFICIENT.search(answer or ""))


def fully_abstained(answer: str) -> bool:
    return (answer or "").lstrip().lower().startswith("insufficient evidence")


def judge_score(r: dict) -> float:
    """The judge's score, except that abstaining on an answerable question counts as 0: a small
    local judge sometimes rates "insufficient evidence" as correct because it is not wrong."""
    if r.get("answerable") and r.get("fully_abstained"):
        return 0.0
    return r["judge_score"]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
    return vals[idx]


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def summarize(results: list[dict]) -> dict[str, Any]:
    """Aggregate per-question results for one system."""
    answerable = [r for r in results if r["answerable"] and r["has_expectations"] and not r.get("error")]
    unanswerable = [r for r in results if not r["answerable"] and not r.get("error")]
    out: dict[str, Any] = {"questions": len(results), "errors": sum(1 for r in results if r.get("error"))}
    for k in K_VALUES:
        out[f"hit@{k}"] = mean([float(r["hits"][f"hit@{k}"]) for r in answerable])
    if any("recall_any" in r for r in answerable):
        out["recall_any"] = mean([float(r.get("recall_any", False)) for r in answerable])
    if not any(r.get("retrieval_only") for r in results):
        cites = [r for r in results if not r.get("error")]
        total = sum(r["citations_total"] for r in cites)
        out["citation_validity"] = (
            round(sum(r["citations_valid"] for r in cites) / total, 4) if total else None
        )
        out["cited_expected"] = mean([float(r["cited_hit"]) for r in answerable])
        judged = [judge_score(r) for r in results if r.get("judge_score") is not None]
        out["judge_correctness"] = mean(judged)
        out["insufficient_accuracy"] = mean([float(r["said_insufficient"]) for r in unanswerable])
        out["over_abstention"] = mean([float(r["fully_abstained"]) for r in answerable])
        lat = [r["latency_ms"] for r in cites]
        out["latency_p50_ms"] = percentile(lat, 50)
        out["latency_p95_ms"] = percentile(lat, 95)
        costs = [r["cost_usd"] for r in cites if r.get("cost_usd") is not None]
        out["cost_mean_usd"] = round(statistics.fmean(costs), 5) if costs else None
        out["cost_total_usd"] = round(sum(costs), 4) if costs else None
        out["llm_calls_mean"] = mean([r["llm_calls"] for r in cites])
        out["tool_calls_mean"] = mean([r["tool_calls"] for r in cites])
        by_cat: dict[str, list[float]] = {}
        for r in results:
            if r.get("judge_score") is not None:
                by_cat.setdefault(r["category"], []).append(judge_score(r))
        out["judge_by_category"] = {c: mean(v) for c, v in sorted(by_cat.items())}
    return out
