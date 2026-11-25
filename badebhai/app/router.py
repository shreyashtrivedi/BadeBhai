"""Auto mode: send simple lookups down the fast path, everything else to the agent.

The agent plans, searches several sources and checks its evidence, which wins clearly on "how does",
"why", history and change-planning questions, but costs 2-5x the time. On plain single-part lookups
("where is X handled?", "how do I install it?", "does it support Y?") the one-search fast path
answers as well in a third of the time. Replayed on the 38-question eval, these rules send 10
questions down the fast path: judge score 0.783 vs 0.789 for agent-only (within judge noise), mean
latency 88 s vs 110 s. Compound lookups ("where is X and what does it raise?") lost quality on the
fast path, so they go to the agent.

Rules, not a model: they are instant, predictable, and their reason is shown to the user. When in
doubt, the agent is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A lookup: where/which/what-file questions, "how do I <use it>", "does it support X".
_LOOKUP = re.compile(
    r"""(?ix)^\s*(
        where\s+(is|are|does|do|can|in)\b
      | in\s+which\s+(file|module|class|function|package)\b
      | which\s+(file|files|module|modules|class|classes|function|functions|method|methods|data\s+structure|object|package)\b
      | what\s+(file|module|class|function|method|data\s+structure)\b
      | (what|which)\s+\S+(\s+\S+)?\s+(is|are)\s+used\s+(to|for)\b
      | how\s+(do|can|should)\s+(i|you|we)\s+(install|configure|set\s+up|setup|enable|disable|use|run|import|pass|send|set|get|add)\b
      | (does|do|is|are|can)\s+\S+(\s+\S+)?\s+(support|have|provide|include|offer|handle)\b
      | is\s+there\s+(a|an|any)\b
    )"""
)
# Anything that needs reasoning over several pieces, history, or a plan.
_COMPLEX = re.compile(
    r"""(?ix)\b(
        why | when | changed? | changes | history | histor(y|ical) | commit(s|ted)? | version | release[sd]?
      | introduced | deprecated | removed | added\s+in | before | after | since | between
      | issues? | bugs? | pull\s+requests? | prs? | regressions?
      | compare | comparison | difference | differ | versus | vs\.?
      | would\s+i | should\s+i | steps | plan | refactor | implement(ing)? | migrate | add\s+(a|an|new|support)
      | explain | walk\s+me | how\s+does | how\s+do\s+(the|these|they|it)\b | what\s+happens | flow | lifecycle
      | relationship | interact | depend(s|ency|encies)? | architecture | overview | main\s+parts | design
      | best | recommend | trade-?offs? | pros | cons | risk
    )\b"""
)
_VERSION = re.compile(r"\bv?\d+\.\d+(\.\d+)?\b")
_REFERS_BACK = re.compile(r"(?i)\b(it|its|that|this|those|these|they|them|there|above|previous|same|also)\b")
_MAX_FAST_WORDS = 22


@dataclass
class Route:
    mode: str  # "naive" (fast path) | "agent"
    reason: str


def route(question: str, follow_up: bool = False) -> Route:
    q = question.strip()
    words = len(q.split())
    if follow_up and _REFERS_BACK.search(q):
        return Route("agent", "follow-up question: the agent uses the conversation so far")
    if _COMPLEX.search(q) or _VERSION.search(q):
        return Route("agent", "needs reasoning over several sources (how/why/history/planning)")
    if q.count("?") > 1 or re.search(
        r"(?i)\?\s*(and|also)\b|,\s*and\s+(how|why|what|where|which|is|are|does)\b", q
    ):
        return Route("agent", "several questions in one")
    if words > _MAX_FAST_WORDS:
        return Route("agent", "long question")
    if _LOOKUP.search(q):
        if re.search(r"(?i)\b(and|or)\b|,", q):  # "where is X and what does it raise": several parts
            return Route("agent", "a lookup with several parts")
        return Route("naive", "simple lookup: one search answers it")
    return Route("agent", "not a simple lookup")
