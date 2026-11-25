"""Memory write policy.

Only durable, user-specific facts are stored: role, focus area, preferences, repeated corrections,
stable project context. Candidates proposed by the LLM extractor are rejected when they are below
the importance threshold, in a non-durable category, look like code, questions, secrets, or
instructions, or are too long. Before inserting, near-duplicates refresh the existing memory and
explicit updates supersede the memory they correct instead of creating a contradicting duplicate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from badebhai.app.memory.store import Memory, MemoryStore
from badebhai.app.security import looks_injected
from badebhai.config import Settings
from badebhai.ingestion.secrets import contains_secret

Category = Literal["role", "focus_area", "preference", "correction", "project_context", "other"]
DURABLE = {"role", "focus_area", "preference", "correction", "project_context"}
MAX_CHARS = 280


class MemoryCandidate(BaseModel):
    text: str = Field(
        description="One short third-person sentence, e.g. 'The user works on the payments module'"
    )
    category: Category
    importance: float = Field(description="0-1: how useful this is across future sessions")
    action: Literal["add", "update", "ignore"]
    target_id: int | None = Field(description="Existing memory id for action=update, else null")
    reason: str = Field(description="Why this is (or is not) worth remembering")


class ExtractionOutput(BaseModel):
    candidates: list[MemoryCandidate]


@dataclass
class Decision:
    text: str
    category: str
    importance: float
    verdict: str  # saved | superseded | refreshed | rejected
    reason: str
    memory_id: int | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_CODE_HINTS = re.compile(
    r"```|\bdef \w+\(|\bclass \w+[:(]|=>|\bfunction\s*\(|;\s*$|[{}]\s*$|\bimport \w+|#include"
)
_FIRST_PERSON = re.compile(
    r"(?i)\b(i|i'm|i am|i've|my|me|mine|we|we're|our|us|myself|prefer|please|always|never|don't|do not|"
    r"stop|remember|call me)\b"
)


def worth_extracting(user_messages: list[str]) -> bool:
    """Cheap pre-filter: skip the extractor LLM call when the user said nothing about themselves."""
    return any(_FIRST_PERSON.search(m) for m in user_messages)


def looks_like_code(text: str) -> bool:
    if _CODE_HINTS.search(text):
        return True
    symbols = sum(text.count(c) for c in "{}[]();=<>")
    return symbols >= 6 and symbols / max(len(text), 1) > 0.05


def rejection_reason(c: MemoryCandidate, settings: Settings) -> str | None:
    text = c.text.strip()
    if c.action == "ignore":
        return "extractor marked it as not worth keeping or a duplicate"
    if c.category not in DURABLE:
        return f"category {c.category!r} is not a durable user fact"
    if c.importance < settings.memory_importance_threshold:
        return f"importance {c.importance:.2f} below threshold {settings.memory_importance_threshold:.2f}"
    if not text or len(text) > MAX_CHARS:
        return "empty or too long to be a single durable fact"
    if text.endswith("?"):
        return "looks like a one-off question"
    if contains_secret(text):
        return "looks like a secret or credential (never stored)"
    if looks_like_code(text):
        return "looks like a code snippet"
    if looks_injected(text):
        return "instruction-like text"
    return None


def apply_candidates(
    store: MemoryStore,
    user_id: str,
    repo: str,
    candidates: list[MemoryCandidate],
    settings: Settings,
    source_thread: str | None,
) -> list[Decision]:
    decisions: list[Decision] = []
    for c in candidates:
        text = " ".join(c.text.split())
        reason = rejection_reason(c, settings)
        if reason:
            decisions.append(Decision(text, c.category, c.importance, "rejected", reason))
            continue
        vec = store.embed(text)
        target: Memory | None = None
        if c.action == "update" and c.target_id is not None:
            target = store.get(user_id, c.target_id)
            if target is not None and (target.repo != repo or not target.active):
                target = None
        if target is not None:
            new = store.supersede(target, text, c.category, c.importance, source_thread, vec)
            decisions.append(
                Decision(
                    text, c.category, c.importance, "superseded", f"replaces memory #{target.id}", new.id
                )
            )
            continue
        near = store.similar(user_id, repo, vec, k=1)
        if near and (near[0].similarity or 0) >= settings.memory_duplicate_similarity:
            store.refresh(near[0].id, c.importance)
            decisions.append(
                Decision(
                    text,
                    c.category,
                    c.importance,
                    "refreshed",
                    f"duplicate of memory #{near[0].id}",
                    near[0].id,
                )
            )
            continue
        new = store.add(user_id, repo, text, c.category, c.importance, source_thread, vec)
        decisions.append(Decision(text, c.category, c.importance, "saved", c.reason[:200], new.id))
    return decisions
