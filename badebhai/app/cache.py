"""Answer cache: the same question on the same commit of a repo is answered instantly.

Keyed by (repo, indexed commit, mode, normalized question), so a sync to a new commit naturally starts
fresh. Only used when the answer cannot depend on who is asking or on an ongoing conversation: no
thread, no long-term memories for that user and repo, and the answer used none. Entries expire
after ANSWER_CACHE_TTL_HOURS (issues and PRs are live data), and a thumbs-down on an answer evicts it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from psycopg.types.json import Jsonb

_NORM = re.compile(r"[^\w]+")


def question_key(question: str) -> str:
    norm = _NORM.sub(" ", question.lower()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


def lookup(pool, repo_id: int, commit: str | None, mode: str, question: str, ttl_hours: int) -> dict | None:
    if not commit or ttl_hours <= 0:
        return None
    with pool.connection() as conn:
        row = conn.execute(
            "UPDATE answer_cache SET hits = hits + 1 WHERE repo_id = %s AND commit = %s AND mode = %s "
            "AND qkey = %s AND created_at > now() - make_interval(hours => %s) RETURNING response, created_at",
            (repo_id, commit, mode, question_key(question), ttl_hours),
        ).fetchone()
    if row is None:
        return None
    response, created_at = row
    return {**response, "cached_at": created_at.isoformat()}


def store(pool, repo_id: int, commit: str | None, mode: str, question: str, response: dict[str, Any]) -> None:
    if not commit:
        return
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO answer_cache (repo_id, commit, mode, qkey, question, response) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (repo_id, commit, mode, qkey) DO UPDATE SET "
            "response = EXCLUDED.response, created_at = now(), hits = 0",
            (repo_id, commit, mode, question_key(question), question, Jsonb(response)),
        )


def evict(pool, repo_id: int, question: str) -> int:
    with pool.connection() as conn:
        return conn.execute(
            "DELETE FROM answer_cache WHERE repo_id = %s AND qkey = %s", (repo_id, question_key(question))
        ).rowcount


def cacheable(response: dict[str, Any]) -> bool:
    return not (response.get("memories_used") or response.get("memories_written")) and bool(
        response.get("answer")
    )
