"""Long-term memory: per (user, repo) facts with embeddings, importance and time decay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
from psycopg.rows import dict_row

from repomind.config import Settings

_COLS = (
    "id, user_id, repo, text, category, importance, created_at, updated_at, last_used_at, use_count, "
    "source_thread, active, superseded_by"
)


@dataclass
class Memory:
    id: int
    user_id: str
    repo: str
    text: str
    category: str
    importance: float
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime
    use_count: int
    source_thread: str | None
    active: bool
    superseded_by: int | None
    similarity: float | None = None
    weight: float | None = None

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("created_at", "updated_at", "last_used_at"):
            d[k] = d[k].isoformat() if d[k] else None
        return d


def decay_factor(idle_days: float, half_life_days: float) -> float:
    """Retrieval weight multiplier for a memory not used for `idle_days`."""
    return 0.5 ** (max(0.0, idle_days) / max(half_life_days, 1e-6))


def retrieval_weight(similarity: float, importance: float, idle_days: float, half_life_days: float) -> float:
    # Profile facts (high importance) stay useful even when not similar to the current question.
    return (0.5 * max(similarity, 0.0) + 0.5 * importance) * decay_factor(idle_days, half_life_days)


class MemoryStore:
    def __init__(self, pool, embedder, settings: Settings):
        self.pool = pool
        self.embedder = embedder
        self.s = settings

    def embed(self, text: str) -> np.ndarray:
        return self.embedder.embed_documents([text])[0]

    # ---------- reads ----------
    def search(self, user_id: str, repo: str, query: str, k: int | None = None) -> list[Memory]:
        qvec = self.embedder.embed_query(query)
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT {_COLS}, 1 - (embedding <=> %(q)s) AS similarity, "
                "EXTRACT(EPOCH FROM now() - last_used_at) / 86400.0 AS idle_days "
                "FROM memories WHERE user_id = %(u)s AND repo = %(r)s AND active "
                "ORDER BY embedding <=> %(q)s LIMIT 50",
                {"q": qvec, "u": user_id, "r": repo},
            )
            rows = cur.fetchall()
        out = []
        for row in rows:
            idle = float(row.pop("idle_days") or 0)
            m = Memory(**row)
            m.similarity = round(float(m.similarity or 0), 4)
            m.weight = round(
                retrieval_weight(m.similarity, m.importance, idle, self.s.memory_half_life_days), 4
            )
            if m.weight >= self.s.memory_min_weight:
                out.append(m)
        out.sort(key=lambda m: m.weight or 0, reverse=True)
        return out[: k or self.s.memory_top_k]

    def similar(self, user_id: str, repo: str, vec: np.ndarray, k: int = 3) -> list[Memory]:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT {_COLS}, 1 - (embedding <=> %(v)s) AS similarity FROM memories "
                "WHERE user_id = %(u)s AND repo = %(r)s AND active ORDER BY embedding <=> %(v)s LIMIT %(k)s",
                {"v": vec, "u": user_id, "r": repo, "k": k},
            )
            return [Memory(**row) for row in cur.fetchall()]

    def get(self, user_id: str, memory_id: int) -> Memory | None:
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT {_COLS} FROM memories WHERE id = %s AND user_id = %s", (memory_id, user_id))
            row = cur.fetchone()
        return Memory(**row) if row else None

    def list(self, user_id: str, repo: str | None = None, include_inactive: bool = False) -> list[Memory]:
        sql = f"SELECT {_COLS} FROM memories WHERE user_id = %s"
        params: list[Any] = [user_id]
        if repo:
            sql += " AND repo = %s"
            params.append(repo)
        if not include_inactive:
            sql += " AND active"
        sql += " ORDER BY importance DESC, updated_at DESC"
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            return [Memory(**r) for r in cur.fetchall()]

    # ---------- writes ----------
    def add(
        self,
        user_id: str,
        repo: str,
        text: str,
        category: str,
        importance: float,
        source_thread: str | None,
        vec: np.ndarray | None = None,
    ) -> Memory:
        vec = vec if vec is not None else self.embed(text)
        with self.pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"INSERT INTO memories (user_id, repo, text, category, embedding, importance, source_thread) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING {_COLS}",
                (user_id, repo, text, category, vec, float(importance), source_thread),
            )
            return Memory(**cur.fetchone())

    def supersede(
        self,
        old: Memory,
        text: str,
        category: str,
        importance: float,
        source_thread: str | None,
        vec: np.ndarray | None = None,
    ) -> Memory:
        """Replace `old` with a corrected fact; the old row is kept (inactive) for auditability."""
        new = self.add(
            old.user_id, old.repo, text, category, max(importance, old.importance), source_thread, vec
        )
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET active = FALSE, superseded_by = %s, updated_at = now() WHERE id = %s",
                (new.id, old.id),
            )
        return new

    def refresh(self, memory_id: int, importance: float) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET importance = GREATEST(importance, %s), last_used_at = now(), "
                "updated_at = now(), use_count = use_count + 1 WHERE id = %s",
                (float(importance), memory_id),
            )

    def touch(self, ids: list[int]) -> None:
        if not ids:
            return
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE memories SET last_used_at = now(), use_count = use_count + 1 WHERE id = ANY(%s)",
                (ids,),
            )

    def delete(self, user_id: str, memory_id: int) -> bool:
        with self.pool.connection() as conn:
            return (
                conn.execute(
                    "DELETE FROM memories WHERE id = %s AND user_id = %s", (memory_id, user_id)
                ).rowcount
                > 0
            )

    def delete_all(self, user_id: str, repo: str | None = None) -> int:
        with self.pool.connection() as conn:
            if repo:
                cur = conn.execute("DELETE FROM memories WHERE user_id = %s AND repo = %s", (user_id, repo))
            else:
                cur = conn.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))
            return cur.rowcount
