"""👍/👎 feedback on answers. Thumbs-down answers are the best source of new eval questions."""

from __future__ import annotations

from typing import Any

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

_COLS = "id, created_at, user_id, repo, commit, mode, thread_id, question, answer, rating, comment, citations"


def add(
    pool: ConnectionPool,
    *,
    user_id: str,
    repo: str,
    question: str,
    answer: str,
    rating: int,
    commit: str | None = None,
    mode: str | None = None,
    thread_id: str | None = None,
    comment: str | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> int:
    with pool.connection() as conn:
        row = conn.execute(
            "INSERT INTO feedback (user_id, repo, commit, mode, thread_id, question, answer, rating, comment, "
            "citations) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (
                user_id,
                repo,
                commit,
                mode,
                thread_id,
                question,
                answer,
                rating,
                comment or None,
                Jsonb(citations or []),
            ),
        ).fetchone()
    return int(row[0])


def list_feedback(
    pool: ConnectionPool, repo: str | None = None, rating: int | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    sql = f"SELECT {_COLS} FROM feedback WHERE true"
    params: list[Any] = []
    if repo:
        sql += " AND repo = %s"
        params.append(repo)
    if rating is not None:
        sql += " AND rating = %s"
        params.append(rating)
    sql += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        return [{**r, "created_at": r["created_at"].isoformat()} for r in cur.fetchall()]


def summary(pool: ConnectionPool, repo: str | None = None) -> dict[str, int]:
    sql = "SELECT rating, count(*) FROM feedback" + (" WHERE repo = %s" if repo else "") + " GROUP BY rating"
    with pool.connection() as conn:
        rows = dict(conn.execute(sql, (repo,) if repo else ()).fetchall())
    return {"up": int(rows.get(1, 0)), "down": int(rows.get(-1, 0))}


def to_golden_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Thumbs-down answers as golden-set stubs: the question is real; expectations are left to fill in."""
    out = []
    for i, r in enumerate(rows, 1):
        out.append(
            {
                "id": f"feedback-{r['id']}",
                "category": "TODO",
                "question": r["question"],
                "expected_files": [],
                "expected_commits": [],
                "expected_issue": False,
                "answerable": True,
                "reference": "TODO: the key facts a correct answer must contain",
                "notes": (
                    f"From thumbs-down #{i} on {r['repo']} at {r.get('commit') or 'unknown commit'} "
                    f"({r.get('mode') or 'agent'} mode). User comment: {r.get('comment') or '(none)'}. "
                    f"Answer given: {r['answer'][:400]}"
                ),
            }
        )
    return out
