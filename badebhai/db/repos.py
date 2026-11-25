"""Data access for the `repos` and `files` tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from badebhai.ingestion.sources import RepoSource, safe_join


@dataclass
class RepoRecord:
    id: int
    name: str
    kind: str
    source: str
    local_path: str
    ref: str | None
    github_full_name: str | None
    wiki_path: str | None
    last_indexed_commit: str | None
    last_indexed_at: datetime | None
    embedding_model: str | None

    @property
    def managed(self) -> bool:
        return self.kind != "local"

    def resolve(self, rel_path: str) -> Path:
        """Absolute path of an indexed file. `wiki/...` paths map into the wiki clone."""
        if rel_path.startswith("wiki/") and self.wiki_path:
            return safe_join(Path(self.wiki_path), rel_path[len("wiki/") :])
        return safe_join(Path(self.local_path), rel_path)

    def file_url(self, path: str, start: int | None = None, end: int | None = None) -> str | None:
        if not self.github_full_name:
            return None
        if path.startswith("wiki/"):
            page = Path(path).stem
            return f"https://github.com/{self.github_full_name}/wiki/{page}"
        ref = self.last_indexed_commit or self.ref or "HEAD"
        url = f"https://github.com/{self.github_full_name}/blob/{ref}/{path}"
        if start:
            url += f"#L{start}" + (f"-L{end}" if end and end != start else "")
        return url

    def to_public(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "ref": self.ref,
            "github": self.github_full_name,
            "last_indexed_commit": self.last_indexed_commit,
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
            "embedding_model": self.embedding_model,
        }


_COLS = (
    "id, name, kind, source, local_path, ref, github_full_name, wiki_path, last_indexed_commit, "
    "last_indexed_at, embedding_model"
)


def _rec(row: dict | None) -> RepoRecord | None:
    return RepoRecord(**row) if row else None


def get_repo(pool: ConnectionPool, name: str) -> RepoRecord | None:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {_COLS} FROM repos WHERE name = %s", (name,))
        return _rec(cur.fetchone())


def find_repo_by_github(pool: ConnectionPool, full_name: str) -> RepoRecord | None:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {_COLS} FROM repos WHERE lower(github_full_name) = lower(%s) ORDER BY id LIMIT 1",
            (full_name,),
        )
        return _rec(cur.fetchone())


def list_repos(pool: ConnectionPool) -> list[RepoRecord]:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT {_COLS} FROM repos ORDER BY name")
        return [RepoRecord(**r) for r in cur.fetchall()]


def upsert_repo(pool: ConnectionPool, src: RepoSource, wiki_path: Path | None) -> RepoRecord:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            INSERT INTO repos (name, kind, source, local_path, ref, github_full_name, wiki_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET kind = EXCLUDED.kind, source = EXCLUDED.source,
                local_path = EXCLUDED.local_path, ref = EXCLUDED.ref,
                github_full_name = EXCLUDED.github_full_name,
                wiki_path = COALESCE(EXCLUDED.wiki_path, repos.wiki_path)
            RETURNING {_COLS}
            """,
            (
                src.name,
                src.kind,
                src.url or str(src.local_path),
                str(src.local_path),
                src.ref,
                src.github_full_name,
                str(wiki_path) if wiki_path else None,
            ),
        )
        return _rec(cur.fetchone())


def mark_indexed(pool: ConnectionPool, repo_id: int, commit: str | None, embedding_model: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            "UPDATE repos SET last_indexed_commit = %s, last_indexed_at = now(), embedding_model = %s WHERE id = %s",
            (commit, embedding_model, repo_id),
        )


def file_hashes(pool: ConnectionPool, repo_id: int) -> dict[str, str]:
    with pool.connection() as conn:
        rows = conn.execute("SELECT path, content_hash FROM files WHERE repo_id = %s", (repo_id,)).fetchall()
    return dict(rows)


def file_line_count(pool: ConnectionPool, repo_id: int, path: str) -> int | None:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT line_count FROM files WHERE repo_id = %s AND path = %s", (repo_id, path)
        ).fetchone()
    return row[0] if row else None


def delete_repo(pool: ConnectionPool, name: str) -> bool:
    with pool.connection() as conn:
        return conn.execute("DELETE FROM repos WHERE name = %s", (name,)).rowcount > 0
