"""Ground truth for citation checks: the `files` table plus the indexed working copy."""

from __future__ import annotations

import subprocess
from pathlib import Path

from repomind.db import repos as repo_db
from repomind.db.repos import RepoRecord
from repomind.ingestion.secrets import redact


class RepoOracle:
    def __init__(self, pool, repo: RepoRecord):
        self.pool = pool
        self.repo = repo
        self._lines: dict[str, int | None] = {}
        self._commits: dict[str, bool] = {}

    def line_count(self, path: str) -> int | None:
        if path not in self._lines:
            self._lines[path] = repo_db.file_line_count(self.pool, self.repo.id, path)
        return self._lines[path]

    def read_lines(self, path: str, start: int, end: int) -> str | None:
        try:
            text = self.repo.resolve(path).read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return None
        lines = redact(text)[0].split("\n")
        return "\n".join(ln.rstrip("\r") for ln in lines[start - 1 : end])

    def commit_exists(self, sha: str) -> bool:
        if sha not in self._commits:
            proc = subprocess.run(
                ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=self.repo.local_path, capture_output=True
            )
            self._commits[sha] = proc.returncode == 0
        return self._commits[sha]

    def find_path(self, partial: str) -> str | None:
        partial = partial.lstrip("./")
        if self.line_count(partial) is not None:
            return partial
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT path FROM files WHERE repo_id = %s AND path LIKE %s LIMIT 2",
                (self.repo.id, "%/" + partial.replace("%", r"\%")),
            ).fetchall()
        return rows[0][0] if len(rows) == 1 else None

    @property
    def root(self) -> Path:
        return Path(self.repo.local_path)
