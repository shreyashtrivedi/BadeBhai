"""Full and incremental indexing.

Incremental rules:
* Each file's sha256 is stored; unchanged files are never re-chunked or re-embedded.
* `sync` narrows the candidate set with `git diff <last_indexed_commit>..HEAD`, so only changed
  files are even hashed. Files that vanished from the tree are always deleted.
* A per-repo Postgres advisory lock serializes concurrent webhook / sync / poll runs.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import socket
import stat
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from badebhai.config import Settings, get_settings
from badebhai.db import repos as repo_db
from badebhai.db.pool import get_pool
from badebhai.embeddings import Embedder, get_embedder
from badebhai.ingestion.chunker import CHUNKER_VERSION, Chunk, Chunker
from badebhai.ingestion.filters import list_candidate_files, read_text_file
from badebhai.ingestion.languages import classify
from badebhai.ingestion.secrets import redact
from badebhai.ingestion.sources import (
    checkout_ref,
    diff_paths,
    ensure_clone,
    ensure_wiki_clone,
    fetch_commit,
    head_commit,
    parse_source,
    update_managed_clone,
    working_tree_changes,
)
from badebhai.log import get_logger

log = get_logger(__name__)

_EMBED_BATCH_CHUNKS = 256
_LEASE_TTL_S = 300  # renewed after every embedded batch

Progress = Callable[[str, int, int], None]  # (stage, done, total): "downloading" | "indexing"


def _no_progress(stage: str, done: int, total: int) -> None:
    pass


def _delete_managed_dir(path: Path, repos_dir: Path) -> bool:
    """rmtree, but only strictly inside DATA_DIR/repos (git marks object files read-only on Windows)."""
    path, root = path.resolve(), repos_dir.resolve()
    if path == root or root not in path.parents or not path.exists():
        return False

    def _force(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_force)
    else:  # pragma: no cover
        shutil.rmtree(path, onerror=_force)
    return True


@dataclass
class IndexReport:
    repo: str
    commit: str | None = None
    mode: str = "full"
    files_seen: int = 0
    files_indexed: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    files_skipped: int = 0
    chunks_written: int = 0
    redactions: int = 0
    seconds: float = 0.0
    changed_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["changed_paths"] = d["changed_paths"][:50]
        return d


@dataclass
class _PendingFile:
    path: str
    content_hash: str
    kind: str
    language: str | None
    size: int
    line_count: int
    chunks: list[Chunk]


class Indexer:
    def __init__(self, settings: Settings | None = None, pool=None, embedder: Embedder | None = None):
        self.settings = settings or get_settings()
        self.pool = pool or get_pool(self.settings)
        self.embedder = embedder or get_embedder(self.settings)
        # Stored per repo; a different embedding model or chunker version forces a full re-index.
        self.signature = f"{self.embedder.name}+chunker-v{CHUNKER_VERSION}"
        self.chunker = Chunker(
            self.settings.chunk_max_lines, self.settings.window_lines, self.settings.window_overlap
        )

    # ---------------- entry points ----------------
    def ingest(
        self,
        source: str,
        ref: str | None = None,
        name: str | None = None,
        progress: Progress | None = None,
        allow_local: bool = True,
        owner: str | None = None,
    ) -> IndexReport:
        progress = progress or _no_progress
        token = self.settings.github_token
        src = parse_source(source, self.settings.repos_dir, ref=ref, name=name, allow_local=allow_local)
        progress("downloading", 0, 0)
        ensure_clone(src, token=token)
        wiki = None
        if self.settings.index_wiki and src.github_full_name and src.managed:
            wiki = ensure_wiki_clone(src.github_full_name, self.settings.repos_dir, token=token)
        repo = repo_db.upsert_repo(self.pool, src, wiki, owner=owner)
        return self.reindex(repo, candidates=None, mode="full", progress=progress)

    def sync(
        self,
        repo_name: str,
        target_commit: str | None = None,
        hinted_paths: set[str] | None = None,
        progress: Progress | None = None,
    ) -> IndexReport:
        """Bring the index up to date with the repo's tracked ref (or `target_commit`)."""
        repo = repo_db.get_repo(self.pool, repo_name)
        if repo is None:
            raise KeyError(f"unknown repo {repo_name!r}; ingest it first")
        path = Path(repo.local_path)
        token = self.settings.github_token
        if repo.managed:
            (progress or _no_progress)("downloading", 0, 0)
            if target_commit:
                fetch_commit(path, target_commit, token=token)
            update_managed_clone(path, repo.ref, token=token)
            if repo.wiki_path and repo.github_full_name:
                ensure_wiki_clone(repo.github_full_name, self.settings.repos_dir, token=token)
        candidates: set[str] | None = None
        mode = "full-scan"
        if repo.last_indexed_commit:
            diff = diff_paths(path, repo.last_indexed_commit, "HEAD")
            if diff is not None:
                changed, removed = diff
                candidates = changed | removed | (hinted_paths or set())
                if not repo.managed:
                    candidates |= working_tree_changes(path)
                if repo.wiki_path:
                    candidates |= {f"wiki/{p}" for p in self._wiki_files(repo)}
                mode = "incremental"
        return self.reindex(repo, candidates=candidates, mode=mode, progress=progress)

    def restore_clone(self, repo: repo_db.RepoRecord) -> bool:
        """Re-clone a managed repo whose working copy is gone (a fresh container on a free host) at the
        exact commit that was indexed, so git history and file reads match the index. No re-embedding."""
        if not repo.managed or (Path(repo.local_path) / ".git").exists():
            return False
        src = parse_source(repo.source, self.settings.repos_dir, ref=repo.ref, name=repo.name)
        ensure_clone(src, token=self.settings.github_token)
        if repo.last_indexed_commit:
            checkout_ref(src.local_path, repo.last_indexed_commit)
        wiki = None
        if repo.wiki_path and repo.github_full_name:
            wiki = ensure_wiki_clone(
                repo.github_full_name, self.settings.repos_dir, token=self.settings.github_token
            )
        repo_db.set_local_paths(self.pool, repo.id, src.local_path, wiki)
        log.info("restored missing clone", extra={"repo": repo.name, "path": str(src.local_path)})
        return True

    def remove(self, repo_name: str, delete_memories: bool = True) -> dict:
        """Delete a repo's index and its managed clone. A user's own local folder is never touched."""
        repo = repo_db.get_repo(self.pool, repo_name)
        if repo is None:
            raise KeyError(f"unknown repo {repo_name!r}")
        with self._lease(repo.id):  # waits for any running index of this repo
            counts = repo_db.remove_repo(self.pool, repo.id, repo.name, delete_memories)
        removed_dirs = []
        if repo.managed:
            for path in (repo.local_path, repo.wiki_path):
                if path and _delete_managed_dir(Path(path), self.settings.repos_dir):
                    removed_dirs.append(path)
        log.info("repo removed", extra={"repo": repo.name, **counts})
        return {"repo": repo.name, **counts, "deleted_clone": bool(removed_dirs)}

    @contextmanager
    def _lease(self, repo_id: int, wait_s: float = 1800.0):
        """Hold the repo's indexing lease (a row with an expiry, renewed while we work). Works behind
        connection poolers, and a crashed holder's lease lapses after LEASE_TTL_S."""
        holder = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        deadline = time.monotonic() + wait_s
        while True:
            with self.pool.connection() as conn:
                got = conn.execute(
                    "INSERT INTO index_leases (repo_id, holder, expires_at) "
                    "VALUES (%s, %s, now() + make_interval(secs => %s)) "
                    "ON CONFLICT (repo_id) DO UPDATE SET holder = EXCLUDED.holder, expires_at = EXCLUDED.expires_at "
                    "WHERE index_leases.expires_at < now() RETURNING holder",
                    (repo_id, holder, _LEASE_TTL_S),
                ).fetchone()
            if got:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"another indexing run of repo {repo_id} is still working")
            time.sleep(2)

        def renew() -> None:
            with self.pool.connection() as conn:
                conn.execute(
                    "UPDATE index_leases SET expires_at = now() + make_interval(secs => %s) "
                    "WHERE repo_id = %s AND holder = %s",
                    (_LEASE_TTL_S, repo_id, holder),
                )

        try:
            yield renew
        finally:
            with self.pool.connection() as conn:
                conn.execute("DELETE FROM index_leases WHERE repo_id = %s AND holder = %s", (repo_id, holder))

    # ---------------- core ----------------
    def _wiki_files(self, repo: repo_db.RepoRecord) -> list[str]:
        if not repo.wiki_path or not Path(repo.wiki_path).exists():
            return []
        return [p for p in list_candidate_files(Path(repo.wiki_path)) if classify(p)[0] == "doc"]

    def _current_files(self, repo: repo_db.RepoRecord) -> set[str]:
        files = set(list_candidate_files(Path(repo.local_path)))
        files |= {f"wiki/{p}" for p in self._wiki_files(repo)}
        return files

    def reindex(
        self,
        repo: repo_db.RepoRecord,
        candidates: set[str] | None,
        mode: str = "full",
        progress: Progress | None = None,
    ) -> IndexReport:
        t0 = time.perf_counter()
        report = IndexReport(repo=repo.name, mode=mode)
        with self._lease(repo.id) as renew:
            self._renew = renew
            try:
                self._reindex_locked(repo, candidates, report, progress or _no_progress)
            finally:
                self._renew = None
        report.seconds = round(time.perf_counter() - t0, 2)
        log.info("index complete", extra={k: v for k, v in report.to_dict().items() if k != "changed_paths"})
        return report

    def _reindex_locked(
        self,
        repo: repo_db.RepoRecord,
        candidates: set[str] | None,
        report: IndexReport,
        progress: Progress,
    ) -> None:
        current = self._current_files(repo)
        known = repo_db.file_hashes(self.pool, repo.id)
        force = bool(known) and repo.embedding_model not in (None, self.signature)
        if force:
            log.warning(
                "embedding model or chunker changed; re-indexing everything",
                extra={"old": repo.embedding_model, "new": self.signature},
            )
        removed = set(known) - current
        to_check = current if (candidates is None or force) else (candidates & current)
        report.files_seen = len(to_check)
        progress("indexing", 0, len(to_check))

        pending: list[_PendingFile] = []
        pending_chunks = 0
        for n, rel in enumerate(sorted(to_check)):
            if n % 10 == 0:
                progress("indexing", n, len(to_check))
            got = read_text_file(repo.resolve(rel), self.settings.max_file_bytes)
            if got is None:
                report.files_skipped += 1
                if rel in known:
                    removed.add(rel)
                continue
            raw, text = got
            digest = hashlib.sha256(raw).hexdigest()
            if not force and known.get(rel) == digest:
                report.files_unchanged += 1
                continue
            kind, language = classify(rel)
            clean, n_redacted = redact(text)
            report.redactions += n_redacted
            chunks = self.chunker.chunk(rel, clean, kind, language)
            line_count = len(clean.split("\n")) - (1 if clean.endswith("\n") else 0)
            pending.append(_PendingFile(rel, digest, kind, language, len(raw), line_count, chunks))
            pending_chunks += len(chunks)
            if pending_chunks >= _EMBED_BATCH_CHUNKS:
                self._write(repo.id, pending, report)
                pending, pending_chunks = [], 0
        if pending:
            self._write(repo.id, pending, report)
        progress("indexing", len(to_check), len(to_check))
        if removed:
            with self.pool.connection() as conn:
                conn.execute(
                    "DELETE FROM files WHERE repo_id = %s AND path = ANY(%s)", (repo.id, sorted(removed))
                )
            report.files_removed = len(removed)
            report.changed_paths.extend(sorted(removed))
        report.commit = head_commit(Path(repo.local_path))
        repo_db.mark_indexed(self.pool, repo.id, report.commit, self.signature)

    def _write(self, repo_id: int, files: list[_PendingFile], report: IndexReport) -> None:
        all_chunks = [c for f in files for c in f.chunks]
        vectors = self.embedder.embed_documents([c.embedding_text() for c in all_chunks])
        rows = [
            (
                repo_id,
                c.path,
                c.kind,
                c.language,
                c.symbol,
                c.symbol_type,
                c.start_line,
                c.end_line,
                c.content,
                vectors[i],
                c.search_text(),
            )
            for i, c in enumerate(all_chunks)
        ]
        with self.pool.connection() as conn, conn.cursor() as cur:
            paths = [f.path for f in files]
            cur.execute("DELETE FROM files WHERE repo_id = %s AND path = ANY(%s)", (repo_id, paths))
            cur.executemany(
                "INSERT INTO files (repo_id, path, content_hash, kind, language, size_bytes, line_count) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(repo_id, f.path, f.content_hash, f.kind, f.language, f.size, f.line_count) for f in files],
            )
            if rows:
                cur.executemany(
                    "INSERT INTO chunks (repo_id, path, kind, language, symbol, symbol_type, start_line, "
                    "end_line, content, embedding, tsv) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s))",
                    rows,
                )
        if getattr(self, "_renew", None):
            self._renew()
        report.files_indexed += len(files)
        report.chunks_written += len(rows)
        report.changed_paths.extend(f.path for f in files)
        log.info("indexed batch", extra={"files": len(files), "chunks": len(rows)})
