"""Full and incremental indexing.

Incremental rules:
* Each file's sha256 is stored; unchanged files are never re-chunked or re-embedded.
* `sync` narrows the candidate set with `git diff <last_indexed_commit>..HEAD`, so only changed
  files are even hashed. Files that vanished from the tree are always deleted.
* A per-repo Postgres advisory lock serializes concurrent webhook / sync / poll runs.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from repomind.config import Settings, get_settings
from repomind.db import repos as repo_db
from repomind.db.pool import get_pool
from repomind.embeddings import Embedder, get_embedder
from repomind.ingestion.chunker import CHUNKER_VERSION, Chunk, Chunker
from repomind.ingestion.filters import list_candidate_files, read_text_file
from repomind.ingestion.languages import classify
from repomind.ingestion.secrets import redact
from repomind.ingestion.sources import (
    diff_paths,
    ensure_clone,
    ensure_wiki_clone,
    fetch_commit,
    head_commit,
    parse_source,
    update_managed_clone,
    working_tree_changes,
)
from repomind.log import get_logger

log = get_logger(__name__)

_EMBED_BATCH_CHUNKS = 256
_LOCK_BASE = 910_000


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
    def ingest(self, source: str, ref: str | None = None, name: str | None = None) -> IndexReport:
        src = parse_source(source, self.settings.repos_dir, ref=ref, name=name)
        ensure_clone(src)
        wiki = None
        if self.settings.index_wiki and src.github_full_name and src.managed:
            wiki = ensure_wiki_clone(src.github_full_name, self.settings.repos_dir)
        repo = repo_db.upsert_repo(self.pool, src, wiki)
        return self.reindex(repo, candidates=None, mode="full")

    def sync(
        self, repo_name: str, target_commit: str | None = None, hinted_paths: set[str] | None = None
    ) -> IndexReport:
        """Bring the index up to date with the repo's tracked ref (or `target_commit`)."""
        repo = repo_db.get_repo(self.pool, repo_name)
        if repo is None:
            raise KeyError(f"unknown repo {repo_name!r}; ingest it first")
        path = Path(repo.local_path)
        if repo.managed:
            if target_commit:
                fetch_commit(path, target_commit)
            update_managed_clone(path, repo.ref)
            if repo.wiki_path and repo.github_full_name:
                ensure_wiki_clone(repo.github_full_name, self.settings.repos_dir)
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
        return self.reindex(repo, candidates=candidates, mode=mode)

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
        self, repo: repo_db.RepoRecord, candidates: set[str] | None, mode: str = "full"
    ) -> IndexReport:
        t0 = time.perf_counter()
        report = IndexReport(repo=repo.name, mode=mode)
        with self.pool.connection() as lock_conn:
            lock_conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_BASE + repo.id,))
            lock_conn.commit()
            try:
                self._reindex_locked(repo, candidates, report)
            finally:
                lock_conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_BASE + repo.id,))
                lock_conn.commit()
        report.seconds = round(time.perf_counter() - t0, 2)
        log.info("index complete", extra={k: v for k, v in report.to_dict().items() if k != "changed_paths"})
        return report

    def _reindex_locked(
        self, repo: repo_db.RepoRecord, candidates: set[str] | None, report: IndexReport
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

        pending: list[_PendingFile] = []
        pending_chunks = 0
        for rel in sorted(to_check):
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
        report.files_indexed += len(files)
        report.chunks_written += len(rows)
        report.changed_paths.extend(f.path for f in files)
        log.info("indexed batch", extra={"files": len(files), "chunks": len(rows)})
