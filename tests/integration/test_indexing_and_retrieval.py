"""Indexer + hybrid search against a real Postgres (hash embedder, no downloads)."""

from pathlib import Path

import pytest

from repomind.db import repos as repo_db
from repomind.db.pool import get_pool
from repomind.embeddings import get_embedder
from repomind.ingestion.indexer import Indexer
from repomind.retrieval.hybrid import hybrid_search

pytestmark = pytest.mark.db


def _chunks(pool, repo_id, path):
    with pool.connection() as conn:
        return conn.execute(
            "SELECT symbol, start_line, end_line, content FROM chunks WHERE repo_id=%s AND path=%s ORDER BY start_line",
            (repo_id, path),
        ).fetchall()


def test_full_ingest_indexes_code_and_docs(db_settings, tiny_repo: Path):
    rep = Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    assert rep.files_indexed == 4  # auth.py, retry.py, README.md, .gitignore
    assert rep.chunks_written > 0
    pool = get_pool(db_settings)
    repo = repo_db.get_repo(pool, "tiny")
    assert repo.kind == "local" and repo.last_indexed_commit
    symbols = [r[0] for r in _chunks(pool, repo.id, "app/auth.py")]
    assert "BasicAuth" in symbols and "verify_token" in symbols
    assert repo_db.file_hashes(pool, repo.id).keys() >= {"app/auth.py", "README.md"}
    assert "logo.png" not in repo_db.file_hashes(pool, repo.id)


def test_hybrid_search_finds_relevant_code(db_settings, tiny_repo: Path):
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    pool = get_pool(db_settings)
    repo = repo_db.get_repo(pool, "tiny")
    hits = hybrid_search(
        pool,
        get_embedder(db_settings),
        repo.id,
        "where is basic authentication handled",
        k=3,
        kinds=("code",),
        settings=db_settings,
    )
    assert hits[0].path == "app/auth.py"
    retry = hybrid_search(
        pool,
        get_embedder(db_settings),
        repo.id,
        "retry with exponential backoff",
        k=3,
        kinds=("code",),
        settings=db_settings,
    )
    assert retry[0].path == "app/retry.py" and retry[0].keyword_rank == 1
    docs = hybrid_search(
        pool, get_embedder(db_settings), repo.id, "authentication", k=3, kinds=("doc",), settings=db_settings
    )
    assert docs and all(h.kind == "doc" for h in docs)
    scoped = hybrid_search(
        pool,
        get_embedder(db_settings),
        repo.id,
        "auth",
        k=5,
        kinds=("code",),
        path_prefix="app/retry",
        settings=db_settings,
    )
    assert all(h.path.startswith("app/retry") for h in scoped)


def test_incremental_sync_reembeds_only_changed_and_deletes_removed(db_settings, tiny_repo: Path, git):
    idx = Indexer(db_settings)
    idx.ingest(str(tiny_repo), name="tiny")
    pool = get_pool(db_settings)
    repo = repo_db.get_repo(pool, "tiny")
    before = repo_db.file_hashes(pool, repo.id)

    # modify auth.py, delete retry.py, add a new module; commit
    auth = tiny_repo / "app" / "auth.py"
    auth.write_text(auth.read_text() + "\n\ndef rotate_api_tokens():\n    return 'rotated'\n")
    (tiny_repo / "app" / "retry.py").unlink()
    (tiny_repo / "app" / "billing.py").write_text("def charge_invoice(amount):\n    return amount * 100\n")
    git(tiny_repo, "add", "-A")
    git(tiny_repo, "commit", "-q", "-m", "rotate tokens, drop retry, add billing")

    rep = idx.sync("tiny")
    assert rep.mode == "incremental"
    assert rep.files_indexed == 2  # auth.py + billing.py; README untouched
    assert rep.files_removed == 1
    assert rep.files_seen == 2  # only the diffed paths still in the tree were read; README never hashed

    after = repo_db.file_hashes(pool, repo.id)
    assert "app/retry.py" not in after
    assert after["README.md"] == before["README.md"]
    assert after["app/auth.py"] != before["app/auth.py"]
    assert _chunks(pool, repo.id, "app/retry.py") == []
    new_hits = hybrid_search(
        pool, get_embedder(db_settings), repo.id, "rotate api tokens", k=2, settings=db_settings
    )
    assert new_hits[0].symbol == "rotate_api_tokens"

    # a second sync with nothing new is a no-op
    again = idx.sync("tiny")
    assert again.files_indexed == 0 and again.files_removed == 0


def test_uncommitted_local_changes_are_indexed_on_sync(db_settings, tiny_repo: Path):
    idx = Indexer(db_settings)
    idx.ingest(str(tiny_repo), name="tiny")
    (tiny_repo / "app" / "auth.py").write_text("def brand_new_login_flow():\n    return True\n")
    rep = idx.sync("tiny")
    assert rep.files_indexed == 1
    pool = get_pool(db_settings)
    repo = repo_db.get_repo(pool, "tiny")
    assert [r[0] for r in _chunks(pool, repo.id, "app/auth.py")] == ["brand_new_login_flow"]


def test_secrets_are_redacted_before_indexing(db_settings, tiny_repo: Path, git):
    (tiny_repo / "config.py").write_text("GITHUB = 'ghp_" + "Z" * 36 + "'\nDEBUG = True\n")
    git(tiny_repo, "add", "-A")
    git(tiny_repo, "commit", "-q", "-m", "oops")
    rep = Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    assert rep.redactions == 1
    pool = get_pool(db_settings)
    repo = repo_db.get_repo(pool, "tiny")
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT content FROM chunks WHERE repo_id=%s AND content LIKE %s", (repo.id, "%ghp_%")
        ).fetchall()
    assert rows == []
