from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest

from repomind.config import Settings

if sys.platform == "win32":  # async psycopg needs a selector loop on Windows (see repomind/compat.py)
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ADMIN_URL = os.environ.get(
    "TEST_ADMIN_DATABASE_URL", "postgresql://repomind:repomind@localhost:5433/repomind"
)
TEST_DB = "repomind_test"


def _test_db_url() -> str | None:
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        return url
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3) as conn:
            exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone()
            if not exists:
                conn.execute(f"CREATE DATABASE {TEST_DB}")
    except psycopg.Error:
        return None
    return ADMIN_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"


_DB_URL = _test_db_url()


def pytest_collection_modifyitems(config, items):
    skip_db = pytest.mark.skip(reason="Postgres not reachable (run `docker compose up -d db`)")
    skip_live = pytest.mark.skip(reason="live test; set RUN_LIVE=1")
    for item in items:
        if "db" in item.keywords and _DB_URL is None:
            item.add_marker(skip_db)
        if "live" in item.keywords and os.environ.get("RUN_LIVE") != "1":
            item.add_marker(skip_live)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        database_url=_DB_URL or "postgresql://invalid/none",
        data_dir=tmp_path / "data",
        embedding_provider="hash",
        embedding_dim=384,
        llm_provider="fake",
        index_wiki=False,
        log_json=False,
        mcp_transport="inmemory",
        github_token=None,
    )


@pytest.fixture
def db_settings(settings: Settings):
    """Settings bound to a clean test database."""
    from repomind.db.pool import close_pools, reset_database

    reset_database(settings)
    yield settings
    close_pools()


def _git(cwd: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, env=env).stdout.decode()


AUTH_PY = '''"""Authentication helpers."""

import base64


class BasicAuth:
    """Attaches HTTP Basic Authentication to a request."""

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def __call__(self, request):
        token = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        request.headers["Authorization"] = "Basic " + token
        return request


def verify_token(token):
    """Check a bearer token against the configured secret."""
    return token.startswith("tok_")
'''

RETRY_PY = '''"""Retry logic for flaky upstream calls."""

import time

MAX_RETRIES = 3


def with_retries(fn, retries=MAX_RETRIES, backoff=0.5):
    """Call fn, retrying with exponential backoff on ConnectionError."""
    for attempt in range(retries):
        try:
            return fn()
        except ConnectionError:
            time.sleep(backoff * 2 ** attempt)
    return fn()
'''

README = """# Tiny Service

A tiny HTTP client used in RepoMind tests.

## Authentication

Use `BasicAuth` from `app/auth.py` to sign requests.

## Retries

Calls are retried by `with_retries` in `app/retry.py`.
"""


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "tiny"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "auth.py").write_text(AUTH_PY)
    (repo / "app" / "retry.py").write_text(RETRY_PY)
    (repo / "README.md").write_text(README)
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0binary")
    (repo / ".gitignore").write_text("secret_dir/\n")
    (repo / "secret_dir").mkdir()
    (repo / "secret_dir" / "keys.py").write_text("KEY = 1\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial commit")
    return repo


@pytest.fixture
def git():
    return _git
