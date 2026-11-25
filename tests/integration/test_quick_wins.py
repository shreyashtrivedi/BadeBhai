"""Removing repos, the citation code viewer, 👍/👎 feedback, and the editor-facing MCP assistant."""

import subprocess

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from mcp import Client

from badebhai.app.api import create_app
from badebhai.app.mcp_client import _parse
from badebhai.db.pool import get_pool
from badebhai.evals.golden import GoldenQuestion
from badebhai.ingestion import indexer as indexer_mod
from badebhai.ingestion.indexer import Indexer
from badebhai.llm.fake import FakeLLM
from badebhai.mcp_servers import assistant
from tests.integration.helpers import toolbox_factory

pytestmark = pytest.mark.db


def _client(settings):
    return TestClient(create_app(settings, llm=FakeLLM(), toolbox_factory=toolbox_factory(settings)))


def _clone_from(tiny_repo):
    def fake_clone(src, token=None):
        if not (src.local_path / ".git").exists():
            subprocess.run(["git", "clone", "-q", tiny_repo.as_uri(), str(src.local_path)], check=True)
        src.ref = src.ref or "main"

    return fake_clone


def test_removing_a_repo_deletes_index_clone_and_memories(db_settings, tiny_repo, monkeypatch):
    monkeypatch.setattr(indexer_mod, "ensure_clone", _clone_from(tiny_repo))
    Indexer(db_settings).ingest("acme/tiny")
    clone = db_settings.repos_dir / "acme__tiny"
    assert (clone / ".git").exists()
    pool = get_pool(db_settings)
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO memories (user_id, repo, text, embedding, importance) VALUES ('u', 'acme/tiny', 'x', "
            "array_fill(0.1, ARRAY[384])::vector, 0.9)"
        )
    with _client(db_settings) as c:
        assert c.delete("/repos/nope/nothing").status_code == 404
        r = c.delete("/repos/acme/tiny")
        assert r.status_code == 200, r.text
        assert r.json()["files"] > 0 and r.json()["memories"] == 1 and r.json()["deleted_clone"]
        assert c.get("/repos").json() == []
    assert not clone.exists()
    with pool.connection() as conn:
        assert conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 0


def test_removing_a_local_folder_repo_never_touches_the_folder(db_settings, tiny_repo):
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    with _client(db_settings) as c:
        r = c.delete("/repos/tiny", params={"keep_memories": True})
        assert r.status_code == 200 and r.json()["deleted_clone"] is False
    assert (tiny_repo / "app" / "auth.py").exists()


def test_file_viewer_serves_only_indexed_files_with_secrets_redacted(db_settings, tiny_repo):
    (tiny_repo / "app" / "settings.py").write_text(
        'API_KEY = "sk-ant-api03-' + "A" * 40 + '"\nDEBUG = True\n'
    )
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    with _client(db_settings) as c:
        f = c.get("/file", params={"repo": "tiny", "path": "app/auth.py"})
        assert f.status_code == 200 and "class BasicAuth" in f.json()["content"]
        assert f.json()["line_count"] > 0 and f.json()["path"] == "app/auth.py"
        s = c.get("/file", params={"repo": "tiny", "path": "./app/settings.py"}).json()
        assert "DEBUG = True" in s["content"] and "A" * 40 not in s["content"]
        for bad in ("secret_dir/keys.py", "../tiny/app/auth.py", ".git/config", "nope.py"):
            assert c.get("/file", params={"repo": "tiny", "path": bad}).status_code == 404, bad


def test_feedback_is_saved_and_exports_as_eval_stubs(db_settings, tiny_repo):
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    with _client(db_settings) as c:
        base = {
            "repo": "tiny",
            "question": "Where is auth?",
            "answer": "In app/auth.py [1].",
            "mode": "agent",
        }
        assert c.post("/feedback", json={**base, "rating": 1}).status_code == 201
        down = c.post("/feedback", json={**base, "rating": -1, "comment": "missed the token rotation"})
        assert down.status_code == 201
        assert c.post("/feedback", json={**base, "rating": 5}).status_code == 422
        assert c.post("/feedback", json={**base, "repo": "nope", "rating": 1}).status_code == 404
        got = c.get("/feedback", params={"repo": "tiny"}).json()
        assert got["totals"] == {"up": 1, "down": 1} and got["items"][0]["commit"]
    from badebhai.db import feedback as feedback_db

    rows = feedback_db.list_feedback(get_pool(db_settings), "tiny", -1)
    stubs = yaml.safe_load(yaml.safe_dump({"questions": feedback_db.to_golden_candidates(rows)}))["questions"]
    q = GoldenQuestion(**stubs[0])  # loads with the eval's own schema
    assert q.question == "Where is auth?" and "missed the token rotation" in q.notes


async def test_assistant_mcp_server_asks_the_running_api():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/repos" and request.method == "GET":
            return httpx.Response(
                200, json=[{"name": "psf/requests", "ref": "v2.32.3", "last_indexed_commit": "abc"}]
            )
        if request.url.path == "/ask":
            return httpx.Response(
                200,
                json={
                    "answer": "Headers use CaseInsensitiveDict [1].",
                    "citations": [
                        {"id": 1, "type": "code", "label": "structures.py#L13-L80", "url": "u", "title": "t"}
                    ],
                    "verification": {"valid_citations": 1, "total_citations": 1},
                    "latency_ms": 41000,
                },
            )
        if request.url.path == "/repos" and request.method == "POST":
            return httpx.Response(404, json={"error": "github.com/x/y wasn't found.", "needs_token": True})
        return httpx.Response(500)

    server = assistant.build_server("http://badebhai.test", transport=httpx.MockTransport(handler))
    async with Client(server) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        assert tools == {"ask", "list_repos", "add_repo"}
        repos = _parse(await client.call_tool("list_repos", {}))
        assert repos["repos"][0] == {"name": "psf/requests", "ref": "v2.32.3", "ready": True}
        data = _parse(
            await client.call_tool("ask", {"question": "Where are headers stored?", "repo": "psf/requests"})
        )
        assert data["citations_verified"] == "1/1" and data["citations"][0]["label"].startswith(
            "structures.py"
        )
        err = _parse(await client.call_tool("add_repo", {"source": "x/y"}))
        assert err["error_type"] == "http_404" and err["needs_token"] is True
    assert ("POST", "/ask") in seen


def test_missing_clone_is_restored_without_reindexing(db_settings, tiny_repo, monkeypatch):
    """A fresh container on a free host keeps the database but loses the disk: clone back, reuse the index."""
    from pathlib import Path

    from badebhai.db import repos as repo_db

    monkeypatch.setattr(indexer_mod, "ensure_clone", _clone_from(tiny_repo))
    idx = Indexer(db_settings)
    idx.ingest("acme/tiny")
    repo = repo_db.get_repo(get_pool(db_settings), "acme/tiny")
    assert indexer_mod._delete_managed_dir(
        Path(repo.local_path), db_settings.repos_dir
    )  # read-only git files too
    assert idx.restore_clone(repo) is True
    restored = repo_db.get_repo(get_pool(db_settings), "acme/tiny")
    assert (db_settings.repos_dir / "acme__tiny" / ".git").exists()
    assert restored.last_indexed_commit == repo.last_indexed_commit
    assert idx.restore_clone(restored) is False  # nothing to do when the clone is there
