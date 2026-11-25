"""Adding a repository through the web API: checks before download, progress, friendly failures."""

import subprocess
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from badebhai.app.api import create_app
from badebhai.ingestion import indexer as indexer_mod
from badebhai.ingestion.sources import GitError
from badebhai.llm.fake import FakeLLM
from badebhai.mcp_servers.github_client import GitHubClient
from tests.integration.helpers import toolbox_factory

pytestmark = pytest.mark.db


def _github(repos: dict[str, dict], token: str | None = None) -> GitHubClient:
    def handler(request: httpx.Request) -> httpx.Response:
        full = request.url.path.removeprefix("/repos/")
        if full in repos:
            return httpx.Response(200, json={"full_name": full, **repos[full]})
        return httpx.Response(404, json={"message": "Not Found"})

    return GitHubClient(token=token, transport=httpx.MockTransport(handler))


def _client(settings, github):
    return TestClient(
        create_app(settings, llm=FakeLLM(), toolbox_factory=toolbox_factory(settings), github=github)
    )


def _wait(client, job_id):
    for _ in range(240):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.25)
    raise AssertionError("job did not finish")


def test_adding_a_github_repo_shows_progress_and_then_refuses_duplicates(db_settings, tiny_repo, monkeypatch):
    s = db_settings.model_copy(update={"github_token": "ghp_test_secret"})
    seen = {}

    def fake_clone(src, token=None):  # stands in for `git clone https://github.com/...`
        seen["token"] = token
        if not (src.local_path / ".git").exists():
            subprocess.run(["git", "clone", "-q", tiny_repo.as_uri(), str(src.local_path)], check=True)
        src.ref = src.ref or "main"

    monkeypatch.setattr(indexer_mod, "ensure_clone", fake_clone)
    gh = _github({"acme/tiny": {"size": 120, "description": "A tiny app", "default_branch": "main"}}, "x")
    with _client(s, gh) as c:
        setup = c.get("/setup").json()
        assert setup["github_token"] is True and "ghp_test_secret" not in str(setup)

        r = c.post("/repos", json={"source": "https://github.com/acme/tiny"})
        assert r.status_code == 202, r.text
        body = r.json()
        assert body["repo"] == "acme/tiny" and body["about"]["description"] == "A tiny app"
        job = _wait(c, body["job_id"])
        assert job["status"] == "done", job
        assert (
            job["progress"]["stage"] == "indexing" and job["progress"]["done"] == job["progress"]["total"] > 0
        )
        assert job["report"]["files_indexed"] > 0 and job["finished_at"] >= job["started_at"]
        assert seen["token"] == "ghp_test_secret" and "ghp_test_secret" not in str(c.get("/jobs").json())

        assert any(r["name"] == "acme/tiny" for r in c.get("/repos").json())
        again = c.post("/repos", json={"source": "acme/tiny"})
        assert again.status_code == 409 and again.json()["already_indexed"] and again.json()["can_update"]


def test_missing_or_private_repo_says_a_token_would_help(db_settings):
    with _client(db_settings, _github({})) as c:
        assert c.get("/setup").json()["github_token"] is False
        r = c.post("/repos", json={"source": "acme/secret-stuff"})
        assert r.status_code == 404
        assert r.json()["needs_token"] is True and "private" in r.json()["error"]


def test_bad_input_and_server_paths_get_plain_errors(db_settings, tiny_repo):
    with _client(db_settings, _github({})) as c:
        empty = c.post("/repos", json={"source": "   "})
        assert empty.status_code == 400 and "address" in empty.json()["error"]
        local = c.post("/repos", json={"source": str(tiny_repo)})
        assert local.status_code == 400 and "badebhai ingest" in local.json()["error"]
        assert c.get("/repos").json() == []


def test_huge_repos_are_refused_from_the_web(db_settings):
    s = db_settings.model_copy(update={"api_max_repo_mb": 100})
    with _client(s, _github({"big/mono": {"size": 300 * 1024}})) as c:
        r = c.post("/repos", json={"source": "big/mono"})
        assert r.status_code == 413 and "300 MB" in r.json()["error"]


def test_clone_failure_becomes_a_friendly_job_error(db_settings, monkeypatch):
    def fail(src, token=None):
        raise GitError(
            "git clone --quiet https://github.com/acme/ghost.git failed: fatal: could not read Username for "
            "'https://github.com': terminal prompts disabled"
        )

    monkeypatch.setattr(indexer_mod, "ensure_clone", fail)
    with _client(db_settings, _github({"acme/ghost": {"size": 1}})) as c:
        job = _wait(c, c.post("/repos", json={"source": "acme/ghost"}).json()["job_id"])
        assert job["status"] == "failed" and job["needs_token"] is True
        assert "couldn't be downloaded" in job["error"] and "terminal prompts" in job["details"]
