"""GitHub bot replies, the Slack command, and comparing repositories, end to end with fakes."""

import json
import re
import shutil
import subprocess
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from badebhai.app.api import create_app
from badebhai.ingestion import indexer as indexer_mod
from badebhai.ingestion.indexer import Indexer
from badebhai.ingestion.webhook import sign
from badebhai.integrations import slack
from badebhai.llm.fake import FakeLLM
from badebhai.mcp_servers.github_client import GitHubClient
from tests.integration.helpers import toolbox_factory

pytestmark = pytest.mark.db


def _cite_first(system, messages, schema):
    user = messages[-1]["content"]
    tag = re.search(r"<(ev-[0-9a-f]{8}) ", user).group(1)
    ids = re.findall(rf'<{tag} id="(\d+)"', user)
    return f"It is handled in app/auth.py [{ids[0]}]."


def _recorder():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, str(request.url), json.loads(request.content or b"{}")))
        return httpx.Response(201, json={"ok": True})

    return calls, httpx.MockTransport(handler)


def _wait_for(calls, n=1, timeout=20):
    for _ in range(int(timeout / 0.2)):
        if len(calls) >= n:
            return
        time.sleep(0.2)
    raise AssertionError(f"expected {n} outgoing call(s), got {calls}")


@pytest.fixture
def github_repo(db_settings, tiny_repo, monkeypatch):
    def fake_clone(src, token=None):
        if not (src.local_path / ".git").exists():
            subprocess.run(["git", "clone", "-q", tiny_repo.as_uri(), str(src.local_path)], check=True)
        src.ref = src.ref or "main"

    monkeypatch.setattr(indexer_mod, "ensure_clone", fake_clone)
    Indexer(db_settings).ingest("acme/tiny")
    return db_settings


def test_github_bot_answers_a_mention_with_a_comment(github_repo):
    s = github_repo.model_copy(
        update={"github_bot_enabled": True, "github_webhook_secret": "hook", "github_token": "ghp_x"}
    )
    calls, transport = _recorder()
    gh = GitHubClient(token="ghp_x", transport=transport)
    app = create_app(s, llm=FakeLLM({"naive": _cite_first}), toolbox_factory=toolbox_factory(s), github=gh)
    payload = {
        "action": "created",
        "comment": {
            "body": "@badebhai where is authentication handled?",
            "author_association": "OWNER",
            "user": {"login": "ana", "type": "User"},
        },
        "issue": {"number": 12, "title": "Auth question"},
        "repository": {"full_name": "acme/tiny"},
    }
    body = json.dumps(payload).encode()
    headers = {"X-Hub-Signature-256": sign("hook", body), "X-GitHub-Event": "issue_comment"}
    with TestClient(app) as c:
        r = c.post("/webhooks/github", content=body, headers=headers)
        assert r.json() == {"status": "accepted", "repo": "acme/tiny", "issue": 12}
        _wait_for(calls)
        stranger = json.dumps(
            {**payload, "comment": {**payload["comment"], "author_association": "NONE"}}
        ).encode()
        r2 = c.post(
            "/webhooks/github",
            content=stranger,
            headers={**headers, "X-Hub-Signature-256": sign("hook", stranger)},
        )
        assert r2.json()["status"] == "ignored"
    method, url, sent = calls[0]
    assert method == "POST" and url.endswith("/repos/acme/tiny/issues/12/comments")
    assert "app/auth.py" in sent["body"] and "citations verified" in sent["body"]


def test_bot_is_off_by_default(github_repo):
    s = github_repo.model_copy(update={"github_webhook_secret": "hook"})
    body = json.dumps({"action": "created"}).encode()
    with TestClient(create_app(s, llm=FakeLLM(), toolbox_factory=toolbox_factory(s))) as c:
        r = c.post(
            "/webhooks/github",
            content=body,
            headers={"X-Hub-Signature-256": sign("hook", body), "X-GitHub-Event": "issue_comment"},
        )
    assert r.json() == {"status": "ignored", "reason": "GITHUB_BOT_ENABLED is off"}


def test_slack_command_acknowledges_then_posts_the_answer(github_repo):
    s = github_repo.model_copy(update={"slack_signing_secret": "shh"})
    calls, transport = _recorder()
    app = create_app(
        s, llm=FakeLLM({"naive": _cite_first}), toolbox_factory=toolbox_factory(s), http_transport=transport
    )
    body = b"text=where+is+authentication+handled%3F&user_id=U9&response_url=https%3A%2F%2Fhooks.slack.com%2Fcommands%2Fabc"
    ts = str(int(time.time()))
    with TestClient(app) as c:
        bad = c.post(
            "/integrations/slack/command",
            content=body,
            headers={"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": "v0=bad"},
        )
        assert bad.status_code == 401
        r = c.post(
            "/integrations/slack/command",
            content=body,
            headers={"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": slack.sign("shh", ts, body)},
        )
        assert r.status_code == 200 and "Looking into it" in r.json()["text"]
        _wait_for(calls)
    _method, url, sent = calls[0]
    assert url == "https://hooks.slack.com/commands/abc" and sent["response_type"] == "in_channel"
    assert "app/auth.py" in sent["text"]


def test_compare_two_repos_with_citations_into_each(db_settings, tiny_repo, tmp_path):
    other = tmp_path / "tiny2"
    shutil.copytree(tiny_repo, other)
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    Indexer(db_settings).ingest(str(other), name="tiny2")

    def compare_llm(system, messages, schema):
        user = messages[-1]["content"]
        assert "### tiny" in user and "### tiny2" in user and "untrusted" in system
        first, second = re.findall(r"\[(\d+)\]", user)[:2]
        return f"Both keep auth in app/auth.py [{first}][{second}]. One also claims magic [99]."

    llm = FakeLLM({"naive": _cite_first, "compare": compare_llm})
    with TestClient(create_app(db_settings, llm=llm, toolbox_factory=toolbox_factory(db_settings))) as c:
        r = c.post(
            "/compare", json={"question": "Where is authentication handled?", "repos": ["tiny", "tiny2"]}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert c.post("/compare", json={"question": "x", "repos": ["tiny", "tiny"]}).status_code == 422
    assert body["mode"] == "compare" and body["repo"] == "tiny vs tiny2"
    assert {c["repo"] for c in body["citations"]} == {"tiny", "tiny2"}
    assert (
        "⚠️(reference not found)" in body["answer"]
        and body["verification"]["invalid_citations"][0]["marker"] == 99
    )
    assert (
        any(s["node"].startswith("tiny2: ") for s in body["trace"]) and body["trace"][-1]["node"] == "compare"
    )
