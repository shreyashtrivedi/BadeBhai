"""Agent over MCP: multi-source answers, streaming, thread history; webhook-driven live updates."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from repomind.app.api import create_app
from repomind.db.pool import get_pool
from repomind.ingestion.indexer import Indexer
from repomind.ingestion.webhook import sign
from repomind.llm.fake import FakeLLM
from tests.integration.helpers import evidence_ids, plan, toolbox_factory

pytestmark = pytest.mark.db

MIXED_PLAN = plan(
    (
        "Where is authentication handled?",
        ["code"],
        "basic auth",
        [("search_code", {"query": "basic auth header"})],
    ),
    (
        "Which commits changed authentication?",
        ["commits"],
        "token",
        [("search_commits", {"query": "token rotation auth"})],
    ),
    (
        "Are there open auth issues?",
        ["issues"],
        "auth",
        [("search_issues", {"query": "basic auth", "state": "open"})],
    ),
)


def cite_one_of_each(system, messages, schema):
    ids = evidence_ids(messages[-1]["content"])
    return (
        f"Authentication is handled by `BasicAuth` in `app/auth.py` [{ids['code'][0]}]. "
        f"Token rotation was added in a later commit [{ids['commit'][0]}]. "
        f"An open issue reports unicode password problems [{ids['issue'][0]}]."
    )


@pytest.fixture
def indexed(db_settings, tiny_repo, git):
    auth = tiny_repo / "app" / "auth.py"
    auth.write_text(auth.read_text() + "\n\ndef rotate_token(old):\n    return 'tok_' + old[::-1]\n")
    git(tiny_repo, "commit", "-qam", "Add token rotation to the auth module")
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    with get_pool(db_settings).connection() as conn:
        conn.execute("UPDATE repos SET github_full_name = 'acme/tiny' WHERE name = 'tiny'")
    return db_settings


def test_agent_answers_from_code_commits_and_issues(indexed):
    llm = FakeLLM({"decompose": MIXED_PLAN, "synthesize": cite_one_of_each})
    app = create_app(indexed, llm=llm, toolbox_factory=toolbox_factory(indexed))
    q = "Where is auth handled, why was token rotation added, and are there open issues about it?"
    with TestClient(app) as client:
        r = client.post("/ask", json={"question": q, "repo": "tiny", "mode": "agent", "user_id": "u1"})
    assert r.status_code == 200, r.text
    body = r.json()
    types = {c["type"] for c in body["citations"]}
    assert types == {"code", "commit", "issue"}  # >= 2 sources, as required
    v = body["verification"]
    assert v["total_citations"] == 3 and v["valid_citations"] == 3
    assert [s["node"] for s in body["trace"]] == [
        "decompose",
        "retrieve",
        "synthesize",
        "verify_citations",
        "finalize",
    ]
    servers = {c["server"] for c in body["tool_calls"]}
    assert servers == {"code_search", "git_history"}
    assert all(c["ok"] for c in body["tool_calls"])
    assert len(body["sub_questions"]) == 3
    commit = next(c for c in body["citations"] if c["type"] == "commit")
    assert commit["url"].startswith("https://github.com/acme/tiny/commit/")
    # the tool catalog reached the planner through MCP discovery
    assert "search_issues" in llm.calls_for("decompose")[0].system


def test_streaming_emits_steps_tools_tokens_and_final(indexed):
    llm = FakeLLM({"decompose": MIXED_PLAN, "synthesize": cite_one_of_each})
    from repomind.app.agent.runner import AgentRunner
    from repomind.app.schemas import AskRequest
    from repomind.db import repos as repo_db
    from repomind.embeddings import get_embedder

    pool = get_pool(indexed)
    runner = AgentRunner(indexed, llm, pool, get_embedder(indexed), toolbox_factory=toolbox_factory(indexed))
    repo = repo_db.get_repo(pool, "tiny")

    async def collect():
        return [e async for e in runner.stream(AskRequest(question="auth?", repo="tiny"), repo)]

    from repomind import compat

    events = compat.run(collect())
    kinds = [e["type"] for e in events]
    assert kinds[0] == "start" and kinds[-1] == "final"
    assert {"step", "tool", "token"} <= set(kinds)
    streamed = "".join(e["text"] for e in events if e["type"] == "token")
    assert "BasicAuth" in streamed
    assert len(events[-1]["response"]["citations"]) == 3


def test_thread_history_carries_into_follow_up(indexed):
    seen = []

    def decompose(system, messages, schema):
        seen.append(messages[-1]["content"])
        return MIXED_PLAN

    llm = FakeLLM({"decompose": decompose, "synthesize": cite_one_of_each})
    app = create_app(indexed, llm=llm, toolbox_factory=toolbox_factory(indexed))
    with TestClient(app) as client:
        first = client.post(
            "/ask", json={"question": "Where is auth handled?", "repo": "tiny", "user_id": "u1"}
        )
        tid = first.json()["thread_id"]
        client.post(
            "/ask",
            json={"question": "And who changed it?", "repo": "tiny", "user_id": "u1", "thread_id": tid},
        )
        # same thread id but another user must not see u1's history
        client.post(
            "/ask", json={"question": "Other user", "repo": "tiny", "user_id": "u2", "thread_id": tid}
        )
    assert "Recent conversation" not in seen[0]
    assert "Where is auth handled?" in seen[1]
    assert "Recent conversation" not in seen[2]


def test_rate_limited_github_degrades_gracefully(indexed):
    from tests.integration.helpers import fake_github

    llm = FakeLLM(
        {
            "decompose": MIXED_PLAN,
            "synthesize": lambda s, m, sc: (
                f"Auth lives in app/auth.py [{evidence_ids(m[-1]['content'])['code'][0]}]."
            ),
        }
    )
    app = create_app(
        indexed, llm=llm, toolbox_factory=toolbox_factory(indexed, fake_github(rate_limited=True))
    )
    with TestClient(app) as client:
        body = client.post("/ask", json={"question": "auth and issues?", "repo": "tiny"}).json()
    assert any("rate limit" in w for w in body["warnings"])
    assert "rate limit" in llm.calls_for("synthesize")[0].messages[-1]["content"]
    assert body["verification"]["valid_citations"] == 1


def test_webhook_push_makes_new_code_searchable_quickly(db_settings, tiny_repo, git):
    db_settings.github_webhook_secret = "s3cret"
    origin = tiny_repo
    Indexer(db_settings).ingest(f"file://{origin.as_posix()}", name="acme/tiny")
    with get_pool(db_settings).connection() as conn:
        conn.execute("UPDATE repos SET github_full_name = 'acme/tiny' WHERE name = 'acme/tiny'")

    before = git(origin, "rev-parse", "HEAD").strip()
    (origin / "app" / "billing.py").write_text("def charge_invoice(amount_cents):\n    return amount_cents\n")
    git(origin, "add", "-A")
    git(origin, "commit", "-qm", "Add invoice charging")
    after = git(origin, "rev-parse", "HEAD").strip()
    payload = {
        "ref": "refs/heads/main",
        "before": before,
        "after": after,
        "repository": {"full_name": "acme/tiny"},
        "commits": [{"added": ["app/billing.py"], "modified": [], "removed": []}],
    }
    body = json.dumps(payload).encode()
    app = create_app(db_settings, llm=FakeLLM(), toolbox_factory=toolbox_factory(db_settings))
    with TestClient(app) as client:
        bad = client.post(
            "/webhooks/github",
            content=body,
            headers={"X-Hub-Signature-256": "sha256=00", "X-GitHub-Event": "push"},
        )
        assert bad.status_code == 401
        t0 = time.time()
        r = client.post(
            "/webhooks/github",
            content=body,
            headers={"X-Hub-Signature-256": sign("s3cret", body), "X-GitHub-Event": "push"},
        )
        assert r.status_code == 202 and r.json()["status"] == "accepted"
        job_id = r.json()["job_id"]
        for _ in range(120):
            job = client.get(f"/jobs/{job_id}").json()
            if job["status"] != "running":
                break
            time.sleep(0.25)
        assert job["status"] == "done", job
        assert job["report"]["files_indexed"] == 1 and job["report"]["mode"] == "incremental"
        hits = client.get("/search", params={"repo": "acme/tiny", "q": "charge invoice"}).json()
        assert hits[0]["path"] == "app/billing.py"
        assert time.time() - t0 < 60
        repos = client.get("/repos").json()
        assert repos[0]["last_indexed_commit"] == after
        other_branch = client.post(
            "/webhooks/github",
            content=json.dumps({**payload, "ref": "refs/heads/feature"}).encode(),
            headers={
                "X-Hub-Signature-256": sign(
                    "s3cret", json.dumps({**payload, "ref": "refs/heads/feature"}).encode()
                ),
                "X-GitHub-Event": "push",
            },
        )
        assert other_branch.json()["status"] == "ignored"
