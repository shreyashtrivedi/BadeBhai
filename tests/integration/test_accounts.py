"""Optional accounts: sign-in, per-user repo lists, private memories, usage limits."""

import re
import subprocess

import pytest
from fastapi.testclient import TestClient

from badebhai.app import auth
from badebhai.app.api import create_app
from badebhai.db.pool import get_pool
from badebhai.ingestion import indexer as indexer_mod
from badebhai.ingestion.indexer import Indexer
from badebhai.llm.fake import FakeLLM
from tests.integration.helpers import toolbox_factory
from tests.integration.test_add_repo import _github, _wait

pytestmark = pytest.mark.db


def _cite_auth(system, messages, schema):
    user = messages[-1]["content"]
    tag = re.search(r"<(ev-[0-9a-f]{8}) ", user).group(1)
    ids = re.findall(rf'<{tag} id="(\d+)"', user)
    return f"Auth lives in app/auth.py [{ids[0]}]."


@pytest.fixture
def setup(db_settings, tiny_repo, monkeypatch):
    s = db_settings.model_copy(
        update={
            "auth_enabled": True,
            "session_secret": "test-secret",
            "asks_per_hour": 3,
            "answer_cache_ttl_hours": 0,
        }
    )

    def fake_clone(src, token=None):
        if not (src.local_path / ".git").exists():
            subprocess.run(["git", "clone", "-q", tiny_repo.as_uri(), str(src.local_path)], check=True)
        src.ref = src.ref or "main"

    monkeypatch.setattr(indexer_mod, "ensure_clone", fake_clone)
    pool = get_pool(s)
    auth.create_user(pool, "root", "rootpass123", "admin")
    auth.create_user(pool, "alice", "alicepass123")
    auth.create_user(pool, "bob", "bobpass1234")
    gh = _github({"acme/tiny": {"size": 1}, "acme/other": {"size": 1}})
    app = create_app(s, llm=FakeLLM({"naive": _cite_auth}), toolbox_factory=toolbox_factory(s), github=gh)
    return s, app, pool


def _login(app, user, password):
    c = TestClient(app)
    c.__enter__()
    r = c.post("/auth/login", json={"username": user, "password": password})
    assert r.status_code == 200, r.text
    return c


def test_everything_but_public_paths_needs_sign_in(setup):
    _, app, _ = setup
    with TestClient(app) as anon:
        assert anon.get("/health").status_code == 200
        assert anon.get("/ui/").status_code == 200
        assert anon.get("/auth/me").json() == {"auth_enabled": True, "user": None}
        r = anon.get("/repos")
        assert r.status_code == 401 and r.json()["auth_required"]
        assert anon.post("/auth/login", json={"username": "alice", "password": "nope"}).status_code == 401
        assert (
            anon.post("/auth/login", json={"username": "ghost", "password": "whatever1"}).status_code == 401
        )


def test_private_repos_memories_and_sharing(setup):
    _, app, pool = setup
    alice, bob, root = (
        _login(app, "alice", "alicepass123"),
        _login(app, "bob", "bobpass1234"),
        _login(app, "root", "rootpass123"),
    )
    try:
        assert alice.get("/auth/me").json()["user"] == {"username": "alice", "role": "member"}
        job = _wait(alice, alice.post("/repos", json={"source": "acme/tiny"}).json()["job_id"])
        assert job["status"] == "done", job
        assert [r["name"] for r in alice.get("/repos").json()] == ["acme/tiny"]
        assert alice.get("/repos").json()[0]["shared"] is False
        assert bob.get("/repos").json() == []  # private to alice
        assert bob.post("/ask", json={"question": "Where is auth?", "repo": "acme/tiny"}).status_code == 404
        assert bob.get("/file", params={"repo": "acme/tiny", "path": "app/auth.py"}).status_code == 404
        assert [r["name"] for r in root.get("/repos").json()] == ["acme/tiny"]  # admins see everything

        # bob adds the same repo: no re-index, it simply joins his list
        r = bob.post("/repos", json={"source": "https://github.com/acme/tiny"})
        assert r.status_code == 200 and r.json()["added_to_your_list"]
        assert (
            bob.post("/ask", json={"question": "Where is auth handled?", "repo": "acme/tiny"}).status_code
            == 200
        )

        # memories belong to the signed-in user whatever user_id the client sends
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO memories (user_id, repo, text, embedding, importance) VALUES "
                "('bob', 'acme/tiny', 'bob works on billing', array_fill(0.1, ARRAY[384])::vector, 0.9)"
            )
        assert alice.get("/memories", params={"user_id": "bob"}).json() == []
        assert [m["text"] for m in bob.get("/memories", params={"user_id": "x"}).json()] == [
            "bob works on billing"
        ]

        # removing: alice leaves, bob is still a member; the last member removes it for real
        assert alice.delete("/repos/acme/tiny").json() == {
            "repo": "acme/tiny",
            "removed_from_your_list": True,
        }
        assert alice.get("/repos").json() == [] and len(bob.get("/repos").json()) == 1
        assert bob.delete("/repos/acme/tiny").json()["files"] > 0
        assert root.get("/repos").json() == []
    finally:
        for c in (alice, bob, root):
            c.__exit__(None, None, None)


def test_shared_repos_are_admin_managed(setup, tiny_repo):
    s, app, _ = setup
    Indexer(s).ingest(str(tiny_repo), name="tiny")  # indexed by the operator: shared
    alice, root = _login(app, "alice", "alicepass123"), _login(app, "root", "rootpass123")
    try:
        assert alice.get("/repos").json()[0]["shared"] is True
        assert alice.delete("/repos/tiny").status_code == 403
        assert alice.patch("/repos/tiny", json={"shared": False}).status_code == 403
        assert root.patch("/repos/tiny", json={"shared": False}).json() == {"repo": "tiny", "shared": False}
        assert alice.get("/repos").json() == []
        assert alice.get("/feedback").status_code == 403 and root.get("/feedback").status_code == 200
    finally:
        alice.__exit__(None, None, None)
        root.__exit__(None, None, None)


def test_question_limits_apply_to_members_not_admins(setup, tiny_repo):
    s, app, pool = setup
    Indexer(s).ingest(str(tiny_repo), name="tiny")
    key = auth.set_api_key(pool, "bob")
    with TestClient(app, headers={"Authorization": f"Bearer {key}"}) as bob:
        assert bob.get("/auth/me").json()["user"]["username"] == "bob"  # API key works too
        for _ in range(3):
            assert (
                bob.post("/ask", json={"question": "Where is auth handled?", "repo": "tiny"}).status_code
                == 200
            )
        limited = bob.post("/ask", json={"question": "Where is auth handled?", "repo": "tiny"})
        assert limited.status_code == 429 and "limit of 3 questions per hour" in limited.json()["error"]
    root = _login(app, "root", "rootpass123")
    try:
        for _ in range(4):
            assert (
                root.post("/ask", json={"question": "Where is auth handled?", "repo": "tiny"}).status_code
                == 200
            )
    finally:
        root.__exit__(None, None, None)


def test_sign_in_attempts_are_throttled(setup):
    _, app, _ = setup
    with TestClient(app) as c:
        codes = [
            c.post("/auth/login", json={"username": "alice", "password": "wrong-pass"}).status_code
            for _ in range(11)
        ]
    assert codes[:10] == [401] * 10 and codes[10] == 429


def test_db_usage_limiter_is_shared_and_survives_restarts(db_settings):
    pool = get_pool(db_settings)
    a, b = auth.DbUsageLimiter(pool), auth.DbUsageLimiter(pool)  # e.g. two API processes
    assert a.check("bob", "ask", 2, 3600) is None
    assert b.check("bob", "ask", 2, 3600) is None
    wait = a.check("bob", "ask", 2, 3600)
    assert wait is not None and 3500 < wait <= 3601
    assert b.check("alice", "ask", 2, 3600) is None and a.check("bob", "ask", 0, 3600) is None
