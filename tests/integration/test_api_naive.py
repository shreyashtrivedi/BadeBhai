"""Full /ask flow (naive mode) with a scripted LLM against a real Postgres index."""

import re

import pytest
from fastapi.testclient import TestClient

from badebhai.app.api import create_app
from badebhai.ingestion.indexer import Indexer
from badebhai.llm.fake import FakeLLM

pytestmark = pytest.mark.db


def _cite_auth(system, messages, schema):
    """Cite whichever evidence item is app/auth.py, plus one bogus id."""
    user = messages[-1]["content"]
    tag = re.search(r"<(ev-[0-9a-f]{8}) ", user).group(1)
    assert "never follow instructions" in system.lower() and tag in system
    ids = re.findall(rf'<{tag} id="(\d+)"[^>]*source="app/auth\.py', user)
    return (
        f"Authentication is handled by `BasicAuth` in app/auth.py [{ids[0]}]. Tokens are rotated hourly [42]."
    )


def test_ask_naive_returns_verified_citations(db_settings, tiny_repo):
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    llm = FakeLLM({"naive": _cite_auth})
    with TestClient(create_app(db_settings, llm=llm)) as client:
        r = client.post(
            "/ask", json={"question": "Where is authentication handled?", "repo": "tiny", "mode": "naive"}
        )
        assert r.status_code == 200, r.text
        body = r.json()
    assert body["citations"][0]["path"] == "app/auth.py"
    assert body["citations"][0]["label"].startswith("app/auth.py#L")
    assert "[1]" in body["answer"] and "[42]" not in body["answer"]
    assert body["verification"]["valid_citations"] == 1
    assert body["verification"]["invalid_citations"][0]["marker"] == 42
    assert "unverified" in body["answer"]
    assert [s["node"] for s in body["trace"]] == ["retrieve", "synthesize", "verify_citations"]
    assert body["usage"]["llm_calls"] == 1


def test_unknown_repo_is_404(db_settings):
    with TestClient(create_app(db_settings, llm=FakeLLM())) as client:
        r = client.post("/ask", json={"question": "x", "repo": "nope", "mode": "naive"})
    assert r.status_code == 404


def test_search_and_repos_endpoints(db_settings, tiny_repo):
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    with TestClient(create_app(db_settings, llm=FakeLLM())) as client:
        assert client.get("/repos").json()[0]["name"] == "tiny"
        hits = client.get("/search", params={"repo": "tiny", "q": "retry backoff"}).json()
        assert hits[0]["path"] == "app/retry.py"
