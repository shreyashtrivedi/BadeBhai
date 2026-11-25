"""Auto mode routes simple lookups to the fast path; repeated questions are answered from the cache."""

import re

import pytest
from fastapi.testclient import TestClient

from badebhai.app.api import create_app
from badebhai.db.pool import get_pool
from badebhai.ingestion.indexer import Indexer
from badebhai.llm.fake import FakeLLM
from tests.integration.helpers import toolbox_factory

pytestmark = pytest.mark.db


def _cite_auth(system, messages, schema):
    user = messages[-1]["content"]
    tag = re.search(r"<(ev-[0-9a-f]{8}) ", user).group(1)
    ids = re.findall(rf'<{tag} id="(\d+)"[^>]*source="app/auth\.py', user)
    return f"Authentication is handled by `BasicAuth` in app/auth.py [{ids[0]}]."


@pytest.fixture
def client(db_settings, tiny_repo):
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    llm = FakeLLM({"naive": _cite_auth})
    with TestClient(create_app(db_settings, llm=llm, toolbox_factory=toolbox_factory(db_settings))) as c:
        yield c


def _ask(c, **kw):
    r = c.post("/ask", json={"question": "Where is authentication handled?", "repo": "tiny", **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_auto_mode_sends_a_simple_lookup_down_the_fast_path_and_caches_it(client):
    first = _ask(client)
    assert first["mode"] == "naive" and first["route"]["mode"] == "naive" and not first["cached"]
    assert first["usage"]["llm_calls"] == 1
    again = _ask(client, question="where is AUTHENTICATION handled")  # normalized to the same key
    assert again["cached"] and again["usage"]["llm_calls"] == 0 and again["answer"] == first["answer"]
    assert again["citations"][0]["path"] == "app/auth.py"


def test_conversations_and_personal_memories_bypass_the_cache(client, db_settings):
    _ask(client)
    assert not _ask(client, thread_id="t-1")["cached"]
    with get_pool(db_settings).connection() as conn:
        conn.execute(
            "INSERT INTO memories (user_id, repo, text, embedding, importance) VALUES ('ana', 'tiny', 'x', "
            "array_fill(0.1, ARRAY[384])::vector, 0.9)"
        )
    assert not _ask(client, user_id="ana")["cached"]
    assert _ask(client, user_id="someone-else")["cached"]


def test_thumbs_down_evicts_the_cached_answer(client):
    first = _ask(client)
    assert _ask(client)["cached"]
    fb = {
        "repo": "tiny",
        "question": "Where is authentication handled?",
        "answer": first["answer"],
        "rating": -1,
    }
    assert client.post("/feedback", json=fb).status_code == 201
    assert not _ask(client)["cached"]


def test_explicit_mode_is_respected(client):
    r = _ask(client, mode="naive")
    assert r["mode"] == "naive" and r["route"] is None
