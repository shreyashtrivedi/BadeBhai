"""Public demo: fixed repositories, per-visitor and daily question limits."""

import re

import pytest
from fastapi.testclient import TestClient

from badebhai.app.api import create_app
from badebhai.ingestion.indexer import Indexer
from badebhai.llm.fake import FakeLLM
from tests.integration.helpers import toolbox_factory

pytestmark = pytest.mark.db


def _cite(system, messages, schema):
    tag = re.search(r"<(ev-[0-9a-f]{8}) ", messages[-1]["content"]).group(1)
    first = re.search(rf'<{tag} id="(\d+)"', messages[-1]["content"]).group(1)
    return f"Auth is in app/auth.py [{first}]."


@pytest.fixture
def demo(db_settings, tiny_repo):
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    s = db_settings.model_copy(
        update={
            "demo_mode": True,
            "demo_asks_per_hour": 2,
            "demo_asks_per_day": 3,
            "answer_cache_ttl_hours": 0,
        }
    )
    with TestClient(create_app(s, llm=FakeLLM({"naive": _cite}), toolbox_factory=toolbox_factory(s))) as c:
        yield c


def _ask(c, ip):
    return c.post(
        "/ask",
        json={"question": "Where is authentication handled?", "repo": "tiny", "mode": "naive"},
        headers={"X-Forwarded-For": f"{ip}, 10.0.0.1"},
    )


def test_demo_repositories_are_read_only(demo):
    setup = demo.get("/setup").json()
    assert setup["demo_mode"] is True and setup["source_url"].startswith("https://github.com/")
    for r in (
        demo.post("/repos", json={"source": "pallets/flask"}),
        demo.delete("/repos/tiny"),
        demo.patch("/repos/tiny", json={"shared": False}),
    ):
        assert r.status_code == 403 and "run your own" in r.json()["error"].lower()
    assert demo.get("/feedback").status_code == 403
    assert [r["name"] for r in demo.get("/repos").json()] == ["tiny"]


def test_per_visitor_and_daily_limits(demo):
    assert [_ask(demo, "1.1.1.1").status_code for _ in range(2)] == [200, 200]
    limited = _ask(demo, "1.1.1.1")
    assert limited.status_code == 429 and "per hour per visitor" in limited.json()["error"]
    assert _ask(demo, "2.2.2.2").status_code == 200  # another visitor; 3 answers today in total
    daily = _ask(demo, "3.3.3.3")
    assert daily.status_code == 429 and "daily quota" in daily.json()["error"]


def test_cached_answers_do_not_count_against_the_limits(db_settings, tiny_repo):
    """Clicking an example question whose answer is cached costs no model call, so it is never limited."""
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    s = db_settings.model_copy(
        update={
            "demo_mode": True,
            "demo_asks_per_hour": 1,
            "demo_asks_per_day": 1,
            "answer_cache_ttl_hours": 24,
        }
    )
    with TestClient(create_app(s, llm=FakeLLM({"naive": _cite}), toolbox_factory=toolbox_factory(s))) as c:
        first = _ask(c, "4.4.4.4")
        assert first.status_code == 200 and not first.json().get("cached")
        again = [_ask(c, "4.4.4.4") for _ in range(3)]  # same visitor, over the hourly limit
        assert all(r.status_code == 200 and r.json()["cached"] for r in again)
        other = c.post(  # a new question still needs the model and is limited
            "/ask",
            json={"question": "Where are passwords hashed?", "repo": "tiny", "mode": "naive"},
            headers={"X-Forwarded-For": "4.4.4.4, 10.0.0.1"},
        )
        assert other.status_code == 429
