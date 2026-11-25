"""The repository map: built from the index, exposed as an MCP tool, used for big-picture questions."""

import pytest
from fastapi.testclient import TestClient

from badebhai.app.api import create_app
from badebhai.app.citations import check_evidence
from badebhai.app.evidence import evidence_from_tool
from badebhai.app.oracle import RepoOracle
from badebhai.db import repos as repo_db
from badebhai.db.pool import get_pool
from badebhai.ingestion.indexer import Indexer
from badebhai.llm.fake import FakeLLM
from badebhai.retrieval.overview import build_overview, doc_summary, leading_doc
from tests.integration.helpers import toolbox_factory

pytestmark = pytest.mark.db


def test_leading_doc_and_summary_find_module_docstrings():
    py = '"""\nrequests.utils\n~~~~~~~~~~~~~~\n\nThis module provides utility functions. More text.\n"""\nimport os\n'
    doc = leading_doc(py)
    assert doc.split("\n")[-1] == '"""' and "import os" not in doc
    assert doc_summary(doc) == "This module provides utility functions."
    js = "// Router for the HTTP API.\n// Handles auth.\nconst x = 1;\n"
    assert leading_doc(js) == "// Router for the HTTP API.\n// Handles auth."
    assert leading_doc("import os\n") is None


def test_map_lists_directories_main_files_and_verifiable_evidence(db_settings, tiny_repo):
    (tiny_repo / "app" / "auth.py").write_text(
        '"""Authentication helpers for the tiny app."""\n\nclass BasicAuth:\n    pass\n\n\ndef rotate(x):\n    return x\n'
    )
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    pool = get_pool(db_settings)
    repo = repo_db.get_repo(pool, "tiny")
    out = build_overview(pool, repo)
    text = out["items"][0]["content"]
    assert out["items"][0]["type"] == "map" and "Repository map of tiny" in text
    assert "app/" in text and "BasicAuth" in text and "Main files" in text
    assert "Authentication helpers for the tiny app" in text
    evidence = evidence_from_tool("repo_overview", out)
    oracle = RepoOracle(pool, repo)
    assert evidence[0].type == "map" and evidence[0].label == "repository map"
    assert all(check_evidence(ev, oracle) is None for ev in evidence), [
        check_evidence(e, oracle) for e in evidence
    ]
    assert any(ev.type == "code" and ev.path == "app/auth.py" for ev in evidence)
    forged = evidence[0].model_copy(update={"source_tool": "search_code"})
    assert check_evidence(forged, oracle) is not None  # only repo_overview may produce a map


async def test_repo_overview_is_an_mcp_tool(db_settings, tiny_repo):
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    async with toolbox_factory(db_settings)() as tb:
        assert "repo_overview" in await tb.tools()
        payload = await tb.call("repo_overview", {"repo": "tiny"})
    assert payload["items"][0]["type"] == "map"


def test_overview_endpoint(db_settings, tiny_repo):
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    with TestClient(
        create_app(db_settings, llm=FakeLLM(), toolbox_factory=toolbox_factory(db_settings))
    ) as c:
        r = c.get("/overview", params={"repo": "tiny"})
        assert r.status_code == 200 and "Repository map of tiny" in r.json()["map"]
        assert c.get("/overview", params={"repo": "nope"}).status_code == 404
