"""The three MCP servers, called through the MCP client toolbox (in-process transport)."""

import pytest

from badebhai.db.pool import get_pool
from badebhai.ingestion.indexer import Indexer
from tests.integration.helpers import fake_github, toolbox_factory

pytestmark = pytest.mark.db


@pytest.fixture
def indexed(db_settings, tiny_repo, git):
    auth = tiny_repo / "app" / "auth.py"
    auth.write_text(auth.read_text() + "\n\ndef rotate_token(old):\n    return 'tok_' + old[::-1]\n")
    git(tiny_repo, "commit", "-qam", "Add token rotation to the auth module")
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    with get_pool(db_settings).connection() as conn:
        conn.execute("UPDATE repos SET github_full_name = 'acme/tiny' WHERE name = 'tiny'")
    return db_settings


async def test_tool_discovery_and_catalog(indexed):
    async with toolbox_factory(indexed)() as tb:
        tools = await tb.tools()
        assert {"search_code", "read_code", "search_docs", "blame", "search_issues", "prs_for_commit"} <= set(
            tools
        )
        assert tools["blame"].server == "git_history"
        catalog = await tb.catalog_text()
        assert "[code_search]" in catalog and "search_code(query: string" in catalog
        assert "repo" not in tools["search_code"].signature()


async def test_code_and_docs_tools(indexed):
    async with toolbox_factory(indexed)() as tb:
        res = await tb.call("search_code", {"repo": "tiny", "query": "basic auth header", "k": 3})
        assert res["items"][0]["path"] == "app/auth.py" and res["items"][0]["type"] == "code"
        read = await tb.call(
            "read_code", {"repo": "tiny", "path": "app/auth.py", "start_line": 1, "end_line": 3}
        )
        assert read["items"][0]["content"].startswith('"""Authentication helpers."""')
        assert read["items"][0]["end_line"] == 3
        outline = await tb.call("file_outline", {"repo": "tiny", "path": "app/auth.py"})
        assert {"BasicAuth", "verify_token", "rotate_token"} <= {s["symbol"] for s in outline["symbols"]}
        missing = await tb.call("read_code", {"repo": "tiny", "path": "auth.py"})
        assert missing["error_type"] == "not_found" and "app/auth.py" in missing["did_you_mean"]
        doc = await tb.call("search_docs", {"repo": "tiny", "query": "retries"})
        assert doc["items"][0]["path"] == "README.md" and doc["items"][0]["type"] == "doc"
        unknown = await tb.call("search_code", {"repo": "nope", "query": "x"})
        assert unknown["error_type"] == "unknown_repo"


async def test_git_history_tools(indexed):
    async with toolbox_factory(indexed)() as tb:
        commits = await tb.call("search_commits", {"repo": "tiny", "query": "token rotation"})
        top = commits["items"][0]
        assert top["type"] == "commit" and top["subject"] == "Add token rotation to the auth module"
        assert top["url"] == f"https://github.com/acme/tiny/commit/{top['sha']}"
        detail = await tb.call("get_commit", {"repo": "tiny", "sha": top["sha"][:10]})
        assert "rotate_token" in detail["items"][0]["content"]
        blame = await tb.call(
            "blame", {"repo": "tiny", "path": "app/auth.py", "start_line": 1, "end_line": 30}
        )
        subjects = {i["subject"] for i in blame["items"]}
        assert subjects == {"initial commit", "Add token rotation to the auth module"}
        history = await tb.call("file_history", {"repo": "tiny", "path": "app/auth.py"})
        assert len(history["items"]) == 2
        recent = await tb.call(
            "recent_changes", {"repo": "tiny", "path_prefix": "app/", "since": "1 day ago"}
        )
        assert recent["summary"]["commits"] == 2
        bad = await tb.call("get_commit", {"repo": "tiny", "sha": "--output=x"})
        assert bad["error_type"] == "bad_arguments"


async def test_issue_tools_with_mocked_github(indexed):
    async with toolbox_factory(indexed)() as tb:
        res = await tb.call("search_issues", {"repo": "tiny", "query": "basic auth unicode", "state": "open"})
        assert [(i["type"], i["number"]) for i in res["items"]] == [("issue", 7), ("pr", 9)]
        one = await tb.call("get_issue", {"repo": "tiny", "number": 7})
        assert "unicode" in one["items"][0]["content"]
        prs = await tb.call("prs_for_commit", {"repo": "tiny", "sha": "abcdef1234"})
        assert prs["items"][0]["type"] == "pr"


async def test_rate_limit_is_reported_not_raised(indexed):
    async with toolbox_factory(indexed, fake_github(rate_limited=True))() as tb:
        res = await tb.call("search_issues", {"repo": "tiny", "query": "auth"})
    assert res["error_type"] == "rate_limited"
    assert "GITHUB_TOKEN" in res["error"]


async def test_tool_budget_and_argument_validation(indexed):
    indexed.max_tool_calls = 2
    async with toolbox_factory(indexed)() as tb:
        missing = await tb.call("blame", {"repo": "tiny", "path": "app/auth.py"})
        assert missing["error_type"] == "bad_arguments"  # rejected client-side, does not count
        await tb.call("search_code", {"repo": "tiny", "query": "a"})
        await tb.call("search_code", {"repo": "tiny", "query": "b"})
        from badebhai.app.mcp_client import ToolBudgetExceeded

        with pytest.raises(ToolBudgetExceeded):
            await tb.call("search_code", {"repo": "tiny", "query": "c"})
        assert [c.tool for c in tb.calls] == ["search_code", "search_code"]
