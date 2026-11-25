"""History questions: release sections in changelogs, version search, release commit ranges, time windows."""

import pytest

from badebhai.app.agent.planning import parse_window, release_calls
from badebhai.ingestion.chunker import Chunker
from badebhai.ingestion.indexer import Indexer
from badebhai.retrieval.hybrid import build_tsquery, keyword_terms
from tests.integration.helpers import toolbox_factory

HISTORY = """Release History
===============

2.1.0 (2024-05-20)
------------------

**Security**
- Fixed `verify=False` leaking to later requests. (#101)

2.0.0 (2024-01-02)
------------------

- Dropped Python 2.

A paragraph line
- a list item, not a heading
---
"""


def test_setext_headings_split_changelogs_by_release():
    chunks = Chunker().chunk("HISTORY.md", HISTORY, "doc", "markdown")
    titles = [c.symbol for c in chunks]
    assert (
        "Release History > 2.1.0 (2024-05-20)" in titles and "Release History > 2.0.0 (2024-01-02)" in titles
    )
    sec = next(c for c in chunks if c.symbol.endswith("2.1.0 (2024-05-20)"))
    assert "verify=False" in sec.content and "Dropped Python 2" not in sec.content
    assert not any("list item" in (t or "") for t in titles)  # "- item\n---" is not a heading


def test_versions_are_single_search_terms():
    assert "2.32.0" in keyword_terms("What was fixed in v2.32.0?")
    assert "2.32.0" in build_tsquery("fixed in 2.32.0").split(" | ")


def test_time_windows_and_release_calls():
    assert parse_window("What changed during May 2024?") == ("2024-05-01", "2024-06-01")
    assert parse_window("changes in December 2023") == ("2023-12-01", "2024-01-01")
    assert parse_window("between 2024-05-01 and 2024-05-31") == ("2024-05-01", "2024-05-31")
    assert parse_window("since March 2024") == ("2024-03-01", None)
    assert parse_window("what happened in 2022") == ("2022-01-01", "2023-01-01")
    assert parse_window("what changed last week") is None
    tools = {"release_notes", "changes_between_versions", "recent_changes"}
    assert release_calls("Which code changed for the fix in 2.1.0?", tools) == [
        {"tool": "release_notes", "args": {"version": "2.1.0"}},
        {"tool": "changes_between_versions", "args": {"version": "2.1.0"}},
    ]
    assert release_calls("What changed in src/app during May 2024?", tools) == [
        {
            "tool": "recent_changes",
            "args": {"since": "2024-05-01", "limit": 30, "until": "2024-06-01", "path_prefix": "src/app"},
        }
    ]


@pytest.mark.db
async def test_release_notes_and_commits_between_versions(db_settings, tiny_repo, git):
    (tiny_repo / "HISTORY.md").write_text(HISTORY)
    git(tiny_repo, "add", "-A")
    git(tiny_repo, "commit", "-q", "-m", "release notes")
    git(tiny_repo, "tag", "v2.0.0")
    (tiny_repo / "app" / "verify.py").write_text("def reset_verify():\n    return True\n")
    git(tiny_repo, "add", "-A")
    git(tiny_repo, "commit", "-q", "-m", "Stop verify=False leaking to later requests (#101)")
    git(tiny_repo, "tag", "v2.1.0-rc1")
    git(tiny_repo, "tag", "v2.1.0")
    Indexer(db_settings).ingest(str(tiny_repo), name="tiny")
    async with toolbox_factory(db_settings)() as tb:
        notes = await tb.call("release_notes", {"repo": "tiny", "version": "v2.1.0"})
        assert notes["items"][0]["path"] == "HISTORY.md" and "verify=False" in notes["items"][0]["content"]
        missing = await tb.call("release_notes", {"repo": "tiny", "version": "9.9"})
        assert missing["items"] == [] and "note" in missing
        rng = await tb.call("changes_between_versions", {"repo": "tiny", "version": "2.1.0"})
        assert rng["range"] == {"from": "v2.0.0", "to": "v2.1.0", "path": None}
        assert [c["subject"] for c in rng["items"]] == ["Stop verify=False leaking to later requests (#101)"]
        bad = await tb.call("changes_between_versions", {"repo": "tiny", "version": "7.0.0"})
        assert bad["error_type"] == "not_found"
