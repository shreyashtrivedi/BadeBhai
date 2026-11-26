"""'Recently' is relative to the indexed snapshot, not to today (demo repos are pinned to old releases)."""

from badebhai.app.agent.planning import release_calls
from badebhai.mcp_servers.git_history import _from_head

TOOLS = {"recent_changes": 1, "release_notes": 1}


def test_recently_asks_for_the_snapshots_recent_commits():
    assert release_calls("What changed recently, and why?", TOOLS) == [
        {"tool": "recent_changes", "args": {"since": "90 days ago", "limit": 30}}
    ]
    assert release_calls("latest changes to src/requests/", TOOLS)[0]["args"]["path_prefix"] == "src/requests"


def test_no_recency_call_for_versions_or_plain_questions():
    assert release_calls("How does retry work?", TOOLS) == []
    assert release_calls("Is this the latest version?", TOOLS) == []
    assert [c["tool"] for c in release_calls("What changed in 2.32.0?", TOOLS)] == ["release_notes"]


def test_relative_dates_count_back_from_the_newest_commit():
    head = "2024-05-29T08:36:10-07:00"
    assert _from_head("30 days ago", head) == "2024-04-29"
    assert _from_head("2 weeks ago", head) == "2024-05-15"
    assert _from_head("2024-01-01", head) == "2024-01-01"  # absolute dates untouched
    assert _from_head("30 days ago", None) == "30 days ago"  # no snapshot date: leave it to git
