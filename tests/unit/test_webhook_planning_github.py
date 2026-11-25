import json

import httpx
import pytest

from badebhai.app.agent.planning import PlannedCall, default_calls, validate_calls
from badebhai.app.mcp_client import ToolSpec, _parse
from badebhai.ingestion.webhook import parse_push, sign, verify_signature
from badebhai.mcp_servers.github_client import GitHubClient, GitHubError, GitHubRateLimited


# ---------------------------------------------------------------- webhook
def test_signature_verification():
    body = b'{"ref":"refs/heads/main"}'
    good = sign("s3cret", body)
    assert verify_signature("s3cret", body, good)
    assert not verify_signature("s3cret", body + b" ", good)
    assert not verify_signature("other", body, good)
    assert not verify_signature("s3cret", body, None)
    assert not verify_signature("s3cret", body, "sha1=abc")
    assert not verify_signature("", body, good)


def test_parse_push_collects_touched_files():
    ev = parse_push(
        {
            "ref": "refs/heads/main",
            "before": "a" * 40,
            "after": "b" * 40,
            "repository": {"full_name": "acme/tiny"},
            "commits": [
                {"added": ["x.py"], "modified": ["y.py"], "removed": []},
                {"added": [], "modified": ["x.py"], "removed": ["z.py"]},
            ],
        }
    )
    assert ev.full_name == "acme/tiny" and ev.branch == "main"
    assert ev.touched == {"x.py", "y.py", "z.py"} and ev.removed == {"z.py"}
    assert parse_push({"ref": "refs/tags/v1"}) is None


# ---------------------------------------------------------------- planning
TOOLS = {
    "search_code": ToolSpec(
        "code_search",
        "search_code",
        "",
        {
            "properties": {"repo": {"type": "string"}, "query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["repo", "query"],
        },
    ),
    "blame": ToolSpec(
        "git_history",
        "blame",
        "",
        {
            "properties": {
                "repo": {},
                "path": {"type": "string"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
            },
            "required": ["repo", "path", "start_line", "end_line"],
        },
    ),
}


def test_validate_calls_filters_and_normalizes():
    calls = [
        PlannedCall(
            tool="search_code", args_json='{"query": "auth", "k": 4, "repo": "evil/repo", "junk": 1}'
        ),
        PlannedCall(tool="search_code", args_json='{"query": "auth", "k": 4}'),  # duplicate after cleanup
        PlannedCall(tool="delete_everything", args_json="{}"),  # unknown tool
        PlannedCall(tool="search_code", args_json="not json"),
        PlannedCall(tool="blame", args_json='{"path": "a.py"}'),  # missing required lines
        PlannedCall(tool="blame", args_json='{"path": "a.py", "start_line": 1, "end_line": 9}'),
    ]
    out = validate_calls(calls, TOOLS, max_calls=5)
    assert out == [
        {"tool": "search_code", "args": {"query": "auth", "k": 4}},
        {"tool": "blame", "args": {"path": "a.py", "start_line": 1, "end_line": 9}},
    ]
    assert len(validate_calls(calls, TOOLS, max_calls=1)) == 1


def test_default_calls_map_sources_to_tools():
    assert default_calls(["code", "issues"], "q", TOOLS) == [{"tool": "search_code", "args": {"query": "q"}}]


# ---------------------------------------------------------------- MCP result parsing
class _Res:
    def __init__(self, text, is_error=False):
        self.structured_content = None
        self.content = [type("T", (), {"text": text})()]
        self.is_error = is_error


def test_parse_tool_results():
    assert _parse(_Res('{"items": [1]}')) == {"items": [1]}
    assert _parse(_Res("boom", is_error=True))["error"] == "boom"
    assert _parse(_Res("not json"))["error_type"] == "bad_output"


# ---------------------------------------------------------------- GitHub client
def _client(handler, **kw):
    return GitHubClient(token=kw.pop("token", None), transport=httpx.MockTransport(handler), **kw)


def test_rate_limit_raises_clear_error():
    def handler(request):
        return httpx.Response(
            403,
            headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1893456000"},
            json={"message": "API rate limit exceeded"},
        )

    with pytest.raises(GitHubRateLimited) as ei:
        _client(handler).get("/repos/a/b/issues/1")
    msg = str(ei.value)
    assert "rate limit" in msg and "GITHUB_TOKEN" in msg and "UTC" in msg
    with pytest.raises(GitHubRateLimited) as ei2:
        _client(handler, token="t").get("/x")
    assert "GITHUB_TOKEN" not in str(ei2.value)


def test_server_errors_are_retried_then_cached(monkeypatch):
    monkeypatch.setattr("badebhai.mcp_servers.github_client.time.sleep", lambda s: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(502)
        return httpx.Response(200, json={"ok": True})

    c = _client(handler)
    assert c.get("/thing", {"a": 1}) == {"ok": True}
    assert calls["n"] == 3
    assert c.get("/thing", {"a": 1}) == {"ok": True}  # served from TTL cache
    assert calls["n"] == 3


def test_not_found_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(GitHubError) as ei:
        _client(handler).get("/nope")
    assert ei.value.status == 404 and calls["n"] == 1


def test_auth_header_and_user_agent_are_generic():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={})

    _client(handler, token="tok").get("/x")
    assert seen["authorization"] == "Bearer tok"
    assert seen["user-agent"] == "badebhai-mcp"
    assert json.loads("{}") == {}
