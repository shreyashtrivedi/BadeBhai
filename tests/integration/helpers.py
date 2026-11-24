"""Shared fixtures-as-functions for MCP / agent integration tests."""

from __future__ import annotations

import json
import re

import httpx

from repomind.app.mcp_client import McpToolbox
from repomind.mcp_servers import code_search, docs, git_history
from repomind.mcp_servers.common import Deps
from repomind.mcp_servers.github_client import GitHubClient

ISSUE = {
    "number": 7,
    "title": "BasicAuth breaks with unicode passwords",
    "state": "open",
    "html_url": "https://github.com/acme/tiny/issues/7",
    "created_at": "2024-03-01T10:00:00Z",
    "closed_at": None,
    "user": {"login": "alice"},
    "labels": [{"name": "bug"}],
    "body": "When the password contains unicode, the Authorization header is wrong.",
    "comments": 0,
}
PR = {
    **ISSUE,
    "number": 9,
    "title": "Add token verification",
    "pull_request": {"url": "x"},
    "state": "closed",
}


def fake_github(rate_limited: bool = False) -> GitHubClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if rate_limited:
            return httpx.Response(
                403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1893456000"}
            )
        path = request.url.path
        if path == "/search/issues":
            return httpx.Response(200, json={"total_count": 2, "items": [ISSUE, PR]})
        if path.endswith("/issues/7"):
            return httpx.Response(200, json=ISSUE)
        if "/commits/" in path and path.endswith("/pulls"):
            return httpx.Response(200, json=[{k: v for k, v in PR.items() if k != "pull_request"}])
        return httpx.Response(404, json={"message": "Not Found"})

    return GitHubClient(token=None, transport=httpx.MockTransport(handler))


def toolbox_factory(settings, github: GitHubClient | None = None):
    deps = Deps(settings)
    servers = {
        "code_search": code_search.build_server(deps),
        "git_history": git_history.build_server(deps, github or fake_github()),
        "docs": docs.build_server(deps),
    }
    return lambda: McpToolbox(targets=servers, timeout_s=20, max_calls=settings.max_tool_calls)


def evidence_ids(user_message: str) -> dict[str, list[str]]:
    """Map evidence type -> ids, parsed from a rendered evidence block list."""
    out: dict[str, list[str]] = {}
    for m in re.finditer(r'<ev-[0-9a-f]{8} id="(\d+)" type="(\w+)"', user_message):
        out.setdefault(m.group(2), []).append(m.group(1))
    return out


def plan(*subs: tuple[str, list[str], str, list[tuple[str, dict]]]) -> dict:
    return {
        "sub_questions": [
            {
                "text": text,
                "sources": sources,
                "query": query,
                "calls": [{"tool": t, "args_json": json.dumps(a)} for t, a in calls],
            }
            for text, sources, query, calls in subs
        ]
    }


def grade_all(score: float = 0.9, sufficient: bool = True):
    """Scripted grader: same relevance for every item, same verdict for every sub-question."""

    def handler(system, messages, schema):
        user = messages[-1]["content"]
        item_ids = [int(x) for x in re.findall(r'<ev-[0-9a-f]{8} id="(\d+)"', user)]
        sq_ids = re.findall(r"## Sub-question (q\d+):", user)
        return {
            "grades": [{"id": i, "relevance": score} for i in item_ids],
            "verdicts": [
                {
                    "sub_question_id": q,
                    "sufficient": sufficient,
                    "missing": "" if sufficient else "nothing relevant",
                }
                for q in sq_ids
            ],
        }

    return handler


STAGE3_NODES = [
    "load_memory",
    "decompose",
    "retrieve",
    "grade_evidence",
    "synthesize",
    "verify_citations",
    "write_memory",
]
