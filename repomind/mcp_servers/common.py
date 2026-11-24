"""Shared plumbing for the RepoMind MCP servers.

Each server is a separate process exposing read-only tools. Tool functions are plain sync
functions: the MCP SDK runs them in worker threads, so blocking DB / git / embedding work is fine.
Errors are returned as `{"error": ..., "error_type": ...}` payloads (raised exceptions would reach
the client as an opaque "Error executing tool").
"""

from __future__ import annotations

import argparse
import functools
import threading
from collections.abc import Callable
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from repomind.config import Settings, get_settings
from repomind.db import repos as repo_db
from repomind.log import get_logger, setup_logging

log = get_logger(__name__)


class ToolError(Exception):
    def __init__(self, message: str, error_type: str = "error", **extra: Any):
        super().__init__(message)
        self.error_type = error_type
        self.extra = extra


class Deps:
    """Lazily-created shared resources (settings, DB pool, embedder) for one server process."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._pool = None
        self._embedder = None
        self._lock = threading.Lock()

    @property
    def pool(self):
        with self._lock:
            if self._pool is None:
                from repomind.db.pool import get_pool

                self._pool = get_pool(self.settings)
            return self._pool

    @property
    def embedder(self):
        with self._lock:
            if self._embedder is None:
                from repomind.embeddings import get_embedder

                self._embedder = get_embedder(self.settings)
            return self._embedder

    def repo(self, name: str) -> repo_db.RepoRecord:
        rec = repo_db.get_repo(self.pool, name)
        if rec is None:
            known = [r.name for r in repo_db.list_repos(self.pool)]
            raise ToolError(f"repo {name!r} is not indexed (known: {known})", "unknown_repo")
        return rec


def tool_errors(fn: Callable[..., dict]) -> Callable[..., dict]:
    """Turn exceptions into structured error payloads the agent can reason about."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError as e:
            return {"error": str(e), "error_type": e.error_type, **e.extra}
        except Exception as e:
            log.exception("tool failed", extra={"tool": fn.__name__})
            return {"error": f"{type(e).__name__}: {e}", "error_type": "internal"}

    return wrapper


def clamp(value: int | None, lo: int, hi: int, default: int) -> int:
    if value is None:
        return default
    return max(lo, min(hi, int(value)))


def run(server: MCPServer, default_port: int) -> None:
    parser = argparse.ArgumentParser(description=f"RepoMind MCP server: {server.name}")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="streamable-http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args()
    s = get_settings()
    setup_logging(s.log_level, s.log_json)
    if args.transport == "stdio":
        server.run("stdio")
        return
    log.info("mcp server starting", extra={"server": server.name, "host": args.host, "port": args.port})
    server.run(
        "streamable-http",
        host=args.host,
        port=args.port,
        stateless_http=True,
        json_response=True,
        # Read-only servers on a private network / localhost; host-header checks would reject
        # container DNS names such as `mcp-code:8101`.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
