"""MCP client side: the agent reaches every data source through this toolbox.

`async with McpToolbox.from_settings(s) as tb:` opens one session per server (Streamable HTTP,
stdio subprocess, or an in-process server for tests), discovers tools with `list_tools`, and
exposes `call()` with a per-call timeout, one retry on transport errors, and a per-question
tool-call budget. Results are parsed into dicts; failures come back as `{"error": ...}`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import Client, StdioServerParameters

from repomind.config import Settings
from repomind.log import get_logger

log = get_logger(__name__)

SERVERS = ("code_search", "git_history", "docs")
_MODULES = {
    "code_search": "repomind.mcp_servers.code_search",
    "git_history": "repomind.mcp_servers.git_history",
    "docs": "repomind.mcp_servers.docs",
}


class ToolBudgetExceeded(RuntimeError):
    pass


@dataclass
class ToolSpec:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def params(self) -> dict[str, Any]:
        return self.input_schema.get("properties", {})

    @property
    def required(self) -> list[str]:
        return list(self.input_schema.get("required", []))

    def signature(self) -> str:
        parts = []
        for name, spec in self.params.items():
            if name == "repo":
                continue
            typ = spec.get("type") or "/".join(
                x.get("type", "?") for x in spec.get("anyOf", []) if x.get("type") != "null"
            )
            opt = "" if name in self.required else "?"
            parts.append(f"{name}{opt}: {typ}")
        return f"{self.name}({', '.join(parts)})"


@dataclass
class ToolCallRecord:
    server: str
    tool: str
    args: dict[str, Any]
    ok: bool
    ms: int
    n_items: int = 0
    error: str | None = None
    error_type: str | None = None


@dataclass
class McpToolbox:
    targets: dict[str, Any]
    timeout_s: float = 25.0
    max_calls: int = 24
    calls: list[ToolCallRecord] = field(default_factory=list)
    _clients: dict[str, Client] = field(default_factory=dict)
    _stack: AsyncExitStack | None = None
    _tools: dict[str, ToolSpec] | None = None

    @classmethod
    def from_settings(cls, s: Settings, overrides: dict[str, Any] | None = None) -> McpToolbox:
        if overrides is not None:
            targets = overrides
        elif s.mcp_transport == "http":
            targets = {
                "code_search": s.mcp_code_search_url,
                "git_history": s.mcp_git_history_url,
                "docs": s.mcp_docs_url,
            }
        elif s.mcp_transport == "stdio":
            targets = {
                name: StdioServerParameters(command=sys.executable, args=["-m", mod, "--transport", "stdio"])
                for name, mod in _MODULES.items()
            }
        elif s.mcp_transport == "inmemory":
            from repomind.mcp_servers import code_search, docs, git_history
            from repomind.mcp_servers.common import Deps

            deps = Deps(s)
            targets = {
                "code_search": code_search.build_server(deps),
                "git_history": git_history.build_server(deps),
                "docs": docs.build_server(deps),
            }
        else:
            raise ValueError(f"unknown MCP_TRANSPORT {s.mcp_transport!r}")
        return cls(targets=targets, timeout_s=s.mcp_tool_timeout_s, max_calls=s.max_tool_calls)

    async def __aenter__(self) -> McpToolbox:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for name, target in self.targets.items():
            try:
                client = Client(target, read_timeout_seconds=self.timeout_s)
                self._clients[name] = await self._stack.enter_async_context(client)
            except Exception as e:  # a down server degrades that source, not the whole answer
                log.warning("mcp server unavailable", extra={"server": name, "error": _err(e)})
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            try:
                await self._stack.__aexit__(*exc)
            except Exception as e:
                log.debug("mcp shutdown error", extra={"error": _err(e)})
        self._clients.clear()

    @property
    def available_servers(self) -> list[str]:
        return list(self._clients)

    async def tools(self) -> dict[str, ToolSpec]:
        """Discover tools from every connected server (cached for this toolbox)."""
        if self._tools is None:
            found: dict[str, ToolSpec] = {}
            for server, client in self._clients.items():
                try:
                    result = await asyncio.wait_for(client.list_tools(), self.timeout_s)
                except Exception as e:
                    log.warning("list_tools failed", extra={"server": server, "error": _err(e)})
                    continue
                for t in result.tools:
                    found[t.name] = ToolSpec(server, t.name, t.description or "", t.input_schema or {})
            self._tools = found
        return self._tools

    async def catalog_text(self) -> str:
        tools = await self.tools()
        by_server: dict[str, list[ToolSpec]] = {}
        for spec in tools.values():
            by_server.setdefault(spec.server, []).append(spec)
        lines = []
        for server, specs in by_server.items():
            lines.append(f"[{server}]")
            for spec in specs:
                desc = " ".join(spec.description.split())[:260]
                lines.append(f"- {spec.signature()}: {desc}")
        return "\n".join(lines)

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - len(self.calls))

    async def call(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        tools = await self.tools()
        spec = tools.get(tool)
        if spec is None:
            return {"error": f"unknown tool {tool!r}", "error_type": "unknown_tool"}
        if len(self.calls) >= self.max_calls:
            raise ToolBudgetExceeded(f"tool-call budget of {self.max_calls} exhausted")
        clean = {k: v for k, v in args.items() if k in spec.params and v is not None}
        missing = [k for k in spec.required if k not in clean]
        if missing:
            return {"error": f"missing required arguments {missing}", "error_type": "bad_arguments"}
        record = ToolCallRecord(spec.server, tool, clean, ok=False, ms=0)
        self.calls.append(record)
        t0 = time.perf_counter()
        payload: dict[str, Any]
        for attempt in (1, 2):
            try:
                res = await asyncio.wait_for(
                    self._clients[spec.server].call_tool(tool, clean), self.timeout_s
                )
                payload = _parse(res)
                break
            except TimeoutError:
                payload = {"error": f"{tool} timed out after {self.timeout_s:.0f}s", "error_type": "timeout"}
                break
            except Exception as e:
                payload = {"error": f"{tool} failed: {_err(e)}", "error_type": "transport"}
                if attempt == 2:
                    break
                await asyncio.sleep(0.5)
        record.ms = int((time.perf_counter() - t0) * 1000)
        record.ok = "error" not in payload
        record.n_items = len(payload.get("items", []) or [])
        record.error = payload.get("error")
        record.error_type = payload.get("error_type")
        log.info(
            "tool call",
            extra={
                "tool": tool,
                "ok": record.ok,
                "ms": record.ms,
                "items": record.n_items,
                "error": record.error,
            },
        )
        return payload


def _parse(res) -> dict[str, Any]:
    if res.structured_content and isinstance(res.structured_content, dict):
        data = res.structured_content.get("result", res.structured_content)
        if isinstance(data, dict):
            return data
    text = "".join(getattr(c, "text", "") for c in res.content or [])
    if res.is_error:
        return {"error": text or "tool error", "error_type": "tool_error"}
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        return {"error": "tool returned non-JSON output", "error_type": "bad_output", "raw": text[:500]}
    return data if isinstance(data, dict) else {"result": data}


def _err(e: BaseException) -> str:
    if isinstance(e, BaseExceptionGroup) and e.exceptions:
        return _err(e.exceptions[0])
    return f"{type(e).__name__}: {e}"[:300]
