"""Convert retrieval hits and MCP tool payloads into `Evidence` items."""

from __future__ import annotations

from typing import Any

from badebhai.app.schemas import Evidence
from badebhai.app.security import looks_injected
from badebhai.retrieval.hybrid import Hit


def evidence_from_hit(hit: Hit, repo) -> Evidence:
    kind = "code" if hit.kind == "code" else "doc"
    title = f"{hit.path}:{hit.start_line}-{hit.end_line}" + (f" ({hit.symbol})" if hit.symbol else "")
    return Evidence(
        key=f"{kind}:{hit.path}#L{hit.start_line}-L{hit.end_line}",
        type=kind,
        source_tool="hybrid_search",
        title=title,
        content=hit.content,
        path=hit.path,
        start_line=hit.start_line,
        end_line=hit.end_line,
        symbol=hit.symbol,
        url=repo.file_url(hit.path, hit.start_line, hit.end_line) if repo else None,
        score=hit.score,
        suspicious=looks_injected(hit.content),
    )


def evidence_from_tool(tool: str, payload: dict[str, Any]) -> list[Evidence]:
    """MCP tool results carry an `items` list; each item has a `type` the agent understands."""
    out: list[Evidence] = []
    for item in payload.get("items", []) or []:
        typ = item.get("type")
        content = item.get("content") or item.get("snippet") or ""
        if typ in ("code", "doc"):
            path, s, e = item.get("path"), item.get("start_line"), item.get("end_line")
            if not (path and s and e):
                continue
            sym = item.get("symbol")
            ev = Evidence(
                key=f"{typ}:{path}#L{s}-L{e}",
                type=typ,
                source_tool=tool,
                title=f"{path}:{s}-{e}" + (f" ({sym})" if sym else ""),
                content=content,
                path=path,
                start_line=s,
                end_line=e,
                symbol=sym,
                url=item.get("url"),
                score=item.get("score"),
            )
        elif typ == "commit":
            sha = item.get("sha")
            if not sha:
                continue
            ev = Evidence(
                key=f"commit:{sha}",
                type="commit",
                source_tool=tool,
                title=f"commit {sha[:10]}: {item.get('subject', '')}"[:200],
                content=content,
                sha=sha,
                url=item.get("url"),
                date=item.get("date"),
                path=item.get("path"),
            )
        elif typ in ("issue", "pr"):
            num = item.get("number")
            if num is None:
                continue
            ev = Evidence(
                key=f"{typ}:{num}",
                type=typ,
                source_tool=tool,
                title=f"{'PR' if typ == 'pr' else 'Issue'} #{num}: {item.get('title', '')}"[:200],
                content=content,
                number=int(num),
                url=item.get("url"),
                date=item.get("date"),
            )
        else:
            continue
        ev.suspicious = looks_injected(ev.content) or looks_injected(ev.title)
        out.append(ev)
    return out
