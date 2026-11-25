"""Index-backed tool implementations shared by the code-search and docs servers."""

from __future__ import annotations

import re
from typing import Any

from badebhai.db.repos import RepoRecord
from badebhai.ingestion.secrets import redact
from badebhai.mcp_servers.common import Deps, ToolError, clamp
from badebhai.retrieval.hybrid import Hit, hybrid_search

MAX_ITEM_CHARS = 3000
MAX_READ_LINES = 200


def hit_item(hit: Hit, repo: RepoRecord) -> dict[str, Any]:
    content = hit.content
    if len(content) > MAX_ITEM_CHARS:
        content = content[:MAX_ITEM_CHARS].rsplit("\n", 1)[0] + "\n…"
    return {
        "type": "code" if hit.kind == "code" else "doc",
        "path": hit.path,
        "start_line": hit.start_line,
        "end_line": hit.end_line,
        "symbol": hit.symbol,
        "symbol_type": hit.symbol_type,
        "language": hit.language,
        "score": hit.score,
        "content": content,
        "url": repo.file_url(hit.path, hit.start_line, hit.end_line),
    }


def search(
    deps: Deps,
    repo: str,
    query: str,
    kind: str,
    k: int | None,
    path_prefix: str | None = None,
    language: str | None = None,
) -> dict:
    if not query or not query.strip():
        raise ToolError("query must not be empty", "bad_arguments")
    r = deps.repo(repo)
    hits = hybrid_search(
        deps.pool,
        deps.embedder,
        r.id,
        query,
        k=clamp(k, 1, 15, 6),
        kinds=(kind,),
        path_prefix=path_prefix or None,
        language=language or None,
        settings=deps.settings,
    )
    return {"repo": r.name, "commit": r.last_indexed_commit, "items": [hit_item(h, r) for h in hits]}


def _file_row(deps: Deps, r: RepoRecord, path: str) -> tuple[str, int] | None:
    with deps.pool.connection() as conn:
        row = conn.execute(
            "SELECT kind, line_count FROM files WHERE repo_id = %s AND path = %s", (r.id, path)
        ).fetchone()
    return (row[0], row[1]) if row else None


def suggest_paths(deps: Deps, r: RepoRecord, path: str, limit: int = 5) -> list[str]:
    base = path.rstrip("/").split("/")[-1]
    with deps.pool.connection() as conn:
        rows = conn.execute(
            "SELECT path FROM files WHERE repo_id = %s AND path ILIKE %s ORDER BY length(path) LIMIT %s",
            (r.id, "%" + base.replace("%", r"\%") + "%", limit),
        ).fetchall()
    return [x[0] for x in rows]


def read_range(deps: Deps, repo: str, path: str, start_line: int | None, end_line: int | None) -> dict:
    r = deps.repo(repo)
    path = path.strip().lstrip("./")
    row = _file_row(deps, r, path)
    if row is None:
        raise ToolError(
            f"{path!r} is not an indexed file", "not_found", did_you_mean=suggest_paths(deps, r, path)
        )
    kind, n_lines = row
    start = clamp(start_line, 1, n_lines, 1)
    end = clamp(end_line, start, n_lines, min(n_lines, start + MAX_READ_LINES - 1))
    end = min(end, start + MAX_READ_LINES - 1)
    try:
        text = r.resolve(path).read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError) as e:
        raise ToolError(f"cannot read {path}: {e}", "not_found") from e
    lines = redact(text)[0].split("\n")
    content = "\n".join(ln.rstrip("\r") for ln in lines[start - 1 : end])
    return {
        "repo": r.name,
        "commit": r.last_indexed_commit,
        "file_lines": n_lines,
        "items": [
            {
                "type": "code" if kind == "code" else "doc",
                "path": path,
                "start_line": start,
                "end_line": end,
                "content": content,
                "url": r.file_url(path, start, end),
            }
        ],
    }


def list_files(deps: Deps, repo: str, path_prefix: str | None, kind: str | None, limit: int | None) -> dict:
    r = deps.repo(repo)
    lim = clamp(limit, 1, 500, 200)
    sql = "SELECT path, kind, language, line_count FROM files WHERE repo_id = %s"
    params: list[Any] = [r.id]
    if path_prefix:
        sql += " AND path LIKE %s"
        params.append(path_prefix.replace("%", r"\%") + "%")
    if kind:
        sql += " AND kind = %s"
        params.append(kind)
    sql += " ORDER BY path LIMIT %s"
    params.append(lim + 1)
    with deps.pool.connection() as conn:
        rows = conn.execute(sql, params).fetchall()
    files = [{"path": p, "kind": k, "language": lang, "lines": n} for p, k, lang, n in rows[:lim]]
    return {"repo": r.name, "files": files, "truncated": len(rows) > lim}


def outline(deps: Deps, repo: str, path: str) -> dict:
    r = deps.repo(repo)
    path = path.strip().lstrip("./")
    with deps.pool.connection() as conn:
        rows = conn.execute(
            "SELECT symbol, symbol_type, start_line, end_line FROM chunks "
            "WHERE repo_id = %s AND path = %s ORDER BY start_line",
            (r.id, path),
        ).fetchall()
    if not rows:
        raise ToolError(
            f"{path!r} is not an indexed file", "not_found", did_you_mean=suggest_paths(deps, r, path)
        )
    symbols = [
        {"symbol": s, "type": t, "start_line": a, "end_line": b}
        for s, t, a, b in rows
        if s and t not in ("module", "body")
    ]
    return {"repo": r.name, "path": path, "symbols": symbols}


_CHANGELOG = r"(^|/)(history|changelog|changes|news|releases?|release[-_]notes)[^/]*$"


def release_notes(deps: Deps, repo: str, version: str) -> dict:
    """The changelog section(s) whose heading names `version` (exact lines, ready to cite)."""
    r = deps.repo(repo)
    v = version.strip().lstrip("vV")
    if not re.fullmatch(r"\d+(\.\d+){1,3}", v):
        raise ToolError(f"{version!r} is not a version number like 2.32.0", "bad_arguments")
    pattern = r"(^|[^0-9.])v?" + re.escape(v) + r"([^0-9.]|$)"
    with deps.pool.connection() as conn:
        rows = conn.execute(
            "SELECT path, start_line, end_line, content, symbol FROM chunks WHERE repo_id = %s AND kind = 'doc' "
            "AND symbol ~ %s ORDER BY (path ~* %s) DESC, length(path), start_line LIMIT 4",
            (r.id, pattern, _CHANGELOG),
        ).fetchall()
    items = [
        {
            "type": "doc",
            "path": path,
            "start_line": s,
            "end_line": e,
            "symbol": sym,
            "content": content,
            "url": r.file_url(path, s, e),
        }
        for path, s, e, content, sym in rows
    ]
    note = (
        None if items else f"no changelog section mentions {v}; try search_docs or changes_between_versions"
    )
    return {"repo": r.name, "version": v, "items": items, **({"note": note} if note else {})}
