"""A map of a repository, built from the index (no LLM, milliseconds).

Search finds small pieces of code, so "what are the main parts of this codebase?" is hard for plain
RAG: no single chunk says it. The map answers it structurally: which directories exist, how big they
are, which languages, and their main classes and functions with file:line, plus the project files and
entry points. It comes with the README's opening and the top modules' own docstrings as ordinary,
line-verified evidence, so an overview answer can still cite real lines.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from psycopg.rows import dict_row

from badebhai.ingestion.languages import is_test_path

_PROJECT_FILES = {
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "package.json", "go.mod", "cargo.toml",
    "pom.xml", "build.gradle", "makefile", "dockerfile", "docker-compose.yml", "gemfile", "composer.json",
}  # fmt: skip
_ENTRY_NAMES = {"__main__.py", "main.py", "app.py", "manage.py", "cli.py", "main.go", "main.rs", "index.js", "index.ts", "server.js", "server.ts"}  # fmt: skip
_DOC_START = re.compile(r'^\s*(?:[rubRUB]{0,2}"""|[rubRUB]{0,2}\'\'\'|/\*\*?|//|#(?!!))')


@dataclass
class _Dir:
    files: int = 0
    lines: int = 0
    code: int = 0
    docs: int = 0
    langs: Counter = field(default_factory=Counter)
    classes: list[tuple[str, str, int]] = field(default_factory=list)
    functions: list[tuple[str, str, int]] = field(default_factory=list)


def _dir_of(path: str, depth: int, prefix: str) -> str:
    rel = path[len(prefix) :] if prefix and path.startswith(prefix) else path
    parts = PurePosixPath(rel).parts[:-1]
    if not parts:
        return prefix or "."
    return prefix + "/".join(parts[:depth]) + "/"


def leading_doc(content: str, max_lines: int = 30) -> str | None:
    """The module docstring / header comment at the top of a chunk, as exact lines, or None."""
    lines = content.split("\n")
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith("#!") or "coding" in lines[i][:30]):
        i += 1
    if i >= len(lines) or not _DOC_START.match(lines[i]):
        return None
    first = lines[i].strip()
    end = i
    if first.lstrip("rubRUB").startswith(('"""', "'''")):
        quote = first.lstrip("rubRUB")[:3]
        if first.count(quote) < 2 or first == quote:  # multi-line docstring
            end = next((j for j in range(i + 1, len(lines)) if quote in lines[j]), None)
            if end is None:
                return None
    elif first.startswith("/*"):
        end = next((j for j in range(i, len(lines)) if "*/" in lines[j]), None)
        if end is None:
            return None
    else:  # run of // or # comments
        while (
            end + 1 < len(lines)
            and _DOC_START.match(lines[end + 1])
            and not lines[end + 1].lstrip().startswith(("/*", '"""'))
        ):
            end += 1
    if end - i + 1 > max_lines or end - i + 1 < 1:
        return None
    text = "\n".join(lines[: end + 1])
    return text if len(text.strip()) > 20 else None


def doc_summary(doc: str | None, max_chars: int = 160) -> str | None:
    """First real sentence of a module docstring, skipping title lines such as `requests.utils` / `~~~~`."""
    if not doc:
        return None
    lines = []
    for ln in doc.split("\n"):
        t = ln.strip().strip("\"'").strip("/*#! ").strip()
        if (
            not t
            or set(t) <= set("~=-_*")
            or re.fullmatch(r"[\w.]+", t)
            or t.startswith((":copyright", ":license"))
        ):
            continue
        lines.append(t)
    text = " ".join(lines)
    if len(text.split()) < 3:
        return None
    first = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return first[:max_chars].rstrip() + ("…" if len(first) > max_chars else "")


def build_overview(pool, repo, path_prefix: str = "", depth: int = 2, max_dirs: int = 30) -> dict:
    prefix = path_prefix.strip().lstrip("./")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT path, kind, language, line_count FROM files WHERE repo_id = %s AND path LIKE %s ORDER BY path",
            (repo.id, prefix.replace("%", r"\%") + "%"),
        )
        files = cur.fetchall()
        cur.execute(
            "SELECT path, symbol, symbol_type, start_line, end_line, content FROM chunks WHERE repo_id = %s "
            "AND path LIKE %s AND (symbol_type IN ('class', 'body', 'function', 'module') OR start_line = 1) "
            "ORDER BY path, start_line",
            (repo.id, prefix.replace("%", r"\%") + "%"),
        )
        chunks = cur.fetchall()

    dirs: dict[str, _Dir] = defaultdict(_Dir)
    langs: Counter = Counter()
    per_file_symbols: Counter = Counter()
    file_syms: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"class": [], "function": []})
    line_counts = {f["path"]: f["line_count"] for f in files}
    for f in files:
        d = dirs[_dir_of(f["path"], depth, prefix)]
        d.files += 1
        d.lines += f["line_count"]
        if f["kind"] == "code":
            d.code += 1
            if f["language"]:
                d.langs[f["language"]] += f["line_count"]
                langs[f["language"]] += f["line_count"]
        else:
            d.docs += 1
    heads: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    for c in chunks:
        if c["start_line"] == 1:
            heads[c["path"]] = c
        sym, typ = c["symbol"], c["symbol_type"]
        if typ == "body" and sym and "." not in sym:
            typ = "class"  # a class too big for one chunk: its header chunk
        if not sym or typ not in ("class", "function") or is_test_path(c["path"]):
            continue
        if (c["path"], sym) in seen:  # long functions are split into several windows
            continue
        seen.add((c["path"], sym))
        name = sym.rsplit(".", 1)[-1]
        if name.startswith("_") or ("." in sym and typ == "function"):  # private helpers, methods
            continue
        d = dirs[_dir_of(c["path"], depth, prefix)]
        entry = (sym, c["path"], c["start_line"])
        (d.classes if typ == "class" else d.functions).append(entry)
        per_file_symbols[c["path"]] += 3 if typ == "class" else 1  # classes say more about a module
        file_syms[c["path"]][typ].append(f"{sym} (:{c['start_line']})")

    code_files = sum(1 for f in files if f["kind"] == "code")
    doc_files = len(files) - code_files
    total = sum(langs.values()) or 1
    lang_line = ", ".join(f"{lang} {round(100 * n / total)}%" for lang, n in langs.most_common(4))
    where = f" under {prefix}" if prefix else ""
    out = [
        f"Repository map of {repo.name}{where} at {(repo.last_indexed_commit or 'unknown')[:10]}, generated by "
        f"BadeBhai from its index: {code_files} code files, {doc_files} documentation files"
        + (f"; languages: {lang_line}." if lang_line else "."),
        "",
        "Directories (files, lines, main classes and functions with file:line):",
    ]
    ranked = sorted(dirs.items(), key=lambda kv: (is_test_path(kv[0] + "x"), -kv[1].lines))
    for name, d in ranked[:max_dirs]:
        kind = "tests" if is_test_path(name + "x") else ("docs" if d.code == 0 and d.docs else "")
        lang = "/".join(lang for lang, _ in d.langs.most_common(2))
        bits = (
            [f"{d.files} files", f"{d.lines:,} lines"] + ([lang] if lang else []) + ([kind] if kind else [])
        )
        line = f"- {name}: " + ", ".join(bits)

        def fmt(items, n):
            return ", ".join(f"{s} ({PurePosixPath(p).name}:{ln})" for s, p, ln in items[:n])

        if d.classes:
            line += f". Classes: {fmt(d.classes, 8)}" + (
                f" and {len(d.classes) - 8} more" if len(d.classes) > 8 else ""
            )
        if d.functions:
            line += f". Functions: {fmt(d.functions, 6)}" + (
                f" and {len(d.functions) - 6} more" if len(d.functions) > 6 else ""
            )
        out.append(line)
    if len(ranked) > max_dirs:
        out.append(f"- … {len(ranked) - max_dirs} more directories")
    main_files = [p for p, _ in per_file_symbols.most_common(12)]
    if main_files:
        out += ["", "Main files (by public classes and functions):"]
        for path in main_files:
            summary = doc_summary(leading_doc(heads[path]["content"])) if path in heads else None
            syms = file_syms[path]
            line = f"- {path} ({line_counts.get(path, 0):,} lines)" + (
                f": {summary.rstrip('.')}" if summary else ""
            )
            if syms["class"]:
                line += ". Classes: " + ", ".join(syms["class"][:6])
            if syms["function"]:
                line += (
                    ". Functions: "
                    + ", ".join(syms["function"][:5])
                    + (" …" if len(syms["function"]) > 5 else "")
                )
            out.append(line)
    project = [
        f["path"]
        for f in files
        if PurePosixPath(f["path"]).name.lower() in _PROJECT_FILES and f["path"].count("/") <= 1
    ]
    entries = [
        f["path"]
        for f in files
        if PurePosixPath(f["path"]).name.lower() in _ENTRY_NAMES and not is_test_path(f["path"])
    ]
    if project:
        out += ["", "Project files: " + ", ".join(project[:12])]
    if entries:
        out += ["Likely entry points: " + ", ".join(entries[:12])]

    items: list[dict] = [
        {
            "type": "map",
            "title": f"repository map of {repo.name}{where}",
            "content": "\n".join(out),
            "repo": repo.name,
            "prefix": prefix,
        }
    ]
    readme = next(
        (
            f["path"]
            for f in files
            if PurePosixPath(f["path"]).name.lower().startswith("readme")
            and f["path"].count("/") == prefix.count("/")
        ),
        None,
    )
    if readme and readme in heads:
        h = heads[readme]
        items.append(
            {
                "type": "doc",
                "path": readme,
                "start_line": h["start_line"],
                "end_line": h["end_line"],
                "content": h["content"],
                "url": repo.file_url(readme, h["start_line"], h["end_line"]),
            }
        )
    for path in (p for p in main_files if p in heads):
        doc = leading_doc(heads[path]["content"])
        if doc:
            n = len(doc.split("\n"))
            items.append(
                {
                    "type": "code",
                    "path": path,
                    "start_line": 1,
                    "end_line": n,
                    "content": doc,
                    "symbol": "module docstring",
                    "url": repo.file_url(path, 1, n),
                }
            )
        if len(items) >= 7:
            break
    return {"repo": repo.name, "commit": repo.last_indexed_commit, "items": items}
