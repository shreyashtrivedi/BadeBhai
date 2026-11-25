"""Chunking: tree-sitter function/class chunks, heading-based doc chunks, sliding-window fallback.

Every chunk's `content` is exactly lines `start_line..end_line` (1-based, inclusive) of the
file, so a citation's line range can always be checked against the source.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from badebhai.ingestion.languages import TS_SPECS, TSSpec, get_parser

# Bump when chunk boundaries change: indexes built with another version are re-chunked on next sync.
CHUNKER_VERSION = 3  # 3: Markdown setext headings (changelog release sections)


@dataclass
class Chunk:
    path: str
    kind: str
    language: str | None
    start_line: int
    end_line: int
    content: str
    symbol: str | None = None
    symbol_type: str | None = None

    def embedding_text(self) -> str:
        head = f"File: {self.path}"
        if self.symbol:
            head += f"\n{self.symbol_type or 'section'}: {self.symbol}"
        return f"{head}\n\n{self.content}"

    def search_text(self) -> str:
        path_words = re.sub(r"[/._\-]+", " ", self.path)
        return f"{self.symbol or ''} {path_words}\n{self.content}\n{expand_identifiers(self.content)}"


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_identifier(ident: str) -> list[str]:
    return [p.lower() for piece in ident.split("_") for p in _CAMEL.findall(piece) if len(p) > 1]


def expand_identifiers(text: str) -> str:
    """'HTTPBasicAuth max_retries' -> 'auth basic http max retries' so keyword search matches parts."""
    extra: set[str] = set()
    for ident in set(_IDENT.findall(text)):
        if "_" in ident.strip("_") or any(c.isupper() for c in ident[1:]):
            parts = split_identifier(ident)
            if len(parts) > 1:
                extra.update(parts)
    return " ".join(sorted(extra))


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT = re.compile(r"^(=+|-+)\s*$")
_NOT_TITLE = re.compile(r"^\s*([-*+>|]|\d+[.)]\s|```|~~~)")  # list items, quotes, tables, fences
_RST_ADORN = re.compile(r"""^([=\-~^"'`#*+:.])\1{2,}\s*$""")


class Chunker:
    def __init__(self, max_lines: int = 80, window: int = 50, overlap: int = 10):
        if overlap >= window:
            raise ValueError("overlap must be smaller than window")
        self.max_lines = max_lines
        self.window = window
        self.overlap = overlap

    # ---------- public ----------
    def chunk(self, path: str, text: str, kind: str, language: str | None) -> list[Chunk]:
        lines = [ln.rstrip("\r") for ln in text.split("\n")]
        if text.endswith("\n"):
            lines = lines[:-1]
        if not any(ln.strip() for ln in lines):
            return []
        ctx = _Ctx(path, kind, language, lines)
        ranges: list[tuple[int, int, str | None, str | None]] | None = None
        if kind == "code" and language in TS_SPECS:
            ranges = self._code_ranges(text, language, ctx)
        elif language == "markdown":
            ranges = self._sections(self._markdown_sections(lines), ctx)
        elif language == "rst":
            ranges = self._sections(self._rst_sections(lines), ctx)
        if not ranges:
            ranges = [(s, e, None, None) for s, e in self._windows(1, len(lines))]
        chunks = []
        for s, e, sym, sym_type in ranges:
            if ctx.blank(s, e):
                continue
            chunks.append(Chunk(path, kind, language, s, e, ctx.slice(s, e), sym, sym_type))
        return chunks

    # ---------- code ----------
    def _code_ranges(self, text: str, language: str, ctx: _Ctx):
        parser = get_parser(language)
        if parser is None:
            return None
        spec = TS_SPECS[language]
        tree = parser.parse(text.encode("utf-8"))
        out: list[tuple[int, int, str | None, str | None]] = []
        self._walk(tree.root_node.named_children, spec, None, out, ctx, header=None)
        return out

    @staticmethod
    def _rows(node) -> tuple[int, int]:
        s, e = node.start_point.row + 1, node.end_point.row + 1
        if node.end_point.column == 0 and e > s:
            e -= 1
        return s, e

    @staticmethod
    def _resolve_def(node, spec: TSSpec):
        if node.type in spec.wrappers:
            inner = node.child_by_field_name(spec.wrappers[node.type])
            if inner is None:
                inner = next((c for c in reversed(node.named_children) if c.type in spec.defs), None)
            if inner is not None and inner.type in spec.defs:
                return inner, spec.defs[inner.type]
            return None
        if node.type in spec.defs:
            if node.type == "lexical_declaration" and node.end_point.row - node.start_point.row < 2:
                return None  # one-line const: keep it with the surrounding module code
            return node, spec.defs[node.type]
        return None

    @staticmethod
    def _name(node) -> str | None:
        def txt(n):
            return n.text.decode("utf-8", errors="replace") if n is not None else None

        name = node.child_by_field_name("name")
        if name is not None:
            return txt(name)
        if node.type == "impl_item":
            typ, trait = node.child_by_field_name("type"), node.child_by_field_name("trait")
            if typ is not None:
                return f"{txt(trait)} for {txt(typ)}" if trait is not None else txt(typ)
        for child in node.named_children:
            if child.type in ("variable_declarator", "type_spec", "type_alias"):
                n = child.child_by_field_name("name")
                if n is not None:
                    return txt(n)
        for child in node.named_children:
            if child.type in ("identifier", "type_identifier", "property_identifier"):
                return txt(child)
        return None

    def _walk(self, nodes, spec: TSSpec, parent: str | None, out, ctx: _Ctx, header):
        group_type = "module" if parent is None else "body"
        pending: list[tuple[int, int, bool]] = [(*header, False)] if header else []
        run: list[tuple[int, int, str, str]] = []  # consecutive tiny definitions awaiting grouping

        def flush_run() -> None:
            self._flush_run(run, out)
            run.clear()

        for node in nodes:
            s, e = self._rows(node)
            resolved = self._resolve_def(node, spec)
            if resolved is None:
                is_lead = "comment" in node.type or node.type in ("attribute_item", "decorator", "annotation")
                if not is_lead:
                    flush_run()  # real code between definitions breaks a run of tiny ones
                pending.append((s, e, is_lead))
                continue
            start = s
            # Attach directly-preceding comments / attributes (doc comments) to the definition.
            while pending and pending[-1][2] and pending[-1][1] >= start - 1:
                start = pending.pop()[0]
            if pending:
                flush_run()
            self._flush(pending, parent, group_type, out)
            pending = []
            if e - start + 1 <= self.SMALL_DEF_LINES:
                name = self._name(resolved[0]) or resolved[1]
                run.append((start, e, f"{parent}.{name}" if parent else name, resolved[1]))
                continue
            flush_run()
            self._emit_def(resolved[0], resolved[1], start, e, spec, parent, out, ctx)
        flush_run()
        self._flush(pending, parent, group_type, out)

    SMALL_DEF_LINES = 6
    MIN_RUN = 3

    def _flush_run(self, run: list[tuple[int, int, str, str]], out) -> None:
        """Emit tiny definitions; runs of >= MIN_RUN are grouped (e.g. 20 two-line exception classes)."""
        if len(run) < self.MIN_RUN:
            out.extend(run)
            return
        budget = max(self.max_lines // 2, self.SMALL_DEF_LINES)
        group: list[tuple[int, int, str, str]] = []
        for item in run:
            if group and item[1] - group[0][0] + 1 > budget:
                out.append(self._group(group))
                group = []
            group.append(item)
        if group:
            out.append(self._group(group) if len(group) > 1 else group[0])

    @staticmethod
    def _group(group: list[tuple[int, int, str, str]]) -> tuple[int, int, str, str]:
        names = ", ".join(g[2] for g in group)
        return (group[0][0], group[-1][1], names[:300], "definitions")

    def _emit_def(self, defn, sym_type: str, start: int, end: int, spec: TSSpec, parent, out, ctx):
        name = self._name(defn) or sym_type
        full = f"{parent}.{name}" if parent else name
        if end - start + 1 <= self.max_lines:
            out.append((start, end, full, sym_type))
            return
        if defn.type in spec.containers:
            body = defn.child_by_field_name(spec.body_field)
            members = body.named_children if body is not None else []
            if members:
                first_member_start = self._rows(members[0])[0]
                header = (start, max(start, first_member_start - 1))
                self._walk(members, spec, full, out, ctx, header=header)
                return
        for ws, we in self._windows(start, end):
            out.append((ws, we, full, sym_type))

    def _flush(self, pending, parent, group_type, out) -> None:
        if not pending:
            return
        groups: list[list[int]] = []
        for s, e, _ in pending:
            if groups and e - groups[-1][0] + 1 <= self.max_lines:
                groups[-1][1] = max(groups[-1][1], e)
            else:
                groups.append([s, e])
        for s, e in groups:
            if e - s + 1 > self.max_lines:
                out.extend((ws, we, parent, group_type) for ws, we in self._windows(s, e))
            else:
                out.append((s, e, parent, group_type))

    # ---------- docs ----------
    def _markdown_sections(self, lines: list[str]) -> list[tuple[int, int, str | None]]:
        sections, stack = [], []
        cur_start, cur_title, in_fence = 1, None, False
        for i, line in enumerate(lines, start=1):
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = _MD_HEADING.match(line)
            setext = (
                not m
                and i < len(lines)
                and line.strip()
                and not _NOT_TITLE.match(line)
                and _SETEXT.match(lines[i])  # next line (lines is 0-based, i is 1-based)
                and (i == 1 or not lines[i - 2].strip())  # a heading starts its own paragraph
            )
            if (m and len(m.group(1)) <= 3) or setext:
                if i > cur_start:
                    sections.append((cur_start, i - 1, cur_title))
                if m:
                    level, title = len(m.group(1)), m.group(2).strip()
                else:
                    level, title = (1 if lines[i].strip().startswith("=") else 2), line.strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                cur_title, cur_start = " > ".join(t for _, t in stack), i
        sections.append((cur_start, len(lines), cur_title))
        return sections

    def _rst_sections(self, lines: list[str]) -> list[tuple[int, int, str | None]]:
        levels: dict[tuple[str, bool], int] = {}
        stack: list[tuple[int, str]] = []
        sections, cur_start, cur_title = [], 1, None
        i = 0
        while i < len(lines) - 1:
            title, under = lines[i], lines[i + 1]
            if (
                title.strip()
                and not _RST_ADORN.match(title)
                and _RST_ADORN.match(under)
                and len(under.rstrip()) >= len(title.rstrip())
            ):
                over = (
                    i > 0
                    and bool(_RST_ADORN.match(lines[i - 1]))
                    and lines[i - 1].strip()[0] == under.strip()[0]
                )
                level = levels.setdefault((under.strip()[0], over), len(levels) + 1)
                start = i if over else i + 1  # 1-based line of the overline or title
                if level <= 3:
                    if start > cur_start:
                        sections.append((cur_start, start - 1, cur_title))
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title.strip()))
                    cur_title, cur_start = " > ".join(t for _, t in stack), start
                i += 2
                continue
            i += 1
        sections.append((cur_start, len(lines), cur_title))
        return sections

    def _sections(self, sections, ctx: _Ctx):
        # Merge heading-only stubs (e.g. "## Usage" directly followed by "### Install") forward.
        merged: list[list] = []
        carry_start = None
        for s, e, title in sections:
            if carry_start is not None:
                s, carry_start = carry_start, None
            if ctx.nonblank_count(s, e) < 3 and (s, e, title) != tuple(sections[-1]):
                carry_start = s
                continue
            merged.append([s, e, title])
        if carry_start is not None:
            if merged:
                merged[-1][1] = len(ctx.lines)
            else:
                merged.append([carry_start, len(ctx.lines), None])
        out = []
        for s, e, title in merged:
            if e - s + 1 > self.max_lines:
                out.extend((ws, we, title, "section") for ws, we in self._windows(s, e))
            else:
                out.append((s, e, title, "section"))
        return out

    # ---------- windows ----------
    def _windows(self, start: int, end: int) -> list[tuple[int, int]]:
        out, cur, step = [], start, self.window - self.overlap
        while cur <= end:
            we = min(end, cur + self.window - 1)
            out.append((cur, we))
            if we == end:
                break
            cur += step
        return out


class _Ctx:
    def __init__(self, path: str, kind: str, language: str | None, lines: list[str]):
        self.path, self.kind, self.language, self.lines = path, kind, language, lines

    def slice(self, s: int, e: int) -> str:
        return "\n".join(self.lines[s - 1 : e])

    def blank(self, s: int, e: int) -> bool:
        return not any(ln.strip() for ln in self.lines[s - 1 : e])

    def nonblank_count(self, s: int, e: int) -> int:
        return sum(1 for ln in self.lines[s - 1 : e] if ln.strip())
