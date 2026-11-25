"""Programmatic citation verification.

The model writes `[n]` markers that refer to numbered evidence items. This module:
* checks each evidence item is real: the file is indexed, the line range exists, and the cited
  lines still match the file (catches stale or fabricated ranges); commits exist in the clone;
* strips markers that point at unknown or invalid evidence, and flags claims left unsupported;
* checks inline `path:line` references in the prose;
* renumbers surviving citations 1..k in order of appearance and builds the citation list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from badebhai.app.schemas import Citation, Evidence, VerificationReport

CITE_RE = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
INLINE_REF_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_][\w./-]*\.[A-Za-z0-9]{1,8})(?::|#L)(?P<start>\d+)(?:\s*[-–]\s*L?(?P<end>\d+))?"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`*_(\"'])")
_NOT_CLAIM = re.compile(
    r"(?i)(insufficient evidence|couldn'?t find|could not find|no evidence|not found in the evidence|"
    r"unable to (find|determine)|don'?t have (enough )?evidence|the evidence does not)"
)
UNVERIFIED_MARK = " ⚠️(unverified)"
# A trailing "Citations: [1][2][3]" line (small models add one despite the prompt) carries no claim.
_REFERENCES_LINE = re.compile(
    r"(?im)^\s*\**(?:citations|sources|references)\**\s*:?\s*(?:\[\d{1,3}\][\s,]*)+\.?\s*$"
)


class SourceOracle(Protocol):
    def line_count(self, path: str) -> int | None: ...

    def read_lines(self, path: str, start: int, end: int) -> str | None: ...

    def commit_exists(self, sha: str) -> bool: ...

    def find_path(self, partial: str) -> str | None: ...


@dataclass
class VerifiedAnswer:
    answer: str
    citations: list[Citation]
    report: VerificationReport


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip("\n").split("\n"))


def check_evidence(ev: Evidence, oracle: SourceOracle | None) -> str | None:
    """Return a problem description, or None if the evidence item is verifiably real."""
    if ev.type in ("code", "doc"):
        if not ev.path or not ev.start_line or not ev.end_line:
            return "missing path or line range"
        if ev.start_line < 1 or ev.end_line < ev.start_line:
            return f"invalid line range L{ev.start_line}-L{ev.end_line}"
        if oracle is None:
            return None
        n_lines = oracle.line_count(ev.path)
        if n_lines is None:
            return f"file {ev.path} is not in the index"
        if ev.end_line > n_lines:
            return f"line range L{ev.start_line}-L{ev.end_line} exceeds file length ({n_lines} lines)"
        actual = oracle.read_lines(ev.path, ev.start_line, ev.end_line)
        if actual is not None:
            claimed = ev.content
            truncated = claimed.endswith("…")
            claimed = _norm(claimed.rstrip("…").rstrip())
            real = _norm(actual)
            ok = real.startswith(claimed) if truncated else real == claimed
            if not ok:
                return "cited lines do not match the file (stale or fabricated range)"
        return None
    if ev.type == "commit":
        if not ev.sha or not re.fullmatch(r"[0-9a-f]{7,40}", ev.sha):
            return "missing or malformed commit sha"
        if oracle is not None and not oracle.commit_exists(ev.sha):
            return f"commit {ev.sha[:10]} not found in the repository"
        return None
    if ev.type in ("issue", "pr"):
        return None if ev.number is not None else "missing issue/PR number"
    return "unknown evidence type"


def _segments(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of sentence/bullet segments, skipping fenced code blocks."""
    segs: list[tuple[int, int]] = []
    pos, in_fence = 0, False
    for line in text.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence or not line.strip():
            continue
        cur = start
        for m in _SENTENCE_SPLIT.finditer(line):
            segs.append((cur, start + m.start()))
            cur = start + m.end()
        segs.append((cur, end))
    return segs


def _is_claim(seg: str) -> bool:
    plain = CITE_RE.sub("", seg).strip()
    plain = re.sub(r"^([#>*\-+]+|\d+[.)])\s*", "", plain)
    if plain.startswith("#") or plain.endswith(":") or len(plain.split()) < 5:
        return False
    return not _NOT_CLAIM.search(plain)


def verify_answer(answer: str, evidence: list[Evidence], oracle: SourceOracle | None) -> VerifiedAnswer:
    answer = _REFERENCES_LINE.sub("", answer).rstrip()
    report = VerificationReport()
    problems: dict[int, str | None] = {}

    def problem_for(n: int) -> str | None:
        if n not in problems:
            problems[n] = (
                f"[{n}] does not refer to any evidence item"
                if not 1 <= n <= len(evidence)
                else check_evidence(evidence[n - 1], oracle)
            )
        return problems[n]

    renumber: dict[int, int] = {}
    edits: list[tuple[int, int, str]] = []
    segs = _segments(answer)

    def seg_index(offset: int) -> int | None:
        for i, (s, e) in enumerate(segs):
            if s <= offset < e:
                return i
        return None

    seg_cited: dict[int, int] = {}
    seg_valid: dict[int, int] = {}
    for m in CITE_RE.finditer(answer):
        ids = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        si = seg_index(m.start())
        kept: list[int] = []
        for n in ids:
            report.total_citations += 1
            prob = problem_for(n)
            if prob is None:
                report.valid_citations += 1
                if n not in renumber:
                    renumber[n] = len(renumber) + 1
                kept.append(renumber[n])
            else:
                report.invalid_citations.append({"marker": n, "problem": prob})
        if si is not None:
            seg_cited[si] = seg_cited.get(si, 0) + len(ids)
            seg_valid[si] = seg_valid.get(si, 0) + len(kept)
        edits.append((m.start(), m.end(), "".join(f"[{k}]" for k in dict.fromkeys(kept))))

    for i, (s, e) in enumerate(segs):
        seg = answer[s:e]
        if seg_cited.get(i, 0) and not seg_valid.get(i, 0):
            report.flagged_claims.append(CITE_RE.sub("", seg).strip())
            edits.append((e, e, UNVERIFIED_MARK))
        elif not seg_cited.get(i) and _is_claim(seg):
            report.uncited_claims.append(seg.strip())

    for m in INLINE_REF_RE.finditer(answer):
        if oracle is None or m.group("path").startswith(("http", "www.")):
            continue
        path = oracle.find_path(m.group("path"))
        start, end = int(m.group("start")), int(m.group("end") or m.group("start"))
        n_lines = oracle.line_count(path) if path else None
        if path is None or n_lines is None or end > n_lines or start < 1:
            report.bad_inline_refs.append(m.group(0))
            edits.append((m.end(), m.end(), " ⚠️(reference not found)"))

    out = answer
    for s, e, rep in sorted(edits, key=lambda x: (x[0], x[1]), reverse=True):
        out = out[:s] + rep + out[e:]
    out = re.sub(r"[ \t]+([.,;:])", r"\1", out)  # tidy spaces left by removed markers

    citations: list[Citation] = []
    for old, new in sorted(renumber.items(), key=lambda kv: kv[1]):
        ev = evidence[old - 1]
        citations.append(
            Citation(
                id=new,
                evidence_key=ev.key,
                type=ev.type,
                label=ev.label,
                title=ev.title,
                url=ev.url,
                path=ev.path,
                start_line=ev.start_line,
                end_line=ev.end_line,
                sha=ev.sha,
                number=ev.number,
                snippet=ev.content[:600],
            )
        )
    return VerifiedAnswer(out.strip(), citations, report)
