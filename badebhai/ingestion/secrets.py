"""Secret detection and line-preserving redaction.

Redaction never adds or removes newlines, so line numbers in citations stay valid.
"""

from __future__ import annotations

import math
import re
from collections import Counter

REDACTED = "[REDACTED]"

# High-confidence token formats.
_TOKEN_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),  # GitHub tokens
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),  # Slack
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}"),  # Anthropic
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}\b"),  # OpenAI-style
    re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),  # Google API key
    re.compile(r"\b[sr]k_live_[0-9a-zA-Z]{20,}\b"),  # Stripe
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),  # JWT
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),  # GitLab
]

# key = "value" assignments where the key looks sensitive. Value redacted only if it looks random.
_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<key>[A-Za-z0-9_.\-]*(?:password|passwd|secret|api[_-]?key|access[_-]?key|auth[_-]?token|
        private[_-]?key|client[_-]?secret|token)[A-Za-z0-9_.\-]*)
    (?P<sep>\s*["']?\s*[:=]\s*)
    (?P<q>["'])(?P<val>[^"'\s]{12,})(?P=q)
    """
)
_PLACEHOLDER = re.compile(r"(?i)(your|example|changeme|placeholder|dummy|xxxx|<|\$\{|test|fake|sample)")
_PEM_BEGIN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_PEM_END = re.compile(r"-----END [A-Z ]*PRIVATE KEY-----")


def _entropy(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _looks_random(val: str) -> bool:
    if _PLACEHOLDER.search(val):
        return False
    classes = sum(bool(re.search(p, val)) for p in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]"))
    return _entropy(val) >= 3.2 and classes >= 2


def redact(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_redactions)."""
    count = 0
    lines = text.split("\n")
    in_pem = False
    for i, line in enumerate(lines):
        if in_pem:
            if _PEM_END.search(line):
                in_pem = False
            else:
                lines[i] = REDACTED
                count += 1
            continue
        if _PEM_BEGIN.search(line) and not _PEM_END.search(line):
            in_pem = True
            continue
        for pat in _TOKEN_PATTERNS:
            line, n = pat.subn(REDACTED, line)
            count += n

        def _sub(m: re.Match) -> str:
            nonlocal count
            if not _looks_random(m.group("val")):
                return m.group(0)
            count += 1
            return f"{m.group('key')}{m.group('sep')}{m.group('q')}{REDACTED}{m.group('q')}"

        lines[i] = _ASSIGNMENT.sub(_sub, line)
    return "\n".join(lines), count


def contains_secret(text: str) -> bool:
    if _PEM_BEGIN.search(text):
        return True
    _, n = redact(text)
    return n > 0
