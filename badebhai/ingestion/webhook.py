"""GitHub push webhooks: signature verification and payload parsing."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    """Constant-time check of `X-Hub-Signature-256: sha256=<hex hmac>`."""
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.strip())


def sign(secret: str, body: bytes) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


@dataclass
class PushEvent:
    full_name: str
    branch: str | None
    before: str
    after: str
    added: set[str] = field(default_factory=set)
    modified: set[str] = field(default_factory=set)
    removed: set[str] = field(default_factory=set)
    deleted: bool = False

    @property
    def touched(self) -> set[str]:
        return self.added | self.modified | self.removed


def parse_push(payload: dict) -> PushEvent | None:
    repo = payload.get("repository") or {}
    full_name = repo.get("full_name")
    ref = payload.get("ref", "")
    if not full_name or not payload.get("after"):
        return None
    ev = PushEvent(
        full_name=full_name,
        branch=ref.removeprefix("refs/heads/") if ref.startswith("refs/heads/") else None,
        before=payload.get("before", ""),
        after=payload["after"],
        deleted=bool(payload.get("deleted")),
    )
    for commit in payload.get("commits", []) or []:
        ev.added.update(commit.get("added", []))
        ev.modified.update(commit.get("modified", []))
        ev.removed.update(commit.get("removed", []))
    return ev
