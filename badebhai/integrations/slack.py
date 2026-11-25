"""Slack slash command: `/badebhai [owner/repo] <question>`.

Slack signs every request (SLACK_SIGNING_SECRET); unsigned or stale (>5 min) requests are rejected.
The command is acknowledged at once ("Looking into it…") and the answer is posted to the
response_url Slack provides, which is only ever a hooks.slack.com URL.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from urllib.parse import parse_qs

_RESPONSE_URL_PREFIX = "https://hooks.slack.com/"


def verify(
    signing_secret: str, timestamp: str | None, body: bytes, signature: str | None, now: float | None = None
) -> bool:
    if not signing_secret or not timestamp or not signature:
        return False
    try:
        if abs((time.time() if now is None else now) - int(timestamp)) > 300:
            return False
    except ValueError:
        return False
    base = b"v0:" + timestamp.encode() + b":" + body
    expected = "v0=" + hmac.new(signing_secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def sign(signing_secret: str, timestamp: str, body: bytes) -> str:
    return (
        "v0="
        + hmac.new(
            signing_secret.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
        ).hexdigest()
    )


@dataclass
class SlackCommand:
    text: str
    user_id: str
    response_url: str


def parse_command(body: bytes) -> SlackCommand | None:
    form = {k: v[0] for k, v in parse_qs(body.decode("utf-8", errors="replace")).items()}
    url = form.get("response_url", "")
    if not url.startswith(_RESPONSE_URL_PREFIX):
        return None
    return SlackCommand(
        text=(form.get("text") or "").strip(), user_id=form.get("user_id", "unknown"), response_url=url
    )


def split_repo(text: str, known: list[str], default: str | None) -> tuple[str | None, str]:
    """`psf/requests where is auth?` -> ("psf/requests", "where is auth?"); else the default repo."""
    first, _, rest = text.partition(" ")
    lowered = {k.lower(): k for k in known}
    if first.lower() in lowered and rest.strip():
        return lowered[first.lower()], rest.strip()
    if default in known:
        return default, text
    if len(known) == 1:
        return known[0], text
    return None, text
