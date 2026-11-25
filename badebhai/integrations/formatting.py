"""Render an answer with its citations for GitHub comments (Markdown) and Slack (mrkdwn)."""

from __future__ import annotations

import re

from badebhai.app.schemas import AskResponse

_DISCLAIMER = "Answers come from a local model and can be wrong; check the cited lines."


def _verified(resp: AskResponse) -> str:
    v = resp.verification
    return (
        f"{v.valid_citations}/{v.total_citations} citations verified" if v.total_citations else "no citations"
    )


def github_markdown(resp: AskResponse, question: str) -> str:
    """A GitHub comment: the answer, a numbered source list with links, and a footer."""
    lines = [f"> {question.strip()[:300]}", "", resp.answer.strip(), ""]
    if resp.citations:
        lines.append("<details><summary>Sources</summary>\n")
        for c in resp.citations:
            label = f"[{c.label}]({c.url})" if c.url else f"`{c.label}`"
            lines.append(f"{c.id}. {label} {c.title[:100]}".rstrip())
        lines.append("\n</details>\n")
    lines.append(f"<sub>🤖 BadeBhai · {_verified(resp)} · {_DISCLAIMER}</sub>")
    return "\n".join(lines)


def slack_mrkdwn(resp: AskResponse, repo: str) -> str:
    """Slack's mrkdwn: *bold*, <url|label> links, citations as links after the answer."""
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", resp.answer.strip())
    text = re.sub(r"^#{1,6}\s+(.*)$", r"*\1*", text, flags=re.MULTILINE)
    parts = [f"*{repo}*", text]
    if resp.citations:
        refs = [f"[{c.id}] <{c.url}|{c.label}>" if c.url else f"[{c.id}] `{c.label}`" for c in resp.citations]
        parts.append("\n".join(refs))
    parts.append(f"_{_verified(resp)} · {_DISCLAIMER}_")
    return "\n\n".join(parts)
