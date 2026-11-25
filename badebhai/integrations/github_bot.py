"""GitHub bot: answer `@badebhai <question>` comments on issues and pull requests.

Off unless GITHUB_BOT_ENABLED=true, because it writes to GitHub (one comment per question). It only
answers:
* `issue_comment` "created" events delivered to the signed webhook (GITHUB_WEBHOOK_SECRET),
* on repos BadeBhai has indexed,
* from the repo's owners, members and collaborators (GITHUB_BOT_ALLOW_ANYONE=true widens it),
* never from bots, so it cannot reply to itself or start loops.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from badebhai.config import Settings

_TRUSTED = {"OWNER", "MEMBER", "COLLABORATOR"}


@dataclass
class BotQuestion:
    full_name: str
    number: int
    question: str
    author: str
    issue_title: str
    is_pull_request: bool


def parse_mention(payload: dict, s: Settings) -> tuple[BotQuestion | None, str]:
    """(question, "") when the comment should be answered, else (None, reason it is ignored)."""
    if payload.get("action") != "created":
        return None, "only new comments are answered"
    comment = payload.get("comment") or {}
    issue = payload.get("issue") or {}
    repo = payload.get("repository") or {}
    user = comment.get("user") or {}
    body = comment.get("body") or ""
    handle = s.github_bot_handle.strip()
    if not re.search(rf"(?i)(^|\s){re.escape(handle)}\b", body):
        return None, f"no {handle} mention"
    if user.get("type") == "Bot" or (user.get("login") or "").endswith("[bot]"):
        return None, "comments by bots are ignored"
    if not s.github_bot_allow_anyone and comment.get("author_association") not in _TRUSTED:
        return None, "only owners, members and collaborators can ask"
    question = re.sub(rf"(?i)(^|\s){re.escape(handle)}\b[:,]?", " ", body).strip()
    question = re.sub(r"(?m)^>.*$", "", question).strip()  # drop quoted text
    if not question:
        question = f"Which code is relevant to this issue: {issue.get('title', '')}?"
    if not repo.get("full_name") or not issue.get("number"):
        return None, "malformed payload"
    return (
        BotQuestion(
            full_name=repo["full_name"],
            number=int(issue["number"]),
            question=question[:2000],
            author=user.get("login") or "someone",
            issue_title=(issue.get("title") or "")[:300],
            is_pull_request="pull_request" in issue,
        ),
        "",
    )
