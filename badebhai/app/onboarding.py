"""Adding a repository from the web UI: forgiving input, a quick GitHub check before cloning, and
plain-language errors that say when (and how) a GitHub token would help.

The token itself never leaves the server: it is read from the environment (GITHUB_TOKEN), used for
GitHub API calls and git clones, and only its presence is reported to the browser.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from badebhai.config import Settings
from badebhai.ingestion.sources import GitError, LocalSourceRefused, RepoSource, parse_source
from badebhai.mcp_servers.github_client import GitHubClient, GitHubError, GitHubRateLimited

TOKEN_URL = "https://github.com/settings/personal-access-tokens/new"
BIG_REPO_MB = 500

_GITHUB_WEB = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/(?P<owner>[\w.-]+)/(?P<repo>[\w.-]+?)(?:\.git)?"
    r"(?:@(?P<at>[\w.+-][\w./+-]*)|/(?:tree|commit|releases/tag)/(?P<ref>[^/?#\s]+))?(?:[/?#].*)?$",
    re.IGNORECASE,
)


class AddRepoError(Exception):
    """A problem the person adding the repo can act on. `message` is shown as is."""

    def __init__(self, message: str, status: int = 400, needs_token: bool = False, **extra: Any):
        super().__init__(message)
        self.message = message
        self.status = status
        self.needs_token = needs_token
        self.extra = extra

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.message, "needs_token": self.needs_token, **self.extra}


@dataclass
class Checked:
    source: str  # normalized, what the indexer receives
    repo: RepoSource
    about: dict[str, Any] = field(default_factory=dict)  # description, size, default branch, ...
    warnings: list[str] = field(default_factory=list)


def normalize_source(text: str) -> str:
    """Accept what people actually paste: `github.com/o/r`, a browser URL with /tree/<branch> or
    /blob/... paths, a trailing slash or `.git`. Returns `owner/name[@ref]` for GitHub input."""
    text = (text or "").strip().strip("<>\"'")
    m = _GITHUB_WEB.match(text)
    if not m:
        return text
    out = f"{m.group('owner')}/{m.group('repo')}"
    ref = m.group("at") or m.group("ref")
    return f"{out}@{ref}" if ref else out


def check_source(raw: str, s: Settings, gh: GitHubClient | None) -> Checked:
    """Validate a repo address before any download. Raises AddRepoError with a friendly message."""
    if not (raw or "").strip():
        raise AddRepoError("Type a repository address first, for example fastapi/fastapi.")
    source = normalize_source(raw)
    try:
        repo = parse_source(source, s.repos_dir, allow_local=s.api_allow_local_paths)
    except LocalSourceRefused:
        raise AddRepoError(
            "That doesn't look like a GitHub address. Use owner/name (like fastapi/fastapi) or paste the "
            "repository's link. To add a folder on this computer, run "
            "`badebhai ingest <folder>` in a terminal instead."
        ) from None
    except ValueError as e:
        raise AddRepoError(
            f"That doesn't look like a repository address ({e}). Use owner/name, like fastapi/fastapi, "
            "or paste the GitHub link."
        ) from None
    checked = Checked(source=source, repo=repo)
    if repo.kind == "github" and gh is not None:
        _check_github(checked, s, gh)
    return checked


def _check_github(checked: Checked, s: Settings, gh: GitHubClient) -> None:
    full = checked.repo.github_full_name
    try:
        info = gh.get(f"/repos/{full}")
    except GitHubRateLimited:
        checked.warnings.append(
            "GitHub's hourly limit for requests without a token was reached, so the repository could not "
            "be checked first. Downloading it anyway."
        )
        return
    except GitHubError as e:
        if e.status == 404:
            if gh.authenticated:
                raise AddRepoError(
                    f"github.com/{full} wasn't found. Check the spelling. If it is private, the GitHub "
                    "token BadeBhai uses doesn't have access to it.",
                    status=404,
                ) from None
            raise AddRepoError(
                f"github.com/{full} wasn't found. Check the spelling. If it is a private repository, "
                "BadeBhai needs a GitHub token to read it (steps below).",
                status=404,
                needs_token=True,
            ) from None
        return  # GitHub API trouble (network, 5xx): the clone will tell us for sure
    size_mb = round((info.get("size") or 0) / 1024)
    checked.about = {
        "full_name": info.get("full_name") or full,
        "description": info.get("description"),
        "private": bool(info.get("private")),
        "default_branch": info.get("default_branch"),
        "size_mb": size_mb,
        "stars": info.get("stargazers_count"),
    }
    if size_mb > s.api_max_repo_mb:
        raise AddRepoError(
            f"github.com/{full} is about {size_mb:,} MB, over this server's limit of "
            f"{s.api_max_repo_mb:,} MB for repositories added from the web page. Run "
            f"`badebhai ingest {full}` in a terminal, or raise API_MAX_REPO_MB.",
            status=413,
        )
    if size_mb > BIG_REPO_MB:
        checked.warnings.append(
            f"This is a big repository (about {size_mb:,} MB), so downloading and reading it can take a "
            "long time. You can keep using BadeBhai meanwhile."
        )


def explain_failure(exc: BaseException, has_token: bool, kind: str | None) -> dict[str, Any]:
    """Turn an ingest exception into {error, needs_token, details} for the web UI."""
    raw = str(exc)
    low = raw.lower()
    needs_token = False
    if isinstance(exc, subprocess.TimeoutExpired):
        msg = (
            "Downloading took too long. The repository may be very large or the connection slow. "
            "Try again, or use `badebhai ingest` in a terminal."
        )
    elif isinstance(exc, GitError) and any(
        k in low
        for k in (
            "authentication failed",
            "could not read username",
            "terminal prompts disabled",
            "repository not found",
            "not found",
            "403",
        )
    ):
        needs_token = not has_token and kind == "github"
        msg = (
            "The repository couldn't be downloaded. It may be private, or the address may be wrong."
            + (" Private repositories need a GitHub token (steps below)." if needs_token else "")
            + (" The GitHub token doesn't have access to it." if has_token and kind == "github" else "")
        )
    elif isinstance(exc, GitError) and any(
        k in low for k in ("did not match", "unknown revision", "not a tree", "pathspec", "invalid reference")
    ):
        msg = "That version (branch, tag or commit) wasn't found in the repository. Check the name after '@'."
    elif "could not resolve host" in low or "unable to access" in low:
        msg = "Couldn't reach the repository's server. Check the internet connection and try again."
    elif isinstance(exc, ValueError):
        msg = raw
    else:
        msg = "Something went wrong while reading the repository."
    return {"error": msg, "needs_token": needs_token, "details": raw[:600]}
