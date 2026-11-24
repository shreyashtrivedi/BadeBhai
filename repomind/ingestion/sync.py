"""Change detection for polling: is there anything newer than the last indexed commit?"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from repomind.db import repos as repo_db
from repomind.db.repos import RepoRecord
from repomind.ingestion.indexer import Indexer, IndexReport
from repomind.ingestion.sources import GitError, git, head_commit, is_remote_branch, working_tree_changes
from repomind.log import get_logger

log = get_logger(__name__)


def remote_head(repo: RepoRecord) -> str | None:
    """Latest sha of the tracked branch on the remote; None for tags/shas (pinned) or errors."""
    if not repo.managed or not repo.ref:
        return None
    path = Path(repo.local_path)
    if not is_remote_branch(path, repo.ref):
        return None  # pinned to a tag or commit: never moves
    try:
        out = git(path, "ls-remote", "origin", f"refs/heads/{repo.ref}", timeout=30).split()
    except GitError as e:
        log.warning("ls-remote failed", extra={"repo": repo.name, "error": str(e)})
        return None
    return out[0] if out else None


def needs_sync(repo: RepoRecord) -> bool:
    if repo.managed:
        head = remote_head(repo)
        return bool(head) and head != repo.last_indexed_commit
    path = Path(repo.local_path)
    return head_commit(path) != repo.last_indexed_commit or bool(working_tree_changes(path))


def sync_due(indexer: Indexer) -> list[IndexReport]:
    reports = []
    for repo in repo_db.list_repos(indexer.pool):
        try:
            if needs_sync(repo):
                reports.append(indexer.sync(repo.name))
        except Exception:
            log.exception("sync failed", extra={"repo": repo.name})
    return reports


async def poll_forever(indexer: Indexer, interval_s: int, stop: asyncio.Event) -> None:
    log.info("sync polling enabled", extra={"interval_s": interval_s})
    while not stop.is_set():
        reports = await asyncio.to_thread(sync_due, indexer)
        for rep in reports:
            log.info(
                "poll sync",
                extra={"repo": rep.repo, "files": rep.files_indexed, "removed": rep.files_removed},
            )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
