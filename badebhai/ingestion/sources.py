"""Repository sources: GitHub URLs, other git URLs, local paths. Git helpers.

Managed clones live under DATA_DIR/repos and may be fetched/reset by sync. A user-supplied
local path is never modified: BadeBhai only reads it.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from badebhai.log import get_logger

log = get_logger(__name__)

_GITHUB = re.compile(
    r"^(?:https?://(?:www\.)?github\.com/|git@github\.com:)?(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"}


class GitError(RuntimeError):
    pass


def git(cwd: Path | str, *args: str, timeout: float = 300, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=off", *args],
        cwd=str(cwd),
        capture_output=True,
        timeout=timeout,
        env=_GIT_ENV,
    )
    if check and proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args[:3])} failed: {proc.stderr.decode(errors='replace').strip()[:500]}"
        )
    return proc.stdout.decode("utf-8", errors="replace")


@dataclass
class RepoSource:
    kind: str  # github | git | local
    url: str | None
    local_path: Path
    name: str
    ref: str | None
    github_full_name: str | None

    @property
    def managed(self) -> bool:
        return self.kind != "local"


def github_full_name_from_url(url: str) -> str | None:
    m = _GITHUB.match(url.strip())
    if m and (url.startswith(("http", "git@")) or url.count("/") == 1):
        return f"{m.group('owner')}/{m.group('repo')}"
    return None


def parse_source(source: str, repos_dir: Path, ref: str | None = None, name: str | None = None) -> RepoSource:
    """Accepts a local path, `owner/name`, a GitHub URL, or any git URL; `@ref` suffix allowed."""
    src = source.strip()
    if ref is None and "@" in src and not src.startswith("git@"):
        base, _, maybe_ref = src.rpartition("@")
        if base and "/" not in maybe_ref:
            src, ref = base, maybe_ref
    local = Path(src).expanduser()
    if local.exists():
        local = local.resolve()
        full = None
        if (local / ".git").exists():
            origin = git(local, "remote", "get-url", "origin", check=False).strip()
            full = github_full_name_from_url(origin) if origin else None
        return RepoSource("local", None, local, name or full or local.name, ref, full)
    full = github_full_name_from_url(src)
    if full:
        slug = full.replace("/", "__")
        return RepoSource(
            "github", f"https://github.com/{full}.git", repos_dir / slug, name or full, ref, full
        )
    if src.startswith(("http://", "https://", "git@", "ssh://", "file://")):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.rstrip("/").removesuffix(".git").split("/")[-1])
        return RepoSource("git", src, repos_dir / slug, name or slug, ref, None)
    raise ValueError(f"cannot resolve repository source {source!r}: not a path, owner/name, or git URL")


def ensure_clone(src: RepoSource) -> None:
    """Clone a managed repo if needed and check out `src.ref` (branch, tag, or sha)."""
    if not src.managed:
        return
    if not (src.local_path / ".git").exists():
        src.local_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("cloning", extra={"url": src.url, "dest": str(src.local_path)})
        git(src.local_path.parent, "clone", "--quiet", src.url, str(src.local_path), timeout=900)
    if src.ref is None:
        src.ref = git(src.local_path, "rev-parse", "--abbrev-ref", "HEAD").strip()
    checkout_ref(src.local_path, src.ref)


def is_remote_branch(path: Path, ref: str) -> bool:
    return bool(
        git(path, "rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{ref}", check=False).strip()
    )


def checkout_ref(path: Path, ref: str) -> None:
    if is_remote_branch(path, ref):
        git(path, "checkout", "--quiet", "-B", ref, f"origin/{ref}")
    else:
        git(path, "checkout", "--quiet", "--detach", ref)


def update_managed_clone(path: Path, ref: str | None) -> None:
    """Fetch and move the managed working copy to the latest commit of its tracked ref."""
    git(path, "fetch", "--quiet", "--tags", "--force", "origin", timeout=600)
    if ref and is_remote_branch(path, ref):
        git(path, "reset", "--quiet", "--hard", f"origin/{ref}")
    elif ref:
        git(path, "checkout", "--quiet", "--detach", ref)


def fetch_commit(path: Path, sha: str) -> None:
    """Make sure `sha` is present locally (webhook pushes may be ahead of our clone)."""
    if not _has_commit(path, sha):
        git(path, "fetch", "--quiet", "origin", timeout=600)


def _has_commit(path: Path, sha: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=str(path), capture_output=True, env=_GIT_ENV
        ).returncode
        == 0
    )


def head_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    out = git(path, "rev-parse", "HEAD", check=False).strip()
    return out or None


def diff_paths(path: Path, old: str, new: str = "HEAD") -> tuple[set[str], set[str]] | None:
    """(changed_or_added, removed) between two commits; None if `old` is unknown."""
    if not _has_commit(path, old):
        return None
    out = git(path, "diff", "--name-status", "-M", old, new)
    changed, removed = set(), set()
    for line in out.splitlines():
        parts = line.split("\t")
        status = parts[0][:1]
        if status == "D":
            removed.add(parts[1])
        elif status == "R":
            removed.add(parts[1])
            changed.add(parts[2])
        elif status in "AMCT" and len(parts) >= 2:
            changed.add(parts[-1])
    return changed, removed


def working_tree_changes(path: Path) -> set[str]:
    """Uncommitted modifications in a local repo (indexed as-is)."""
    out = git(path, "status", "--porcelain=v1", "-z", "--untracked-files=normal", check=False)
    entries = [e for e in out.split("\0") if e]
    return {e[3:] for e in entries if len(e) > 3}


def ensure_wiki_clone(full_name: str, repos_dir: Path) -> Path | None:
    """Clone (or update) the GitHub wiki of `full_name`. Returns None when the repo has no wiki."""
    dest = repos_dir / (full_name.replace("/", "__") + ".wiki")
    try:
        if (dest / ".git").exists():
            git(dest, "pull", "--quiet", "--ff-only", timeout=120, check=False)
        else:
            git(
                repos_dir,
                "clone",
                "--quiet",
                "--depth",
                "50",
                f"https://github.com/{full_name}.wiki.git",
                str(dest),
                timeout=120,
            )
        return dest
    except (GitError, subprocess.TimeoutExpired):
        return None


def safe_join(base: Path, rel: str) -> Path:
    """Join a repo-relative path to its base directory, refusing traversal outside it."""
    base = Path(base).resolve()
    target = (base / rel).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"path escapes repository: {rel!r}")
    return target
