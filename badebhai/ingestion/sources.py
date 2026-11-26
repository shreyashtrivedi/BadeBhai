"""Repository sources: GitHub URLs, other git URLs, local paths. Git helpers.

Managed clones live under DATA_DIR/repos and may be fetched/reset by sync. A user-supplied
local path is never modified: BadeBhai only reads it.

A GitHub token, when configured, reaches git only through environment config for the git process
(an `http.https://github.com/.extraheader`), so it is sent to github.com alone and is never written
into the clone's .git/config, its remote URL, the command line, or logs.
"""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from badebhai.log import get_logger

log = get_logger(__name__)

_GITHUB = re.compile(
    r"^(?:https?://(?:www\.)?github\.com/|git@github\.com:)?(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)
_REF = re.compile(r"^[A-Za-z0-9_.][A-Za-z0-9_./+-]{0,199}$")  # never starts with '-': no git option injection
_REMOTE = re.compile(r"^(?:https?://|ssh://|git@)[^\s]+$")
# GIT_CLONE_FILTER=blob:none: full commit history, file contents fetched only when needed. Much
# lighter for a small hosted instance; git log/blame still work (they fetch what they read).
_CLONE_FILTER = os.environ.get("GIT_CLONE_FILTER", "").strip()
# git runs at low priority on POSIX so it cannot starve the web server on a fractional-CPU host.
_NICE = ["nice", "-n", "10"] if os.name == "posix" and shutil.which("nice") else []


def _memory_limit_bytes() -> int | None:
    """The container's memory limit (cgroup v2 or v1), or None when unlimited / not in a container."""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw.isdigit() and int(raw) < 1 << 50:
            return int(raw)
    return None


# On a small container git must not size itself to the host: index-pack starts one thread per host CPU
# (a container sees them all), each with its own 96 MB delta-base cache, which alone can exceed 512 MB.
_limit = _memory_limit_bytes()
_LOW_MEMORY_GIT = (
    ["-c", "pack.threads=1", "-c", "core.deltaBaseCacheLimit=16m", "-c", "pack.windowMemory=32m"]
    if os.environ.get("GIT_LOW_MEMORY", "").strip().lower() in ("1", "true", "yes")
    or _CLONE_FILTER  # set for small hosted instances
    or (_limit is not None and _limit < 2 << 30)
    else []
)
_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"}


class GitError(RuntimeError):
    pass


def git_env(token: str | None = None) -> dict[str, str]:
    """Environment for git; with a token, authenticate requests to github.com (only) with it."""
    if not token:
        return _GIT_ENV
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return {
        **_GIT_ENV,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}",
    }


def git(
    cwd: Path | str, *args: str, timeout: float = 300, check: bool = True, token: str | None = None
) -> str:
    proc = subprocess.run(
        [*_NICE, "git", "-c", "core.quotepath=off", *_LOW_MEMORY_GIT, *args],
        cwd=str(cwd),
        capture_output=True,
        timeout=timeout,
        env=git_env(token),
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
    if m and (m.group("owner").startswith(".") or m.group("repo") in (".", "..")):
        return None  # ./x, ../x are paths, not GitHub names
    if m and (url.startswith(("http", "git@")) or url.count("/") == 1):
        return f"{m.group('owner')}/{m.group('repo')}"
    return None


class LocalSourceRefused(ValueError):
    pass


def parse_source(
    source: str,
    repos_dir: Path,
    ref: str | None = None,
    name: str | None = None,
    allow_local: bool = True,
) -> RepoSource:
    """Accepts a local path, `owner/name`, a GitHub URL, or any git URL; `@ref` suffix allowed.

    `allow_local=False` (requests from the web API) never reads a path on the server's disk:
    `owner/name` is always GitHub, and paths and file:// URLs are refused."""
    src = source.strip()
    if ref is None and "@" in src and not src.startswith("git@"):
        base, _, maybe_ref = src.rpartition("@")
        if base and "/" not in maybe_ref:
            src, ref = base, maybe_ref
    if ref is not None and not _REF.match(ref):
        raise ValueError(f"{ref!r} is not a valid branch, tag or commit name")
    local = Path(src).expanduser()
    if not allow_local:
        if src.startswith("file:") or (not github_full_name_from_url(src) and not _REMOTE.match(src)):
            raise LocalSourceRefused(
                f"{source!r} is not a GitHub address or git URL (local folders can only be added "
                "from the command line)"
            )
    elif local.exists():
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
    if _REMOTE.match(src) or (allow_local and src.startswith("file://")):
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", src.rstrip("/").removesuffix(".git").split("/")[-1])
        return RepoSource("git", src, repos_dir / slug, name or slug, ref, None)
    raise ValueError(f"cannot resolve repository source {source!r}: not a path, owner/name, or git URL")


def ensure_clone(src: RepoSource, token: str | None = None) -> None:
    """Clone a managed repo if needed and check out `src.ref` (branch, tag, or sha)."""
    if not src.managed:
        return
    if not (src.local_path / ".git").exists():
        src.local_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("cloning", extra={"url": src.url, "dest": str(src.local_path)})
        extra = [f"--filter={_CLONE_FILTER}"] if _CLONE_FILTER else []
        git(
            src.local_path.parent,
            "clone",
            "--quiet",
            *extra,
            src.url,
            str(src.local_path),
            timeout=900,
            token=token,
        )
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


def update_managed_clone(path: Path, ref: str | None, token: str | None = None) -> None:
    """Fetch and move the managed working copy to the latest commit of its tracked ref."""
    git(path, "fetch", "--quiet", "--tags", "--force", "origin", timeout=600, token=token)
    if ref and is_remote_branch(path, ref):
        git(path, "reset", "--quiet", "--hard", f"origin/{ref}")
    elif ref:
        git(path, "checkout", "--quiet", "--detach", ref)


def fetch_commit(path: Path, sha: str, token: str | None = None) -> None:
    """Make sure `sha` is present locally (webhook pushes may be ahead of our clone)."""
    if not _has_commit(path, sha):
        git(path, "fetch", "--quiet", "origin", timeout=600, token=token)


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


def ensure_wiki_clone(full_name: str, repos_dir: Path, token: str | None = None) -> Path | None:
    """Clone (or update) the GitHub wiki of `full_name`. Returns None when the repo has no wiki."""
    dest = repos_dir / (full_name.replace("/", "__") + ".wiki")
    try:
        if (dest / ".git").exists():
            git(dest, "pull", "--quiet", "--ff-only", timeout=120, check=False, token=token)
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
                token=token,
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
