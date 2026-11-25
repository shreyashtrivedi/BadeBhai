"""Which files get indexed: .gitignore, vendored dirs, binaries, size caps."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath

import pathspec

SKIP_DIRS = {
    ".git",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "target",
    ".tox",
    ".nox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "coverage",
    ".next",
    ".nuxt",
    "site-packages",
    ".idea",
    ".vscode",
    "bower_components",
    ".gradle",
    ".terraform",
}
SKIP_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "uv.lock",
    "Cargo.lock",
    "go.sum",
    "composer.lock",
    "Gemfile.lock",
    "Pipfile.lock",
    ".DS_Store",
}
BINARY_EXT = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".webp",
    ".svg",
    ".pdf",
    ".zip",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".tar",
    ".rar",
    ".whl",
    ".egg",
    ".jar",
    ".war",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
    ".bin",
    ".o",
    ".a",
    ".class",
    ".pyc",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".wav",
    ".flac",
    ".ogg",
    ".psd",
    ".db",
    ".sqlite",
    ".parquet",
    ".npy",
    ".pkl",
    ".pt",
    ".onnx",
    ".csv",
    ".tsv",
    ".ipynb",
    ".lockb",
    ".icns",
    ".map",
}


def _path_allowed(rel: str) -> bool:
    p = PurePosixPath(rel)
    if any(part in SKIP_DIRS for part in p.parts[:-1]):
        return False
    name = p.name
    if name in SKIP_NAMES or name.endswith((".min.js", ".min.css")):
        return False
    return p.suffix.lower() not in BINARY_EXT


def is_git_repo(root: Path) -> bool:
    return (root / ".git").exists()


def list_candidate_files(root: Path) -> list[str]:
    """Repo-relative POSIX paths, respecting .gitignore. Uses `git ls-files` when possible."""
    root = Path(root)
    paths: list[str]
    if is_git_repo(root):
        out = subprocess.run(
            [
                "git",
                "-c",
                "core.quotepath=off",
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8", errors="replace")
        paths = [p for p in out.split("\0") if p]
    else:
        spec = None
        gi = root / ".gitignore"
        if gi.exists():
            spec = pathspec.PathSpec.from_lines("gitwildmatch", gi.read_text(errors="ignore").splitlines())
        paths = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                rel = Path(dirpath, fn).relative_to(root).as_posix()
                if spec is None or not spec.match_file(rel):
                    paths.append(rel)
    return sorted(p for p in paths if _path_allowed(p) and (root / p).is_file())


def read_text_file(path: Path, max_bytes: int) -> tuple[bytes, str] | None:
    """Return (raw_bytes, text) or None when the file is too big, binary, or looks generated."""
    try:
        size = path.stat().st_size
        if size > max_bytes or size == 0:
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw:  # binary (and Postgres text cannot hold NUL)
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError:
            return None
    lines = text.count("\n") + 1
    if len(text) / lines > 400:  # minified / generated
        return None
    return raw, text
