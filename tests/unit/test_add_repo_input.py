"""Adding repos from the web UI: forgiving input, refusing server paths, token handling, plain errors."""

import base64
import subprocess
from pathlib import Path

import pytest

from badebhai.app.onboarding import explain_failure, normalize_source
from badebhai.ingestion.sources import GitError, LocalSourceRefused, git_env, parse_source


@pytest.mark.parametrize(
    ("pasted", "expected"),
    [
        ("fastapi/fastapi", "fastapi/fastapi"),
        ("  https://github.com/pallets/flask  ", "pallets/flask"),
        ("github.com/pallets/flask/", "pallets/flask"),
        ("https://www.github.com/pallets/flask.git", "pallets/flask"),
        ("https://github.com/psf/requests/tree/v2.32.3/src/requests", "psf/requests@v2.32.3"),
        ("https://github.com/psf/requests/blob/main/README.md", "psf/requests"),
        ("github.com/psf/requests@v2.32.3", "psf/requests@v2.32.3"),
        ("psf/requests@v2.32.3", "psf/requests@v2.32.3"),
        ("https://gitlab.com/group/project", "https://gitlab.com/group/project"),
    ],
)
def test_pasted_addresses_are_normalized(pasted, expected):
    assert normalize_source(pasted) == expected


def test_web_requests_never_read_server_paths(tmp_path: Path, monkeypatch):
    (tmp_path / "acme" / "tiny").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    # From the CLI an existing folder wins; from the web, owner/name always means GitHub.
    assert parse_source("acme/tiny", tmp_path / "repos").kind == "local"
    assert parse_source("acme/tiny", tmp_path / "repos", allow_local=False).kind == "github"
    for bad in (str(tmp_path), "./acme", "file:///etc", "C:\\Users\\someone\\code", "~/.ssh"):
        with pytest.raises(LocalSourceRefused):
            parse_source(bad, tmp_path / "repos", allow_local=False)
    assert parse_source("https://gitlab.com/g/p.git", tmp_path, allow_local=False).kind == "git"


def test_refs_cannot_smuggle_git_options(tmp_path: Path):
    for bad in ("o/r@--upload-pack=touch", "o/r@-x", "o/r@a b"):
        with pytest.raises(ValueError):
            parse_source(bad, tmp_path)
    assert parse_source("o/r@v1.2.3", tmp_path).ref == "v1.2.3"


def test_token_reaches_git_only_as_a_github_scoped_header():
    assert "GIT_CONFIG_COUNT" not in git_env(None)
    env = git_env("ghp_secret")
    assert env["GIT_CONFIG_KEY_0"] == "http.https://github.com/.extraheader"
    assert base64.b64decode(env["GIT_CONFIG_VALUE_0"].split()[-1]).decode() == "x-access-token:ghp_secret"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_clone_with_token_keeps_it_out_of_the_clone(tmp_path: Path):
    from badebhai.ingestion.sources import git

    src = tmp_path / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    git(tmp_path, "clone", "--quiet", src.as_uri(), str(tmp_path / "dst"), token="ghp_secret")
    config = (tmp_path / "dst" / ".git" / "config").read_text()
    assert "ghp_secret" not in config and "extraheader" not in config


def test_failures_are_explained_and_say_when_a_token_helps():
    private = GitError(
        "git clone --quiet https://github.com/acme/secret.git failed: fatal: could not read Username for "
        "'https://github.com': terminal prompts disabled"
    )
    out = explain_failure(private, has_token=False, kind="github")
    assert out["needs_token"] and "private" in out["error"] and "token" in out["error"]
    assert "terminal prompts" in out["details"]
    with_token = explain_failure(private, has_token=True, kind="github")
    assert not with_token["needs_token"] and "doesn't have access" in with_token["error"]
    bad_ref = GitError("git checkout --quiet --detach failed: error: pathspec 'v9' did not match any file(s)")
    assert "version" in explain_failure(bad_ref, False, "github")["error"]
    timeout = subprocess.TimeoutExpired(["git", "clone"], 900)
    assert "too long" in explain_failure(timeout, False, "github")["error"]
