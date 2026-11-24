"""`repomind` command-line interface."""

from __future__ import annotations

import json
import sys

import typer

from repomind.config import get_settings
from repomind.log import setup_logging

app = typer.Typer(
    help="RepoMind: agentic RAG over a live codebase.", no_args_is_help=True, pretty_exceptions_enable=False
)


def _init() -> None:
    s = get_settings()
    setup_logging(s.log_level, json_lines=False)


def _print(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.command()
def ingest(
    source: str = typer.Argument(..., help="local path, owner/name, GitHub URL or git URL; '@ref' allowed"),
    ref: str = typer.Option(None, help="branch, tag or sha to index"),
    name: str = typer.Option(None, help="name to register the repo under"),
) -> None:
    """Clone (if needed) and fully index a repository."""
    _init()
    from repomind.ingestion.indexer import Indexer

    _print(Indexer().ingest(source, ref=ref, name=name).to_dict())


@app.command()
def sync(
    repo: str = typer.Argument(None, help="repo name; omit with --all"),
    all_repos: bool = typer.Option(False, "--all", help="sync every indexed repo"),
) -> None:
    """Re-index only what changed since the last indexed commit."""
    _init()
    from repomind.db import repos as repo_db
    from repomind.db.pool import get_pool
    from repomind.ingestion.indexer import Indexer

    idx = Indexer()
    names = [r.name for r in repo_db.list_repos(get_pool())] if all_repos else [repo]
    if not names or names == [None]:
        raise typer.BadParameter("give a repo name or --all")
    for n in names:
        _print(idx.sync(n).to_dict())


@app.command()
def repos() -> None:
    """List indexed repositories."""
    _init()
    from repomind.db import repos as repo_db
    from repomind.db.pool import get_pool

    _print([r.to_public() for r in repo_db.list_repos(get_pool())])


@app.command()
def search(
    repo: str,
    query: str,
    kind: str = typer.Option("code", help="code | doc | all"),
    k: int = typer.Option(8, "-k", "--k", help="number of results"),
) -> None:
    """Run hybrid search directly (debugging)."""
    _init()
    from repomind.db import repos as repo_db
    from repomind.db.pool import get_pool
    from repomind.embeddings import get_embedder
    from repomind.retrieval.hybrid import hybrid_search

    pool = get_pool()
    r = repo_db.get_repo(pool, repo)
    if r is None:
        raise typer.BadParameter(f"unknown repo {repo!r}")
    kinds = ("code", "doc") if kind == "all" else (kind,)
    for h in hybrid_search(pool, get_embedder(), r.id, query, k=k, kinds=kinds):
        typer.echo(
            f"{h.score:7.3f}  {h.path}:{h.start_line}-{h.end_line}  {h.symbol or ''}"
            f"  (vec={h.vector_rank}, kw={h.keyword_rank})"
        )


@app.command()
def ask(
    repo: str,
    question: str,
    mode: str = typer.Option("agent", help="agent | naive"),
    user: str = typer.Option("cli-user"),
    thread: str = typer.Option(None),
) -> None:
    """Ask a question from the terminal (runs the same pipeline as the API)."""
    _init()
    from repomind import compat
    from repomind.app.service import ask_once

    resp = compat.run(ask_once(repo, question, mode=mode, user_id=user, thread_id=thread))
    typer.echo(resp.answer + "\n")
    for c in resp.citations:
        typer.echo(f"[{c.id}] {c.label}  {c.url or ''}")
    v = resp.verification
    typer.echo(
        f"\ncitations valid: {v.valid_citations}/{v.total_citations}   "
        f"latency: {resp.latency_ms} ms   cost: ${resp.usage.get('cost_usd', 0):.4f}"
    )


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Run the API (and web UI)."""
    import uvicorn

    from repomind.compat import UVICORN_LOOP

    uvicorn.run(
        "repomind.app.main:app", host=host, port=port, reload=reload, loop=UVICORN_LOOP, log_level="info"
    )


@app.command("reset-db")
def reset_db(yes: bool = typer.Option(False, "--yes", help="skip confirmation")) -> None:
    """Drop all RepoMind tables (indexed data and memories)."""
    _init()
    from repomind.db.pool import reset_database

    if not yes and not typer.confirm("Drop all RepoMind tables (index + memories)?"):
        raise typer.Abort()
    reset_database()
    typer.echo("dropped")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
