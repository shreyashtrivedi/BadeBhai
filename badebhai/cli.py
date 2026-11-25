"""`badebhai` command-line interface."""

from __future__ import annotations

import json
import sys

import typer

from badebhai.config import configure_tracing, get_settings
from badebhai.log import setup_logging

app = typer.Typer(
    help="BadeBhai: agentic RAG over a live codebase.", no_args_is_help=True, pretty_exceptions_enable=False
)


def _init() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    s = get_settings()
    setup_logging(s.log_level, json_lines=False)
    configure_tracing(s)


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
    from badebhai.ingestion.indexer import Indexer

    _print(Indexer().ingest(source, ref=ref, name=name).to_dict())


@app.command()
def sync(
    repo: str = typer.Argument(None, help="repo name; omit with --all"),
    all_repos: bool = typer.Option(False, "--all", help="sync every indexed repo"),
    watch: bool = typer.Option(False, "--watch", help="keep polling for new commits"),
    interval: int = typer.Option(30, help="seconds between polls with --watch"),
) -> None:
    """Re-index only what changed since the last indexed commit (optionally keep polling)."""
    _init()
    import time

    from badebhai.db import repos as repo_db
    from badebhai.db.pool import get_pool
    from badebhai.ingestion.indexer import Indexer
    from badebhai.ingestion.sync import needs_sync

    idx = Indexer()
    if not all_repos and not repo:
        raise typer.BadParameter("give a repo name or --all")
    while True:
        records = repo_db.list_repos(get_pool())
        targets = [r for r in records if all_repos or r.name == repo]
        if not targets:
            raise typer.BadParameter(f"unknown repo {repo!r}")
        for r in targets:
            if not watch or needs_sync(r):
                _print(idx.sync(r.name).to_dict())
        if not watch:
            break
        time.sleep(interval)


@app.command()
def repos() -> None:
    """List indexed repositories."""
    _init()
    from badebhai.db import repos as repo_db
    from badebhai.db.pool import get_pool

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
    from badebhai.db import repos as repo_db
    from badebhai.db.pool import get_pool
    from badebhai.embeddings import get_embedder
    from badebhai.retrieval.hybrid import hybrid_search

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
    from badebhai import compat
    from badebhai.app.service import ask_once

    resp = compat.run(ask_once(repo, question, mode=mode, user_id=user, thread_id=thread))
    typer.echo(resp.answer + "\n")
    for c in resp.citations:
        typer.echo(f"[{c.id}] {c.label}  {c.url or ''}")
    v = resp.verification
    typer.echo(
        f"\ncitations valid: {v.valid_citations}/{v.total_citations}   "
        f"latency: {resp.latency_ms} ms   cost: ${resp.usage.get('cost_usd', 0):.4f}"
    )


MCP_SERVERS = {
    "code_search": ("badebhai.mcp_servers.code_search", 8101),
    "git_history": ("badebhai.mcp_servers.git_history", 8102),
    "docs": ("badebhai.mcp_servers.docs", 8103),
}
mcp_app = typer.Typer(help="Run the MCP data-source servers.", no_args_is_help=True)
app.add_typer(mcp_app, name="mcp")


def _spawn_mcp_servers(host: str = "127.0.0.1"):
    import socket
    import subprocess
    import time

    procs = []
    for _name, (module, port) in MCP_SERVERS.items():
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    module,
                    "--transport",
                    "streamable-http",
                    "--host",
                    host,
                    "--port",
                    str(port),
                ]
            )
        )
    deadline = time.time() + 90
    for _name, (_, port) in MCP_SERVERS.items():
        while time.time() < deadline:
            try:  # a GET on /mcp opens a long-lived stream, so probe the TCP port instead
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
    return procs


@mcp_app.command("serve")
def mcp_serve(
    server: str = typer.Argument(..., help="code_search | git_history | docs"),
    transport: str = typer.Option("streamable-http", help="streamable-http | stdio"),
    host: str = "127.0.0.1",
    port: int = typer.Option(None, help="defaults: code_search 8101, git_history 8102, docs 8103"),
) -> None:
    """Run one MCP server in the foreground."""
    import runpy

    if server not in MCP_SERVERS:
        raise typer.BadParameter(f"unknown server {server!r}")
    module, default_port = MCP_SERVERS[server]
    sys.argv = [module, "--transport", transport, "--host", host, "--port", str(port or default_port)]
    runpy.run_module(module, run_name="__main__")


@mcp_app.command("up")
def mcp_up(host: str = "127.0.0.1") -> None:
    """Run all three MCP servers (Streamable HTTP) until Ctrl-C."""
    procs = _spawn_mcp_servers(host)
    typer.echo("MCP servers: " + ", ".join(f"{n} :{p}" for n, (_, p) in MCP_SERVERS.items()))
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    with_mcp: bool = typer.Option(True, "--with-mcp/--no-mcp", help="also start the 3 MCP servers locally"),
) -> None:
    """Run the API (and web UI). By default also launches the MCP servers as child processes."""
    import uvicorn

    from badebhai.compat import UVICORN_LOOP

    procs = _spawn_mcp_servers() if with_mcp and get_settings().mcp_transport == "http" else []
    try:
        uvicorn.run(
            "badebhai.app.main:app", host=host, port=port, reload=reload, loop=UVICORN_LOOP, log_level="info"
        )
    finally:
        for p in procs:
            p.terminate()


memories_app = typer.Typer(help="Inspect and delete long-term memories.", no_args_is_help=True)
app.add_typer(memories_app, name="memories")


def _memory_store():
    from badebhai.app.memory.store import MemoryStore
    from badebhai.db.pool import get_pool
    from badebhai.embeddings import get_embedder

    s = get_settings()
    return MemoryStore(get_pool(s), get_embedder(s), s)


@memories_app.command("list")
def memories_list(user: str = typer.Option(..., help="user id"), repo: str = typer.Option(None)) -> None:
    """List a user's active memories."""
    _init()
    for m in _memory_store().list(user, repo):
        typer.echo(
            f"#{m.id:<5} {m.repo:24s} [{m.category}] imp={m.importance:.2f} used={m.use_count}  {m.text}"
        )


@memories_app.command("delete")
def memories_delete(memory_id: int, user: str = typer.Option(..., help="user id")) -> None:
    """Delete one memory."""
    _init()
    typer.echo("deleted" if _memory_store().delete(user, memory_id) else "not found")


@memories_app.command("clear")
def memories_clear(user: str = typer.Option(..., help="user id"), repo: str = typer.Option(None)) -> None:
    """Delete all of a user's memories (optionally for one repo)."""
    _init()
    typer.echo(f"deleted {_memory_store().delete_all(user, repo)}")


@app.command("eval")
def eval_cmd(
    systems: str = typer.Option("naive,agent", help="comma-separated: naive, agent"),
    limit: int = typer.Option(None, help="only the first N questions"),
    ids: str = typer.Option(None, help="comma-separated question ids"),
    retrieval_only: bool = typer.Option(False, help="hit@k of naive retrieval only (no LLM calls)"),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="LLM-judge answer correctness"),
    golden: str = typer.Option(None, help="golden YAML (default: psf/requests v2.32.3)"),
    mcp: str = typer.Option("inmemory", help="MCP transport for the agent: inmemory | http | stdio"),
    judge_model: str = typer.Option(None, help="model for the LLM judge (default: the main model)"),
    resume: str = typer.Option(None, help="run id to resume (results are checkpointed as they finish)"),
) -> None:
    """Run the golden-question eval and write badebhai/evals/reports/latest.md."""
    _init()
    from badebhai import compat
    from badebhai.evals.golden import DEFAULT_GOLDEN
    from badebhai.evals.report import REPORTS_DIR, markdown, write
    from badebhai.evals.runner import EvalRunner
    from badebhai.llm import get_llm

    s = get_settings().model_copy(update={"mcp_transport": mcp})
    judge_llm = None
    if judge and not retrieval_only:
        judge_llm = get_llm(s.model_copy(update={"llm_model": judge_model}) if judge_model else s)
    runner = EvalRunner(s, judge_llm=judge_llm)
    report = compat.run(
        runner.evaluate(
            golden or DEFAULT_GOLDEN,
            tuple(x.strip() for x in systems.split(",") if x.strip()),
            limit=limit,
            ids=[x.strip() for x in ids.split(",")] if ids else None,
            retrieval_only=retrieval_only,
            checkpoint_dir=REPORTS_DIR / "raw",
            resume=resume,
        )
    )
    raw, md = write(report)
    typer.echo(markdown(report))
    typer.echo(f"raw results: {raw}")
    typer.echo(f"report: {md}")


@app.command("reset-db")
def reset_db(yes: bool = typer.Option(False, "--yes", help="skip confirmation")) -> None:
    """Drop all BadeBhai tables (indexed data and memories)."""
    _init()
    from badebhai.db.pool import reset_database

    if not yes and not typer.confirm("Drop all BadeBhai tables (index + memories)?"):
        raise typer.Abort()
    reset_database()
    typer.echo("dropped")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
