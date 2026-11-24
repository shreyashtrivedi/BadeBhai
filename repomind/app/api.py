"""FastAPI backend."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from repomind import __version__
from repomind.app.agent.checkpoint import postgres_checkpointer
from repomind.app.agent.runner import AgentRunner
from repomind.app.mcp_client import McpToolbox
from repomind.app.memory.store import MemoryStore
from repomind.app.naive import answer_naive
from repomind.app.schemas import AskRequest, AskResponse
from repomind.config import Settings, configure_tracing, get_settings
from repomind.db import repos as repo_db
from repomind.db.pool import get_pool
from repomind.embeddings import get_embedder
from repomind.ingestion.indexer import Indexer
from repomind.ingestion.sync import poll_forever
from repomind.ingestion.webhook import parse_push, verify_signature
from repomind.llm import LLMError, LLMProvider, LLMUnavailable, get_llm, track_usage
from repomind.log import get_logger, setup_logging
from repomind.retrieval.hybrid import hybrid_search

log = get_logger(__name__)
UI_DIR = Path(__file__).resolve().parent.parent / "ui"


class IngestRequest(BaseModel):
    source: str
    ref: str | None = None
    name: str | None = None


def create_app(
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
    toolbox_factory: Callable[[], McpToolbox] | None = None,
    checkpointer=None,
) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(s.log_level, s.log_json)
        tracing = configure_tracing(s)
        st = app.state
        st.settings = s
        st.pool = await asyncio.to_thread(get_pool, s)
        st.embedder = get_embedder(s)
        # Load the embedding model now, not during the first question (memory search embeds the query).
        await asyncio.to_thread(st.embedder.embed_query, "warm up")
        st.llm = llm or get_llm(s)
        st.jobs = {}
        st.indexer = Indexer(s, st.pool, st.embedder)
        st.memory = MemoryStore(st.pool, st.embedder, s)
        saver_cm = None
        saver = checkpointer
        if saver is None:  # short-term memory survives restarts: Postgres checkpointer
            saver_cm = postgres_checkpointer(s.database_url)
            saver = await saver_cm.__aenter__()
        st.runner = AgentRunner(
            s,
            st.llm,
            st.pool,
            st.embedder,
            checkpointer=saver,
            toolbox_factory=toolbox_factory,
            memory_store=st.memory,
        )
        stop = asyncio.Event()
        background: list[asyncio.Task] = []
        if s.sync_poll_interval_s > 0:
            background.append(asyncio.create_task(poll_forever(st.indexer, s.sync_poll_interval_s, stop)))
        if s.auto_ingest_repo:
            background.append(asyncio.create_task(_auto_ingest(app, s.auto_ingest_repo)))
        log.info(
            "api ready",
            extra={
                "llm": st.llm.name,
                "embedder": st.embedder.name,
                "mcp": s.mcp_transport,
                "langsmith": tracing,
            },
        )
        yield
        stop.set()
        for task in background:
            task.cancel()
        if saver_cm is not None:
            await saver_cm.__aexit__(None, None, None)

    app = FastAPI(title="RepoMind", version=__version__, lifespan=lifespan)

    @app.exception_handler(LLMUnavailable)
    async def _llm_unavailable(_: Request, exc: LLMUnavailable):
        return JSONResponse(status_code=503, content={"error": str(exc)})

    @app.exception_handler(LLMError)
    async def _llm_error(_: Request, exc: LLMError):
        return JSONResponse(status_code=502, content={"error": str(exc)})

    @app.exception_handler(TimeoutError)
    async def _timeout(_: Request, exc: TimeoutError):
        return JSONResponse(
            status_code=504, content={"error": "the question took too long to answer; try narrowing it"}
        )

    def _repo(request: Request, name: str) -> repo_db.RepoRecord:
        repo = repo_db.get_repo(request.app.state.pool, name)
        if repo is None:
            raise HTTPException(404, f"repo {name!r} is not indexed; POST /repos to ingest it")
        return repo

    def _start_job(app: FastAPI, kind: str, fn: Callable[[], Any], **meta: Any) -> str:
        job_id = uuid.uuid4().hex[:12]
        jobs = app.state.jobs
        jobs[job_id] = {"status": "running", "kind": kind, **meta}

        def run() -> None:
            try:
                rep = fn()
                jobs[job_id] = {"status": "done", "kind": kind, **meta, "report": rep.to_dict()}
            except Exception as e:  # surfaced through /jobs
                log.exception("job failed", extra={"kind": kind})
                jobs[job_id] = {"status": "failed", "kind": kind, **meta, "error": str(e)}

        asyncio.get_running_loop().run_in_executor(None, run)
        return job_id

    async def _auto_ingest(app: FastAPI, source: str) -> None:
        from repomind.ingestion.sources import parse_source

        src = parse_source(source, s.repos_dir)
        existing = await asyncio.to_thread(repo_db.get_repo, app.state.pool, src.name)
        if existing is not None:
            if existing.embedding_model != app.state.indexer.signature:  # new embedder / chunker version
                log.info("index signature changed; re-syncing", extra={"repo": existing.name})
                _start_job(app, "sync", lambda: app.state.indexer.sync(existing.name), repo=existing.name)
            return
        log.info("auto-ingesting", extra={"source": source})
        _start_job(app, "ingest", lambda: app.state.indexer.ingest(source), source=source)

    # ------------------------------------------------------------------ routes
    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        repos = await asyncio.to_thread(repo_db.list_repos, request.app.state.pool)
        return {"status": "ok", "version": __version__, "repos": len(repos)}

    @app.get("/tools")
    async def tools(request: Request) -> dict[str, Any]:
        """MCP servers reachable from the API and the tools they expose (discovered via list_tools)."""
        factory = toolbox_factory or (lambda: McpToolbox.from_settings(s))
        async with factory() as tb:
            specs = await tb.tools()
            return {
                "transport": s.mcp_transport,
                "servers": tb.available_servers,
                "tools": [
                    {"server": t.server, "name": t.name, "signature": t.signature()} for t in specs.values()
                ],
            }

    @app.get("/repos")
    async def list_repos(request: Request) -> list[dict]:
        repos = await asyncio.to_thread(repo_db.list_repos, request.app.state.pool)
        return [r.to_public() for r in repos]

    @app.post("/repos", status_code=202)
    async def ingest(request: Request, body: IngestRequest) -> dict:
        idx = request.app.state.indexer
        job_id = _start_job(
            request.app,
            "ingest",
            lambda: idx.ingest(body.source, ref=body.ref, name=body.name),
            source=body.source,
        )
        return {"job_id": job_id, "status": "running"}

    @app.post("/repos/{owner}/{name}/sync", status_code=202)
    async def sync_repo(request: Request, owner: str, name: str) -> dict:
        full = f"{owner}/{name}"
        _repo(request, full)
        job_id = _start_job(request.app, "sync", lambda: request.app.state.indexer.sync(full), repo=full)
        return {"job_id": job_id, "status": "running"}

    @app.get("/jobs/{job_id}")
    async def job(request: Request, job_id: str) -> dict:
        if job_id not in request.app.state.jobs:
            raise HTTPException(404, "unknown job")
        return request.app.state.jobs[job_id]

    @app.get("/search")
    async def search(request: Request, repo: str, q: str, kind: str = "code", k: int = 8) -> list[dict]:
        r = _repo(request, repo)
        kinds = ("code", "doc") if kind == "all" else (kind,)
        st = request.app.state
        hits = await asyncio.to_thread(
            hybrid_search, st.pool, st.embedder, r.id, q, k=k, kinds=kinds, settings=s
        )
        return [h.to_dict(max_chars=800) for h in hits]

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: Request, body: AskRequest) -> AskResponse:
        repo = _repo(request, body.repo)
        st = request.app.state
        with track_usage() as usage:
            if body.mode == "naive":
                resp = await answer_naive(
                    body.question, repo, llm=st.llm, pool=st.pool, embedder=st.embedder, settings=s
                )
            else:
                resp = await st.runner.ask(body, repo)
        resp.usage = usage.summary()
        return resp

    @app.post("/ask/stream")
    async def ask_stream(request: Request, body: AskRequest) -> EventSourceResponse:
        """Server-sent events: start, step, tool, token ... then final (or error)."""
        repo = _repo(request, body.repo)
        st = request.app.state

        async def events():
            with track_usage() as usage:
                try:
                    if body.mode == "naive":
                        resp = await answer_naive(
                            body.question, repo, llm=st.llm, pool=st.pool, embedder=st.embedder, settings=s
                        )
                        resp.usage = usage.summary()
                        yield {
                            "event": "final",
                            "data": json.dumps({"type": "final", "response": resp.model_dump()}),
                        }
                        return
                    async for ev in st.runner.stream(body, repo):
                        if ev["type"] == "final":
                            ev["response"]["usage"] = usage.summary()
                        yield {"event": ev["type"], "data": json.dumps(ev, default=str)}
                except LLMUnavailable as e:
                    yield {"event": "error", "data": json.dumps({"error": str(e), "status": 503})}
                except LLMError as e:
                    yield {"event": "error", "data": json.dumps({"error": str(e), "status": 502})}
                except TimeoutError:
                    yield {"event": "error", "data": json.dumps({"error": "timed out", "status": 504})}
                except Exception:
                    log.exception("stream failed")
                    yield {"event": "error", "data": json.dumps({"error": "internal error", "status": 500})}

        return EventSourceResponse(events())

    # ------------------------------------------------------------------ memories (user control)
    @app.get("/memories")
    async def list_memories(
        request: Request, user_id: str, repo: str | None = None, include_inactive: bool = False
    ) -> list[dict]:
        mems = await asyncio.to_thread(request.app.state.memory.list, user_id, repo, include_inactive)
        return [m.to_public() for m in mems]

    @app.delete("/memories/{memory_id}")
    async def delete_memory(request: Request, memory_id: int, user_id: str) -> dict:
        if not await asyncio.to_thread(request.app.state.memory.delete, user_id, memory_id):
            raise HTTPException(404, "memory not found for this user")
        return {"deleted": memory_id}

    @app.delete("/memories")
    async def delete_all_memories(request: Request, user_id: str, repo: str | None = None) -> dict:
        n = await asyncio.to_thread(request.app.state.memory.delete_all, user_id, repo)
        return {"deleted": n}

    @app.post("/webhooks/github", status_code=202)
    async def github_webhook(request: Request) -> dict:
        if not s.github_webhook_secret:
            raise HTTPException(503, "GITHUB_WEBHOOK_SECRET is not configured; webhooks are disabled")
        body = await request.body()
        if not verify_signature(s.github_webhook_secret, body, request.headers.get("X-Hub-Signature-256")):
            raise HTTPException(401, "invalid webhook signature")
        event = request.headers.get("X-GitHub-Event", "")
        if event == "ping":
            return {"status": "pong"}
        if event != "push":
            return {"status": "ignored", "reason": f"event {event!r} is not handled"}
        try:
            push = parse_push(json.loads(body))
        except (json.JSONDecodeError, AttributeError):
            raise HTTPException(400, "malformed push payload") from None
        if push is None or push.deleted:
            return {"status": "ignored", "reason": "no commits to index"}
        st = request.app.state
        repo = await asyncio.to_thread(repo_db.find_repo_by_github, st.pool, push.full_name)
        if repo is None:
            return {"status": "ignored", "reason": f"{push.full_name} is not indexed"}
        if repo.ref and push.branch != repo.ref:
            return {"status": "ignored", "reason": f"push to {push.branch!r}, tracking {repo.ref!r}"}
        job_id = _start_job(
            request.app,
            "webhook-sync",
            lambda: st.indexer.sync(repo.name, target_commit=push.after, hinted_paths=push.touched),
            repo=repo.name,
            after=push.after,
        )
        log.info(
            "webhook accepted", extra={"repo": repo.name, "after": push.after, "files": len(push.touched)}
        )
        return {"status": "accepted", "job_id": job_id, "repo": repo.name, "after": push.after}

    # ------------------------------------------------------------------ web UI
    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/ui/")

    return app
