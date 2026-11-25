"""FastAPI backend."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from badebhai import __version__
from badebhai.app import cache
from badebhai.app.agent.checkpoint import postgres_checkpointer
from badebhai.app.agent.runner import AgentRunner
from badebhai.app.mcp_client import McpToolbox
from badebhai.app.memory.store import MemoryStore
from badebhai.app.naive import answer_naive
from badebhai.app.onboarding import TOKEN_URL, AddRepoError, check_source, explain_failure
from badebhai.app.router import route
from badebhai.app.schemas import AskRequest, AskResponse
from badebhai.config import Settings, configure_tracing, get_settings
from badebhai.db import feedback as feedback_db
from badebhai.db import repos as repo_db
from badebhai.db.pool import get_pool
from badebhai.embeddings import get_embedder
from badebhai.ingestion.indexer import Indexer
from badebhai.ingestion.secrets import redact
from badebhai.ingestion.sync import poll_forever
from badebhai.ingestion.webhook import parse_push, verify_signature
from badebhai.llm import LLMError, LLMProvider, LLMUnavailable, get_llm, track_usage
from badebhai.log import get_logger, setup_logging
from badebhai.mcp_servers.github_client import GitHubClient
from badebhai.retrieval.hybrid import hybrid_search

log = get_logger(__name__)
UI_DIR = Path(__file__).resolve().parent.parent / "ui"


class IngestRequest(BaseModel):
    source: str
    ref: str | None = None
    name: str | None = None


class FeedbackRequest(BaseModel):
    repo: str
    question: str = Field(min_length=1, max_length=4000)
    answer: str = Field(max_length=40000)
    rating: Literal[-1, 1]
    user_id: str = "anonymous"
    mode: str | None = None
    thread_id: str | None = None
    comment: str | None = Field(default=None, max_length=2000)
    citations: list[dict[str, Any]] = Field(default_factory=list, max_length=50)


_MAX_JOBS_KEPT = 50
_VIEW_MAX_LINES = 4000


def create_app(
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
    toolbox_factory: Callable[[], McpToolbox] | None = None,
    checkpointer=None,
    github: GitHubClient | None = None,
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
        st.github = github or GitHubClient(
            s.github_token, s.github_api_url, s.github_cache_ttl_s, timeout_s=8.0
        )
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

    app = FastAPI(title="BadeBhai", version=__version__, lifespan=lifespan)

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

    def _start_job(
        app: FastAPI,
        kind: str,
        fn: Callable[[Callable[[str, int, int], None]], Any],
        explain: Callable[[BaseException], dict] | None = None,
        **meta: Any,
    ) -> str:
        """Run `fn(progress)` in a worker thread; its state is polled through GET /jobs/{id}."""
        job_id = uuid.uuid4().hex[:12]
        jobs = app.state.jobs
        finished = [k for k, v in jobs.items() if v["status"] != "running"]
        for old in finished[: max(0, len(jobs) - _MAX_JOBS_KEPT)]:
            jobs.pop(old, None)
        base = {"id": job_id, "kind": kind, "started_at": time.time(), **meta}
        jobs[job_id] = {**base, "status": "running", "progress": {"stage": "starting", "done": 0, "total": 0}}

        def progress(stage: str, done: int, total: int) -> None:
            jobs[job_id]["progress"] = {"stage": stage, "done": done, "total": total}

        def run() -> None:
            try:
                rep = fn(progress)
                end = {"progress": jobs[job_id]["progress"], "finished_at": time.time()}
                jobs[job_id] = {**base, **end, "status": "done", "report": rep.to_dict()}
            except Exception as e:  # surfaced through /jobs
                log.exception("job failed", extra={"kind": kind})
                err = explain(e) if explain else {"error": str(e)}
                end = {"progress": jobs[job_id]["progress"], "finished_at": time.time()}
                jobs[job_id] = {**base, **end, "status": "failed", **err}

        asyncio.get_running_loop().run_in_executor(None, run)
        return job_id

    async def _auto_ingest(app: FastAPI, source: str) -> None:
        from badebhai.ingestion.sources import parse_source

        src = parse_source(source, s.repos_dir)
        existing = await asyncio.to_thread(repo_db.get_repo, app.state.pool, src.name)
        if existing is not None:
            if existing.embedding_model != app.state.indexer.signature:  # new embedder / chunker version
                log.info("index signature changed; re-syncing", extra={"repo": existing.name})
                _start_job(
                    app,
                    "sync",
                    lambda p: app.state.indexer.sync(existing.name, progress=p),
                    repo=existing.name,
                )
            return
        log.info("auto-ingesting", extra={"source": source})
        _start_job(
            app,
            "ingest",
            lambda p: app.state.indexer.ingest(source, progress=p),
            source=source,
            repo=src.name,
        )

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

    @app.get("/setup")
    async def setup() -> dict[str, Any]:
        """What the web UI needs to guide someone adding a repo. Never includes the token itself."""
        return {
            "github_token": bool(s.github_token),
            "allow_local_paths": s.api_allow_local_paths,
            "max_repo_mb": s.api_max_repo_mb,
            "token_url": TOKEN_URL,
        }

    @app.get("/repos")
    async def list_repos(request: Request) -> list[dict]:
        repos = await asyncio.to_thread(repo_db.list_repos, request.app.state.pool)
        return [r.to_public() for r in repos]

    def _running_job(app: FastAPI, repo: str) -> dict | None:
        return next(
            (j for j in app.state.jobs.values() if j["status"] == "running" and j.get("repo") == repo), None
        )

    @app.post("/repos", status_code=202)
    async def ingest(request: Request, body: IngestRequest) -> JSONResponse:
        """Add a repository: check it first (friendly errors, nothing downloaded yet), then index it
        in the background. Poll GET /jobs/{job_id} for progress."""
        st = request.app.state
        source = f"{body.source.strip()}@{body.ref}" if body.ref else body.source
        try:
            checked = await asyncio.to_thread(check_source, source, s, st.github)
        except AddRepoError as e:
            return JSONResponse(status_code=e.status, content=e.to_dict())
        name = body.name or checked.repo.name
        if job := _running_job(request.app, name):
            return JSONResponse(
                status_code=202,
                content={"job_id": job["id"], "status": "running", "repo": name, "already_running": True},
            )
        existing = await asyncio.to_thread(repo_db.get_repo, st.pool, name)
        if existing is not None and existing.last_indexed_commit and checked.repo.ref in (None, existing.ref):
            return JSONResponse(
                status_code=409,
                content={
                    "error": f"{name} is already added. You can ask about it now, or check it for new commits.",
                    "repo": name,
                    "already_indexed": True,
                    "can_update": existing.managed and name.count("/") == 1,
                },
            )
        has_token, kind = bool(s.github_token), checked.repo.kind
        job_id = _start_job(
            request.app,
            "ingest",
            lambda p: st.indexer.ingest(
                checked.source, name=body.name, progress=p, allow_local=s.api_allow_local_paths
            ),
            explain=lambda e: explain_failure(e, has_token, kind),
            source=checked.source,
            repo=name,
            about=checked.about,
            warnings=checked.warnings,
        )
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "running",
                "repo": name,
                "about": checked.about,
                "warnings": checked.warnings,
            },
        )

    @app.post("/repos/{owner}/{name}/sync", status_code=202)
    async def sync_repo(request: Request, owner: str, name: str) -> dict:
        full = f"{owner}/{name}"
        repo = _repo(request, full)
        if job := _running_job(request.app, full):
            return {"job_id": job["id"], "status": "running", "repo": full, "already_running": True}
        has_token = bool(s.github_token)
        kind = "github" if repo.github_full_name else repo.kind
        job_id = _start_job(
            request.app,
            "sync",
            lambda p: request.app.state.indexer.sync(full, progress=p),
            explain=lambda e: explain_failure(e, has_token, kind),
            repo=full,
        )
        return {"job_id": job_id, "status": "running", "repo": full}

    @app.delete("/repos/{name:path}")
    async def remove_repo(request: Request, name: str, keep_memories: bool = False) -> dict:
        """Remove a repo: its index, its downloaded copy (never a local folder you pointed at), and,
        unless keep_memories, what users told BadeBhai about it."""
        _repo(request, name)
        if _running_job(request.app, name):
            raise HTTPException(409, f"{name} is being indexed right now; try again when it finishes")
        return await asyncio.to_thread(request.app.state.indexer.remove, name, not keep_memories)

    @app.get("/file")
    async def view_file(request: Request, repo: str, path: str) -> dict:
        """An indexed file's text for the citation viewer (secrets redacted, like everything indexed).
        Only files in the index can be read: never .env, binaries, or anything outside the repo."""
        r = _repo(request, repo)
        rel = path.strip().lstrip("./")
        n_lines = await asyncio.to_thread(repo_db.file_line_count, request.app.state.pool, r.id, rel)
        if n_lines is None:
            raise HTTPException(404, f"{rel!r} is not an indexed file of {repo}")
        try:
            raw = await asyncio.to_thread(r.resolve(rel).read_text, encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            raise HTTPException(404, f"{rel!r} can no longer be read; the repo may need an update") from None
        lines = redact(raw)[0].split("\n")
        truncated = len(lines) > _VIEW_MAX_LINES
        return {
            "repo": r.name,
            "path": rel,
            "commit": r.last_indexed_commit,
            "url": r.file_url(rel),
            "line_count": n_lines,
            "truncated": truncated,
            "content": "\n".join(ln.rstrip("\r") for ln in lines[:_VIEW_MAX_LINES]),
        }

    @app.post("/feedback", status_code=201)
    async def add_feedback(request: Request, body: FeedbackRequest) -> dict:
        r = _repo(request, body.repo)
        if body.rating < 0:  # a thumbs-down answer must not be served again from the cache
            await asyncio.to_thread(cache.evict, request.app.state.pool, r.id, body.question)
        fid = await asyncio.to_thread(
            feedback_db.add,
            request.app.state.pool,
            user_id=body.user_id,
            repo=r.name,
            commit=r.last_indexed_commit,
            question=body.question,
            answer=body.answer,
            rating=body.rating,
            mode=body.mode,
            thread_id=body.thread_id,
            comment=body.comment,
            citations=body.citations,
        )
        return {"id": fid}

    @app.get("/feedback")
    async def list_feedback(
        request: Request, repo: str | None = None, rating: int | None = None, limit: int = 100
    ) -> dict:
        pool = request.app.state.pool
        rows = await asyncio.to_thread(feedback_db.list_feedback, pool, repo, rating, min(limit, 500))
        totals = await asyncio.to_thread(feedback_db.summary, pool, repo)
        return {"totals": totals, "items": rows}

    @app.get("/jobs")
    async def list_jobs(request: Request, status: str | None = None) -> list[dict]:
        return [j for j in request.app.state.jobs.values() if status is None or j["status"] == status]

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

    def _resolve_mode(body: AskRequest) -> tuple[AskRequest, dict | None]:
        if body.mode != "auto":
            return body, None
        rt = route(body.question, follow_up=bool(body.thread_id))
        return body.model_copy(update={"mode": rt.mode}), {"mode": rt.mode, "reason": rt.reason}

    async def _cache_allowed(st, body: AskRequest, repo) -> bool:
        """Only answers that cannot depend on the asker: no conversation, no memories for this repo."""
        if body.thread_id or s.answer_cache_ttl_hours <= 0 or not repo.last_indexed_commit:
            return False
        return not await asyncio.to_thread(st.memory.list, body.user_id, repo.name)

    async def _from_cache(st, req: AskRequest, repo, rt: dict | None) -> AskResponse | None:
        t0 = time.perf_counter()
        hit = await asyncio.to_thread(
            cache.lookup,
            st.pool,
            repo.id,
            repo.last_indexed_commit,
            req.mode,
            req.question,
            s.answer_cache_ttl_hours,
        )
        if hit is None:
            return None
        resp = AskResponse.model_validate(hit)
        resp.cached, resp.thread_id, resp.route = True, None, rt
        resp.latency_ms = int((time.perf_counter() - t0) * 1000)
        resp.usage = {"llm_calls": 0, "cost_usd": 0.0, "cost_complete": True, "cached": True}
        return resp

    async def _remember(st, req: AskRequest, repo, payload: dict) -> None:
        if cache.cacheable(payload):
            await asyncio.to_thread(
                cache.store, st.pool, repo.id, repo.last_indexed_commit, req.mode, req.question, payload
            )

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: Request, body: AskRequest) -> AskResponse:
        repo = _repo(request, body.repo)
        st = request.app.state
        req, rt = _resolve_mode(body)
        use_cache = await _cache_allowed(st, req, repo)
        if use_cache and (hit := await _from_cache(st, req, repo, rt)):
            return hit
        with track_usage() as usage:
            if req.mode == "naive":
                resp = await answer_naive(
                    req.question, repo, llm=st.llm, pool=st.pool, embedder=st.embedder, settings=s
                )
            else:
                resp = await st.runner.ask(req, repo)
        resp.usage = usage.summary()
        resp.route = rt
        if use_cache:
            await _remember(st, req, repo, resp.model_dump())
        return resp

    @app.post("/ask/stream")
    async def ask_stream(request: Request, body: AskRequest) -> EventSourceResponse:
        """Server-sent events: start, step, tool, token ... then final (or error)."""
        repo = _repo(request, body.repo)
        st = request.app.state

        def final(resp: AskResponse) -> dict:
            return {
                "event": "final",
                "data": json.dumps({"type": "final", "response": resp.model_dump()}, default=str),
            }

        async def events():
            req, rt = _resolve_mode(body)
            with track_usage() as usage:
                try:
                    if rt:
                        label = "fast path" if req.mode == "naive" else "agent"
                        step = {
                            "type": "step",
                            "node": "route",
                            "summary": f"{label}: {rt['reason']}",
                            "ms": 0,
                        }
                        yield {"event": "step", "data": json.dumps(step)}
                    use_cache = await _cache_allowed(st, req, repo)
                    if use_cache and (hit := await _from_cache(st, req, repo, rt)):
                        yield final(hit)
                        return
                    if req.mode == "naive":
                        resp = await answer_naive(
                            req.question, repo, llm=st.llm, pool=st.pool, embedder=st.embedder, settings=s
                        )
                        resp.usage, resp.route = usage.summary(), rt
                        if use_cache:
                            await _remember(st, req, repo, resp.model_dump())
                        yield final(resp)
                        return
                    async for ev in st.runner.stream(req, repo):
                        if ev["type"] == "final":
                            ev["response"]["usage"] = usage.summary()
                            ev["response"]["route"] = rt
                            if use_cache:
                                await _remember(st, req, repo, ev["response"])
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
            lambda p: st.indexer.sync(
                repo.name, target_commit=push.after, hinted_paths=push.touched, progress=p
            ),
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
