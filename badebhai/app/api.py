"""FastAPI backend."""

from __future__ import annotations

import asyncio
import json
import secrets
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from badebhai import __version__
from badebhai.app import auth, cache
from badebhai.app.agent.checkpoint import postgres_checkpointer
from badebhai.app.agent.runner import AgentRunner
from badebhai.app.compare import compare as compare_repos
from badebhai.app.mcp_client import McpToolbox
from badebhai.app.memory.store import MemoryStore
from badebhai.app.naive import answer_naive
from badebhai.app.onboarding import TOKEN_URL, AddRepoError, check_source, explain_failure
from badebhai.app.router import route
from badebhai.app.schemas import AskRequest, AskResponse, CompareRequest
from badebhai.config import Settings, configure_tracing, get_settings
from badebhai.db import feedback as feedback_db
from badebhai.db import repos as repo_db
from badebhai.db.pool import get_pool
from badebhai.embeddings import get_embedder
from badebhai.ingestion.indexer import Indexer
from badebhai.ingestion.secrets import redact
from badebhai.ingestion.sync import poll_forever
from badebhai.ingestion.webhook import parse_push, verify_signature
from badebhai.integrations import github_bot, slack
from badebhai.integrations.formatting import github_markdown, slack_mrkdwn
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


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class ShareRequest(BaseModel):
    shared: bool


_PUBLIC_PATHS = ("/health", "/auth/login", "/auth/logout", "/auth/me", "/webhooks/", "/integrations/")


def _is_public(path: str) -> bool:
    return (
        path == "/"
        or path.startswith("/ui")
        or any(path == p or (p.endswith("/") and path.startswith(p)) for p in _PUBLIC_PATHS)
    )


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
    http_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(s.log_level, s.log_json)
        tracing = configure_tracing(s)
        st = app.state
        st.settings = s
        st.pool = await asyncio.to_thread(get_pool, s)
        st.embedder = get_embedder(s)  # loaded in the background once clones are restored (_startup_repos)
        st.llm = llm or get_llm(s)
        st.jobs = {}
        st.started_at = time.time()
        st.limiter = auth.DbUsageLimiter(st.pool) if s.auth_enabled else auth.UsageLimiter()
        st.http = httpx.AsyncClient(timeout=15.0, transport=http_transport)
        st.bot_tasks = set()
        st.session_secret = s.session_secret or secrets.token_urlsafe(32)
        if s.auth_enabled:
            if not s.session_secret:
                log.warning("SESSION_SECRET is not set: sign-ins last until the API restarts")
            if await asyncio.to_thread(auth.count_users, st.pool) == 0:
                log.warning("AUTH_ENABLED but no users yet: run `badebhai users add <name> --admin`")
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
        background.append(asyncio.create_task(_resume_bot_replies(app)))
        sources = [x.strip() for x in (s.auto_ingest_repo or "").split(",") if x.strip()]
        background.append(asyncio.create_task(_startup_repos(app, sources)))
        if s.sync_poll_interval_s > 0:
            background.append(asyncio.create_task(poll_forever(st.indexer, s.sync_poll_interval_s, stop)))
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
        await st.http.aclose()
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

    @app.middleware("http")
    async def require_sign_in(request: Request, call_next):
        request.state.user = None
        if not s.auth_enabled:
            return await call_next(request)
        request.state.user = await asyncio.to_thread(_identify, request)
        if request.state.user is None and not _is_public(request.url.path):
            return JSONResponse(status_code=401, content={"error": "Please sign in.", "auth_required": True})
        return await call_next(request)

    def _identify(request: Request) -> auth.User | None:
        pool = request.app.state.pool
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            return auth.user_by_api_key(pool, header[7:].strip())
        name = auth.read_session(
            request.cookies.get(auth.SESSION_COOKIE),
            request.app.state.session_secret,
            s.session_max_age_hours * 3600,
        )
        return auth.get_user(pool, name) if name else None

    def _user(request: Request) -> auth.User | None:
        return getattr(request.state, "user", None)

    def _uid(request: Request, supplied: str | None) -> str:
        """With accounts, memories and feedback always belong to the signed-in user."""
        user = _user(request)
        return user.username if user else (supplied or "anonymous")

    def _limit(request: Request, action: str, limit: int, window_s: int, what: str) -> None:
        user = _user(request)
        if user is None or user.is_admin:
            return
        wait = request.app.state.limiter.check(user.username, action, limit, window_s)
        if wait is not None:
            mins = max(1, round(wait / 60))
            raise HTTPException(
                429, f"You've reached the limit of {limit} {what}. Try again in about {mins} min."
            )

    def _demo_readonly() -> None:
        if s.demo_mode:
            raise HTTPException(
                403,
                "This is a public demo with fixed repositories. Run your own BadeBhai (free) to add any repo: "
                + s.demo_source_url,
            )

    def _visitor(request: Request) -> str:
        # Behind a hosting proxy every request comes from the proxy; the first X-Forwarded-For hop is the visitor.
        fwd = request.headers.get("x-forwarded-for", "")
        return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")

    def _demo_limit(request: Request) -> None:
        if not s.demo_mode:
            return
        lim = request.app.state.limiter
        # The visitor's own limit first, so their rejected requests do not use up everyone's daily quota.
        wait = lim.check(_visitor(request), "demo-hour", s.demo_asks_per_hour, 3600)
        if wait is not None:
            raise HTTPException(
                429,
                f"Demo limit: {s.demo_asks_per_hour} questions per hour per visitor. Try again in about "
                f"{max(1, round(wait / 60))} min, or run your own copy (free): {s.demo_source_url}",
            )
        if lim.check("*", "demo-day", s.demo_asks_per_day, 86400) is not None:
            raise HTTPException(
                429,
                "The demo has answered its daily quota of questions. Try again tomorrow, or run your own copy.",
            )

    def _can_see(request: Request, repo: repo_db.RepoRecord) -> bool:
        user = _user(request)
        if user is None or user.is_admin or repo.shared:
            return True
        return repo_db.is_member(request.app.state.pool, repo.id, user.username)

    def _repo(request: Request, name: str) -> repo_db.RepoRecord:
        repo = repo_db.get_repo(request.app.state.pool, name)
        if repo is None or not _can_see(request, repo):
            raise HTTPException(404, f"repo {name!r} is not indexed; POST /repos to ingest it")
        return repo

    @app.exception_handler(HTTPException)
    async def _http_error(_: Request, exc: HTTPException):  # one error shape for the UI: {"error": ...}
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "detail": exc.detail})

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

    async def _restore_clones(app: FastAPI) -> None:
        """Indexed repos whose working copy vanished (fresh container): clone them back, no re-index."""
        for repo in await asyncio.to_thread(repo_db.list_repos, app.state.pool):
            try:
                await asyncio.to_thread(app.state.indexer.restore_clone, repo)
            except Exception:
                log.exception("could not restore clone", extra={"repo": repo.name})

    async def _startup_repos(app: FastAPI, sources: list[str]) -> None:
        """Restore missing clones, then index the auto-ingest repos one at a time: concurrent clones and
        embedding batches trip over each other and spike memory on a small instance."""
        await _restore_clones(app)
        # Load the embedding model now, not during the first question, but after the clones: git and
        # the model together can exceed a 512 MB instance.
        try:
            await asyncio.to_thread(app.state.embedder.embed_query, "warm up")
        except Exception:
            log.exception("could not load the embedding model")
        for source in sources:
            try:
                job_id = await _auto_ingest(app, source)
            except Exception:
                log.exception("auto-ingest failed", extra={"source": source})
                continue
            while job_id and app.state.jobs.get(job_id, {}).get("status") == "running":  # noqa: ASYNC110 (job runs in a thread)
                await asyncio.sleep(2)

    async def _auto_ingest(app: FastAPI, source: str) -> str | None:
        from badebhai.ingestion.sources import parse_source

        src = parse_source(source, s.repos_dir)
        existing = await asyncio.to_thread(repo_db.get_repo, app.state.pool, src.name)
        if existing is not None and existing.last_indexed_commit:
            if existing.embedding_model != app.state.indexer.signature:  # new embedder / chunker version
                log.info("index signature changed; re-syncing", extra={"repo": existing.name})
                return _start_job(
                    app,
                    "sync",
                    lambda p: app.state.indexer.sync(existing.name, progress=p),
                    repo=existing.name,
                )
            return None
        # new, or an earlier indexing run never finished: (re)index it in full
        log.info("auto-ingesting", extra={"source": source})
        return _start_job(
            app,
            "ingest",
            lambda p: app.state.indexer.ingest(source, progress=p),
            source=source,
            repo=src.name,
        )

    # ------------------------------------------------------------------ routes
    @app.get("/health")
    async def health(request: Request, deep: bool = False) -> dict[str, Any]:
        """Liveness, answered without the database so a busy small instance still passes health checks
        (Render restarts a service whose health check times out). ?deep=1 also counts repos."""
        out: dict[str, Any] = {
            "status": "ok",
            "version": __version__,
            "uptime_s": int(time.time() - request.app.state.started_at),
        }
        if deep:
            out["repos"] = len(await asyncio.to_thread(repo_db.list_repos, request.app.state.pool))
        return out

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

    @app.post("/auth/login")
    async def login(request: Request, body: LoginRequest) -> JSONResponse:
        ip = request.client.host if request.client else "?"
        wait = request.app.state.limiter.check(f"{body.username}@{ip}", "login", 10, 900)
        if wait is not None:
            raise HTTPException(429, "Too many sign-in attempts. Wait a few minutes and try again.")
        user = await asyncio.to_thread(
            auth.authenticate, request.app.state.pool, body.username, body.password
        )
        if user is None:
            raise HTTPException(401, "Wrong username or password.")
        resp = JSONResponse({"user": user.to_public()})
        resp.set_cookie(
            auth.SESSION_COOKIE,
            auth.sign_session(user.username, request.app.state.session_secret),
            max_age=s.session_max_age_hours * 3600,
            httponly=True,
            samesite="lax",
            secure=s.cookie_secure,
        )
        return resp

    @app.post("/auth/logout")
    async def logout() -> JSONResponse:
        resp = JSONResponse({"signed_out": True})
        resp.delete_cookie(auth.SESSION_COOKIE)
        return resp

    @app.get("/auth/me")
    async def me(request: Request) -> dict:
        user = _user(request)
        return {"auth_enabled": s.auth_enabled, "user": user.to_public() if user else None}

    @app.get("/setup")
    async def setup() -> dict[str, Any]:
        """What the web UI needs to guide someone adding a repo. Never includes the token itself."""
        return {
            "github_token": bool(s.github_token),
            "allow_local_paths": s.api_allow_local_paths,
            "max_repo_mb": s.api_max_repo_mb,
            "token_url": TOKEN_URL,
            "demo_mode": s.demo_mode,
            "source_url": s.demo_source_url,
            "model": s.main_model,
        }

    @app.get("/repos")
    async def list_repos(request: Request) -> list[dict]:
        user, pool = _user(request), request.app.state.pool
        if user is None or user.is_admin:
            repos = await asyncio.to_thread(repo_db.list_repos, pool)
        else:
            repos = await asyncio.to_thread(repo_db.list_visible_repos, pool, user.username)
        return [r.to_public() for r in repos]

    @app.patch("/repos/{name:path}")
    async def share_repo(request: Request, name: str, body: ShareRequest) -> dict:
        """Admins: make a repo visible to everyone (shared) or only to its members."""
        _demo_readonly()
        user = _user(request)
        if user is not None and not user.is_admin:
            raise HTTPException(403, "Only an admin can change who sees a repository.")
        repo = _repo(request, name)
        await asyncio.to_thread(repo_db.set_shared, request.app.state.pool, repo.id, body.shared)
        return {"repo": repo.name, "shared": body.shared}

    def _running_job(app: FastAPI, repo: str) -> dict | None:
        return next(
            (j for j in app.state.jobs.values() if j["status"] == "running" and j.get("repo") == repo), None
        )

    @app.post("/repos", status_code=202)
    async def ingest(request: Request, body: IngestRequest) -> JSONResponse:
        """Add a repository: check it first (friendly errors, nothing downloaded yet), then index it
        in the background. Poll GET /jobs/{job_id} for progress."""
        _demo_readonly()
        st = request.app.state
        source = f"{body.source.strip()}@{body.ref}" if body.ref else body.source
        try:
            checked = await asyncio.to_thread(check_source, source, s, st.github)
        except AddRepoError as e:
            return JSONResponse(status_code=e.status, content=e.to_dict())
        name = body.name or checked.repo.name
        user = _user(request)
        owner = user.username if user and not user.is_admin else None
        if job := _running_job(request.app, name):
            return JSONResponse(
                status_code=202,
                content={"job_id": job["id"], "status": "running", "repo": name, "already_running": True},
            )
        existing = await asyncio.to_thread(repo_db.get_repo, st.pool, name)
        if existing is not None and owner and not await asyncio.to_thread(_can_see, request, existing):
            await asyncio.to_thread(
                repo_db.add_member, st.pool, existing.id, owner
            )  # already indexed: share it
            return JSONResponse(
                status_code=200,
                content={
                    "repo": name,
                    "status": "done",
                    "added_to_your_list": True,
                    "message": f"{name} was already indexed, so it was added to your list right away.",
                },
            )
        if owner:
            _limit(request, "add", s.repo_adds_per_day, 86400, "repositories added per day")
            if await asyncio.to_thread(repo_db.count_owned, st.pool, owner) >= s.max_private_repos_per_user:
                raise HTTPException(
                    403,
                    f"You can have up to {s.max_private_repos_per_user} repositories. Remove one first.",
                )
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
                checked.source, name=body.name, progress=p, allow_local=s.api_allow_local_paths, owner=owner
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
        _demo_readonly()
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
        _demo_readonly()
        repo = _repo(request, name)
        if _running_job(request.app, name):
            raise HTTPException(409, f"{name} is being indexed right now; try again when it finishes")
        user = _user(request)
        if user is not None and not user.is_admin:
            if repo.shared:
                raise HTTPException(403, "Only an admin can remove a repository that everyone shares.")
            left = await asyncio.to_thread(
                repo_db.remove_member, request.app.state.pool, repo.id, user.username
            )
            if left:
                return {"repo": name, "removed_from_your_list": True}
        return await asyncio.to_thread(request.app.state.indexer.remove, name, not keep_memories)

    @app.get("/overview")
    async def overview(request: Request, repo: str, path_prefix: str = "") -> dict:
        """The repository map (directories, main symbols, entry points), as the agent's repo_overview."""
        from badebhai.retrieval.overview import build_overview

        r = _repo(request, repo)
        data = await asyncio.to_thread(build_overview, request.app.state.pool, r, path_prefix)
        return {"repo": r.name, "commit": r.last_indexed_commit, "map": data["items"][0]["content"]}

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
            user_id=_uid(request, body.user_id),
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
        user = _user(request)
        if (user is not None and not user.is_admin) or (s.demo_mode and user is None):
            raise HTTPException(403, "Only an admin can read everyone's feedback.")
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

    def _ask_guard(request: Request, body: AskRequest) -> AskRequest:
        _demo_limit(request)
        _limit(request, "ask", s.asks_per_hour, 3600, "questions per hour")
        return body.model_copy(update={"user_id": _uid(request, body.user_id)})

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: Request, body: AskRequest) -> AskResponse:
        body = _ask_guard(request, body)
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

    def _compare_setup(request: Request, body: CompareRequest):
        _demo_limit(request)
        _limit(request, "ask", s.asks_per_hour, 3600, "questions per hour")
        names = list(dict.fromkeys(body.repos))
        if len(names) < 2:
            raise HTTPException(422, "Pick two different repositories to compare.")
        repos = [_repo(request, n) for n in names]
        st, uid = request.app.state, _uid(request, body.user_id)

        async def answer_one(repo) -> AskResponse:
            req, _ = _resolve_mode(
                AskRequest(question=body.question, repo=repo.name, mode=body.mode, user_id=uid)
            )
            if req.mode == "naive":
                return await answer_naive(
                    req.question, repo, llm=st.llm, pool=st.pool, embedder=st.embedder, settings=s
                )
            return await st.runner.ask(req, repo)

        return repos, answer_one

    @app.post("/compare", response_model=AskResponse)
    async def compare(request: Request, body: CompareRequest) -> AskResponse:
        """Answer one question for 2-3 repos, then compare them with citations into each."""
        repos, answer_one = _compare_setup(request, body)
        with track_usage() as usage:
            resp = await compare_repos(body.question, repos, answer_one, request.app.state.llm)
        resp.usage = usage.summary()
        return resp

    @app.post("/compare/stream")
    async def compare_stream(request: Request, body: CompareRequest) -> EventSourceResponse:
        """/compare as server-sent events: a `step` per repo researched, then `final` (or `error`)."""
        repos, answer_one = _compare_setup(request, body)
        queue: asyncio.Queue = asyncio.Queue()

        async def run() -> None:
            with track_usage() as usage:
                try:
                    resp = await compare_repos(
                        body.question, repos, answer_one, request.app.state.llm, queue.put
                    )
                    resp.usage = usage.summary()
                    await queue.put({"type": "final", "response": resp.model_dump()})
                except LLMUnavailable as e:
                    await queue.put({"type": "error", "error": str(e), "status": 503})
                except LLMError as e:
                    await queue.put({"type": "error", "error": str(e), "status": 502})
                except TimeoutError:
                    await queue.put({"type": "error", "error": "timed out", "status": 504})
                except Exception:
                    log.exception("compare stream failed")
                    await queue.put({"type": "error", "error": "internal error", "status": 500})

        async def events():
            task = asyncio.create_task(run())
            try:
                while True:
                    ev = await queue.get()
                    yield {"event": ev["type"], "data": json.dumps(ev, default=str)}
                    if ev["type"] in ("final", "error"):
                        return
            finally:
                if not task.done():
                    task.cancel()  # the client went away

        return EventSourceResponse(events())

    @app.post("/ask/stream")
    async def ask_stream(request: Request, body: AskRequest) -> EventSourceResponse:
        """Server-sent events: start, step, tool, token ... then final (or error)."""
        body = _ask_guard(request, body)
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
        request: Request, user_id: str = "anonymous", repo: str | None = None, include_inactive: bool = False
    ) -> list[dict]:
        user_id = _uid(request, user_id)
        mems = await asyncio.to_thread(request.app.state.memory.list, user_id, repo, include_inactive)
        return [m.to_public() for m in mems]

    @app.delete("/memories/{memory_id}")
    async def delete_memory(request: Request, memory_id: int, user_id: str = "anonymous") -> dict:
        user_id = _uid(request, user_id)
        if not await asyncio.to_thread(request.app.state.memory.delete, user_id, memory_id):
            raise HTTPException(404, "memory not found for this user")
        return {"deleted": memory_id}

    @app.delete("/memories")
    async def delete_all_memories(
        request: Request, user_id: str = "anonymous", repo: str | None = None
    ) -> dict:
        n = await asyncio.to_thread(request.app.state.memory.delete_all, _uid(request, user_id), repo)
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
        if event == "issue_comment":
            try:
                return await _github_mention(request.app, json.loads(body))
            except (json.JSONDecodeError, AttributeError):
                raise HTTPException(400, "malformed issue_comment payload") from None
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

    # ------------------------------------------------------------------ bots (GitHub, Slack)
    async def _answer_for_bot(
        app: FastAPI, question: str, repo, user_id: str, context: str = ""
    ) -> AskResponse:
        """Auto mode decides on the person's own words; `context` (e.g. the issue) is only passed along."""
        st = app.state
        req, rt = _resolve_mode(AskRequest(question=question, repo=repo.name, user_id=user_id))
        req = req.model_copy(update={"question": question + context})
        with track_usage() as usage:
            if req.mode == "naive":
                resp = await answer_naive(
                    req.question, repo, llm=st.llm, pool=st.pool, embedder=st.embedder, settings=s
                )
            else:
                resp = await st.runner.ask(req, repo)
        resp.usage, resp.route = usage.summary(), rt
        return resp

    def _background(app: FastAPI, coro) -> None:
        task = asyncio.create_task(coro)
        app.state.bot_tasks.add(task)  # keep a reference until it finishes
        task.add_done_callback(app.state.bot_tasks.discard)

    def _bot_db(sql: str, params: tuple) -> Any:
        with app.state.pool.connection() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchone() if cur.description else None

    async def _queue_reply(app: FastAPI, kind: str, payload: dict) -> None:
        """Record the reply before answering, so a restart mid-answer resumes it instead of losing it."""
        row = await asyncio.to_thread(
            _bot_db,
            "INSERT INTO bot_replies (kind, payload) VALUES (%s, %s) RETURNING id",
            (kind, Jsonb(payload)),
        )
        _background(app, _run_reply(app, row[0], kind, payload))

    async def _run_reply(app: FastAPI, reply_id: int, kind: str, payload: dict) -> None:
        await asyncio.to_thread(
            _bot_db,
            "UPDATE bot_replies SET attempts = attempts + 1, updated_at = now() WHERE id = %s",
            (reply_id,),
        )
        posted = await (_github_reply(app, payload) if kind == "github" else _slack_reply(app, payload))
        await asyncio.to_thread(
            _bot_db,
            "UPDATE bot_replies SET status = %s, updated_at = now() WHERE id = %s",
            ("done" if posted else "failed", reply_id),
        )

    async def _resume_bot_replies(app: FastAPI) -> None:
        """Pending replies from before a restart: GitHub within a day, Slack within its 30-minute window."""
        try:
            rows = await asyncio.to_thread(_pending_replies)
        except Exception:
            log.exception("could not load pending bot replies")
            return
        for reply_id, kind, payload in rows:
            log.info("resuming bot reply", extra={"id": reply_id, "kind": kind})
            _background(app, _run_reply(app, reply_id, kind, payload))

    def _pending_replies() -> list[tuple]:
        with app.state.pool.connection() as conn:
            conn.execute(
                "UPDATE bot_replies SET status = 'failed' WHERE status = 'pending' AND (attempts >= 3 OR "
                "(kind = 'slack' AND created_at < now() - interval '25 minutes') OR created_at < now() - interval '1 day')"
            )
            return conn.execute(
                "SELECT id, kind, payload FROM bot_replies WHERE status = 'pending' ORDER BY id"
            ).fetchall()

    async def _github_reply(app: FastAPI, p: dict) -> bool:
        repo = await asyncio.to_thread(repo_db.get_repo, app.state.pool, p["repo"])
        try:
            if repo is None:
                raise LookupError(f"{p['repo']} is no longer indexed")
            resp = await _answer_for_bot(
                app, p["question"], repo, f"github:{p['author']}", p.get("context", "")
            )
            text = github_markdown(resp, p["question"])
        except Exception as e:
            log.exception("github bot answer failed", extra={"repo": p["repo"], "issue": p["number"]})
            text = f"Sorry @{p['author']}, I couldn't answer that ({type(e).__name__}).\n\n<sub>🤖 BadeBhai</sub>"
        try:
            await asyncio.to_thread(
                app.state.github.post,
                f"/repos/{p['full_name']}/issues/{p['number']}/comments",
                {"body": text},
            )
            return True
        except Exception:
            log.exception("github bot could not post", extra={"repo": p["repo"], "issue": p["number"]})
            return False

    async def _slack_reply(app: FastAPI, p: dict) -> bool:
        repo = await asyncio.to_thread(repo_db.get_repo, app.state.pool, p["repo"])
        try:
            if repo is None:
                raise LookupError(f"{p['repo']} is no longer indexed")
            resp = await _answer_for_bot(app, p["question"], repo, f"slack:{p['user_id']}")
            text = slack_mrkdwn(resp, repo.name)
        except Exception as e:
            log.exception("slack answer failed", extra={"repo": p["repo"]})
            text = f"Sorry, I couldn't answer that ({type(e).__name__})."
        try:
            r = await app.state.http.post(
                p["response_url"], json={"response_type": "in_channel", "text": text}
            )
            return r.status_code < 400
        except httpx.HTTPError:
            log.exception("slack reply failed")
            return False

    async def _github_mention(app: FastAPI, payload: dict) -> dict:
        if not s.github_bot_enabled:
            return {"status": "ignored", "reason": "GITHUB_BOT_ENABLED is off"}
        q, why = github_bot.parse_mention(payload, s)
        if q is None:
            return {"status": "ignored", "reason": why}
        repo = await asyncio.to_thread(repo_db.find_repo_by_github, app.state.pool, q.full_name)
        if repo is None or not repo.last_indexed_commit:
            return {"status": "ignored", "reason": f"{q.full_name} is not indexed"}
        if not s.github_token:
            return {"status": "ignored", "reason": "GITHUB_TOKEN is needed to post replies"}

        context = (
            f" (asked on {'pull request' if q.is_pull_request else 'issue'} #{q.number}: {q.issue_title})"
        )
        payload = {
            "repo": repo.name,
            "full_name": q.full_name,
            "number": q.number,
            "question": q.question,
            "author": q.author,
            "context": context,
        }
        await _queue_reply(app, "github", payload)
        log.info("github bot accepted", extra={"repo": repo.name, "issue": q.number, "author": q.author})
        return {"status": "accepted", "repo": repo.name, "issue": q.number}

    @app.post("/integrations/slack/command")
    async def slack_command(request: Request) -> dict:
        """Slack slash command `/badebhai [owner/repo] <question>` (see integrations/slack.py)."""
        if not s.slack_signing_secret:
            raise HTTPException(503, "SLACK_SIGNING_SECRET is not configured; the Slack command is disabled")
        body = await request.body()
        ok = slack.verify(
            s.slack_signing_secret,
            request.headers.get("X-Slack-Request-Timestamp"),
            body,
            request.headers.get("X-Slack-Signature"),
        )
        if not ok:
            raise HTTPException(401, "invalid Slack signature")
        cmd = slack.parse_command(body)
        if cmd is None:
            raise HTTPException(400, "malformed Slack command")
        st = request.app.state
        repos = [r for r in await asyncio.to_thread(repo_db.list_repos, st.pool) if r.last_indexed_commit]
        if s.auth_enabled:  # Slack users are not BadeBhai accounts: shared repos only
            repos = [r for r in repos if r.shared]
        names = [r.name for r in repos]
        if not cmd.text:
            return {
                "response_type": "ephemeral",
                "text": "Usage: `/badebhai [owner/repo] your question`. Repos: " + ", ".join(names),
            }
        name, question = slack.split_repo(cmd.text, names, s.slack_default_repo)
        if name is None:
            return {
                "response_type": "ephemeral",
                "text": "Which repository? Start with one of: " + ", ".join(names),
            }
        repo = next(r for r in repos if r.name == name)
        payload = {
            "repo": repo.name,
            "question": question,
            "user_id": cmd.user_id,
            "response_url": cmd.response_url,
        }
        await _queue_reply(request.app, "slack", payload)
        return {
            "response_type": "ephemeral",
            "text": f"Looking into it in *{repo.name}*… (usually a minute or two)",
        }

    # ------------------------------------------------------------------ web UI
    if UI_DIR.exists():
        app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")

        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse("/ui/")

    return app
