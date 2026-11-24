"""FastAPI backend."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from repomind import __version__
from repomind.app.naive import answer_naive
from repomind.app.schemas import AskRequest, AskResponse
from repomind.config import Settings, get_settings
from repomind.db import repos as repo_db
from repomind.db.pool import get_pool
from repomind.embeddings import get_embedder
from repomind.ingestion.indexer import Indexer
from repomind.llm import LLMError, LLMProvider, LLMUnavailable, get_llm, track_usage
from repomind.log import get_logger, setup_logging
from repomind.retrieval.hybrid import hybrid_search

log = get_logger(__name__)


class IngestRequest(BaseModel):
    source: str
    ref: str | None = None
    name: str | None = None


def create_app(settings: Settings | None = None, llm: LLMProvider | None = None) -> FastAPI:
    s = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(s.log_level, s.log_json)
        app.state.settings = s
        app.state.pool = await asyncio.to_thread(get_pool, s)
        app.state.embedder = get_embedder(s)
        app.state.llm = llm or get_llm(s)
        app.state.jobs = {}
        log.info("api ready", extra={"llm": app.state.llm.name, "embedder": app.state.embedder.name})
        yield

    app = FastAPI(title="RepoMind", version=__version__, lifespan=lifespan)

    @app.exception_handler(LLMUnavailable)
    async def _llm_unavailable(_: Request, exc: LLMUnavailable):
        return JSONResponse(status_code=503, content={"error": str(exc)})

    @app.exception_handler(LLMError)
    async def _llm_error(_: Request, exc: LLMError):
        return JSONResponse(status_code=502, content={"error": str(exc)})

    def _repo(request: Request, name: str) -> repo_db.RepoRecord:
        repo = repo_db.get_repo(request.app.state.pool, name)
        if repo is None:
            raise HTTPException(404, f"repo {name!r} is not indexed; POST /repos to ingest it")
        return repo

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        repos = await asyncio.to_thread(repo_db.list_repos, request.app.state.pool)
        return {"status": "ok", "version": __version__, "repos": len(repos)}

    @app.get("/repos")
    async def list_repos(request: Request) -> list[dict]:
        repos = await asyncio.to_thread(repo_db.list_repos, request.app.state.pool)
        return [r.to_public() for r in repos]

    @app.post("/repos", status_code=202)
    async def ingest(request: Request, body: IngestRequest) -> dict:
        job_id = uuid.uuid4().hex[:12]
        jobs = request.app.state.jobs
        jobs[job_id] = {"status": "running", "source": body.source}

        def run() -> None:
            try:
                rep = Indexer(s, request.app.state.pool, request.app.state.embedder).ingest(
                    body.source, ref=body.ref, name=body.name
                )
                jobs[job_id] = {"status": "done", "report": rep.to_dict()}
            except Exception as e:  # reported through /jobs
                log.exception("ingest failed")
                jobs[job_id] = {"status": "failed", "error": str(e)}

        asyncio.get_running_loop().run_in_executor(None, run)
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
        hits = await asyncio.to_thread(
            hybrid_search,
            request.app.state.pool,
            request.app.state.embedder,
            r.id,
            q,
            k=k,
            kinds=kinds,
            settings=s,
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
                raise HTTPException(501, "agent mode arrives in Stage 2; use mode=naive")
        resp.usage = usage.summary()
        return resp

    return app
