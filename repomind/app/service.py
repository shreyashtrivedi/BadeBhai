"""One-shot question answering outside the API (CLI, evals)."""

from __future__ import annotations

import asyncio

from repomind.app.naive import answer_naive
from repomind.app.schemas import AskResponse
from repomind.config import Settings, get_settings
from repomind.db import repos as repo_db
from repomind.db.pool import get_pool
from repomind.embeddings import get_embedder
from repomind.llm import LLMProvider, get_llm, track_usage


async def ask_once(
    repo_name: str,
    question: str,
    *,
    mode: str = "agent",
    user_id: str = "cli-user",
    thread_id: str | None = None,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
) -> AskResponse:
    s = settings or get_settings()
    pool = await asyncio.to_thread(get_pool, s)
    repo = repo_db.get_repo(pool, repo_name)
    if repo is None:
        raise KeyError(f"repo {repo_name!r} is not indexed")
    llm = llm or get_llm(s)
    embedder = get_embedder(s)
    with track_usage() as usage:
        if mode == "naive":
            resp = await answer_naive(question, repo, llm=llm, pool=pool, embedder=embedder, settings=s)
        else:
            raise NotImplementedError("agent mode arrives in Stage 2")
    resp.usage = usage.summary()
    return resp
