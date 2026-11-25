"""One-shot question answering outside the API (CLI, evals)."""

from __future__ import annotations

import asyncio

from badebhai.app.agent.checkpoint import postgres_checkpointer
from badebhai.app.agent.runner import AgentRunner
from badebhai.app.naive import answer_naive
from badebhai.app.schemas import AskRequest, AskResponse
from badebhai.config import Settings, get_settings
from badebhai.db import repos as repo_db
from badebhai.db.pool import get_pool
from badebhai.embeddings import get_embedder
from badebhai.llm import LLMProvider, get_llm, track_usage


async def ask_once(
    repo_name: str,
    question: str,
    *,
    mode: str = "agent",
    user_id: str = "cli-user",
    thread_id: str | None = None,
    settings: Settings | None = None,
    llm: LLMProvider | None = None,
    runner: AgentRunner | None = None,
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
            req = AskRequest(question=question, repo=repo_name, user_id=user_id, thread_id=thread_id)
            if runner is not None:
                resp = await runner.ask(req, repo)
            else:  # persistent threads across CLI invocations
                async with postgres_checkpointer(s.database_url) as saver:
                    resp = await AgentRunner(s, llm, pool, embedder, checkpointer=saver).ask(req, repo)
    resp.usage = usage.summary()
    return resp
