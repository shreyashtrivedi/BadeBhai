"""LangGraph Postgres checkpointer (short-term, per-thread memory)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from badebhai.db.pool import ensure_database


@asynccontextmanager
async def postgres_checkpointer(database_url: str) -> AsyncIterator[AsyncPostgresSaver]:
    """Yields an AsyncPostgresSaver backed by its own async pool (needs a selector loop on Windows)."""
    ensure_database(database_url)
    pool = AsyncConnectionPool(
        database_url,
        min_size=1,
        max_size=10,
        # hosted Postgres (Neon) drops idle connections: check before use, retire idle ones (see db.pool)
        check=AsyncConnectionPool.check_connection,
        max_idle=120,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    try:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        yield saver
    finally:
        await pool.close()
