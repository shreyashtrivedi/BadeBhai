"""Postgres connection pool and idempotent schema bootstrap."""

from __future__ import annotations

import threading
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import ConnectionPool

from repomind.config import Settings, get_settings
from repomind.log import get_logger

log = get_logger(__name__)

_SCHEMA = Path(__file__).with_name("schema.sql")
_pools: dict[str, ConnectionPool] = {}
_lock = threading.Lock()


class SchemaMismatch(RuntimeError):
    pass


def ensure_database(database_url: str) -> None:
    """Create the target database if it does not exist yet (connects to the `postgres` maintenance DB)."""
    try:
        psycopg.connect(database_url, connect_timeout=10).close()
        return
    except psycopg.OperationalError as e:
        if "does not exist" not in str(e):
            raise
    params = conninfo_to_dict(database_url)
    name = params.get("dbname")
    admin = make_conninfo(database_url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        try:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            log.info("created database", extra={"database": name})
        except psycopg.errors.DuplicateDatabase:
            pass


def ensure_schema(database_url: str, dim: int) -> None:
    ensure_database(database_url)
    sql_text = _SCHEMA.read_text(encoding="utf-8").replace("{dim}", str(int(dim)))
    with psycopg.connect(database_url, autocommit=True) as conn:
        # Serialize concurrent bootstraps (api + 3 MCP servers start together).
        conn.execute("SELECT pg_advisory_lock(727001)")
        try:
            conn.execute(sql_text)
            row = conn.execute(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
            ).fetchone()
            if row and row[0] not in (-1, dim):
                raise SchemaMismatch(
                    f"chunks.embedding has dimension {row[0]} but EMBEDDING_DIM={dim}. "
                    "Run `repomind reset-db` (drops indexed data) or set EMBEDDING_DIM to match."
                )
        finally:
            conn.execute("SELECT pg_advisory_unlock(727001)")


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    try:
        # pgvector >= 0.8: keep scanning the HNSW index until the repo filter is satisfied.
        conn.execute("SET hnsw.iterative_scan = relaxed_order")
    except psycopg.Error:
        conn.rollback()
    conn.commit()


def get_pool(settings: Settings | None = None) -> ConnectionPool:
    s = settings or get_settings()
    with _lock:
        pool = _pools.get(s.database_url)
        if pool is None:
            ensure_schema(s.database_url, s.embedding_dim)
            pool = ConnectionPool(
                s.database_url, min_size=1, max_size=10, configure=_configure, open=True, timeout=30
            )
            _pools[s.database_url] = pool
            log.info("db pool ready")
        return pool


def close_pools() -> None:
    with _lock:
        for pool in _pools.values():
            pool.close()
        _pools.clear()


def reset_database(settings: Settings | None = None) -> None:
    """Drop all RepoMind tables (used by `repomind reset-db` and tests)."""
    s = settings or get_settings()
    close_pools()
    with psycopg.connect(s.database_url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, files, memories, repos, schema_meta CASCADE")
