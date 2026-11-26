"""Keep the API's memory flat on a small container (the free 512 MB demo instance).

A container sees every CPU of its host, so glibc creates up to 8 malloc arenas per host core and
asyncio's default thread pool grows to 32 threads; each thread that touches an arena keeps its freed
memory. On a small host we cap both and hand freed memory back to the OS now and then.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import os
from concurrent.futures import ThreadPoolExecutor

from badebhai.ingestion.sources import memory_limit_bytes
from badebhai.log import get_logger

log = get_logger(__name__)

_M_ARENA_MAX = -8
_SMALL_HOST_BYTES = 2 << 30


def _libc():
    if os.name != "posix":
        return None
    try:
        return ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    except OSError:
        return None


def small_host() -> bool:
    limit = memory_limit_bytes()
    return limit is not None and limit < _SMALL_HOST_BYTES


def apply(loop: asyncio.AbstractEventLoop) -> bool:
    """Cap malloc arenas and the default thread pool when running in a small container."""
    if not small_host():
        return False
    libc = _libc()
    if libc is not None and hasattr(libc, "mallopt"):
        libc.mallopt(_M_ARENA_MAX, 2)
    loop.set_default_executor(ThreadPoolExecutor(max_workers=6, thread_name_prefix="badebhai"))
    log.info(
        "small host: capped malloc arenas and worker threads", extra={"limit_mb": memory_limit_bytes() >> 20}
    )
    return True


def trim() -> None:
    """Return freed heap memory to the OS (glibc only; a no-op elsewhere)."""
    libc = _libc()
    if libc is not None and hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
