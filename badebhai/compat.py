"""Platform shims.

psycopg's async mode (used by the LangGraph Postgres checkpointer) cannot run on Windows'
default ProactorEventLoop, so on Windows every BadeBhai event loop is a SelectorEventLoop.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def new_event_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == "win32":
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


def run(coro: Coroutine[Any, Any, T]) -> T:
    with asyncio.Runner(loop_factory=new_event_loop) as runner:
        return runner.run(coro)


UVICORN_LOOP = "badebhai.compat:new_event_loop"
