"""Structured logging. JSON lines by default; key=value text when LOG_JSON=false."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extras = " ".join(
            f"{k}={v}" for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        )
        base = f"{record.levelname.lower():7s} {record.name}: {record.getMessage()}"
        out = f"{base} {extras}".rstrip()
        if record.exc_info:
            out += "\n" + self.formatException(record.exc_info)
        return out


_configured = False


def setup_logging(level: str = "INFO", json_lines: bool = True) -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)  # stderr: stdout is reserved for MCP stdio transport
    handler.setFormatter(JsonFormatter() if json_lines else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in (
        "httpx",
        "httpx2",
        "httpcore",
        "urllib3",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
        "mcp.server",
        "mcp.client",
        "uvicorn.access",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
