"""Minimal GitHub REST client: TTL cache, retries with backoff, clear rate-limit errors."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from repomind.log import get_logger

log = get_logger(__name__)


class GitHubError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class GitHubRateLimited(GitHubError):
    def __init__(self, reset_at: datetime | None, authenticated: bool):
        when = reset_at.strftime("%H:%M UTC") if reset_at else "soon"
        hint = (
            ""
            if authenticated
            else " Set GITHUB_TOKEN to raise the limit from 60 to 5,000 requests per hour."
        )
        super().__init__(f"GitHub API rate limit exceeded; it resets at {when}.{hint}", status=403)
        self.reset_at = reset_at


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        cache_ttl_s: int = 120,
        timeout_s: float = 15.0,
        max_attempts: int = 3,
        transport: httpx.BaseTransport | None = None,
    ):
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "repomind-mcp",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.authenticated = bool(token)
        self.http = httpx.Client(base_url=base_url, headers=headers, timeout=timeout_s, transport=transport)
        self.ttl = cache_ttl_s
        self.max_attempts = max_attempts
        self._cache: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] > now:
                return hit[1]
        data = self._fetch(path, params)
        with self._lock:
            self._cache[key] = (now + self.ttl, data)
        return data

    def _fetch(self, path: str, params: dict[str, Any] | None) -> Any:
        delay = 1.0
        last: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                r = self.http.get(path, params=params)
            except httpx.TransportError as e:
                last = GitHubError(f"network error talking to GitHub: {e}")
            else:
                if r.status_code in (403, 429) and self._is_rate_limited(r):
                    reset = r.headers.get("x-ratelimit-reset")
                    reset_at = datetime.fromtimestamp(int(reset), UTC) if reset else None
                    raise GitHubRateLimited(reset_at, self.authenticated)
                if r.status_code == 404:
                    raise GitHubError(f"not found: {path}", 404)
                if r.status_code == 422:
                    raise GitHubError(f"GitHub rejected the query: {r.text[:200]}", 422)
                if r.status_code < 400:
                    return r.json()
                last = GitHubError(f"GitHub returned HTTP {r.status_code}", r.status_code)
                if r.status_code < 500:
                    raise last
            if attempt < self.max_attempts:
                time.sleep(delay)
                delay *= 2
        raise last or GitHubError("GitHub request failed")

    @staticmethod
    def _is_rate_limited(r: httpx.Response) -> bool:
        if r.headers.get("x-ratelimit-remaining") == "0":
            return True
        if "retry-after" in r.headers:
            return True
        return "rate limit" in r.text.lower()
