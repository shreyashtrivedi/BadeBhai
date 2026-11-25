"""Optional accounts for shared deployments (AUTH_ENABLED=true).

* Passwords: scrypt (stdlib) with a per-user salt.
* Browser sessions: an HttpOnly, SameSite=Lax cookie holding `username|issued|signature` (HMAC-SHA256
  with SESSION_SECRET). Nothing is stored server-side; changing the secret signs everyone out.
* API keys (`bb_...`) for scripts and the MCP assistant: shown once, stored as a SHA-256 hash.
* Usage limits per user (questions per hour, repos added per day), kept in memory per API process.

Without AUTH_ENABLED none of this runs and BadeBhai behaves as a single-user local tool.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

SESSION_COOKIE = "bb_session"
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


@dataclass
class User:
    id: int
    username: str
    role: str
    disabled: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def to_public(self) -> dict:
        return {"username": self.username, "role": self.role}


# ------------------------------------------------------------------ passwords and tokens
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, dk = stored.split("$")
        got = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt), dklen=32, n=int(n), r=int(r), p=int(p)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got.hex(), dk)


def sign_session(username: str, secret: str, now: float | None = None) -> str:
    payload = f"{username}|{int(time.time() if now is None else now)}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode()


def read_session(token: str | None, secret: str, max_age_s: int, now: float | None = None) -> str | None:
    if not token:
        return None
    try:
        username, issued, sig = base64.urlsafe_b64decode(token.encode()).decode().rsplit("|", 2)
    except (ValueError, UnicodeDecodeError):
        return None
    expected = hmac.new(secret.encode(), f"{username}|{issued}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    if (time.time() if now is None else now) - int(issued) > max_age_s:
        return None
    return username


def new_api_key() -> str:
    return "bb_" + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# ------------------------------------------------------------------ user records
def _user(row: dict | None) -> User | None:
    return User(row["id"], row["username"], row["role"], row["disabled"]) if row else None


def create_user(pool: ConnectionPool, username: str, password: str, role: str = "member") -> User:
    if role not in ("admin", "member"):
        raise ValueError("role must be admin or member")
    if not username or not username.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ValueError("usernames may contain letters, digits, '-', '_' and '.'")
    if len(password) < 8:
        raise ValueError("passwords need at least 8 characters")
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (%s, %s, %s) "
            "RETURNING id, username, role, disabled",
            (username, hash_password(password), role),
        )
        return _user(cur.fetchone())


def authenticate(pool: ConnectionPool, username: str, password: str) -> User | None:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, username, role, disabled, password_hash FROM users WHERE username = %s", (username,)
        )
        row = cur.fetchone()
    if row is None:
        verify_password(password, hash_password("timing-equalizer"))  # same cost for unknown users
        return None
    if row["disabled"] or not verify_password(password, row["password_hash"]):
        return None
    return _user(row)


def get_user(pool: ConnectionPool, username: str) -> User | None:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, username, role, disabled FROM users WHERE username = %s", (username,))
        user = _user(cur.fetchone())
    return user if user and not user.disabled else None


def user_by_api_key(pool: ConnectionPool, key: str) -> User | None:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, username, role, disabled FROM users WHERE api_key_hash = %s AND NOT disabled",
            (hash_api_key(key),),
        )
        return _user(cur.fetchone())


def set_api_key(pool: ConnectionPool, username: str) -> str:
    key = new_api_key()
    with pool.connection() as conn:
        n = conn.execute(
            "UPDATE users SET api_key_hash = %s WHERE username = %s", (hash_api_key(key), username)
        ).rowcount
    if not n:
        raise KeyError(username)
    return key


def set_password(pool: ConnectionPool, username: str, password: str) -> None:
    if len(password) < 8:
        raise ValueError("passwords need at least 8 characters")
    with pool.connection() as conn:
        if not conn.execute(
            "UPDATE users SET password_hash = %s WHERE username = %s", (hash_password(password), username)
        ).rowcount:
            raise KeyError(username)


def list_users(pool: ConnectionPool) -> list[dict]:
    with pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT username, role, disabled, api_key_hash IS NOT NULL AS has_api_key, created_at "
            "FROM users ORDER BY username"
        )
        return cur.fetchall()


def delete_user(pool: ConnectionPool, username: str) -> bool:
    with pool.connection() as conn:
        conn.execute("DELETE FROM repo_members WHERE username = %s", (username,))
        return conn.execute("DELETE FROM users WHERE username = %s", (username,)).rowcount > 0


def count_users(pool: ConnectionPool) -> int:
    with pool.connection() as conn:
        return int(conn.execute("SELECT count(*) FROM users").fetchone()[0])


# ------------------------------------------------------------------ usage limits
class UsageLimiter:
    """Sliding-window counters per (user, action). Admins are never limited."""

    def __init__(self):
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(
        self, user: str, action: str, limit: int, window_s: int, now: float | None = None
    ) -> int | None:
        """Record one event; return None if allowed, else seconds until the next one is allowed."""
        if limit <= 0:
            return None
        now = time.time() if now is None else now
        with self._lock:
            q = self._events[(user, action)]
            while q and q[0] <= now - window_s:
                q.popleft()
            if len(q) >= limit:
                return int(q[0] + window_s - now) + 1
            q.append(now)
            return None


class DbUsageLimiter:
    """UsageLimiter kept in Postgres: shared by every API process and kept across restarts."""

    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    def check(
        self, user: str, action: str, limit: int, window_s: int, now: float | None = None
    ) -> int | None:
        if limit <= 0:
            return None
        with self.pool.connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"usage|{user}|{action}",))
            conn.execute(
                "DELETE FROM usage_events WHERE username = %s AND action = %s "
                "AND at <= now() - make_interval(secs => %s)",
                (user, action, window_s),
            )
            count, wait = conn.execute(
                "SELECT count(*), extract(epoch FROM min(at) + make_interval(secs => %s) - now()) "
                "FROM usage_events WHERE username = %s AND action = %s",
                (window_s, user, action),
            ).fetchone()
            if count >= limit:
                return int(wait or 0) + 1
            conn.execute("INSERT INTO usage_events (username, action) VALUES (%s, %s)", (user, action))
        return None
