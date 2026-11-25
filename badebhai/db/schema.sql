-- BadeBhai schema. `{dim}` is substituted with the configured embedding dimension.
-- Idempotent: safe to run on every startup.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repos (
    id                  SERIAL PRIMARY KEY,
    name                TEXT UNIQUE NOT NULL,       -- "owner/name" for GitHub repos, folder name for local ones
    kind                TEXT NOT NULL DEFAULT 'local', -- github | git (managed clones) | local (read-only)
    source              TEXT NOT NULL,              -- clone URL or local path
    local_path          TEXT NOT NULL,              -- working copy that indexing reads from
    ref                 TEXT,                       -- branch or tag being tracked
    github_full_name    TEXT,                       -- owner/name when the repo is on GitHub
    wiki_path           TEXT,                       -- local clone of the GitHub wiki, if any
    last_indexed_commit TEXT,
    last_indexed_at     TIMESTAMPTZ,
    embedding_model     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS files (
    repo_id      INT  NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path         TEXT NOT NULL,
    content_hash TEXT NOT NULL,                     -- sha256 of raw file bytes
    kind         TEXT NOT NULL,                     -- code | doc
    language     TEXT,
    size_bytes   INT  NOT NULL,
    line_count   INT  NOT NULL,
    indexed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo_id, path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    repo_id     INT  NOT NULL,
    path        TEXT NOT NULL,
    kind        TEXT NOT NULL,                      -- code | doc
    language    TEXT,
    symbol      TEXT,                               -- function / class / heading
    symbol_type TEXT,
    start_line  INT  NOT NULL,                      -- 1-based, inclusive
    end_line    INT  NOT NULL,                      -- 1-based, inclusive
    content     TEXT NOT NULL,                      -- exact (redacted) source lines start..end
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector NOT NULL,
    FOREIGN KEY (repo_id, path) REFERENCES files(repo_id, path) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_repo_kind_idx ON chunks (repo_id, kind);
CREATE INDEX IF NOT EXISTS chunks_repo_path_idx ON chunks (repo_id, path);

CREATE TABLE IF NOT EXISTS memories (
    id            BIGSERIAL PRIMARY KEY,
    user_id       TEXT NOT NULL,
    repo          TEXT NOT NULL,
    text          TEXT NOT NULL,
    category      TEXT NOT NULL DEFAULT 'other',
    embedding     vector({dim}) NOT NULL,
    importance    REAL NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    use_count     INT NOT NULL DEFAULT 0,
    source_thread TEXT,
    active        BOOLEAN NOT NULL DEFAULT TRUE,
    superseded_by BIGINT REFERENCES memories(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS memories_scope_idx ON memories (user_id, repo) WHERE active;

-- 👍/👎 on answers. Thumbs-down rows become candidate eval questions (`badebhai feedback export`).
CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id     TEXT NOT NULL,
    repo        TEXT NOT NULL,
    commit      TEXT,
    mode        TEXT,
    thread_id   TEXT,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT,
    citations   JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS feedback_repo_idx ON feedback (repo, rating, created_at DESC);

-- Recent answers, reused for the same question on the same commit of a repo (see app/cache.py).
CREATE TABLE IF NOT EXISTS answer_cache (
    repo_id     INT  NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    commit      TEXT NOT NULL,
    mode        TEXT NOT NULL,
    qkey        TEXT NOT NULL,                      -- sha256 of the normalized question
    question    TEXT NOT NULL,
    response    JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    hits        INT NOT NULL DEFAULT 0,
    PRIMARY KEY (repo_id, commit, mode, qkey)
);

-- Optional accounts (AUTH_ENABLED=true). Without auth, none of this is used.
CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,                    -- scrypt, see app/auth.py
    role          TEXT NOT NULL DEFAULT 'member',   -- admin | member
    api_key_hash  TEXT,                             -- sha256 of the user's API key (shown once)
    disabled      BOOLEAN NOT NULL DEFAULT FALSE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Shared repos are visible to everyone; others only to their members (and admins).
ALTER TABLE repos ADD COLUMN IF NOT EXISTS shared BOOLEAN NOT NULL DEFAULT TRUE;

CREATE TABLE IF NOT EXISTS repo_members (
    repo_id  INT  NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    username TEXT NOT NULL,
    added_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo_id, username)
);

-- Usage-limit events (AUTH_ENABLED), shared by every API process and kept across restarts.
CREATE TABLE IF NOT EXISTS usage_events (
    username TEXT NOT NULL,
    action   TEXT NOT NULL,
    at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS usage_events_idx ON usage_events (username, action, at);

-- Bot replies (GitHub @badebhai, Slack /badebhai) not yet posted; resumed after a restart.
CREATE TABLE IF NOT EXISTS bot_replies (
    id         BIGSERIAL PRIMARY KEY,
    kind       TEXT NOT NULL,                       -- github | slack
    payload    JSONB NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',     -- pending | done | failed
    attempts   INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
