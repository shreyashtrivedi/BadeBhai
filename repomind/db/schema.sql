-- RepoMind schema. `{dim}` is substituted with the configured embedding dimension.
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
