# RepoMind (BadeBhai): build plan

This is the Stage 0 plan. It stays in the repo so the reasoning behind the layout is visible.

## Module plan

```
repomind/
  config.py               Settings (pydantic-settings, .env)
  log.py                  JSON structured logging
  cli.py                  `repomind` CLI: ingest, sync, serve, mcp, ask, memories, eval
  db/
    schema.sql            repos, files, chunks (pgvector + tsvector), memories
    pool.py               psycopg connection pool + schema bootstrap
  llm/
    base.py               LLMProvider interface (complete / complete_json / stream) + usage accounting
    anthropic_provider.py Claude implementation (structured outputs, refusal fallback, streaming)
    fake.py               scripted fake LLM for tests
  embeddings/             Embedder interface, sentence-transformers (bge-small), hashing (tests)
  ingestion/
    filters.py            .gitignore (via git ls-files / pathspec), binary + size caps, vendored dirs
    secrets.py            secret redaction (line-preserving)
    languages.py          extension -> language, tree-sitter grammars and node types
    chunker.py            tree-sitter function/class chunks, markdown/rst heading chunks, sliding-window fallback
    sources.py            clone / local path / wiki clone, git helpers
    indexer.py            full + incremental indexing keyed by content hash and last indexed commit
    webhook.py            GitHub signature verification + push handling
    sync.py               `repomind sync` (diff from last indexed commit) + polling loop
  retrieval/hybrid.py     vector (HNSW cosine) + full-text (tsvector) fused with reciprocal rank fusion
  mcp_servers/
    code_search.py        search_code, read_code, list_files, file_outline
    git_history.py        search_commits, get_commit, blame, file_history, recent_changes,
                          search_issues, get_issue, prs_for_commit (local git + GitHub REST)
    docs.py               search_docs, read_doc, list_docs
    github_client.py      REST client: TTL cache, retries with backoff, rate-limit errors
  app/
    api.py                FastAPI: /ask, /ask/stream (SSE), /memories, /webhooks/github, /repos, UI
    schemas.py            Evidence, Citation, SubQuestion, TraceStep, request/response models
    prompts/*.md          versioned prompt files (name.vN.md)
    mcp_client.py         MCP toolbox: http / stdio / in-memory transports, timeouts, budget
    naive.py              naive RAG baseline (Stage 1 /ask, eval baseline)
    agent/                LangGraph graph + nodes, citation verifier, injection defense, guardrails
    memory/               long-term memory store (decay) + write policy (threshold, conflicts, secrets)
  evals/                  golden set (psf/requests @ pinned tag), runner, metrics, report
  ui/                     static chat page: streamed answer, citations, step trace, memories
tests/                    unit + integration (fake LLM, in-memory MCP, real Postgres when available)
```

## Assumptions and decisions

| Decision | Choice | Why |
|---|---|---|
| Project layout | `repomind/` is the single Python package; evals/ui live inside it | one installable package, `repomind` CLI |
| MCP transport | streamable HTTP by default (stdio supported) | servers run as separate processes/containers; on Windows, async psycopg (checkpointer) needs the selector loop, which can't spawn stdio subprocesses |
| LLM | `claude-opus-5-5` for all roles; `LLM_FAST_MODEL` overrides decompose/grade/memory | cost is your call; one switch to change it |
| Structured LLM output | JSON-schema structured outputs (`output_config.format`) | forced tool choice is not available on current models |
| Refusal fallback | server-side `fallbacks: "default"` on, `LLM_FALLBACKS=off` disables | recommended default for current Claude models |
| Commits / blame / diffs | local git clone | no rate limit, exact line data |
| Issues / PRs | GitHub REST, live, TTL cache | always current; never bulk-indexed |
| Code + docs storage | one `chunks` table, `kind` = code or doc | the same hybrid search serves both MCP servers |
| Citations | the model cites `[n]`; the citation list is built programmatically from evidence | the model cannot fabricate a citation target |
| Memory extraction input | the user's own messages only, never retrieved repo text | repo content cannot plant memories (injection defense) |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d); `hash` embedder for tests | no downloads in CI |
| Demo / eval repo | `psf/requests`, pinned tag | small, well known, rich issue/PR history; matches the example questions |
| User identity | `user_id` passed by the client (no auth) | demo scope; noted as a limitation |

## Stages

1. Core: compose with Postgres+pgvector, ingestion, tree-sitter chunking, embeddings, hybrid search, `/ask` (naive RAG) with file:line citations.
2. Live + multi-source: three MCP servers, incremental indexing, webhook + `repomind sync` + polling; agent calls tools via MCP.
3. Agent brain: decompose, parallel retrieve, grade with query-rewrite retries, synthesize, citation verifier, long-term memory, LangSmith, SSE.
4. Quality: 30+ golden questions, hit@k, citation validity, LLM-judge correctness, latency/cost; naive vs agentic table.
5. Polish: web UI with trace, one-command Docker start, README with a Mermaid diagram, demo script, limitations.
