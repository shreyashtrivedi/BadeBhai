# RepoMind (BadeBhai)

An agentic RAG assistant that answers questions about a GitHub repository by combining code,
commit history, issues/PRs and docs, with verified citations and per-user memory.

> Work in progress (Stage 2 of 5). See [docs/PLAN.md](docs/PLAN.md) for the build plan.

## One-command start (Docker)

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY (and optionally GITHUB_TOKEN)
docker compose up --build       # db + 3 MCP servers + API on http://localhost:8000
```

The demo repo (`psf/requests@v2.32.3`) is indexed automatically on first start (~1 min).
Set `AUTO_INGEST_REPO` to index something else, or `POST /repos {"source": "owner/name"}`.

## Local development

```bash
python -m pip install uv && uv sync
docker compose up -d db                       # Postgres + pgvector on localhost:5433
uv run repomind ingest psf/requests@v2.32.3   # clone + index
uv run repomind serve                         # API on :8000, also starts the 3 MCP servers
uv run repomind ask psf/requests "Why was the retry logic added, and is it related to any open issues?"
```

## Data sources are MCP servers

| Server | Port | Tools |
|---|---|---|
| `code_search` | 8101 | `search_code`, `read_code`, `file_outline`, `list_files` |
| `git_history` | 8102 | `search_commits`, `get_commit`, `blame`, `file_history`, `recent_changes`, `search_issues`, `get_issue`, `prs_for_commit` |
| `docs` | 8103 | `search_docs`, `read_doc`, `list_docs` |

Run them alone with `uv run repomind mcp up` (Streamable HTTP) or
`uv run repomind mcp serve code_search --transport stdio` (e.g. for Claude Desktop).
`GET /tools` shows what the API discovered.

## Keeping the index live

- **Webhook:** point a GitHub push webhook at `POST /webhooks/github` (content type JSON) and set
  `GITHUB_WEBHOOK_SECRET`. Signatures are verified; only the tracked branch is synced, and only
  changed files are re-chunked and re-embedded. (For local testing, forward with smee.io.)
- **CLI:** `uv run repomind sync psf/requests` diffs from the last indexed commit;
  `uv run repomind sync --all --watch --interval 30` keeps polling.
- **API polling:** `SYNC_POLL_INTERVAL_S=60`.
- Issues, PRs and commits are never bulk-indexed: they are read live through the `git_history` server.

## Tests

```bash
docker compose up -d db
uv run pytest           # DB tests use a separate `repomind_test` database
uv run ruff check .
```
