# RepoMind (BadeBhai)

An agentic RAG assistant that answers questions about a GitHub repository by combining code,
commit history, issues/PRs and docs, with verified citations and per-user memory.

> Work in progress (Stage 4 of 5). See [docs/PLAN.md](docs/PLAN.md) for the build plan.

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

## Agent and memory

`POST /ask` (JSON) or `POST /ask/stream` (server-sent events: `start`, `step`, `tool`, `token`, `final`).
The graph: `load_memory -> decompose -> retrieve (parallel MCP calls) -> grade_evidence
-> [rewrite_queries -> retrieve]x<=2 -> synthesize -> verify_citations -> write_memory`.

- Sub-questions with weak evidence are re-planned (new queries, follow-up calls such as blame on a
  found range), at most `MAX_RETRIES=2` times; then they are reported as **insufficient evidence**.
- Every `[n]` citation is checked against the index and the working copy; invalid ones are stripped
  and the claim is flagged.
- Long-term memory is per user and repo. Only durable facts (role, focus area, preferences,
  corrections, project context) above `MEMORY_IMPORTANCE_THRESHOLD` are kept; contradictions
  supersede, duplicates refresh, secrets and code are never stored, weight decays when unused.
  `GET /memories?user_id=`, `DELETE /memories/{id}?user_id=`, `DELETE /memories?user_id=`
  (CLI: `repomind memories list|delete|clear`).
- Thread state (short-term memory) lives in the LangGraph Postgres checkpointer.
- Tracing: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`; otherwise structured JSON logs.

## Evaluation

**Golden set:** 38 questions on `psf/requests` pinned at `v2.32.3`
([repomind/evals/golden/requests_v2_32_3.yaml](repomind/evals/golden/requests_v2_32_3.yaml)):
14 location, 5 how-it-works, 7 history/why, 2 live-issue, 4 change-planning, 3 docs and
3 unanswerable questions. Each has expected files / commit shas and a reference answer that was
checked against the snapshot (source, git history, HISTORY.md).

**Metrics:** retrieval hit@k (an expected file, commit or issue among the top-k evidence items),
citation validity (verified / total citations), whether the answer cites an expected source,
LLM-judge correctness (0-1 against the reference), "insufficient evidence" accuracy on unanswerable
questions, over-abstention, latency p50/p95, cost and LLM / tool calls per question.

```bash
uv run repomind eval --retrieval-only        # no LLM calls: hit@k of the naive retrieval
uv run repomind eval --limit 5               # both systems + judge on 5 questions (needs ANTHROPIC_API_KEY)
uv run repomind eval                         # full run; writes repomind/evals/reports/latest.md
```

### Results

| Metric | Naive RAG (1 hybrid search + 1 LLM call) | Agentic pipeline |
|---|---|---|
| Retrieval hit@1 | 71.4% | not run yet |
| Retrieval hit@3 | 91.4% | not run yet |
| Retrieval hit@5 | 91.4% | not run yet |
| Retrieval hit@10 | 91.4% | not run yet |
| Citation validity | not run yet | not run yet |
| LLM-judge correctness | not run yet | not run yet |
| Insufficient-evidence accuracy | not run yet | not run yet |
| Latency p50 / p95 | not run yet | not run yet |
| Cost per question | not run yet | not run yet |

Only the LLM-free rows have been measured so far: the build machine had no `ANTHROPIC_API_KEY`,
and these cells are left empty rather than estimated. Run `uv run repomind eval` to fill them in
(the agent column needs the LLM even for retrieval, because the planner chooses the tool calls).

Notes on the measured rows:
- The naive hit@k ceiling is 91.4% because 3 of the 35 answerable questions can only be answered
  from commits or live issues, which a code/doc index cannot return; the agent reaches them
  through the `git_history` MCP server.
- hit@1 went from 42.9% to 71.4% after two changes: presenting code and doc hits interleaved
  instead of sorted by score (docs used to bury code for "where is X" questions), and grouping runs
  of tiny definitions (for example the 25 small exception classes) into one chunk
  ([before](repomind/evals/reports/20261124-230021-before-interleave.md),
  [after](repomind/evals/reports/latest.md)).

## Tests

```bash
docker compose up -d db
uv run pytest           # DB tests use a separate `repomind_test` database
uv run ruff check .
RUN_LIVE=1 uv run pytest -m live   # real Claude API (needs ANTHROPIC_API_KEY)
```
