# RepoMind (BadeBhai)

RepoMind answers questions about a GitHub repository by combining its **code, commit history,
issues/PRs and docs**. Every claim carries a citation that is checked programmatically, the index
stays **live** as the repo changes, and the assistant **remembers** durable facts about each user.

It is an agentic RAG system: a LangGraph agent decomposes the question, calls data sources that are
separate **MCP servers**, grades its own evidence (re-planning weak searches), and writes a cited
answer, or says "insufficient evidence" when the repository does not support one.

```text
You:      Why was the retry logic added, and is it related to any open issues?
RepoMind: Retries live in `HTTPAdapter` in `src/requests/adapters.py`, which turns `max_retries` into a
          urllib3 `Retry` [1]. The option was added in 2013 so callers could set retries when constructing
          the adapter [2]; retry handling was aligned with the docs in 2014 [3]. Issue #6512 discusses
          requests not being retried when the body is shorter than Content-Length [4].
          [1] src/requests/adapters.py#L202-L222  [2] commit 36dcce1a06  [3] commit 2bc7762cb6  [4] issue #6512
```
*(Illustrative output shape; see [Evaluation](#evaluation) for measured results.)*

## Contents
[Quick start](#quick-start) · [Architecture](#architecture) · [Using it](#using-it) ·
[How it works](#how-it-works) · [Evaluation](#evaluation) · [Configuration](#configuration) ·
[Development](#development) · [Limitations and next steps](#limitations-and-next-steps)

## Quick start

**Docker (one command):**

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY (GITHUB_TOKEN is optional but avoids rate limits)
docker compose up --build     # Postgres+pgvector, 3 MCP servers, API + web UI
```

Open **http://localhost:8000**. The demo repo (`psf/requests` pinned at `v2.32.3`) is cloned and
indexed automatically on first start, which takes about a minute. To index another repo, set
`AUTO_INGEST_REPO=owner/name` or `POST /repos {"source": "owner/name"}`.

**Local (Python 3.11+):**

```bash
python -m pip install uv && uv sync
cp .env.example .env                          # set ANTHROPIC_API_KEY
docker compose up -d db                       # Postgres+pgvector on localhost:5433
uv run repomind ingest psf/requests@v2.32.3   # clone + index (any public repo or local path works)
uv run repomind serve                         # http://localhost:8000, also starts the 3 MCP servers
```

Try: *Where is authentication handled?* · *Why was the retry logic added, and is it related to any
open issues?* · *What changed in src/requests between 2024-05-01 and 2024-05-31?* · *Which files would
I touch to add a new top-level helper like requests.get?*

## Architecture

```mermaid
flowchart LR
    subgraph clients[Clients]
        ui[Web UI]
        cli[repomind CLI]
    end

    subgraph api[FastAPI backend]
        direction TB
        lm[load_memory] --> dc[decompose]
        dc --> rt[retrieve<br/>parallel MCP calls]
        rt --> gr[grade_evidence]
        gr -- weak, retries left --> rw[rewrite_queries]
        rw --> rt
        gr -- done --> sy[synthesize<br/>streamed]
        sy --> vc[verify_citations]
        vc --> wm[write_memory]
    end

    subgraph mcp[MCP servers, Streamable HTTP]
        cs[code_search]
        gh[git_history]
        ds[docs]
    end

    subgraph ingest[Live ingestion]
        wh[POST /webhooks/github]
        sync[repomind sync / polling]
        idx[Indexer<br/>tree-sitter chunks + embeddings]
    end

    ui -- SSE /ask/stream --> api
    cli --> api
    rt --> cs & gh & ds
    dc & gr & rw & sy & wm -.-> llm[(Claude<br/>Anthropic API)]
    cs & ds --> pg[(Postgres + pgvector<br/>chunks, files, memories,<br/>LangGraph checkpoints)]
    lm & wm --> pg
    gh --> clone[(git clone)]
    gh --> ghapi[(GitHub REST:<br/>issues, PRs)]
    vc --> pg
    wh & sync --> idx --> pg
    idx --> clone
```

| Piece | Choice |
|---|---|
| Agent | LangGraph; state checkpointed per thread in Postgres (`AsyncPostgresSaver`) |
| LLM | Claude (`claude-opus-5-5` by default) behind a small `LLMProvider` interface; structured outputs for planning, grading and memory; streamed synthesis |
| Data sources | three MCP servers built with the official MCP Python SDK; the agent is an MCP client and discovers tools with `list_tools` |
| Storage | one Postgres: pgvector HNSW + `tsvector` full-text (hybrid search with reciprocal rank fusion), files and hashes, memories, checkpoints |
| Chunking | tree-sitter function/class chunks (Python, JS, TS/TSX, Go, Java, Rust), heading-based Markdown/RST sections, sliding windows otherwise |
| Embeddings | local `BAAI/bge-small-en-v1.5` (sentence-transformers, CPU), configurable |

## Using it

**Web UI** (`/`): ask questions, watch the agent's steps stream in (sub-questions, MCP tool calls,
grading, verification), click `[n]` chips to open the cited code, commit or issue, and see or delete
what RepoMind remembers about you.

**API**

| Endpoint | Purpose |
|---|---|
| `POST /ask` | `{"question", "repo", "mode": "agent"\|"naive", "user_id", "thread_id"}` returns the answer, citations, verification report, trace, sub-questions, tool calls, usage |
| `POST /ask/stream` | the same as server-sent events: `start`, `step`, `tool`, `token`, `final`, `error` |
| `GET /memories?user_id=&repo=` · `DELETE /memories/{id}?user_id=` · `DELETE /memories?user_id=` | user control over long-term memory |
| `POST /repos` · `GET /repos` · `POST /repos/{owner}/{name}/sync` · `GET /jobs/{id}` | ingest, list, sync |
| `POST /webhooks/github` | GitHub push webhook (HMAC-SHA256 verified) |
| `GET /search?repo=&q=&kind=code\|doc\|all` · `GET /tools` | debugging: raw hybrid search, discovered MCP tools |

**CLI**: `repomind ingest | sync [--watch] | repos | search | ask | serve | mcp up | mcp serve |
memories list|delete|clear | eval | reset-db`.

**MCP servers** can also be used on their own, for example from Claude Desktop:
`uv run repomind mcp serve git_history --transport stdio`.

| Server | Port | Tools |
|---|---|---|
| `code_search` | 8101 | `search_code`, `read_code`, `file_outline`, `list_files` |
| `git_history` | 8102 | `search_commits`, `get_commit`, `blame`, `file_history`, `recent_changes`, `search_issues`, `get_issue`, `prs_for_commit` |
| `docs` | 8103 | `search_docs`, `read_doc`, `list_docs` |

## How it works

**The agent.** `decompose` splits the question into up to 5 sub-questions, each with the sources it
needs (code, docs, commits, issues) and first tool calls planned from the MCP tool catalog. `retrieve`
runs all planned calls in parallel. `grade_evidence` scores each evidence item's relevance to its
sub-question and decides whether the evidence is sufficient. Weak sub-questions go to
`rewrite_queries`, which plans new calls from what was found (for example `blame` on a code range it
found, then `get_commit` / `prs_for_commit`), never repeating a call, at most `MAX_RETRIES=2` times;
after that the sub-question is marked **insufficient**. `synthesize` writes the answer from graded
evidence only (if nothing passed grading, the answer is a fixed "Insufficient evidence" message, so
there is no path to a guessed answer). `write_memory` updates long-term memory.

**Citations.** The model writes `[n]` markers that refer to numbered evidence; it never writes the
citation targets itself. `verify_citations` then checks every marker: the evidence id exists, the
file is indexed, the line range exists, and the cited lines still match the file (catching stale or
fabricated ranges); commits must exist in the clone. Invalid markers are stripped, claims left
unsupported are flagged `⚠️(unverified)`, inline `path:line` references are checked too, and
surviving citations are renumbered.

**Live data.** Each file's sha256 and each repo's last indexed commit are stored. A signed push
webhook, `repomind sync`, or polling diffs from the last indexed commit and re-chunks and re-embeds
only changed files, deleting chunks of removed files; a per-repo advisory lock serializes runs.
Issues, PRs and commits are not bulk-indexed: `git_history` reads commits, blame and diffs from the
local clone and issues/PRs live from GitHub (short TTL cache, retries with backoff, a clear message
on rate limits). Ingestion respects `.gitignore`, skips binaries, vendored and huge files, and redacts
secrets (keys, tokens, PEM blocks) without changing line numbers.

**Memory.** Short-term: the thread's recent turns live in the LangGraph checkpointer (thread ids are
namespaced per user). Long-term: a `memories` table per user and repo. An extractor proposes facts from
the user's own messages only (never from repository text), and a write policy keeps only durable,
user-specific facts (role, focus area, preferences, repeated corrections, project context) with
importance at or above `MEMORY_IMPORTANCE_THRESHOLD`; it rejects code, questions, secrets and
instruction-like text. A contradicting fact supersedes the old one, a duplicate refreshes it, and
retrieval weight decays with a 30-day half-life when a memory is not used. The extractor is only
called when the user says something about themselves, which saves an LLM call on most questions.

**Safety.** The agent is read-only: it never writes to the repo or runs its code; git arguments are
validated (no option injection) and paths are confined to the repo. All repository text is
untrusted: it is placed only inside evidence blocks delimited by a fresh random tag per request,
delimiter look-alikes are neutralized, instruction-like text is flagged, and every prompt tells the
model to ignore instructions inside evidence (tested end to end, plus an opt-in live test). Tool
calls, retries and request time are capped per question.

## Evaluation

**Golden set:** 38 questions on `psf/requests` pinned at `v2.32.3`
([golden YAML](repomind/evals/golden/requests_v2_32_3.yaml)): 14 location, 5 how-it-works,
7 history/why, 2 live-issue, 4 change-planning, 3 docs and 3 unanswerable. Each has expected files or
commit shas and a reference answer checked against the snapshot (source, git history, HISTORY.md).

**Metrics:** retrieval hit@k (an expected file, commit or issue among the top-k evidence items),
citation validity (verified / total), whether the answer cites an expected source, LLM-judge
correctness (0-1 against the reference), "insufficient evidence" accuracy on unanswerable questions,
over-abstention, latency p50/p95, cost and LLM / tool calls per question.

```bash
uv run repomind eval --retrieval-only     # no LLM calls: hit@k of the naive retrieval
uv run repomind eval --limit 5            # both systems + judge on 5 questions (needs ANTHROPIC_API_KEY)
uv run repomind eval                      # full run; writes repomind/evals/reports/latest.md
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

Only the LLM-free rows have been measured: the build machine had no `ANTHROPIC_API_KEY`, and the other
cells are left empty rather than estimated. `uv run repomind eval` fills them in. The agent column
needs the LLM even for retrieval, because the planner chooses the tool calls.

- Naive hit@k tops out at 91.4% because 3 of the 35 answerable questions can only be answered from
  commits or live issues, which a code/doc index cannot return; the agent reaches them through the
  `git_history` server.
- hit@1 went from 42.9% to 71.4% after two changes the eval motivated: presenting code and doc hits
  interleaved instead of sorted by score (docs used to bury code on "where is X" questions), and
  grouping runs of tiny definitions such as the 25 small exception classes into one chunk
  ([before](repomind/evals/reports/20261124-230021-before-interleave.md),
  [after](repomind/evals/reports/latest.md)).

## Configuration

All settings come from environment variables or `.env` ([.env.example](.env.example)). The main ones:

| Variable | Default | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | (none) | required for answers |
| `LLM_MODEL` / `LLM_FAST_MODEL` | `claude-opus-5-5` / same | main model; model for decompose/grade/rewrite/memory (for example `claude-haiku-5-5` to cut cost) |
| `LLM_EFFORT` / `LLM_FAST_EFFORT` | `medium` / `low` | effort for synthesis / for the small structured steps |
| `LLM_FALLBACKS` | `default` | server-side refusal fallback (`off` to disable) |
| `GITHUB_TOKEN` | (none) | raises GitHub limits from 60 to 5,000 requests/hour |
| `GITHUB_WEBHOOK_SECRET` | (none) | enables `POST /webhooks/github` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / `384` | changing the model triggers a full re-embed |
| `MCP_TRANSPORT` | `http` | `http`, `stdio` or `inmemory` |
| `MAX_TOOL_CALLS` / `MAX_RETRIES` / `MAX_SUBQUESTIONS` | `24` / `2` / `5` | per-question guardrails |
| `GRADE_THRESHOLD` | `0.5` | minimum graded relevance for evidence to be used |
| `MEMORY_IMPORTANCE_THRESHOLD` / `MEMORY_HALF_LIFE_DAYS` | `0.6` / `30` | memory write policy and decay |
| `SYNC_POLL_INTERVAL_S` | `0` | poll for new commits every N seconds (0 = off) |
| `AUTO_INGEST_REPO` | (none; compose sets `psf/requests@v2.32.3`) | ingest on startup if missing |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` | `false` / (none) | LangSmith traces of every graph node and LLM call; otherwise JSON logs |

Prompts are versioned files in [repomind/app/prompts](repomind/app/prompts) (`name.vN.md`); pin one
with `PROMPT_VERSION_<NAME>=N`.

## Development

```bash
docker compose up -d db
uv run pytest                        # 87 tests; DB tests use a separate `repomind_test` database
uv run ruff check . && uv run ruff format --check .
RUN_LIVE=1 uv run pytest -m live     # 2 tests against the real Claude API
```

Tests use a deterministic hashing embedder and a scripted fake LLM, so they need no downloads or keys.
They cover chunking, redaction, hybrid retrieval, incremental sync, the three MCP servers (with a
mocked GitHub), the citation verifier, the memory policy, the full `/ask` and `/ask/stream` flows,
grading and retries, insufficient evidence, prompt-injection containment, webhooks, and the eval runner.

Recording a demo: see [docs/demo-script.md](docs/demo-script.md) (browser walkthrough) and
[docs/demo.tape](docs/demo.tape) (terminal GIF with VHS).

```
repomind/
  app/          api, agent graph + nodes, prompts, memory, citations, MCP client, naive baseline
  mcp_servers/  code_search, git_history, docs (+ GitHub client)
  ingestion/    chunker, filters, secret redaction, indexer, webhook, sync
  retrieval/    hybrid search
  llm/          provider interface, Claude implementation, fake
  evals/        golden set, runner, metrics, judge, reports
  ui/           web UI served at /
tests/          unit + integration
```

## Limitations and next steps

**Limitations**
- **The LLM-dependent evaluation has not been run yet** (no API key during the build). Retrieval
  numbers are real; answer quality, citation validity on live answers, latency and cost are unmeasured.
- **No authentication.** `user_id` is supplied by the client, so anyone who knows a user id can read
  or delete that user's memories. Fine for a local demo, not for a shared deployment.
- The verifier proves that cited evidence is real and unchanged; it does not prove that the claim
  follows from the cited text (entailment).
- Issues and PRs come from GitHub's keyword search API (10 searches/minute unauthenticated), not
  semantic search; issue questions are non-deterministic in the eval because the data is live.
- tree-sitter chunking covers six languages; others use sliding windows. Wiki indexing is
  best-effort (public GitHub wikis only).
- All repos share one HNSW index, which relies on pgvector's iterative scan for filtered queries;
  very large monorepos would want per-repo partitioning, and history tools need a full clone.
- On Windows the API must use the HTTP MCP transport (async psycopg needs a selector event loop,
  which cannot spawn stdio subprocesses); Docker and Linux/macOS are unaffected.

**Next steps**
1. Run the full eval with a key; tune `GRADE_THRESHOLD`, prompts per question category, and measure
   `LLM_FAST_MODEL=claude-haiku-5-5` for the structured steps against quality.
2. Add a cross-encoder reranker after hybrid search, and an entailment check to the verifier.
3. Put the API behind auth (OIDC) and derive `user_id` from the token; per-repo webhook secrets or a
   GitHub App.
4. Cache issue/PR embeddings for semantic search over discussions, including PR review comments.
5. Partition chunks per repo and support shallow clones with on-demand history.
