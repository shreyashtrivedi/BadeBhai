# RepoMind (BadeBhai)

An agentic RAG assistant that answers questions about a GitHub repository by combining code,
commit history, issues/PRs and docs, with verified citations and per-user memory.

> Work in progress. See [docs/PLAN.md](docs/PLAN.md) for the build plan.

## Quick start (local)

```bash
python -m pip install uv
uv sync
cp .env.example .env            # add ANTHROPIC_API_KEY
docker compose up -d db         # Postgres + pgvector on localhost:5433
uv run repomind ingest psf/requests@v2.32.3
uv run repomind search psf/requests "where is authentication handled"
uv run repomind ask psf/requests "Where is authentication handled?" --mode naive
uv run repomind serve           # API on http://localhost:8000
```

## Tests

```bash
docker compose up -d db
uv run pytest           # DB tests use a separate `repomind_test` database
uv run ruff check .
```
