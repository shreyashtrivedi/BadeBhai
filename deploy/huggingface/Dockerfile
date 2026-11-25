# BadeBhai as one container for a free Hugging Face Space (2 vCPU, 16 GB RAM, no GPU).
#
#   * Postgres 16 + pgvector run inside the container, listening on 127.0.0.1 only.
#   * The MCP data servers run in-process (MCP_TRANSPORT=inmemory): one process, no extra ports.
#   * The model is a free OpenAI-compatible API (Google Gemini by default); a 2-vCPU box is far too
#     slow to run one. Set OPENAI_API_KEY as a Space secret.
#   * The demo repositories are indexed at BUILD time, so a cold start answers within seconds.
#
# Built from the repository root by deploy/huggingface/deploy.sh (or the deploy-space workflow).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/.cache/hf \
    HF_HUB_OFFLINE=1 \
    DATA_DIR=/app/data \
    PGDATA=/app/pg \
    PATH="/app/.venv/bin:/usr/lib/postgresql/16/bin:$PATH"

# Postgres 16 + pgvector from the PostgreSQL project's apt repository (Debian's pgvector is too old for HNSW).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl gnupg \
    && install -d /usr/share/postgresql-common/pgdg \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc -o /usr/share/postgresql-common/pgdg/apt.postgresql.org.asc \
    && echo "deb [signed-by=/usr/share/postgresql-common/pgdg/apt.postgresql.org.asc] https://apt.postgresql.org/pub/repos/apt $(. /etc/os-release && echo $VERSION_CODENAME)-pgdg main" \
       > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16 postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*' \
    && pip install --no-cache-dir uv==0.13.10 \
    && useradd -m -u 1000 user

# Hugging Face runs Spaces as uid 1000: everything the app writes lives under /app, owned by that user.
WORKDIR /app
RUN chown user:user /app
USER user

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
RUN HF_HUB_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer as S; S('BAAI/bge-small-en-v1.5')"

COPY --chown=user badebhai ./badebhai
RUN uv sync --frozen --no-dev

# Public-demo configuration (override any of these as Space variables).
ENV DATABASE_URL=postgresql://badebhai@127.0.0.1:5432/badebhai \
    MCP_TRANSPORT=inmemory \
    LLM_PROVIDER=openai \
    OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
    LLM_MODEL=gemini-2.5-flash \
    OPENAI_FREE=true \
    OPENAI_MAX_RPM=10 \
    MAX_SUBQUESTIONS=3 \
    MAX_RETRIES=1 \
    DEMO_MODE=true \
    DEMO_REPOS="psf/requests@v2.32.3 sindresorhus/ky@v2.2.0" \
    LOG_JSON=true

COPY --chown=user deploy/huggingface/start.sh deploy/huggingface/index.sh ./
RUN chmod +x start.sh index.sh && ./index.sh

EXPOSE 7860
CMD ["./start.sh"]
