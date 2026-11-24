FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    DATA_DIR=/data \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*' \
    && pip install --no-cache-dir uv==0.13.10

WORKDIR /app

# Dependencies first (cached layer). torch comes from the CPU-only index (see pyproject.toml).
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Bake the embedding model into the image (before the code, so code edits keep this layer cached).
# The image runs with HF_HUB_OFFLINE=1; if you switch EMBEDDING_MODEL, set HF_HUB_OFFLINE=0 in .env
# (or rebuild with --build-arg PRELOAD_MODEL=<model>).
ARG PRELOAD_MODEL=BAAI/bge-small-en-v1.5
RUN if [ -n "$PRELOAD_MODEL" ]; then \
      HF_HUB_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer as S; S('$PRELOAD_MODEL')"; \
    fi

COPY repomind ./repomind
RUN uv sync --frozen --no-dev

EXPOSE 8000
CMD ["repomind", "serve", "--host", "0.0.0.0", "--port", "8000", "--no-mcp"]
