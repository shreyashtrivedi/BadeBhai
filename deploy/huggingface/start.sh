#!/bin/sh
# Container start: Postgres (local only), then the API + web UI on the port Hugging Face expects.
set -eu

if [ -z "${OPENAI_API_KEY:-}" ] && [ "${LLM_PROVIDER:-}" = "openai" ]; then
  echo "WARNING: OPENAI_API_KEY is not set; add it as a Space secret (Settings -> Variables and secrets)." >&2
fi

pg_ctl -D "$PGDATA" -l /tmp/pg.log -w \
  -o "-c listen_addresses=127.0.0.1 -c port=5432 -c unix_socket_directories=/tmp" start

exec badebhai serve --host 0.0.0.0 --port "${PORT:-7860}" --no-mcp
