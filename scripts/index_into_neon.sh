#!/usr/bin/env bash
# Index the hosted demo's repositories from this machine straight into the Neon database, so the small
# Render instance never has to embed anything (indexing there runs it out of memory). Render then only
# restores the git clones on start; its own auto-ingest finds every file unchanged and skips the work.
#
#   1. Put the Neon connection string in .env.neon (git-ignored), one line:
#        DATABASE_URL=postgresql://...neon.tech/neondb?sslmode=require
#   2. bash scripts/index_into_neon.sh
#
# The embedding settings must match render.yaml, or Render re-indexes on start.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
envfile="${NEON_ENV_FILE:-$root/.env.neon}"
[ -f "$envfile" ] || { echo "missing $envfile (one line: DATABASE_URL=postgresql://...)" >&2; exit 1; }
DATABASE_URL="$(sed -n 's/^[[:space:]]*DATABASE_URL[[:space:]]*=[[:space:]]*//p' "$envfile" | tr -d '\r"'"'" | head -n1)"
[ -n "$DATABASE_URL" ] || { echo "no DATABASE_URL line in $envfile" >&2; exit 1; }
export DATABASE_URL

export EMBEDDING_PROVIDER=fastembed
export EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
export EMBEDDING_DIM=384
export EMBEDDING_BATCH_SIZE=32
export EMBEDDING_RESET_ON_MISMATCH=true
export DATA_DIR="${DATA_DIR:-$root/data/neon-index}"
export LOG_JSON=false

repos="${DEMO_REPOS:-psf/requests@v2.32.3 sindresorhus/ky@v2.2.0}"
if [ -x "$root/.venv/Scripts/badebhai.exe" ]; then cli="$root/.venv/Scripts/badebhai.exe"
elif [ -x "$root/.venv/bin/badebhai" ]; then cli="$root/.venv/bin/badebhai"
else cli="uv run --no-sync badebhai"; fi

for repo in $repos; do
  echo "== indexing $repo into Neon"
  $cli ingest "$repo"
done
$cli repos
