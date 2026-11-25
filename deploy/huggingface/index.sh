#!/bin/sh
# Build step: create the database and index the demo repositories into the image.
set -eu

pg_up() {
  pg_ctl -D "$PGDATA" -l /tmp/pg.log -w \
    -o "-c listen_addresses=127.0.0.1 -c port=5432 -c unix_socket_directories=/tmp" start
}

initdb -D "$PGDATA" -U badebhai --auth=trust --encoding=UTF8 >/dev/null
pg_up
createdb -h 127.0.0.1 -U badebhai badebhai

for repo in $DEMO_REPOS; do
  echo "indexing $repo"
  badebhai ingest "$repo"
done
badebhai repos

pg_ctl -D "$PGDATA" -m fast -w stop
