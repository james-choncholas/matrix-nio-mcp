#!/bin/sh
set -eu

REPO_ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

cleanup() {
  docker compose -f docker-compose.integration.yml down -v
}

trap cleanup EXIT INT TERM

docker compose -f docker-compose.integration.yml up -d

until curl -fsS http://localhost:6333/collections >/dev/null; do
  sleep 1
done

until curl -fsS http://localhost:8008/_matrix/client/versions >/dev/null; do
  sleep 1
done

.venv/bin/pytest tests/integration -v "$@"
