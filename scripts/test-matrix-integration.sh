#!/bin/sh
set -eu

REPO_ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

cleanup() {
  docker compose -f docker-compose.integration.yml down -v
}

trap cleanup EXIT INT TERM

docker compose -f docker-compose.integration.yml up -d --wait

.venv/bin/pytest tests/integration -v "$@"
