#!/bin/sh
set -eu

REPO_ROOT="$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
    return
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
    return
  fi

  echo "docker compose or docker-compose is required" >&2
  exit 1
}

run_pytest() {
  if [ -x ".venv/bin/pytest" ]; then
    .venv/bin/pytest tests/integration -v "$@"
    return
  fi

  python -m pytest tests/integration -v "$@"
}

cleanup() {
  status=$?

  if [ "${status}" -ne 0 ]; then
    echo "Integration stack failed; dumping compose status and logs..." >&2
    compose -f docker-compose.integration.yml ps -a || true
    compose -f docker-compose.integration.yml logs --no-color || true
  fi

  compose -f docker-compose.integration.yml down -v
  trap - EXIT INT TERM
  exit "${status}"
}

trap cleanup EXIT INT TERM

compose -f docker-compose.integration.yml up -d --wait

run_pytest "$@"
