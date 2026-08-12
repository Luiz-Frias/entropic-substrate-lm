#!/usr/bin/env bash
# =============================================================================
# run_container_test.sh — the LIVE gate for phase p2-mlflow.
#
# Brings up postgres + mlflow (compose.yaml, podman-compose), waits for the
# tracking server to report healthy, runs the integration-marked test suite
# against it (real localhost tracking server, real Cloudflare R2 artifacts —
# never MinIO), then tears the stack down. Teardown runs on ANY exit path
# (trap on EXIT) so a red test run never leaves the stack up blocking the
# next invocation.
#
# Usage: just container-test   (wraps this script)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

# Explicit .env load, not left to ambient mise shell-hook activation: mise's
# `_.file = ".env"` (mise.toml) only reaches THIS process's environment when
# mise's shell hook is active in the calling shell. A bare `bash
# scripts/run_container_test.sh` (CI, a non-interactive invocation, or a
# fresh non-login shell) does not carry that — verified empirically: the
# pytest child process below raised botocore.exceptions.NoCredentialsError
# with AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY present in `.env` but absent
# from `$(env)` in that shell. `set -a` exports every var `source .env`
# defines, so both podman-compose's own `.env`-based ${VAR} interpolation
# AND this script's later `uv run pytest` child process see the same values.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  echo "[container-test] AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY not set (.env missing or empty) — R2 artifact tests would fail with a confusing NoCredentialsError; aborting now instead." >&2
  exit 1
fi

# Project name pinned (not left to podman-compose's default $(basename $PWD))
# so the container names below are deterministic regardless of the checkout
# directory's actual name (a worktree checkout is not named
# "entropic-substrate-lm"). Matches tests/test_mlflow_live.py's constant.
PROJECT="entropic-substrate-lm-mlflow-test"
COMPOSE=(podman compose -p "$PROJECT" -f compose.yaml)
MLFLOW_CONTAINER="${PROJECT}_mlflow_1"
MLFLOW_URL="http://localhost:5001"

say() { printf '\n[container-test] %s\n' "$*"; }

cleanup() {
  say "tearing down the stack..."
  "${COMPOSE[@]}" down -v --remove-orphans || true
}
trap cleanup EXIT

say "building + starting postgres + mlflow..."
"${COMPOSE[@]}" up -d --build

# NOTE: `podman-compose ps -q <service>` (1.6.0) does not accept a service
# filter — "unrecognized arguments: mlflow" — verified empirically. `podman
# ps --filter name=...` against the deterministic container name (above) is
# what actually works, so that is what this polls.
say "waiting for mlflow to report healthy..."
status="starting"
for _ in $(seq 1 60); do
  status="$(podman inspect --format '{{.State.Health.Status}}' "$MLFLOW_CONTAINER" 2>/dev/null || echo "starting")"
  [[ "$status" == "healthy" ]] && break
  sleep 2
done

if [[ "$status" != "healthy" ]]; then
  say "mlflow did not become healthy in time — logs:"
  "${COMPOSE[@]}" logs mlflow
  exit 1
fi

say "podman ps:"
podman ps

say "running integration tests against ${MLFLOW_URL}..."
export MLFLOW_TRACKING_URI="$MLFLOW_URL"
export ENTROPIC_MLFLOW_TINY=1
PYTHONPATH=src uv run pytest -m integration -q --disable-warnings
