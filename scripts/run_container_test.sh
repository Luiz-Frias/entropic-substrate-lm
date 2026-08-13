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

# MLFLOW_S3_ENDPOINT_URL is equally required for the R2 artifact path
# (compose.yaml passes it straight through to the mlflow container's boto3
# client) — without it, R2 upload fails server-side with a boto3
# `ValueError: Invalid endpoint:` that surfaces client-side as a generic
# "too many 500 error responses" / MaxRetryError, exactly the confusing
# failure class the AWS-creds guard above exists to prevent. Verified
# empirically: an isolated stop/start with this var unset reproduces the
# generic retry-exhaustion error, not a clear abort message.
if [[ -z "${MLFLOW_S3_ENDPOINT_URL:-}" ]]; then
  echo "[container-test] MLFLOW_S3_ENDPOINT_URL not set (.env missing or empty) — the mlflow server's R2 artifact store would fail with a confusing boto3 'Invalid endpoint' / MaxRetryError; aborting now instead." >&2
  exit 1
fi

# Project name derived PER-WORKTREE, not fixed — NOT $$ either: a PID is not
# stable if teardown runs from a different shell invocation than the up,
# which is exactly the trap-on-EXIT case. Basename of this checkout's
# toplevel, sanitised to the charset podman-compose accepts for a project
# name ([a-z0-9_-] — the fixed prefix below guarantees the result still
# starts alphanumeric regardless of what the worktree happens to be named).
#
# WHY THIS MATTERS: a FIXED, shared project name is what let one agent's
# mlflow container fuse to a DIFFERENT agent's postgres under concurrent
# `just container-test` runs (verified empirically via the
# `com.docker.compose.project.working_dir` container label, and two
# concurrent `down -v` processes racing from different worktrees) — a run
# could VALIDATE AGAINST ANOTHER RUN'S DATABASE and report green. Exported
# (not just a local var) so tests/test_mlflow_live.py's own `podman compose
# -p ...` calls (the mid-run mlflow-kill test) target the SAME project this
# script brought up, instead of guessing or falling back to the old shared
# name.
TOPLEVEL="$(git rev-parse --show-toplevel)"
WORKTREE_NAME="$(basename "$TOPLEVEL")"
SANITIZED_WORKTREE="$(printf '%s' "$WORKTREE_NAME" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '-')"
# The basename ALONE is not unique, and fails in exactly the case this fix exists
# for: the Round C guard-teeth recipe mandates WT="$(mktemp -d)/wt", so every
# reviewer's worktree basename is literally "wt" and they would all resolve to one
# shared project name — restoring the container-fusion and cross-teardown hazard
# this derivation is meant to remove. Hash the FULL path so two worktrees can never
# collide however their directories are named. cksum is POSIX and always present;
# CRC32 over a handful of concurrent paths has negligible collision risk, and the
# readable basename is retained purely so `podman ps` stays human-legible.
PATH_HASH="$(printf '%s' "$TOPLEVEL" | cksum | cut -d' ' -f1)"
PROJECT="entropic-substrate-lm-mlflow-test-${SANITIZED_WORKTREE}-${PATH_HASH}"
export MLFLOW_COMPOSE_PROJECT="$PROJECT"
COMPOSE=(podman compose -p "$PROJECT" -f compose.yaml)
MLFLOW_CONTAINER="${PROJECT}_mlflow_1"

say() { printf '\n[container-test] %s\n' "$*"; }

cleanup() {
  # Capture the triggering exit status as the FIRST statement — `$?` here is
  # still the script's real exit code, not whatever `local`/the trap itself
  # would produce if this were computed later.
  local exit_status=$?
  if [[ "$exit_status" -eq 0 ]]; then
    say "success — tearing down the stack, including volumes..."
    "${COMPOSE[@]}" down -v --remove-orphans || true
  else
    # PRESERVE EVIDENCE ON FAILURE: no "-v" here — a red gate must not
    # destroy the very Postgres state an operator would need to diagnose it.
    # Volumes ARE still cleaned up on the success path above, so nothing
    # accumulates across green runs.
    say "FAILURE (exit ${exit_status}) — tearing down containers only; PRESERVING the postgres volume '${PROJECT}_pgdata' for diagnosis..."
    "${COMPOSE[@]}" down --remove-orphans || true
    say "inspect with: podman volume inspect ${PROJECT}_pgdata"
    say "remove manually once done: podman volume rm ${PROJECT}_pgdata"
  fi
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

# EPHEMERAL host port discovery: compose.yaml publishes only the bare
# container port ("5000", no "host:container" mapping — see its own
# comment), so podman assigns a random host port. `podman port <container>
# <container-port>` reports what was actually assigned as
# "0.0.0.0:<port>" (verified empirically against a scratch compose stack
# with this exact publish shape); take the text after the last colon.
# Port 5001 existed only to dodge macOS AirPlay squatting on 5000 — nothing
# is load-bearing about any specific number now, which is the point: a
# fixed number is exactly what let two concurrent worktrees collide.
say "discovering the ephemeral host port podman assigned..."
HOST_PORT="$(podman port "$MLFLOW_CONTAINER" 5000 | head -n1 | sed -E 's/.*:([0-9]+)$/\1/')"
if [[ -z "$HOST_PORT" ]]; then
  say "could not discover the host port mlflow's container port 5000 maps to — 'podman port' output:"
  podman port "$MLFLOW_CONTAINER" || true
  exit 1
fi
MLFLOW_URL="http://localhost:${HOST_PORT}"
say "mlflow reachable at ${MLFLOW_URL} (container port 5000 -> host port ${HOST_PORT})"

# Container-healthy is NOT REST-API-ready. The container's own HEALTHCHECK
# (compose.yaml) hits mlflow's lightweight, liveness-only `/health` endpoint,
# which never touches Postgres. VERIFIED EMPIRICALLY (isolated curl probes
# through this same host->podman-machine port forward, immediately after a
# fresh stack reported healthy): the first real, DB-backed REST call —
# get-experiment-by-name — took 7.27s; the second 1.22s; the third 0.11s. A
# one-time cold-start tax (SQLAlchemy engine/connection-pool init, first
# query-shape compile), not sustained network jitter — confirmed by the
# monotonic falloff to baseline over 3 calls.
#
# A GET-based warm-up (the previous shape here) only proves a READ path is
# warm — MlflowSink's first real call is a WRITE (create_run), a different
# connection/worker path with its own cold-start tax. This is exactly the
# gap that produced MlflowStartupError read-timeouts on `POST
# .../experiments/create` roughly 10 SECONDS AFTER this script's own
# warm-up had already reported the server warm — reproduced verbatim
# against the live gate. So the warm-up itself must be the same kind of
# call: a POST that performs a real write. Warming here — once, before
# pytest ever constructs a Trainer/MlflowSink — is a one-time setup cost
# with no GPU-hours downside, so it can afford to wait out the cold start
# properly instead of racing it.
say "warming the tracking server's REST API with a real WRITE (create, not just GET)..."
warm_response_file="$(mktemp)"
warm_status="cold"
warm_experiment_id=""
for i in $(seq 1 45); do
  # UNIQUE name per attempt: a create that client-side TIMES OUT may still
  # have landed server-side (that is precisely the failure mode above) — a
  # retry must never reuse a name a prior, timed-out-but-landed attempt
  # already claimed, or the retry itself would fail with a spurious
  # "experiment already exists".
  warmup_name="container-test-warmup-$(date +%s)-${i}-$$"
  code="$(curl -s -m 15 -o "$warm_response_file" -w '%{http_code}' \
    -X POST "${MLFLOW_URL}/api/2.0/mlflow/experiments/create" \
    -H 'Content-Type: application/json' \
    -d "{\"name\": \"${warmup_name}\"}" || echo "000")"
  if [[ "$code" == "200" ]]; then
    warm_status="warm"
    warm_experiment_id="$(python3 -c "
import json
with open('${warm_response_file}') as f:
    print(json.load(f).get('experiment_id', ''))
" 2>/dev/null || echo "")"
    break
  fi
  sleep 2
done
rm -f "$warm_response_file"

if [[ "$warm_status" != "warm" ]]; then
  say "mlflow REST API never warmed up on a real write (still cold after the wait budget) — logs:"
  "${COMPOSE[@]}" logs mlflow
  exit 1
fi

# Best-effort cleanup of the throwaway experiment: warmth is already proven
# by the create succeeding, so a delete failure here must never fail the
# gate — the whole stack (Postgres included) is torn down at the end of
# this script regardless.
if [[ -n "$warm_experiment_id" ]]; then
  curl -s -m 15 -X POST "${MLFLOW_URL}/api/2.0/mlflow/experiments/delete" \
    -H 'Content-Type: application/json' \
    -d "{\"experiment_id\": \"${warm_experiment_id}\"}" >/dev/null || true
fi

say "podman ps:"
podman ps

say "running integration tests against ${MLFLOW_URL}..."
export MLFLOW_TRACKING_URI="$MLFLOW_URL"
export ENTROPIC_MLFLOW_TINY=1
PYTHONPATH=src uv run pytest -m integration -q --disable-warnings
