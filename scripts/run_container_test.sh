#!/usr/bin/env bash
# =============================================================================
# run_container_test.sh — the LIVE gate for phases p2-mlflow / p3-allowlist /
# p4-caddy-auth.
#
# THREE SEQUENTIAL PHASES, never concurrent:
#   1. Brings up postgres + mlflow (compose.yaml, podman-compose), waits for
#      the tracking server to report healthy, runs the main integration
#      suite against it (real localhost tracking server, real Cloudflare R2
#      artifacts — never MinIO), then tears the stack down BEFORE phase 2
#      starts.
#   2. tests/test_mlflow_allowed_hosts.py — self-contained (its own
#      module-scoped fixture owns a SEPARATE compose stack, own project
#      name, own teardown).
#   3. tests/test_caddy_auth.py (p4) — likewise self-contained, its own THIRD
#      compose stack (`--profile proxy`, so it alone includes the caddy
#      service), own project name, own teardown. Proves the Caddy-fronted
#      auth path this script's own live gate is the only thing that
#      exercises (see CLAUDE.md's House rules on why the gate is a hard
#      dependency, not a courtesy).
#
# WHY SEQUENTIAL, NOT CONCURRENT (measured, not assumed): an earlier attempt
# ran phase 2 as ordinary pytest inside phase 1's `-m integration` call,
# while the main stack was still up. MEASURED on this 4-vCPU podman VM: a
# second full postgres+mlflow pair, booted while the first sits idle,
# consumed only ~39% CPU share (`podman top` on the starving process) and
# its healthcheck was still failing (FailingStreak 34/40) after 5+ minutes —
# approaching compose.yaml's own healthcheck retry budget (interval 3s x
# retries 40) with NO margin, i.e. a coin flip between "eventually healthy"
# and "permanently unhealthy," not a slow-but-bounded delay. Raising a
# client-side wait timeout would not fix that: the ceiling is the
# healthcheck's own retry budget, which invariant #5 forbids touching (it
# would change the main stack's behavior for everyone). Root cause is two
# full stacks contending for one machine's CPU — the fix is not running
# them at the same time, not a bigger number. Solo (this shape), the
# allowlist stack's mlflow container was observed healthy well inside its
# existing 120s bound every time — no bound in that module needed raising.
#
# Teardown runs on ANY exit path for phase 1 (trap on EXIT) so a red test
# run never leaves that stack up blocking the next invocation; phases 2 and 3
# each own their own teardown entirely inside their own pytest fixture
# (down -v in a finally), so nothing at this script's level needs to clean
# either up on the normal path (this script's re-armed EXIT trap below is
# only a SIGTERM backup net for both).
#
# Usage: just container-test   (wraps this script)
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

say() { printf '\n[container-test] %s\n' "$*"; }

# IDENTITY GUARD (p5, defects #4 / #5-fix-1) — extracted to its own file so
# it is directly, hermetically testable; see that file's header for the
# full rationale and the measured defect it closes.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/assert_expected_toplevel.sh"

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

# ENFORCED, not merely requested: neutralize the host-allowlist knobs for THIS
# stack. A set allowlist REPLACES mlflow's compiled-in localhost default rather
# than extending it (verified empirically against the pinned image), and this
# script reaches its own mlflow container at localhost:<ephemeral port> — so a
# Railway-domain value would silently 403 every integration test below, on one
# developer's machine and nobody else's. tests/test_mlflow_allowed_hosts.py
# boots a SEPARATE, short-lived stack to exercise the allowlist instead.
#
# EMPTY EXPORT, NOT `unset` — the intuitive fix does not work. podman-compose
# reads the .env beside compose.yaml ITSELF, independently of this process's
# environment, so removing these from the process env leaves the .env value in
# full force. MEASURED against podman-compose 1.6.0 (scratch stack, .env
# containing MLFLOW_ALLOWED_HOSTS=FROM_DOTENV_FILE, compose interpolating
# ${MLFLOW_ALLOWED_HOSTS:-UNSET_DEFAULT}):
#
#   process env UNSET               -> FROM_DOTENV_FILE   (.env wins!)
#   process env = FROM_PROCESS_ENV  -> FROM_PROCESS_ENV
#   process env = "" (empty)        -> UNSET_DEFAULT      (empty wins)
#
# An empty EXPORTED value outranks .env and lets compose.yaml's `:-` default
# apply; `unset` does not. Pinned by tests/test_run_container_test_config.py.
export MLFLOW_ALLOWED_HOSTS=""
export MLFLOW_CORS_ALLOWED_ORIGINS=""

# Project identity — derivation EXTRACTED to scripts/derive_compose_project.sh
# (p5) so its own runtime behaviour (same worktree, same second, two
# DIFFERENT project names) is directly testable by invoking that file twice
# as a subprocess (tests/test_derive_compose_project.py), which a grep of
# THIS script's source text never could. Sets PROJECT_FAMILY (stable per
# worktree — the identity tests/helpers.py's assert_no_foreign_mlflow_stack
# uses to recognise "another run of mine") and PROJECT (PROJECT_FAMILY plus
# per-run entropy — defect #1, see that script's own header for the
# measurement and the retraction of the earlier "NOT $$ either" reasoning).
#
# Both exported, not just local vars: MLFLOW_COMPOSE_PROJECT so
# tests/test_mlflow_live.py's own `podman compose -p ...` calls (the
# mid-run mlflow-kill test) target the SAME project this script brought up
# instead of guessing; MLFLOW_COMPOSE_PROJECT_FAMILY so
# tests/test_mlflow_allowed_hosts.py and tests/test_caddy_auth.py's own
# assert_no_foreign_mlflow_stack calls (phases 2 and 3) can pass `family=`
# and get the twin-detecting, poll-before-refusing behaviour p5 added there —
# see that function's docstring for why `family` is required for a project
# name that carries per-run entropy.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/derive_compose_project.sh"
export MLFLOW_COMPOSE_PROJECT="$PROJECT"
export MLFLOW_COMPOSE_PROJECT_FAMILY="$PROJECT_FAMILY"
COMPOSE=(podman compose -p "$PROJECT" -f compose.yaml)
MLFLOW_CONTAINER="${PROJECT}_mlflow_1"

# IDENTITY HEADER (p5, defect #5-fix-2): container names were hardened
# (per-worktree hash, now per-run entropy too) but the LOG was not, and the
# log is what a reviewer reports from. MEASURED 2026-08-14/15: two agents'
# `/tmp/container-test-round-c-N.log` artifacts (43694 and 6067 bytes)
# coexisted under UNNAMESPACED shared filenames, and a report built from one
# described a DIFFERENT worktree's run in every container name, phase
# result, and timing — caught only because one agent happened to check its
# own derived project name against the log body. This script cannot control
# what an external harness names its captured-output file (grep across this
# repo found no such naming here — it is orchestration-layer, out of this
# phase's scope), but it CAN make the log's own CONTENT self-identifying, so
# that exact cross-check is a one-line grep instead of a rediscovery. sha256
# (not a weaker checksum) of the files this run's conclusion actually
# depends on: this script, its extracted project-identity derivation, and
# the compose file that defines every container/healthcheck bound quoted
# above.
say "identity: cwd=$(pwd) toplevel=${TOPLEVEL}"
say "identity: project=${PROJECT} family=${PROJECT_FAMILY}"
say "identity: sha256 of the files this run's conclusion depends on:"
for f in "${SCRIPT_DIR}/run_container_test.sh" "${SCRIPT_DIR}/derive_compose_project.sh" compose.yaml; do
  printf '[container-test]   %s\n' "$(shasum -a 256 "$f")"
done

# Takes the exit status EXPLICITLY (as $1) rather than reading `$?` itself,
# because this is now called twice: once from the EXIT trap (where `$?` is
# the right source) and once DELIBERATELY right after phase 1's pytest call
# succeeds, to tear the main stack down before phase 2 starts (see the
# module header on why concurrent stacks are a contention hazard, not just
# a resource nicety).
cleanup_main_stack() {
  local exit_status="$1"
  if [[ "$exit_status" -eq 0 ]]; then
    say "success — tearing down the mlflow/postgres stack, including volumes..."
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
  # Reap healthcheck timers podman left behind for the containers just removed.
  # BOTH exit paths, because an orphan outlives a red run exactly as happily as
  # a green one — and a red run is the more likely to have left one. Never fails
  # the gate: `|| true` plus the script's own exit 0.
  bash "$(dirname "${BASH_SOURCE[0]}")/reap_healthcheck_orphans.sh" || true
}
trap 'cleanup_main_stack $?' EXIT

# WAIT FOR THE VM TO SETTLE AT EVERY PHASE SEAM — and note what this is NOT:
# it is not a bigger timeout. The phases' 120s health bounds are not too small
# on a quiet machine; the problem is that each phase starts on a machine that
# is BY CONSTRUCTION at its busiest. For phases 2/3 the producer is tearing
# down the PRIOR phase (stopping containers, removing a network and a volume);
# for phase 1 it is `just check` under ship-it, which saturates every host
# core with pytest immediately before this script boots the main stack.
# Called at all three seams — before phase 1's `up --build`, after phase 1's
# teardown, and after phase 2's own (fixture-owned) teardown, before phase 3 —
# same cause, same fix, at every seam. (MEASURED 2026-08-16: the red run's
# between-phase settles logged load1 7.97/8 and 7.58/8 — the same saturated
# window phase 1 had already booted into unguarded.)
#
# MEASURED on this 4-vCPU VM, one stack at a time, identical image and compose
# file — time from `up` to the mlflow container reporting healthy:
#
#     VM settled (load1 0.86)      29.64s
#     VM at load1 4.4             135.71s      <-- past the 120s bound
#
# So the bound has NO margin at the exact moment a phase uses it, and the gate
# reported that as a code failure three separate times today. Waiting for the
# cause to clear is the fix; raising the bound would only move the cliff (see
# CLAUDE.md House rule 3, and sinks.py on why bigger numbers are not fixes).
#
# Never fails the gate: settling is best-effort. If the machine will not settle
# it says so and proceeds, so a genuinely busy host degrades to today's
# behaviour rather than to a hang.
settle_podman_vm() {
  # Measure the HOST, not the guest. The podman VM's vCPUs are time-sliced by
  # the hypervisor against macOS, so a guest that looks idle is still starved
  # when the host is oversubscribed — and the guest's own load average reflects
  # that starvation rather than causing it. MEASURED 2026-08-14 while this gate
  # was failing: guest reported 1 runnable process out of 193 and 3231MB free,
  # 0 containers, 1 network — completely clean — while the 8-core host sat at
  # load 42.86 under an unrelated Rust build and a Codex process. Polling the
  # guest would have declared that machine settled.
  #
  # Normalised per core, because a load of 8 means something different on 8
  # cores than on 2. Threshold is HALF a runnable process per core.
  #
  # RETRACTED 2026-08-16 (was: one runnable process per core — "at or below
  # that the VM gets a fair share"). That claim failed at its own boundary:
  # this gate declared "host settled" at load1 7.97/8 before phase 2 and
  # 7.58/8 before phase 3, then phase 3's single-worker MLflow (behind Caddy,
  # inside the 4-vCPU VM on this 8-core host) missed its clients' 10s read
  # deadlines until retries exhausted — tests/test_caddy_auth.py red on its
  # zero-failed-writes assertion. Starting a phase at 95-99.6% of saturation
  # is not a settled machine; the healthy measurement above (29.64s at load1
  # 0.86) had real headroom, not boundary compliance. cores/2 demands that
  # headroom. Computed as a float so a 1-core host gets 0.5, not
  # integer-division 0 (which `load1 < target` could never satisfy).
  local cores
  cores="$(sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"
  local half_cores
  half_cores="$(awk -v c="$cores" 'BEGIN{print c / 2}')"
  local target="${LIVE_GATE_SETTLE_LOAD:-$half_cores}"
  local waited=0
  local load1=""
  while [[ "$waited" -lt 180 ]]; do
    load1="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')"
    if [[ -z "$load1" ]]; then
      say "could not read host load (non-macOS?) — continuing without settling"
      return 0
    fi
    if awk -v a="$load1" -v b="$target" 'BEGIN{exit !(a<b)}'; then
      say "host settled (load1 ${load1} < ${target} on ${cores} cores) after ${waited}s — starting the next phase"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  say "WARNING: host still busy (load1 ${load1} vs ${cores} cores) after ${waited}s — starting the next phase anyway; a timing failure below is the MACHINE, not the code. Check \`ps aux -r | head\` for what is competing."
  return 0
}

# Seam 1 of 3: under ship-it, `just check` has just saturated every host core.
settle_podman_vm

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

say "phase 1/3: running the main integration suite against ${MLFLOW_URL}..."
export MLFLOW_TRACKING_URI="$MLFLOW_URL"
export ENTROPIC_MLFLOW_TINY=1
# --ignore: tests/test_mlflow_allowed_hosts.py is phase 2 and
# tests/test_caddy_auth.py is phase 3 (p4), both run below AFTER this stack
# is torn down — see the module header comment on why no two of the three
# ever run concurrently. Without both ignores here, `-m integration` would
# sweep phase 2 and phase 3's own modules into THIS pytest process, and their
# fixtures would try to bring up a SECOND/THIRD live stack while phase 1's
# is still up — assert_no_foreign_mlflow_stack() correctly refuses that
# (MEASURED: this was the failure mode the first time
# tests/test_caddy_auth.py was added without updating this ignore list).
PYTHONPATH=src uv run pytest -m integration -q --disable-warnings \
  --ignore=tests/test_mlflow_allowed_hosts.py \
  --ignore=tests/test_caddy_auth.py

# Tear the main stack down NOW, deliberately, rather than waiting for the
# EXIT trap — phase 2 must never see this stack still up (see the module
# header's measured contention finding).
cleanup_main_stack 0

# Seam 2 of 3: phase 2 must not start on the machine phase 1's teardown just
# made busy. Definition and measurements live above phase 1's `up`.
settle_podman_vm

# Re-arm the EXIT trap to cover phase 2's AND phase 3's stacks — a BACKUP
# net, not either phase's primary cleanup. Both tests/test_mlflow_allowed_hosts.py
# and tests/test_caddy_auth.py own `down -v --remove-orphans` in their own
# pytest fixture's `finally`, which covers every normal Python-level exit
# path (return, raised exception, KeyboardInterrupt — SIGINT's default
# handler raises a catchable exception). What a `finally` block does NOT
# cover is this SCRIPT receiving SIGTERM (e.g. a CI job cancellation) while
# the pytest subprocess is mid-run: SIGTERM's default disposition
# terminates the process without running Python-level `finally` blocks.
# Re-arming this script's own EXIT trap ONCE, here, ahead of both remaining
# phases, closes that gap at the shell level for both — `down -v
# --remove-orphans` against a stack a fixture already tore down cleanly is a
# documented no-op (same idiom each fixture itself relies on for a stack
# that never fully came up), so this never double-reports a teardown that
# already happened, and it costs nothing to leave armed across the phase 3
# boundary too.
ALLOWLIST_PROJECT="${PROJECT}-allowlist"
CADDY_PROJECT="${PROJECT}-caddy"
cleanup_backup_stacks() {
  podman compose -p "$ALLOWLIST_PROJECT" -f compose.yaml down -v --remove-orphans \
    >/dev/null 2>&1 || true
  # --profile proxy: `down` removing a profile-gated service's resources
  # does not strictly require the flag (compose tears down everything the
  # PROJECT owns regardless of profile), but passing it mirrors the `up`
  # invocation that created this stack and costs nothing extra.
  podman compose -p "$CADDY_PROJECT" -f compose.yaml --profile proxy down -v --remove-orphans \
    >/dev/null 2>&1 || true
}
trap cleanup_backup_stacks EXIT

say "phase 2/3: running the host-allowlist integration suite (own compose stack, own project name — see tests/test_mlflow_allowed_hosts.py)..."
PYTHONPATH=src uv run pytest tests/test_mlflow_allowed_hosts.py -m integration -q --disable-warnings

# Same cause as the phase-1 -> phase-2 seam above: phase 2's own fixture just
# tore its stack down (down -v --remove-orphans, inside its `finally`), which
# is precisely what makes the VM busiest right as phase 3 is about to start.
settle_podman_vm

say "phase 3/3: running the Caddy-fronted auth integration suite (own compose stack + --profile proxy, own project name — see tests/test_caddy_auth.py)..."
PYTHONPATH=src uv run pytest tests/test_caddy_auth.py -m integration -q --disable-warnings
