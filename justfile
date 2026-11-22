set shell := ["bash", "-cu"]

import? 'justfile.local'

# Deliverable 10 (dreamy-questing-scroll.md, E-4 ruling): the dotted `python -m`
# entrypoint names are parameterized from ONE source, scripts/entrypoints.env,
# shared with run_ladder.sh and the worker allowlists in provider_launch.py /
# d1_runpod_adapter.py / h_probe_launch.py / d1_external_analysis_adapter.py
# (see patches/entrypoints-python.md for how those four read the same values).
# `set dotenv-load` + `set dotenv-path` load it into every recipe's environment
# AND make it visible to `env_var()` at parse time, so a missing key fails
# immediately (`just --evaluate` / any recipe invocation), never silently as an
# empty string.
set dotenv-load
set dotenv-path := "scripts/entrypoints.env"

cli := env_var("ENTRYPOINT_CLI")
telemetry := env_var("ENTRYPOINT_TELEMETRY")
h_worker := env_var("ENTRYPOINT_H_WORKER")

# List all recipes (default target)
default:
    @just --list --unsorted

# Local synthetic smoke only; no scientific gate or Runpod spending authority.
d-probe *ARGS:
    uv run python -m {{ cli }} d-probe {{ ARGS }}

# H numerical training only; full probe/scientific completion is a separate gate.
h-train-probe *ARGS:
    uv run python -m {{ cli }} h-train-probe {{ ARGS }}

h-inspect-training *ARGS:
    uv run python -m {{ cli }} h-inspect-training {{ ARGS }}

h-generate-probe *ARGS:
    uv run python -m {{ cli }} h-generate-probe {{ ARGS }}

h-inspect-generation *ARGS:
    uv run python -m {{ cli }} h-inspect-generation {{ ARGS }}

h-capture-model *ARGS:
    uv run python -m {{ cli }} h-capture-model {{ ARGS }}

h-inspect-model *ARGS:
    uv run python -m {{ cli }} h-inspect-model {{ ARGS }}

h-inspect-custody *ARGS:
    uv run python -m {{ cli }} h-inspect-custody {{ ARGS }}

h-diagnose-spans *ARGS:
    uv run python -m {{ cli }} h-diagnose-spans {{ ARGS }}

h-inspect-spans *ARGS:
    uv run python -m {{ cli }} h-inspect-spans {{ ARGS }}

h-worker *ARGS:
    uv run python -m {{ h_worker }} {{ ARGS }}

# e.g. just h-tiny-trace --profile <remote>/profile.json --corpus <capture> --out results/<dir>
# Re-seals the capture in this regime; JSONL sink always, MLflow when opted in.
# Local CPU tracer for a remote tiny H profile (no scientific or spending authority).
h-tiny-trace *ARGS:
    uv run python -m entropic_substrate.composition.wiring.h_tiny_tracer {{ ARGS }}

# e.g. just h-solver-preflight --profile results/<attempt>/profile.json --corpus <pod>/prepared-corpus.json
# Replays prediction step 1 on CPU; exit 0 PASS, 3 REFUSE; verdict JSON beside the profile.
# CPU solver-budget preflight that gates every H pod launch (no scientific or spending authority).
h-solver-preflight *ARGS:
    PYTHONPATH=src uv run python tools/h_solver_preflight.py {{ ARGS }}

# e.g. just h-probe-stop --launch-root <dir> --launch-file launch.json --repo-root . --max-launch-bytes 1000000 --wait-seconds 300 --poll-seconds 5
# The only sanctioned way to end an in-flight H probe run (FOR-1617): publishes
# the R2 stop marker, waits for the worker's `stopped` terminal, prints its keys.
h-probe-stop *ARGS:
    uv run python -m entropic_substrate.ladder.run.worker.h_probe_stop {{ ARGS }}

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Provision all mise tools, sync the Python venv, and install prek hooks.
setup:
    @echo "→ Installing mise-managed tools..."
    mise install
    @echo "→ Syncing the Python environment with uv..."
    uv sync --all-groups
    @echo "→ Validating prek config..."
    uv run prek validate-config .pre-commit-config.yaml
    @echo "→ Installing prek hooks..."
    prek install
    prek install-hooks
    @echo "✓ Development environment ready"

# Sync the Python venv (faster than setup; skips tool install).
sync:
    uv sync --all-groups

# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

# Apply ruff formatting in-place.
fmt:
    @echo "→ Formatting..."
    uv run ruff format .

# Check formatting without modifying files (CI-safe).
fmt-check:
    @echo "→ Checking formatting..."
    uv run ruff format --check .

# ---------------------------------------------------------------------------
# Lint + typecheck
# ---------------------------------------------------------------------------

# Run ruff linter (auto-fix enabled for safe fixes).
lint:
    @echo "→ Linting..."
    uv run ruff check . --fix

# Run basedpyright strict type checking against src/ + scripts/ (mirrors
# pyproject [tool.basedpyright] include and the prek basedpyright hook).
typecheck:
    @echo "→ Type checking..."
    uv run basedpyright src scripts

# Dependency hygiene + security lint (deptry + bandit, configured in pyproject).
audit:
    @echo "→ Auditing dependencies and security..."
    uv run deptry src notebooks
    uv run bandit -c pyproject.toml -q -r src

# p0-contract-pin (D2/DN-P1): the layers contract over the LANDED layers
# kernel < core < ladder < historical < composition, `__legacy` deliberately
# OUTSIDE it until p13. `--no-sync`: a bare `uv run` re-syncs the env and
# would replace the CPU-only torch wheel CI pins with the CUDA build (see
# every other `--no-sync` step in ci.yaml for the same measured reason).
# Echoes the resolved package location so the scanned tree is evidenced, not
# assumed — a stale editable install pointing elsewhere would otherwise pass
# silently.
lint-imports:
    @echo "→ Checking import layers (kernel < core < ladder < historical < composition)..."
    @uv run --no-sync python -c "import entropic_substrate; print('scanning', entropic_substrate.__file__)"
    uv run --no-sync lint-imports

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Hermetic unit tests only (CPU). Wall time scales with physical cores: the
# "<60s" this comment used to claim was never measured here and is withdrawn.
# CI deliberately diverges from this line (ci.yaml unit step, FOR-1618,
# 2026-09-28): the ubuntu-latest runner reports psutil physical=1, so `-n auto`
# is ONE worker there and CI runs `-n logical` (2) plus OMP/MKL_NUM_THREADS=1
# for the subprocess children tests spawn, measured 23m01s -> 18m29s on the same
# host model. On a multi-core dev box `-n auto` is already N workers and the
# local regime is unmeasured, so the flags stay. tests/xdist_policy.py pins one
# torch thread per xdist worker on both.
test:
    @echo "→ Running unit tests..."
    PYTHONPATH=src uv run pytest -m unit -q --maxfail=1 --disable-warnings -n auto --dist loadfile

# Fast tier: the unit suite minus `slow`. The inner loop and the prek pre-commit
# stage run this; `check`, the pre-push hook, CI and `ship-it` all run the full
# `test` recipe above, and tests/repo/test_suite_tiers.py pins that none of them
# deselects `slow`.
# MEASURED 2026-11-21 at main 8c1f0c13: `slow` covers tests/architecture/test_cold_import.py
# (360 s) and 59 functions with a case at or above 5 s (633 s) out of 2,335
# summed test-seconds.
test-fast:
    @echo "→ Running fast unit tier (unit and not slow)..."
    PYTHONPATH=src uv run pytest -m "unit and not slow" -q --maxfail=1 --disable-warnings -n auto --dist loadfile

# All tests (unit + integration). Stays `--dist loadfile` on purpose: the
# integration files tests/live/test_caddy_auth.py and tests/live/test_mlflow_allowed_hosts.py
# bring up podman compose under ONE fixed project name, so any mode that splits a
# file across workers (worksteal) can run both at once and collide. worksteal and
# loadscope measured null for the unit suite anyway (FOR-1618 ledger).
test-all:
    @echo "→ Running full test suite..."
    PYTHONPATH=src uv run pytest -q --maxfail=1 --disable-warnings -n auto --dist loadfile

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

# Standard quality gate: fmt-check + lint + typecheck + unit tests + import layers.
# This is the gate CI runs on every push/PR and prek runs pre-push.
check: fmt-check lint typecheck test lint-imports
    @echo "✓ All quality checks passed"

# CI parity gate — runs prek hooks (commit + push stages).
# CI-parity gate on the BRANCH DIFF, never on the staged set. MEASURED 2026-09-18
# (sprint-preflight wf_f189da1b-7c8): plain `prek run` examines only staged files, so on
# a clean tree — and on every lifecycle lane after its commits — it ran zero hooks and
# reported PASS. A gate that examines nothing must refuse, not pass (measurement path
# fails LOUD). base = the phase's parent under a stacked sprint ($PC_CI_BASE), default
# origin/main; the diff is merge-base(HEAD, base)..HEAD, i.e. exactly what this branch
# added. An empty diff exits 2 instead of reporting a vacuous PASS.
# MEASURED 2026-09-28 (p14 rung pre-for-1567, L2 gate-check on e28c3f8): the former
# default `origin/dev` no longer exists (`git ls-remote --heads origin dev` is empty
# since dev was retired into main at v0.1.0), so the gate died at `git merge-base`
# with exit 128 before running a single hook, on the parent tip as well. A default
# that names a retired ref is a gate that cannot run; `origin/main` always exists.
# Under a stacked sprint the range vs main is a superset of the branch's own diff
# (the trunk's already-gated commits ride along), which is correct, only slower.
pc-ci base=env("PC_CI_BASE", "origin/main"):
    #!/usr/bin/env bash
    set -euo pipefail
    from_ref="$(git merge-base HEAD "{{base}}")"
    changed="$(git diff --name-only "${from_ref}..HEAD" | wc -l | tr -d ' ')"
    if [ "${changed}" -eq 0 ]; then
        echo "✗ pc-ci: empty diff vs {{base}} (${from_ref:0:7}..HEAD) — nothing examined, refusing to report PASS" >&2
        exit 2
    fi
    echo "→ Running prek hooks on ${changed} file(s) changed vs {{base}} (${from_ref:0:7}..HEAD)..."
    prek run --from-ref "${from_ref}" --to-ref HEAD
    prek run --stage pre-push --from-ref "${from_ref}" --to-ref HEAD
    # FOR-1536 (MEASURED 2026-09-21): prek range runs cover the pre-commit and
    # pre-push stages only; gitlint lives in the commit-msg stage, so a commit
    # made with hooks skipped reached submit with an 84-char title. Lint every
    # message in the range with the same tool the hook pins.
    # MEASURED 2026-09-22 (p4 ladder): the whole-range form flagged a commit already
    # published in the p3 PR (21d146d, B6), which no later phase may rewrite. The
    # gate exists to stop a bad message BEFORE it is pushed, so it lints exactly the
    # commits no remote ref can reach; an empty set is a pass, never an error.
    unpublished="$(git rev-list HEAD --not --remotes)"
    echo "→ gitlint over $(printf '%s\n' "${unpublished}" | grep -c . || true) unpublished commit(s)..."
    for sha in ${unpublished}; do uv run gitlint --commit "${sha}"; done
    echo "✓ All prek gates passed on the branch diff"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Download enwik8 into data/ (then `dvc add data/enwik8` to track it).
download-data:
    @echo "→ Downloading enwik8..."
    PYTHONPATH=src uv run python scripts/download_enwik8.py
    @echo "✓ Downloaded. Run: dvc add data/enwik8"

# ---------------------------------------------------------------------------
# Training / eval entrypoints
# ---------------------------------------------------------------------------

# Run a training experiment, e.g.: just train exp=e1-fisher
# Valid exp values: e0-byte, e0-bpe, e1-fisher, e1-random, e2, e3,
# e3-anchor, e3-vantage (anchored arms are execution-gated on the E3 readout)
train exp="tiny" *ARGS:
    PYTHONPATH=src uv run python -m {{ cli }} train --exp {{ exp }} {{ ARGS }}

# Gated ladder E1 → E2 → E3 (stops with DECISION-NEEDED before the anchored
# arms / E4). `just ladder --tiny` = CPU smoke of the whole ladder in minutes.
ladder *ARGS:
    PYTHONPATH=src uv run --no-sync bash scripts/run_ladder.sh {{ ARGS }}

# Evaluate a checkpoint, e.g.: just eval ckpt=models/e3-latest.pt
eval ckpt *ARGS:
    PYTHONPATH=src uv run python -m {{ cli }} eval --ckpt {{ ckpt }} {{ ARGS }}

# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

# Build the training container image (single-stage Containerfile — see file header).
# platform defaults to linux/amd64: GPU pods are x86_64, and a bare build on
# Apple Silicon would produce an arm64 image they cannot run.
#
# RETRACTION (2026-08-17): this comment previously claimed "qemu makes the
# cross-build slower but correct". MEASURED wrong on this repo's own build: the
# dependency layer's uv installer segfaults (exit 139) under qemu-user on the
# podman VM (rosetta=false), while the trivial `uv --version` under the same
# emulation passes — heavy multithreaded work is the trigger class, so the
# failure is load-dependent, not deterministic-per-binary. Local cross-builds
# are therefore NOT a supported publish lane; use `just container-publish`,
# which builds on a native amd64 Docker Build Cloud builder.

# Build the training container image locally (defaults to linux/amd64 for GPU pods)
container-build tag="entropic-substrate-lm:latest" platform="linux/amd64":
    @echo "→ Building container image {{ tag }} for {{ platform }}..."
    podman build --platform {{ platform }} -t {{ tag }} -f Containerfile .

# Build on a native amd64 Docker Build Cloud builder and push straight to GHCR
# (the image never transits the local uplink; the context upload respects
# .dockerignore — see tests/repo/test_deploy_config.py's parity pin). Needs a
# Docker Hub login (`docker login`) and the cloud-capable buildx from
# docker/buildx-desktop; the OSS Homebrew buildx lacks the `cloud` driver.
container-publish tag="ghcr.io/luiz-frias/entropic-substrate-lm:latest" builder="cloud-luizffrias-small-lang-model":
    docker buildx build --builder {{ builder }} --platform linux/amd64 -f Containerfile -t {{ tag }} --push .

# LIVE gate for phase p2-mlflow: bring up postgres + mlflow (compose.yaml,
# native arm64, real R2 artifacts), run the integration suite against it,
# tear down. See scripts/run_container_test.sh for the orchestration.

# LIVE gate: real MLflow + Postgres + R2 stack, integration suite, teardown
container-test:
    bash scripts/run_container_test.sh

# Logger smoke inside the digest-pinned image from docs/deploy-h100.md, with
# the CURRENT source mounted over the image's: the image supplies only its own
# interpreter and installed packages. One container, never beside another
# (house rule 2); `ship-it-full` runs it after `container-test` has torn down.
# A missing image is pulled; a failed pull or a down podman FAILS, never skips.
# MEASURED 2026-11-22 on Apple silicon (amd64 under qemu, podman VM 4 vCPU /
# 4 GB): this recipe 7.1 s with the image cached, both active paths PASS; the
# one-time pull was 250 s for 4.08 GB compressed.
logger-smoke-image:
    podman run --rm --platform linux/amd64 -v "{{ justfile_directory() }}/src:/app/src:ro" -v "{{ justfile_directory() }}/scripts:/app/scripts:ro" "$(uv run python scripts/logger_smoke.py image-ref)" python scripts/logger_smoke.py run --status active

# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

# Publish the Graphite stack as drafts. Two recipes since 2026-11-21:
#
#   ship-it        static checks, then `gt submit` (whose pre-push hook runs
#                  tests/unit + tests/e2e in full and reads the post-commit
#                  record), then `post-ship`. NO containers.
#   ship-it-full   `just check` + the live `container-test` gate + the logger
#                  smoke inside the pinned image, then submit, then `post-ship`.
#
# Nothing but `container-test` executes the live MLflow path: the tests under
# tests/live skipif on MLFLOW_TRACKING_URI, so with no stack up they SILENTLY
# SKIP, reporting success by testing nothing. That is exactly how the live
# gate went red at main (deps commit 81bd4dd, mlflow -> mlflow-skinny) and
# stayed red under a fully green CI. See CLAUDE.md's House rules.
#
# RETRACTED 2026-11-21 (operator ruling): this block used to say "Deliberately
# NOT provided: a `ship-it-fast` that skips the live gate. An official bypass
# is how a gate stops being a gate." The slim recipe now exists, and what keeps
# it from being a bypass is mechanical, not a habit: scripts/ship_scope.py
# reads the stack's diff and REFUSES (exit 3, naming the files and
# `just ship-it-full`) when it touches compose, the image definition, the
# dependency files, the MLflow adapter or sinks, the container gate,
# tests/live, or the logging substrate and the series emission paths. Pinned by tests/repo/test_ship_scope.py and
# tests/repo/test_suite_tiers.py. Not covered by any local recipe: a
# dependency PR opened by a bot never passes through either one.

# Deliberately NO `--ai`: it would REGENERATE the PR title from the diff,
# blind to gitlint/commitizen, and the PR title is what GitHub bakes into the
# permanent squash-merge subject (`<title> (#N)`). That is how PRs #32/#33
# landed code-fenced subjects that turned the pre-push gate red. Without it,
# gt uses the first commit's subject — already generated under the rules by
# `just stack`. Pinned by tests/repo/test_pr_authoring.py.

# Both recipes hold the host suite lock (tests/suite_lock.py) across every
# step: their suites run as this holder's descendants and pass through, while
# any other session's `-n` suite or ship-it refuses with exit 75 instead of
# racing. MEASURED 2026-10-30: a bare pre-push suite flaked a 10 s watchdog
# deadline at swap 96% / load 22 beside another worktree's suite; alone it
# passed 3/3. Steps are inside the held command, not recipe prerequisites,
# because prerequisites run before the body and so outside the lock.
ship-it:
    uv run python tests/suite_lock.py ship-it -- bash -c 'uv run python scripts/ship_scope.py && just fmt-check lint typecheck lint-imports && gt submit --stack --no-edit --no-interactive --always && just post-ship'

# The full gate: CI-parity check and the live MLflow stack, then submit. Run it
# whenever `just ship-it` refuses, and on demand for infra work.
ship-it-full:
    uv run python tests/suite_lock.py ship-it-full -- bash -c 'just check && just container-test && just logger-smoke-image && gt submit --stack --no-edit --no-interactive --always && just post-ship'

# After the submit: the folders the push did not run, slow tests included, then
# the dead-code WARNING report (never fails; candidates for a later refactor).
# A failure here is reported after the drafts are already up, by design.
post-ship:
    uv run python scripts/suite_stages.py run post-ship
    uv run python scripts/dead_code_report.py

# Run one stage of the suite by name (pre-commit, post-commit, pre-push,
# post-ship). The stage table is scripts/suite_stages.py; the hooks call the
# same entry, so this is exactly what a hook runs.
test-stage stage:
    uv run python scripts/suite_stages.py run {{ stage }}
