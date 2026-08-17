set shell := ["bash", "-cu"]

import? 'justfile.local'

# List all recipes (default target)
default:
    @just --list --unsorted

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
    uv run deptry src
    uv run bandit -c pyproject.toml -q -r src

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Fast hermetic unit tests only (CPU, <60s total).
test:
    @echo "→ Running unit tests..."
    PYTHONPATH=src uv run pytest -m unit -q --maxfail=1 --disable-warnings

# All tests (unit + integration).
test-all:
    @echo "→ Running full test suite..."
    PYTHONPATH=src uv run pytest -q --maxfail=1 --disable-warnings

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------

# Standard quality gate: fmt-check + lint + typecheck + unit tests.
# This is the gate CI runs on every push/PR and prek runs pre-push.
check: fmt-check lint typecheck test
    @echo "✓ All quality checks passed"

# CI parity gate — runs prek hooks (commit + push stages).
pc-ci:
    @echo "→ Running prek hooks on changed files..."
    prek run
    prek run --stage pre-push
    @echo "✓ All prek gates passed"

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
    PYTHONPATH=src uv run python -m entropic_substrate.cli train --exp {{exp}} {{ARGS}}

# Gated ladder E1 → E2 → E3 (stops with DECISION-NEEDED before the anchored
# arms / E4). `just ladder --tiny` = CPU smoke of the whole ladder in minutes.
ladder *ARGS:
    PYTHONPATH=src uv run --no-sync bash scripts/run_ladder.sh {{ARGS}}

# Evaluate a checkpoint, e.g.: just eval ckpt=models/e3-latest.pt
eval ckpt *ARGS:
    PYTHONPATH=src uv run python -m entropic_substrate.cli eval --ckpt {{ckpt}} {{ARGS}}

# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

# Build the training container image (single-stage Containerfile — see file header).
# platform defaults to linux/amd64: GPU pods are x86_64, and a bare build on
# Apple Silicon would produce an arm64 image they cannot run (qemu makes the
# cross-build slower but correct).

# Build the training container image (defaults to linux/amd64 for GPU pods)
container-build tag="entropic-substrate-lm:latest" platform="linux/amd64":
    @echo "→ Building container image {{tag}} for {{platform}}..."
    podman build --platform {{platform}} -t {{tag}} -f Containerfile .

# LIVE gate for phase p2-mlflow: bring up postgres + mlflow (compose.yaml,
# native arm64, real R2 artifacts), run the integration suite against it,
# tear down. See scripts/run_container_test.sh for the orchestration.

# LIVE gate: real MLflow + Postgres + R2 stack, integration suite, teardown
container-test:
    bash scripts/run_container_test.sh

# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

# Full pre-flight, then submit the whole Graphite stack as drafts.
#
# `container-test` is a HARD dependency, not a courtesy. Nothing else in this
# repo executes the live MLflow path:
#
#   prek pre-commit  `pytest -m unit`  no containers
#   prek pre-push    `pytest -q`       no containers, AND
#                                      tests/test_mlflow_live.py skipif's on
#                                      MLFLOW_TRACKING_URI, so with no stack up
#                                      every live test SILENTLY SKIPS —
#                                      reporting success by testing nothing
#   CI               `just check`      no containers
#
# Because those tests SKIP rather than FAIL, an ungated push is green while
# covering nothing. That is exactly how the live gate went red at main (deps
# commit 81bd4dd, mlflow -> mlflow-skinny) and stayed red under a fully green
# CI. See CLAUDE.md's House rules.
#
# Deliberately NOT provided: a `ship-it-fast` that skips the live gate. An
# official bypass is how a gate stops being a gate. If a branch genuinely
# cannot run containers, call `gt submit` directly and own that choice.

# Deliberately NO `--ai`: it would REGENERATE the PR title from the diff,
# blind to gitlint/commitizen, and the PR title is what GitHub bakes into the
# permanent squash-merge subject (`<title> (#N)`). That is how PRs #32/#33
# landed code-fenced subjects that turned the pre-push gate red. Without it,
# gt uses the first commit's subject — already generated under the rules by
# `just stack`. Pinned by tests/test_pr_authoring.py.

# Gate (check + live container-test) then submit the Graphite stack as drafts
ship-it: check container-test
    gt submit --stack --no-edit --no-interactive --always
