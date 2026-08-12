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

# Run basedpyright strict type checking against src/.
typecheck:
    @echo "→ Type checking..."
    uv run basedpyright src

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
container-build tag="entropic-substrate-lm:latest":
    @echo "→ Building container image {{tag}}..."
    podman build -t {{tag}} -f Containerfile .
