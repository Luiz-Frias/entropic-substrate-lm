"""Static contracts for the RunPod/H100 deployment dependency boundary."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from helpers import mlflow_command_script

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict[str, Any]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def _dependency_names(entries: list[str | dict[str, str]]) -> set[str]:
    return {
        entry.split("@", 1)[0].split("[", 1)[0].split("=", 1)[0].strip().lower()
        for entry in entries
        if isinstance(entry, str)
    }


def test_cuda_base_and_project_pin_the_same_torch_release() -> None:
    """Prevent dependency resolution from replacing the base image's CUDA wheel."""
    pyproject = _pyproject()
    dependencies = pyproject["project"]["dependencies"]
    torch_specs = [entry for entry in dependencies if entry.startswith("torch")]
    assert len(torch_specs) == 1

    containerfile = (REPO_ROOT / "Containerfile").read_text(encoding="utf-8")
    base_match = re.search(r"^FROM pytorch/pytorch:(\d+\.\d+\.\d+)-", containerfile, re.MULTILINE)
    assert base_match is not None
    assert torch_specs[0] == f"torch=={base_match.group(1)}"
    assert tuple(int(part) for part in base_match.group(1).split(".")) >= (2, 13, 0)


def test_runpod_dependency_layer_is_frozen_and_includes_deploy_tools() -> None:
    """The image must install DVC from the audited lock without test-only Geoopt."""
    pyproject = _pyproject()
    groups = pyproject["dependency-groups"]

    assert "dvc" in _dependency_names(groups["deploy"])
    assert {"include-group": "deploy"} in groups["dev"]
    assert "geoopt" not in _dependency_names(groups["deploy"])

    containerfile = (REPO_ROOT / "Containerfile").read_text(encoding="utf-8")
    assert "COPY pyproject.toml uv.lock ./" in containerfile
    assert "uv export --frozen --no-dev --group deploy" in containerfile
    assert "--prune torch" in containerfile
    assert (
        "COPY --from=ghcr.io/astral-sh/uv:0.11.6@sha256:"
        "43cb71695fcad1516c2fbe0f56e500184c42d8bce838d9f64593b8aff2c16298 "
        "/uv /uvx /bin/"
    ) in containerfile
    assert 'pip install --no-cache-dir "uv' not in containerfile


def test_live_stack_pins_a_single_mlflow_worker() -> None:
    """A worker pool is what made the live gate flaky; keep the server single-process.

    MLflow's default worker count has every uvicorn worker independently import the
    full FastAPI/SQLAlchemy/Alembic stack. On a 4-vCPU podman VM they contend, the
    supervisor kills whichever workers overrun its startup timeout, and startup
    stretches unboundedly under load — measured at 41s with repeated ``Child process
    [N] died`` versus 8s and zero deaths at ``--workers 1``. Concurrency is not under
    test in this stack (one pytest process drives it), so the pool buys nothing.
    """
    entries = mlflow_command_script().split()
    assert "--workers" in entries, "compose.yaml must pin an explicit MLflow worker count"
    assert entries[entries.index("--workers") + 1] == "1"


def test_live_stack_host_allowlist_is_optional_and_never_wildcard() -> None:
    """The p3 allowlist knobs must be OPTIONAL FLAGS (absent when unset —
    not present-with-an-empty-value; --cors-allowed-origins "" measurably
    HANGS the pinned mlflow image, see compose.yaml's own comment).

    What this test PINS is compose.yaml's own TEXT: no unconditional flag,
    no literal '*' written into the command as a default. It does NOT (and,
    parsing static YAML, cannot) prove an operator's runtime
    MLFLOW_ALLOWED_HOSTS/MLFLOW_CORS_ALLOWED_ORIGINS value is never '*' —
    that value arrives from the deploy environment at container-start time,
    after this test has already run. Enforcing that at runtime (a fail-fast
    guard inside compose.yaml's own sh -c script) is tracked separately, not
    pinned here.
    """
    script = mlflow_command_script()
    assert "${MLFLOW_ALLOWED_HOSTS:+--allowed-hosts" in script, script
    assert "${MLFLOW_CORS_ALLOWED_ORIGINS:+--cors-allowed-origins" in script, script
    # Neither flag may be present unconditionally (i.e. outside a `:+` guard)
    # — a bare `--allowed-hosts "${MLFLOW_ALLOWED_HOSTS:-}"` shape would pass
    # an empty string on the unset path, which is the exact footgun this
    # phase measured and rejected for --cors-allowed-origins.
    assert re.search(r"(?<!\+)--allowed-hosts \"\$\{MLFLOW_ALLOWED_HOSTS:-", script) is None
    assert (
        re.search(r"(?<!\+)--cors-allowed-origins \"\$\{MLFLOW_CORS_ALLOWED_ORIGINS:-", script)
        is None
    )
    assert "*" not in script, "compose.yaml must never WRITE a wildcard allowlist default"


def test_mlflow_sidecar_matches_the_skinny_client_release() -> None:
    """Keep the isolated full server on the same protocol release as its client."""
    pyproject = _pyproject()
    dependencies = pyproject["project"]["dependencies"]
    client_spec = next(entry for entry in dependencies if entry.startswith("mlflow-skinny=="))
    client_version = client_spec.partition("==")[2]

    sidecar = (REPO_ROOT / "Containerfile.mlflow").read_text(encoding="utf-8")
    assert f"FROM ghcr.io/mlflow/mlflow:v{client_version}" in sidecar
    assert "psycopg2-binary boto3" in sidecar
