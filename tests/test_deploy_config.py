"""Static contracts for the RunPod/H100 deployment dependency boundary."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

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


def test_mlflow_sidecar_matches_the_skinny_client_release() -> None:
    """Keep the isolated full server on the same protocol release as its client."""
    pyproject = _pyproject()
    dependencies = pyproject["project"]["dependencies"]
    client_spec = next(entry for entry in dependencies if entry.startswith("mlflow-skinny=="))
    client_version = client_spec.partition("==")[2]

    sidecar = (REPO_ROOT / "Containerfile.mlflow").read_text(encoding="utf-8")
    assert f"FROM ghcr.io/mlflow/mlflow:v{client_version}" in sidecar
    assert "psycopg2-binary boto3" in sidecar
