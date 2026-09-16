"""Static contracts for the deployment dependency boundaries.

Two deploy surfaces share this module: the RunPod/H100 training image
(Containerfile) and the observability sidecar images the tracking stack
deploys from (Containerfile.mlflow, Containerfile.caddy).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from helpers import load_compose, mlflow_command

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict[str, Any]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)


def _dependency_names(entries: list[str | dict[str, str]]) -> set[str]:
    # Split at the FIRST PEP 508 version-specifier/extras/URL character
    # (`[`, `<`, `>`, `=`, `!`, `~`, `@`). A plain `.split("=", 1)` (the prior
    # shape here) mis-parsed any bare `>=`/`<=`/`~=` entry with no `[extras]`
    # to shield it — e.g. "psutil>=6.0.0" -> "psutil>" — because ">" was never
    # a split character; "dvc[s3]>=3.59.0" only ever passed because its `[`
    # split ran first and discarded the `>=` along with the extras.
    return {
        re.split(r"[\[<>=!~@]", entry, maxsplit=1)[0].strip().lower()
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
    # The base image's system Python 3.12 is Debian externally-managed
    # (PEP 668): the FIRST real build of this file (Docker Build Cloud,
    # native amd64, 2026-08-17) failed exit-2 at the dependency layer with
    # uv refusing `--system` against the marker. ENV form (not a per-RUN
    # flag) so both install layers inherit the opt-out and cannot drift.
    assert "ENV UV_BREAK_SYSTEM_PACKAGES=1" in containerfile
    assert (
        "COPY --from=ghcr.io/astral-sh/uv:0.11.6@sha256:"
        "43cb71695fcad1516c2fbe0f56e500184c42d8bce838d9f64593b8aff2c16298 "
        "/uv /uvx /bin/"
    ) in containerfile
    assert 'pip install --no-cache-dir "uv' not in containerfile


def test_runpod_image_contains_only_committed_dvc_remote_topology() -> None:
    """The pod must discover R2 through the same credential-free DVC config as Git."""
    containerfile = (REPO_ROOT / "Containerfile").read_text(encoding="utf-8")
    dvc_config = (REPO_ROOT / ".dvc/config").read_text(encoding="utf-8")
    active_instructions = {
        line.strip()
        for line in containerfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "COPY .dvc/config ./.dvc/config" in active_instructions
    assert "remote = r2-entropic-substrate-lm" in dvc_config
    assert "url = s3://entropic-substrate-lm" in dvc_config
    assert "endpointurl = https://" in dvc_config
    for forbidden in ("access_key_id", "secret_access_key", "credentialpath"):
        assert forbidden not in dvc_config.lower()


def test_deploy_group_pins_system_metrics_dependencies() -> None:
    """MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING (entropic_substrate.sinks) is
    mlflow-native, but mlflow-skinny does NOT declare psutil or pynvml as a
    dependency (verified 2026-08-17 via ``importlib.metadata.requires``) —
    without an explicit pin, GPU/CPU system metrics silently no-op for want
    of an import rather than failing loud on a training pod. Training-pod-only
    observability belongs in ``deploy``, not ``dev``: the unit suite injects a
    fake monitor factory (tests/test_sinks.py) and never imports either.
    """
    pyproject = _pyproject()
    groups = pyproject["dependency-groups"]
    deploy_names = _dependency_names(groups["deploy"])

    assert "psutil" in deploy_names
    assert "pynvml" in deploy_names


def test_deployment_and_analysis_are_audited_runtime_capabilities() -> None:
    pyproject = _pyproject()
    groups = pyproject["dependency-groups"]
    assert "boto3" in _dependency_names(groups["deploy"])
    assert "boto3" not in _dependency_names(groups["dev"])
    assert "scipy" in _dependency_names(groups["deploy"])
    assert "scipy" in _dependency_names(groups["analysis"])
    audit = pyproject["tool"]["deptry"]
    assert set(audit["non_dev_dependency_groups"]) == {"deploy", "analysis"}
    assert "DEP004" not in audit.get("ignore", [])
    assert audit["per_rule_ignores"] == {"DEP002": ["dvc", "psutil"]}
    assert "uv run deptry src notebooks" in (REPO_ROOT / "justfile").read_text()


def test_analysis_group_is_dev_only_and_absent_from_deploy() -> None:
    """matplotlib/marimo/plotly render interpretation charts (entropic_substrate.
    analysis, notebooks/) but must never grow the training-pod image: mirrors
    ``test_deploy_group_pins_system_metrics_dependencies``'s idiom, inverted —
    these three belong in a dev-side ``analysis`` group ONLY, included by
    ``dev`` so the unit suite can import them, and absent from ``deploy`` AND
    from ``[project.dependencies]`` (the training-pod runtime).
    """
    pyproject = _pyproject()
    groups = pyproject["dependency-groups"]
    analysis_names = _dependency_names(groups["analysis"])
    deploy_names = _dependency_names(groups["deploy"])
    project_names = _dependency_names(pyproject["project"]["dependencies"])

    for name in ("matplotlib", "marimo", "plotly"):
        assert name in analysis_names, f"{name} must be pinned in the analysis group"
        assert name not in deploy_names, f"{name} must never enter the deploy group"
        assert name not in project_names, f"{name} must never enter the runtime deps"

    assert {"include-group": "analysis"} in groups["dev"]

    containerfile = (REPO_ROOT / "Containerfile").read_text(encoding="utf-8")
    assert "uv export --frozen --no-dev --group deploy" in containerfile
    for name in ("matplotlib", "marimo", "plotly"):
        assert name not in containerfile


def test_live_stack_pins_a_single_mlflow_worker() -> None:
    """A worker pool is what made the live gate flaky; keep the server single-process.

    MLflow's default worker count has every uvicorn worker independently import the
    full FastAPI/SQLAlchemy/Alembic stack. On a 4-vCPU podman VM they contend, the
    supervisor kills whichever workers overrun its startup timeout, and startup
    stretches unboundedly under load — measured at 41s with repeated ``Child process
    [N] died`` versus 8s and zero deaths at ``--workers 1``. Concurrency is not under
    test in this stack (one pytest process drives it), so the pool buys nothing.
    """
    command = mlflow_command()
    assert "--workers" in command, "compose.yaml must pin an explicit MLflow worker count"
    assert command[command.index("--workers") + 1] == "1"


# The allowlist FLAG-SHAPE test that used to live here is deliberately gone,
# not relocated. It pinned an `sh -c` script that no longer exists: the
# allowlist now reaches MLflow through MLFLOW_SERVER_* environment variables,
# so there is no flag whose optionality needs guarding. tests/
# test_compose_config.py owns that contract in full — single owner, which also
# retires a genuinely weak assertion this module carried. Its
# `"${MLFLOW_ALLOWED_HOSTS:+--allowed-hosts" in script` check was ALSO a
# substring of the correctly-escaped `"$${MLFLOW_ALLOWED_HOSTS:+..."` form, so
# it would have kept passing had the command reverted to the injectable
# single-`$` shape — inviting a future reader to trust the wrong test as the
# escaping guard. This module keeps only what it is uniquely responsible for:
# the worker count above and the image/dependency pins below.


def test_mlflow_sidecar_matches_the_skinny_client_release() -> None:
    """Keep the isolated full server on the same protocol release as its client."""
    pyproject = _pyproject()
    dependencies = pyproject["project"]["dependencies"]
    client_spec = next(entry for entry in dependencies if entry.startswith("mlflow-skinny=="))
    client_version = client_spec.partition("==")[2]

    sidecar = (REPO_ROOT / "Containerfile.mlflow").read_text(encoding="utf-8")
    assert f"FROM ghcr.io/mlflow/mlflow:v{client_version}" in sidecar
    assert "psycopg2-binary boto3" in sidecar


def test_mlflow_sidecar_cmd_is_flat_exec_form_with_no_flags() -> None:
    """A deploy platform runs the sidecar image's own CMD; the local stack never does.

    compose.yaml's ``command:`` overrides CMD outright (the base image ships
    ``ENTRYPOINT []`` — nothing wraps around it), so this CMD exists for exactly
    one consumer: a platform that runs the image as-built. Exec form with NO
    flags: in the pinned release every server option this deployment needs binds
    to a native env var via click ``envvar=`` (mlflow/utils/cli_args.py —
    MLFLOW_HOST / MLFLOW_PORT / MLFLOW_WORKERS / MLFLOW_ARTIFACTS_DESTINATION /
    MLFLOW_SERVER_ALLOWED_HOSTS / MLFLOW_SERVER_CORS_ALLOWED_ORIGINS;
    mlflow/cli/__init__.py — MLFLOW_BACKEND_STORE_URI), so deploy configuration
    stays a pure data channel — no shell form, no interpolation site, the same
    idiom compose.yaml's flat argv already pins for the local stack.
    """
    sidecar = (REPO_ROOT / "Containerfile.mlflow").read_text(encoding="utf-8")
    cmds = [line.strip() for line in sidecar.splitlines() if line.strip().startswith("CMD")]
    assert cmds == ['CMD ["mlflow", "server"]'], (
        f"Containerfile.mlflow must declare exactly one flat exec-form CMD with "
        f"no flags — configuration reaches the server via env vars only; got {cmds!r}"
    )


def test_caddy_containerfile_bakes_only_the_caddyfile_from_the_compose_pinned_image() -> None:
    """Deploy platforms build images; they do not bind-mount files. The Caddyfile
    therefore ships inside a dedicated image — same pinned base as the local
    compose service (one Caddy version, two venues) — carrying NOTHING but the
    config: credentials arrive from the environment at boot (the Caddyfile's
    ``{$VAR}`` placeholders), never from a layer, and the base image's own
    ``caddy run --config /etc/caddy/Caddyfile`` stays the run contract.
    """
    caddy_image = load_compose()["services"]["caddy"]["image"]
    containerfile = (REPO_ROOT / "Containerfile.caddy").read_text(encoding="utf-8")
    lines = [
        stripped
        for line in containerfile.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]
    assert lines and lines[0] == f"FROM {caddy_image}", (
        f"Containerfile.caddy must build FROM the exact image compose.yaml pins "
        f"({caddy_image}); got {lines[:1]!r}"
    )
    copies = [line for line in lines if line.startswith("COPY")]
    assert copies == ["COPY Caddyfile /etc/caddy/Caddyfile"], (
        f"the ONLY COPY must be the Caddyfile itself — no credentials, no extra "
        f"files in the layer; got {copies!r}"
    )
    overrides = [line for line in lines if line.startswith(("CMD", "ENTRYPOINT"))]
    assert not overrides, (
        f"must not override the base image's `caddy run` contract; got {overrides!r}"
    )


def test_railway_config_as_code_pins_per_service_build_and_watch_paths() -> None:
    """One repo feeds two Railway services; each needs its OWN build recipe.

    A single root railway.json cannot express two dockerfilePaths, so each
    service carries a dedicated config file (its path is set once in the
    service's settings). watchPatterns keep GitHub-source deploys scoped: a
    push touching only src/ must not rebuild the proxy, and vice versa. The
    dockerfilePath here must agree with the RAILWAY_DOCKERFILE_PATH variable
    each service already carries — config-as-code and env must never diverge.
    """
    import json as _json

    mlflow_cfg = _json.loads((REPO_ROOT / "railway.mlflow.json").read_text(encoding="utf-8"))
    assert mlflow_cfg["build"]["builder"] == "DOCKERFILE"
    assert mlflow_cfg["build"]["dockerfilePath"] == "Containerfile.mlflow"
    assert "Containerfile.mlflow" in mlflow_cfg["build"]["watchPatterns"]

    caddy_cfg = _json.loads((REPO_ROOT / "railway.caddy.json").read_text(encoding="utf-8"))
    assert caddy_cfg["build"]["builder"] == "DOCKERFILE"
    assert caddy_cfg["build"]["dockerfilePath"] == "Containerfile.caddy"
    for watched in ("Caddyfile", "Containerfile.caddy"):
        assert watched in caddy_cfg["build"]["watchPatterns"], (
            f"caddy service must redeploy when {watched} changes"
        )


def test_railwayignore_keeps_secrets_and_bulk_out_of_the_upload_context() -> None:
    """``railway up`` ships the working directory to a remote builder.

    ``.env`` holds live R2/HF/RunPod credentials and MUST never ride along —
    gitleaks guards the git history, but an upload context is a separate
    egress path the hooks never see. The bulk entries (venv, DVC-tracked
    data/checkpoints, reference submodules) are excluded because neither
    deploy image COPYs anything beyond the Caddyfile, so shipping gigabytes
    of context buys nothing and slows every deploy.
    """
    entries = (REPO_ROOT / ".railwayignore").read_text(encoding="utf-8").split()
    for required in (".env", ".venv", "data", "models", "results", "references"):
        assert required in entries, (
            f".railwayignore must exclude {required!r} from the `railway up` "
            f"upload context; entries found: {entries!r}"
        )


def test_dockerignore_matches_containerignore_byte_for_byte() -> None:
    """podman reads .containerignore; the docker CLI reads ONLY .dockerignore.

    The Build Cloud lane (`docker buildx build --builder cloud-...`) uploads
    the context to a REMOTE builder, so without parity it ships everything
    .containerignore exists to exclude — ``.env`` included. One contract, two
    filenames: a symlink satisfies this test, drift fails it.
    """
    container = (REPO_ROOT / ".containerignore").read_text(encoding="utf-8")
    docker = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert docker == container, (
        ".dockerignore must be byte-identical to .containerignore (symlink it) — "
        "docker-CLI builds otherwise upload a context the containerignore was "
        "written to prevent"
    )
    assert ".env" in container.split(), ".containerignore must exclude .env"


def test_baked_model_copies_carry_a_matching_containerignore_negation() -> None:
    """E1.5c's frozen arm ships its p_phi weights INSIDE the image.

    Pods are credential-free (no R2 keys), so the frozen field checkpoint
    cannot be pulled at runtime — the preregistration
    (docs/e1-5c-preregistration.md Pilots) delivers it baked in, which is
    also what puts the field weights under the image digest the launch
    record pins.

    `models/` is excluded wholesale from the build context, so every baked
    model path needs its own `!`-negation. The two must move together: a
    COPY without its negation fails the BUILD (context missing), while a
    negation without its COPY silently fattens the context. This pins the
    pair rather than either half.
    """
    containerfile = (REPO_ROOT / "Containerfile").read_text(encoding="utf-8")
    ignore_lines = (REPO_ROOT / ".containerignore").read_text(encoding="utf-8").split()
    copied = [
        line.split()[1]
        for line in containerfile.splitlines()
        if line.startswith("COPY ") and line.split()[1].startswith("models/")
    ]
    assert copied, "expected the frozen p_phi checkpoint to be COPYed into the image"
    for path in copied:
        assert f"!{path}" in ignore_lines, (
            f"Containerfile COPYs {path!r} but .containerignore has no "
            f"'!{path}' negation — `models/` excludes it from the build context"
        )


def test_training_image_invokes_the_pinned_uv_copy_not_a_path_shadow() -> None:
    containerfile = (Path(__file__).resolve().parents[1] / "Containerfile").read_text()
    assert "RUN /bin/uv export --frozen" in containerfile
    assert "&& /bin/uv pip install --system" in containerfile
    assert "RUN /bin/uv pip install --system --no-cache --no-deps ." in containerfile
