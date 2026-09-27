"""H0 deployment pins cannot become broader experiment or spending authority."""

import json
import shlex
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from tests.test_sentence_worker_request import request

from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch, parse_probe_launch
from entropic_substrate.ladder.run.provider.d1_command_runner import D1BoundedProcessRunner
from entropic_substrate.ladder.run.worker_request import ProbeRequest

pytestmark = pytest.mark.unit


def remote_request() -> ProbeRequest:
    value = request()
    bundle = ArtifactIdentity("code.bundle", ContentHash("sha256", "b" * 64))
    return replace(
        value,
        profile=replace(value.profile, run_class="remote_tiny", device="cuda", precision="bf16"),
        inputs=(*value.inputs, bundle),
        code_bundle=bundle,
        code_commit="c" * 40,
        image="ghcr.io/luiz-frias/entropic-substrate-lm@sha256:" + "d" * 64,
    )


def launch() -> H0ProbeLaunch:
    return H0ProbeLaunch(
        request=remote_request(),
        template_id="template-fixture",
        registry_auth_id="registry-fixture",
        gpu_type_id="NVIDIA A100 80GB PCIe",
        data_center_id="US-TX-3",
        cloud_type="SECURE",
        container_disk_gb=80,
        r2_access_secret_name="R2_ACCESS",  # pragma: allowlist secret
        r2_private_secret_name="R2_PRIVATE",  # pragma: allowlist secret
        mlflow_uri_secret_name="MLFLOW_URI",  # pragma: allowlist secret
        mlflow_username_secret_name="MLFLOW_USER",  # pragma: allowlist secret
        mlflow_password_secret_name="MLFLOW_PASS",  # pragma: allowlist secret
        mlflow_s3_endpoint_secret_name="MLFLOW_S3",  # pragma: allowlist secret
        controller_timeout_seconds=3600,
        watchdog_timeout_seconds=3900,
        cost_guard="host_watchdog_tiny_only",
    )


def test_roundtrip_binds_every_deployment_choice_without_authority() -> None:
    value = launch()
    assert parse_probe_launch(value.payload, max_bytes=1_000_000) == value
    assert value.manifest["scientific_authority"] == "none"
    assert value.manifest["spending_authority"] == "none"
    assert value.manifest["provider_deadline"] is None
    assert value.manifest["schema"] == "h0-probe-launch-v2"
    assert value.manifest["runpodctl_version"] == "runpodctl 2.14.0-dd55bcf"
    assert len(value.manifest["runpodctl_sha256"]) == 64
    for updated in (
        replace(value, container_disk_gb=81),
        replace(value, cloud_type="COMMUNITY"),
        replace(value, controller_timeout_seconds=3500),
        replace(value, watchdog_timeout_seconds=4000),
    ):
        assert updated.digest != value.digest


def test_command_runs_bound_worker_without_ssh_or_shared_volume() -> None:
    value = launch()
    args = value.create_arguments
    assert args[:2] == ("pod", "create")
    assert "--ssh=false" in args
    paired = tuple(arg for arg in args[2:] if arg != "--ssh=false")
    flags = dict(zip(paired[::2], paired[1::2], strict=True))
    assert flags["--image"] == value.request.image
    assert flags["--gpu-count"] == "1" and flags["--min-cuda-version"] == "12.6"
    assert flags["--volume-in-gb"] == "0"
    assert "--wait" not in args and "--terminate-after" not in args
    assert "--network-volume-id" not in args
    assert flags["--name"] == value.pod_name
    assert "--env" not in args
    assert flags["--template-id"] == value.template_id
    # EXIT-GATE LANE (conductor, operator-cleared 2026-09-26): the worker builds
    # CompositeSink(JsonlSink, MlflowSink, tiny=True), and sinks.mlflow_enabled
    # admits MLflow only when MLFLOW_TRACKING_URI is set AND ENTROPIC_MLFLOW_TINY
    # == "1" (silent JSONL-only otherwise, measured 2026-09-26). The template env
    # is therefore the R2 pair, the four MLflow secret references and the literal
    # tiny opt-in -- exactly seven keys, nothing else.
    assert value.template_environment == {
        "AWS_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_R2_ACCESS }}",
        "AWS_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_R2_PRIVATE }}",
        "MLFLOW_TRACKING_URI": "{{ RUNPOD_SECRET_MLFLOW_URI }}",
        "MLFLOW_TRACKING_USERNAME": "{{ RUNPOD_SECRET_MLFLOW_USER }}",
        "MLFLOW_TRACKING_PASSWORD": "{{ RUNPOD_SECRET_MLFLOW_PASS }}",
        "MLFLOW_S3_ENDPOINT_URL": "{{ RUNPOD_SECRET_MLFLOW_S3 }}",
        "ENTROPIC_MLFLOW_TINY": "1",
    }
    shell, flag, script = shlex.split(flags["--docker-args"])
    assert (shell, flag) == ("/bin/sh", "-c")
    commands = shlex.split(script)
    assert value.workspace.startswith("/app/h-probe-")
    assert commands[:6] == ["mkdir", "-m", "700", "--", value.workspace, "&&"]
    worker = commands[6:]
    assert worker[:5] == [
        "exec",
        "python",
        "-m",
        "entropic_substrate.ladder.run.worker_command",
        "run",
    ]
    worker_flags = dict(zip(worker[5::2], worker[6::2], strict=True))
    assert worker_flags["--request-key"] == f"{value.request.prefix}/request.json"
    assert worker_flags["--request-sha256"] == value.request.digest
    assert worker_flags["--request-size"] == str(len(value.request.payload))
    assert worker_flags["--workspace-root"] == value.workspace
    assert worker_flags["--repo-root"] == "/app"


@pytest.mark.parametrize(
    "field,value",
    [
        ("registry_auth_id", ""),
        ("registry_auth_id", "--evil"),
        ("template_id", ""),
        ("gpu_type_id", "GPU\ncommand"),
        ("gpu_type_id", ""),
        ("data_center_id", " ../elsewhere"),
        ("cloud_type", "secure"),
        ("container_disk_gb", 0),
        ("container_disk_gb", True),
        ("r2_access_secret_name", "{{raw-key}}"),
        ("r2_private_secret_name", "R2_ACCESS"),
        ("mlflow_uri_secret_name", "{{raw-key}}"),
        ("mlflow_username_secret_name", "R2_ACCESS"),
        ("mlflow_password_secret_name", "MLFLOW_USER"),
        ("mlflow_s3_endpoint_secret_name", ""),
        ("controller_timeout_seconds", 0),
        ("controller_timeout_seconds", True),
        ("watchdog_timeout_seconds", 3600),
        ("watchdog_timeout_seconds", float("inf")),
        ("cost_guard", "provider_deadline"),
        ("cost_guard", "none"),
    ],
)
def test_invalid_deployments_fail_before_any_provider_action(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(launch(), **{field: value})


def test_local_requests_cannot_use_the_paid_launch_path() -> None:
    with pytest.raises(ValueError, match="remote H0"):
        replace(launch(), request=request())


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "authority", "deadline", "schema", "duplicate", "oversize", "cli"],
)
def test_manifest_reader_rejects_partial_or_relabelled_declarations(mutation: str) -> None:
    value = launch()
    row = value.manifest
    if mutation == "missing":
        del row["watchdog_timeout_seconds"]
    elif mutation == "extra":
        row["fallback"] = True
    elif mutation == "authority":
        row["spending_authority"] = "approved"
    elif mutation == "deadline":
        row["provider_deadline"] = "pretend"
    elif mutation == "schema":
        row["schema"] = "d1-launch"
    elif mutation == "cli":
        row["runpodctl_sha256"] = "0" * 64
    raw = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    if mutation == "duplicate":
        raw = b'{"schema":"h0-probe-launch-v1",' + raw[1:]
    with pytest.raises(ValueError):
        parse_probe_launch(raw, max_bytes=1 if mutation == "oversize" else 1_000_000)


def test_request_identity_changes_launch_identity_and_owned_namespace() -> None:
    value = launch()
    changed = replace(value, request=replace(value.request, nonce="2" * 64))
    assert changed.digest != value.digest
    assert changed.pod_name != value.pod_name and changed.workspace != value.workspace


@pytest.mark.parametrize("maximum", [0, True, -1])
def test_reader_requires_explicit_positive_bounds(maximum: int) -> None:
    with pytest.raises(ValueError):
        parse_probe_launch(launch().payload, max_bytes=maximum)


def test_shell_composition_executes_once_and_preserves_existing_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute the actual composed shell with an argv-recording worker stand-in."""
    workspace = tmp_path / "owned"
    monkeypatch.setattr(H0ProbeLaunch, "workspace", property(lambda _self: str(workspace)))
    binary = tmp_path / "python"
    binary.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o700)
    value = launch()
    args = value.create_arguments
    command = tuple(shlex.split(args[args.index("--docker-args") + 1]))
    runner = D1BoundedProcessRunner(timeout_seconds=5, max_output_bytes=10_000)
    result = runner.run(command, cwd=tmp_path, input=b"", env={"PATH": f"{tmp_path}:/bin:/usr/bin"})
    assert result.returncode == 0 and workspace.is_dir()
    assert workspace.stat().st_mode & 0o777 == 0o700
    worker = json.loads(result.stdout)
    assert worker[:3] == ["-m", "entropic_substrate.ladder.run.worker_command", "run"]
    assert worker[worker.index("--workspace-root") + 1] == str(workspace)
    retained = workspace / "forensic.txt"
    retained.write_text("must survive")
    again = runner.run(command, cwd=tmp_path, input=b"", env={"PATH": f"{tmp_path}:/bin:/usr/bin"})
    assert again.returncode != 0 and not again.stdout
    assert retained.read_text() == "must survive"


def test_mutated_nested_request_is_revalidated() -> None:
    value = remote_request()
    object.__setattr__(value.profile, "stage", "H1")
    with pytest.raises(ValueError):
        replace(launch(), request=value)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "id",
        "image",
        "env",
        "public",
        "serverless",
        "ports",
        "entrypoint",
        "disk",
        "volume",
        "registry",
        "env-missing-mlflow",
        "env-missing-tiny",
        "env-tiny-off",
        "env-r2-only",
    ],
)
def test_template_binding_is_checked_not_assumed(mutation: str | None) -> None:
    value = launch()
    template = {
        "id": value.template_id,
        "name": "private-test",
        "imageName": value.request.image,
        "env": value.template_environment,
        "containerDiskInGb": value.container_disk_gb,
        "containerRegistryAuthId": value.registry_auth_id,
    }
    changes = {
        "id": ("id", "foreign"),
        "image": ("imageName", "foreign:latest"),
        "env": ("env", {}),
        "public": ("isPublic", True),
        "serverless": ("isServerless", True),
        "ports": ("ports", ["22/tcp"]),
        "entrypoint": ("dockerEntrypoint", ["unrequested"]),
        "disk": ("containerDiskInGb", value.container_disk_gb + 1),
        "volume": ("volumeInGb", 10),
        "registry": ("containerRegistryAuthId", "foreign"),
        "env-missing-mlflow": (
            "env",
            {k: v for k, v in value.template_environment.items() if k != "MLFLOW_TRACKING_URI"},
        ),
        "env-missing-tiny": (
            "env",
            {k: v for k, v in value.template_environment.items() if k != "ENTROPIC_MLFLOW_TINY"},
        ),
        "env-tiny-off": ("env", {**value.template_environment, "ENTROPIC_MLFLOW_TINY": "0"}),
        "env-r2-only": (
            "env",
            {
                "AWS_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_R2_ACCESS }}",
                "AWS_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_R2_PRIVATE }}",
            },
        ),
    }
    if mutation is None:
        value.validate_template(template)
        return
    key, replacement = changes[mutation]
    template[key] = replacement
    with pytest.raises(ValueError, match="template"):
        value.validate_template(template)


def test_v1_launch_records_without_mlflow_references_are_refused() -> None:
    """A pre-exit-gate launch record (schema v1, no MLflow secret names) never parses
    as a v2 deployment: the MLflow delivery is part of the pinned identity, not a
    default the parser may fill (house rule: never fill defaults)."""
    value = launch()
    row = json.loads(value.payload)
    row["schema"] = "h0-probe-launch-v1"
    for key in (
        "mlflow_uri_secret_name",
        "mlflow_username_secret_name",
        "mlflow_password_secret_name",
        "mlflow_s3_endpoint_secret_name",
    ):
        row.pop(key)
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        parse_probe_launch(payload, max_bytes=1_000_000)
