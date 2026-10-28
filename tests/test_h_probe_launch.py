"""H0 deployment pins cannot become broader experiment or spending authority."""

import hashlib
import json
import shlex
import sys
from dataclasses import fields, replace
from pathlib import Path

import pytest
from tests.test_run_authority import authority as sealed_authority
from tests.test_sentence_worker_request import request

from entropic_substrate.core.sentence.profile import (
    DEVELOPMENT_DIAGNOSTIC_SAMPLES,
    DEVELOPMENT_GENERATION_CONTINUATIONS,
    DEVELOPMENT_IMPORTANCE_SAMPLES,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.reader.sinks import (
    MLFLOW_EXPERIMENT_ENV,
    MLFLOW_SYSTEM_METRICS_ENV,
)
from entropic_substrate.ladder.run.h_probe_launch import (
    EXTENSION_REGISTRY,
    H_SERIES_EXPERIMENT,
    LAUNCH_SCHEMA_V2,
    LAUNCH_SCHEMA_V3,
    LAUNCH_SCHEMA_V4,
    H0ProbeLaunch,
    LaunchExtension,
    named_experiment,
    parse_probe_launch,
    system_metrics,
)
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


def launch_nine() -> H0ProbeLaunch:
    """The H0 wave's declaration: base plus both registry extensions (nine keys)."""
    return replace(launch(), extensions=(named_experiment(H_SERIES_EXPERIMENT), system_metrics()))


def pilot_request(**authority_changes: object) -> ProbeRequest:
    """A valid H1 pilot request (development profile) under a matching sealed authority."""
    value = remote_request()
    profile = replace(
        value.profile,
        name="h1-isotropic-pilot",
        stage="H1",
        run_class="pilot",
        consumer="isotropic",
        optimization=replace(value.profile.optimization, future_weight=0),
        schedule=replace(
            value.profile.schedule,
            importance_samples=DEVELOPMENT_IMPORTANCE_SAMPLES,
            diagnostic_samples=DEVELOPMENT_DIAGNOSTIC_SAMPLES,
            generation_continuations=DEVELOPMENT_GENERATION_CONTINUATIONS,
        ),
    )
    return replace(value, profile=profile, authority=sealed_authority(**authority_changes))


def pilot_launch() -> H0ProbeLaunch:
    return replace(launch(), request=pilot_request(), cost_guard="host_watchdog_budgeted")


def pilot_launch_eight() -> H0ProbeLaunch:
    """The scientific H wave's declaration: six base keys plus both extensions."""
    return replace(
        pilot_launch(), extensions=(named_experiment(H_SERIES_EXPERIMENT), system_metrics())
    )


def test_roundtrip_binds_every_deployment_choice_without_authority() -> None:
    value = launch()
    assert parse_probe_launch(value.payload, max_bytes=1_000_000) == value
    assert value.manifest["scientific_authority"] == "none"
    assert value.manifest["spending_authority"] == "none"
    assert value.manifest["provider_deadline"] is None
    assert value.manifest["schema"] == "h-probe-launch-v4"
    assert value.manifest["runpodctl_version"] == "runpodctl 3.0.1-551d6a4"
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


# ---------------------------------------------------------------------------
# FOR-1647 / DN-OBS-TEMPLATE: schema v3 = seven-key base + declared extensions
# from a closed registry; v2 is read-old as v3 with no extensions.
# ---------------------------------------------------------------------------

# The v2 payload the PARENT encoder (7c87208, h0-probe-launch-v2) rendered for this
# file's fixture, byte for byte (captured by running the parent's launch().payload
# before the v3 change); its digest and derived names are pinned beside it.
#
# RE-FROZEN 2026-10-20 (FOR-1664 door A (ii), operator ruling): the embedded profile gained
# schedule.diagnostic_samples and its schema literal became h-run-profile-v2, so the old
# bytes no longer parse (v1 is refused; no read-old branch exists). RETRACTION (house rule
# 4): the paragraph above says these bytes are the parent encoder's own capture. That held
# until this re-freeze. They are now the production encoder's output on the FOR-1664 tree:
# replace(launch(), manifest_schema=LAUNCH_SCHEMA_V2, extensions=()).payload. Control: the
# same recipe on the unmodified parent 0fd202bb reproduces the old bytes and digest exactly,
# and the decoded old and new payloads differ at exactly two JSON paths,
# request.profile.schema and request.profile.schedule.diagnostic_samples (8, tiny smoke).
# Old digest:
# 31d1687bd6ba229ee5e36b8c506d1307b3de3e1fd5b71a4eb1bd6ce3ea729e19  # pragma: allowlist secret
# Old pod name: h0-tiny-31d1687bd6ba229ee5e36b8c  # pragma: allowlist secret
#
# RE-FROZEN 2026-10-23 (FOR-1674, operator ruling D1): the embedded profile gained
# schedule.generation_continuations (2, tiny smoke) and its schema literal became
# h-run-profile-v3 (v2 is refused: no read-old branch for profiles). RETRACTION (house
# rule 4): the FOR-1664 paragraph above says these bytes are the production encoder's
# output "on the FOR-1664 tree". That held until this re-freeze; they are now its output on
# the FOR-1674 tree, by the same recipe. Control: the decoded FOR-1664 bytes and the new
# bytes differ at exactly two JSON paths, request.profile.schema and
# request.profile.schedule.generation_continuations. Previous values, kept for the record:
# digest e37391e793dd6e9ae6f95102586119d775622408a0143aef87b819ae801357ae
# (pod name h0-tiny-e37391e793dd6e9ae6f95102)  # pragma: allowlist secret
# Re-sealed 2026-10-26 for the runpodctl 3.0.1-551d6a4 pin (the parser binds every launch
# to the verified CLI): previous digest 62690c796e9c6b608d84360a2dc8f841726847e5e3f285b3dc2480fc2dbf1fa3
# (pod name h0-tiny-62690c796e9c6b608d84360a)  # pragma: allowlist secret
_PARENT_V2_PAYLOAD = b'{"cloud_type":"SECURE","container_disk_gb":80,"controller_timeout_seconds":3600,"cost_guard":"host_watchdog_tiny_only","data_center_id":"US-TX-3","gpu_type_id":"NVIDIA A100 80GB PCIe","mlflow_password_secret_name":"MLFLOW_PASS","mlflow_s3_endpoint_secret_name":"MLFLOW_S3","mlflow_uri_secret_name":"MLFLOW_URI","mlflow_username_secret_name":"MLFLOW_USER","provider_deadline":null,"r2_access_secret_name":"R2_ACCESS","r2_private_secret_name":"R2_PRIVATE","registry_auth_id":"registry-fixture","request":{"code_bundle":{"path":"code.bundle","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},"code_commit":"cccccccccccccccccccccccccccccccccccccccc","image":"ghcr.io/luiz-frias/entropic-substrate-lm@sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd","inputs":[{"path":"source/prepared-corpus.json","sha256":"2a7cc94b2c40ad16ec8141fdf2dd5c5f42b29652d2e8f0c45477d11b119f22e8"},{"path":"code.bundle","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}],"limits":{"max_object_bytes":10000000,"max_objects":100,"max_total_bytes":50000000},"nonce":"1111111111111111111111111111111111111111111111111111111111111111","pause":null,"probe":{"batch_size":4,"eval_positions":8,"lags":[2,4,16,64],"learning_rate":0.001,"seed":42,"train_positions":8,"updates":2,"weight_decay":0.0001},"profile":{"codec":{"decoder_layers":1,"decoder_width":8,"encoder_layers":1,"head_width":4,"max_sentence_bytes":64,"width":8},"consumer":"fisher","device":"cuda","law":{"log_radius_std":0.1,"mean_resultant":0.9,"mode":"magnitude","width":8},"name":"h0-fisher-tiny","optimization":{"betas":[0.9,0.95],"clip_norm":1.0,"epsilon":1e-08,"future_weight":0.3,"learning_rate":0.001,"weight_decay":0.01},"permutation":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255],"precision":"bf16","run_class":"remote_tiny","schedule":{"batch_size":2,"checkpoint_interval":2,"diagnostic_samples":8,"evaluation_batches":1,"evaluation_interval":2,"generation_continuations":2,"generation_sentences":2,"importance_chunk":1,"importance_samples":[2,4],"steps":4,"warmup_steps":2},"schema":"h-run-profile-v3","seed":42,"solver":{"absolute_tolerance":1e-08,"damping":1.0,"max_iterations":32,"relative_tolerance":1e-05},"source_digest":"2a7cc94b2c40ad16ec8141fdf2dd5c5f42b29652d2e8f0c45477d11b119f22e8","source_seed":42,"spine":{"alpha_init":0.05,"carrier":"magnitude","context_sentences":4,"head_width":4,"layers":2,"width":8},"stage":"H0"},"resume":null,"schema":"h-probe-worker-request-v1","scientific_authority":"none","spending_authority":"none"},"runpodctl_sha256":"da72de4c4a2deaeb3642378e7fbb71304f89463a5c7b3af7c7459d6683391696","runpodctl_version":"runpodctl 3.0.1-551d6a4","schema":"h0-probe-launch-v2","scientific_authority":"none","spending_authority":"none","template_id":"template-fixture","watchdog_timeout_seconds":3900}'  # pragma: allowlist secret
_PARENT_V2_DIGEST = (
    "ffcaf756fb553e1e68ab798dd2ec8431fd1e97e91a8323f97dbf56f7030ff5e3"  # pragma: allowlist secret
)
_PARENT_V2_POD_NAME = "h0-tiny-ffcaf756fb553e1e68ab798d"
_BASE_KEYS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_S3_ENDPOINT_URL",
    "ENTROPIC_MLFLOW_TINY",
)


def test_a_parent_v2_payload_reads_as_the_same_launch_with_no_extensions() -> None:
    """C1: read-old preserves identity. Payload, digest, pod name and workspace are
    the v2 bytes' own; the schema marker is state, so it never equals a v3 launch."""
    assert hashlib.sha256(_PARENT_V2_PAYLOAD).hexdigest() == _PARENT_V2_DIGEST
    read = parse_probe_launch(_PARENT_V2_PAYLOAD, max_bytes=1_000_000)
    assert type(read) is H0ProbeLaunch
    assert read.payload == _PARENT_V2_PAYLOAD
    assert read.digest == _PARENT_V2_DIGEST
    assert read.pod_name == _PARENT_V2_POD_NAME
    assert read.workspace == f"/app/h-probe-{_PARENT_V2_DIGEST}"
    assert read.manifest_schema == LAUNCH_SCHEMA_V2 and read.extensions == ()
    assert read.manifest["schema"] == LAUNCH_SCHEMA_V2 and "extensions" not in read.manifest
    assert tuple(read.template_environment) == _BASE_KEYS
    args = read.create_arguments
    assert args[args.index("--name") + 1] == _PARENT_V2_POD_NAME
    assert args[args.index("--template-id") + 1] == "template-fixture"
    # write-new: the same choices built in-process render v4 and a different identity
    built = launch()
    assert built.manifest_schema == LAUNCH_SCHEMA_V4 and built != read
    assert built.digest != read.digest and built.pod_name != read.pod_name
    assert parse_probe_launch(read.payload, max_bytes=1_000_000) == read


def test_the_v4_manifest_records_extension_names_and_parameters_and_roundtrips() -> None:
    value = launch_nine()
    assert value.manifest["schema"] == LAUNCH_SCHEMA_V4
    assert value.manifest["extensions"] == [
        {"name": "named_experiment", "params": {"name": "h-series"}},
        {"name": "system_metrics", "params": {}},
    ]
    assert parse_probe_launch(value.payload, max_bytes=1_000_000) == value
    assert launch().manifest["extensions"] == []
    assert parse_probe_launch(launch().payload, max_bytes=1_000_000) == launch()
    other = replace(launch(), extensions=(named_experiment("other"), system_metrics()))
    assert other.digest != value.digest
    assert replace(value, extensions=(named_experiment("h-series"),)).digest != value.digest


def test_the_h0_wave_declaration_is_exactly_nine_keys_and_the_union_is_validated() -> None:
    value = launch_nine()
    assert value.template_environment == {
        "AWS_ACCESS_KEY_ID": "{{ RUNPOD_SECRET_R2_ACCESS }}",
        "AWS_SECRET_ACCESS_KEY": "{{ RUNPOD_SECRET_R2_PRIVATE }}",
        "MLFLOW_TRACKING_URI": "{{ RUNPOD_SECRET_MLFLOW_URI }}",
        "MLFLOW_TRACKING_USERNAME": "{{ RUNPOD_SECRET_MLFLOW_USER }}",
        "MLFLOW_TRACKING_PASSWORD": "{{ RUNPOD_SECRET_MLFLOW_PASS }}",
        "MLFLOW_S3_ENDPOINT_URL": "{{ RUNPOD_SECRET_MLFLOW_S3 }}",
        "ENTROPIC_MLFLOW_TINY": "1",
        "MLFLOW_EXPERIMENT_NAME": "h-series",
        "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "true",
    }
    assert tuple(value.template_environment) == (
        *_BASE_KEYS,
        *(spec.env_key for spec in EXTENSION_REGISTRY.values()),
    )
    template = {
        "id": value.template_id,
        "imageName": value.request.image,
        "env": value.template_environment,
        "containerDiskInGb": value.container_disk_gb,
        "containerRegistryAuthId": value.registry_auth_id,
    }
    value.validate_template(template)
    for env in (
        launch().template_environment,  # the seven-key base lacks both extension keys
        {k: v for k, v in value.template_environment.items() if k != "MLFLOW_EXPERIMENT_NAME"},
        {**value.template_environment, "MLFLOW_EXPERIMENT_NAME": "other"},
        {**value.template_environment, "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": "false"},
    ):
        with pytest.raises(ValueError, match="template"):
            value.validate_template({**template, "env": env})
    # and the converse: a base launch refuses a nine-key template
    with pytest.raises(ValueError, match="template"):
        launch().validate_template({**template, "env": value.template_environment})
    # create_arguments still carries the template id and no inline env
    args = value.create_arguments
    assert args[args.index("--template-id") + 1] == value.template_id and "--env" not in args


def test_extension_keys_match_their_consumers(monkeypatch: pytest.MonkeyPatch) -> None:
    """C4: the registry names the exact env keys the consumers read, and the system-metrics
    literal is one the consumer predicate enables."""
    from entropic_substrate.ladder.capture.adapters.mlflow import _system_metrics_logging_enabled

    assert EXTENSION_REGISTRY["named_experiment"].env_key == MLFLOW_EXPERIMENT_ENV
    assert EXTENSION_REGISTRY["system_metrics"].env_key == MLFLOW_SYSTEM_METRICS_ENV
    assert tuple(EXTENSION_REGISTRY) == ("named_experiment", "system_metrics")
    keys = [spec.env_key for spec in EXTENSION_REGISTRY.values()]
    assert len(set(keys)) == len(keys) and not set(keys) & set(_BASE_KEYS)
    monkeypatch.setenv(
        MLFLOW_SYSTEM_METRICS_ENV, launch_nine().template_environment[MLFLOW_SYSTEM_METRICS_ENV]
    )
    assert _system_metrics_logging_enabled() is True
    assert launch_nine().template_environment[MLFLOW_EXPERIMENT_ENV] == H_SERIES_EXPERIMENT


def test_the_registry_is_closed_and_immutable() -> None:
    with pytest.raises(TypeError):
        EXTENSION_REGISTRY["plugin"] = EXTENSION_REGISTRY["system_metrics"]  # type: ignore[index]
    with pytest.raises(ValueError, match="registry"):
        LaunchExtension("plugin")
    fixed = {
        "schema",
        "scientific_authority",
        "spending_authority",
        "provider_deadline",
        "runpodctl_version",
        "runpodctl_sha256",
    }
    assert not fixed & {f.name for f in fields(H0ProbeLaunch)}


@pytest.mark.parametrize(
    "extensions",
    [
        (system_metrics(), named_experiment("h-series")),  # not registry order
        (system_metrics(), system_metrics()),  # duplicate
        (named_experiment("a"), named_experiment("b")),  # duplicate name
        ("system_metrics",),  # not a declared extension
    ],
)
def test_noncanonical_or_duplicate_extensions_are_refused_at_construction(
    extensions: object,
) -> None:
    with pytest.raises(ValueError):
        replace(launch(), extensions=extensions)


@pytest.mark.parametrize(
    "build",
    [
        lambda: LaunchExtension("named_experiment"),  # missing parameter
        lambda: LaunchExtension("system_metrics", (("name", "x"),)),  # undeclared parameter
        lambda: named_experiment("h series"),  # unsafe value
        lambda: named_experiment(""),
        lambda: named_experiment("--evil"),
        lambda: LaunchExtension("named_experiment", (("other", "x"),)),
    ],
)
def test_extension_parameters_obey_the_declared_shape_and_identifier_discipline(
    build: object,
) -> None:
    with pytest.raises(ValueError):
        build()  # type: ignore[operator]


def test_a_v2_launch_cannot_declare_extensions_and_unknown_schemas_are_refused() -> None:
    with pytest.raises(ValueError):
        replace(launch(), manifest_schema=LAUNCH_SCHEMA_V2, extensions=(system_metrics(),))
    with pytest.raises(ValueError):
        replace(launch(), manifest_schema="h0-probe-launch-v4")
    with pytest.raises(ValueError):
        replace(launch(), manifest_schema="h-probe-launch-v5")


def _reencode(row: dict[str, object]) -> bytes:
    return json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


@pytest.mark.parametrize("mutation", ["unknown", "v3_missing", "v2_extra", "extra_param", "order"])
def test_the_manifest_reader_refuses_unknown_or_unfilled_extensions(mutation: str) -> None:
    """C2: unknown names are refused at parse (M4 teeth), a v3 payload never gets an empty
    extension set filled in, and the v2 branch accepts no extension key."""
    row = launch_nine().manifest
    extensions = row["extensions"]
    assert isinstance(extensions, list)
    if mutation == "unknown":
        extensions[1] = {"name": "plugin", "params": {}}
    elif mutation == "v3_missing":
        del row["extensions"]
    elif mutation == "v2_extra":
        row["schema"] = LAUNCH_SCHEMA_V2
    elif mutation == "extra_param":
        extensions[1] = {"name": "system_metrics", "params": {"name": "x"}}
    else:
        extensions.reverse()
    with pytest.raises(ValueError):
        parse_probe_launch(_reencode(row), max_bytes=1_000_000)


@pytest.mark.parametrize(
    "variant", ["v2_read", "v3_read", "v4_base", "v4_nine", "v4_pilot", "v4_pilot_eight"]
)
def test_every_launch_variant_survives_the_reparse_sites_outside_this_module(
    tmp_path: Path, variant: str
) -> None:
    """G7: the control directory, the provider transport and the process child each re-parse
    ``launch.payload`` and require the nominal class. A v2-read launch is the same class, so all
    three keep working for it, and parse(x.payload) == x holds for every variant."""
    from entropic_substrate.ladder.run.h_probe_control import own_probe_role, read_probe_control
    from entropic_substrate.ladder.run.h_runpod_transport import HRunpodTransport

    value = {
        "v2_read": lambda: parse_probe_launch(_PARENT_V2_PAYLOAD, max_bytes=1_000_000),
        "v3_read": lambda: parse_probe_launch(
            replace(launch_nine(), manifest_schema=LAUNCH_SCHEMA_V3).payload, max_bytes=1_000_000
        ),
        "v4_base": launch,
        "v4_nine": launch_nine,
        "v4_pilot": pilot_launch,
        "v4_pilot_eight": pilot_launch_eight,
    }[variant]()
    assert type(value) is H0ProbeLaunch
    assert parse_probe_launch(value.payload, max_bytes=1_000_000) == value
    directory = tmp_path / "probe"
    directory.mkdir(mode=0o700)
    with own_probe_role(directory, value, "controller") as controller:
        controller.write("create-intent", {"attempt": "one"}, max_bytes=10_000)
    assert read_probe_control(
        directory, value, "controller", "create-intent", max_bytes=10_000
    ) == {"attempt": "one"}
    transport = HRunpodTransport(
        executable=Path("/fixture/runpodctl"),
        runner=object(),  # type: ignore[arg-type]
        max_response_bytes=1_000_000,
        max_launch_bytes=1_000_000,
    )
    assert transport._launch(value) == value  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Rung B (operator ruling 2026-10-27, "option B"): LAUNCH SCHEMA v4. A scientific
# launch names its sealed authority as both authorities, takes the budgeted guard,
# binds the authority's GPU class and hours, and drops the tiny MLflow opt-in.
# v3 and v2 payloads read old as the same class and never equal a v4 launch.
# ---------------------------------------------------------------------------
def test_a_v3_payload_reads_old_as_the_same_class_and_never_equals_v4() -> None:
    v3 = replace(launch_nine(), manifest_schema=LAUNCH_SCHEMA_V3)
    assert v3.manifest["schema"] == "h0-probe-launch-v3"
    read = parse_probe_launch(v3.payload, max_bytes=1_000_000)
    assert type(read) is H0ProbeLaunch and read == v3
    assert read.payload == v3.payload and read.manifest_schema == LAUNCH_SCHEMA_V3
    v4 = launch_nine()
    assert read != v4 and read.digest != v4.digest and read.pod_name != v4.pod_name
    assert read.template_environment == v4.template_environment
    # the v3 and v4 smoke manifests differ in the schema marker alone
    assert {k: v for k, v in read.manifest.items() if k != "schema"} == {
        k: v for k, v in v4.manifest.items() if k != "schema"
    }


def test_a_v1_payload_stays_refused() -> None:
    row = launch().manifest
    row["schema"] = "h0-probe-launch-v1"
    with pytest.raises(ValueError, match="scope or authority mismatch"):
        parse_probe_launch(_reencode(row), max_bytes=1_000_000)


def test_a_pilot_launch_names_its_authority_and_roundtrips() -> None:
    value = pilot_launch_eight()
    authority = value.request.authority
    assert authority is not None
    assert value.manifest["schema"] == LAUNCH_SCHEMA_V4
    assert value.manifest["scientific_authority"] == authority.digest
    assert value.manifest["spending_authority"] == authority.digest
    assert value.manifest["cost_guard"] == "host_watchdog_budgeted"
    assert parse_probe_launch(value.payload, max_bytes=1_000_000) == value
    assert value.pod_name == f"h1-pilot-{value.digest[:24]}"
    args = value.create_arguments
    assert args[args.index("--name") + 1] == value.pod_name
    assert args[args.index("--gpu-id") + 1] == authority.gpu_class
    assert launch().pod_name.startswith("h0-tiny-")


def test_the_scientific_template_is_eight_keys_without_the_tiny_opt_in() -> None:
    value = pilot_launch_eight()
    environment = value.template_environment
    assert "ENTROPIC_MLFLOW_TINY" not in environment
    assert tuple(environment) == (
        *(k for k in _BASE_KEYS if k != "ENTROPIC_MLFLOW_TINY"),
        *(spec.env_key for spec in EXTENSION_REGISTRY.values()),
    )
    assert len(environment) == 8
    template = {
        "id": value.template_id,
        "imageName": value.request.image,
        "env": environment,
        "containerDiskInGb": value.container_disk_gb,
        "containerRegistryAuthId": value.registry_auth_id,
    }
    value.validate_template(template)
    # a smoke nine-key template cannot carry a pilot, nor the converse
    with pytest.raises(ValueError, match="template"):
        value.validate_template({**template, "env": launch_nine().template_environment})
    with pytest.raises(ValueError, match="template"):
        launch_nine().validate_template({**template, "env": environment})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"cost_guard": "host_watchdog_tiny_only"}, "requires the budgeted host watchdog"),
        ({"gpu_type_id": "NVIDIA H100 80GB HBM3"}, "GPU differs from the run authority"),
        (
            {"controller_timeout_seconds": 21_000, "watchdog_timeout_seconds": 21_601},
            "watchdog outlasts the run authority's hours",
        ),
        ({"manifest_schema": LAUNCH_SCHEMA_V3}, "a pre-v4 launch carries no run authority"),
        ({"manifest_schema": LAUNCH_SCHEMA_V2}, "a pre-v4 launch carries no run authority"),
    ],
)
def test_a_pilot_launch_is_bound_by_its_authority(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(pilot_launch(), **changes)


def test_the_watchdog_may_use_exactly_the_authority_hours() -> None:
    value = replace(
        pilot_launch(), controller_timeout_seconds=21_000, watchdog_timeout_seconds=21_600
    )
    assert value.watchdog_timeout_seconds == 6 * 3600


def test_a_budgeted_guard_on_a_smoke_launch_is_refused_with_the_tiny_message() -> None:
    with pytest.raises(ValueError, match="requires the declared tiny-only watchdog exception"):
        replace(launch(), cost_guard="host_watchdog_budgeted")


@pytest.mark.parametrize(
    "mutation", ["authority_none", "authority_other", "smoke_with_authority", "stage_tamper"]
)
def test_the_launch_reader_refuses_a_mandate_that_differs_from_the_request(
    mutation: str,
) -> None:
    row = pilot_launch().manifest if mutation != "smoke_with_authority" else launch().manifest
    if mutation == "authority_none":
        row["scientific_authority"] = row["spending_authority"] = "none"
    elif mutation == "authority_other":
        row["spending_authority"] = sealed_authority(stage="H2").digest
    elif mutation == "smoke_with_authority":
        row["scientific_authority"] = row["spending_authority"] = sealed_authority().digest
    else:
        request = row["request"]
        assert isinstance(request, dict)
        authority = request["authority"]
        assert isinstance(authority, dict)
        authority["stage"] = "H2"
    # The canonical re-render would also refuse these bytes; the explicit check is what
    # names the defect, so the name is what this pins.
    message = (
        "run authority digest does not match its content"
        if mutation == "stage_tamper"
        else "scope or authority mismatch"
    )
    with pytest.raises(ValueError, match=message):
        parse_probe_launch(_reencode(row), max_bytes=1_000_000)


@pytest.mark.parametrize(
    ("rate", "admitted"),
    [
        # cap 40 USD over the fixture's 3900 s watchdog: the ceiling is 40 * 3600 / 3900
        (40 * 3600 / 3900, True),
        (36.9, True),
        (37.0, False),
        (0, False),
        (-1, False),
        (float("inf"), False),
        (float("nan"), False),
    ],
)
def test_a_pilot_hourly_ceiling_must_keep_the_run_inside_its_spend_cap(
    rate: float, admitted: bool
) -> None:
    value = pilot_launch()
    if admitted:
        value.require_hourly_rate_within_cap(rate)
    else:
        with pytest.raises(ValueError, match="exceeds the spend cap"):
            value.require_hourly_rate_within_cap(rate)


def test_a_smoke_ceiling_is_the_invokers_approval_alone() -> None:
    launch().require_hourly_rate_within_cap(1_000_000.0)
