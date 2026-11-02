"""H0 deployment pins cannot become broader experiment or spending authority.

RE-FROZEN 2026-11-02 (FOR-1694, h1-l4c, C-L4c-1, launch schema v5). Every launch these helpers
build in-process now renders ``h-probe-launch-v5`` (write-new), so each frozen launch digest and
pod name moved once. RETRACTION (house rule 4): the digest, pod name and workspace the v4
fixtures pinned before this commit (every ``launch()``, ``launch_nine()``, ``pilot_launch()``
and ``pilot_launch_eight()`` value, whose manifests carried ``"schema":"h-probe-launch-v4"``)
are retracted, not renamed: the v4 manifest bytes still parse and re-render as the same class
(``test_a_v4_payload_reads_old_as_the_same_class_and_never_equals_v5``), but no in-process
launch equals one. The legacy ``h0-probe-launch-v2`` payload below keeps its byte and name
pins and is now REFUSED BY NAME (its read retired at this bump).
"""

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
    LAUNCH_SCHEMA_V5,
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


# The fixture's sealed hourly price ceiling: 36.9 USD/h over the 3900 s watchdog is 39.975 USD,
# inside the authority's 40 USD cap (37.0 is 40.083 USD, past it).
PILOT_CEILING = 36.9


def pilot_launch() -> H0ProbeLaunch:
    """A v5 scientific launch: it SEALS ``PILOT_CEILING``."""
    return replace(
        launch(),
        request=pilot_request(),
        cost_guard="host_watchdog_budgeted",
        hourly_price_ceiling=PILOT_CEILING,
    )


def pilot_launch_v4() -> H0ProbeLaunch:
    """The same scientific launch under the v4 manifest: it seals no ceiling, so the
    ``--max-hourly-rate`` flag keeps its v3 semantics (the rung B cap check)."""
    return replace(pilot_launch(), manifest_schema=LAUNCH_SCHEMA_V4, hourly_price_ceiling=None)


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
    assert value.manifest["schema"] == "h-probe-launch-v5"
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


def test_a_v2_payload_is_refused_by_name_and_the_refusal_names_v3_as_the_minimum() -> None:
    """C-L4c-1: the v2 read the v3 section promised to retire at the next bump is closed. The
    legacy bytes, digest and name pin stay as a record (the rule ``<h0-tiny>-<digest[:24]>`` that
    named a v2 pod is unchanged), but ``parse_probe_launch`` refuses them naming the minimum."""
    assert hashlib.sha256(_PARENT_V2_PAYLOAD).hexdigest() == _PARENT_V2_DIGEST
    assert f"h0-tiny-{_PARENT_V2_DIGEST[:24]}" == _PARENT_V2_POD_NAME
    with pytest.raises(ValueError) as raised:
        parse_probe_launch(_PARENT_V2_PAYLOAD, max_bytes=1_000_000)
    message = str(raised.value)
    assert LAUNCH_SCHEMA_V2 in message and "minimum supported version" in message
    assert LAUNCH_SCHEMA_V3 in message and "not admitted" in message
    # in-process construction under the retired name is refused too, never rendered
    with pytest.raises(ValueError, match="manifest schema is not admitted"):
        replace(launch(), manifest_schema=LAUNCH_SCHEMA_V2)


def test_the_v5_manifest_records_extension_names_and_parameters_and_roundtrips() -> None:
    value = launch_nine()
    assert value.manifest["schema"] == LAUNCH_SCHEMA_V5
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


def test_unknown_launch_schemas_are_refused() -> None:
    """RETRACTION (house rule 4, FOR-1694): this test used to pin ``h-probe-launch-v5`` as an
    unknown schema and a v2 launch as unable to declare extensions. v5 is now the schema every
    launch renders, and v2 is retired, so the tooth moves to names that stay unknown: a
    schema one past v5, the legacy rendering of v4's name, and a v2 launch of any extensions."""
    for name in ("h-probe-launch-v6", "h0-probe-launch-v4", "h-probe-launch-v1"):
        with pytest.raises(ValueError, match="manifest schema is not admitted"):
            replace(launch(), manifest_schema=name)
    with pytest.raises(ValueError):
        replace(launch(), manifest_schema=LAUNCH_SCHEMA_V2, extensions=(system_metrics(),))
    for name in ("h-probe-launch-v6", "h0-probe-launch-v4"):
        row = launch().manifest
        row["schema"] = name
        with pytest.raises(ValueError, match="scope or authority mismatch"):
            parse_probe_launch(_reencode(row), max_bytes=1_000_000)


def _reencode(row: dict[str, object]) -> bytes:
    return json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


@pytest.mark.parametrize("mutation", ["unknown", "v3_missing", "v2_extra", "extra_param", "order"])
def test_the_manifest_reader_refuses_unknown_or_unfilled_extensions(mutation: str) -> None:
    """C2: unknown names are refused at parse (M4 teeth), a payload never gets an empty
    extension set filled in, and a v2 label is refused by name (its read retired at v5)."""
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
    "variant", ["v3_read", "v4_read", "v5_base", "v5_nine", "v5_pilot", "v5_pilot_eight"]
)
def test_every_launch_variant_survives_the_reparse_sites_outside_this_module(
    tmp_path: Path, variant: str
) -> None:
    """G7: the control directory, the provider transport and the process child each re-parse
    ``launch.payload`` and require the nominal class. A v3- or v4-read launch is the same class, so
    all three keep working for it, and parse(x.payload) == x holds for every variant."""
    from entropic_substrate.ladder.run.h_probe_control import own_probe_role, read_probe_control
    from entropic_substrate.ladder.run.h_runpod_transport import HRunpodTransport

    value = {
        "v3_read": lambda: parse_probe_launch(
            replace(launch_nine(), manifest_schema=LAUNCH_SCHEMA_V3).payload, max_bytes=1_000_000
        ),
        "v4_read": lambda: parse_probe_launch(
            replace(
                pilot_launch_eight(), manifest_schema=LAUNCH_SCHEMA_V4, hourly_price_ceiling=None
            ).payload,
            max_bytes=1_000_000,
        ),
        "v5_base": launch,
        "v5_nine": launch_nine,
        "v5_pilot": pilot_launch,
        "v5_pilot_eight": pilot_launch_eight,
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
V5_ONLY_KEYS = ("hourly_price_ceiling", "hours_cap", "pair", "cells")


def test_a_v3_payload_reads_old_as_the_same_class_and_never_equals_v5() -> None:
    v3 = replace(launch_nine(), manifest_schema=LAUNCH_SCHEMA_V3)
    assert v3.manifest["schema"] == "h0-probe-launch-v3"
    read = parse_probe_launch(v3.payload, max_bytes=1_000_000)
    assert type(read) is H0ProbeLaunch and read == v3
    assert read.payload == v3.payload and read.manifest_schema == LAUNCH_SCHEMA_V3
    v5 = launch_nine()
    assert read != v5 and read.digest != v5.digest and read.pod_name != v5.pod_name
    assert read.template_environment == v5.template_environment
    # a v3 manifest carries no v5 key; otherwise the two smoke manifests differ in the schema
    # marker alone
    assert not set(V5_ONLY_KEYS) & set(read.manifest)
    assert {k: v for k, v in read.manifest.items() if k != "schema"} == {
        k: v for k, v in v5.manifest.items() if k != "schema" and k not in V5_ONLY_KEYS
    }
    # a v3 payload that smuggles a v5 key is refused (the key list is chosen by the schema)
    smuggled = {**v3.manifest, "hourly_price_ceiling": None}
    with pytest.raises(ValueError):
        parse_probe_launch(_reencode(smuggled), max_bytes=1_000_000)


def test_a_v4_payload_reads_old_as_the_same_class_and_never_equals_v5() -> None:
    """The v4 pilot launch (flag-semantics ceiling) re-renders its own bytes; its pod name is
    ``<stage>-<class>-<digest prefix>`` of ITS digest, and the rule is unchanged."""
    v4 = pilot_launch_v4()
    assert v4.manifest["schema"] == "h-probe-launch-v4" and not set(V5_ONLY_KEYS) & set(v4.manifest)
    read = parse_probe_launch(v4.payload, max_bytes=1_000_000)
    assert type(read) is H0ProbeLaunch and read == v4 and read.payload == v4.payload
    assert read.manifest_schema == LAUNCH_SCHEMA_V4 and read.hourly_price_ceiling is None
    assert read.pod_name == f"h1-pilot-{read.digest[:24]}"
    v5 = pilot_launch()
    assert read != v5 and read.digest != v5.digest and read.pod_name != v5.pod_name
    # a v4 payload that smuggles a v5 key, and a v5 payload that drops one, are refused
    with pytest.raises(ValueError):
        parse_probe_launch(
            _reencode({**v4.manifest, "hourly_price_ceiling": PILOT_CEILING}), max_bytes=1_000_000
        )
    dropped = v5.manifest
    del dropped["hourly_price_ceiling"]
    with pytest.raises(ValueError):
        parse_probe_launch(_reencode(dropped), max_bytes=1_000_000)


def test_a_v1_payload_stays_refused() -> None:
    row = launch().manifest
    row["schema"] = "h0-probe-launch-v1"
    with pytest.raises(ValueError, match="scope or authority mismatch"):
        parse_probe_launch(_reencode(row), max_bytes=1_000_000)


def test_a_pilot_launch_names_its_authority_and_roundtrips() -> None:
    value = pilot_launch_eight()
    authority = value.request.authority
    assert authority is not None
    assert value.manifest["schema"] == LAUNCH_SCHEMA_V5
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
        ({"manifest_schema": LAUNCH_SCHEMA_V2}, "manifest schema is not admitted"),
        ({"manifest_schema": LAUNCH_SCHEMA_V4}, "a pre-v5 launch seals no hourly price ceiling"),
    ],
)
def test_a_pilot_launch_is_bound_by_its_authority(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(pilot_launch(), **changes)


def test_the_watchdog_may_use_exactly_the_authority_hours() -> None:
    # the sealed ceiling is re-bounded by the longer window: 6 USD/h x 6 h = 36 USD <= 40 USD
    value = replace(
        pilot_launch(),
        controller_timeout_seconds=21_000,
        watchdog_timeout_seconds=21_600,
        hourly_price_ceiling=6.0,
    )
    assert value.watchdog_timeout_seconds == 6 * 3600
    with pytest.raises(ValueError, match="exceeds the spend cap"):
        replace(value, hourly_price_ceiling=7.0)  # 7 x 6 = 42 USD


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
    value = pilot_launch_v4()  # unsealed: the flag keeps its rung B cap semantics
    if admitted:
        value.require_hourly_rate_within_cap(rate)
    else:
        with pytest.raises(ValueError, match="exceeds the spend cap"):
            value.require_hourly_rate_within_cap(rate)


def test_a_smoke_ceiling_is_the_invokers_approval_alone() -> None:
    launch().require_hourly_rate_within_cap(1_000_000.0)


# ---------------------------------------------------------------------------
# FOR-1694 (C-L4c-1, launch schema v5): the SEALED hourly price ceiling and hours cap.
# ---------------------------------------------------------------------------
def test_a_v5_scientific_launch_seals_its_ceiling_and_the_authoritys_hours() -> None:
    value = pilot_launch()
    manifest = value.manifest
    assert manifest["hourly_price_ceiling"] == PILOT_CEILING
    assert manifest["hours_cap"] == 6.0  # copied from authority.spend_cap.hours, never typed
    assert manifest["pair"] is False and manifest["cells"] is None
    assert parse_probe_launch(value.payload, max_bytes=1_000_000) == value
    # the digest, and with it the pod name and workspace, binds the ceiling and the hours
    assert replace(value, hourly_price_ceiling=30.0).digest != value.digest
    assert replace(value, hourly_price_ceiling=30.0).pod_name != value.pod_name
    other_hours = replace(
        pilot_launch(),
        request=replace(
            pilot_request(), authority=sealed_authority(spend_cap=replace_cap(hours=7.0))
        ),
    )
    assert other_hours.manifest["hours_cap"] == 7.0 and other_hours.digest != value.digest


def replace_cap(**changes: float) -> object:
    from entropic_substrate.ladder.run.run_authority import SpendCap

    return SpendCap(**{"usd": 40.0, "hours": 6.0, **changes})  # type: ignore[arg-type]


def test_a_smoke_v5_launch_carries_explicit_null_for_both_sealed_fields() -> None:
    value = launch()
    manifest = value.manifest
    assert manifest["schema"] == LAUNCH_SCHEMA_V5
    assert manifest["hourly_price_ceiling"] is None and manifest["hours_cap"] is None
    assert manifest["pair"] is False and manifest["cells"] is None
    # and a smoke launch seals nothing: a ceiling on it is refused, never silently ignored
    with pytest.raises(ValueError, match="only a v5 scientific launch seals"):
        replace(launch(), hourly_price_ceiling=1.0)


@pytest.mark.parametrize("ceiling", [None, 0, -1.0, float("inf"), float("nan"), True, "36.9"])
def test_a_v5_scientific_launch_needs_a_real_ceiling_inside_its_cap(ceiling: object) -> None:
    with pytest.raises(ValueError):
        replace(pilot_launch(), hourly_price_ceiling=ceiling)
    with pytest.raises(ValueError, match="exceeds the spend cap"):
        replace(pilot_launch(), hourly_price_ceiling=37.0)  # 37 x 3900 / 3600 = 40.08 USD > 40


@pytest.mark.parametrize(
    "tamper",
    [
        {"hours_cap": 7.0},
        {"hours_cap": None},
        {"pair": True},
        {"cells": ["a" * 64, "b" * 64]},
        {"hourly_price_ceiling": 37.0},  # past the cap: the product rule re-runs at parse
        {"hourly_price_ceiling": None},
        {"hourly_price_ceiling": "36.9"},
    ],
)
def test_the_v5_reader_re_derives_every_copied_field_and_refuses_a_disagreement(
    tamper: dict[str, object],
) -> None:
    row = {**pilot_launch().manifest, **tamper}
    with pytest.raises(ValueError):
        parse_probe_launch(_reencode(row), max_bytes=1_000_000)


def test_a_sealed_ceiling_admits_only_itself_and_refuses_every_other_value_by_name() -> None:
    value = pilot_launch()
    value.require_hourly_rate_within_cap(PILOT_CEILING)
    assert value.hourly_ceiling(None) == PILOT_CEILING
    assert value.hourly_ceiling(PILOT_CEILING) == PILOT_CEILING
    # no tolerance in either direction: a lower flag is as unsealed as a higher one
    for flag in (PILOT_CEILING - 1e-9, PILOT_CEILING + 1e-9, 1.0, 37.0, 0, float("nan"), True):
        with pytest.raises(ValueError, match="differs from the launch's sealed hourly price"):
            value.require_hourly_rate_within_cap(flag)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="differs from the launch's sealed hourly price"):
            value.hourly_ceiling(flag)  # type: ignore[arg-type]


def test_an_unsealed_launch_keeps_the_flag_with_its_v3_semantics_and_refuses_its_absence() -> None:
    for unsealed in (launch(), pilot_launch_v4()):
        assert unsealed.hourly_price_ceiling is None
        with pytest.raises(ValueError, match="seals no hourly price ceiling"):
            unsealed.hourly_ceiling(None)
    assert launch().hourly_ceiling(0.6) == 0.6
    assert pilot_launch_v4().hourly_ceiling(36.9) == 36.9
    with pytest.raises(ValueError, match="exceeds the spend cap"):
        pilot_launch_v4().hourly_ceiling(37.0)


def test_the_pod_name_rule_is_unchanged_for_v5_and_every_read_old_launch() -> None:
    """D8 (verify-and-close): ``<stage>-<class>-<digest[:24]>`` for a scientific launch,
    ``h0-tiny-<digest[:24]>`` for a smoke one, in every schema, with no two launches sharing one."""
    pilot, smoke = pilot_launch(), launch()
    assert pilot.pod_name == f"h1-pilot-{pilot.digest[:24]}"
    assert smoke.pod_name == f"h0-tiny-{smoke.digest[:24]}"
    v4, v3 = pilot_launch_v4(), replace(launch(), manifest_schema=LAUNCH_SCHEMA_V3)
    assert v4.pod_name == f"h1-pilot-{v4.digest[:24]}"
    assert v3.pod_name == f"h0-tiny-{v3.digest[:24]}"
    names = {pilot.pod_name, smoke.pod_name, v4.pod_name, v3.pod_name}
    assert len(names) == 4
    # two ceilings, two authorities, two nonces: three more launches, three more names
    assert len({pilot.pod_name, replace(pilot, hourly_price_ceiling=30.0).pod_name}) == 2
