"""Explicit H0-only deployment choices for the pinned Runpod command boundary.

LAUNCH SCHEMA v3 (FOR-1647, DN-OBS-TEMPLATE, operator 2026-10-13). The pod
environment is a seven-key BASE (four R2/MLflow secret references, the S3
endpoint reference, the tiny opt-in and the tracking URI) plus DECLARED
EXTENSIONS drawn from a CLOSED registry (``EXTENSION_REGISTRY``: an immutable
module constant, no registration function, no discovery, no env or CLI
selection; adding an extension is a source edit). Each extension contributes
exactly one environment key. The manifest records each extension's name AND
its parameters, so ``parse_probe_launch`` rebuilds an identical environment.
The H0 wave launches with ``named_experiment("h-series")`` and
``system_metrics()``: nine keys.

READ-OLD (documentation of an in-tree parse, NOT a reader-identity bump):
``parse_probe_launch`` reads a ``h0-probe-launch-v2`` payload as the same
class with ``manifest_schema == "h0-probe-launch-v2"`` and no extensions. A
v2-read launch re-renders v2: its payload, digest, pod name and workspace are
the bytes it was read from, and it never compares equal to a v3 launch. Every
launch built in-process renders v3 (write-new). v1 stays refused. The v2 read
window retires at the next launch-schema bump; this module implements no
retirement today.
"""

import hashlib
import json
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import cast

from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.ladder.run.worker_request import (
    ProbeRequest,
    parse_probe_request,
)

RUNPODCTL_VERSION = "runpodctl 3.0.1-551d6a4"
RUNPODCTL_SHA256 = (
    "da72de4c4a2deaeb3642378e7fbb71304f89463a5c7b3af7c7459d6683391696"  # pragma: allowlist secret
)


LAUNCH_SCHEMA_V2 = "h0-probe-launch-v2"
LAUNCH_SCHEMA_V3 = "h0-probe-launch-v3"
# The H lane's experiment. Exactly one definition in the tree: the wave monitor
# reads this constant, and a launch selects it with ``named_experiment``.
H_SERIES_EXPERIMENT = "h-series"

_EXTENSION_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


@dataclass(frozen=True, slots=True)
class _ExtensionSpec:
    """One registry row: the single env key an extension contributes."""

    env_key: str
    parameters: tuple[str, ...]
    value: Callable[[Mapping[str, str]], str]


# CLOSED registry, in canonical (registry) order. The env keys equal the consumer
# constants ``ladder.capture.reader.sinks.MLFLOW_EXPERIMENT_ENV`` and
# ``MLFLOW_SYSTEM_METRICS_ENV``; tests/test_h_probe_launch.py pins that equality
# and that "true" is a literal ``adapters.mlflow._system_metrics_logging_enabled``
# accepts, so a rename on the consumer side cannot silently orphan a launch.
EXTENSION_REGISTRY: Mapping[str, _ExtensionSpec] = MappingProxyType(
    {
        "named_experiment": _ExtensionSpec(
            "MLFLOW_EXPERIMENT_NAME", ("name",), lambda params: params["name"]
        ),
        "system_metrics": _ExtensionSpec(
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", (), lambda _: "true"
        ),
    }
)

_BASE_ENVIRONMENT_KEYS = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_S3_ENDPOINT_URL",
        "ENTROPIC_MLFLOW_TINY",
    }
)
_extension_keys = [spec.env_key for spec in EXTENSION_REGISTRY.values()]
if len(set(_extension_keys)) != len(_extension_keys) or _BASE_ENVIRONMENT_KEYS & set(
    _extension_keys
):
    raise RuntimeError("launch extension environment keys must not collide")


@dataclass(frozen=True, slots=True)
class LaunchExtension:
    """A declared, validated member of the closed registry; its name and parameters
    are what the manifest records."""

    name: str
    parameters: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        spec = EXTENSION_REGISTRY.get(self.name) if type(self.name) is str else None
        if spec is None:
            raise ValueError("launch extension is not in the closed registry")
        if type(self.parameters) is not tuple or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            or _EXTENSION_VALUE.fullmatch(pair[1]) is None
            for pair in self.parameters
        ):
            raise ValueError("launch extension parameters must be safe identifier text")
        if tuple(key for key, _ in self.parameters) != spec.parameters:
            raise ValueError("launch extension parameters differ from its registry declaration")

    @property
    def env_value(self) -> str:
        return EXTENSION_REGISTRY[self.name].value(dict(self.parameters))

    @property
    def manifest(self) -> dict[str, object]:
        return {"name": self.name, "params": dict(self.parameters)}


def named_experiment(name: str) -> LaunchExtension:
    """Route the pod's MLflow runs into the named experiment."""
    return LaunchExtension("named_experiment", (("name", name),))


def system_metrics() -> LaunchExtension:
    """Turn on MLflow's own system-metrics monitor in the pod."""
    return LaunchExtension("system_metrics")


@dataclass(frozen=True, slots=True)
class H0ProbeLaunch:
    """A reproducible command, not permission to allocate or proof of a watchdog.

    The controller must establish a separately supervised watchdog before create.
    Both time budgets start at that arming event, not at trainer readiness. R2
    remains authoritative custody; this tiny lane does not attach shared volumes.
    """

    request: ProbeRequest
    template_id: str
    registry_auth_id: str
    gpu_type_id: str
    data_center_id: str
    cloud_type: str
    container_disk_gb: int
    r2_access_secret_name: str
    r2_private_secret_name: str
    mlflow_uri_secret_name: str
    mlflow_username_secret_name: str
    mlflow_password_secret_name: str
    mlflow_s3_endpoint_secret_name: str
    controller_timeout_seconds: int
    watchdog_timeout_seconds: int
    cost_guard: str
    # Schema state, not a subclass: a launch read from v2 is the same class with
    # ``manifest_schema == LAUNCH_SCHEMA_V2``. Neither name may equal one of the six
    # fixed manifest keys, because ``manifest`` and ``parse_probe_launch`` both
    # iterate ``fields()``.
    extensions: tuple[LaunchExtension, ...] = ()
    manifest_schema: str = LAUNCH_SCHEMA_V3

    def __post_init__(self) -> None:
        if type(self.request) is not ProbeRequest:
            raise TypeError("H0 deployment requires a validated worker request")
        request = parse_probe_request(self.request.payload, max_bytes=len(self.request.payload))
        if request.profile.run_class != "remote_tiny" or request.profile.stage != "H0":
            raise ValueError("deployment is restricted to remote H0 tiny probes")
        object.__setattr__(self, "request", request)
        for value in (self.registry_auth_id, self.data_center_id, self.template_id):
            if (
                type(value) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value) is None
            ):
                raise ValueError("provider placement identity is unsafe")
        gpu = self.gpu_type_id
        if (
            type(gpu) is not str
            or not gpu
            or gpu.strip() != gpu
            or not gpu[0].isalnum()
            or len(gpu) > 128
            or any(not (c.isascii() and (c.isalnum() or c in " ._-")) for c in gpu)
        ):
            raise ValueError("provider GPU type must be a safe explicit identifier")
        if self.cloud_type not in ("SECURE", "COMMUNITY"):
            raise ValueError("provider cloud type must be explicit")
        for value in (
            self.container_disk_gb,
            self.controller_timeout_seconds,
            self.watchdog_timeout_seconds,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("deployment storage and time bounds must be positive integers")
        if self.controller_timeout_seconds >= self.watchdog_timeout_seconds:
            raise ValueError(
                "watchdog must leave time after the controller deadline for capture/teardown"
            )
        if self.cost_guard != "host_watchdog_tiny_only":
            raise ValueError("H0 deployment requires the declared tiny-only watchdog exception")
        secret_names = (
            self.r2_access_secret_name,
            self.r2_private_secret_name,
            self.mlflow_uri_secret_name,
            self.mlflow_username_secret_name,
            self.mlflow_password_secret_name,
            self.mlflow_s3_endpoint_secret_name,
        )
        for name in secret_names:
            if type(name) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name) is None:
                raise ValueError(
                    "R2 and MLflow credentials must use provider secret-reference names"
                )
        if len(set(secret_names)) != len(secret_names):
            raise ValueError("provider secret-reference names must be distinct")
        if self.manifest_schema not in (LAUNCH_SCHEMA_V2, LAUNCH_SCHEMA_V3):
            raise ValueError("launch manifest schema is not admitted")
        if type(self.extensions) is not tuple or any(
            type(item) is not LaunchExtension for item in self.extensions
        ):
            raise ValueError("launch extensions must be a tuple of declared extensions")
        declared = [item.name for item in self.extensions]
        if declared != [name for name in EXTENSION_REGISTRY if name in declared] or len(
            set(declared)
        ) != len(declared):
            raise ValueError("launch extensions must be unique and in registry order")
        if self.manifest_schema == LAUNCH_SCHEMA_V2 and self.extensions:
            raise ValueError("a v2 launch declares no extensions")

    @property
    def manifest(self) -> dict[str, object]:
        row: dict[str, object] = {
            "schema": self.manifest_schema,
            "scientific_authority": "none",
            "spending_authority": "none",
            "provider_deadline": None,
            "runpodctl_version": RUNPODCTL_VERSION,
            "runpodctl_sha256": RUNPODCTL_SHA256,
            **{f.name: getattr(self, f.name) for f in fields(self) if f.name in _BASE_FIELDS},
            "request": self.request.manifest,
        }
        if self.manifest_schema == LAUNCH_SCHEMA_V3:
            row["extensions"] = [item.manifest for item in self.extensions]
        return row

    @property
    def payload(self) -> bytes:
        return _json(self.manifest)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def pod_name(self) -> str:
        return f"h0-tiny-{self.digest[:24]}"

    @property
    def workspace(self) -> str:
        return f"/app/h-probe-{self.digest}"

    @property
    def template_environment(self) -> dict[str, str]:
        """The pod's entire environment, every value a provider secret reference
        except the one literal opt-in.

        EXIT-GATE LANE (conductor, operator-cleared 2026-09-26, schema v2): the H
        worker builds ``CompositeSink(JsonlSink, MlflowSink, tiny=True)`` and
        ``sinks.mlflow_enabled`` admits the MLflow mirror only when
        ``MLFLOW_TRACKING_URI`` is set AND ``ENTROPIC_MLFLOW_TINY == "1"``; with
        either missing it falls back to JSONL-only SILENTLY (measured 2026-09-26
        on the pinned 6b4e68fa image regime). The v1 contract carried only the
        two R2 references, so no v1 launch could ever reach the Railway MLflow.
        The username/password references feed the mlflow client's basic-auth
        env contract; the S3 endpoint reference feeds the artifact upload of the
        JSONL mirror. Exactly seven keys; validate_template compares the whole
        mapping, so a template missing any reference or the opt-in is refused.

        v3: the seven base keys, then each declared extension's one key in registry
        order (nine keys for the H0 wave). ``validate_template`` compares the union.
        """
        environment = {
            "AWS_ACCESS_KEY_ID": f"{{{{ RUNPOD_SECRET_{self.r2_access_secret_name} }}}}",
            "AWS_SECRET_ACCESS_KEY": f"{{{{ RUNPOD_SECRET_{self.r2_private_secret_name} }}}}",
            "MLFLOW_TRACKING_URI": f"{{{{ RUNPOD_SECRET_{self.mlflow_uri_secret_name} }}}}",
            "MLFLOW_TRACKING_USERNAME": f"{{{{ RUNPOD_SECRET_{self.mlflow_username_secret_name} }}}}",
            "MLFLOW_TRACKING_PASSWORD": f"{{{{ RUNPOD_SECRET_{self.mlflow_password_secret_name} }}}}",
            "MLFLOW_S3_ENDPOINT_URL": f"{{{{ RUNPOD_SECRET_{self.mlflow_s3_endpoint_secret_name} }}}}",
            "ENTROPIC_MLFLOW_TINY": "1",
        }
        for item in self.extensions:
            environment[EXTENSION_REGISTRY[item.name].env_key] = item.env_value
        return environment

    def validate_template(self, value: object) -> None:
        """Check a pinned-CLI template get result before creating the pod.

        CLI 2.14.0 omits false booleans, zero volumes and empty command/port lists.
        Their absence here follows that serialization contract, not a guessed
        scientific default. A provider read must still succeed independently.
        Measured 2026-10-26: 3.0.1 (every command moved to REST v2 in 3.0.0)
        returns byte-identical ``template get``, ``user`` and ``pod list`` JSON
        to 2.14.0 for this template, so the contract carries over unchanged.
        """
        if type(value) is not dict:
            raise ValueError("H0 template must be a provider object")
        row = cast(dict[str, object], value)
        if (
            row.get("id") != self.template_id
            or row.get("imageName") != self.request.image
            or row.get("env") != self.template_environment
            or row.get("containerRegistryAuthId") != self.registry_auth_id
            or type(row.get("containerDiskInGb")) is not int
            or row.get("containerDiskInGb") != self.container_disk_gb
            or type(row.get("volumeInGb", 0)) is not int
            or row.get("volumeInGb", 0) != 0
            or any(row.get(k, False) is not False for k in ("isPublic", "isServerless", "isRunpod"))
            or any(row.get(k, []) != [] for k in ("ports", "dockerEntrypoint", "dockerStartCmd"))
        ):
            raise ValueError("H0 template differs from the declared private worker configuration")

    @property
    def create_arguments(self) -> tuple[str, ...]:
        """Arguments for verified runpodctl 3.0.1; execution belongs to the controller.

        ``pod create --help`` is byte-identical between 2.14.0 and 3.0.1 (measured
        2026-10-26); the create response itself is only exercised by a live launch.

        The fixed shell composition only creates a new owned workspace and execs
        the existing image worker. No installs, source overlays or SSH readiness.
        An existing workspace is a failure, never an implicit resume or overwrite.
        """
        request = self.request
        if request.image is None:
            raise ValueError("remote H0 deployment requires an immutable image")
        worker = (
            "python",
            "-m",
            "entropic_substrate.ladder.run.worker_command",
            "run",
            "--repo-root",
            "/app",
            "--workspace-root",
            self.workspace,
            "--max-object-bytes",
            str(request.limits.max_object_bytes),
            "--max-objects",
            str(request.limits.max_objects),
            "--request-key",
            f"{request.prefix}/request.json",
            "--request-sha256",
            request.digest,
            "--request-size",
            str(len(request.payload)),
        )
        script = f"mkdir -m 700 -- {shlex.quote(self.workspace)} && exec {shlex.join(worker)}"
        return (
            "pod",
            "create",
            "--name",
            self.pod_name,
            "--template-id",
            self.template_id,
            "--image",
            request.image,
            "--gpu-id",
            self.gpu_type_id,
            "--gpu-count",
            "1",
            "--cloud-type",
            self.cloud_type,
            "--data-center-ids",
            self.data_center_id,
            "--container-disk-in-gb",
            str(self.container_disk_gb),
            "--volume-in-gb",
            "0",
            "--min-cuda-version",
            "12.6",
            "--ssh=false",
            "--registry-auth-id",
            self.registry_auth_id,
            "--docker-args",
            shlex.join(("/bin/sh", "-c", script)),
            "--output",
            "json",
        )


_BASE_FIELDS = frozenset(
    f.name
    for f in fields(H0ProbeLaunch)
    if f.name not in ("request", "extensions", "manifest_schema")
)


def parse_probe_launch(payload: bytes, *, max_bytes: int) -> H0ProbeLaunch:
    """Require every deployment choice and canonical bytes; never fill defaults.

    Schema-first: the key list is chosen by the declared schema before any key is
    checked. Only the v2 branch supplies an empty extension set (the documented
    read-old); a v3 payload without ``extensions`` is refused.
    """
    if (
        type(max_bytes) is not int
        or max_bytes < 1
        or type(payload) is not bytes
        or not 0 < len(payload) <= max_bytes
    ):
        raise ValueError("H0 deployment exceeds its explicit byte budget")
    decoded = json.loads(payload)
    schema = decoded.get("schema") if type(decoded) is dict else None
    if schema not in (LAUNCH_SCHEMA_V2, LAUNCH_SCHEMA_V3):
        raise ValueError("H0 deployment scope or authority mismatch")
    names = " ".join(sorted(_BASE_FIELDS | {"request"}))
    keys = f"schema scientific_authority spending_authority provider_deadline runpodctl_version runpodctl_sha256 {names}"
    row = recovery_record(
        decoded, keys + (" extensions" if schema == LAUNCH_SCHEMA_V3 else "")
    ).copy()
    if (
        row.pop("schema") != schema
        or row.pop("scientific_authority") != "none"
        or row.pop("spending_authority") != "none"
        or row.pop("provider_deadline") is not None
        or row.pop("runpodctl_version") != RUNPODCTL_VERSION
        or row.pop("runpodctl_sha256") != RUNPODCTL_SHA256
    ):
        raise ValueError("H0 deployment scope or authority mismatch")
    extensions: list[LaunchExtension] = []
    if schema == LAUNCH_SCHEMA_V3:
        declared = row.pop("extensions")
        if type(declared) is not list:
            raise ValueError("launch extensions must be a list")
        for item in declared:
            entry = recovery_record(item, "name params")
            params = entry["params"]
            if type(params) is not dict or any(type(v) is not str for v in params.values()):
                raise ValueError("launch extension parameters must be text")
            extensions.append(
                LaunchExtension(
                    cast(str, entry["name"]), tuple(cast(dict[str, str], params).items())
                )
            )
    row["request"] = parse_probe_request(_json(row["request"]), max_bytes=max_bytes)
    factory = cast(Callable[..., H0ProbeLaunch], H0ProbeLaunch)
    result = factory(**row, extensions=tuple(extensions), manifest_schema=cast(str, schema))
    if result.payload != payload:
        raise ValueError("H0 deployment must use canonical JSON")
    return result
