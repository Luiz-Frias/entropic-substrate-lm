"""Explicit H0-only deployment choices for the pinned Runpod command boundary."""

import hashlib
import json
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import cast

from entropic_substrate.sentence_prediction.recovery_state import recovery_record
from entropic_substrate.sentence_prediction.worker_request import ProbeRequest, parse_probe_request

RUNPODCTL_VERSION = "runpodctl 2.14.0-dd55bcf"
RUNPODCTL_SHA256 = (
    "377ea9f6b72c68cb766be2784f137ab4cefdd8d10022f0f2910104d386e48918"  # pragma: allowlist secret
)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


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
    controller_timeout_seconds: int
    watchdog_timeout_seconds: int
    cost_guard: str

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
        for name in (self.r2_access_secret_name, self.r2_private_secret_name):
            if type(name) is not str or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", name) is None:
                raise ValueError("R2 credentials must use provider secret-reference names")
        if self.r2_access_secret_name == self.r2_private_secret_name:
            raise ValueError("R2 secret-reference names must be distinct")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "schema": "h0-probe-launch-v1",
            "scientific_authority": "none",
            "spending_authority": "none",
            "provider_deadline": None,
            "runpodctl_version": RUNPODCTL_VERSION,
            "runpodctl_sha256": RUNPODCTL_SHA256,
            **{f.name: getattr(self, f.name) for f in fields(self) if f.name != "request"},
            "request": self.request.manifest,
        }

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
        return {
            "AWS_ACCESS_KEY_ID": f"{{{{ RUNPOD_SECRET_{self.r2_access_secret_name} }}}}",
            "AWS_SECRET_ACCESS_KEY": f"{{{{ RUNPOD_SECRET_{self.r2_private_secret_name} }}}}",
        }

    def validate_template(self, value: object) -> None:
        """Check a pinned-CLI template get result before creating the pod.

        CLI 2.14.0 omits false booleans, zero volumes and empty command/port lists.
        Their absence here follows that serialization contract, not a guessed
        scientific default. A provider read must still succeed independently.
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
        """Arguments for verified runpodctl 2.14.0; execution belongs to the controller.

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
            "entropic_substrate.sentence_prediction.worker_command",
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


def parse_probe_launch(payload: bytes, *, max_bytes: int) -> H0ProbeLaunch:
    """Require every deployment choice and canonical bytes; never fill defaults."""
    if (
        type(max_bytes) is not int
        or max_bytes < 1
        or type(payload) is not bytes
        or not 0 < len(payload) <= max_bytes
    ):
        raise ValueError("H0 deployment exceeds its explicit byte budget")
    names = " ".join(f.name for f in fields(H0ProbeLaunch))
    row = recovery_record(
        json.loads(payload),
        f"schema scientific_authority spending_authority provider_deadline runpodctl_version runpodctl_sha256 {names}",
    ).copy()
    if (
        row.pop("schema") != "h0-probe-launch-v1"
        or row.pop("scientific_authority") != "none"
        or row.pop("spending_authority") != "none"
        or row.pop("provider_deadline") is not None
        or row.pop("runpodctl_version") != RUNPODCTL_VERSION
        or row.pop("runpodctl_sha256") != RUNPODCTL_SHA256
    ):
        raise ValueError("H0 deployment scope or authority mismatch")
    row["request"] = parse_probe_request(_json(row["request"]), max_bytes=max_bytes)
    factory = cast(Callable[..., H0ProbeLaunch], H0ProbeLaunch)
    result = factory(**row)
    if result.payload != payload:
        raise ValueError("H0 deployment must use canonical JSON")
    return result
