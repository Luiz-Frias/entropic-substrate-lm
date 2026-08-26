"""Structured-JSON runpodctl adapter for the D1 accelerator boundary."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import NoReturn, cast

from entropic_substrate.d1_command_runner import SubprocessD1CommandRunner
from entropic_substrate.d1_operator_ports import (
    D1CommandResult,
    D1CommandRunner,
    D1OperatorTransportError,
    D1PodDeletionAcknowledgedError,
    D1RunPodAuthReceipt,
)
from entropic_substrate.d1_operator_values import (
    D1OperatorLaunchIntent,
    D1PodPlacement,
    require_d1_pod_id,
)
from entropic_substrate.evidence.resource_billing import (
    BillingGap,
    ExecutionSegmentTermination,
    ProviderBilling,
)
from entropic_substrate.evidence.resource_provenance import (
    BillingUnit,
    EvidenceGap,
    Money,
    PurchaseOption,
    RunStartProvenance,
)
from entropic_substrate.experiments.d1_primitives import EvidenceClass

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_GO_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{1,9})? [+-]\d{4} UTC$")
_WAIT_TIMEOUT = re.compile(r"^[1-9][0-9]*[smhd]$")
_FAILURE_CODE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_RUNPODCTL_VERSION = "runpodctl 2.9.0-c094cac"
_POD_REQUIRED = frozenset(
    "id name desiredStatus runtimeStatus imageName gpuCount createdAt machine ssh".split()
)
_SSH_REQUIRED = frozenset({"ssh_command", "ip", "port"})
_MARKETS = {"secure-cloud": "SECURE", "community-cloud": "COMMUNITY"}


def _error(operation: str, subject: str, detail: str, cause: Exception | None = None) -> NoReturn:
    del cause
    error = D1OperatorTransportError(operation=operation, subject=subject, detail=detail)
    raise error from None


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("provider JSON repeats a key")
        result[key] = value
    return result


def _nonfinite(value: str) -> NoReturn:
    del value
    raise ValueError("provider JSON contains a non-finite constant")


def _decode(raw: str, *, operation: str, subject: str) -> dict[str, object]:
    try:
        value: object = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (json.JSONDecodeError, ValueError):
        _error(operation, subject, "provider response is invalid JSON")
    if type(value) is not dict:
        _error(operation, subject, "provider response must be a JSON object")
    return cast(dict[str, object], value)


def _exact(value: dict[str, object], keys: set[str], *, operation: str, subject: str) -> None:
    if set(value) != keys:
        _error(operation, subject, "provider response keys differ from the adapter schema")


def _text(value: object, *, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _timestamp(value: object, *, label: str) -> datetime:
    text = _text(value, label=label)
    if _TIMESTAMP.fullmatch(text) is not None:
        return datetime.fromisoformat(text[:-1] + "+00:00")
    if _GO_TIMESTAMP.fullmatch(text) is not None:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f %z UTC")
    raise ValueError(f"{label} must be a supported RunPod UTC timestamp")


def _integer(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


class RunpodctlD1Adapter:
    """Use configured runpodctl auth; no credential value crosses this API."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        registry_auth_id: str,
        network_volume_id: str,
        network_volume_mount_path: str,
        wait_timeout: str,
        runner: D1CommandRunner | None = None,
    ) -> None:
        selected = SubprocessD1CommandRunner() if runner is None else runner
        if not hasattr(selected, "run") or not callable(clock):
            raise TypeError("RunPod adapter requires a command runner")
        mount = PurePosixPath(_text(network_volume_mount_path, label="network volume mount"))
        if not mount.is_absolute() or ".." in mount.parts or mount.as_posix() != str(mount):
            raise ValueError("RunPod network volume mount must be a normalized absolute POSIX path")
        if _WAIT_TIMEOUT.fullmatch(wait_timeout) is None:
            raise ValueError("RunPod wait timeout must be a positive duration")
        self._runner, self._clock = selected, clock
        self._registry_auth_id = _text(registry_auth_id, label="registry auth id")
        self._network_volume_id = _text(network_volume_id, label="network volume id")
        self._network_volume_mount_path = str(mount)
        self._wait_timeout = wait_timeout

    def _now(self) -> datetime:
        value = self._clock()
        if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("RunPod control clock must return an exact UTC datetime")
        return value

    @staticmethod
    def _coded_failure(stderr: str) -> tuple[str, int | None, str | None, bool] | None:
        lines = [line for line in stderr.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            value: object = json.loads(
                lines[-1],
                object_pairs_hook=_pairs,
                parse_constant=_nonfinite,
            )
        except (json.JSONDecodeError, ValueError):
            return None
        if type(value) is not dict:
            return None
        root = cast(dict[str, object], value)
        if not {"error", "code"} <= set(root) or set(root) - {"error", "code", "status", "id"}:
            return None
        code, message, status, resource_id = (
            root.get("code"),
            root.get("error"),
            root.get("status"),
            root.get("id"),
        )
        if (
            type(code) is not str
            or _FAILURE_CODE.fullmatch(code) is None
            or type(message) is not str
            or (status is not None and type(status) is not int)
        ):
            return None
        unsafe_resource_id = False
        if resource_id is not None:
            try:
                resource_id = require_d1_pod_id(resource_id)
            except (TypeError, ValueError):
                resource_id, unsafe_resource_id = None, True
        return code, status, resource_id, unsafe_resource_id

    @staticmethod
    def _failure_detail(code: str, status: int | None) -> str:
        suffix = "" if status is None else f" status={status}"
        return f"runpodctl failed code={code}{suffix}"

    def _execute(
        self,
        operation: str,
        subject: str,
        argv: tuple[str, ...],
        *,
        stdin: str | None = None,
    ) -> D1CommandResult:
        try:
            return self._runner.run(argv, cwd=None, stdin=stdin)
        except Exception:
            _error(operation, subject, "command execution failed")

    def _invoke(
        self,
        operation: str,
        subject: str,
        argv: tuple[str, ...],
        *,
        stdin: str | None = None,
    ) -> D1CommandResult:
        result = self._execute(operation, subject, argv, stdin=stdin)
        if result.returncode != 0:
            failure = self._coded_failure(result.stderr)
            if failure is None:
                _error(operation, subject, "runpodctl failure payload is malformed")
            code, status, _, _ = failure
            _error(operation, subject, self._failure_detail(code, status))
        return result

    def _delete_attempt(self, pod_id: str, *, operation: str) -> tuple[str, int | None] | None:
        pod_id = require_d1_pod_id(pod_id)
        result = self._execute(
            operation,
            pod_id,
            ("runpodctl", "pod", "delete", pod_id, "--output", "json"),
        )
        if result.returncode != 0:
            failure = self._coded_failure(result.stderr)
            if failure is None:
                return "uncoded_failure", None
            code, status, _, _ = failure
            return code, status
        payload = _decode(result.stdout, operation=operation, subject=pod_id)
        _exact(payload, {"deleted", "id"}, operation=operation, subject=pod_id)
        if payload["deleted"] is not True or payload["id"] != pod_id:
            _error(operation, pod_id, "termination response identity or schema differs")
        return None

    def _cleanup_and_raise(self, subject: str, resource_id: str, primary: str) -> NoReturn:
        try:
            cleanup = self._delete_attempt(resource_id, operation="cleanup_pod")
        except D1OperatorTransportError:
            cleanup = ("uncoded_failure", None)
        if cleanup is None:
            _error(
                "create_pod",
                subject,
                f"{primary}; resource_id={resource_id}; cleanup=deleted",
            )
        cleanup_code, cleanup_status = cleanup
        status = "" if cleanup_status is None else f" cleanup_status={cleanup_status}"
        _error(
            "create_pod",
            subject,
            f"{primary}; COST LEAK: resource_id={resource_id} may still be billing; "
            f"cleanup_code={cleanup_code}{status}",
        )

    def _run(
        self,
        operation: str,
        subject: str,
        argv: tuple[str, ...],
        *,
        stdin: str | None = None,
    ) -> dict[str, object]:
        result = self._invoke(operation, subject, argv, stdin=stdin)
        return _decode(result.stdout, operation=operation, subject=subject)

    def verify_authenticated(self) -> D1RunPodAuthReceipt:
        operation = "authenticate"
        version = self._invoke(
            operation,
            "configured-session",
            ("runpodctl", "version"),
        )
        if version.stdout.removesuffix("\n") != _RUNPODCTL_VERSION:
            _error(operation, "configured-session", "unsupported runpodctl version")
        payload = self._run(
            operation,
            "configured-session",
            ("runpodctl", "user", "--output", "json"),
        )
        try:
            return D1RunPodAuthReceipt(_text(payload.get("id"), label="account id"))
        except (TypeError, ValueError):
            _error(operation, "configured-session", "RunPod user identity is invalid")

    def create(self, intent: D1OperatorLaunchIntent) -> D1PodPlacement:
        if type(intent) is not D1OperatorLaunchIntent:
            raise TypeError("RunPod create requires D1OperatorLaunchIntent")
        if intent.resource.provider != "runpod":
            raise ValueError("RunPod adapter requires provider='runpod'")
        if intent.resource.purchase_option is not PurchaseOption.ON_DEMAND:
            raise ValueError("RunPod adapter requires ON_DEMAND purchase option")
        if intent.resource.provider_sku != intent.resource.provider_gpu_type_id:
            raise ValueError("RunPod provider_sku must equal the measurable gpuTypeId")
        cloud_type = _MARKETS.get(intent.resource.market)
        if cloud_type is None:
            raise ValueError("RunPod market must map to secure or community cloud")
        subject = intent.plan.logical_run_id
        terminate_after = self._now() + timedelta(seconds=intent.terminate_after_seconds)
        argv = [
            "runpodctl",
            "pod",
            "create",
            "--name",
            subject,
            "--image",
            intent.image.reference,
            "--gpu-id",
            intent.resource.provider_gpu_type_id,
            "--gpu-count",
            str(intent.resource.accelerator_count),
            "--cloud-type",
            cloud_type,
            "--data-center-ids",
            intent.resource.region,
        ]
        if cloud_type == "COMMUNITY":
            argv.append("--public-ip")
        argv.extend(
            [
                "--ssh",
                "--ports",
                "22/tcp",
                "--registry-auth-id",
                self._registry_auth_id,
                "--network-volume-id",
                self._network_volume_id,
                "--volume-mount-path",
                self._network_volume_mount_path,
                "--terminate-after",
                terminate_after.isoformat().replace("+00:00", "Z"),
                "--wait",
                "--wait-timeout",
                self._wait_timeout,
                "--output",
                "json",
            ]
        )
        try:
            result = self._execute("create_pod", subject, tuple(argv))
        except D1OperatorTransportError:
            _error(
                "create_pod",
                subject,
                "create command outcome is unknown; allocation state is unknown; COST RISK: "
                "reconcile by logical run name; automatic cleanup impossible",
            )
        if result.returncode != 0:
            failure = self._coded_failure(result.stderr)
            if failure is None:
                _error(
                    "create_pod",
                    subject,
                    "runpodctl failure payload is malformed; COST RISK: allocation state is "
                    "unknown; reconcile by logical run name; automatic cleanup impossible",
                )
            code, status, resource_id, unsafe_resource_id = failure
            primary = self._failure_detail(code, status)
            if unsafe_resource_id:
                _error(
                    "create_pod",
                    subject,
                    f"{primary}; COST LEAK: provider returned an unsafe pod id; cleanup impossible",
                )
            if resource_id is not None:
                self._cleanup_and_raise(subject, resource_id, primary)
            _error(
                "create_pod",
                subject,
                f"{primary}; COST RISK: allocation state is unknown; reconcile by logical "
                "run name; automatic cleanup impossible",
            )
        try:
            payload = _decode(result.stdout, operation="create_pod", subject=subject)
        except D1OperatorTransportError:
            _error(
                "create_pod",
                subject,
                "COST LEAK: provider reported success without a usable pod response; "
                "cleanup impossible",
            )
        try:
            resource_id = require_d1_pod_id(payload.get("id"))
        except (TypeError, ValueError):
            _error(
                "create_pod",
                subject,
                "COST LEAK: provider reported success without a safe pod id; cleanup impossible",
            )
        try:
            if not _POD_REQUIRED <= set(payload):
                raise ValueError("pod response fields differ from the v2.9 schema")
            machine, ssh = payload["machine"], payload["ssh"]
            if type(machine) is not dict or machine.get("dataCenterId") != intent.resource.region:
                raise ValueError("pod region differs from request")
            observed_gpu = machine.get("gpuId")
            if observed_gpu != intent.resource.provider_gpu_type_id:
                raise ValueError("pod GPU differs from request")
            if machine.get("secureCloud") is not (cloud_type == "SECURE"):
                raise ValueError("pod cloud market differs from request")
            if (
                type(ssh) is not dict
                or not _SSH_REQUIRED <= set(ssh)
                or "error" in ssh
                or "setup" in ssh
            ):
                raise ValueError("pod SSH is not usable")
            ssh_key = ssh.get("ssh_key")
            if (
                type(ssh_key) is not dict
                or ssh_key.get("exists") is not True
                or ssh_key.get("in_account") is not True
            ):
                raise ValueError("pod SSH key is not usable")
            if ssh.get("id") != resource_id or ssh.get("name") != subject:
                raise ValueError("pod SSH identity differs from placement")
            if payload["desiredStatus"] != "RUNNING":
                raise ValueError("pod desiredStatus is not RUNNING")
            if payload["runtimeStatus"] != "running":
                raise ValueError("pod runtimeStatus is not running")
            if payload["imageName"] != intent.image.reference:
                raise ValueError("pod image differs from request")
            if payload["gpuCount"] != intent.resource.accelerator_count:
                raise ValueError("pod accelerator count differs from request")
            if payload["name"] != subject:
                raise ValueError("pod name differs from request")
            _text(ssh["ssh_command"], label="SSH command")
            _text(ssh["ip"], label="SSH ip")
            port = _integer(ssh["port"], label="SSH port")
            if not 1 <= port <= 65535:
                raise ValueError("SSH port must be usable")
            # The on-demand mutation and this adapter's provider identity are
            # command constraints; GPU/count/region/cloud/timestamp are response facts.
            return D1PodPlacement(
                pod_id=resource_id,
                accelerator_form_factor=intent.resource.accelerator_form_factor,
                accelerator_model=_text(observed_gpu, label="GPU type"),
                accelerator_count=_integer(payload["gpuCount"], label="GPU count"),
                provider=intent.resource.provider,
                provider_sku=_text(observed_gpu, label="GPU type"),
                region=intent.resource.region,
                market=intent.resource.market,
                purchase_option=intent.resource.purchase_option,
                pod_created_at=_timestamp(payload["createdAt"], label="pod createdAt"),
            )
        except (TypeError, ValueError) as error:
            self._cleanup_and_raise(subject, resource_id, str(error))

    def cancel(self, pod_id: str) -> datetime:
        pod = require_d1_pod_id(pod_id)
        operation = "terminate_pod"
        try:
            self._now()
        except Exception:
            raise ValueError("RunPod pre-delete clock is not valid UTC") from None
        failure = self._delete_attempt(pod, operation=operation)
        if failure is not None:
            code, status = failure
            _error(operation, pod, self._failure_detail(code, status))
        try:
            return self._now()
        except Exception:
            raise D1PodDeletionAcknowledgedError(
                pod_id=pod,
                detail="post-delete UTC measurement is invalid; do not retry deletion",
            ) from None

    def terminate(
        self,
        pod_id: str,
        *,
        start: RunStartProvenance,
        trainer_ended_at: datetime,
    ) -> ExecutionSegmentTermination:
        pod = require_d1_pod_id(pod_id)
        if type(start) is not RunStartProvenance:
            raise TypeError("RunPod termination requires RunStartProvenance")
        if (
            type(trainer_ended_at) is not datetime
            or trainer_ended_at.tzinfo is None
            or trainer_ended_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("trainer_ended_at must be an exact UTC datetime")
        pod_terminated_at = self.cancel(pod)
        try:
            return ExecutionSegmentTermination(
                start.logical_run_id,
                start.execution_segment_index,
                start.execution_segment_id,
                trainer_ended_at,
                pod_terminated_at,
            )
        except Exception:
            raise D1PodDeletionAcknowledgedError(
                pod_id=pod,
                detail="termination evidence is invalid after deletion; do not retry deletion",
            ) from None

    def billing(self, pod_id: str, *, start: RunStartProvenance) -> ProviderBilling | BillingGap:
        pod = require_d1_pod_id(pod_id)
        if type(start) is not RunStartProvenance:
            raise TypeError("RunPod billing requires RunStartProvenance")
        operation = "provider_billing"
        payload = self._run(
            operation,
            pod,
            ("runpodctl", "billing", "pod", pod, "--output", "json"),
        )
        if payload.get("pod_id") != pod or payload.get("schema_version") != "d1-runpod-billing-v1":
            _error(operation, pod, "billing response identity or schema differs")
        if payload.get("kind") == "billing_gap":
            _exact(
                payload,
                {"kind", "pod_id", "reason", "schema_version"},
                operation=operation,
                subject=pod,
            )
            reason = _text(payload["reason"], label="billing gap reason")
            return BillingGap(
                start.logical_run_id,
                start.execution_segment_index,
                start.execution_segment_id,
                EvidenceGap("provider_billed_duration_seconds", reason),
                EvidenceGap("provider_billed_amount", reason),
                EvidenceGap("provider_billing_unit", reason),
                EvidenceGap("provider_billing_record_id", reason),
            )
        keys = {
            "billed_amount",
            "billed_duration_seconds",
            "billing_record_id",
            "billing_unit",
            "kind",
            "pod_id",
            "schema_version",
        }
        _exact(payload, keys, operation=operation, subject=pod)
        if payload["kind"] != "provider_billing":
            _error(operation, pod, "unsupported billing response kind")
        try:
            money = cast(dict[str, object], payload["billed_amount"])
            return ProviderBilling(
                start.logical_run_id,
                start.execution_segment_index,
                start.execution_segment_id,
                Decimal(_text(payload["billed_duration_seconds"], label="duration")),
                Money(
                    Decimal(_text(money["amount"], label="billing amount")),
                    _text(money["currency"], label="billing currency"),
                ),
                BillingUnit(payload["billing_unit"]),
                _text(payload["billing_record_id"], label="billing_record_id"),
                EvidenceClass.PROVIDER_BILLED,
            )
        except (InvalidOperation, KeyError, TypeError, ValueError) as error:
            _error(operation, pod, f"invalid provider billing: {error}", error)
