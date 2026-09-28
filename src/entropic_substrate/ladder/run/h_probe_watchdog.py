"""Independent H0 teardown policy; a controller supplies attributed allocation evidence."""

import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.lifecycle import ContentHash
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.h_probe_control import own_probe_role, read_probe_control
from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch
from entropic_substrate.ladder.run.h_runpod_transport import (
    ProbeAccount,
    ProbePod,
    ProbeProviderError,
)

UNCERTAIN_ABSENT_CONFIRMATIONS = 3
"""K consecutive valid empty inventory reads required to resolve a cancelled,
never-attributed create as `cancelled_uncertain_absent` (see run_probe_watchdog)."""


class WatchdogProvider(Protocol):
    def account(self) -> ProbeAccount: ...
    def check_template(self, launch: H0ProbeLaunch) -> None: ...
    def list_owned(self, launch: H0ProbeLaunch) -> tuple[ProbePod, ...]: ...
    def get_owned(self, launch: H0ProbeLaunch, pod_id: str) -> ProbePod | None: ...
    def delete_owned(self, launch: H0ProbeLaunch, pod_id: str) -> str: ...


class ProbeWatchdogFailure(RuntimeError):
    def __init__(self, record: dict[str, object]) -> None:
        self.record = record
        super().__init__(f"probe watchdog: {record['status']}")


def _message(
    root: Path, launch: H0ProbeLaunch, name: str, fields: str, maximum: int
) -> dict[str, object] | None:
    value = read_probe_control(root, launch, "controller", name, max_bytes=maximum)
    if value is None:
        return None
    row = recovery_record(value, f"schema guard_pid {fields}")
    if (
        row["schema"] != f"h-probe-{name}-v1"
        or type(row["guard_pid"]) is not int
        or row["guard_pid"] != os.getpid()
    ):
        raise ValueError("control record belongs to another watchdog")
    return row


_ALLOCATION_V1_FIELDS = "outcome pod_id response_sha256"
_ALLOCATION_V2_FIELDS = (
    "outcome pod_id response_sha256 provider_error_code allocation_uncertain "
    "possible_pod_id provider_detail provider_detail_sha256"
)


def _allocation_message(
    root: Path, launch: H0ProbeLaunch, maximum: int
) -> dict[str, object] | None:
    """Read-old, write-new: accept exactly the v1 field set or exactly the v2 one."""
    value = read_probe_control(root, launch, "controller", "allocation", max_bytes=maximum)
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("control record has an invalid shape")
    schema = value.get("schema")
    if schema == "h-probe-allocation-v2":
        fields = _ALLOCATION_V2_FIELDS
    elif schema == "h-probe-allocation-v1":
        fields = _ALLOCATION_V1_FIELDS
    else:
        raise ValueError("unknown allocation schema version")
    row = recovery_record(value, f"schema guard_pid {fields}")
    if type(row["guard_pid"]) is not int or row["guard_pid"] != os.getpid():
        raise ValueError("control record belongs to another watchdog")
    return row


def _allocation(row: dict[str, object]) -> tuple[str | None, bool]:
    outcome, pod_id, digest = row["outcome"], row["pod_id"], row["response_sha256"]
    if outcome not in ("created", "uncertain"):
        raise ValueError("unknown allocation outcome")
    if pod_id is not None:
        pod_id = require_d1_pod_id(pod_id)
    if digest is not None:
        if type(digest) is not str:
            raise ValueError("allocation response hash must be text")
        ContentHash("sha256", digest)
    if outcome == "created" and (pod_id is None or digest is None):
        raise ValueError("created allocation requires its pod and response identities")
    if "provider_error_code" in row:
        code = row["provider_error_code"]
        if code is not None and type(code) is not str:
            raise ValueError("provider error code must be text or absent")
        uncertain_flag = row["allocation_uncertain"]
        if type(uncertain_flag) is not bool:
            raise ValueError("allocation_uncertain must be boolean")
        possible = row["possible_pod_id"]
        if possible is not None:
            require_d1_pod_id(possible)
        detail = row["provider_detail"]
        if detail is not None and type(detail) is not str:
            raise ValueError("provider detail must be text or absent")
        detail_digest = row["provider_detail_sha256"]
        if detail_digest is not None:
            if type(detail_digest) is not str:
                raise ValueError("provider detail hash must be text")
            ContentHash("sha256", detail_digest)
        if outcome == "created" and (code is not None or uncertain_flag is not False):
            raise ValueError("created allocation cannot carry a provider failure")
    return pod_id, outcome == "uncertain"


def _clock(clock: Callable[[], float], previous: float) -> float:
    value = clock()
    if type(value) not in (int, float) or not math.isfinite(value) or value < previous:
        raise ValueError("watchdog requires a finite nondecreasing clock")
    return float(value)


def run_probe_watchdog(
    root: Path,
    launch: H0ProbeLaunch,
    provider: WatchdogProvider,
    *,
    poll_seconds: float,
    max_control_bytes: int,
    monotonic: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Own teardown independently of the trainer/controller; never create a pod.

    Forced cleanup starts at the controller deadline; the watchdog deadline bounds
    cleanup polling. Individual provider calls must themselves be bounded. A host
    outage or unresolved provider call cannot be turned into a cost guarantee.
    The caller must supervise this process and verify its live PID before create;
    the ready file alone is not liveness or spending authorization.
    """
    if (
        type(poll_seconds) not in (int, float)
        or not math.isfinite(poll_seconds)
        or poll_seconds <= 0
    ):
        raise ValueError("watchdog polling interval must be finite and positive")
    if type(max_control_bytes) is not int or max_control_bytes < 1:
        raise ValueError("watchdog control bound must be positive")
    with own_probe_role(root, launch, "watchdog") as writer:
        account = provider.account()
        provider.check_template(launch)
        if provider.list_owned(launch):
            raise ValueError("preexisting candidates require separate reconciliation before arming")
        origin = _clock(monotonic, 0)
        controller_deadline = origin + launch.controller_timeout_seconds
        watchdog_deadline = origin + launch.watchdog_timeout_seconds
        writer.write(
            "ready",
            {
                "schema": "h-probe-watchdog-ready-v1",
                "pid": os.getpid(),
                "account_id": account.account_id,
                "observed_at_utc": datetime.now(UTC).isoformat(),
                "monotonic_origin": origin,
                "controller_deadline": controller_deadline,
                "watchdog_deadline": watchdog_deadline,
            },
            max_bytes=max_control_bytes,
        )
        now, pod_id, intent_seen = origin, None, False
        uncertain, acknowledged, verified = False, False, False
        reason, fault, last_error = None, None, None
        candidates: set[str] = set()
        errors = 0
        inventory_valid = False
        absent_confirmations = 0
        unattributed_seen = False
        resolved_uncertain_absent = False

        def note_error(error: ProbeProviderError | ValueError) -> None:
            nonlocal errors, last_error, fault
            errors += 1
            last_error = (
                error.code if isinstance(error, ProbeProviderError) else type(error).__name__
            )
            if not isinstance(error, ProbeProviderError) or error.code not in (
                "network_error",
                "timeout",
                "rate_limited",
                "server_error",
            ):
                fault = last_error
            writer.write(
                f"error-{errors:08d}",
                {"error_code": last_error, "elapsed_seconds": now - origin},
                max_bytes=max_control_bytes,
            )

        while now < watchdog_deadline:
            try:
                if fault is None:
                    intent = _message(root, launch, "create-intent", "", max_control_bytes)
                    intent_seen = intent is not None
                    allocation = _allocation_message(root, launch, max_control_bytes)
                    if allocation is not None:
                        if not intent_seen:
                            raise ValueError("allocation lacks durable create intent")
                        observed_id, uncertain = _allocation(allocation)
                        if pod_id is not None and pod_id != observed_id:
                            raise ValueError("allocation identity changed")
                        pod_id = observed_id
                    elif intent_seen:
                        uncertain = True
                    stop = _message(root, launch, "stop", "reason pod_id", max_control_bytes)
                    if stop is not None:
                        if (
                            stop["reason"]
                            not in ("capture_verified", "partial_capture", "cancelled")
                            or stop["pod_id"] != pod_id
                        ):
                            raise ValueError("stop record differs from the attributed allocation")
                        if pod_id is None and stop["reason"] != "cancelled":
                            raise ValueError("capture stop requires an attributed pod")
                        reason = str(stop["reason"])
                if provider.account().account_id != account.account_id:
                    raise ValueError("provider account changed")
                inventory_valid = False
                try:
                    observed = provider.list_owned(launch)
                    current_candidates: set[str] = set()
                    for pod in observed:
                        if (
                            type(pod) is not ProbePod
                            or pod.name != launch.pod_name
                            or pod.image != launch.request.image
                        ):
                            raise ValueError("inventory supplied a foreign candidate")
                        current_candidates.add(require_d1_pod_id(pod.pod_id))
                    candidates, inventory_valid = current_candidates, True
                except (ProbeProviderError, ValueError) as error:
                    note_error(error)
                if pod_id is None and inventory_valid and candidates:
                    # A candidate appeared while nothing is attributed: never adopt
                    # or delete it here (unchanged conservative behavior below), and
                    # never again resolve this run as uncertain-absent either.
                    unattributed_seen = True
                if reason == "cancelled" and pod_id is None and inventory_valid and not candidates:
                    absent_confirmations += 1
                else:
                    absent_confirmations = 0
                now = _clock(monotonic, now)
                if now >= watchdog_deadline:
                    break
                if now >= controller_deadline and reason is None:
                    reason = "controller_deadline"
                if reason is not None or fault is not None:
                    if pod_id is not None:
                        current = provider.get_owned(launch, pod_id)
                        if current is not None and not acknowledged:
                            if (
                                current.pod_id != pod_id
                                or current.name != launch.pod_name
                                or current.image != launch.request.image
                            ):
                                raise ValueError("ownership check supplied a foreign pod")
                            now = _clock(monotonic, now)
                            if now >= watchdog_deadline:
                                break
                            answer = provider.delete_owned(launch, pod_id)
                            if answer not in ("already_absent", "deletion_acknowledged"):
                                raise ValueError("unrecognized deletion acknowledgement")
                            acknowledged = True
                            current = provider.get_owned(launch, pod_id)
                        if current is None:
                            if provider.account().account_id != account.account_id:
                                raise ValueError("absence check used a different account")
                            verified = True
                            now = _clock(monotonic, now)
                            if inventory_valid and not candidates - {pod_id}:
                                break
                    elif not intent_seen and inventory_valid and not candidates and fault is None:
                        break
                    elif (
                        intent_seen
                        and reason == "cancelled"
                        and fault is None
                        and not unattributed_seen
                        and inventory_valid
                        and not candidates
                        and absent_confirmations >= UNCERTAIN_ABSENT_CONFIRMATIONS
                    ):
                        verified = True
                        resolved_uncertain_absent = True
                        break
            except (ProbeProviderError, ValueError) as error:
                note_error(error)
            now = _clock(monotonic, now)
            if now >= watchdog_deadline:
                break
            wait(min(float(poll_seconds), watchdog_deadline - now))
            now = _clock(monotonic, now)
        status = (
            "terminated"
            if verified and inventory_valid and not candidates - {pod_id} and fault is None
            else "cleanup_unresolved"
        )
        if not intent_seen and inventory_valid and not candidates and fault is None:
            status = (
                "cancelled_before_create" if reason == "cancelled" else "expired_without_intent"
            )
        if resolved_uncertain_absent:
            status = "cancelled_uncertain_absent"
        record: dict[str, object] = {
            "schema": "h-probe-watchdog-result-v1",
            "scientific_authority": "none",
            "status": status,
            "reason": reason,
            "pod_id": pod_id,
            "termination_verified": verified and inventory_valid and not candidates - {pod_id},
            "attributed_pod_absent": verified,
            "inventory_verified": inventory_valid,
            "allocation_uncertain": uncertain,
            "unattributed_pod_ids": sorted(candidates - {pod_id}),
            "guard_fault": fault,
            "provider_error_count": errors,
            "last_error": last_error,
            "elapsed_seconds": now - origin,
            "observed_at_utc": datetime.now(UTC).isoformat(),
        }
        writer.write("result", record, max_bytes=max_control_bytes)
        if status == "cleanup_unresolved":
            raise ProbeWatchdogFailure(record)
        return record
