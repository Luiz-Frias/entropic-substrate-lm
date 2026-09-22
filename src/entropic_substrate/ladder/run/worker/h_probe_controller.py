"""Compose one authorized H0 allocation, terminal capture and supervised teardown."""

import math
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from entropic_substrate.__legacy.d1_artifact_snapshot import normalize_d1_artifact_root
from entropic_substrate.__legacy.d1_trainer_object_transport import D1TrainerMailboxObjectPort
from entropic_substrate.ladder.run.h_probe_collection import collect_probe_terminal
from entropic_substrate.ladder.run.h_probe_control import read_probe_control
from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch
from entropic_substrate.ladder.run.h_probe_process import ProbeWatchdogProcess
from entropic_substrate.ladder.run.h_runpod_transport import ProbePod, ProbeProviderError
from entropic_substrate.ladder.run.worker.h_probe_allocation import (
    ProbeAllocationHandoffError,
    ProbeAllocationProvider,
    allocate_probe_once,
)


class ProbeControllerProvider(ProbeAllocationProvider, Protocol):
    def get_owned(self, launch: H0ProbeLaunch, pod_id: str) -> ProbePod | None: ...
    def delete_owned(self, launch: H0ProbeLaunch, pod_id: str) -> str: ...


class ProbeControllerFailure(RuntimeError):
    def __init__(self, record: dict[str, object], causes: tuple[BaseException, ...]) -> None:
        self.record, self.causes = record, causes
        super().__init__("H0 controller failed; inspect retained capture and cleanup evidence")


def _termination(record: dict[str, object], pod_id: str | None) -> bool:
    required = {
        "schema",
        "scientific_authority",
        "status",
        "pod_id",
        "termination_verified",
        "attributed_pod_absent",
        "inventory_verified",
        "unattributed_pod_ids",
        "guard_fault",
    }
    return (
        required <= record.keys()
        and record["schema"] == "h-probe-watchdog-result-v1"
        and record["scientific_authority"] == "none"
        and record["status"] == "terminated"
        and pod_id is not None
        and record["pod_id"] == pod_id
        and record["termination_verified"] is True
        and record["attributed_pod_absent"] is True
        and record["inventory_verified"] is True
        and record["unattributed_pod_ids"] == []
        and record["guard_fault"] is None
    )


def _emergency_cleanup(
    guard: ProbeWatchdogProcess,
    provider: ProbeControllerProvider,
    pod_id: str | None,
) -> dict[str, object]:
    """An explicit emergency attempt, never a substitute for an admitted guard result."""
    if pod_id is None:
        return {"status": "no_attributed_pod", "owned_pod_absent": False}
    if guard.ready is None or provider.account().account_id != guard.ready["account_id"]:
        raise ValueError("emergency cleanup account differs from the supervised account")
    acknowledgement = provider.delete_owned(guard.launch, pod_id)
    if acknowledgement not in ("already_absent", "deletion_acknowledged"):
        raise ValueError("unknown emergency deletion acknowledgement")
    absent = provider.get_owned(guard.launch, pod_id) is None
    if provider.account().account_id != guard.ready["account_id"]:
        raise ValueError("emergency absence observation used a different account")
    return {
        "status": "emergency_attempt",
        "pod_id": pod_id,
        "acknowledgement": acknowledgement,
        "owned_pod_absent": absent,
    }


def run_probe_controller(
    guard: ProbeWatchdogProcess,
    provider: ProbeControllerProvider,
    objects: D1TrainerMailboxObjectPort,
    destination: Path,
    *,
    poll_seconds: float,
    max_hourly_rate: float,
    monotonic: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Run inside supervise_probe_watchdog after explicit spending approval.

    The request/input graph must already be published. No retries create another
    pod. Collection precedes ordinary teardown; failure still requests cleanup.
    A finite scientific gap remains a collected outcome, not a host exception.
    Emergency cleanup retains its own scope and cannot turn a failed guardian
    into success. Callers must inspect unresolved cleanup before further launches.
    """
    for value in (poll_seconds, max_hourly_rate):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("controller polling and price ceiling must be finite and positive")
    if guard.ready is None:
        raise ValueError("controller requires the admitted live watchdog readiness record")
    deadline, cleanup_deadline = (
        guard.ready["controller_deadline"],
        guard.ready["watchdog_deadline"],
    )
    if (type(deadline) is not int and type(deadline) is not float) or (
        type(cleanup_deadline) is not int and type(cleanup_deadline) is not float
    ):
        raise ValueError("controller deadlines must be numeric")
    if (
        not math.isfinite(deadline)
        or not math.isfinite(cleanup_deadline)
        or deadline >= cleanup_deadline
    ):
        raise ValueError("controller deadlines must be finite and ordered")
    root = normalize_d1_artifact_root(destination)
    if any(root.iterdir()):
        raise ValueError("controller capture requires a fresh empty directory")
    errors: list[BaseException] = []
    failure_stages: list[str] = []
    pod: ProbePod | None = None
    pod_id: str | None = None
    collection: dict[str, object] | None = None
    watchdog: dict[str, object] | None = None
    emergency: dict[str, object] | None = None
    now = monotonic()
    if type(now) not in (int, float) or not math.isfinite(now):
        raise ValueError("controller clock must start finite")

    def clock() -> float:
        nonlocal now
        value = monotonic()
        if type(value) not in (int, float) or not math.isfinite(value) or value < now:
            raise ValueError("controller clock must be finite and nondecreasing")
        now = float(value)
        return now

    stage = "allocation"
    try:
        if clock() >= deadline:
            raise TimeoutError("controller deadline expired before allocation")
        pod = allocate_probe_once(guard, provider, objects)
        pod_id = pod.pod_id
        stage = "allocation_custody"
        guard.control.write(
            "pod",
            {"schema": "h-probe-pod-observation-v1", **asdict(pod)},
            max_bytes=guard.config.max_control_bytes,
        )
        stage = "price_admission"
        if (
            type(pod.cost_per_hour) is not int and type(pod.cost_per_hour) is not float
        ) or not 0 <= pod.cost_per_hour <= max_hourly_rate:
            raise ValueError("measured allocation price is missing or exceeds the approved ceiling")
        while clock() < deadline:
            stage = "watchdog_liveness"
            guard.require_alive()
            stage = "terminal_capture"
            collection = collect_probe_terminal(
                objects,
                guard.launch.request,
                root,
                expected_pod_id=pod_id,
                max_marker_bytes=guard.launch.request.limits.max_object_bytes,
            )
            if collection is not None:
                break
            remaining = deadline - clock()
            if remaining > 0:
                wait(min(float(poll_seconds), remaining))
        if collection is None:
            raise TimeoutError("controller deadline expired without terminal evidence")
    except BaseException as error:
        errors.append(error)
        failure_stages.append(stage)
        if isinstance(error, ProbeAllocationHandoffError):
            pod_id = error.pod_id
        elif isinstance(error, ProbeProviderError):
            pod_id = error.possible_pod_id

    stop_failed = False
    try:
        allocation = read_probe_control(
            guard.root,
            guard.launch,
            "controller",
            "allocation",
            max_bytes=guard.config.max_control_bytes,
        )
        recorded_id = None if allocation is None else allocation["pod_id"]
        if recorded_id != pod_id:
            raise ValueError("allocation control differs from the attributed provider response")
        reason = "cancelled"
        if collection is not None and recorded_id is not None:
            reason = (
                "capture_verified" if collection["workload_verified"] is True else "partial_capture"
            )
        guard.control.write(
            "stop",
            {
                "schema": "h-probe-stop-v1",
                "guard_pid": guard.pid,
                "reason": reason,
                "pod_id": recorded_id,
            },
            max_bytes=guard.config.max_control_bytes,
        )
    except BaseException as error:
        errors.append(error)
        failure_stages.append("stop_handoff")
        stop_failed = True

    def emergency_attempt() -> None:
        nonlocal emergency
        try:
            emergency = _emergency_cleanup(guard, provider, pod_id)
        except BaseException as error:
            errors.append(error)
            failure_stages.append("emergency_cleanup")
            emergency = {"status": "emergency_failed", "owned_pod_absent": False}

    if (
        stop_failed
        or not guard.alive
        or any(isinstance(e, ProbeAllocationHandoffError) for e in errors)
    ):
        emergency_attempt()
    try:
        timeout = max(0.0, float(cleanup_deadline) - clock())
        timeout += 3 * guard.config.provider_timeout_seconds + guard.config.shutdown_timeout_seconds
        watchdog = guard.wait_result(timeout_seconds=timeout)
        if not _termination(watchdog, pod_id):
            raise ValueError("watchdog did not verify this allocation's complete termination")
    except BaseException as error:
        errors.append(error)
        failure_stages.append("watchdog_join")
        if not guard.alive and emergency is None:
            emergency_attempt()

    terminated = watchdog is not None and _termination(watchdog, pod_id)
    verified = collection is not None and collection.get("workload_verified") is True
    record: dict[str, object] = {
        "schema": "h-probe-controller-result-v1",
        "scientific_authority": "none",
        "request_sha256": guard.launch.request.digest,
        "launch_sha256": guard.launch.digest,
        "pod_id": pod_id,
        "pod": None if pod is None else asdict(pod),
        "status": "controller_failed" if errors else "terminal_collected",
        "workload_verified": verified,
        "termination_verified": terminated,
        "collection": collection,
        "watchdog": watchdog,
        "emergency_cleanup": emergency,
        "error_types": [type(error).__name__ for error in errors],
        "failure_stages": failure_stages,
    }
    try:
        guard.control.write("result", record, max_bytes=guard.config.max_control_bytes)
    except BaseException as error:
        errors.append(error)
        failure_stages.append("result_custody")
        record.update(status="controller_failed", error_types=[type(e).__name__ for e in errors])
    if errors:
        raise ProbeControllerFailure(record, tuple(errors)) from errors[0]
    return record
