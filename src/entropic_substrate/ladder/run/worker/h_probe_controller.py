"""Compose one authorized H0 allocation, terminal capture and supervised teardown.

Host narration (FOR-1645/1646/1638): the controller defines no ``main()`` and runs
only inside ``h_probe_command``, so its events ride that process's log substrate
(``configure_logging()`` is the command's first statement). The vocabulary is the
closed set in ``worker.h_probe_log_events``; fields travel only through
``logger.bind`` and every record binds ``pod_name`` (the launch's own name) so
concurrent arms that share one JSON file can be told apart.
"""

import math
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from entropic_substrate.kernel.artifacts import normalize_d1_artifact_root
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.run.h_probe_collection import collect_probe_terminal
from entropic_substrate.ladder.run.h_probe_control import read_probe_control
from entropic_substrate.ladder.run.h_probe_events import read_probe_event
from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch
from entropic_substrate.ladder.run.h_probe_process import ProbeWatchdogProcess
from entropic_substrate.ladder.run.h_probe_watchdog import UNCERTAIN_ABSENT_CONFIRMATIONS
from entropic_substrate.ladder.run.h_runpod_transport import ProbePod, ProbeProviderError
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker.h_probe_allocation import (
    ProbeAllocationHandoffError,
    ProbeAllocationProvider,
    allocate_probe_once,
    provider_error_code,
)
from entropic_substrate.ladder.run.worker.h_probe_log_events import (
    HOST_ALLOCATION_END,
    HOST_ALLOCATION_START,
    HOST_EMERGENCY_CLEANUP,
    HOST_POD_ATTRIBUTED,
    HOST_POLL_TICK,
    HOST_PROGRESS_UNAVAILABLE,
    HOST_PROVIDER_ABSENT_OBSERVED,
    HOST_PROVIDER_TRANSIENT,
    HOST_STOP_HANDOFF,
    HOST_TERMINAL_OBSERVED,
    HOST_WATCHDOG_JOIN_END,
    HOST_WATCHDOG_JOIN_START,
    SOURCE_R2_CHECKPOINT,
    STATE_NO_PROGRESS,
    STATE_READ_FAILED,
    STATE_STEP,
    checkpoint_steps,
    progress_fields,
    render,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest

# Provider codes the poll treats as transient: logged as a named host event and the
# poll continues at the existing cadence, adding no fast-fail path. This duplicates
# the watchdog's own tolerated set (ladder/run/h_probe_watchdog.py:191-196, outside
# this phase's envelope); tests/test_h_probe_log_events.py pins the equality.
_TRANSIENT_PROVIDER_CODES = ("network_error", "timeout", "rate_limited", "server_error")


class ProbeControllerProvider(ProbeAllocationProvider, Protocol):
    def get_owned(self, launch: H0ProbeLaunch, pod_id: str) -> ProbePod | None: ...
    def delete_owned(self, launch: H0ProbeLaunch, pod_id: str) -> str: ...


class ProbeControllerFailure(RuntimeError):
    def __init__(self, record: dict[str, object], causes: tuple[BaseException, ...]) -> None:
        self.record, self.causes = record, causes
        super().__init__("H0 controller failed; inspect retained capture and cleanup evidence")


class ProbeProviderAbsent(RuntimeError):
    """The provider no longer lists the attributed pod, and no terminal was published.

    A named terminal observation (FOR-1638), not a swallowed error: it ends the poll
    loop before the controller deadline. The result stays ``controller_failed`` and the
    stop reason stays ``cancelled`` (the watchdog's closed set admits no other value for
    an uncollected run).
    """

    def __init__(self) -> None:
        super().__init__("provider reports the attributed pod absent without terminal evidence")


def _emit(level: str, event: str, pod_name: str, **fields: object) -> None:
    """One host event: structured fields through ``bind``, one human sentence."""
    bound: dict[str, object] = {"pod_name": pod_name, **fields}
    getattr(logger.bind(event=event, **bound), level)(render(event, **bound))


class _CheckpointProgress:
    """Advisory R2 progress: the newest ``checkpoint-N`` event, found without listing.

    The seal schedule is a pure function of the request, so the candidate event names
    are known. Presence is monotone (a later checkpoint implies every earlier one), so a
    poll probes the next unseen candidate and, only when it exists, binary-searches the
    rest. Reads per poll are bounded by ``1 + ceil(log2(candidates))``, independent of
    run length and of how many objects sit under the request prefix. The newest event is
    chosen by parsed step, never by lexical order. Nothing here can change the loop's
    exit, the collection, the stop reason or the result record.
    """

    def __init__(self, request: ProbeRequest) -> None:
        self._request = request
        self._steps: tuple[int, ...] | None = None
        self._known = 0

    @property
    def total(self) -> int:
        schedule = self._request.profile.schedule
        return schedule.warmup_steps + schedule.steps

    def advance(self, objects: D1TrainerMailboxObjectPort) -> int | None:
        """Probe forward and return the newest proven global step, or None if none yet."""
        if self._steps is None:
            schedule = self._request.profile.schedule
            self._steps = checkpoint_steps(
                warmup_steps=schedule.warmup_steps,
                steps=schedule.steps,
                checkpoint_interval=schedule.checkpoint_interval,
                evaluation_interval=schedule.evaluation_interval,
            )
        steps = self._steps
        if self._known < len(steps) and self._present(objects, steps[self._known]):
            self._known += 1
            low, high = self._known, len(steps)
            while low < high:
                middle = (low + high) // 2
                if self._present(objects, steps[middle]):
                    low = middle + 1
                    self._known = low
                else:
                    high = middle
        return steps[self._known - 1] if self._known else None

    def _present(self, objects: D1TrainerMailboxObjectPort, step: int) -> bool:
        event = read_probe_event(
            objects,
            self._request,
            f"checkpoint-{step:08d}",
            expected_pod_id=None,
            max_bytes=self._request.limits.max_object_bytes,
        )
        return event is not None


def _termination(record: dict[str, object], pod_id: str | None) -> bool:
    """Accept either a verified attributed teardown or a verified pre-pod absence.

    The two admissible watchdog statuses require a different pod_id shape each:
    `terminated` only ever certifies an attributed pod, `cancelled_uncertain_absent`
    only ever certifies that no pod was ever attributed. Every other flag keeps
    identical, unconditional semantics between the two.
    """
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
    if not (
        required <= record.keys()
        and record["schema"] == "h-probe-watchdog-result-v1"
        and record["scientific_authority"] == "none"
        and record["termination_verified"] is True
        and record["attributed_pod_absent"] is True
        and record["inventory_verified"] is True
        and record["unattributed_pod_ids"] == []
        and record["guard_fault"] is None
    ):
        return False
    if pod_id is None:
        return record["status"] == "cancelled_uncertain_absent" and record["pod_id"] is None
    return record["status"] == "terminated" and record["pod_id"] == pod_id


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


def _owned_pod_absent(
    guard: ProbeWatchdogProcess,
    provider: ProbeControllerProvider,
    pod_id: str,
    pod_name: str,
    account_id: object,
) -> bool:
    """One per-poll ownership observation: True only for a provider None on the same account.

    A transient provider code is logged as a named event and the poll continues at the
    existing cadence (no counter, sleep or timeout is added). Any other error propagates
    into the existing failure path, where the attributed id is preserved (C5).
    """
    try:
        if provider.get_owned(guard.launch, pod_id) is not None:
            return False
        if provider.account().account_id != account_id:
            raise ValueError("provider account differs from the supervised account")
        return True
    except ProbeProviderError as error:
        if error.code not in _TRANSIENT_PROVIDER_CODES:
            raise
        _emit("warning", HOST_PROVIDER_TRANSIENT, pod_name, code=error.code)
        return False


def _progress_tick(
    progress: _CheckpointProgress, objects: D1TrainerMailboxObjectPort, pod_name: str
) -> None:
    """The per-poll progress line. Advisory: any failure is its own WARNING event only."""
    try:
        step = progress.advance(objects)
        fields = progress_fields(
            source=SOURCE_R2_CHECKPOINT,
            state=STATE_NO_PROGRESS if step is None else STATE_STEP,
            step=step,
            total=progress.total,
        )
        if step is not None:
            fields["last_event"] = f"checkpoint-{step:08d}"
        _emit("info", HOST_POLL_TICK, pod_name, **fields)
    except Exception as error:
        fields = progress_fields(source=SOURCE_R2_CHECKPOINT, state=STATE_READ_FAILED)
        _emit(
            "warning",
            HOST_PROGRESS_UNAVAILABLE,
            pod_name,
            error_type=type(error).__name__,
            **fields,
        )


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
    provider_failure_code: str | None = None
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

    pod_name = guard.launch.pod_name
    account_id = guard.ready["account_id"]
    progress = _CheckpointProgress(guard.launch.request)
    stage = "allocation"
    try:
        if clock() >= deadline:
            raise TimeoutError("controller deadline expired before allocation")
        _emit("info", HOST_ALLOCATION_START, pod_name)
        try:
            pod = allocate_probe_once(guard, provider, objects)
        except BaseException as error:
            _emit(
                "warning",
                HOST_ALLOCATION_END,
                pod_name,
                outcome="failed",
                error_type=type(error).__name__,
            )
            raise
        _emit("info", HOST_ALLOCATION_END, pod_name, outcome="created")
        pod_id = pod.pod_id
        _emit("info", HOST_POD_ATTRIBUTED, pod_name, pod_id=pod_id)
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
            stage = "provider_ownership"
            absent = _owned_pod_absent(guard, provider, pod_id, pod_name, account_id)
            stage = "terminal_capture"
            collection = collect_probe_terminal(
                objects,
                guard.launch.request,
                root,
                expected_pod_id=pod_id,
                max_marker_bytes=guard.launch.request.limits.max_object_bytes,
            )
            if collection is not None:
                _emit(
                    "info",
                    HOST_TERMINAL_OBSERVED,
                    pod_name,
                    reported_outcome=collection.get("reported_outcome"),
                    workload_verified=collection.get("workload_verified"),
                )
                break
            if absent:
                # The None, the account match and a terminal-capture attempt AFTER it
                # (above) all hold: a published terminal is never shadowed (C7).
                stage = "provider_absent"
                _emit("warning", HOST_PROVIDER_ABSENT_OBSERVED, pod_name, pod_id=pod_id)
                raise ProbeProviderAbsent
            _progress_tick(progress, objects, pod_name)
            remaining = deadline - clock()
            if remaining > 0:
                wait(min(float(poll_seconds), remaining))
        if collection is None:
            raise TimeoutError("controller deadline expired without terminal evidence")
    except BaseException as error:
        errors.append(error)
        failure_stages.append(stage)
        # C5: custody is immutable once allocation returned an attributed pod. A later
        # provider error (the per-poll ownership probe) must never rebind the id to its
        # own possible_pod_id, which may be None.
        if isinstance(error, ProbeAllocationHandoffError):
            if pod_id is None:
                pod_id = error.pod_id
            provider_failure_code = provider_error_code(error.provider_error)
        elif isinstance(error, ProbeProviderError):
            if pod_id is None:
                pod_id = error.possible_pod_id
            provider_failure_code = provider_error_code(error)

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
        _emit("info", HOST_STOP_HANDOFF, pod_name, reason=reason, pod_id=recorded_id)
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
        _emit(
            "warning",
            HOST_EMERGENCY_CLEANUP,
            pod_name,
            outcome=emergency["status"],
            acknowledgement=emergency.get("acknowledgement"),
            owned_pod_absent=emergency["owned_pod_absent"],
        )

    if (
        stop_failed
        or not guard.alive
        or any(isinstance(e, ProbeAllocationHandoffError) for e in errors)
    ):
        emergency_attempt()
    try:
        if pod_id is None:
            # No pod was ever attributed: the watchdog resolves this from its own
            # K-poll uncertain-absent window (h_probe_watchdog), not by polling
            # toward the full cleanup_deadline sized for an attributed teardown.
            # Per poll the watchdog makes up to two bounded provider calls
            # (account, list_owned) plus the sleep; +1 poll covers a stop
            # landing just after a poll tick.
            timeout = (UNCERTAIN_ABSENT_CONFIRMATIONS + 1) * (
                guard.config.poll_seconds + 2 * guard.config.provider_timeout_seconds
            ) + guard.config.shutdown_timeout_seconds
        else:
            timeout = max(0.0, float(cleanup_deadline) - clock())
            timeout += (
                3 * guard.config.provider_timeout_seconds + guard.config.shutdown_timeout_seconds
            )
        _emit("info", HOST_WATCHDOG_JOIN_START, pod_name, timeout_seconds=float(timeout))
        watchdog = guard.wait_result(timeout_seconds=timeout)
        _emit("info", HOST_WATCHDOG_JOIN_END, pod_name, status=watchdog.get("status"))
        if not _termination(watchdog, pod_id):
            raise ValueError("watchdog did not verify this allocation's complete termination")
    except BaseException as error:
        errors.append(error)
        failure_stages.append("watchdog_join")
        if watchdog is None:
            # The join itself failed. A join that returned a record whose termination was not
            # verified already narrated that record's status above.
            _emit(
                "warning",
                HOST_WATCHDOG_JOIN_END,
                pod_name,
                status="join_failed",
                error_type=type(error).__name__,
            )
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
        "provider_error_code": provider_failure_code,
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
