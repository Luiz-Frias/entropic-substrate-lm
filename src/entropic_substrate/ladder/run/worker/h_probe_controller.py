"""Compose one authorized H0 allocation, terminal capture and supervised teardown.

Host narration (FOR-1645/1646/1638): the controller defines no ``main()`` and runs
only inside ``h_probe_command``, so its events ride that process's log substrate
(``configure_logging()`` is the command's first statement). The vocabulary is the
closed set in ``worker.h_probe_log_events``; fields travel only through
``logger.bind`` and every record binds ``pod_name`` (the launch's own name) so
concurrent arms that share one JSON file can be told apart.
"""

import math
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from entropic_substrate.kernel.artifacts import normalize_d1_artifact_root
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
from entropic_substrate.ladder.run.transport.d1_trainer_s3_object_adapter import (
    D1TrainerS3ObjectError,
)
from entropic_substrate.ladder.run.worker.h_probe_allocation import (
    ProbeAllocationHandoffError,
    ProbeAllocationProvider,
    allocate_probe_once,
    provider_error_code,
)
from entropic_substrate.ladder.run.worker.h_probe_host_log import emit
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
    HOST_TERMINAL_PROBE_TRANSIENT,
    HOST_WATCHDOG_JOIN_END,
    HOST_WATCHDOG_JOIN_START,
    SOURCE_R2_CHECKPOINT,
    STATE_NO_PROGRESS,
    STATE_READ_FAILED,
    STATE_STEP,
    checkpoint_steps,
    progress_fields,
)
from entropic_substrate.ladder.run.worker_request import AnyProbeRequest, PairProbeRequest

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


# RETRACTED 2026-11-02: this regex used to read ``(warmup|prediction)``. ``runner.py`` names a
# receipt with ``execution.phase`` and ``SentencePhase`` is ``codec_warmup | prediction``
# (training.py), so a real warm-up receipt never matched and the advisory progress tick turned
# the refusal into ``read_failed``. The pair path carried the right alternation in a second
# regex; there is now one rule for both.
# Measured 2026-11-02 at 1e9e6fd3: the hand-off receipt ``handoff-codec_warmup-00000002-progress.json``
# DOES match this pattern (old ``(warmup|prediction)`` form: False; this form: True), so the regex
# alone never refuses a hand-off. The hand-off is kept from being a resume target by the resume
# request gate in ``worker_request._check_resumes`` (refuses a ``handoff-`` name) and by
# ``h_probe_resume``'s paused-status check (R3). This regex is not the R3 guard.
# RETRACTED 2026-11-02: the parenthetical "that gate and its pin ship in the staged DG-A patch,
# not on this tip" was false. The gate is on this tip (``worker_request.py``, the ``handoff-``
# check at line 472), and ``tests/test_h_pair_remote_dga.py`` pins that the hand-off receipt is
# never a resume target.
_RESUME_RECEIPT = re.compile(r"-(codec_warmup|prediction)-(\d{8})-progress\.json")


def _stopped_global_step(path: str, warmup_steps: int) -> int:
    """The global optimizer step a resume receipt stopped at, from its artifact name."""
    match = _RESUME_RECEIPT.search(PurePosixPath(path).name)
    if match is None:
        raise ValueError(f"resume receipt name {path!r} carries no stopped step")
    phase, step = match.group(1), int(match.group(2))
    return step + (warmup_steps if phase == "prediction" else 0)


@dataclass
class _Ladder:
    """One cell's (or the one profile's) seal ladder and how far presence has been proven."""

    cell: str | None
    steps: tuple[int, ...]
    known: int = 0

    def event(self, step: int) -> str:
        return (
            f"checkpoint-{step:08d}" if self.cell is None else f"checkpoint-{self.cell}-{step:08d}"
        )


class _CheckpointProgress:
    """Advisory R2 progress: the newest ``checkpoint-N`` event, found without listing.

    The seal schedule is a pure function of the request, so the candidate event names
    are known. Presence is monotone (a later checkpoint implies every earlier one), so a
    poll probes the next unseen candidate and, only when it exists, binary-searches the
    rest. Reads per poll are bounded by ``1 + ceil(log2(candidates))``, independent of
    run length and of how many objects sit under the request prefix. The newest event is
    chosen by parsed step, never by lexical order. Nothing here can change the loop's
    exit, the collection, the stop reason or the result record.

    A resumed request publishes only the seals after the step its receipt stopped at, so
    the candidates are filtered to global steps above it; probing from the first
    candidate never finds a present event and the walk cannot advance. MEASURED
    2026-10-26, arm B of wave h0-final-probe-resume-v4.1755b0: stopped at global 500,
    R2 held ``checkpoint-00000750`` under B's prefix and MLflow reached 1000, yet every
    tick logged ``state=no_progress_yet``. The stopped step is parsed from the resume
    receipt name ``...-{phase}-{step:08d}-progress.json`` (the receipt's ``step`` is
    phase-local, so a prediction stop adds ``warmup_steps``). A stop that is not on the
    seal ladder is refused, never rounded.

    FOR-1694: a PAIR request has no single profile; it keeps one ladder PER CELL, over that
    cell's own schedule and its cell-named events (``checkpoint-<cell>-<total:08d>``). The
    recipient's ladder starts after its inherited warm-up (its hand-off is not a checkpoint
    event), a resumed cell's ladder starts after its resume step, and ``advance`` returns
    the newest global step ANY cell has been proven to have published, naming the event it
    read in ``last_event``: a step is never claimed that no event was read for.
    """

    def __init__(self, request: AnyProbeRequest) -> None:
        self._request = request
        self._ladders: list[_Ladder] | None = None
        self.last_event: str | None = None

    @property
    def total(self) -> int:
        request = self._request
        profile = request.donor if isinstance(request, PairProbeRequest) else request.profile
        return profile.schedule.warmup_steps + profile.schedule.steps

    def _build(self) -> list[_Ladder]:
        request = self._request
        if isinstance(request, PairProbeRequest):
            ladders: list[_Ladder] = []
            for index, cell in enumerate(request.cells):
                schedule = cell.profile.schedule
                steps = checkpoint_steps(
                    warmup_steps=schedule.warmup_steps,
                    steps=schedule.steps,
                    checkpoint_interval=schedule.checkpoint_interval,
                    evaluation_interval=schedule.evaluation_interval,
                )
                floor = 0
                if index == 1:  # the recipient arrives warmed: no checkpoint of its own before
                    floor = schedule.warmup_steps
                if cell.resume is not None:
                    stopped = _stopped_global_step(cell.resume.path, schedule.warmup_steps)
                    if stopped not in steps:
                        raise ValueError(
                            f"resume stop at global step {stopped} is not on the seal ladder"
                        )
                    floor = max(floor, stopped)
                ladders.append(_Ladder(cell.token, tuple(s for s in steps if s > floor)))
            return ladders
        schedule = request.profile.schedule
        steps = checkpoint_steps(
            warmup_steps=schedule.warmup_steps,
            steps=schedule.steps,
            checkpoint_interval=schedule.checkpoint_interval,
            evaluation_interval=schedule.evaluation_interval,
        )
        resume = request.resume
        if resume is not None:
            stopped = _stopped_global_step(resume.path, schedule.warmup_steps)
            if stopped not in steps:
                raise ValueError(f"resume stop at global step {stopped} is not on the seal ladder")
            steps = tuple(step for step in steps if step > stopped)
        return [_Ladder(None, steps)]

    def advance(self, objects: D1TrainerMailboxObjectPort) -> int | None:
        """Probe forward and return the newest proven global step, or None if none yet."""
        if self._ladders is None:
            self._ladders = self._build()
        newest: tuple[int, str] | None = None
        for ladder in self._ladders:
            steps = ladder.steps
            if ladder.known < len(steps) and self._present(objects, ladder, steps[ladder.known]):
                ladder.known += 1
                low, high = ladder.known, len(steps)
                while low < high:
                    middle = (low + high) // 2
                    if self._present(objects, ladder, steps[middle]):
                        low = middle + 1
                        ladder.known = low
                    else:
                        high = middle
            if ladder.known and (newest is None or steps[ladder.known - 1] >= newest[0]):
                newest = (steps[ladder.known - 1], ladder.event(steps[ladder.known - 1]))
        self.last_event = None if newest is None else newest[1]
        return None if newest is None else newest[0]

    def _present(self, objects: D1TrainerMailboxObjectPort, ladder: _Ladder, step: int) -> bool:
        event = read_probe_event(
            objects,
            self._request,
            ladder.event(step),
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
        emit("warning", HOST_PROVIDER_TRANSIENT, pod_name=pod_name, code=error.code)
        return False


@dataclass
class _TerminalProbeTally:
    """What the per-tick terminal probe observed: advisory, never an exit authority.

    ``count`` is the number of RECOVERABLE read failures absorbed as logged ticks. RETRACTED
    (FOR-1667): ``last_error`` is no longer "the last absorbed error"; it is the
    ``safe_diagnostic()`` of the last classified read failure, absorbed (recoverable) or
    propagated (permanent, so the result record names the code that ended the run while
    ``count`` stays 0 for it).
    """

    count: int = 0
    last_error: dict[str, str | int | None] | None = None


def _probe_terminal(
    objects: D1TrainerMailboxObjectPort,
    guard: ProbeWatchdogProcess,
    pod_id: str,
    pod_name: str,
    root: Path,
    tally: _TerminalProbeTally,
) -> dict[str, object] | None:
    """One per-poll terminal observation; a RECOVERABLE marker READ failure is a logged tick.

    Mirrors ``_owned_pod_absent``: a ``D1TrainerS3ObjectError`` the adapter classifies as
    recoverable (``is_recoverable``) is emitted as a named WARNING and the poll continues at
    the existing cadence (no counter abort, sleep or timeout is added); the controller
    deadline and the watchdog remain the only removal authorities for it. RETRACTED
    (FOR-1667): this docstring and the FOR-1665 pin once said every ``D1TrainerS3ObjectError``
    is transient. It is not: a permanent code (an integrity code, or an auth or request-shape
    provider failure) cannot be cured by the next tick, so it propagates at once with the
    stage ``terminal_capture`` instead of masking itself, and the provider-absent exit,
    behind the deadline. The conflicting-claims ``ValueError`` and every other error also
    propagate into the existing failure path. An error raised after the
    capture began (the destination is no longer empty) is not a marker probe: a retry
    would meet the fresh-destination refusal, so it propagates with its partial evidence.
    The payload is ``safe_diagnostic()`` only (never a message, URL, key or request id).
    """
    try:
        return collect_probe_terminal(
            objects,
            guard.launch.request,
            root,
            expected_pod_id=pod_id,
            max_marker_bytes=guard.launch.request.limits.max_object_bytes,
        )
    except D1TrainerS3ObjectError as error:
        if any(root.iterdir()):
            raise
        diagnostic = error.safe_diagnostic()
        if not error.is_recoverable():
            tally.last_error = diagnostic
            raise
        tally.count += 1
        tally.last_error = diagnostic
        emit(
            "warning",
            HOST_TERMINAL_PROBE_TRANSIENT,
            pod_name=pod_name,
            operation=diagnostic["operation"],
            code=diagnostic["code"],
            provider_code=diagnostic["provider_code"],
            status=diagnostic["http_status"],
            attempt=tally.count,
        )
        return None


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
            fields["last_event"] = progress.last_event
        emit("info", HOST_POLL_TICK, pod_name=pod_name, **fields)
    except Exception as error:
        fields = progress_fields(source=SOURCE_R2_CHECKPOINT, state=STATE_READ_FAILED)
        emit(
            "warning",
            HOST_PROGRESS_UNAVAILABLE,
            pod_name=pod_name,
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
    max_hourly_rate: float | None,
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
    # FOR-1694: an absent ``max_hourly_rate`` is admitted only for a launch that seals its
    # ceiling (``hourly_ceiling`` below refuses it for any other launch, by name).
    for value in (poll_seconds, *(() if max_hourly_rate is None else (max_hourly_rate,))):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("controller polling and price ceiling must be finite and positive")
    # Rung B: every caller, not only the command, keeps a scientific run inside the
    # spend cap its sealed authority grants (a no-op for a smoke launch).
    # FOR-1694: ``ceiling`` is the ONE value the price check below compares against: the
    # launch's sealed hourly price ceiling when it has one (an argument that is present and
    # not exactly equal to it is refused by name, before any provider call), else the argument.
    ceiling = guard.launch.hourly_ceiling(max_hourly_rate)
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
    probe_tally = _TerminalProbeTally()

    def emergency_attempt() -> None:
        nonlocal emergency
        try:
            emergency = _emergency_cleanup(guard, provider, pod_id)
        except BaseException as error:
            errors.append(error)
            failure_stages.append("emergency_cleanup")
            emergency = {"status": "emergency_failed", "owned_pod_absent": False}
        emit(
            "warning",
            HOST_EMERGENCY_CLEANUP,
            pod_name=pod_name,
            outcome=emergency["status"],
            acknowledgement=emergency.get("acknowledgement"),
            owned_pod_absent=emergency["owned_pod_absent"],
        )

    stage = "allocation"
    try:
        if clock() >= deadline:
            raise TimeoutError("controller deadline expired before allocation")
        emit("info", HOST_ALLOCATION_START, pod_name=pod_name)
        try:
            pod = allocate_probe_once(guard, provider, objects)
        except BaseException as error:
            emit(
                "warning",
                HOST_ALLOCATION_END,
                pod_name=pod_name,
                outcome="failed",
                error_type=type(error).__name__,
            )
            raise
        emit("info", HOST_ALLOCATION_END, pod_name=pod_name, outcome="created")
        pod_id = pod.pod_id
        emit("info", HOST_POD_ATTRIBUTED, pod_name=pod_name, pod_id=pod_id)
        stage = "allocation_custody"
        guard.control.write(
            "pod",
            {"schema": "h-probe-pod-observation-v1", **asdict(pod)},
            max_bytes=guard.config.max_control_bytes,
        )
        # The per-pod rate is first readable HERE, in the create response (costPerHr). Read
        # 2026-10-30 against the pinned transport (ladder/run/h_runpod_transport.py): its only
        # pre-create provider reads are `user` (account balance and the account's current
        # spend per hour, not this pod's quote) and `template get` (validate_template's
        # contract carries no price field). A runpodctl 3.0.1 GPU price listing was NOT
        # measured, and a listed quote would not bind the price create bills. So a refused
        # rate is already billing: the except handler below deletes the owned pod at once,
        # before the stop handoff, instead of leaving it to the watchdog's next poll.
        stage = "price_admission"
        if (
            type(pod.cost_per_hour) is not int and type(pod.cost_per_hour) is not float
        ) or not 0 <= pod.cost_per_hour <= ceiling:
            raise ValueError("measured allocation price is missing or exceeds the approved ceiling")
        while clock() < deadline:
            stage = "watchdog_liveness"
            guard.require_alive()
            stage = "provider_ownership"
            absent = _owned_pod_absent(guard, provider, pod_id, pod_name, account_id)
            stage = "terminal_capture"
            transient_before = probe_tally.count
            collection = _probe_terminal(objects, guard, pod_id, pod_name, root, probe_tally)
            if collection is not None:
                emit(
                    "info",
                    HOST_TERMINAL_OBSERVED,
                    pod_name=pod_name,
                    reported_outcome=collection.get("reported_outcome"),
                    workload_verified=collection.get("workload_verified"),
                )
                break
            # A tick whose marker read raised a RECOVERABLE error did not return None: it is a
            # transient (a permanent read already propagated above), so the absence is deferred
            # to the next tick, which re-asks the provider and re-reads the markers (FOR-1665).
            # Otherwise the None, the account match and a terminal-capture attempt AFTER it
            # (above) all hold: a published terminal is never shadowed (C7).
            if absent and probe_tally.count == transient_before:
                stage = "provider_absent"
                emit("warning", HOST_PROVIDER_ABSENT_OBSERVED, pod_name=pod_name, pod_id=pod_id)
                raise ProbeProviderAbsent
            _progress_tick(progress, objects, pod_name)
            remaining = deadline - clock()
            if remaining > 0:
                wait(min(float(poll_seconds), remaining))
        if collection is None:
            if probe_tally.count:
                # The deadline, not a failing read, ended the run: every tick that raised
                # was absorbed above, so the exit is named for what removed the pod.
                stage = "deadline"
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
        if stage == "price_admission":
            emergency_attempt()

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
        emit("info", HOST_STOP_HANDOFF, pod_name=pod_name, reason=reason, pod_id=recorded_id)
    except BaseException as error:
        errors.append(error)
        failure_stages.append("stop_handoff")
        stop_failed = True

    if emergency is None and (
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
        emit("info", HOST_WATCHDOG_JOIN_START, pod_name=pod_name, timeout_seconds=float(timeout))
        watchdog = guard.wait_result(timeout_seconds=timeout)
        emit("info", HOST_WATCHDOG_JOIN_END, pod_name=pod_name, status=watchdog.get("status"))
        if not _termination(watchdog, pod_id):
            raise ValueError("watchdog did not verify this allocation's complete termination")
    except BaseException as error:
        errors.append(error)
        failure_stages.append("watchdog_join")
        if watchdog is None:
            # The join itself failed. A join that returned a record whose termination was not
            # verified already narrated that record's status above.
            emit(
                "warning",
                HOST_WATCHDOG_JOIN_END,
                pod_name=pod_name,
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
        "terminal_probe_transient_count": probe_tally.count,
        "last_terminal_probe_error": probe_tally.last_error,
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
