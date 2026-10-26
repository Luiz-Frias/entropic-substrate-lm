"""One request, checkpoint snapshots, and marker-last worker outcomes over R2 ports."""

import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.native_verification import read_native_verification
from entropic_substrate.ladder.run.runner import read_training_progress
from entropic_substrate.ladder.run.seals import SealMemo, narrated_admission
from entropic_substrate.ladder.run.span_states import SpanPopulationGap
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker_execution import execute_probe_request, narrated_stage
from entropic_substrate.ladder.run.worker_log_events import (
    COMPLETION_WRITTEN,
    STOP_MARKER_OBSERVED,
    STOP_MARKER_RETRY,
    render,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest, worker_artifact
from entropic_substrate.ladder.run.worker_snapshot import PublicationLedger, publish_worker_snapshot
from entropic_substrate.ladder.run.worker_transport import (
    VerificationReadRetry,
    materialize_probe_request,
    publish_verified_object,
    read_published_probe_request,
)


def _canonical(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


# R2 stop marker door (FOR-1617): its own schema, distinct from the host
# control message `h-probe-stop-v1` (worker/h_probe_controller.py,
# h_probe_watchdog.py). Defined once here -- never in worker/h_probe_stop.py,
# which would close an import cycle through worker_command.py -- and
# imported by the verb, never duplicated.
STOP_MARKER_SCHEMA = "h-probe-stop-marker-v1"

# h-run-lifecycle-v1 (FOR-1639): the run_start / run_end records this worker
# writes through its MetricSink. Read back by
# ladder.capture.reader.h_telemetry.final_h_run_end, which holds its own copy
# of the literal (the capture layer never imports from ladder/run).
LIFECYCLE_SCHEMA = "h-run-lifecycle-v1"


def stop_marker_key(request: ProbeRequest) -> str:
    return f"{request.prefix}/control/stop.json"


def _valid_target(target: object) -> tuple[str, int]:
    if (
        type(target) is not tuple
        or len(target) != 2
        or type(target[0]) is not str
        or not target[0]
        or type(target[1]) is not int
        or target[1] < 1
    ):
        raise ValueError("stop marker target must be (non-empty phase, positive step)")
    return target[0], target[1]


def encode_stop_marker(request: ProbeRequest, target: tuple[str, int] | None = None) -> bytes:
    """A pure function of (schema, request digest, target): no `at`, no nonce.

    ``target`` names the one seal ``(phase, step)`` at which the worker stops;
    ``None`` omits the key, so the bytes equal the pre-target marker and the
    worker stops at the next seal it polls.
    """
    row: dict[str, object] = {"schema": STOP_MARKER_SCHEMA, "request_sha256": request.digest}
    if target is not None:
        phase, step = _valid_target(target)
        # Same coordinate rule as ``ProbeRequest.pause``: the marker is immutable
        # once published, so a target the runner never reaches must be refused
        # here, at the verb, rather than discovered as a timeout.
        schedule = request.profile.schedule
        budget = schedule.warmup_steps if phase == "codec_warmup" else schedule.steps
        if phase not in ("codec_warmup", "prediction") or not 1 <= step <= budget:
            raise ValueError("stop marker target is outside the request schedule")
        row["target"] = {"phase": phase, "step": step}
    return _canonical(row)


def _read_stop_marker(payload: bytes, request: ProbeRequest) -> tuple[bool, tuple[str, int] | None]:
    value = json.loads(payload)
    keys = "schema request_sha256 target" if "target" in value else "schema request_sha256"
    row = recovery_record(value, keys)
    if row["schema"] != STOP_MARKER_SCHEMA:
        raise ValueError("stop marker schema is foreign")
    digest = row["request_sha256"]
    if type(digest) is not str:
        raise ValueError("stop marker request digest is invalid")
    target = None
    if "target" in row:
        raw = row["target"]
        if type(raw) is not dict or set(raw) != {"phase", "step"}:
            raise ValueError("stop marker target must be (non-empty phase, positive step)")
        target = _valid_target((raw["phase"], raw["step"]))
    if digest != request.digest:
        return False, None
    if _canonical(row) != payload:
        raise ValueError("stop marker is not canonical JSON")
    return True, target


def classify_stop_marker(payload: bytes, request: ProbeRequest) -> bool:
    """True: this marker is this request's own. False: another request's, ignored.

    Everything else -- bad JSON/UTF-8, non-canonical bytes, a missing or
    extra key, a foreign schema, a malformed target -- raises. The schema
    check runs before the request digest check.
    """
    return _read_stop_marker(payload, request)[0]


def stop_marker_hits(payload: bytes, request: ProbeRequest, seal: tuple[str, int]) -> bool:
    """True when the marker is this request's own AND names no target or ``seal``."""
    own, target = _read_stop_marker(payload, request)
    return own and (target is None or target == seal)


def _paths(root: Path, sections: tuple[str, ...]) -> tuple[str, ...]:
    paths = []
    for section in sections:
        selected = root / section
        if selected.is_symlink():
            raise ValueError("worker output directory cannot be a symlink")
        if not selected.exists():
            continue
        for path in selected.rglob("*"):
            if path.is_symlink():
                raise ValueError("worker outputs cannot contain symlinks")
            if path.is_file():
                paths.append(path.relative_to(root).as_posix())
    return tuple(sorted(paths))


# Terminal outcome markers (G12): `complete` is the contract; the others carry the
# marker name so a stopped/paused/failed run leaves a diagnosable receipt too.
_TERMINAL_MARKERS = frozenset({"complete", "paused", "stopped", "failure"})


def _event(
    objects: D1TrainerMailboxObjectPort, request: ProbeRequest, name: str, value: dict[str, object]
) -> D1TrainerMailboxObjectIdentity:
    payload = _canonical(
        {
            "schema": "h-worker-event-v1",
            "request_sha256": request.digest,
            "run_identity": request.profile.identity,
            "scientific_authority": "none",
            "at": datetime.now(UTC).isoformat(),
            **value,
        }
    )
    if len(payload) > request.limits.max_object_bytes:
        raise ValueError("worker event exceeds its byte budget")
    identity = publish_verified_object(objects, f"{request.prefix}/{name}.json", payload)
    if name in _TERMINAL_MARKERS:
        # FOR-1637 (ruling D1): reported only AFTER the marker's put and its
        # verification read returned; never for `started` or `checkpoint-*`.
        written = {"key": identity.key, "bytes": identity.size_bytes, "marker": name}
        logger.bind(event=COMPLETION_WRITTEN, **written).info(render(COMPLETION_WRITTEN, **written))
    return identity


def _publication(
    ledger: PublicationLedger,
    ledgered_before: int,
    paths: tuple[str, ...],
    l3_seconds: float,
    retry: VerificationReadRetry,
    timing: dict[str, float],
) -> dict[str, object]:
    """The ONE `publication`-block builder shared by every leg that calls
    :func:`~entropic_substrate.ladder.run.worker_snapshot.publish_worker_snapshot`
    with a ledger (FOR-1567/pre-for-1567c). ``timing`` is the runner's L1/L2
    mirror for the checkpoint leg, or an empty mapping for the final-snapshot
    leg (see its call site for why that leg's L1/L2 are unmeasured, never
    reused).
    """
    # +1 in both counts below: the snapshot MARKER itself (h-worker-snapshot-v1)
    # is always a fresh key, never ledgered.
    new_objects = (len(ledger) - ledgered_before) + 1
    return {
        "l1_seconds": timing.get("l1_seconds", 0.0),
        "l2_seconds": timing.get("l2_seconds", 0.0),
        "l3_seconds": l3_seconds,
        "objects_put": new_objects,
        "objects_skipped": len(paths) - (new_objects - 1),
        "verification_reads": new_objects,
        "retries": retry.failures,
    }


def _admit_result(
    root: Path, request: ProbeRequest, value: object, memo: SealMemo | None = None
) -> dict[str, object]:
    row = recovery_record(
        value,
        "schema request_sha256 run_identity scientific_authority status runtime successful_optimizer_steps training generation diagnostics capture verification wall_seconds",
    )
    if (
        row["schema"] != "h-probe-worker-result-v1"
        or row["request_sha256"] != request.digest
        or row["run_identity"] != request.profile.identity
        or row["scientific_authority"] != "none"
    ):
        raise ValueError("worker result identity mismatch")
    duration = row["wall_seconds"]
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
        or duration < 0
    ):
        raise ValueError("worker result duration is invalid")
    maximum = request.limits.max_object_bytes
    training = worker_artifact(row["training"], "run")
    # ``memo`` is the /run memo this process's completion walk populated (None
    # for a path that did not run training here: nothing is reused or measured).
    with narrated_admission("result_admission", "run", memo):
        report = read_training_progress(
            root / "run",
            training,
            request.profile,
            root / "source",
            max_source_bytes=maximum,
            max_artifact_bytes=maximum,
            memo=memo,
        )
    if (
        type(row["successful_optimizer_steps"]) is not int
        or row["successful_optimizer_steps"] != report["successful_optimizer_steps"]
    ):
        raise ValueError("worker result step count differs from admitted training")
    if row["status"] in ("paused", "stopped_by_operator"):
        if report["status"] != "paused" or any(
            row[k] is not None for k in ("generation", "diagnostics", "capture", "verification")
        ):
            raise ValueError("paused worker result cannot claim later stages")
        return row
    if row["status"] != "probe_workload_complete" or report["status"] != "training_complete":
        raise ValueError("worker completion lacks complete training")
    if any(row[key] is None for key in ("generation", "diagnostics", "capture", "verification")):
        raise ValueError("worker completion requires observer and native verification receipts")
    captured, verified = (
        worker_artifact(row["capture"], "capture"),
        worker_artifact(row["verification"], "verification"),
    )
    read_native_verification(
        root / "verification",
        verified,
        root / "capture",
        captured,
        expected_request=ContentHash("sha256", request.digest),
        max_bytes=maximum,
        max_source_bytes=maximum,
        max_total_bytes=request.limits.max_total_bytes,
        max_artifacts=request.limits.max_objects,
    )
    manifest = read_json(root / "capture", captured, max_bytes=maximum)
    diagnostics = manifest["diagnostic_roots"]
    if type(diagnostics) is not list:
        raise ValueError("worker capture must declare its diagnostic roots")
    if (
        artifact_from_mapping(manifest["training"]) != training
        or artifact_from_mapping(manifest["generation"])
        != worker_artifact(row["generation"], "run")
        or worker_artifact(row["diagnostics"], "run")
        not in tuple(artifact_from_mapping(a) for a in diagnostics)
    ):
        raise ValueError("worker result is not the verified capture lineage")
    return row


def run_object_worker(
    objects: D1TrainerMailboxObjectPort,
    marker: D1TrainerMailboxObjectIdentity,
    root: Path,
    *,
    max_request_bytes: int,
    repo_root: Path | None = None,
    sink: MetricSink | None = None,
) -> D1TrainerMailboxObjectIdentity:
    """Publish terminal status only after its snapshot is externally readable.

    Invalid request markers are not trusted enough to select a response namespace.
    Once a request is admitted, failure publication preserves partial owned outputs.
    This function never provisions a pod or authorizes another experiment.

    ``sink``, when given, is threaded to
    :func:`~entropic_substrate.ladder.run.worker_execution.execute_probe_request`
    unchanged -- an additive observability mirror (RUNG 2b), never part of
    this function's own custody/snapshot contract. Its lifecycle (including
    ``finish()``) belongs to the caller that built it.
    """
    request = read_published_probe_request(objects, marker, max_request_bytes=max_request_bytes)
    maximum = request.limits.max_object_bytes
    stage = "materialization"
    # Owned by exactly this ONE run_object_worker invocation (never
    # module-level state): the checkpoint-snapshot publication ledger and
    # the runner's L1/L2 timing mirror it is combined with in each
    # checkpoint event's `publication` block (FOR-1567).
    ledger: PublicationLedger = {}
    publication_timing: dict[str, float] = {}
    # FOR-1662: the /run root's admission memo, handed back by
    # execute_probe_request (a caller-owned out-parameter, like
    # publication_timing). Owned by this ONE invocation, never module-level.
    admission_memos: dict[str, SealMemo] = {}
    # h-run-lifecycle-v1 pairing (FOR-1639): run_end is written only for a
    # run_start that start() logged earlier in THIS invocation, and the failed
    # run_end's clock starts immediately before execute_probe_request.
    started_logged = False
    executed_at = 0.0
    try:
        with narrated_stage("materialization"):
            materialize_probe_request(objects, marker, root, max_request_bytes=max_request_bytes)
        (root / "worker").mkdir()

        def start(runtime: dict[str, object]) -> None:
            nonlocal started_logged
            started = write_json(
                root,
                "worker/start.json",
                {
                    "schema": "h-worker-start-v1",
                    "request_sha256": request.digest,
                    "runtime": runtime,
                    "meaning": "native_inputs_admitted_before_training_dispatch",
                },
                max_bytes=maximum,
            )
            snapshot = publish_worker_snapshot(
                objects, request, root, ("worker/start.json",), name="started"
            )
            _event(
                objects,
                request,
                "started",
                {
                    "status": "inputs_admitted",
                    "runtime": runtime,
                    "record": artifact_reference(root, started),
                    "snapshot": asdict(snapshot),
                },
            )
            if sink is not None:
                sink.log(
                    {
                        "schema": LIFECYCLE_SCHEMA,
                        "event": "run_start",
                        "request_sha256": request.digest,
                        "image": request.image,
                        "code_commit": request.code_commit,
                        "profile_identity": request.profile.identity,
                        "run_class": request.profile.run_class,
                        "runtime": runtime,
                    }
                )
                started_logged = True

        def checkpoint(artifact: ArtifactIdentity) -> None:
            row = read_json(root, artifact, max_bytes=maximum)
            total = row["successful_optimizer_steps"]
            if type(total) is not int or total < 1:
                raise ValueError("worker checkpoint must report measured successful steps")
            paths = _paths(root, ("run", "worker"))
            retry = VerificationReadRetry()
            ledgered_before = len(ledger)
            l3_started = time.perf_counter()
            snapshot = publish_worker_snapshot(
                objects,
                request,
                root,
                paths,
                name=f"checkpoint-{total:08d}",
                ledger=ledger,
                retry=retry,
            )
            l3_seconds = time.perf_counter() - l3_started
            publication = _publication(
                ledger, ledgered_before, paths, l3_seconds, retry, publication_timing
            )
            _event(
                objects,
                request,
                f"checkpoint-{total:08d}",
                {
                    "status": "checkpoint",
                    "successful_optimizer_steps": total,
                    "training": artifact_reference(root, artifact),
                    "snapshot": asdict(snapshot),
                    "publication": publication,
                },
            )
            if sink is not None:
                # FOR-1642 (DN-1643-PORT): the sealed checkpoint's custody evidence on the
                # run itself, so the MLflow run answers "which checkpoints exist and where"
                # without an R2 listing. Only after the checkpoint event was published, and
                # only here: never for `started` or the final snapshot. The values are the
                # snapshot MARKER identity publish_worker_snapshot returned (the identity
                # materialize_worker_snapshot accepts, binding the sealed files' sha256s),
                # and the SAME l3_seconds float the event's `publication` block carries
                # (`float(tag) == publication.l3_seconds`): no new clock. Tag values are
                # strings (MetricSink.annotate refuses anything else).
                evidence = f"checkpoint/{total:08d}"
                sink.annotate(
                    {
                        f"{evidence}/key": snapshot.key,
                        f"{evidence}/sha256": snapshot.sha256,
                        f"{evidence}/size_bytes": str(snapshot.size_bytes),
                        f"{evidence}/publication_seconds": repr(l3_seconds),
                    }
                )

        def stop_requested(phase: str, step: int) -> bool:
            """One bounded read per call site; never retries an absent marker.

            Reuses rung 1's :class:`VerificationReadRetry` for its attempt
            count and backoff schedule only (its public fields), never its
            private ``_verified_read`` (which needs an expected identity and
            payload this poll does not have). A raised exception is retried
            up to ``attempts`` total and then re-raised unchanged; a clean
            ``None`` (absent) or an object read is never retried.
            """
            retry = VerificationReadRetry()
            key = stop_marker_key(request)
            received = None
            for index in range(retry.attempts):
                try:
                    received = objects.read(key)
                    break
                except Exception as poll_error:
                    if index == retry.attempts - 1:
                        raise
                    # FOR-1637: a re-attempted poll is a WARNING of its own event
                    # (not publish.retry, and no event for a clean/absent poll).
                    polled = {
                        "key": key,
                        "attempt": index + 1,
                        "max_attempts": retry.attempts,
                        "reason": type(poll_error).__name__,
                    }
                    logger.bind(event=STOP_MARKER_RETRY, **polled).warning(
                        render(STOP_MARKER_RETRY, **polled)
                    )
                    retry.sleep(retry.backoff_seconds[min(index, len(retry.backoff_seconds) - 1)])
            if received is None:
                return False
            stop = stop_marker_hits(received.payload, request, (phase, step))
            if stop:
                logger.bind(event=STOP_MARKER_OBSERVED, key=key).info(
                    render(STOP_MARKER_OBSERVED, key=key)
                )
            return stop

        stage = "execution"
        executed_at = time.perf_counter()
        result = execute_probe_request(
            request,
            root,
            repo_root=repo_root,
            on_checkpoint=checkpoint,
            on_start=start,
            sink=sink,
            publication_timing=publication_timing,
            operator_stop=stop_requested,
            admission_memos=admission_memos,
        )
        with narrated_stage("result_admission"):
            result = _admit_result(root, request, result, admission_memos.get("run"))
        result_artifact = write_json(root, "worker/result.json", result, max_bytes=maximum)
        stage = "final_snapshot"
        name = (
            "paused"
            if result["status"] == "paused"
            else "stopped"
            if result["status"] == "stopped_by_operator"
            else "complete"
        )
        final_paths = _paths(root, ("run", "capture", "verification", "worker"))
        final_retry = VerificationReadRetry()
        final_ledgered_before = len(ledger)
        l3_started = time.perf_counter()
        snapshot = publish_worker_snapshot(
            objects,
            request,
            root,
            final_paths,
            name=name,
            ledger=ledger,
            retry=final_retry,
        )
        l3_seconds = time.perf_counter() - l3_started
        # DN-1567-C (operator, 2026-09-29): the final-snapshot leg -- shared
        # by all three terminal names, no branch on `name` at this call site
        # -- reuses the SAME run's ledger (never a fresh {}) and a fresh
        # VerificationReadRetry(), exactly like the checkpoint leg.
        #
        # l1_seconds/l2_seconds are 0.0 here: NOT MEASURED for this leg. L1
        # (seal_step's ancestor walk) and L2 (read_training_progress) time
        # the training loop's own seal chain, and that walk already
        # completed -- and was already reported -- on the last checkpoint-*
        # event the runner emitted immediately before this terminal status
        # (runner.py fires `on_checkpoint` at every seal boundary it
        # reaches, unconditionally, including the final one). Reusing those
        # same numbers here would double-report one measurement under two
        # event names, not record a second one.
        #
        # RETRACTION (house rule 4, FOR-1662): this comment used to say two
        # further FULL-CHAIN re-walks happen on this leg (`worker_execution.py`'s
        # post-training readback and `_admit_result`'s `read_training_progress`)
        # and that neither is instrumented, citing a line (:133) that no
        # longer points there. Now both reuse the /run memo the runner's
        # completion walk populated (they read the chain from memory, not from
        # disk) and each is instrumented by its own `admission.start`/
        # `admission.end` pair. Those seconds are NEVER reused in this block:
        # 0.0 still means "not measured for this leg", never "measured zero".
        publication = _publication(
            ledger, final_ledgered_before, final_paths, l3_seconds, final_retry, {}
        )
        event_value: dict[str, object] = {
            "status": result["status"],
            "result": artifact_reference(root, result_artifact),
            "snapshot": asdict(snapshot),
            "publication": publication,
        }
        if result["status"] == "stopped_by_operator":
            event_value["successful_optimizer_steps"] = result["successful_optimizer_steps"]
        terminal = _event(objects, request, name, event_value)
    except Exception as error:
        status = (
            "scientific_gap"
            if isinstance(error, UndefinedDirection)
            else "diagnostic_gap"
            if isinstance(error, SpanPopulationGap)
            else "execution_defect"
        )
        failure = {
            "schema": "h-worker-failure-v1",
            "request_sha256": request.digest,
            "scientific_authority": "none",
            "status": status,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_code": getattr(error, "code", type(error).__name__),
            "detail": str(error),
        }
        try:
            try:
                (root / "worker").mkdir(exist_ok=True)
                write_json(root, "worker/failure.json", failure, max_bytes=maximum)
                snapshot = publish_worker_snapshot(
                    objects,
                    request,
                    root,
                    _paths(root, ("run", "capture", "verification", "worker")),
                    name="failure",
                )
            except Exception as delivery_error:
                try:
                    _event(
                        objects,
                        request,
                        "failure",
                        {
                            "status": status,
                            "stage": stage,
                            "error_type": type(error).__name__,
                            "error_code": getattr(error, "code", type(error).__name__),
                            "snapshot": None,
                            "evidence_delivery": "failed",
                            "delivery_error_type": type(delivery_error).__name__,
                        },
                    )
                except Exception as notification_error:
                    raise ExceptionGroup(
                        "worker failure and unavailable evidence delivery",
                        [error, delivery_error, notification_error],
                    ) from error
                raise ExceptionGroup(
                    "worker failure and evidence delivery defects", [error, delivery_error]
                ) from error
            try:
                _event(
                    objects,
                    request,
                    "failure",
                    {
                        "status": status,
                        "stage": stage,
                        "error_type": type(error).__name__,
                        "error_code": getattr(error, "code", type(error).__name__),
                        "snapshot": asdict(snapshot),
                        "evidence_delivery": "captured",
                    },
                )
            except Exception as notification_error:
                raise ExceptionGroup(
                    "worker failure and terminal publication defect", [error, notification_error]
                ) from error
            raise
        except Exception as propagated:
            # Every exit of this arm (the three ExceptionGroup raises and the
            # bare raise above) lands here, after R2 custody was attempted in
            # full. A sink defect surfaces beside the original error, never
            # replaces it, and a healthy sink leaves the propagated exception
            # exactly as the arm raised it.
            if started_logged and sink is not None:
                try:
                    sink.log(
                        {
                            "schema": LIFECYCLE_SCHEMA,
                            "event": "run_end",
                            "status": "failed",
                            "result_status": status,
                            "wall_seconds": time.perf_counter() - executed_at,
                            "successful_optimizer_steps": None,
                        }
                    )
                except Exception as sink_error:
                    raise ExceptionGroup(
                        "worker failure and lifecycle sink defect", [propagated, sink_error]
                    ) from propagated
            raise
    # Written only after the terminal marker above was published, and outside
    # the try: a sink defect here must not publish a failure marker after a
    # complete/paused/stopped one, so it propagates as the sink's own error.
    if started_logged and sink is not None:
        sink.log(
            {
                "schema": LIFECYCLE_SCHEMA,
                "event": "run_end",
                "status": "ok",
                "result_status": result["status"],
                "wall_seconds": result["wall_seconds"],
                "successful_optimizer_steps": result["successful_optimizer_steps"],
            }
        )
    return terminal
