"""One request, checkpoint snapshots, and marker-last worker outcomes over R2 ports."""

import json
import math
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import HProfile, RunSchedule
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.core.sentence.scoring import LatentUsageEstimator
from entropic_substrate.kernel.artifacts import artifact_from_mapping
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.register.h_preservation import H1ExclusionReason
from entropic_substrate.ladder.run.journal import (
    artifact_reference,
    read_json,
    schema_records,
    write_json,
)
from entropic_substrate.ladder.run.native_verification import read_native_verification
from entropic_substrate.ladder.run.pair_run_tags import CellLifecycle, PairOutcome
from entropic_substrate.ladder.run.runner import (
    PROGRESS_SCHEMA_V2,
    CellScope,
    ReceiptBinding,
    read_training_progress,
)
from entropic_substrate.ladder.run.seals import CorpusMemo, SealMemo, narrated_admission, read_seal
from entropic_substrate.ladder.run.span_states import SpanPopulationGap
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker_execution import (
    CellHooks,
    CellSinks,
    execute_probe_request,
    narrated_stage,
    verdict_record,
)
from entropic_substrate.ladder.run.worker_log_events import (
    COMPLETION_WRITTEN,
    STOP_MARKER_OBSERVED,
    STOP_MARKER_RETRY,
    render,
)
from entropic_substrate.ladder.run.worker_request import (
    RESULT_SCHEMA_V1,
    RESULT_SCHEMA_V2,
    AnyProbeRequest,
    PairProbeRequest,
    ProbeRequest,
    parse_pair_result,
    refuse_pair_result_for_single,
    refuse_recipient_warmup_target,
    refuse_version,
    worker_artifact,
    worker_cell_artifact,
)
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
# FOR-1693 (DH-44, D10): a pair's marker names the CELL it stops, target {phase, step, cell}.
# A single-profile marker stays byte-identical v1; a v1 marker on a pair and a v2 marker on a
# single-profile request are each refused by name.
STOP_MARKER_SCHEMA_V2 = "h-probe-stop-marker-v2"

# h-run-lifecycle-v1 (FOR-1639): the run_start / run_end records this worker
# writes through its MetricSink. Read back by
# ladder.capture.reader.h_telemetry.final_h_run_end, which holds its own copy
# of the literal (the capture layer never imports from ladder/run).
LIFECYCLE_SCHEMA = "h-run-lifecycle-v1"

type StopTarget = tuple[str, int] | tuple[str, int, str]


def stop_marker_key(request: AnyProbeRequest) -> str:
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


def _require_seal(schedule: RunSchedule, phase: str, step: int) -> None:
    """Same coordinate rule as ``ProbeRequest.pause``: the marker is immutable once published,
    so a target the runner never reaches must be refused here, at the verb, rather than
    discovered as a timeout."""
    budget = schedule.warmup_steps if phase == "codec_warmup" else schedule.steps
    is_seal = step == budget or step % schedule.checkpoint_interval == 0
    if phase == "prediction":
        is_seal = is_seal or step % schedule.evaluation_interval == 0
    if phase not in ("codec_warmup", "prediction") or not 1 <= step <= budget or not is_seal:
        raise ValueError("stop marker target is outside the request schedule")


def _cell_target(target: object, request: PairProbeRequest) -> tuple[str, int, str]:
    """``(phase, step, cell)`` with a declared cell token; the schedule is checked by the caller."""
    if type(target) is not tuple or len(target) != 3 or target[2] not in request.tokens:
        raise ValueError("a pair stop marker target must be (non-empty phase, positive step, cell)")
    phase, step = _valid_target(target[:2])
    return phase, step, cast(str, target[2])


def _require_cell_seal(request: PairProbeRequest, phase: str, step: int, cell: str) -> None:
    """The named cell's schedule rule, then the recipient's prediction-only rule (the request
    gate's own, so a target the recipient never reaches is refused here, at the verb)."""
    _require_seal(request.cells[request.tokens.index(cell)].profile.schedule, phase, step)
    if cell == request.tokens[1]:
        refuse_recipient_warmup_target(phase)


def encode_stop_marker(request: AnyProbeRequest, target: StopTarget | None = None) -> bytes:
    """A pure function of (schema, request digest, target): no `at`, no nonce.

    ``target`` names the one seal ``(phase, step)`` at which the worker stops;
    ``None`` omits the key, so the bytes equal the pre-target marker and the
    worker stops at the next seal it polls. A pair (FOR-1693) always names its cell,
    ``(phase, step, cell)``, validated against THAT cell's schedule, and the marker is
    ``h-probe-stop-marker-v2``.
    """
    if isinstance(request, PairProbeRequest):
        if target is None:
            raise ValueError("a pair stop marker names its cell: (phase, step, cell)")
        phase, step, cell = _cell_target(target, request)
        _require_cell_seal(request, phase, step, cell)
        return _canonical(
            {
                "schema": STOP_MARKER_SCHEMA_V2,
                "request_sha256": request.digest,
                "target": {"phase": phase, "step": step, "cell": cell},
            }
        )
    row: dict[str, object] = {"schema": STOP_MARKER_SCHEMA, "request_sha256": request.digest}
    if target is not None:
        phase, step = _valid_target(target)
        _require_seal(request.profile.schedule, phase, step)
        row["target"] = {"phase": phase, "step": step}
    return _canonical(row)


def _read_stop_marker(payload: bytes, request: AnyProbeRequest) -> tuple[bool, StopTarget | None]:
    value = json.loads(payload)
    schema = value.get("schema") if type(value) is dict else None
    if isinstance(request, PairProbeRequest):
        if schema == STOP_MARKER_SCHEMA:
            refuse_version(STOP_MARKER_SCHEMA, STOP_MARKER_SCHEMA_V2, "stop marker")
        pair_row = recovery_record(value, "schema request_sha256 target")
        if pair_row["schema"] != STOP_MARKER_SCHEMA_V2:
            raise ValueError("stop marker schema is foreign")
        pair_digest = pair_row["request_sha256"]
        if type(pair_digest) is not str:
            raise ValueError("stop marker request digest is invalid")
        raw_target = pair_row["target"]
        if type(raw_target) is not dict or set(raw_target) != {"phase", "step", "cell"}:
            raise ValueError(
                "a pair stop marker target must be (non-empty phase, positive step, cell)"
            )
        cell_target = _cell_target(
            (raw_target["phase"], raw_target["step"], raw_target["cell"]), request
        )
        if pair_digest != request.digest:
            return False, None
        if _canonical(pair_row) != payload:
            raise ValueError("stop marker is not canonical JSON")
        _require_cell_seal(request, cell_target[0], cell_target[1], cell_target[2])
        return True, cell_target
    if schema == STOP_MARKER_SCHEMA_V2:
        raise ValueError(
            f"stop marker schema {STOP_MARKER_SCHEMA_V2!r} names a cell; "
            f"a single-profile request admits {STOP_MARKER_SCHEMA}"
        )
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


def classify_stop_marker(payload: bytes, request: AnyProbeRequest) -> bool:
    """True: this marker is this request's own. False: another request's, ignored.

    Everything else -- bad JSON/UTF-8, non-canonical bytes, a missing or
    extra key, a foreign schema, a malformed target -- raises. The schema
    check runs before the request digest check.
    """
    return _read_stop_marker(payload, request)[0]


def stop_marker_hits(
    payload: bytes, request: AnyProbeRequest, seal: tuple[str, int], cell: str | None = None
) -> bool:
    """True when the marker is this request's own AND names no target or ``seal``.

    A pair's poll names its ``cell``: a v2 marker stops exactly the named cell at exactly its
    named seal, and a pair marker always carries a target.
    """
    own, target = _read_stop_marker(payload, request)
    if not own:
        return False
    if isinstance(request, PairProbeRequest):
        if cell is None:
            raise ValueError("a pair stop poll names its cell")
        return target == (seal[0], seal[1], cell)
    return target is None or target == seal


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
#
# FOR-1693: `excluded` is the pair's one NEW non-failure terminal. A training-phase scientific
# exclusion (a recorded MISSING_ENDPOINT or NONFINITE_ENDPOINT) returns normally and must be neither
# `complete` with another status (a false completion) nor `failure` (a relabelled v1 status); its
# status is `pair_excluded`, and the event schema stays `h-worker-event-v1` with its key set.
_TERMINAL_MARKERS = frozenset({"complete", "paused", "stopped", "failure", "excluded"})


def _event(
    objects: D1TrainerMailboxObjectPort,
    request: AnyProbeRequest,
    name: str,
    value: dict[str, object],
    *,
    run_identity: str | None = None,
) -> D1TrainerMailboxObjectIdentity:
    """``run_identity`` defaults to the request's own (one profile's identity, or a pair's pair
    id); a CELL-scoped event of a pair passes that cell's profile identity (FOR-1693, P5-19)."""
    payload = _canonical(
        {
            "schema": "h-worker-event-v1",
            "request_sha256": request.digest,
            "run_identity": request.run_identity if run_identity is None else run_identity,
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
    timing: Mapping[str, float],
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


def _checkpoint_callback(
    objects: D1TrainerMailboxObjectPort,
    request: AnyProbeRequest,
    root: Path,
    *,
    ledger: PublicationLedger,
    publication_timing: MutableMapping[str, float],
    sink: MetricSink | None,
    maximum: int,
    cell: str | None,
    run_identity: str,
) -> Callable[[ArtifactIdentity], None]:
    """The ONE checkpoint publisher: single-profile (``cell is None``) or one cell of a pair.

    Names differ only by the cell token (``checkpoint-<cell>-<total>``), so donor and recipient
    sealing at the same global step never collide; the event carries ``run_identity``.
    """
    prefix = "" if cell is None else f"{cell}-"

    def checkpoint(artifact: ArtifactIdentity) -> None:
        row = read_json(root, artifact, max_bytes=maximum)
        total = row["successful_optimizer_steps"]
        if type(total) is not int or total < 1:
            raise ValueError("worker checkpoint must report measured successful steps")
        name = f"checkpoint-{prefix}{total:08d}"
        paths = _paths(root, ("run", "worker"))
        retry = VerificationReadRetry()
        ledgered_before = len(ledger)
        l3_started = time.perf_counter()
        snapshot = publish_worker_snapshot(
            objects,
            request,
            root,
            paths,
            name=name,
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
            name,
            {
                "status": "checkpoint",
                "successful_optimizer_steps": total,
                "training": artifact_reference(root, artifact),
                "snapshot": asdict(snapshot),
                "publication": publication,
            },
            run_identity=run_identity,
        )
        if sink is not None:
            # FOR-1642 (DN-1643-PORT): the sealed checkpoint's custody evidence on the
            # run itself, so the MLflow run answers "which checkpoints exist and where"
            # without an R2 listing. Only after the checkpoint event was published, and
            # only here: never for ``started`` or the final snapshot. The values are the
            # snapshot MARKER identity publish_worker_snapshot returned (the identity
            # materialize_worker_snapshot accepts, binding the sealed files' sha256s),
            # and the SAME l3_seconds float the event's ``publication`` block carries
            # (``float(tag) == publication.l3_seconds``): no new clock. Tag values are
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

    return checkpoint


def _stop_poll(
    objects: D1TrainerMailboxObjectPort, request: AnyProbeRequest, cell: str | None
) -> Callable[[str, int], bool]:
    """The ONE operator-stop poll: single-profile (``cell is None``) or one cell of a pair."""

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
        stop = stop_marker_hits(received.payload, request, (phase, step), cell)
        if stop:
            logger.bind(event=STOP_MARKER_OBSERVED, key=key).info(
                render(STOP_MARKER_OBSERVED, key=key)
            )
        return stop

    return stop_requested


def _admit_result(
    root: Path,
    request: ProbeRequest,
    value: object,
    memo: SealMemo | None = None,
    corpora: CorpusMemo | None = None,
) -> dict[str, object]:
    if type(value) is dict and value.get("schema") == RESULT_SCHEMA_V2:
        # FOR-1693: result v2 belongs to a pair; a single-profile request admits v1 only.
        refuse_pair_result_for_single()
    row = recovery_record(
        value,
        "schema request_sha256 run_identity scientific_authority status runtime successful_optimizer_steps training generation diagnostics capture verification wall_seconds",
    )
    if (
        row["schema"] != RESULT_SCHEMA_V1
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
            corpora=corpora,
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


def _execution_stop_confirmed(
    output: Path,
    cell: str,
    profile: HProfile,
    failure: dict[str, object] | None,
    *,
    max_bytes: int,
) -> bool:
    """P5-13's closed derivation sources for an execution-class cell, in order: the cell's own
    on-disk ``h-probe-training-failure-v2`` record (status ``execution_defect``); else the
    donor's missing warm-up-final seal; else the recipient root holding no progress receipt.

    A refused resume admission (a donor resuming from its own receipt, a recipient from the
    hand-off receipt or from its own) writes the first source since PR #430 round 1, so its
    present receipt never reads as "did not stop"."""
    if failure is not None:
        return failure.get("status") == "execution_defect"
    if cell == "donor":
        warmup_final = [
            seal
            for _, seal in schema_records(output, "h-checkpoint-seal-v1", max_bytes=max_bytes)
            if seal.get("phase") == "codec_warmup"
            and seal.get("step") == profile.schedule.warmup_steps
        ]
        return not warmup_final
    return not schema_records(output, PROGRESS_SCHEMA_V2, max_bytes=max_bytes)


def _admit_pair_result(
    root: Path,
    request: PairProbeRequest,
    value: object,
    memos: Mapping[str, SealMemo],
    corpora: CorpusMemo | None = None,
) -> dict[str, object]:
    """Re-derive every cell outcome and the pair verdict FROM DISK (FOR-1693, P5-16).

    The structure is :func:`parse_pair_result`'s (a v1 result for a pair is refused by name
    there). Each cell's training is re-read through ``read_training_progress`` against ITS OWN
    root, memo and expected binding; each completed cell's native verification is re-read from
    ``capture/<cell>/`` and ``verification/<cell>/``; a scientific or execution stop is
    confirmed against the training failure record the runner wrote. The verdict the result
    claims must equal the one those facts derive, so it cannot drift from the evidence.
    """
    row = parse_pair_result(value, request)
    maximum = request.limits.max_object_bytes
    entries = cast(list[dict[str, object]], row["cells"])
    derived_scientific: tuple[H1ExclusionReason, str] | None = None
    derived_defect: str | None = None
    for entry, cell in zip(entries, request.cells, strict=True):
        token, profile, outcome = cell.token, cell.profile, entry["outcome"]
        output = root / "run" / token
        expected = ReceiptBinding.of(
            CellScope(request.binding.block_id, request.binding.pair_id, token), request.authority
        )
        report = None
        if entry["training"] is not None:
            training = worker_cell_artifact(entry["training"], "run", token)
            memo = memos.get(f"run/{token}")
            with narrated_admission("result_admission", "run", memo):
                report = read_training_progress(
                    output,
                    training,
                    profile,
                    root / "source",
                    max_source_bytes=maximum,
                    max_artifact_bytes=maximum,
                    memo=memo,
                    expected=expected,
                    corpora=corpora,
                )
        steps = entry["successful_optimizer_steps"]
        if steps != (0 if report is None else report["successful_optimizer_steps"]):
            raise ValueError("worker result step count differs from admitted training")
        failure = None
        if entry["failure"] is not None:
            failure = read_json(
                output, worker_cell_artifact(entry["failure"], "run", token), max_bytes=maximum
            )
            if (
                failure.get("schema") != "h-probe-training-failure-v2"
                or failure.get("run_identity") != profile.identity
            ):
                raise ValueError("worker result failure record belongs to another run")
        if outcome in ("paused", "stopped_by_operator"):
            if report is None or report["status"] != "paused":
                raise ValueError("paused worker result cannot claim later stages")
        elif outcome == "complete":
            if report is None or report["status"] != "training_complete":
                raise ValueError("worker completion lacks complete training")
            captured = worker_cell_artifact(entry["capture"], "capture", token)
            verified = worker_cell_artifact(entry["verification"], "verification", token)
            read_native_verification(
                root / "verification" / token,
                verified,
                root / "capture" / token,
                captured,
                expected_request=ContentHash("sha256", request.digest),
                max_bytes=maximum,
                max_source_bytes=maximum,
                max_total_bytes=request.limits.max_total_bytes,
                max_artifacts=request.limits.max_objects,
            )
            manifest = read_json(root / "capture" / token, captured, max_bytes=maximum)
            diagnostics = manifest["diagnostic_roots"]
            if (
                type(diagnostics) is not list
                or artifact_from_mapping(manifest["training"])
                != worker_cell_artifact(entry["training"], "run", token)
                or artifact_from_mapping(manifest["generation"])
                != worker_cell_artifact(entry["generation"], "run", token)
                or worker_cell_artifact(entry["diagnostics"], "run", token)
                not in tuple(artifact_from_mapping(a) for a in diagnostics)
            ):
                raise ValueError("worker result is not the verified capture lineage")
        elif outcome == "scientific_gap":
            if failure is None or failure.get("status") != "scientific_gap":
                raise ValueError("a scientific cell outcome needs its scientific failure record")
            last_seal = failure["last_seal"]
            warmed = (
                None
                if last_seal is None
                else read_seal(
                    output,
                    artifact_from_mapping(last_seal),
                    profile,
                    root / "source",
                    max_bytes=maximum,
                    max_source_bytes=maximum,
                    corpora=corpora,
                ).warmup_final
            )
            if derived_scientific is None:
                derived_scientific = (
                    H1ExclusionReason.NONFINITE_ENDPOINT
                    if warmed is not None
                    else H1ExclusionReason.MISSING_ENDPOINT,
                    token,
                )
        elif outcome == "execution_defect":
            if not _execution_stop_confirmed(output, token, profile, failure, max_bytes=maximum):
                raise ValueError("an execution cell outcome needs its failure evidence on disk")
            if derived_defect is None:
                derived_defect = token
    outcomes = [entry["outcome"] for entry in entries]
    derived: dict[str, object]
    if derived_defect is not None:
        derived = verdict_record("excluded", H1ExclusionReason.READER_REFUSAL, derived_defect)
    elif derived_scientific is not None:
        derived = verdict_record("excluded", derived_scientific[0], derived_scientific[1])
    elif outcomes == ["complete", "complete"]:
        derived = verdict_record("complete")
    else:
        derived = verdict_record("none")
    if row["verdict"] != derived:
        raise ValueError("worker result verdict differs from the verdict its cells derive")
    return row


def _run_start_record(
    request: AnyProbeRequest, profile: HProfile, runtime: dict[str, object]
) -> dict[str, object]:
    """The one ``run_start`` record both write sites build (single profile and pair cell).

    DH-51 (R1): ``profile`` is the manifest of the SAME ``HProfile`` whose ``identity`` fills
    ``profile_identity`` (the cell's profile on the pair path, the request's on the single path),
    so a reader can authenticate it by rebuilding the profile and comparing the identity. The
    schema name is unchanged: an absent ``profile`` (every earlier capture) is the declared
    default. RETRACTION (house rule 4, 2026-11-04): this record had exactly eight keys, pinned by
    equality; it has nine now.
    """
    return {
        "schema": LIFECYCLE_SCHEMA,
        "event": "run_start",
        "request_sha256": request.digest,
        "image": request.image,
        "code_commit": request.code_commit,
        "profile_identity": profile.identity,
        "run_class": profile.run_class,
        "runtime": runtime,
        "profile": profile.manifest,
    }


def _cell_lifecycle(result_status: str) -> CellLifecycle | None:
    """The DH-50 ``esl.lifecycle`` value a cell's close writes; ``None`` writes none.

    A paused cell is ``paused``; an operator stop and every scientific-class stop (a DH-42
    training gap, a DH-45 post-training scientific or diagnostic gap) is ``stopped``; a complete
    workload is ``completed``. An execution-class close writes nothing, so the cell keeps the
    ``live`` it opened with: the ruling's "a crash never resumed keeps live".
    """
    match result_status:
        case "probe_workload_complete":
            return "completed"
        case "paused":
            return "paused"
        case "stopped_by_operator" | "scientific_gap" | "diagnostic_gap":
            return "stopped"
        case "execution_defect":
            return None
        case other:
            raise ValueError(f"cell result status {other!r} has no DH-50 lifecycle value")


class _PairObserver:
    """The per-cell side channels of one pair run (FOR-1693): lifecycle, events, stop poll.

    Implements :class:`~entropic_substrate.ladder.run.worker_execution.CellObserver`. Exactly one
    ``run_start`` and one ``run_end`` per constructed cell, each bound to THAT cell's own
    on-disk outcome and written to THAT cell's sink; the cell's MLflow run terminates FINISHED
    iff the cell's training returned complete, paused or stopped and its workload returned.
    DH-50: each close replaces the cell's ``esl.lifecycle`` tag (``_cell_lifecycle``) before its
    run terminates, and the pair's hand-off decision lands ONCE on the donor's run.
    """

    def __init__(
        self,
        objects: D1TrainerMailboxObjectPort,
        request: PairProbeRequest,
        root: Path,
        *,
        ledger: PublicationLedger,
        publication_timing: MutableMapping[str, float],
        sinks: CellSinks | None,
        maximum: int,
    ) -> None:
        self._objects, self._request, self._root = objects, request, root
        self._ledger, self._timing, self._sinks = ledger, publication_timing, sinks
        self._maximum = maximum
        self._logged: dict[str, MetricSink] = {}
        self._began: dict[str, float] = {}
        self._settled = False
        self.focused: str | None = None
        self.open: str | None = None

    def identity_of(self, cell: str) -> str:
        return self._request.cells[self._request.tokens.index(cell)].profile.identity

    @property
    def run_identity(self) -> str:
        """The failing cell's identity while a cell is in scope, else the pair's."""
        scoped = self.open or self.focused
        return self._request.run_identity if scoped is None else self.identity_of(scoped)

    def focus(self, cell: str) -> None:
        self.focused = cell

    def begin(self, cell: str, profile: HProfile, runtime: dict[str, object]) -> CellHooks:
        request = self._request
        sink = None if self._sinks is None else self._sinks.begin(cell, profile)
        if sink is not None:
            sink.log(_run_start_record(request, profile, runtime))
            self._logged[cell] = sink
        self._began[cell] = time.perf_counter()
        self.open = cell
        return CellHooks(
            sink,
            _checkpoint_callback(
                self._objects,
                request,
                self._root,
                ledger=self._ledger,
                publication_timing=self._timing,
                sink=sink,
                maximum=self._maximum,
                cell=cell,
                run_identity=profile.identity,
            ),
            _stop_poll(self._objects, request, cell),
        )

    def end(
        self,
        cell: str,
        *,
        status: Literal["ok", "failed"],
        result_status: str,
        wall_seconds: float,
        steps: int | None,
    ) -> None:
        sink = self._logged.pop(cell, None)
        if self.open == cell:
            self.open = None
        if self.focused == cell and result_status != "execution_defect":
            # a cell that ended well, or stopped SCIENTIFICALLY, leaves its scope: the pair goes
            # on to its verdict, and a later pair-level failure (result admission, the final
            # snapshot) names the pair. Only an EXECUTION stop keeps it: the raise that follows
            # is that cell's own. RETRACTION (house rule 4, PR #430 round 1): this was keyed on
            # ``status == "ok"``, which kept a scientifically stopped cell in scope, so a later
            # pair-level failure named that cell's identity instead of the pair's.
            self.focused = None
        if sink is None:
            return
        sink.log(
            {
                "schema": LIFECYCLE_SCHEMA,
                "event": "run_end",
                "status": status,
                "result_status": result_status,
                "wall_seconds": wall_seconds,
                "successful_optimizer_steps": steps,
            }
        )
        if self._sinks is not None:
            lifecycle = _cell_lifecycle(result_status)
            if lifecycle is not None:
                self._sinks.lifecycle(cell, lifecycle)
            self._sinks.finish(cell, "FINISHED" if status == "ok" else "FAILED")

    def pair_outcome(self, value: PairOutcome) -> None:
        """Record the pair's hand-off decision on the donor's run (DH-50), exactly once.

        Settled BEFORE the write, so a failing write is not retried by the failure arm's
        :meth:`stop_before_handoff`. No donor run (no provider, or the donor never began) has
        nothing to tag.
        """
        if self._settled:
            raise ValueError(f"the pair's DH-50 outcome is already recorded, refusing {value!r}")
        self._settled = True
        if self._sinks is not None and self._request.tokens[0] in self._began:
            self._sinks.pair_outcome(value)

    def stop_before_handoff(self) -> None:
        """The pair failed before its decision was recorded: the donor never handed off."""
        if not self._settled:
            self.pair_outcome("stopped_before_handoff")

    def fail_open(self, result_status: str) -> None:
        """A post-training stage of the open cell raised: close THAT cell as failed (DH-45)."""
        if self.open is not None:
            cell = self.open
            self.end(
                cell,
                status="failed",
                result_status=result_status,
                wall_seconds=time.perf_counter() - self._began[cell],
                steps=None,
            )


def run_object_worker(
    objects: D1TrainerMailboxObjectPort,
    marker: D1TrainerMailboxObjectIdentity,
    root: Path,
    *,
    max_request_bytes: int,
    repo_root: Path | None = None,
    sink: MetricSink | None = None,
    cell_sinks: CellSinks | None = None,
    latent_usage: LatentUsageEstimator,
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

    FOR-1693 (DH-40, DH-44, DH-45, DH-47): a PAIR request (v3) is observed per cell. ``cell_sinks``
    is the provider ``worker_command`` owns (one sink per cell; a pair handed ``sink`` is refused
    and so is a single-profile request handed ``cell_sinks``). The pair publishes
    ``h-probe-worker-result-v2`` under the terminals ``complete``, ``paused``, ``stopped`` and the
    new ``excluded``; its events carry the failing cell's identity (cell-scoped) or the pair id
    (pair-level). A training-phase or hand-off EXECUTION stop leaves result v2 beside the v1
    failure record in the failure snapshot; a post-training stage failure of either cell raises
    through the v1 failure arm exactly as a single profile's does, with no result v2.
    """
    request = read_published_probe_request(objects, marker, max_request_bytes=max_request_bytes)
    pair = request if isinstance(request, PairProbeRequest) else None
    if pair is not None and sink is not None:
        raise ValueError("a pair is observed per cell: hand it a cell sink provider, not one sink")
    if pair is None and cell_sinks is not None:
        raise ValueError("a single-profile request has no cells to give sinks to")
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
    # A pair keeps one memo per cell root, keyed `run/<cell>`.
    admission_memos: dict[str, SealMemo] = {}
    # DH-41: the process's corpus memos, one per source root (`source`, `capture[/<cell>]`), owned
    # by this ONE invocation and shared by execution and result admission.
    corpus_memos: dict[str, CorpusMemo] = {}
    observer = (
        None
        if pair is None
        else _PairObserver(
            objects,
            pair,
            root,
            ledger=ledger,
            publication_timing=publication_timing,
            sinks=cell_sinks,
            maximum=maximum,
        )
    )
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
                objects,
                request,
                root,
                ("worker/start.json",),
                name="started",
                # FOR-1694: a pair's later snapshots re-list worker/start.json; ledgering it here
                # keeps the failure snapshot from re-PUTting what the run already published
                ledger=ledger if pair is not None else None,
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
            if sink is not None and isinstance(request, ProbeRequest):
                sink.log(_run_start_record(request, request.profile, runtime))
                started_logged = True

        stage = "execution"
        executed_at = time.perf_counter()
        if observer is not None:
            result = execute_probe_request(
                request,
                root,
                repo_root=repo_root,
                on_start=start,
                publication_timing=publication_timing,
                admission_memos=admission_memos,
                cells=observer,
                corpus_memos=corpus_memos,
                latent_usage=latent_usage,
            )
        else:
            result = execute_probe_request(
                request,
                root,
                repo_root=repo_root,
                on_checkpoint=_checkpoint_callback(
                    objects,
                    request,
                    root,
                    ledger=ledger,
                    publication_timing=publication_timing,
                    sink=sink,
                    maximum=maximum,
                    cell=None,
                    run_identity=request.run_identity,
                ),
                on_start=start,
                sink=sink,
                publication_timing=publication_timing,
                operator_stop=_stop_poll(objects, request, None),
                admission_memos=admission_memos,
                corpus_memos=corpus_memos,
                latent_usage=latent_usage,
            )
        with narrated_stage("result_admission"):
            if pair is not None:
                result = _admit_pair_result(
                    root, pair, result, admission_memos, corpus_memos.get("source")
                )
            else:
                if not isinstance(request, ProbeRequest):
                    raise TypeError("a single-profile result admission reads a ProbeRequest")
                result = _admit_result(
                    root,
                    request,
                    result,
                    admission_memos.get("run"),
                    corpus_memos.get("source"),
                )
        result_artifact = write_json(root, "worker/result.json", result, max_bytes=maximum)
        stage = "final_snapshot"
        name = {
            "paused": "paused",
            "stopped_by_operator": "stopped",
            "pair_excluded": "excluded",
        }.get(cast(str, result["status"]), "complete")
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
        if result["status"] == "stopped_by_operator" and pair is None:
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
        # P5-19: the failure EVENT names the failing cell (a hand-off failure is the
        # recipient's); the failure RECORD keeps its v1 key set.
        failing_identity = request.run_identity if observer is None else observer.run_identity
        try:
            try:
                (root / "worker").mkdir(exist_ok=True)
                write_json(root, "worker/failure.json", failure, max_bytes=maximum)
                if pair is not None and (root / "worker" / "result.json").is_file():
                    # P5-13: the result v2 an execution stop left is read back through the same
                    # v2 admission before the snapshot that carries it is published.
                    _admit_pair_result(
                        root,
                        pair,
                        json.loads((root / "worker" / "result.json").read_bytes()),
                        admission_memos,
                        corpus_memos.get("source"),
                    )
                snapshot = publish_worker_snapshot(
                    objects,
                    request,
                    root,
                    _paths(root, ("run", "capture", "verification", "worker")),
                    name="failure",
                    # FOR-1694: a pair's failure snapshot must not re-PUT what this run already
                    # published and verified (measured: 62 files re-published for 1,382.7 s)
                    ledger=ledger if pair is not None else None,
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
                        run_identity=failing_identity,
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
                    run_identity=failing_identity,
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
            if observer is not None:
                # a pair's cell still open (a post-training stage raised) is closed as failed
                # with THIS error's v1 status; a cell the loop already closed is left alone
                try:
                    observer.fail_open(status)
                    # DH-50: a raise before the hand-off decision (a donor execution-class or
                    # DH-45 stop, a failed hand-off, a recipient that never opened) stops the pair
                    observer.stop_before_handoff()
                except Exception as sink_error:
                    raise ExceptionGroup(
                        "worker failure and lifecycle sink defect", [propagated, sink_error]
                    ) from propagated
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
    # (A pair's cells wrote their own run_end as each cell's workload returned.)
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
