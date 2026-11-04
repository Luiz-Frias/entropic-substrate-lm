"""Collect one terminal worker lineage without promoting partial custody to completion.

FOR-1694 (h1-l4c): a PAIR request is collected as ``h-probe-terminal-collection-v2``: ONE entry
per cell (its receipt, its result v2 entry, its capture and verification roots), the pair verdict
and the terminal name it was collected under (``complete``, ``paused``, ``stopped``, ``excluded``,
``failure``). Result v2 is read under every terminal that can carry it: ``complete``, ``excluded``
and, per DH-47, the ``failure`` terminal (a training-phase execution stop leaves it beside the v1
failure record). Every pair verdict is keyed on the VERDICT, never on a terminal's status:
completion v2 reports ``probe_completion_verified`` for an excluded pair too, so
``workload_verified`` (which the controller reads to pick the stop reason) is true only for the
``complete`` verdict. A pair whose pod never began a cell leaves a failure terminal whose snapshot
holds no cell evidence and no result v2, and is collected as a typed ``pair_not_started``
failure; with no terminal at all the answer is still ``None``. A single-profile request keeps
``h-probe-terminal-collection-v1`` byte for byte, and a pair collection presented as v1 is
refused by name, naming v2 as the minimum (``admit_pair_collection``).
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    publish_d1_atomic_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.run.h_probe_events import ObservedProbeEvent, read_probe_event
from entropic_substrate.ladder.run.journal import read_json, write_json
from entropic_substrate.ladder.run.runner import (
    PROGRESS_FIELDS,
    PROGRESS_SCHEMA_V1,
    PROGRESS_SCHEMA_V2,
    refuse_progress_version,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker_completion import (
    admit_worker_runtime,
    receive_worker_completion,
)
from entropic_substrate.ladder.run.worker_request import (
    RESULT_SCHEMA_V1,
    RESULT_SCHEMA_V2,
    AnyProbeRequest,
    PairProbeRequest,
    ProbeRequest,
    parse_pair_result,
    refuse_pair_result_for_single,
    refuse_version,
    worker_artifact,
    worker_cell_artifact,
)
from entropic_substrate.ladder.run.worker_snapshot import (
    materialize_worker_snapshot,
)

COLLECTION_SCHEMA_V1 = "h-probe-terminal-collection-v1"
COLLECTION_SCHEMA_V2 = "h-probe-terminal-collection-v2"
# A pair is collected under any of these terminals; the order is the probe order.
_PAIR_TERMINAL_NAMES = ("complete", "paused", "stopped", "excluded", "failure")
_FAILURE_FIELDS = (
    "schema request_sha256 scientific_authority status stage error_type error_code detail"
)

_RESULT_FIELDS = "schema request_sha256 run_identity scientific_authority status runtime successful_optimizer_steps training generation diagnostics capture verification wall_seconds"


def _single_result(value: dict[str, object]) -> dict[str, object]:
    """A single-profile request's result record: v1 only; a pair's v2 is refused BY NAME."""
    if value.get("schema") == RESULT_SCHEMA_V2:
        refuse_pair_result_for_single()
    return recovery_record(value, _RESULT_FIELDS)


def collect_probe_terminal(
    objects: D1TrainerMailboxObjectPort,
    request: AnyProbeRequest,
    destination: Path,
    *,
    expected_pod_id: str | None,
    max_marker_bytes: int,
) -> dict[str, object] | None:
    """One observation, one fresh destination; conflicts and invalid evidence fail loud.

    Completion reuses native-receipt admission. Pause/failure capture proves bytes
    and request bindings, not host replay of native model state or a scientific
    classification. A failed copy retains unadmitted partial files for inspection.

    A pair request is collected by :func:`_collect_pair` (collection v2); everything below
    is the single-profile path, byte for byte.
    """
    if isinstance(request, PairProbeRequest):
        return _collect_pair(
            objects,
            request,
            destination,
            expected_pod_id=expected_pod_id,
            max_marker_bytes=max_marker_bytes,
        )
    found = [
        (name, event)
        for name in ("complete", "paused", "stopped", "failure")
        if (
            event := read_probe_event(
                objects, request, name, expected_pod_id=expected_pod_id, max_bytes=max_marker_bytes
            )
        )
        is not None
    ]
    if not found:
        return None
    if len(found) != 1:
        raise ValueError("conflicting terminal claims for one worker request")
    root = normalize_d1_artifact_root(destination)
    if any(root.iterdir()):
        raise ValueError("terminal collection requires a fresh empty destination")
    name, event = found[0]
    report: dict[str, object] = {
        "schema": COLLECTION_SCHEMA_V1,
        "scientific_authority": "none",
        "request_sha256": request.digest,
        "terminal": asdict(event.identity),
        "reported_outcome": event.reported_status,
        "workload_verified": False,
        "reported_optimizer_steps": None,
        "native_validation": "not_replayed_by_host",
    }
    try:
        _capture(objects, request, root, name, event, report, expected_pod_id, max_marker_bytes)
    except Exception as primary:
        try:
            publish_d1_atomic_bytes(root, "terminal.json", event.payload, label="terminal evidence")
        except Exception as custody_error:
            raise ExceptionGroup(
                "terminal recovery and evidence preservation failed", [primary, custody_error]
            ) from primary
        raise
    terminal = publish_d1_atomic_bytes(
        root, "terminal.json", event.payload, label="terminal evidence"
    )
    report["terminal_artifact"] = {"path": "terminal.json", "sha256": terminal.content_hash.digest}
    write_json(root, "collection.json", report, max_bytes=max_marker_bytes)
    return report


def _capture(
    objects: D1TrainerMailboxObjectPort,
    request: ProbeRequest,
    root: Path,
    name: str,
    event: ObservedProbeEvent,
    report: dict[str, object],
    expected_pod_id: str | None,
    max_marker_bytes: int,
) -> None:
    if name == "complete":
        completion = receive_worker_completion(
            objects,
            request,
            event.identity,
            root,
            expected_pod_id=expected_pod_id,
            max_marker_bytes=max_marker_bytes,
        )
        report.update(
            workload_verified=True,
            capture_status="native_completion_receipt_verified",
            reported_optimizer_steps=completion["successful_optimizer_steps"],
            native_validation=completion["native_validation"],
            completion=completion,
        )
    elif event.snapshot is None:
        report["capture_status"] = "missing_failure_snapshot"
    else:
        snapshot = materialize_worker_snapshot(
            objects, request, event.snapshot, root, max_marker_bytes=max_marker_bytes
        )
        row: dict[str, object] = json.loads(event.payload)
        if name == "paused":
            paused = _single_result(
                read_json(
                    root / "worker",
                    worker_artifact(row["result"], "worker"),
                    max_bytes=request.limits.max_object_bytes,
                )
            )
            pause = request.pause
            if pause is None:
                raise ValueError("pause evidence requires an explicitly requested pause")
            expected_steps = pause[1] + (
                request.profile.schedule.warmup_steps if pause[0] == "prediction" else 0
            )
            if (
                paused["schema"] != RESULT_SCHEMA_V1
                or paused["request_sha256"] != request.digest
                or paused["run_identity"] != request.profile.identity
                or paused["scientific_authority"] != "none"
                or paused["status"] != "paused"
                or type(paused["successful_optimizer_steps"]) is not int
                or paused["successful_optimizer_steps"] != expected_steps
                or any(
                    paused[k] is not None
                    for k in ("generation", "diagnostics", "capture", "verification")
                )
            ):
                raise ValueError("paused result differs from its requested lineage or endpoint")
            admit_worker_runtime(paused["runtime"], request, expected_pod_id)
            read_json(
                root / "run",
                worker_artifact(paused["training"], "run"),
                max_bytes=request.limits.max_object_bytes,
            )
            report.update(
                capture_status="pause_bytes_verified",
                reported_optimizer_steps=expected_steps,
                resume_artifact=paused["training"],
            )
        elif name == "stopped":
            stopped = _single_result(
                read_json(
                    root / "worker",
                    worker_artifact(row["result"], "worker"),
                    max_bytes=request.limits.max_object_bytes,
                )
            )
            reported_step = event.reported_optimizer_steps
            if (
                stopped["schema"] != RESULT_SCHEMA_V1
                or stopped["request_sha256"] != request.digest
                or stopped["run_identity"] != request.profile.identity
                or stopped["scientific_authority"] != "none"
                or stopped["status"] != "stopped_by_operator"
                or type(stopped["successful_optimizer_steps"]) is not int
                or stopped["successful_optimizer_steps"] != reported_step
                or any(
                    stopped[k] is not None
                    for k in ("generation", "diagnostics", "capture", "verification")
                )
            ):
                raise ValueError("stopped result differs from its terminal event")
            admit_worker_runtime(stopped["runtime"], request, expected_pod_id)
            raw_receipt = read_json(
                root / "run",
                worker_artifact(stopped["training"], "run"),
                max_bytes=request.limits.max_object_bytes,
            )
            if raw_receipt.get("schema") == PROGRESS_SCHEMA_V1:
                refuse_progress_version(PROGRESS_SCHEMA_V1)
            receipt = recovery_record(raw_receipt, PROGRESS_FIELDS)
            if (
                receipt["schema"] != PROGRESS_SCHEMA_V2
                or receipt["status"] != "paused"
                or receipt["successful_optimizer_steps"] != reported_step
            ):
                raise ValueError("stopped receipt differs from its reported terminal step")
            report.update(
                capture_status="stop_bytes_verified",
                reported_optimizer_steps=reported_step,
                resume_artifact=stopped["training"],
            )
        else:
            _verify_failure_record(
                root, request, cast(list[dict[str, object]], snapshot["files"]), row
            )
            report["capture_status"] = "failure_bytes_verified"
        report["snapshot"] = asdict(event.snapshot)


def _verify_failure_record(
    root: Path,
    request: AnyProbeRequest,
    files: list[dict[str, object]],
    row: dict[str, object],
) -> dict[str, object]:
    """The failure snapshot's unique ``worker/failure.json`` agrees with its terminal event.

    Returns the verified record, so a pair reader can key on the stage it names.
    """
    entries = [item for item in files if item["path"] == "worker/failure.json"]
    if len(entries) != 1:
        raise ValueError("failure snapshot lacks its unique failure record")
    reference = ArtifactIdentity(
        "failure.json", ContentHash("sha256", cast(str, entries[0]["sha256"]))
    )
    failure = recovery_record(
        read_json(root / "worker", reference, max_bytes=request.limits.max_object_bytes),
        _FAILURE_FIELDS,
    )
    if (
        failure["schema"] != "h-worker-failure-v1"
        or failure["request_sha256"] != request.digest
        or failure["scientific_authority"] != "none"
        or any(failure[key] != row[key] for key in ("status", "stage", "error_type", "error_code"))
        or type(failure["detail"]) is not str
    ):
        raise ValueError("failure record differs from its terminal event")
    return failure


# ----------------------------------------------------------------------- the pair (v2)
def admit_pair_collection(value: object, request: PairProbeRequest) -> dict[str, object]:
    """Admit a pair's terminal collection: v2 only, bound to its request and its two cells.

    A collection presented as ``h-probe-terminal-collection-v1`` is refused BY NAME with the
    minimum supported version (v1 names no cell, so it cannot describe a pair). At the committed
    tip no src module imports it yet: the resume builder (``h_probe_resume``) adopts it when the
    staged DG-A reader-member patch lands, and until then that builder still reads v1 only.

    RETRACTED 2026-11-02: the sentence above is false at the tip. ``h_probe_resume`` imports
    ``admit_pair_collection`` and calls it from ``build_pair_resume_request`` (symbols only: line
    numbers drift with every edit).
    """
    if type(value) is not dict:
        raise ValueError("a pair terminal collection must be an object")
    row = cast(dict[str, object], value)
    if row.get("schema") == COLLECTION_SCHEMA_V1:
        refuse_version(COLLECTION_SCHEMA_V1, COLLECTION_SCHEMA_V2, "pair terminal collection")
    cells = row.get("cells")
    if (
        row.get("schema") != COLLECTION_SCHEMA_V2
        or row.get("request_sha256") != request.digest
        or row.get("pair_id") != request.run_identity
        or type(cells) is not list
        or [cast(dict[str, object], c).get("cell") for c in cast(list[object], cells)]
        != list(request.tokens)
    ):
        raise ValueError("the collection is not a pair terminal collection of this request")
    return row


def _cell_entry(
    root: Path, request: PairProbeRequest, index: int, result_entry: dict[str, object] | None
) -> dict[str, object]:
    cell = request.cells[index]
    token = cell.token
    capture_root, verification_root = f"capture/{token}", f"verification/{token}"
    return {
        "cell": token,
        "run_identity": cell.profile.identity,
        "outcome": "unreported" if result_entry is None else result_entry["outcome"],
        "successful_optimizer_steps": None
        if result_entry is None
        else result_entry["successful_optimizer_steps"],
        "receipt": None if result_entry is None else result_entry["training"],
        "result": result_entry,
        "capture_root": capture_root if (root / capture_root).is_dir() else None,
        "verification_root": verification_root if (root / verification_root).is_dir() else None,
    }


def _pair_result(
    root: Path,
    request: PairProbeRequest,
    reference: object,
    *,
    expected_pod_id: str | None,
) -> dict[str, object]:
    """The verified result v2 a terminal (or a failure snapshot) cites, admitted structurally
    and bound to the pod; its evidence is re-read by the caller that needs it."""
    result = parse_pair_result(
        read_json(
            root / "worker",
            worker_artifact(reference, "worker"),
            max_bytes=request.limits.max_object_bytes,
        ),
        request,
    )
    admit_worker_runtime(result["runtime"], request, expected_pod_id)
    return result


def _verify_cell_receipts(
    root: Path, request: PairProbeRequest, result: dict[str, object], outcome: str
) -> None:
    """A paused or stopped cell's receipt is a paused ``h-probe-training-progress-v2`` of ITS cell,
    bound to this pair, at the step its result entry reports."""
    for index, entry in enumerate(cast(list[dict[str, object]], result["cells"])):
        if entry["outcome"] != outcome:
            continue
        cell = request.cells[index]
        raw = read_json(
            root / "run" / cell.token,
            worker_cell_artifact(entry["training"], "run", cell.token),
            max_bytes=request.limits.max_object_bytes,
        )
        if raw.get("schema") == PROGRESS_SCHEMA_V1:
            refuse_progress_version(PROGRESS_SCHEMA_V1)
        receipt = recovery_record(raw, PROGRESS_FIELDS)
        if (
            receipt["schema"] != PROGRESS_SCHEMA_V2
            or receipt["status"] != "paused"
            or receipt["run_identity"] != cell.profile.identity
            or receipt["successful_optimizer_steps"] != entry["successful_optimizer_steps"]
            or (receipt["block"], receipt["pair"], receipt["cell"])
            != (request.binding.block_id, request.binding.pair_id, cell.token)
        ):
            raise ValueError(
                f"the {cell.token} receipt differs from its reported lineage, cell or step"
            )
        if outcome == "paused":
            pause = cell.pause
            if pause is None:
                raise ValueError("pause evidence requires an explicitly requested pause")
            expected = pause[1] + (
                cell.profile.schedule.warmup_steps if pause[0] == "prediction" else 0
            )
            if entry["successful_optimizer_steps"] != expected:
                raise ValueError("paused result differs from its requested lineage or endpoint")


def _collect_pair(
    objects: D1TrainerMailboxObjectPort,
    request: PairProbeRequest,
    destination: Path,
    *,
    expected_pod_id: str | None,
    max_marker_bytes: int,
) -> dict[str, object] | None:
    found = [
        (name, event)
        for name in _PAIR_TERMINAL_NAMES
        if (
            event := read_probe_event(
                objects, request, name, expected_pod_id=expected_pod_id, max_bytes=max_marker_bytes
            )
        )
        is not None
    ]
    if not found:
        return None
    if len(found) != 1:
        raise ValueError("conflicting terminal claims for one worker request")
    root = normalize_d1_artifact_root(destination)
    if any(root.iterdir()):
        raise ValueError("terminal collection requires a fresh empty destination")
    name, event = found[0]
    report: dict[str, object] = {
        "schema": COLLECTION_SCHEMA_V2,
        "scientific_authority": "none",
        "request_sha256": request.digest,
        "pair_id": request.run_identity,
        "terminal": asdict(event.identity),
        "terminal_name": name,
        "reported_outcome": event.reported_status,
        "workload_verified": False,
        "reported_optimizer_steps": None,
        "native_validation": "not_replayed_by_host",
        "verdict": None,
        "cells": [],
    }
    try:
        _capture_pair(
            objects, request, root, name, event, report, expected_pod_id, max_marker_bytes
        )
    except Exception as primary:
        try:
            publish_d1_atomic_bytes(root, "terminal.json", event.payload, label="terminal evidence")
        except Exception as custody_error:
            raise ExceptionGroup(
                "terminal recovery and evidence preservation failed", [primary, custody_error]
            ) from primary
        raise
    terminal = publish_d1_atomic_bytes(
        root, "terminal.json", event.payload, label="terminal evidence"
    )
    report["terminal_artifact"] = {"path": "terminal.json", "sha256": terminal.content_hash.digest}
    write_json(root, "collection.json", report, max_bytes=max_marker_bytes)
    return report


def _capture_pair(
    objects: D1TrainerMailboxObjectPort,
    request: PairProbeRequest,
    root: Path,
    name: str,
    event: ObservedProbeEvent,
    report: dict[str, object],
    expected_pod_id: str | None,
    max_marker_bytes: int,
) -> None:
    row: dict[str, object] = json.loads(event.payload)
    result: dict[str, object] | None = None
    if name in ("complete", "excluded"):
        completion = receive_worker_completion(
            objects,
            request,
            event.identity,
            root,
            expected_pod_id=expected_pod_id,
            max_marker_bytes=max_marker_bytes,
        )
        result = _pair_result(root, request, row["result"], expected_pod_id=expected_pod_id)
        verdict = cast(dict[str, object], completion["verdict"])
        # KEYED ON THE VERDICT, never on a status: completion v2 reports
        # ``probe_completion_verified`` for an excluded pair as well (it only proves the bytes)
        report.update(
            workload_verified=verdict["verdict"] == "complete",
            capture_status="native_completion_receipt_verified"
            if name == "complete"
            else "pair_exclusion_bytes_verified",
            native_validation=completion["native_validation"],
            completion=completion,
        )
    elif event.snapshot is None:
        report["capture_status"] = "missing_failure_snapshot"
    else:
        snapshot = materialize_worker_snapshot(
            objects, request, event.snapshot, root, max_marker_bytes=max_marker_bytes
        )
        files = cast(list[dict[str, object]], snapshot["files"])
        if name in ("paused", "stopped"):
            result = _pair_result(root, request, row["result"], expected_pod_id=expected_pod_id)
            wanted = "paused" if name == "paused" else "stopped_by_operator"
            if result["status"] != wanted:
                raise ValueError(f"{name} result differs from its terminal event")
            _verify_cell_receipts(root, request, result, wanted)
            report["capture_status"] = (
                "pause_bytes_verified" if name == "paused" else "stop_bytes_verified"
            )
        else:
            failure = _verify_failure_record(root, request, files, row)
            paths = {cast(str, item["path"]) for item in files}
            if "worker/result.json" in paths:
                # A failure snapshot may carry result v2 in two producer shapes, both read here
                # keyed on the verdict: DH-47 (a training-phase execution stop: execution_defect,
                # reader_refusal naming a cell, written BEFORE the stage fails) and a failure AT
                # final_snapshot (object_worker writes worker/result.json, THEN sets that stage,
                # so its failure snapshot ships an admitted complete, excluded, paused or stopped
                # result). The second is evidence the host records, as the single-profile path
                # records any failure terminal with a valid failure record.
                reference = {
                    "path": "worker/result.json",
                    "sha256": next(
                        item["sha256"] for item in files if item["path"] == "worker/result.json"
                    ),
                }
                result = _pair_result(root, request, reference, expected_pod_id=expected_pod_id)
                # ``parse_pair_result`` admits ``execution_defect`` only with an excluded
                # ``reader_refusal`` verdict naming a cell, so no second check of that pair is
                # kept here (review probe T29d, 2026-11-03: flipping the copy stayed green)
                if result["status"] != "execution_defect" and failure["stage"] != "final_snapshot":
                    # any other status was written only by a worker that reached final_snapshot
                    raise ValueError("failure result differs from an execution stop")
                report["capture_status"] = "failure_bytes_verified"
            elif not any(
                path.split("/", 1)[0] in ("run", "capture", "verification") for path in paths
            ):
                # positive evidence only: a failure terminal whose snapshot holds no cell file at
                # all and cites no result. Missing evidence never reads as "never started".
                report["capture_status"] = "pair_not_started"
            else:
                report["capture_status"] = "failure_bytes_verified"
        report["snapshot"] = asdict(event.snapshot)
    if result is not None:
        report["verdict"] = result["verdict"]
    report["cells"] = [
        _cell_entry(
            root,
            request,
            index,
            None if result is None else cast(list[dict[str, object]], result["cells"])[index],
        )
        for index in range(2)
    ]
