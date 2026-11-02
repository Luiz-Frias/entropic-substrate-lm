"""One admitted H probe lineage; no allocation and no scientific successor authority."""

import hashlib
import json
import os
import time
from collections.abc import Callable, Generator, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, cast

from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.runtime_precision import (
    require_observed_execution_precision,
)
from entropic_substrate.kernel.artifacts import artifact_from_mapping, read_d1_artifact_bytes
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.kernel.runtime.paths import source_root
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.register.h_preservation import H1ExclusionReason
from entropic_substrate.ladder.run.admission import RUN_ADMISSION, RUNTIME_KIND
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.codec_handoff import materialize_codec_handoff
from entropic_substrate.ladder.run.generation import run_probe_generation
from entropic_substrate.ladder.run.journal import artifact_reference, write_json
from entropic_substrate.ladder.run.native_verification import verify_native_capture
from entropic_substrate.ladder.run.runner import (
    CellScope,
    ReceiptBinding,
    TrainingProgress,
    read_training_progress,
    run_probe_training,
)
from entropic_substrate.ladder.run.seals import CorpusMemo, SealMemo, narrated_admission, read_seal
from entropic_substrate.ladder.run.span_diagnostics import run_span_diagnostics
from entropic_substrate.ladder.run.trace_capture import capture_model_trace
from entropic_substrate.ladder.run.worker.d1_predecessor_loader import verified_git_bundle
from entropic_substrate.ladder.run.worker.hardware import measure_hardware, verify_source_tree
from entropic_substrate.ladder.run.worker_log_events import STAGE_END, STAGE_START, render
from entropic_substrate.ladder.run.worker_request import (
    RESULT_SCHEMA_V2,
    AnyProbeRequest,
    PairProbeRequest,
)


@contextmanager
def narrated_stage(stage: str) -> Generator[None]:
    """Bracket one potentially blocking worker stage with ``stage.start``/``stage.end``.

    FOR-1637: ``.start`` is logged BEFORE the stage runs, so a stage that never
    returns is visible as a start with no end; ``.end`` (with ``seconds``) is
    logged only when the stage returns. A raising stage logs no ``.end`` and
    the exception propagates unchanged. Public because ``object_worker`` brackets
    its own stages (materialization, result admission) with the same event pair.
    """
    started = time.perf_counter()
    logger.bind(event=STAGE_START, stage=stage).info(render(STAGE_START, stage=stage))
    yield
    seconds = time.perf_counter() - started
    logger.bind(event=STAGE_END, stage=stage, seconds=seconds).info(
        render(STAGE_END, stage=stage, seconds=seconds)
    )


def _runtime(request: AnyProbeRequest, root: Path, repo_root: Path | None) -> dict[str, object]:
    """Determine execution kind for a probe request.

    The remote-worker guard is layout-invariant (E-3/DN-8): the "actual"
    side is the installed ``entropic_substrate`` package's own resolved
    location (``entropic_substrate.kernel.runtime.paths.source_root()``), not this
    module's own ``__file__`` depth, so relocating a copy of this module
    elsewhere does not by itself make a legitimately bound tree read as
    unbound.
    """
    # a pair's two cells share one run class, stage, device and precision (require_pair)
    profile = request.donor if isinstance(request, PairProbeRequest) else request.profile
    precision = require_observed_execution_precision(profile)
    if RUN_ADMISSION.worker_surface(profile.run_class) == "local_worker":
        return {
            "kind": RUNTIME_KIND[profile.run_class],
            "pod_id": None,
            "source_attestation": "not_remote",
            "accelerator": None,
            "precision": precision,
        }
    if (
        repo_root is None
        or request.code_bundle is None
        or request.code_commit is None
        or (source_root() / "entropic_substrate" / "sentence_prediction").resolve()
        != (repo_root / "src/entropic_substrate/sentence_prediction").resolve()
    ):
        raise ValueError("remote worker must execute its bound source tree")
    with verified_git_bundle(root, request.code_bundle) as git:
        verify_source_tree(repo_root, request.code_commit, git)
    model, count = measure_hardware()
    if count != 1:
        raise ValueError("H probe worker requires one visible accelerator")
    pod_id = require_d1_pod_id(os.environ.get("RUNPOD_POD_ID"))
    runtime: dict[str, object] = {
        "kind": RUNTIME_KIND[profile.run_class],
        "pod_id": pod_id,
        "source_attestation": "exact_git_tree_match",
        "code_commit": request.code_commit,
        "requested_image": request.image,
        "accelerator": {"model": model, "count": count},
        "precision": precision,
    }
    # Rung B: a scientific run echoes the sealed authority it runs under. The runtime
    # rides into worker/start.json, the ``started`` event, the ``run_start`` lifecycle
    # record and the worker result, so every one of them proves the run's mandate.
    if request.authority is not None:
        runtime["authority"] = request.authority.digest
    return runtime


def _reference(root: Path, section: str, artifact: ArtifactIdentity) -> dict[str, str]:
    ref = artifact_reference(root / section, artifact)
    return {"path": f"{section}/{ref['path']}", "sha256": ref["sha256"]}


def execute_probe_request(
    request: AnyProbeRequest,
    root: Path,
    *,
    repo_root: Path | None = None,
    on_checkpoint: Callable[[ArtifactIdentity], None] | None = None,
    on_start: Callable[[dict[str, object]], None] | None = None,
    sink: MetricSink | None = None,
    publication_timing: MutableMapping[str, float] | None = None,
    operator_stop: Callable[[str, int], bool] | None = None,
    admission_memos: MutableMapping[str, SealMemo] | None = None,
    cells: "CellObserver | None" = None,
    corpus_memos: MutableMapping[str, CorpusMemo] | None = None,
) -> dict[str, object]:
    """Run the existing numerical/custody path after native input admission.

    Completion means the requested engineering workload and readers finished.
    It does not claim useful generation, a scientific verdict, or a new launch grant.

    ``sink``, when given, is threaded to :func:`~entropic_substrate.ladder.run.runner.run_probe_training`
    unchanged -- see that function's docstring. This function neither
    constructs nor finishes it.

    ``publication_timing``, when given, is threaded to ``run_probe_training``
    unchanged -- an ADDITIVE instrumentation mirror (FOR-1567), never part of
    this function's own completion contract.

    ``operator_stop``, when given, is composed with ``request.pause`` into
    :func:`~entropic_substrate.ladder.run.runner.run_probe_training`'s single
    ``stop_after_seal`` callback (FOR-1617): it is called at every seal
    boundary the runner already reaches, unconditionally (handed that seal's
    ``(phase, step)`` so a marker can name its target), so a control-key
    read count matches the boundary count regardless of whether ``pause`` is
    also set. When both a declared pause and the operator poll agree on the
    same boundary, the existing pause label (``"paused"``) wins -- the result
    status becomes ``"stopped_by_operator"`` only when the poll alone caused
    the stop.

    ``admission_memos``, when given, receives the ``/run`` root's admission
    memo under the key ``"run"`` (FOR-1662), so a caller that admits the
    returned result afterwards (``object_worker``'s result admission) can reuse
    the walk this process already performed. It is a caller-owned out-parameter
    in the idiom of ``publication_timing``: the memo never enters the returned
    result dict or any serialized artifact, and omitting it changes nothing.

    FOR-1693: a pair request (v3) runs :func:`_execute_pair` and returns an
    ``h-probe-worker-result-v2`` record; its per-cell sink, checkpoint callback and stop poll come
    from ``cells`` (the observer ``object_worker`` owns), so ``sink``, ``on_checkpoint`` and
    ``operator_stop`` must be absent: a pair handed one plain sink is refused. ``admission_memos``
    then receives one memo per cell root under ``run/<cell>``. A single-profile request refuses
    ``cells``.

    ``corpus_memos`` (FOR-1693, DH-41), when given, is the caller-owned map of the process's
    :class:`~entropic_substrate.ladder.run.seals.CorpusMemo` objects, one per source root: key
    ``source`` for the workspace ``source/`` and ``capture`` (or ``capture/<cell>``) for the
    capture roots. It is an in/out parameter in the idiom of ``admission_memos`` (so
    ``object_worker``'s result admission reuses the same memos); omitting it keeps every
    read memo-less, exactly as before.
    """
    if isinstance(request, PairProbeRequest):
        if sink is not None or on_checkpoint is not None or operator_stop is not None:
            raise ValueError("a pair is observed per cell: hand it a cell observer, not one sink")
        return _execute_pair(
            request,
            root,
            repo_root=repo_root,
            on_start=on_start,
            cells=cells,
            publication_timing=publication_timing,
            admission_memos=admission_memos,
            corpus_memos=corpus_memos,
        )
    if cells is not None:
        raise ValueError("a single-profile request has no cells to observe")
    started = time.perf_counter()
    request.__post_init__()
    limits = request.limits
    _, payload, digest = read_d1_artifact_bytes(
        root, "request.json", label="worker request", max_bytes=limits.max_object_bytes
    )
    if digest != request.digest or payload != request.payload:
        raise ValueError("worker materialized request identity mismatch")
    with narrated_stage("runtime_admission"):
        runtime = _runtime(request, root, repo_root)
    profile, source, output = request.profile, root / "source", root / "run"
    source_corpora = (
        None if corpus_memos is None else corpus_memos.setdefault("source", CorpusMemo())
    )
    if source_corpora is None:
        read_prepared_corpus(
            source, ContentHash("sha256", profile.source_digest), max_bytes=limits.max_object_bytes
        )
    else:
        source_corpora.read(
            source, ContentHash("sha256", profile.source_digest), max_bytes=limits.max_object_bytes
        )
    if on_start is not None:
        on_start(runtime)
    resume = (
        None
        if request.resume is None
        else ArtifactIdentity(PurePosixPath(request.resume.path).name, request.resume.content_hash)
    )

    def checkpoint(artifact: ArtifactIdentity) -> None:
        if on_checkpoint is not None:
            ref = _reference(root, "run", artifact)
            on_checkpoint(ArtifactIdentity(ref["path"], ContentHash("sha256", ref["sha256"])))

    operator_stopped = False

    def stop_after_seal(phase: str, step: int) -> bool:
        nonlocal operator_stopped
        matched = request.pause is not None and (phase, step) == request.pause
        # Called unconditionally (never short-circuited by `matched`) so a
        # control-key read happens on every boundary this callback reaches,
        # independent of whether a declared pause is also set (C13(a)).
        polled = operator_stop(phase, step) if operator_stop is not None else False
        operator_stopped = polled and not matched
        return matched or polled

    trained = run_probe_training(
        profile,
        source,
        output,
        max_source_bytes=limits.max_object_bytes,
        max_artifact_bytes=limits.max_object_bytes,
        resume=resume,
        stop_after_seal=None
        if request.pause is None and operator_stop is None
        else stop_after_seal,
        on_checkpoint=checkpoint if on_checkpoint is not None else None,
        sink=sink,
        publication_timing=publication_timing,
        authority=request.authority,
        corpora=source_corpora,
    )
    # FOR-1662: ONE fresh full-chain admission walk per artifact root per process.
    # The runner's completion/pause walk populated this memo for the /run root;
    # the stages below that read /run reuse it. The capture root is a different
    # root (byte-identical copies, digest-keyed memo) and gets its own memo.
    run_memo = trained.admission_memo
    if admission_memos is not None and run_memo is not None:
        admission_memos["run"] = run_memo
    with (
        narrated_stage("training_readback"),
        narrated_admission("training_readback", "run", run_memo),
    ):
        training = read_training_progress(
            output,
            trained.receipt,
            profile,
            source,
            max_source_bytes=limits.max_object_bytes,
            max_artifact_bytes=limits.max_object_bytes,
            memo=run_memo,
            corpora=source_corpora,
        )
    result: dict[str, object] = {
        "schema": "h-probe-worker-result-v1",
        "request_sha256": request.digest,
        "run_identity": profile.identity,
        "scientific_authority": "none",
        "status": "stopped_by_operator" if operator_stopped else "paused",
        "runtime": runtime,
        "successful_optimizer_steps": training["successful_optimizer_steps"],
        "training": _reference(root, "run", trained.receipt),
        "generation": None,
        "diagnostics": None,
        "capture": None,
        "verification": None,
    }
    if trained.status == "paused":
        result["wall_seconds"] = time.perf_counter() - started
        return result
    with narrated_stage("generation"), narrated_admission("generation", "run", run_memo):
        generated = run_probe_generation(
            output,
            trained.receipt,
            profile,
            source,
            max_source_bytes=limits.max_object_bytes,
            max_artifact_bytes=limits.max_object_bytes,
            memo=run_memo,
            corpora=source_corpora,
        )
    with (
        narrated_stage("span_diagnostics"),
        narrated_admission("span_diagnostics", "run", run_memo),
    ):
        diagnostic = run_span_diagnostics(
            output,
            trained.receipt,
            profile,
            source,
            request.probe,
            max_source_bytes=limits.max_object_bytes,
            max_artifact_bytes=limits.max_object_bytes,
            memo=run_memo,
            corpora=source_corpora,
        )
    capture_root = root / "capture"
    capture_root.mkdir()
    with narrated_stage("trace_capture"), narrated_admission("trace_capture", "run", run_memo):
        captured = capture_model_trace(
            output,
            trained.receipt,
            generated.receipt,
            profile,
            source,
            capture_root,
            max_bytes=limits.max_object_bytes,
            max_source_bytes=limits.max_object_bytes,
            max_total_bytes=limits.max_total_bytes,
            max_artifacts=limits.max_objects,
            diagnostic_roots=(diagnostic,),
            memo=run_memo,
            corpora=source_corpora,
        )
    verification_root = root / "verification"
    verification_root.mkdir()
    capture_memo = SealMemo()
    with (
        narrated_stage("native_verification"),
        narrated_admission("native_verification", "capture", capture_memo),
    ):
        verified = verify_native_capture(
            capture_root,
            captured,
            verification_root,
            request_id=ContentHash("sha256", request.digest),
            max_bytes=limits.max_object_bytes,
            max_source_bytes=limits.max_object_bytes,
            max_total_bytes=limits.max_total_bytes,
            max_artifacts=limits.max_objects,
            memo=capture_memo,
            corpora=None
            if corpus_memos is None
            else corpus_memos.setdefault("capture", CorpusMemo()),
        )
    result.update(
        status="probe_workload_complete",
        generation=_reference(root, "run", generated.receipt),
        diagnostics=_reference(root, "run", diagnostic),
        capture=_reference(root, "capture", captured),
        verification=_reference(root, "verification", verified),
        wall_seconds=time.perf_counter() - started,
    )
    return result


# ------------------------------------------------------------------------------ the pair (v3)
class CellSinks(Protocol):
    """Owned by ``worker_command``: one MLflow run, metrics file and worker log PER CELL.

    ``begin`` builds the cell's sink (re-pointing the worker log once, inside the provider);
    ``finish`` terminates that cell's run exactly once. This module never builds or finishes a
    sink (FOR-1693, P5-15).
    """

    def begin(self, cell: str, profile: HProfile) -> MetricSink: ...

    def finish(self, cell: str, status: Literal["FINISHED", "FAILED"]) -> None: ...


@dataclass(frozen=True)
class CellHooks:
    """What a cell's training run is handed: its sink, its checkpoint callback, its stop poll."""

    sink: MetricSink | None
    on_checkpoint: Callable[[ArtifactIdentity], None] | None
    operator_stop: Callable[[str, int], bool] | None


class CellObserver(Protocol):
    """The per-cell side channels ``object_worker`` owns (events, lifecycle records, stop poll).

    ``focus`` names the cell the loop is entering, BEFORE anything of it can fail (so a hand-off
    failure is the recipient's); ``begin`` opens the cell (its ``run_start`` and sink) and returns
    its hooks; ``end`` closes it with its own on-disk outcome (its ``run_end`` and run status).
    """

    def focus(self, cell: str) -> None: ...

    def begin(self, cell: str, profile: HProfile, runtime: dict[str, object]) -> CellHooks: ...

    def end(
        self,
        cell: str,
        *,
        status: Literal["ok", "failed"],
        result_status: str,
        wall_seconds: float,
        steps: int | None,
    ) -> None: ...


@dataclass
class _Cell:
    token: str
    profile: HProfile
    outcome: str = "not_run"
    steps: int = 0
    training: dict[str, str] | None = None
    failure: dict[str, str] | None = None
    generation: dict[str, str] | None = None
    diagnostics: dict[str, str] | None = None
    capture: dict[str, str] | None = None
    verification: dict[str, str] | None = None
    reason: H1ExclusionReason | None = None
    last_seal: ArtifactIdentity | None = None

    def entry(self) -> dict[str, object]:
        return {
            "cell": self.token,
            "run_identity": self.profile.identity,
            "outcome": self.outcome,
            "successful_optimizer_steps": self.steps,
            "training": self.training,
            "failure": self.failure,
            "generation": self.generation,
            "diagnostics": self.diagnostics,
            "capture": self.capture,
            "verification": self.verification,
        }


def _cell_reference(
    root: Path, section: str, cell: str, artifact: ArtifactIdentity
) -> dict[str, str]:
    ref = artifact_reference(root / section / cell, artifact)
    return {"path": f"{section}/{cell}/{ref['path']}", "sha256": ref["sha256"]}


def verdict_record(
    kind: Literal["complete", "excluded", "none"],
    reason: H1ExclusionReason | None = None,
    cell: str | None = None,
) -> dict[str, object]:
    """The pair verdict: ``complete``, an ``H1ExclusionReason`` member naming a cell, or no verdict."""
    if (kind == "excluded") != (reason is not None) or (kind == "excluded") != (cell is not None):
        raise ValueError("an excluded verdict names its reason and its cell; no other kind does")
    if reason is not None and type(reason) is not H1ExclusionReason:
        raise TypeError("a recorded exclusion reason is an H1ExclusionReason member")
    return {"verdict": kind, "reason": None if reason is None else reason.value, "cell": cell}


def pair_result(
    request: PairProbeRequest,
    runtime: dict[str, object],
    cells: Sequence[_Cell],
    *,
    status: str,
    verdict: dict[str, object],
    wall_seconds: float,
) -> dict[str, object]:
    """THE builder of every ``h-probe-worker-result-v2`` (complete, paused, stopped, excluded,
    and the execution-defect result that rides a failure snapshot)."""
    return {
        "schema": RESULT_SCHEMA_V2,
        "request_sha256": request.digest,
        "run_identity": request.run_identity,
        "scientific_authority": "none",
        "status": status,
        "runtime": runtime,
        "cells": [cell.entry() for cell in cells],
        "verdict": verdict,
        "wall_seconds": wall_seconds,
    }


def _training_failure_record(output: Path) -> tuple[Path, dict[str, object]] | None:
    """The one ``h-probe-training-failure-v2`` record in a cell's run root, or ``None``.

    Located by reading every ``*.json`` of the root and keeping those whose ``schema`` IS that
    name (never a file-name glob alone); two is a defect and raises, none is ``None`` (the
    runner raised before its own record could be written).
    """
    found: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(output.glob("*.json")):
        row = json.loads(path.read_bytes())
        if type(row) is dict and row.get("schema") == "h-probe-training-failure-v2":
            found.append((path, cast(dict[str, object], row)))
    if len(found) > 1:
        raise ValueError("a cell root holds more than one training failure record")
    return found[0] if found else None


def _failure_reference(output: Path, record_path: Path, cell: str) -> dict[str, str]:
    digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
    ref = artifact_reference(
        output, ArtifactIdentity(record_path.name, ContentHash("sha256", digest))
    )
    return {"path": f"run/{cell}/{ref['path']}", "sha256": ref["sha256"]}


def _write_defect_result(
    request: PairProbeRequest,
    root: Path,
    runtime: dict[str, object],
    cells: Sequence[_Cell],
    cell: str,
    started: float,
) -> None:
    """P5-13: an execution-class stop writes result v2 (verdict READER_REFUSAL naming ``cell``)
    at ``worker/result.json`` BEFORE the original exception propagates; the failure snapshot the
    worker's failure arm publishes then carries it beside the v1 failure record."""
    (root / "worker").mkdir(exist_ok=True)
    result = pair_result(
        request,
        runtime,
        cells,
        status="execution_defect",
        verdict=verdict_record("excluded", H1ExclusionReason.READER_REFUSAL, cell),
        wall_seconds=time.perf_counter() - started,
    )
    write_json(root, "worker/result.json", result, max_bytes=request.limits.max_object_bytes)


def _execute_pair(
    request: PairProbeRequest,
    root: Path,
    *,
    repo_root: Path | None,
    on_start: Callable[[dict[str, object]], None] | None,
    cells: CellObserver | None,
    publication_timing: MutableMapping[str, float] | None,
    admission_memos: MutableMapping[str, SealMemo] | None,
    corpus_memos: MutableMapping[str, CorpusMemo] | None,
) -> dict[str, object]:
    """Donor to completion, hand-off, recipient (FOR-1693, D1/D2/D11, DH-15, DH-42, DH-45, DH-47).

    Each cell runs its WHOLE workload (training, generation, span diagnostics, trace capture,
    native verification) before the next cell has a sink, a root or a hand-off. The classifying
    ``except`` wraps ONLY ``run_probe_training`` and ``materialize_codec_handoff``, over one
    closed table:

    ===  =================================================  ===============================
    row  event                                              recorded
    ===  =================================================  ===============================
    1    scientific gap, no warm-up-final seal in the       MISSING_ENDPOINT; the pair
         failing cell's chain                               returns, no hand-off
    2    scientific gap, warm-up-final seal admitted        NONFINITE_ENDPOINT; a donor
                                                            still hands off, a recipient
                                                            ends the pair
    3    any other ``run_probe_training`` failure           READER_REFUSAL naming that cell
    4    any ``materialize_codec_handoff`` failure          READER_REFUSAL naming the
                                                            recipient
    ===  =================================================  ===============================

    Rows 3 and 4 write result v2 and re-raise the ORIGINAL exception (so the failure arm still
    classifies it as ``execution_defect``); a failure of any post-training stage propagates
    untouched, with no member recorded and no result v2 (DH-45). Classification keys on the
    runner's own ``isinstance`` split and on the warm-up-final seal the failing chain admits,
    never on phase fields, step counts or file-name globs.
    """
    started = time.perf_counter()
    request.__post_init__()
    limits = request.limits
    _, payload, digest = read_d1_artifact_bytes(
        root, "request.json", label="worker request", max_bytes=limits.max_object_bytes
    )
    if digest != request.digest or payload != request.payload:
        raise ValueError("worker materialized request identity mismatch")
    with narrated_stage("runtime_admission"):
        runtime = _runtime(request, root, repo_root)
    source = root / "source"
    # one memo for the workspace source root, shared by both cells and every stage; each cell's
    # capture root (its own copy of the source) gets a memo of its own
    source_corpora = (
        None if corpus_memos is None else corpus_memos.setdefault("source", CorpusMemo())
    )
    donor_digest = ContentHash("sha256", request.donor.source_digest)
    if source_corpora is None:
        read_prepared_corpus(source, donor_digest, max_bytes=limits.max_object_bytes)
    else:
        source_corpora.read(source, donor_digest, max_bytes=limits.max_object_bytes)
    if on_start is not None:
        on_start(runtime)
    donor, recipient = (_Cell(cell.token, cell.profile) for cell in request.cells)
    states = (donor, recipient)

    def binding_of(cell: _Cell) -> ReceiptBinding:
        scope = CellScope(request.binding.block_id, request.binding.pair_id, cell.token)
        return ReceiptBinding.of(scope, request.authority)

    def train(
        cell: _Cell, index: int, resume: ArtifactIdentity | None, hooks: CellHooks
    ) -> tuple[TrainingProgress, bool]:
        """Run the cell's training; the caller owns the classifying ``except``."""
        pause = request.cells[index].pause
        operator_stopped = False

        def stop_after_seal(phase: str, step: int) -> bool:
            nonlocal operator_stopped
            matched = pause is not None and (phase, step) == pause
            polled = hooks.operator_stop(phase, step) if hooks.operator_stop is not None else False
            operator_stopped = polled and not matched
            return matched or polled

        def checkpoint(artifact: ArtifactIdentity) -> None:
            # the runner hands over a path relative to the cell's run root; the observer's
            # callback reads and snapshots from the WORKSPACE root, so it gets run/<cell>/<name>
            if hooks.on_checkpoint is not None:
                ref = _cell_reference(root, "run", cell.token, artifact)
                hooks.on_checkpoint(
                    ArtifactIdentity(ref["path"], ContentHash("sha256", ref["sha256"]))
                )

        trained = run_probe_training(
            cell.profile,
            source,
            root / "run" / cell.token,
            max_source_bytes=limits.max_object_bytes,
            max_artifact_bytes=limits.max_object_bytes,
            resume=resume,
            stop_after_seal=None
            if pause is None and hooks.operator_stop is None
            else stop_after_seal,
            on_checkpoint=checkpoint if hooks.on_checkpoint is not None else None,
            sink=hooks.sink,
            publication_timing=publication_timing,
            authority=request.authority,
            scope=CellScope(request.binding.block_id, request.binding.pair_id, cell.token),
            corpora=source_corpora,
        )
        return trained, operator_stopped

    def close(
        cell: _Cell, *, ok: bool, result_status: str, began: float, steps: int | None
    ) -> None:
        if cells is not None:
            cells.end(
                cell.token,
                status="ok" if ok else "failed",
                result_status=result_status,
                wall_seconds=time.perf_counter() - began,
                steps=steps,
            )

    def classify(error: Exception, cell: _Cell, began: float) -> bool:
        """Rows 1-3 for a training failure. ``True``: a scientific stop recorded (the pair goes
        on to its verdict); ``False``: an execution-class stop (result v2 written; the caller
        re-raises the original exception)."""
        output = root / "run" / cell.token
        found = _training_failure_record(output)
        scientific = isinstance(error, UndefinedDirection)
        if found is not None:
            record_path, record = found
            if record["status"] != ("scientific_gap" if scientific else "execution_defect"):
                raise ValueError("the on-disk training failure disagrees with the raised error")
            cell.failure = _failure_reference(output, record_path, cell.token)
            # an execution-class stop re-reads nothing: the evidence may be what broke, and a
            # second failure here would replace the original exception
            if scientific and record["last_receipt"] is not None:
                receipt = artifact_from_mapping(record["last_receipt"])
                cell.training = _cell_reference(root, "run", cell.token, receipt)
                cell.steps = cast(
                    int,
                    read_training_progress(
                        output,
                        receipt,
                        cell.profile,
                        source,
                        max_source_bytes=limits.max_object_bytes,
                        max_artifact_bytes=limits.max_object_bytes,
                        expected=binding_of(cell),
                        corpora=source_corpora,
                    )["successful_optimizer_steps"],
                )
            if scientific and record["last_seal"] is not None:
                cell.last_seal = artifact_from_mapping(record["last_seal"])
        if scientific:
            if found is None:
                raise ValueError("a scientific stop left no training failure record")
            warmed = (
                None
                if cell.last_seal is None
                else read_seal(
                    output,
                    cell.last_seal,
                    cell.profile,
                    source,
                    max_bytes=limits.max_object_bytes,
                    max_source_bytes=limits.max_object_bytes,
                    corpora=source_corpora,
                ).warmup_final
            )
            cell.outcome = "scientific_gap"
            cell.reason = (
                H1ExclusionReason.NONFINITE_ENDPOINT
                if warmed is not None
                else H1ExclusionReason.MISSING_ENDPOINT
            )
            close(cell, ok=False, result_status="scientific_gap", began=began, steps=None)
            return True
        cell.outcome = "execution_defect"
        cell.reason = H1ExclusionReason.READER_REFUSAL
        close(cell, ok=False, result_status="execution_defect", began=began, steps=None)
        _write_defect_result(request, root, runtime, states, cell.token, started)
        return False

    def workload(
        cell: _Cell, trained: TrainingProgress, stopped_by_operator: bool, began: float
    ) -> None:
        """Readback, then (complete training only) the four later stages; may raise unwrapped."""
        output = root / "run" / cell.token
        memo = trained.admission_memo
        if admission_memos is not None and memo is not None:
            admission_memos[f"run/{cell.token}"] = memo
        with (
            narrated_stage("training_readback"),
            narrated_admission("training_readback", "run", memo),
        ):
            training = read_training_progress(
                output,
                trained.receipt,
                cell.profile,
                source,
                max_source_bytes=limits.max_object_bytes,
                max_artifact_bytes=limits.max_object_bytes,
                memo=memo,
                expected=binding_of(cell),
                corpora=source_corpora,
            )
        cell.steps = cast(int, training["successful_optimizer_steps"])
        cell.training = _cell_reference(root, "run", cell.token, trained.receipt)
        cell.last_seal = trained.last_seal
        if trained.status == "paused":
            cell.outcome = "stopped_by_operator" if stopped_by_operator else "paused"
            close(cell, ok=True, result_status=cell.outcome, began=began, steps=cell.steps)
            return
        with narrated_stage("generation"), narrated_admission("generation", "run", memo):
            generated = run_probe_generation(
                output,
                trained.receipt,
                cell.profile,
                source,
                max_source_bytes=limits.max_object_bytes,
                max_artifact_bytes=limits.max_object_bytes,
                memo=memo,
                corpora=source_corpora,
            )
        with (
            narrated_stage("span_diagnostics"),
            narrated_admission("span_diagnostics", "run", memo),
        ):
            diagnostic = run_span_diagnostics(
                output,
                trained.receipt,
                cell.profile,
                source,
                request.probe,
                max_source_bytes=limits.max_object_bytes,
                max_artifact_bytes=limits.max_object_bytes,
                memo=memo,
                corpora=source_corpora,
            )
        capture_root = root / "capture" / cell.token
        capture_root.mkdir(parents=True)
        with narrated_stage("trace_capture"), narrated_admission("trace_capture", "run", memo):
            captured = capture_model_trace(
                output,
                trained.receipt,
                generated.receipt,
                cell.profile,
                source,
                capture_root,
                max_bytes=limits.max_object_bytes,
                max_source_bytes=limits.max_object_bytes,
                max_total_bytes=limits.max_total_bytes,
                max_artifacts=limits.max_objects,
                diagnostic_roots=(diagnostic,),
                memo=memo,
                corpora=source_corpora,
            )
        verification_root = root / "verification" / cell.token
        verification_root.mkdir(parents=True)
        capture_memo = SealMemo()
        with (
            narrated_stage("native_verification"),
            narrated_admission("native_verification", "capture", capture_memo),
        ):
            verified = verify_native_capture(
                capture_root,
                captured,
                verification_root,
                request_id=ContentHash("sha256", request.digest),
                max_bytes=limits.max_object_bytes,
                max_source_bytes=limits.max_object_bytes,
                max_total_bytes=limits.max_total_bytes,
                max_artifacts=limits.max_objects,
                memo=capture_memo,
                corpora=None
                if corpus_memos is None
                else corpus_memos.setdefault(f"capture/{cell.token}", CorpusMemo()),
            )
        cell.outcome = "complete"
        cell.generation = _cell_reference(root, "run", cell.token, generated.receipt)
        cell.diagnostics = _cell_reference(root, "run", cell.token, diagnostic)
        cell.capture = _cell_reference(root, "capture", cell.token, captured)
        cell.verification = _cell_reference(root, "verification", cell.token, verified)
        close(cell, ok=True, result_status="probe_workload_complete", began=began, steps=cell.steps)

    def run_cell(cell: _Cell, index: int, resume: ArtifactIdentity | None) -> None:
        """One cell, from its (already focused) entry to its workload or its recorded stop."""
        began = time.perf_counter()
        # opening the cell (its run_start and sink) is NOT part of the classified training
        hooks = (
            CellHooks(None, None, None)
            if cells is None
            else cells.begin(cell.token, cell.profile, runtime)
        )
        try:
            trained, operator_stopped = train(cell, index, resume, hooks)
        except Exception as error:
            if not classify(error, cell, began):
                raise
            return
        workload(cell, trained, operator_stopped, began)

    # ---------------------------------------------------------------------- the donor
    # a cell's resume names ``run/<cell>/<receipt>`` in the REQUEST (its inputs' own paths); the
    # runner reads it relative to that cell's run root, exactly as a hand-off receipt is named
    donor_resume, recipient_resume = (
        None
        if cell.resume is None
        else ArtifactIdentity(PurePosixPath(cell.resume.path).name, cell.resume.content_hash)
        for cell in request.cells
    )
    if cells is not None:
        cells.focus(donor.token)
    # a resuming donor's root exists already (materialized from its inputs); a fresh one must not
    (root / "run" / donor.token).mkdir(parents=True, exist_ok=donor_resume is not None)
    run_cell(donor, 0, donor_resume)

    # ----------------------------------------------- the hand-off, then the recipient
    handed_off = donor.outcome == "complete" or (
        donor.outcome == "scientific_gap" and donor.reason is H1ExclusionReason.NONFINITE_ENDPOINT
    )
    if handed_off and recipient_resume is not None:
        # C-L4c-2 case 3: the recipient resumes from its OWN receipt (never the hand-off receipt:
        # R3); its root came with the inputs, the donor's root too, and no hand-off is made again
        if cells is not None:
            cells.focus(recipient.token)
        run_cell(recipient, 1, recipient_resume)
    elif handed_off:
        if cells is not None:
            cells.focus(recipient.token)
        donor_root = root / "run" / donor.token
        recipient_root = root / "run" / recipient.token
        recipient_root.mkdir(parents=True)
        try:
            warmed = (
                None
                if donor.last_seal is None
                else read_seal(
                    donor_root,
                    donor.last_seal,
                    donor.profile,
                    source,
                    max_bytes=limits.max_object_bytes,
                    max_source_bytes=limits.max_object_bytes,
                    corpora=source_corpora,
                ).warmup_final
            )
            if warmed is None:
                raise ValueError("the donor's chain holds no warm-up-final seal to hand off")
            handoff = materialize_codec_handoff(
                donor_root,
                warmed,
                donor.profile,
                recipient_root,
                recipient.profile,
                source,
                max_source_bytes=limits.max_object_bytes,
                max_artifact_bytes=limits.max_object_bytes,
                binding=binding_of(recipient),
            )
        except Exception as handoff_error:
            recipient.outcome = "execution_defect"
            recipient.reason = H1ExclusionReason.READER_REFUSAL
            if any(recipient_root.glob("*-progress.json")):
                raise ValueError(
                    "a failed hand-off left a progress receipt behind"
                ) from handoff_error
            _write_defect_result(request, root, runtime, states, recipient.token, started)
            raise
        run_cell(recipient, 1, handoff.receipt)

    # --------------------------------------------------------------------- the verdict
    excluded = next((cell for cell in states if cell.outcome == "scientific_gap"), None)
    verdict = verdict_record("none")
    status = "paused"
    if excluded is not None:
        if excluded.reason is None:
            raise ValueError(
                f"cell {excluded.token!r} is excluded without a recorded H1ExclusionReason member"
            )
        status = "pair_excluded"
        verdict = verdict_record("excluded", excluded.reason, excluded.token)
    elif all(cell.outcome == "complete" for cell in states):
        status = "probe_workload_complete"
        verdict = verdict_record("complete")
    elif any(cell.outcome == "stopped_by_operator" for cell in states):
        status = "stopped_by_operator"
    return pair_result(
        request,
        runtime,
        states,
        status=status,
        verdict=verdict,
        wall_seconds=time.perf_counter() - started,
    )
