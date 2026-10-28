"""One admitted H probe lineage; no allocation and no scientific successor authority."""

import os
import time
from collections.abc import Callable, Generator, MutableMapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from entropic_substrate.core.sentence.runtime_precision import (
    require_observed_execution_precision,
)
from entropic_substrate.kernel.artifacts import read_d1_artifact_bytes
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.kernel.runtime.paths import source_root
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.run.admission import RUN_ADMISSION, RUNTIME_KIND
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.generation import run_probe_generation
from entropic_substrate.ladder.run.journal import artifact_reference
from entropic_substrate.ladder.run.native_verification import verify_native_capture
from entropic_substrate.ladder.run.runner import read_training_progress, run_probe_training
from entropic_substrate.ladder.run.seals import SealMemo, narrated_admission
from entropic_substrate.ladder.run.span_diagnostics import run_span_diagnostics
from entropic_substrate.ladder.run.trace_capture import capture_model_trace
from entropic_substrate.ladder.run.worker.d1_predecessor_loader import verified_git_bundle
from entropic_substrate.ladder.run.worker.hardware import measure_hardware, verify_source_tree
from entropic_substrate.ladder.run.worker_log_events import STAGE_END, STAGE_START, render
from entropic_substrate.ladder.run.worker_request import ProbeRequest


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


def _runtime(request: ProbeRequest, root: Path, repo_root: Path | None) -> dict[str, object]:
    """Determine execution kind for a probe request.

    The remote-worker guard is layout-invariant (E-3/DN-8): the "actual"
    side is the installed ``entropic_substrate`` package's own resolved
    location (``entropic_substrate.kernel.runtime.paths.source_root()``), not this
    module's own ``__file__`` depth, so relocating a copy of this module
    elsewhere does not by itself make a legitimately bound tree read as
    unbound.
    """
    precision = require_observed_execution_precision(request.profile)
    if RUN_ADMISSION.worker_surface(request.profile.run_class) == "local_worker":
        return {
            "kind": RUNTIME_KIND["local_worker"],
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
    return {
        "kind": RUNTIME_KIND["remote_worker"],
        "pod_id": pod_id,
        "source_attestation": "exact_git_tree_match",
        "code_commit": request.code_commit,
        "requested_image": request.image,
        "accelerator": {"model": model, "count": count},
        "precision": precision,
    }


def _reference(root: Path, section: str, artifact: ArtifactIdentity) -> dict[str, str]:
    ref = artifact_reference(root / section, artifact)
    return {"path": f"{section}/{ref['path']}", "sha256": ref["sha256"]}


def execute_probe_request(
    request: ProbeRequest,
    root: Path,
    *,
    repo_root: Path | None = None,
    on_checkpoint: Callable[[ArtifactIdentity], None] | None = None,
    on_start: Callable[[dict[str, object]], None] | None = None,
    sink: MetricSink | None = None,
    publication_timing: MutableMapping[str, float] | None = None,
    operator_stop: Callable[[str, int], bool] | None = None,
    admission_memos: MutableMapping[str, SealMemo] | None = None,
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
    profile, source, output = request.profile, root / "source", root / "run"
    read_prepared_corpus(
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
