"""One admitted H probe lineage; no allocation and no scientific successor authority."""

import os
import time
from collections.abc import Callable, MutableMapping
from pathlib import Path, PurePosixPath

from entropic_substrate.core.sentence.runtime_precision import require_execution_precision
from entropic_substrate.kernel.artifacts import read_d1_artifact_bytes
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.kernel.runtime.paths import source_root
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.run.campaign.d1_operator_values import require_d1_pod_id
from entropic_substrate.ladder.run.generation import run_probe_generation
from entropic_substrate.ladder.run.journal import artifact_reference
from entropic_substrate.ladder.run.native_verification import verify_native_capture
from entropic_substrate.ladder.run.runner import read_training_progress, run_probe_training
from entropic_substrate.ladder.run.span_diagnostics import run_span_diagnostics
from entropic_substrate.ladder.run.trace_capture import capture_model_trace
from entropic_substrate.ladder.run.worker.d1_predecessor_loader import verified_git_bundle
from entropic_substrate.ladder.run.worker.hardware import measure_hardware, verify_source_tree
from entropic_substrate.ladder.run.worker_request import ProbeRequest


def _runtime(request: ProbeRequest, root: Path, repo_root: Path | None) -> dict[str, object]:
    """Determine execution kind for a probe request.

    The remote-worker guard is layout-invariant (E-3/DN-8): the "actual"
    side is the installed ``entropic_substrate`` package's own resolved
    location (``entropic_substrate.kernel.runtime.paths.source_root()``), not this
    module's own ``__file__`` depth, so relocating a copy of this module
    elsewhere does not by itself make a legitimately bound tree read as
    unbound.
    """
    precision = require_execution_precision()
    if request.profile.run_class == "local_tiny":
        return {
            "kind": "local_tiny",
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
        "kind": "remote_tiny",
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
    """
    started = time.perf_counter()
    request.__post_init__()
    limits = request.limits
    _, payload, digest = read_d1_artifact_bytes(
        root, "request.json", label="worker request", max_bytes=limits.max_object_bytes
    )
    if digest != request.digest or payload != request.payload:
        raise ValueError("worker materialized request identity mismatch")
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

    trained = run_probe_training(
        profile,
        source,
        output,
        max_source_bytes=limits.max_object_bytes,
        max_artifact_bytes=limits.max_object_bytes,
        resume=resume,
        stop_after_seal=None
        if request.pause is None
        else lambda phase, step: (phase, step) == request.pause,
        on_checkpoint=checkpoint if on_checkpoint is not None else None,
        sink=sink,
        publication_timing=publication_timing,
    )
    training = read_training_progress(
        output,
        trained.receipt,
        profile,
        source,
        max_source_bytes=limits.max_object_bytes,
        max_artifact_bytes=limits.max_object_bytes,
    )
    result: dict[str, object] = {
        "schema": "h-probe-worker-result-v1",
        "request_sha256": request.digest,
        "run_identity": profile.identity,
        "scientific_authority": "none",
        "status": "paused",
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
    generated = run_probe_generation(
        output,
        trained.receipt,
        profile,
        source,
        max_source_bytes=limits.max_object_bytes,
        max_artifact_bytes=limits.max_object_bytes,
    )
    diagnostic = run_span_diagnostics(
        output,
        trained.receipt,
        profile,
        source,
        request.probe,
        max_source_bytes=limits.max_object_bytes,
        max_artifact_bytes=limits.max_object_bytes,
    )
    capture_root = root / "capture"
    capture_root.mkdir()
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
    )
    verification_root = root / "verification"
    verification_root.mkdir()
    verified = verify_native_capture(
        capture_root,
        captured,
        verification_root,
        request_id=ContentHash("sha256", request.digest),
        max_bytes=limits.max_object_bytes,
        max_source_bytes=limits.max_object_bytes,
        max_total_bytes=limits.max_total_bytes,
        max_artifacts=limits.max_objects,
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
