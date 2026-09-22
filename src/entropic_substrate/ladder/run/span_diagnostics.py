"""Join report-only span probes to a real sealed training lineage."""

import math
import time
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import torch

from entropic_substrate.__legacy.analysis.predictive_probe import ProbeConfig
from entropic_substrate.__legacy.evidence.ladder_campaign import artifact_from_mapping
from entropic_substrate.__legacy.evidence.resource_meter_models import ResourcePhase
from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.corpus import CorpusSplit, PreparedCorpus
from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.lifecycle import ArtifactIdentity, ContentHash
from entropic_substrate.ladder.capture.capture import read_prepared_corpus
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.runner import read_training_progress
from entropic_substrate.ladder.run.seals import SealedStep, read_seal, resume_seal
from entropic_substrate.ladder.run.span_probe_artifacts import (
    SpanProbeBundle,
    capture_span_probes,
    read_span_probes,
    reconcile_span_evidence,
)
from entropic_substrate.ladder.run.span_states import (
    SpanPopulationGap,
    SpanStateSet,
    collect_span_states,
)


def _trained(
    root: Path,
    training: ArtifactIdentity,
    profile: HProfile,
    source: Path,
    source_limit: int,
    artifact_limit: int,
) -> tuple[dict[str, object], SealedStep]:
    row = read_training_progress(
        root,
        training,
        profile,
        source,
        max_source_bytes=source_limit,
        max_artifact_bytes=artifact_limit,
    )
    if row["status"] != "training_complete":
        raise ValueError("span diagnostics require complete sealed training")
    seal = read_seal(
        root,
        artifact_from_mapping(row["last_seal"]),
        profile,
        source,
        max_bytes=artifact_limit,
        max_source_bytes=source_limit,
    )
    return row, seal


def _source_positions(data: SpanStateSet, corpus: PreparedCorpus, profile: HProfile) -> int:
    documents = {r.segmentation.document.identity: r for r in corpus.records}
    windows: dict[tuple[str, int], int] = {}
    for index, position in enumerate(data.positions):
        record = documents.get(position.document_id)
        if record is None or record.split != data.split:
            raise ValueError("diagnostic source document or split mismatch")
        spans = record.segmentation.spans
        if position.event_index >= len(spans):
            raise ValueError("diagnostic event is outside its source document")
        span = spans[position.event_index]
        start = (
            position.event_index
            // profile.spine.context_sentences
            * profile.spine.context_sentences
        )
        if (
            position.span_start != span.start
            or position.span_end != span.end
            or position.window_start != spans[start].start
            or position.target_position >= len(span.document.source)
            or int(data.labels[index]) != span.document.source[position.target_position]
        ):
            raise ValueError("diagnostic source coordinates or canonical target mismatch")
        key = (position.document_id, position.window_start)
        windows[key] = max(windows.get(key, 0), position.span_end)
    return sum(end - start for (_, start), end in windows.items())


def _summaries(bundle: SpanProbeBundle) -> list[dict[str, object]]:
    return [
        {
            "control": probe.manifest.control,
            "probe_id": probe.manifest.probe_id,
            "gain": asdict(report.summary),
            "uniform": asdict(report.uniform_summary),
            "before_accuracy": report.before_accuracy,
            "after_accuracy": report.after_accuracy,
            "fit_seconds": probe.manifest.fit_seconds,
            "score_seconds": report.score_seconds,
        }
        for probe, report in zip(bundle.probes, bundle.reports, strict=True)
    ]


def _duration(value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("diagnostic duration must be finite and nonnegative")


def read_span_diagnostics(
    root: Path,
    artifact: ArtifactIdentity,
    training: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
) -> dict[str, object]:
    """Verify checkpoint/source bindings and replay frozen scores, not model forward states.

    The selected model states are content-bound producer evidence. This reader
    does not claim a second forward execution or grant scientific eligibility.
    """
    _, seal = _trained(root, training, profile, source_root, max_source_bytes, max_artifact_bytes)
    row = recovery_record(
        read_json(root, artifact, max_bytes=max_artifact_bytes),
        "schema run_identity run_class scientific_authority status validation_scope training checkpoint probe_bundle collection summaries",
    )
    if (
        row["schema"] != "h-span-diagnostics-v1"
        or row["run_identity"] != profile.identity
        or row["run_class"] != profile.run_class
        or row["scientific_authority"] != "none"
        or row["status"] != "diagnostics_complete"
        or row["validation_scope"] != "checkpoint-source-binding-and-frozen-score-replay"
        or row["training"] != artifact_reference(root, training)
        or row["checkpoint"] != artifact_reference(root, seal.checkpoint)
    ):
        raise ValueError("span diagnostic lineage binding mismatch")
    bundle = read_span_probes(
        root,
        artifact_from_mapping(row["probe_bundle"]),
        expected_checkpoint=seal.checkpoint.content_hash.digest,
        expected_codec=SentenceByteCodec(profile.permutation).identity,
        expected_source=profile.source_digest,
        max_bytes=max_artifact_bytes,
    )
    if bundle.train.split != "train" or bundle.heldout.split != "validation":
        raise ValueError("diagnostic producer split protocol mismatch")
    corpus = read_prepared_corpus(
        source_root, ContentHash("sha256", profile.source_digest), max_bytes=max_source_bytes
    )
    collection = recovery_record(row["collection"], "train validation")
    for states in (bundle.train, bundle.heldout):
        metric = recovery_record(
            collection[states.split], "wall_seconds cuda_seconds encoder_source_bytes"
        )
        _duration(metric["wall_seconds"])
        if profile.device == "cuda":
            _duration(metric["cuda_seconds"])
        elif metric["cuda_seconds"] is not None:
            raise ValueError("CPU diagnostic cannot claim CUDA duration")
        count = metric["encoder_source_bytes"]
        if type(count) is not int or count != _source_positions(states, corpus, profile):
            raise ValueError("diagnostic exposure differs from captured source windows")
    reconcile_span_evidence(_summaries(bundle), row["summaries"])
    return row


def run_span_diagnostics(
    root: Path,
    training: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    config: ProbeConfig,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
) -> ArtifactIdentity:
    """Restore the actual trained checkpoint before collecting any fitting states."""
    config.__post_init__()
    row, seal = _trained(root, training, profile, source_root, max_source_bytes, max_artifact_bytes)
    prefix = uuid4().hex
    stage = "restore"
    bundle_artifact = None
    try:
        with torch.random.fork_rng(devices=[0] if profile.device == "cuda" else []):
            execution = SentenceExecution(profile, source_root, max_source_bytes=max_source_bytes)
            resume_seal(
                root,
                artifact_from_mapping(row["last_seal"]),
                execution,
                max_bytes=max_artifact_bytes,
            )
            collection: dict[str, dict[str, object]] = {}
            states = []
            roster: tuple[tuple[CorpusSplit, int], ...] = (
                ("train", config.train_positions),
                ("validation", config.eval_positions),
            )
            for split, count in roster:
                stage = "collection"
                started = time.perf_counter()
                ticket = (
                    None if execution.timer is None else execution.timer.start(ResourcePhase.EVAL)
                )
                try:
                    selected = collect_span_states(
                        execution,
                        seal.checkpoint.content_hash.digest,
                        split=split,
                        count=count,
                        lags=config.lags,
                        seed=config.seed,
                    )
                finally:
                    cuda_seconds = (
                        None
                        if execution.timer is None
                        else execution.timer.stop(ticket, ResourcePhase.EVAL)
                    )
                collection[split] = {
                    "wall_seconds": time.perf_counter() - started,
                    "cuda_seconds": cuda_seconds,
                    "encoder_source_bytes": _source_positions(selected, execution.corpus, profile),
                }
                states.append(selected)
            stage = "fit_and_score"
            bundle_artifact = capture_span_probes(
                root,
                f"{prefix}-span-probes.pt",
                states[0],
                states[1],
                config,
                max_bytes=max_artifact_bytes,
            )
        stage = "verification"
        bundle = read_span_probes(
            root,
            bundle_artifact,
            expected_checkpoint=seal.checkpoint.content_hash.digest,
            expected_codec=execution.codec.identity,
            expected_source=profile.source_digest,
            max_bytes=max_artifact_bytes,
        )
        receipt = write_json(
            root,
            f"{prefix}-span-diagnostics.json",
            {
                "schema": "h-span-diagnostics-v1",
                "run_identity": profile.identity,
                "run_class": profile.run_class,
                "scientific_authority": "none",
                "status": "diagnostics_complete",
                "validation_scope": "checkpoint-source-binding-and-frozen-score-replay",
                "training": artifact_reference(root, training),
                "checkpoint": artifact_reference(root, seal.checkpoint),
                "probe_bundle": artifact_reference(root, bundle_artifact),
                "collection": collection,
                "summaries": _summaries(bundle),
            },
            max_bytes=max_artifact_bytes,
        )
        read_span_diagnostics(
            root,
            receipt,
            training,
            profile,
            source_root,
            max_source_bytes=max_source_bytes,
            max_artifact_bytes=max_artifact_bytes,
        )
        return receipt
    except Exception as error:
        failure = {
            "schema": "h-span-diagnostics-failure-v1",
            "run_identity": profile.identity,
            "scientific_authority": "none",
            "training": artifact_reference(root, training),
            "stage": stage,
            "status": "scientific_gap"
            if isinstance(error, UndefinedDirection)
            else "diagnostic_gap"
            if isinstance(error, SpanPopulationGap)
            else "diagnostic_defect",
            "error_code": getattr(error, "code", type(error).__name__),
            "error_type": type(error).__name__,
            "detail": str(error),
            "probe_bundle": None
            if bundle_artifact is None
            else artifact_reference(root, bundle_artifact),
        }
        if isinstance(error, SpanPopulationGap):
            failure["population"] = {"requested": error.requested, "available": error.available}
        elif isinstance(error, UndefinedDirection):
            failure["affected_positions"] = error.positions
        try:
            write_json(root, f"{prefix}-span-failure.json", failure, max_bytes=max_artifact_bytes)
        except Exception as recording_error:
            raise ExceptionGroup(
                "diagnostic and failure-recording defects", [error, recording_error]
            ) from error
        raise
