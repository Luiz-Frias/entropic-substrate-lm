"""Generation snapshots reference a sealed trained model and retain exact feedback."""

import json
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import uuid4
from zipfile import ZipFile

import torch

from entropic_substrate.__legacy.evidence.ladder_campaign import artifact_from_mapping
from entropic_substrate.__legacy.evidence.resource_meter_models import ResourcePhase
from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import (
    capture_rng,
    recovery_record,
    restore_rng,
    rollout_from_payload,
    rollout_to_payload,
)
from entropic_substrate.core.sentence.rollout import (
    SentenceRollout,
    advance_rollout,
    begin_rollout,
)
from entropic_substrate.kernel.artifacts import (
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
)
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.runner import read_training_progress
from entropic_substrate.ladder.run.seals import resume_seal


@dataclass(frozen=True)
class GenerationProgress:
    status: Literal["paused", "generation_complete"]
    receipt: ArtifactIdentity


def _trained(
    root: Path,
    training: ArtifactIdentity,
    profile: HProfile,
    source: Path,
    source_limit: int,
    artifact_limit: int,
) -> tuple[SentenceExecution, dict[str, object]]:
    row = read_training_progress(
        root,
        training,
        profile,
        source,
        max_source_bytes=source_limit,
        max_artifact_bytes=artifact_limit,
    )
    if row["status"] != "training_complete":
        raise ValueError("generation requires complete training")
    with torch.random.fork_rng(devices=[0] if profile.device == "cuda" else []):
        execution = SentenceExecution(profile, source, max_source_bytes=source_limit)
        resume_seal(
            root, artifact_from_mapping(row["last_seal"]), execution, max_bytes=artifact_limit
        )
        rng = capture_rng()
    execution.model.eval()
    return execution, rng


def _prompt(execution: SentenceExecution) -> tuple[tuple[bytes, ...], dict[str, object]]:
    record = next(r for r in execution.corpus.records if r.split == "validation")
    span = record.segmentation.spans[0]
    return (span.source,), {
        "document": span.document.identity,
        "span": [span.start, span.end, span.index],
    }


def _summary(state: SentenceRollout, execution: SentenceExecution) -> dict[str, object]:
    outcome = (
        state.status
        if state.status != "active"
        else "sentence_budget"
        if len(state.emissions) >= execution.profile.schedule.generation_sentences
        else "paused"
    )
    return {
        "outcome": outcome,
        "sentences": [e.source.decode("utf-8") for e in state.emissions],
        "observations": list(state.observations),
        "pending": [execution.codec.canonical_byte(b) for b in state.pending],
        "log_scores": list(state.log_scores),
        "prompt": [source.decode("utf-8") for source in _prompt(execution)[0]],
    }


def _snapshot(root: Path, artifact: ArtifactIdentity, limit: int) -> dict[str, object]:
    ref = artifact_reference(root, artifact)
    _, data, digest = read_d1_artifact_bytes(
        root, ref["path"], label="generation snapshot", max_bytes=limit
    )
    if digest != ref["sha256"]:
        raise ValueError("generation snapshot hash mismatch")
    with ZipFile(BytesIO(data)) as archive:
        if sum(i.file_size for i in archive.infolist()) > limit:
            raise ValueError("expanded generation snapshot exceeds byte limit")
    return recovery_record(
        torch.load(BytesIO(data), map_location="cpu", weights_only=True),
        "schema run_identity training prompt rollout rng",
    )


def _load(
    root: Path,
    artifact: ArtifactIdentity,
    training: ArtifactIdentity,
    execution: SentenceExecution,
    limit: int,
) -> tuple[dict[str, object], SentenceRollout, object]:
    raw = read_json(root, artifact, max_bytes=limit)
    if raw.get("schema") == "h-generation-receipt-v1":
        raise ValueError("generation v1 requires its original reader; no implicit migration")
    receipt = recovery_record(
        raw,
        "schema run_identity scientific_authority training snapshot status summary wall_seconds cuda_seconds parent_receipt",
    )
    reference = artifact_reference(root, training)
    if (
        receipt["schema"] != "h-generation-receipt-v2"
        or receipt["run_identity"] != execution.profile.identity
        or receipt["training"] != reference
        or receipt["scientific_authority"] != "none"
    ):
        raise ValueError("generation receipt belongs to another trained lineage")
    saved = _snapshot(root, artifact_from_mapping(receipt["snapshot"]), limit)
    if (
        saved["schema"] != "h-generation-state-v1"
        or saved["run_identity"] != execution.profile.identity
        or saved["training"] != reference
        or saved["prompt"] != _prompt(execution)[1]
    ):
        raise ValueError("generation snapshot binding differs from trained lineage")
    state = rollout_from_payload(saved["rollout"], execution.model, execution.codec)
    if state is None:
        raise ValueError("generation snapshot lacks feedback state")
    prompt = _prompt(execution)[0]
    if state.history[: len(prompt)] != prompt:
        raise ValueError("generation history does not contain the bound prompt")
    with torch.random.fork_rng(devices=[0] if execution.device.type == "cuda" else []):
        restore_rng(saved["rng"])
    if json.dumps(receipt["summary"], sort_keys=True) != json.dumps(
        _summary(state, execution), sort_keys=True
    ):
        raise ValueError("generation summary differs from observed feedback")
    expected_status = (
        "paused" if _summary(state, execution)["outcome"] == "paused" else "generation_complete"
    )
    if receipt["status"] != expected_status:
        raise ValueError("generation status differs from its observed outcome")
    if receipt["parent_receipt"] is not None:
        parent = read_json(root, artifact_from_mapping(receipt["parent_receipt"]), max_bytes=limit)
        if (
            parent.get("schema") != "h-generation-receipt-v2"
            or parent.get("run_identity") != execution.profile.identity
            or parent.get("training") != reference
            or parent.get("status") != "paused"
        ):
            raise ValueError("generation parent does not belong to this resumable lineage")
        summary = recovery_record(
            parent.get("summary"), "outcome sentences observations pending log_scores prompt"
        )
        observations = summary["observations"]
        if (
            type(observations) is not list
            or not len(observations) < len(state.observations)
            or tuple(observations) != state.observations[: len(observations)]
        ):
            raise ValueError("generation resume does not extend the previous observation prefix")
    return receipt, state, saved["rng"]


def read_generation(
    root: Path,
    artifact: ArtifactIdentity,
    training: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
) -> dict[str, object]:
    execution, _ = _trained(
        root, training, profile, source_root, max_source_bytes, max_artifact_bytes
    )
    return _load(root, artifact, training, execution, max_artifact_bytes)[0]


def _write_snapshot(
    root: Path,
    name: str,
    training: ArtifactIdentity,
    execution: SentenceExecution,
    state: SentenceRollout,
    rng: object,
    limit: int,
) -> ArtifactIdentity:
    stream = BytesIO()
    torch.save(
        {
            "schema": "h-generation-state-v1",
            "run_identity": execution.profile.identity,
            "training": artifact_reference(root, training),
            "prompt": _prompt(execution)[1],
            "rollout": rollout_to_payload(state, execution.model, execution.codec),
            "rng": rng,
        },
        stream,
    )
    data = stream.getvalue()
    if len(data) > limit:
        raise ValueError("generation snapshot exceeds byte limit")
    return publish_d1_atomic_bytes(root, name, data, label="generation state")


def run_probe_generation(
    root: Path,
    training: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
    resume: ArtifactIdentity | None = None,
    pause_after: int | None = None,
) -> GenerationProgress:
    """Generate to an outcome/budget; controlled pauses persist even partial sentences."""
    if pause_after is not None and (type(pause_after) is not int or pause_after < 1):
        raise ValueError("generation pause must be a positive observation count")
    execution, rng = _trained(
        root, training, profile, source_root, max_source_bytes, max_artifact_bytes
    )
    if resume is None:
        with torch.autocast(
            execution.device.type, dtype=torch.bfloat16, enabled=profile.precision == "bf16"
        ):
            state = begin_rollout(execution.model, _prompt(execution)[0], execution.codec)
    else:
        receipt, state, rng = _load(root, resume, training, execution, max_artifact_bytes)
        if receipt["status"] == "generation_complete":
            return GenerationProgress(
                "generation_complete", artifact_from_mapping(artifact_reference(root, resume))
            )
        if pause_after is not None and pause_after <= len(state.observations):
            raise ValueError("generation pause must follow the recovered observation cursor")
    restore_rng(rng)
    started = time.perf_counter_ns()
    ticket = None
    failure: Exception | None = None
    attempt_rng = capture_rng()
    try:
        ticket = None if execution.timer is None else execution.timer.start(ResourcePhase.EVAL)
        while (
            state.status == "active"
            and len(state.emissions) < profile.schedule.generation_sentences
        ):
            attempt_rng = capture_rng()
            with torch.autocast(
                execution.device.type, dtype=torch.bfloat16, enabled=profile.precision == "bf16"
            ):
                state = advance_rollout(execution.model, state, execution.codec)
            if pause_after is not None and len(state.observations) >= pause_after:
                break
    except Exception as error:
        failure = error
    finally:
        try:
            cuda_seconds = (
                None
                if execution.timer is None or ticket is None
                else execution.timer.stop(ticket, ResourcePhase.EVAL)
            )
        except Exception as error:
            cuda_seconds = None
            failure = (
                error
                if failure is None
                else ExceptionGroup("generation and meter failures", [failure, error])
            )
    prefix = uuid4().hex
    if failure is not None:
        snapshot = _write_snapshot(
            root,
            prefix + "-generation-failure.pt",
            training,
            execution,
            state,
            attempt_rng,
            max_artifact_bytes,
        )
        write_json(
            root,
            prefix + "-generation-failure.json",
            {
                "schema": "h-generation-failure-v1",
                "run_identity": profile.identity,
                "scientific_authority": "none",
                "training": artifact_reference(root, training),
                "snapshot": artifact_reference(root, snapshot),
                "forensic_only": True,
                "status": "scientific_gap"
                if isinstance(failure, UndefinedDirection)
                else "execution_defect",
                "error_type": type(failure).__name__,
                "detail": str(failure),
            },
            max_bytes=max_artifact_bytes,
        )
        raise failure
    outcome = _summary(state, execution)
    status: Literal["paused", "generation_complete"] = (
        "paused" if outcome["outcome"] == "paused" else "generation_complete"
    )
    snapshot = _write_snapshot(
        root,
        prefix + "-generation.pt",
        training,
        execution,
        state,
        capture_rng(),
        max_artifact_bytes,
    )
    receipt = write_json(
        root,
        prefix + "-generation.json",
        {
            "schema": "h-generation-receipt-v2",
            "parent_receipt": None if resume is None else artifact_reference(root, resume),
            "run_identity": profile.identity,
            "scientific_authority": "none",
            "training": artifact_reference(root, training),
            "snapshot": artifact_reference(root, snapshot),
            "status": status,
            "summary": outcome,
            "wall_seconds": (time.perf_counter_ns() - started) / 1e9,
            "cuda_seconds": cuda_seconds,
        },
        max_bytes=max_artifact_bytes,
    )
    return GenerationProgress(status, artifact_from_mapping(artifact_reference(root, receipt)))
