"""Recoverable engineering-probe training driver over the shared numerical engine."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.core.sentence.recovery_state import recovery_record
from entropic_substrate.kernel.artifacts import artifact_from_mapping, normalize_d1_artifact_root
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.capture.ports.metric_sink import MetricSink
from entropic_substrate.ladder.run.evaluation_records import read_evaluation
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.journal import artifact_reference, read_json, write_json
from entropic_substrate.ladder.run.seals import read_seal, resume_seal, seal_step


@dataclass(frozen=True)
class TrainingProgress:
    status: Literal["paused", "training_complete"]
    last_seal: ArtifactIdentity
    receipt: ArtifactIdentity


def _progress(
    root: Path,
    status: Literal["paused", "training_complete"],
    seal: ArtifactIdentity,
    receipt: ArtifactIdentity,
) -> TrainingProgress:
    return TrainingProgress(
        status,
        artifact_from_mapping(artifact_reference(root, seal)),
        artifact_from_mapping(artifact_reference(root, receipt)),
    )


def read_training_progress(
    root: Path,
    artifact: ArtifactIdentity,
    profile: HProfile,
    source_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
) -> dict[str, object]:
    """Admit resume/completion only against the exact source and sealed ancestry."""
    if profile.run_class not in ("local_tiny", "remote_tiny"):
        raise ValueError("probe receipt has no scientific authority")
    row = recovery_record(
        read_json(root, artifact, max_bytes=max_artifact_bytes),
        "schema run_identity run_class scientific_authority profile status phase step successful_optimizer_steps last_seal evaluations parent_receipt inventory",
    )
    if (
        row["schema"] != "h-probe-training-progress-v1"
        or row["run_identity"] != profile.identity
        or row["run_class"] != profile.run_class
        or row["scientific_authority"] != "none"
    ):
        raise ValueError("training receipt construction or authority mismatch")
    bound_profile = HProfile.from_manifest(
        read_json(root, artifact_from_mapping(row["profile"]), max_bytes=max_artifact_bytes)
    )
    if bound_profile != profile:
        raise ValueError("training receipt profile differs from requested profile")
    seal = read_seal(
        root,
        artifact_from_mapping(row["last_seal"]),
        profile,
        source_root,
        max_bytes=max_artifact_bytes,
        max_source_bytes=max_source_bytes,
    )
    total = seal.step + (profile.schedule.warmup_steps if seal.phase == "prediction" else 0)
    if (
        row["phase"] != seal.phase
        or type(row["step"]) is not int
        or row["step"] != seal.step
        or type(row["successful_optimizer_steps"]) is not int
        or row["successful_optimizer_steps"] != total
    ):
        raise ValueError("training receipt counters differ from sealed work")
    if row["status"] not in ("checkpoint", "paused", "training_complete"):
        raise ValueError("unknown training receipt status")
    if row["status"] == "training_complete" and (
        seal.phase != "prediction" or seal.step != profile.schedule.steps
    ):
        raise ValueError("training completion lacks its full optimizer budget")
    evaluations = row["evaluations"]
    if type(evaluations) is not list:
        raise ValueError("evaluation roster must be an array")
    observed: list[int] = []
    for item in evaluations:
        entry = recovery_record(item, "step artifact")
        step = entry["step"]
        if type(step) is not int:
            raise ValueError("evaluation step must be integer")
        evaluation = read_evaluation(
            root,
            artifact_from_mapping(entry["artifact"]),
            profile,
            source_root,
            max_bytes=max_artifact_bytes,
            max_source_bytes=max_source_bytes,
        )
        point = artifact_from_mapping(evaluation.get("checkpoint_seal"))
        if point.content_hash.digest not in seal.seal_hashes:
            raise ValueError("evaluation belongs to another checkpoint lineage")
        point_state = read_seal(
            root,
            point,
            profile,
            source_root,
            max_bytes=max_artifact_bytes,
            max_source_bytes=max_source_bytes,
        )
        if (
            point_state.phase != "prediction"
            or point_state.step != step
            or evaluation.get("schema") != "h-predictive-evaluation-v1"
            or evaluation.get("run_identity") != profile.identity
            or type(evaluation.get("step")) is not int
            or evaluation["step"] != step
        ):
            raise ValueError("evaluation identity differs from its sealed checkpoint")
        observed.append(step)
    expected = (
        []
        if seal.phase == "codec_warmup"
        else [
            s
            for s in range(1, seal.step + 1)
            if s % profile.schedule.evaluation_interval == 0 or s == profile.schedule.steps
        ]
    )
    if observed != expected:
        raise ValueError("training receipt lacks the complete declared evaluation grid")
    if row["parent_receipt"] is not None:
        parent = read_json(
            root, artifact_from_mapping(row["parent_receipt"]), max_bytes=max_artifact_bytes
        )
        if (
            parent.get("run_identity") != profile.identity
            or artifact_from_mapping(parent.get("last_seal")).content_hash.digest
            not in seal.seal_hashes
        ):
            raise ValueError("resume receipt belongs to another lineage")
    return row


def run_probe_training(
    profile: HProfile,
    source_root: Path,
    output_root: Path,
    *,
    max_source_bytes: int,
    max_artifact_bytes: int,
    resume: ArtifactIdentity | None = None,
    stop_after_seal: Callable[[str, int], bool] | None = None,
    on_checkpoint: Callable[[ArtifactIdentity], None] | None = None,
    sink: MetricSink | None = None,
) -> TrainingProgress:
    """Run all declared training steps; this result is not a full flywheel verdict.

    Checkpoint/evaluation failures retain partial evidence and propagate. A
    stop callback is observed only after a durable seal and evaluation receipt.
    Scientific adapters and successor remote probes require separate authority.

    ``sink``, when given, observes every committed training-step row exactly
    as :mod:`entropic_substrate.ladder.run.journal` receives it -- an
    ADDITIVE observability mirror (RUNG 2b), never a substitute for the D1
    journal's authoritative, content-addressed record. This function never
    constructs a sink and never calls ``sink.finish()``; the caller that
    built it owns its lifecycle.
    """
    if profile.run_class not in ("local_tiny", "remote_tiny") or (
        profile.run_class == "remote_tiny" and profile.stage != "H0"
    ):
        raise ValueError("probe driver lacks scientific or successor authority")
    root = normalize_d1_artifact_root(output_root)
    evaluations: list[dict[str, object]] = []
    previous_receipt = resume
    last_seal = None
    if resume is None:
        if any(root.iterdir()):
            raise ValueError("fresh probe output must be empty")
        profile_artifact = write_json(
            root, "profile.json", profile.manifest, max_bytes=max_artifact_bytes
        )
    else:
        previous = read_training_progress(
            root,
            resume,
            profile,
            source_root,
            max_source_bytes=max_source_bytes,
            max_artifact_bytes=max_artifact_bytes,
        )
        last_seal = artifact_from_mapping(previous["last_seal"])
        if previous["status"] == "training_complete":
            return _progress(root, "training_complete", last_seal, resume)
        profile_artifact = artifact_from_mapping(previous["profile"])
        evaluations = cast(list[dict[str, object]], previous["evaluations"]).copy()
    execution: SentenceExecution | None = None
    prefix = uuid4().hex
    pending: list[dict[str, object]] = []
    try:
        execution = SentenceExecution(profile, source_root, max_source_bytes=max_source_bytes)
        if last_seal is not None:
            resume_seal(root, last_seal, execution, max_bytes=max_artifact_bytes)
        while True:
            if (
                execution.phase == "codec_warmup"
                and execution.step == profile.schedule.warmup_steps
            ):
                execution.begin_prediction()
            step_row = execution.advance()
            pending.append(step_row)
            if sink is not None:
                sink.log(step_row)
            final = execution.step == (
                profile.schedule.warmup_steps
                if execution.phase == "codec_warmup"
                else profile.schedule.steps
            )
            evaluate = execution.phase == "prediction" and (
                final or execution.step % profile.schedule.evaluation_interval == 0
            )
            if not (
                final or evaluate or execution.step % profile.schedule.checkpoint_interval == 0
            ):
                continue
            last_seal = seal_step(
                root,
                prefix,
                execution,
                tuple(pending),
                parent=last_seal,
                max_bytes=max_artifact_bytes,
            )
            pending.clear()
            if evaluate:
                report = execution.evaluate()
                report["checkpoint_seal"] = artifact_reference(root, last_seal)
                evaluation = write_json(
                    root,
                    f"{prefix}-{execution.step:08d}-evaluation.json",
                    report,
                    max_bytes=max_artifact_bytes,
                )
                read_evaluation(
                    root,
                    evaluation,
                    profile,
                    source_root,
                    max_bytes=max_artifact_bytes,
                    max_source_bytes=max_source_bytes,
                )
                evaluations.append(
                    {"step": execution.step, "artifact": artifact_reference(root, evaluation)}
                )
            stopped = (
                False
                if stop_after_seal is None
                else stop_after_seal(execution.phase, execution.step)
            )
            if type(stopped) is not bool:
                raise ValueError("stop callback must return a boolean")
            complete = execution.phase == "prediction" and final
            status = "training_complete" if complete else "paused" if stopped else "checkpoint"
            receipt = write_json(
                root,
                f"{prefix}-{execution.phase}-{execution.step:08d}-progress.json",
                {
                    "schema": "h-probe-training-progress-v1",
                    "run_identity": profile.identity,
                    "run_class": profile.run_class,
                    "scientific_authority": "none",
                    "profile": artifact_reference(root, profile_artifact),
                    "status": status,
                    "phase": execution.phase,
                    "step": execution.step,
                    "successful_optimizer_steps": execution.step
                    + (profile.schedule.warmup_steps if execution.phase == "prediction" else 0),
                    "last_seal": artifact_reference(root, last_seal),
                    "evaluations": evaluations,
                    "parent_receipt": None
                    if previous_receipt is None
                    else artifact_reference(root, previous_receipt),
                    "inventory": execution.inventory(),
                },
                max_bytes=max_artifact_bytes,
            )
            previous_receipt = receipt
            if complete or stopped or on_checkpoint is not None:
                read_training_progress(
                    root,
                    receipt,
                    profile,
                    source_root,
                    max_source_bytes=max_source_bytes,
                    max_artifact_bytes=max_artifact_bytes,
                )
            if on_checkpoint is not None:
                on_checkpoint(artifact_from_mapping(artifact_reference(root, receipt)))
            if complete or stopped:
                return _progress(
                    root, "training_complete" if complete else "paused", last_seal, receipt
                )
    except Exception as error:
        write_json(
            root,
            f"{prefix}-failure.json",
            {
                "schema": "h-probe-training-failure-v2",
                "run_identity": profile.identity,
                "run_class": profile.run_class,
                "scientific_authority": "none",
                "profile": artifact_reference(root, profile_artifact),
                "status": "scientific_gap"
                if isinstance(error, UndefinedDirection)
                else "execution_defect",
                "error_type": type(error).__name__,
                "error_code": getattr(error, "code", type(error).__name__),
                "phase": "initialization" if execution is None else execution.phase,
                "step": 0 if execution is None else execution.step,
                "error": str(error),
                "last_attempt": None if execution is None else execution.last_attempt,
                "last_evaluation": None if execution is None else execution.last_evaluation,
                "unsealed_rows": pending,
                "last_seal": None if last_seal is None else artifact_reference(root, last_seal),
                "last_receipt": None
                if previous_receipt is None
                else artifact_reference(root, previous_receipt),
            },
            max_bytes=max_artifact_bytes,
        )
        raise
