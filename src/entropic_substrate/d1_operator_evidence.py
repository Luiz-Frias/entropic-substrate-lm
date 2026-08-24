"""Trainer-input and pre-termination export custody for D1 operator runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

from entropic_substrate.d1_billing_finalization_values import D1TrainerHandoff
from entropic_substrate.d1_resume_trainer import D1ResumeRunCompletion
from entropic_substrate.d1_trainer_handoff import (
    D1FreshRunCompletion,
    D1TrainerFailureHandoff,
)
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.lifecycle import (
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.evidence.resource_provenance import (
    EvidenceGap,
    RunStartProvenance,
    SegmentStartMode,
)
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.experiments.d1_plan import D1RunPlan
from entropic_substrate.predictive_state.checkpoint_repository import (
    load_checkpoint_candidate,
)

_RECEIPT_TOKEN: Final = object()


@dataclass(frozen=True, slots=True)
class D1TrainerInputEnvelope:
    """Canonical plan/start bytes staged before invoking the trainer CLI."""

    plan: D1RunPlan
    start: RunStartProvenance

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.plan), D1RunPlan):
            raise TypeError("trainer input requires a typed D1 run plan")
        if type(self.start) is not RunStartProvenance:
            raise TypeError("trainer input requires RunStartProvenance")
        if self.start.logical_run_id != self.plan.logical_run_id:
            raise ValueError("trainer input plan and start logical identities differ")
        container = self.plan.run.execution.container
        if container is None or self.start.image_digest != container.image_digest:
            raise ValueError("trainer input start image differs from the run plan")
        compile_expected = self.plan.profile.draft.execution_protocol.optimization.compile
        if self.start.compiler_enabled is not compile_expected:
            raise ValueError("trainer input compiler posture differs from the run plan")

    @property
    def command(self) -> str:
        return (
            "d1-run-fresh"
            if self.start.segment_start_mode is SegmentStartMode.FRESH
            else "d1-run-resume"
        )

    @property
    def payload(self) -> dict[str, object]:
        return {
            "command": self.command,
            "plan": self.plan.plan_manifest,
            "run_start": run_start_to_mapping(self.start),
            "schema_version": "d1-operator-trainer-input-v1",
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class D1ExportedTrainerHandoff:
    """A typed trainer result paired with its exported authoritative journal."""

    handoff: D1TrainerHandoff
    journal_path: Path

    def __post_init__(self) -> None:
        if type(self.handoff) not in (
            D1FreshRunCompletion,
            D1ResumeRunCompletion,
            D1TrainerFailureHandoff,
        ):
            raise TypeError("export candidate requires a typed D1 trainer handoff")
        if not isinstance(cast(object, self.journal_path), Path):
            raise TypeError("export candidate journal_path must be a Path")
        if self.journal_path.is_symlink() or not self.journal_path.is_file():
            raise ValueError("exported journal must be a regular non-symlink file")


@dataclass(frozen=True, slots=True)
class D1TrainerExportReceipt:
    """Factory-attested bytes safe to retain before terminating the pod."""

    handoff: D1TrainerHandoff
    journal: ArtifactIdentity
    checkpoint: ArtifactIdentity | EvidenceGap
    quality_export_ready: bool
    _token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _RECEIPT_TOKEN:
            raise TypeError("D1TrainerExportReceipt is factory-only")
        if type(self.journal) is not ArtifactIdentity:
            raise TypeError("trainer export receipt requires journal identity")
        if type(self.checkpoint) not in (ArtifactIdentity, EvidenceGap):
            raise TypeError("trainer export checkpoint must be identity or gap")
        if self.quality_export_ready is not isinstance(self.checkpoint, ArtifactIdentity):
            raise ValueError("quality readiness differs from checkpoint evidence")

    @property
    def handoff_kind(self) -> str:
        if type(self.handoff) is D1FreshRunCompletion:
            return "fresh_completion"
        if type(self.handoff) is D1ResumeRunCompletion:
            return "resume_completion"
        return "trainer_failure"

    @property
    def manifest(self) -> dict[str, object]:
        checkpoint = self.checkpoint
        checkpoint_manifest: dict[str, object]
        if isinstance(checkpoint, ArtifactIdentity):
            checkpoint_manifest = {
                "kind": "artifact",
                "path": checkpoint.path,
                "sha256": checkpoint.content_hash.digest,
            }
        else:
            checkpoint_manifest = {
                "field": checkpoint.field,
                "kind": "gap",
                "reason": checkpoint.reason,
            }
        return {
            "checkpoint": checkpoint_manifest,
            "handoff_kind": self.handoff_kind,
            "journal": {
                "path": self.journal.path,
                "sha256": self.journal.content_hash.digest,
            },
            "quality_export_ready": self.quality_export_ready,
            "schema_version": "d1-trainer-export-receipt-v1",
        }


def _journal_identity(candidate: D1ExportedTrainerHandoff) -> ArtifactIdentity:
    path = candidate.journal_path
    try:
        data = path.read_bytes()
        events = read_journal(path)
    except Exception as error:
        raise ValueError(f"exported journal is invalid: {error}") from error
    digest = hashlib.sha256(data).hexdigest()
    state = candidate.handoff.termination.journal_state
    last = events[-1]
    if (
        digest != state.prefix_sha256
        or last.event_seq != state.last_event_sequence
        or last.successful_step != state.successful_step
        or last.logical_run_id != state.logical_run_id
        or last.execution_digest != state.execution_digest
    ):
        raise ValueError("exported journal bytes differ from trainer handoff state")
    expected_tail = (
        JournalEventKind.TRAINER_ABORTED
        if type(candidate.handoff) is D1TrainerFailureHandoff
        else JournalEventKind.CHECKPOINT
    )
    if last.event_kind is not expected_tail:
        raise ValueError(f"exported journal must end in {expected_tail.value}")
    return ArtifactIdentity(str(path), ContentHash("sha256", digest))


def _checkpoint_identity(
    candidate: D1ExportedTrainerHandoff,
) -> ArtifactIdentity | EvidenceGap:
    handoff = candidate.handoff
    if isinstance(handoff, D1TrainerFailureHandoff):
        return EvidenceGap(
            "terminal_checkpoint",
            "trainer failure handoff carries no successful terminal checkpoint",
        )
    receipt = handoff.final_checkpoint
    try:
        _, actual = load_checkpoint_candidate(
            receipt.path,
            expected_sha256=receipt.sha256,
        )
    except Exception as error:
        raise ValueError(f"exported checkpoint is invalid: {error}") from error
    if actual != receipt:
        raise ValueError("exported checkpoint bytes differ from trainer handoff receipt")
    events = read_journal(candidate.journal_path)
    if events[-1].payload.get("checkpoint_sha256") != receipt.sha256:
        raise ValueError("exported checkpoint differs from authoritative journal")
    return ArtifactIdentity(
        str(receipt.path),
        ContentHash("sha256", receipt.sha256),
    )


def verify_d1_trainer_export(
    candidate: D1ExportedTrainerHandoff,
) -> D1TrainerExportReceipt:
    """Re-open exported bytes; never trust the transport handoff alone."""
    if type(candidate) is not D1ExportedTrainerHandoff:
        raise TypeError("trainer export verification requires a typed candidate")
    journal = _journal_identity(candidate)
    checkpoint = _checkpoint_identity(candidate)
    return D1TrainerExportReceipt(
        handoff=candidate.handoff,
        journal=journal,
        checkpoint=checkpoint,
        quality_export_ready=isinstance(checkpoint, ArtifactIdentity),
        _token=_RECEIPT_TOKEN,
    )
