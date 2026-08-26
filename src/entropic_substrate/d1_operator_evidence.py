"""Trainer-input and pre-termination export custody for D1 operator runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, NoReturn, cast

from entropic_substrate.d1_billing_finalization_values import D1TrainerHandoff
from entropic_substrate.d1_operator_values import D1RuntimeObservation
from entropic_substrate.d1_resume_trainer import D1ResumeRunCompletion
from entropic_substrate.d1_trainer_handoff import (
    D1FreshRunCompletion,
    D1TrainerFailureHandoff,
)
from entropic_substrate.d1_trainer_request_codec import (
    D1TrainerRequest,
    d1_trainer_request_sha256,
)
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import (
    JournalCheckpointState,
    JournalEventKind,
    canonical_event_line,
)
from entropic_substrate.evidence.lifecycle import (
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.evidence.resource_provenance import EvidenceGap
from entropic_substrate.experiments.d1_primitives import (
    canonical_digest,
    require_nonempty_string,
    require_sha256,
)
from entropic_substrate.factory_attestation import FactoryAttestation
from entropic_substrate.predictive_state.checkpoint_repository import (
    load_checkpoint_candidate,
)

_RECEIPT_TOKEN: Final = object()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True)


def d1_runtime_observation_sha256(runtime: D1RuntimeObservation) -> str:
    if type(runtime) is not D1RuntimeObservation:
        raise TypeError("D1 runtime identity requires D1RuntimeObservation")
    return canonical_digest(
        {
            "compiler_enabled": runtime.compiler_enabled,
            "compiler_version": runtime.compiler_version,
            "cuda_version": runtime.cuda_version,
            "driver_version": runtime.driver_version,
            "pod_id": runtime.pod_id,
            "pytorch_version": runtime.pytorch_version,
            "trainer_monotonic_origin_ns": runtime.trainer_monotonic_origin_ns,
            "trainer_started_at": runtime.trainer_started_at.isoformat(),
        }
    )


class _NoncopyableTrainerReceipt:
    __slots__ = ()

    def require_factory_attested(self) -> None:
        if not _TRAINER_RECEIPTS.registered(self):
            raise TypeError("D1 trainer receipt is not factory-attested")
        if not _TRAINER_RECEIPTS.contains(self):
            raise TypeError("D1 trainer receipt changed after minting")

    def _reject_copy(self, *_: object) -> NoReturn:
        raise TypeError("D1 trainer receipt cannot be copied")

    __copy__ = __deepcopy__ = __reduce_ex__ = _reject_copy


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class D1StagedTrainerRequestReceipt(_NoncopyableTrainerReceipt):
    request: D1TrainerRequest
    request_sha256: str
    artifact: ArtifactIdentity

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("D1StagedTrainerRequestReceipt is factory-attested")


def attest_d1_staged_trainer_request(
    *, request: D1TrainerRequest, artifact: ArtifactIdentity
) -> D1StagedTrainerRequestReceipt:
    request_sha256 = d1_trainer_request_sha256(request)
    if type(artifact) is not ArtifactIdentity:
        raise TypeError("D1 staged trainer request requires ArtifactIdentity")
    if artifact.content_hash != ContentHash("sha256", request_sha256):
        raise ValueError("D1 staged trainer artifact differs from the canonical request digest")
    receipt = object.__new__(D1StagedTrainerRequestReceipt)
    object.__setattr__(receipt, "request", request)
    object.__setattr__(receipt, "request_sha256", request_sha256)
    object.__setattr__(receipt, "artifact", artifact)
    _TRAINER_RECEIPTS.register(receipt)
    return receipt


@dataclass(frozen=True, slots=True, init=False, weakref_slot=True)
class D1TrainerStartReceipt(_NoncopyableTrainerReceipt):
    """Acknowledges the raw bootstrap process and its exact trainer command."""

    pod_id: str
    attempt_sha256: str
    request_sha256: str
    trainer_started_at: datetime
    runtime_sha256: str
    process_id: str
    command_sha256: str

    def __init__(self, *_: object, **__: object) -> None:
        raise TypeError("D1TrainerStartReceipt is factory-attested")


def _receipt_fp(value: _NoncopyableTrainerReceipt) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


_TRAINER_RECEIPTS = FactoryAttestation[_NoncopyableTrainerReceipt, str](_receipt_fp)


def attest_d1_trainer_start(
    *,
    pod_id: str,
    attempt_sha256: str,
    request_sha256: str,
    trainer_started_at: datetime,
    runtime_sha256: str,
    process_id: str,
    command_sha256: str,
) -> D1TrainerStartReceipt:
    require_nonempty_string(pod_id, label="D1 trainer start pod_id")
    require_nonempty_string(process_id, label="D1 trainer start process_id")
    if process_id.strip() != process_id:
        raise ValueError("D1 trainer start process_id must be trimmed")
    for value, label in (
        (attempt_sha256, "attempt"),
        (request_sha256, "request"),
        (runtime_sha256, "runtime"),
        (command_sha256, "command"),
    ):
        require_sha256(value, label=f"D1 trainer start {label}")
    if (
        type(trainer_started_at) is not datetime
        or trainer_started_at.tzinfo is None
        or trainer_started_at.utcoffset() != timedelta(0)
    ):
        raise ValueError("D1 trainer start timestamp must be UTC")
    receipt = object.__new__(D1TrainerStartReceipt)
    object.__setattr__(receipt, "pod_id", pod_id)
    object.__setattr__(receipt, "attempt_sha256", attempt_sha256)
    object.__setattr__(receipt, "request_sha256", request_sha256)
    object.__setattr__(receipt, "trainer_started_at", trainer_started_at)
    object.__setattr__(receipt, "runtime_sha256", runtime_sha256)
    object.__setattr__(receipt, "process_id", process_id)
    object.__setattr__(receipt, "command_sha256", command_sha256)
    _TRAINER_RECEIPTS.register(receipt)
    return receipt


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
    if isinstance(candidate.handoff, D1TrainerFailureHandoff):
        termination = candidate.handoff.termination
        abort_payload = cast(dict[str, object], last.manifest["payload"])
        observed_model_step = abort_payload.get("model_successful_step")
        if (
            type(observed_model_step) is not int
            or observed_model_step != termination.model_successful_step
        ):
            raise ValueError("exported TRAINER_ABORTED model_successful_step differs from handoff")
        prior = events[-2]
        pre_abort = JournalCheckpointState(
            logical_run_id=prior.logical_run_id,
            execution_digest=prior.execution_digest,
            last_event_sequence=prior.event_seq,
            prefix_sha256=hashlib.sha256(
                b"".join(canonical_event_line(event) for event in events[:-1])
            ).hexdigest(),
            successful_step=prior.successful_step,
        )
        if _canonical_json(abort_payload.get("journal_cursor")) != _canonical_json(
            pre_abort.manifest
        ):
            raise ValueError("exported TRAINER_ABORTED journal_cursor differs from pre-abort bytes")
        if _canonical_json(abort_payload.get("resource_ledger_state")) != _canonical_json(
            termination.resource_ledger_state.manifest
        ):
            raise ValueError("exported TRAINER_ABORTED resource_ledger_state differs from handoff")
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
