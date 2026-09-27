"""Initial READY checkpoint creation and one-cell finalized crash adoption."""

from __future__ import annotations

import hashlib
from typing import NoReturn, Protocol

from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_values import (
    D1ProcessEventKind,
    D1ProcessState,
    canonical_lifecycle_event_line,
)
from entropic_substrate.ladder.capture.ledger.d1_process_manager import D1ProcessManager
from entropic_substrate.ladder.register.d1_launch_attempts import (
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.ladder.register.d1_registered_design import D1RegisteredDesign
from entropic_substrate.ladder.run.campaign.d1_campaign_checkpoint import (
    D1CampaignCheckpoint,
    D1CampaignCompletedCell,
    D1CampaignDisposition,
)
from entropic_substrate.ladder.run.campaign.d1_campaign_checkpoint_repository import (
    D1CampaignCheckpointReceipt,
)
from entropic_substrate.ladder.run.campaign.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.ladder.run.closure.d1_finalized_cell_custody import (
    D1FinalizedCellCustodyRecord,
)


class D1CampaignBootstrapError(ValueError):
    """Bootstrap or crash adoption authorities are missing, ambiguous, or forked."""


class D1CampaignCheckpointRepositoryPort(Protocol):
    def load(
        self, *, expected_sha256: str | None = None
    ) -> tuple[D1CampaignCheckpoint, D1CampaignCheckpointReceipt]: ...

    def save(
        self,
        checkpoint: D1CampaignCheckpoint,
        *,
        expected_previous_sha256: str | None,
    ) -> D1CampaignCheckpointReceipt: ...


def _fail(message: str, error: Exception | None = None) -> NoReturn:
    if error is None:
        raise D1CampaignBootstrapError(message)
    raise D1CampaignBootstrapError(message) from error


def bootstrap_d1_ready_checkpoint(
    *,
    design: D1RegisteredDesign,
    process_manager: D1ProcessManager,
    attempt_ledger: D1LaunchAttemptLedger,
    repository: D1CampaignCheckpointRepositoryPort,
) -> tuple[D1CampaignCheckpoint, D1CampaignCheckpointReceipt]:
    """Persist the sole empty READY checkpoint before any provider call."""
    if (
        type(design) is not D1RegisteredDesign
        or type(process_manager) is not D1ProcessManager
        or type(attempt_ledger) is not D1LaunchAttemptLedger
    ):
        raise TypeError("D1 bootstrap requires nominal design, lifecycle, and ledger")
    if (
        process_manager.state is not D1ProcessState.PREREGISTERED
        or process_manager.active_subject is not None
        or process_manager.finalized_subjects
        or attempt_ledger.events
    ):
        _fail("initial READY checkpoint requires preregistered lifecycle and empty ledger")
    prefix = process_manager.history.prefix
    checkpoint = D1CampaignCheckpoint(
        design.design_digest,
        design.decision_contract_digest,
        (),
        attempt_ledger,
        process_manager.state,
        prefix.prefix_sha256,
        prefix.event_count,
        None,
        D1CampaignDisposition.READY,
        None,
    )
    try:
        receipt = repository.save(checkpoint, expected_previous_sha256=None)
    except Exception as error:
        _fail(f"initial READY checkpoint save failed: {error}", error)
    return checkpoint, receipt


def _verify_checkpoint_prefix(
    checkpoint: D1CampaignCheckpoint,
    manager: D1ProcessManager,
) -> None:
    count = checkpoint.lifecycle_event_count
    if len(manager.events) < count:
        _fail("hydrated lifecycle is behind the checkpoint")
    prefix = b"".join(canonical_lifecycle_event_line(item) for item in manager.events[:count])
    if hashlib.sha256(prefix).hexdigest() != checkpoint.lifecycle_prefix:
        _fail("checkpoint lifecycle prefix differs from hydrated authority")


def _adopted_checkpoint(
    *,
    design: D1RegisteredDesign,
    intent: D1OperatorLaunchIntent,
    checkpoint: D1CampaignCheckpoint,
    manager: D1ProcessManager,
    record: D1FinalizedCellCustodyRecord,
    artifact: ArtifactIdentity,
) -> D1CampaignCheckpoint:
    record.require_factory_attested()
    _verify_checkpoint_prefix(checkpoint, manager)
    if (
        checkpoint.design_digest != design.design_digest
        or checkpoint.decision_contract_digest != design.decision_contract_digest
    ):
        _fail("checkpoint registration differs from the active design")
    if checkpoint.disposition is not D1CampaignDisposition.READY:
        _fail("only a READY checkpoint may adopt finalized progress")
    if (
        record.registered_design_digest != design.design_digest
        or intent.registered_design_digest != design.design_digest
        or record.logical_run_id != intent.plan.logical_run_id
        or record.arm is not intent.plan.profile.arm
        or record.seed != intent.plan.seed
        or record.plan_digest != intent.plan.plan_digest
        or hashlib.sha256(record.intent_json.encode()).hexdigest()
        != canonical_digest(intent.manifest)
    ):
        _fail("finalized candidate differs from the registered intent")
    if (
        artifact.content_hash.algorithm != "sha256"
        or artifact.content_hash.digest != record.record_digest
    ):
        _fail("finalized candidate artifact differs from its custody record")
    prior_events = checkpoint.attempt_ledger.events
    ledger = record.attempt_ledger
    extension = ledger.events[len(prior_events) :]
    terminal_status = (
        D1LaunchAttemptStatus.FAILED
        if record.numerical_defect is not None
        else D1LaunchAttemptStatus.COMPLETED
    )
    if ledger.events[: len(prior_events)] != prior_events or tuple(
        item.status for item in extension
    ) != (
        D1LaunchAttemptStatus.RECORDED,
        D1LaunchAttemptStatus.TRAINER_STARTED,
        terminal_status,
    ):
        _fail("finalized candidate attempt ledger is not exactly one completed cell ahead")
    expected_subjects = (
        *(cell.logical_run_id for cell in checkpoint.completed_cells),
        record.logical_run_id,
    )
    if (
        manager.state is not D1ProcessState.OPERATOR_FINALIZED
        or manager.active_subject is not None
        or manager.finalized_subjects != expected_subjects
        or len(manager.events) <= checkpoint.lifecycle_event_count
    ):
        _fail("hydrated lifecycle is not exactly one finalized cell ahead")
    boundary = manager.events[-1]
    if (
        boundary.event_kind is not D1ProcessEventKind.OPERATOR_FINALIZED
        or boundary.subject_id != record.logical_run_id
        or boundary.artifacts != (artifact,)
    ):
        _fail("operator-finalized lifecycle artifact differs from finalized custody")
    prefix = manager.history.prefix
    cell = D1CampaignCompletedCell(
        checkpoint.next_index,
        record.logical_run_id,
        record.arm,
        record.seed,
        record.plan_digest,
        canonical_digest(intent.manifest),
        record.segment_starts[0],
        ledger.ledger_digest,
        prefix.prefix_sha256,
        prefix.event_count,
        "numerical_defect" if record.numerical_defect is not None else "completed",
        artifact,
    )
    return D1CampaignCheckpoint(
        design.design_digest,
        design.decision_contract_digest,
        (*checkpoint.completed_cells, cell),
        ledger,
        manager.state,
        prefix.prefix_sha256,
        prefix.event_count,
        None,
        D1CampaignDisposition.READY,
        None,
    )


def adopt_d1_finalized_cell_after_crash(
    *,
    design: D1RegisteredDesign,
    intent: D1OperatorLaunchIntent,
    checkpoint: D1CampaignCheckpoint,
    process_manager: D1ProcessManager,
    candidates: tuple[tuple[D1FinalizedCellCustodyRecord, ArtifactIdentity], ...],
    repository: D1CampaignCheckpointRepositoryPort,
    generation: D1CampaignCheckpointReceipt,
) -> tuple[D1CampaignCheckpoint, D1CampaignCheckpointReceipt]:
    """Adopt exactly one durable cell and CAS-save it without provider access."""
    if (
        type(design) is not D1RegisteredDesign
        or type(intent) is not D1OperatorLaunchIntent
        or type(checkpoint) is not D1CampaignCheckpoint
        or type(process_manager) is not D1ProcessManager
        or type(generation) is not D1CampaignCheckpointReceipt
    ):
        raise TypeError("D1 crash adoption requires nominal authorities")
    if type(candidates) is not tuple or len(candidates) != 1:
        _fail("crash adoption requires exactly one finalized-cell candidate")
    if (
        type(candidates[0]) is not tuple
        or len(candidates[0]) != 2
        or type(candidates[0][0]) is not D1FinalizedCellCustodyRecord
        or type(candidates[0][1]) is not ArtifactIdentity
    ):
        raise TypeError("D1 crash adoption candidate must be a record/artifact pair")
    try:
        current, observed = repository.load(expected_sha256=generation.sha256)
    except Exception as error:
        _fail(f"checkpoint generation reload failed: {error}", error)
    if current != checkpoint or observed != generation:
        _fail("checkpoint generation differs from crash-adoption authority")
    record, artifact = candidates[0]
    adopted = _adopted_checkpoint(
        design=design,
        intent=intent,
        checkpoint=checkpoint,
        manager=process_manager,
        record=record,
        artifact=artifact,
    )
    try:
        receipt = repository.save(adopted, expected_previous_sha256=generation.sha256)
    except Exception as error:
        _fail(f"adopted checkpoint CAS failed: {error}", error)
    return adopted, receipt
