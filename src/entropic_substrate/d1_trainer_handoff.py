"""Typed D1 trainer termination handoffs before operator finalization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast

from entropic_substrate.d1_evaluation import D1EvalResult
from entropic_substrate.evidence.d1_journal_values import JournalCheckpointState
from entropic_substrate.evidence.resource_ledger import ResourceLedgerState
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    execution_segment_id,
)
from entropic_substrate.evidence.resource_work import ResourceWorkSnapshot
from entropic_substrate.predictive_state.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
)

__all__ = [
    "D1FreshRunCompletion",
    "D1FreshRunFailure",
    "D1TrainerFailureHandoff",
    "D1TrainerTermination",
    "capture_d1_trainer_termination",
]

_PENDING_ABORT_EVENT = "TRAINER_ABORTED"


def _utc(value: datetime, *, label: str) -> None:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be a UTC timestamp")


@dataclass(frozen=True, slots=True)
class D1TrainerTermination:
    """Measured process stop joined to exact journal and work-ledger cursors."""

    logical_run_id: str
    execution_segment_index: int
    execution_segment_id: str
    trainer_ended_at: datetime
    successful_step: int
    journal_state: JournalCheckpointState
    resource_ledger_state: ResourceLedgerState
    schema_version: str = "d1-trainer-termination-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "d1-trainer-termination-v1":
            raise ValueError("unsupported D1 trainer termination schema")
        if type(self.logical_run_id) is not str or not self.logical_run_id.strip():
            raise ValueError("D1 trainer termination logical_run_id must be nonempty")
        if type(self.execution_segment_index) is not int or self.execution_segment_index < 0:
            raise ValueError("D1 trainer termination segment index must be nonnegative")
        if self.execution_segment_id != execution_segment_id(
            self.logical_run_id, self.execution_segment_index
        ):
            raise ValueError("D1 trainer termination execution-segment identity is inconsistent")
        _utc(self.trainer_ended_at, label="D1 trainer_ended_at")
        if type(self.successful_step) is not int or self.successful_step < 0:
            raise ValueError("D1 trainer termination successful_step must be nonnegative")
        if type(self.journal_state) is not JournalCheckpointState:
            raise TypeError("D1 trainer termination requires JournalCheckpointState")
        if type(self.resource_ledger_state) is not ResourceLedgerState:
            raise TypeError("D1 trainer termination requires ResourceLedgerState")
        if self.journal_state.logical_run_id != self.logical_run_id:
            raise ValueError("D1 termination journal logical_run_id differs")
        if self.resource_ledger_state.logical_run_id != self.logical_run_id:
            raise ValueError("D1 termination resource-ledger logical_run_id differs")
        if self.resource_ledger_state.execution_segment_index != self.execution_segment_index:
            raise ValueError("D1 termination resource-ledger segment differs")
        if self.journal_state.successful_step > self.successful_step:
            raise ValueError("D1 termination journal is ahead of successful_step")
        ledger_step = self.resource_ledger_state.last_training_step
        if ledger_step is not None and ledger_step > self.successful_step:
            raise ValueError("D1 termination resource ledger is ahead of successful_step")

    @property
    def manifest(self) -> dict[str, object]:
        return {
            "execution_segment_id": self.execution_segment_id,
            "execution_segment_index": self.execution_segment_index,
            "journal_state": self.journal_state.manifest,
            "logical_run_id": self.logical_run_id,
            "resource_ledger_state": self.resource_ledger_state.manifest,
            "schema_version": self.schema_version,
            "successful_step": self.successful_step,
            "trainer_ended_at": self.trainer_ended_at.isoformat().replace("+00:00", "Z"),
        }


def capture_d1_trainer_termination(
    *,
    start: RunStartProvenance,
    trainer_ended_at: datetime,
    successful_step: int,
    journal_state: JournalCheckpointState,
    resource_ledger_state: ResourceLedgerState,
) -> D1TrainerTermination:
    """Join a measured end timestamp to one admitted execution segment."""
    if type(start) is not RunStartProvenance:
        raise TypeError("D1 trainer termination requires RunStartProvenance")
    _utc(trainer_ended_at, label="D1 trainer_ended_at")
    if trainer_ended_at < start.trainer_started_at:
        raise ValueError("D1 trainer_ended_at precedes trainer_started_at")
    return D1TrainerTermination(
        logical_run_id=start.logical_run_id,
        execution_segment_index=start.execution_segment_index,
        execution_segment_id=start.execution_segment_id,
        trainer_ended_at=trainer_ended_at,
        successful_step=successful_step,
        journal_state=journal_state,
        resource_ledger_state=resource_ledger_state,
    )


@dataclass(frozen=True, slots=True)
class D1FreshRunCompletion:
    """Successful trainer-to-operator handoff; RUN_END remains outstanding."""

    termination: D1TrainerTermination
    final_checkpoint: D1CheckpointArtifactReceipt
    final_evaluation: D1EvalResult
    final_resource_work: ResourceWorkSnapshot
    data_order_sha256: str
    data_order_batches_attested: int
    operator_finalization_required: bool = True

    def __post_init__(self) -> None:
        if type(self.termination) is not D1TrainerTermination:
            raise TypeError("D1 completion requires trainer termination identity")
        if self.termination.successful_step < 1:
            raise ValueError("D1 completion successful_step must be positive")
        if self.termination.journal_state.successful_step != self.successful_step:
            raise ValueError("D1 completion journal and successful-step cursors differ")
        if self.termination.resource_ledger_state.last_training_step != self.successful_step:
            raise ValueError("D1 completion resource ledger is not at the terminal eval")
        if type(self.final_checkpoint) is not D1CheckpointArtifactReceipt:
            raise TypeError("D1 completion requires a checkpoint receipt")
        if type(self.final_evaluation) is not D1EvalResult:
            raise TypeError("D1 completion requires a final evaluation")
        if type(self.final_resource_work) is not ResourceWorkSnapshot:
            raise TypeError("D1 completion requires final resource work")
        if self.final_resource_work.logical_run_id != self.termination.logical_run_id:
            raise ValueError("D1 completion resource snapshot logical run differs")
        if (
            self.final_resource_work.execution_segment_index
            != self.termination.execution_segment_index
            or self.final_resource_work.training_step != self.successful_step
        ):
            raise ValueError("D1 completion resource snapshot cursor differs")
        if self.termination.trainer_ended_at < self.final_resource_work.observed_at:
            raise ValueError("D1 completion trainer ended before final resource observation")
        if len(self.data_order_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.data_order_sha256
        ):
            raise ValueError("D1 completion data-order digest must be lowercase SHA-256")
        if (
            type(self.data_order_batches_attested) is not int
            or self.data_order_batches_attested < 1
        ):
            raise ValueError("D1 completion must attest at least one data-order batch")
        if self.operator_finalization_required is not True:
            raise ValueError("D1 trainer cannot finalize provider billing or RUN_END")

    @property
    def successful_step(self) -> int:
        return self.termination.successful_step

    @property
    def journal_state(self) -> JournalCheckpointState:
        return self.termination.journal_state


@dataclass(frozen=True, slots=True)
class D1TrainerFailureHandoff:
    """Nonterminal defect evidence pending a journal-level TRAINER_ABORTED event."""

    termination: D1TrainerTermination
    error_type: str
    error_message: str
    pending_journal_event: str = _PENDING_ABORT_EVENT
    rollback_claimed: bool = False
    operator_finalization_required: bool = True

    def __post_init__(self) -> None:
        if type(self.termination) is not D1TrainerTermination:
            raise TypeError("D1 failure handoff requires trainer termination identity")
        if type(self.error_type) is not str or not self.error_type.strip():
            raise ValueError("D1 failure handoff error_type must be nonempty")
        if type(self.error_message) is not str or not self.error_message.strip():
            raise ValueError("D1 failure handoff error_message must be nonempty")
        if self.pending_journal_event != _PENDING_ABORT_EVENT:
            raise ValueError("D1 failure handoff must name pending TRAINER_ABORTED")
        if self.rollback_claimed is not False:
            raise ValueError("D1 failure handoff cannot claim rollback")
        if self.operator_finalization_required is not True:
            raise ValueError("D1 failure still requires operator finalization")

    @classmethod
    def from_error(
        cls, termination: D1TrainerTermination, error: BaseException
    ) -> D1TrainerFailureHandoff:
        if not isinstance(cast(object, error), BaseException):
            raise TypeError("D1 failure handoff requires a caught exception")
        return cls(
            termination=termination,
            error_type=type(error).__name__,
            error_message=str(error) or repr(error),
        )


class D1FreshRunFailure(RuntimeError):
    """Typed application failure whose ``__cause__`` is the original defect."""

    def __init__(self, handoff: D1TrainerFailureHandoff) -> None:
        if type(handoff) is not D1TrainerFailureHandoff:
            raise TypeError("D1FreshRunFailure requires D1TrainerFailureHandoff")
        self.handoff = handoff
        super().__init__(
            f"D1 fresh trainer failed after step {handoff.termination.successful_step}: "
            f"{handoff.error_type}: {handoff.error_message}"
        )
