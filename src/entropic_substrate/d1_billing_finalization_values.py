"""Typed values at the D1 operator-finalization application boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from entropic_substrate.d1_resume_trainer import D1ResumeRunCompletion
from entropic_substrate.d1_trainer_handoff import (
    D1FreshRunCompletion,
    D1TrainerFailureHandoff,
    D1TrainerTermination,
)
from entropic_substrate.evidence.d1_journal import JournalTailQuarantineReceipt
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalEvent,
    JournalCheckpointState,
    JournalEventKind,
)
from entropic_substrate.evidence.resource_billing import (
    BillingGap,
    ExecutionSegmentTermination,
    ProviderBilling,
    ProviderBillingPartial,
    TerminalProviderBillingEvidence,
)
from entropic_substrate.evidence.resource_billing_reconciliation import (
    LogicalBillingMissingCost,
    LogicalBillingOutcome,
    LogicalBillingReconciliation,
)
from entropic_substrate.evidence.resource_ledger import ResourceLedgerState
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    validate_execution_segments,
)
from entropic_substrate.evidence.resource_work import ResourceWorkSnapshot
from entropic_substrate.experiments.d1_plan import D1RunPlan
from entropic_substrate.experiments.d1_primitives import EvidenceClass

__all__ = [
    "D1BillingFinalization",
    "D1BillingFinalizationRequest",
    "D1TerminalResourceEvidence",
    "D1TrainerHandoff",
    "attest_durable_d1_finalization",
]

_RESULT_TOKEN = object()
type D1TrainerHandoff = D1FreshRunCompletion | D1ResumeRunCompletion | D1TrainerFailureHandoff


@dataclass(frozen=True, slots=True)
class D1BillingFinalizationRequest:
    """Fully joined operator inputs before billing calculation or journal mutation."""

    plan: D1RunPlan
    segment_starts: tuple[RunStartProvenance, ...]
    segment_terminations: tuple[ExecutionSegmentTermination, ...]
    provider_billings: tuple[TerminalProviderBillingEvidence, ...]
    handoff: D1TrainerHandoff
    journal_path: Path
    expected_journal_state: JournalCheckpointState

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.plan), D1RunPlan):
            raise TypeError("D1 billing finalization requires a typed run plan")
        if not isinstance(cast(object, self.journal_path), Path):
            raise TypeError("D1 billing finalization journal_path must be a Path")
        if type(self.expected_journal_state) is not JournalCheckpointState:
            raise TypeError("expected_journal_state must be JournalCheckpointState")
        for label, values, admitted in (
            ("segment_starts", self.segment_starts, (RunStartProvenance,)),
            ("segment_terminations", self.segment_terminations, (ExecutionSegmentTermination,)),
            (
                "provider_billings",
                self.provider_billings,
                (ProviderBilling, ProviderBillingPartial, BillingGap),
            ),
        ):
            if type(values) is not tuple or any(type(value) not in admitted for value in values):
                raise TypeError(f"D1 {label} must be a tuple of typed evidence values")
        validate_execution_segments(self.segment_starts)
        if not (
            len(self.segment_starts)
            == len(self.segment_terminations)
            == len(self.provider_billings)
        ):
            raise ValueError("D1 finalization segment evidence counts differ")
        if type(self.handoff) not in (
            D1FreshRunCompletion,
            D1ResumeRunCompletion,
            D1TrainerFailureHandoff,
        ):
            raise TypeError("D1 billing finalization requires a typed trainer handoff")
        if self.handoff.operator_finalization_required is not True:
            raise ValueError("D1 trainer handoff does not require operator finalization")
        termination = self.handoff.termination
        if type(termination) is not D1TrainerTermination:
            raise TypeError("D1 trainer handoff requires a typed termination")
        if self.expected_journal_state != termination.journal_state:
            raise ValueError("expected journal state differs from the trainer handoff")
        if termination.logical_run_id != self.plan.logical_run_id:
            raise ValueError("D1 plan and trainer handoff logical identities differ")
        if termination.journal_state.execution_digest != self.plan.run.execution_digest:
            raise ValueError("D1 plan and trainer handoff execution identities differ")
        final = self.segment_terminations[-1]
        if (
            final.logical_run_id,
            final.execution_segment_index,
            final.execution_segment_id,
            final.trainer_ended_at,
        ) != (
            termination.logical_run_id,
            termination.execution_segment_index,
            termination.execution_segment_id,
            termination.trainer_ended_at,
        ):
            raise ValueError("final segment termination differs from the trainer handoff")
        if self.segment_starts[-1].execution_segment_id != termination.execution_segment_id:
            raise ValueError("final segment start differs from the trainer handoff")
        if (
            type(self.handoff) is D1ResumeRunCompletion
            and self.handoff.run_start_provenance != self.segment_starts[-1]
        ):
            raise ValueError("resume handoff start differs from final segment evidence")

    @property
    def termination(self) -> D1TrainerTermination:
        return self.handoff.termination

    @property
    def trainer_outcome(self) -> str:
        return "failed" if type(self.handoff) is D1TrainerFailureHandoff else "completed"


@dataclass(frozen=True, slots=True)
class D1TerminalResourceEvidence:
    """Terminal cumulative work with non-quality work explicitly derived."""

    resource_ledger_state: ResourceLedgerState
    final_eval_resource_work: ResourceWorkSnapshot | None
    tail_quarantine: JournalTailQuarantineReceipt | None
    non_quality_online_train_token_positions: int

    @classmethod
    def from_handoff(cls, handoff: D1TrainerHandoff) -> D1TerminalResourceEvidence:
        state = handoff.termination.resource_ledger_state
        non_quality = state.online_train_token_positions - state.committed_quality_model_tokens
        if non_quality < 0:
            raise ValueError("D1 resource ledger quality tokens exceed attempted online work")
        final_work: ResourceWorkSnapshot | None = None
        quarantine: JournalTailQuarantineReceipt | None = None
        if not isinstance(handoff, D1TrainerFailureHandoff):
            final_work = handoff.final_resource_work
            if type(final_work) is not ResourceWorkSnapshot:
                raise TypeError("successful D1 handoff requires final resource work")
            for field_name in (
                "committed_quality_source_bytes",
                "committed_quality_model_tokens",
                "online_train_token_positions",
                "target_train_token_positions",
                "baseline_calibration_positions",
                "eval_positions",
            ):
                if getattr(final_work, field_name) != getattr(state, field_name):
                    raise ValueError(
                        f"terminal resource snapshot and ledger differ at {field_name}"
                    )
            if type(handoff) is D1ResumeRunCompletion:
                quarantine = handoff.tail_quarantine
        return cls(state, final_work, quarantine, non_quality)

    @property
    def manifest(self) -> dict[str, object]:
        quarantine: dict[str, object] = {"kind": "none"}
        if self.tail_quarantine is not None:
            value = self.tail_quarantine
            quarantine = {
                "first_tail_event_sequence": value.first_tail_event_sequence,
                "kind": "quarantined_tail",
                "last_tail_event_sequence": value.last_tail_event_sequence,
                "quarantine_path": str(value.quarantine_path),
                "quarantined_journal_sha256": value.quarantined_journal_sha256,
                "recovered_prefix_sha256": value.recovered_prefix_sha256,
                "tail_event_count": value.tail_event_count,
            }
        return {
            "final_eval_resource_work": (
                None
                if self.final_eval_resource_work is None
                else self.final_eval_resource_work.manifest
            ),
            "non_quality_counter_evidence": EvidenceClass.DERIVED_FROM_MEASURED.value,
            "non_quality_online_train_token_positions": (
                self.non_quality_online_train_token_positions
            ),
            "resource_ledger_state": self.resource_ledger_state.manifest,
            "tail_quarantine": quarantine,
        }


@dataclass(frozen=True, slots=True)
class D1BillingFinalization:
    """Factory-attested result returned only after durable ``RUN_END`` append."""

    billing: LogicalBillingOutcome
    journal_state: JournalCheckpointState
    terminal_event: D1JournalEvent
    trainer_outcome: str
    quality_export_ready: bool
    _token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self._token is not _RESULT_TOKEN:
            raise TypeError("D1BillingFinalization is factory-only")
        if type(self.billing) not in (LogicalBillingReconciliation, LogicalBillingMissingCost):
            raise TypeError("D1 billing finalization requires a logical billing outcome")
        if type(self.journal_state) is not JournalCheckpointState:
            raise TypeError("D1 billing finalization requires journal state")
        if (
            type(self.terminal_event) is not D1JournalEvent
            or self.terminal_event.event_kind is not JournalEventKind.RUN_END
        ):
            raise ValueError("D1 billing finalization requires the durable run_end event")
        if self.journal_state.last_event_sequence != self.terminal_event.event_seq:
            raise ValueError("D1 finalized journal state does not receipt run_end")
        if self.trainer_outcome not in {"completed", "failed"}:
            raise ValueError("D1 trainer_outcome must be completed or failed")
        if type(self.quality_export_ready) is not bool:
            raise TypeError("D1 quality_export_ready must be bool")


def attest_durable_d1_finalization(
    *,
    billing: LogicalBillingOutcome,
    journal_state: JournalCheckpointState,
    terminal_event: D1JournalEvent,
    trainer_outcome: str,
    quality_export_ready: bool,
) -> D1BillingFinalization:
    """Construct the result after the caller's durable journal append succeeds."""
    return D1BillingFinalization(
        billing,
        journal_state,
        terminal_event,
        trainer_outcome,
        quality_export_ready,
        _RESULT_TOKEN,
    )
