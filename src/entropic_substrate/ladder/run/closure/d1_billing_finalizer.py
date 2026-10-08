"""Operator-side D1 billing reconciliation and authoritative run closure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from entropic_substrate.kernel.evidence_class.resource_provenance import EvidenceGap
from entropic_substrate.kernel.evidence_class.resource_provenance_codec import run_start_to_mapping
from entropic_substrate.ladder.capture.records.d1_journal import (
    AuthoritativeRunJournal,
    read_journal,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    FrozenJson,
    JournalEventKind,
)
from entropic_substrate.ladder.capture.resource.resource_billing_reconciliation import (
    LogicalBillingReconciliation,
    reconcile_logical_billing,
)
from entropic_substrate.ladder.run.closure.d1_billing_finalization_values import (
    D1BillingFinalization,
    D1BillingFinalizationRequest,
    D1TerminalResourceEvidence,
    D1TrainerHandoff,
    attest_durable_d1_finalization,
)
from entropic_substrate.ladder.run.trainer.checkpoint_repository import (
    D1CheckpointArtifactReceipt,
    load_checkpoint_candidate,
)
from entropic_substrate.ladder.run.trainer.checkpoint_values import (
    D1_CHECKPOINT_KIND,
    D1_CHECKPOINT_SCHEMA_VERSION,
)
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import (
    D1FreshRunCompletion,
    D1TrainerFailureHandoff,
)

__all__ = ["D1BillingFinalization", "finalize_d1_billing"]

_FINALIZATION_SCHEMA = "d1-operator-finalization-v1"


def _gap_manifest(gap: EvidenceGap) -> dict[str, object]:
    return {"field": gap.field, "kind": "gap", "reason": gap.reason}


def _validate_plan_and_journal(
    request: D1BillingFinalizationRequest,
) -> tuple[D1JournalEvent, ...]:
    events = read_journal(request.journal_path)
    plan = request.plan
    identity = events[0].identity
    expected_identity = (
        plan.logical_run_id,
        plan.run.execution_digest,
        plan.run.aggregation_digest,
        plan.run.construction_digest,
        plan.profile.family.family_digest,
        plan.run.run_class,
    )
    actual_identity = (
        identity.logical_run_id,
        identity.execution_digest,
        identity.aggregation_digest,
        identity.construction_digest,
        identity.family_digest,
        identity.run_class,
    )
    if actual_identity != expected_identity:
        raise ValueError("D1 plan and journal identities differ")
    if events[0].payload.get("plan_digest") != plan.plan_digest:
        raise ValueError("D1 journal plan digest differs from the supplied plan")
    observed_starts = [events[0].payload.get("run_start_provenance")]
    observed_starts.extend(
        event.payload.get("run_start_provenance")
        for event in events
        if event.event_kind is JournalEventKind.RUN_RESUME
    )
    expected_starts = [run_start_to_mapping(start) for start in request.segment_starts]
    if observed_starts != expected_starts:
        raise ValueError("D1 segment starts differ from authoritative journal evidence")
    expected_tail = (
        JournalEventKind.TRAINER_ABORTED
        if request.trainer_outcome == "failed"
        else JournalEventKind.CHECKPOINT
    )
    if events[-1].event_kind is not expected_tail:
        raise ValueError("D1 journal tail differs from the trainer handoff outcome")
    return events


def _checkpoint_export(
    request: D1BillingFinalizationRequest,
    events: tuple[D1JournalEvent, ...],
) -> tuple[dict[str, object], bool]:
    handoff = request.handoff
    if isinstance(handoff, D1TrainerFailureHandoff):
        return (
            _gap_manifest(
                EvidenceGap(
                    "terminal_checkpoint",
                    "trainer failure has no successful terminal checkpoint handoff",
                )
            ),
            False,
        )
    receipt = handoff.final_checkpoint
    if type(receipt) is not D1CheckpointArtifactReceipt:
        raise TypeError("successful D1 handoff requires a checkpoint receipt")
    terminal_step = request.plan.profile.draft.execution_protocol.optimization.max_steps
    if handoff.successful_step != terminal_step:
        raise ValueError("successful D1 handoff did not reach the registered terminal step")
    if events[-1].payload.get("checkpoint_sha256") != receipt.sha256:
        raise ValueError("terminal journal checkpoint differs from the trainer receipt")
    candidate, actual = load_checkpoint_candidate(receipt.path, expected_sha256=receipt.sha256)
    if actual != receipt:
        raise ValueError("terminal checkpoint receipt path, size, or digest changed")
    if (
        candidate.get("kind") != D1_CHECKPOINT_KIND
        or candidate.get("schema_version") != D1_CHECKPOINT_SCHEMA_VERSION
        or candidate.get("rung") != "d1"
        or candidate.get("successful_step") != handoff.successful_step
    ):
        raise ValueError("terminal checkpoint envelope is not export-ready D1 evidence")
    identity = candidate.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("terminal checkpoint identity is missing")
    expected_identity = {
        "execution_digest": request.plan.run.execution_digest,
        "aggregation_digest": request.plan.run.aggregation_digest,
        "construction_digest": request.plan.run.construction_digest,
        "family_digest": request.plan.profile.family.family_digest,
    }
    if any(identity.get(name) != value for name, value in expected_identity.items()):
        raise ValueError("terminal checkpoint identity differs from the run plan")
    return (
        {
            "kind": "verified_terminal_checkpoint",
            "path": str(receipt.path),
            "sha256": receipt.sha256,
            "size_bytes": receipt.size_bytes,
            "successful_step": handoff.successful_step,
        },
        True,
    )


def _handoff_manifest(handoff: D1TrainerHandoff) -> dict[str, object]:
    if type(handoff) is D1TrainerFailureHandoff:
        return handoff.manifest
    return {
        "kind": "fresh_completion"
        if type(handoff) is D1FreshRunCompletion
        else "resume_completion",
        "operator_finalization_required": handoff.operator_finalization_required,
        "termination": handoff.termination.manifest,
    }


def finalize_d1_billing(request: D1BillingFinalizationRequest) -> D1BillingFinalization:
    """Reconcile every physical segment, then durably close one logical D1 run."""
    events = _validate_plan_and_journal(request)
    billing = reconcile_logical_billing(
        request.segment_starts,
        request.segment_terminations,
        request.provider_billings,
    )
    checkpoint, quality_ready = _checkpoint_export(request, events)
    payload = cast(
        Mapping[str, FrozenJson],
        {
            "billing": billing.manifest,
            "checkpoint_export": checkpoint,
            "provider_billing_complete": type(billing) is LogicalBillingReconciliation,
            "quality_export_ready": quality_ready,
            "schema_version": _FINALIZATION_SCHEMA,
            "terminal_resource_evidence": D1TerminalResourceEvidence.from_handoff(
                request.handoff
            ).manifest,
            "trainer_handoff": _handoff_manifest(request.handoff),
            "trainer_outcome": request.trainer_outcome,
        },
    )
    with AuthoritativeRunJournal.open_for_finalization(
        request.journal_path,
        request.expected_journal_state,
    ) as journal:
        state = journal.state
        terminal = D1JournalEvent(
            event_kind=JournalEventKind.RUN_END,
            event_seq=state.last_event_sequence + 1,
            logical_run_id=request.plan.logical_run_id,
            execution_digest=request.plan.run.execution_digest,
            aggregation_digest=request.plan.run.aggregation_digest,
            construction_digest=request.plan.run.construction_digest,
            family_digest=request.plan.profile.family.family_digest,
            run_class=request.plan.run.run_class,
            successful_step=request.termination.successful_step,
            payload=payload,
        )
        finalized_state = journal.append(terminal)
    return attest_durable_d1_finalization(
        billing=billing,
        journal_state=finalized_state,
        terminal_event=terminal,
        trainer_outcome=request.trainer_outcome,
        quality_export_ready=quality_ready,
    )
