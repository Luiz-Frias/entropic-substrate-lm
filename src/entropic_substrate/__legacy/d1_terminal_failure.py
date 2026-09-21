"""Join a typed numeric source observation to the trainer lifecycle cursor."""

from __future__ import annotations

from entropic_substrate.d1_numerical_defect import (
    D1DefectCheckpointReference,
    D1NonfiniteMetricError,
    D1TerminalNumericalDefectRecord,
)
from entropic_substrate.evidence.resource_ledger import ResourceLedgerState
from entropic_substrate.evidence.resource_provenance import RunStartProvenance
from entropic_substrate.experiments.d1_plan import D1RunPlan

__all__ = ["terminal_numerical_defect_for_failure"]


def terminal_numerical_defect_for_failure(
    error: Exception,
    *,
    plan: D1RunPlan,
    start: RunStartProvenance,
    resource_state: ResourceLedgerState,
    successful_step: int,
    checkpoint_sha256: str | None,
) -> D1TerminalNumericalDefectRecord | None:
    """Enrich only a typed defect observed at the registered terminal cursor."""
    if type(error) is not D1NonfiniteMetricError:
        return None
    maximum = plan.profile.draft.execution_protocol.optimization.max_steps
    if successful_step != maximum:
        return None
    if resource_state.logical_run_id != plan.logical_run_id:
        raise ValueError("terminal defect resource ledger belongs to another run")
    if resource_state.execution_segment_index != start.execution_segment_index:
        raise ValueError("terminal defect resource ledger belongs to another segment")
    checkpoint = (
        D1DefectCheckpointReference.receipt(checkpoint_sha256)
        if checkpoint_sha256 is not None
        else D1DefectCheckpointReference.gap("no_prior_checkpoint")
    )
    return D1TerminalNumericalDefectRecord.from_error(
        error,
        plan=plan,
        start=start,
        successful_step=successful_step,
        eval_index=resource_state.next_eval_index,
        checkpoint=checkpoint,
    )
