"""Typed operator-finalization inputs reject unjoined or stale evidence."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import cast

import pytest

from d1_billing_finalizer_fixtures import completed_case, provider_bill
from d1_trainer_fixtures import micro_plan
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.capture.records.d1_journal import read_journal
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalEvent,
    JournalEventKind,
)
from entropic_substrate.ladder.capture.resource.resource_billing_reconciliation import (
    reconcile_logical_billing,
)
from entropic_substrate.ladder.run.closure.d1_billing_finalization_values import (
    D1BillingFinalization,
    D1BillingFinalizationRequest,
    D1TerminalResourceEvidence,
)
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import D1TrainerFailureHandoff

pytestmark = pytest.mark.unit


def _request(tmp_path: Path) -> D1BillingFinalizationRequest:
    case = completed_case(tmp_path)
    return D1BillingFinalizationRequest(
        case.plan,
        (case.start,),
        (case.termination,),
        (provider_bill(case.start),),
        case.completion,
        case.journal_path,
        case.completion.journal_state,
    )


def test_request_joins_exact_handoff_segment_plan_and_resource_evidence(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)

    assert request.termination == request.handoff.termination
    assert request.trainer_outcome == "completed"
    resources = D1TerminalResourceEvidence.from_handoff(request.handoff)
    assert resources.non_quality_online_train_token_positions == 0
    assert resources.manifest["non_quality_counter_evidence"] == "derived_from_measured"
    assert resources.manifest["tail_quarantine"] == {"kind": "none"}


@pytest.mark.parametrize("mismatch", ["state", "timestamp", "plan"])
def test_request_refuses_unjoined_prefix_time_and_plan(
    tmp_path: Path,
    mismatch: str,
) -> None:
    request = _request(tmp_path)
    state = request.expected_journal_state
    terminations = request.segment_terminations
    plan = request.plan
    if mismatch == "state":
        state = replace(state, prefix_sha256="f" * 64)
    elif mismatch == "timestamp":
        terminations = (
            replace(
                terminations[0],
                trainer_ended_at=terminations[0].trainer_ended_at + timedelta(seconds=1),
            ),
        )
    else:
        plan = micro_plan(D1Arm.AC)

    with pytest.raises(ValueError):
        replace(
            request,
            plan=plan,
            segment_terminations=terminations,
            expected_journal_state=state,
        )


def test_request_refuses_untyped_handoff(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(TypeError, match="handoff"):
        replace(request, handoff=cast(D1TrainerFailureHandoff, object()))


def test_finalized_result_cannot_be_constructed_without_factory_attestation(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    billing = reconcile_logical_billing(
        request.segment_starts,
        request.segment_terminations,
        request.provider_billings,
    )
    prior = read_journal(request.journal_path)[-1]
    terminal = D1JournalEvent(
        JournalEventKind.RUN_END,
        prior.event_seq + 1,
        prior.logical_run_id,
        prior.execution_digest,
        prior.aggregation_digest,
        prior.construction_digest,
        prior.family_digest,
        prior.run_class,
        prior.successful_step,
        {},
    )

    with pytest.raises(TypeError, match="factory-only"):
        D1BillingFinalization(
            billing,
            replace(
                request.expected_journal_state,
                last_event_sequence=terminal.event_seq,
            ),
            terminal,
            "completed",
            True,
            object(),
        )
