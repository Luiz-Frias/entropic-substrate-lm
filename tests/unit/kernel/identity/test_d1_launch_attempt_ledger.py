"""TDD contract for append-only D1 launch-attempt evidence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from entropic_substrate.kernel.identity.d1_primitives import D1Arm, RunClass
from entropic_substrate.ladder.register.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)

pytestmark = pytest.mark.unit
NOW = datetime(2026, 8, 24, 15, 0, tzinfo=UTC)


def identity(
    *,
    run_class: RunClass = RunClass.CONFIRMATORY,
    logical_run_id: str = "d1-confirmatory-sg-s101",
    seed: int = 101,
) -> D1LaunchAttemptIdentity:
    return D1LaunchAttemptIdentity(
        run_class=run_class,
        family_digest="a" * 64,
        arm=D1Arm.SG,
        seed=seed,
        logical_run_id=logical_run_id,
        registered_design_digest=("b" * 64 if run_class is RunClass.CONFIRMATORY else None),
    )


def test_ledger_is_immutable_append_only_and_retains_failed_pilot_attempts() -> None:
    empty = D1LaunchAttemptLedger()
    pilot = identity(run_class=RunClass.PILOT, logical_run_id="d1-pilot-sg-s101")
    recorded = empty.record(pilot, observed_at=NOW)
    failed = recorded.transition(
        pilot,
        status=D1LaunchAttemptStatus.FAILED,
        observed_at=NOW + timedelta(seconds=1),
        reason="provider rejected placement",
    )
    assert empty.events == ()
    assert len(recorded.events) == 1
    assert [event.status.value for event in failed.events] == ["recorded", "failed"]
    assert failed.events[-1].reason == "provider rejected placement"
    assert failed.events[-1].identity.run_class is RunClass.PILOT
    assert len(failed.ledger_digest) == 64


def test_renaming_a_run_cannot_hide_a_duplicate_optional_stopping_identity() -> None:
    first = identity()
    ledger = D1LaunchAttemptLedger().record(first, observed_at=NOW)
    renamed = identity(logical_run_id="please-pretend-this-is-new")
    assert renamed.optional_stopping_digest == first.optional_stopping_digest
    with pytest.raises(ValueError, match="optional-stopping"):
        ledger.record(renamed, observed_at=NOW + timedelta(seconds=1))


def test_confirmatory_identity_requires_design_while_pilot_forbids_it() -> None:
    with pytest.raises(ValueError, match="registered_design_digest"):
        D1LaunchAttemptIdentity(
            run_class=RunClass.CONFIRMATORY,
            family_digest="a" * 64,
            arm=D1Arm.NONE,
            seed=1,
            logical_run_id="confirmatory",
            registered_design_digest=None,
        )
    with pytest.raises(ValueError, match="pilot"):
        D1LaunchAttemptIdentity(
            run_class=RunClass.PILOT,
            family_digest="a" * 64,
            arm=D1Arm.NONE,
            seed=1,
            logical_run_id="pilot",
            registered_design_digest="b" * 64,
        )


def test_attempt_state_machine_requires_registration_monotonic_time_and_terminality() -> None:
    item = identity()
    empty = D1LaunchAttemptLedger()
    with pytest.raises(ValueError, match="unregistered"):
        empty.transition(
            item,
            status=D1LaunchAttemptStatus.TRAINER_STARTED,
            observed_at=NOW,
        )

    recorded = empty.record(item, observed_at=NOW)
    with pytest.raises(ValueError, match="precede"):
        recorded.transition(
            item,
            status=D1LaunchAttemptStatus.TRAINER_STARTED,
            observed_at=NOW - timedelta(seconds=1),
        )
    abandoned = recorded.transition(
        item,
        status=D1LaunchAttemptStatus.ABANDONED,
        observed_at=NOW + timedelta(seconds=1),
        reason="operator declined price",
    )
    with pytest.raises(ValueError, match="terminal"):
        abandoned.transition(
            item,
            status=D1LaunchAttemptStatus.TRAINER_STARTED,
            observed_at=NOW + timedelta(seconds=2),
        )


def test_failure_and_abandonment_require_reasons_but_normal_progress_forbids_one() -> None:
    item = identity()
    ledger = D1LaunchAttemptLedger().record(item, observed_at=NOW)
    with pytest.raises(ValueError, match="reason"):
        ledger.transition(
            item,
            status=D1LaunchAttemptStatus.FAILED,
            observed_at=NOW + timedelta(seconds=1),
        )
    with pytest.raises(ValueError, match="reason"):
        ledger.transition(
            item,
            status=D1LaunchAttemptStatus.TRAINER_STARTED,
            observed_at=NOW + timedelta(seconds=1),
            reason="should not exist",
        )
