"""Terminal meter-refresh defects never replace the primary D1 trainer failure."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest

import entropic_substrate.d1_fresh_trainer as fresh_module
import entropic_substrate.d1_resume_trainer as resume_module
from d1_resume_fixtures import RecordingObserver, captured_step_one_case, origin, resume_start
from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    output_paths,
    run_start,
)
from entropic_substrate.d1_dual_failure import (
    D1PrimarySecondaryFailure,
    secondary_failure_from_mapping,
)
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_numerical_defect import (
    D1NonfiniteMetricError,
    d1_nonfinite_metric_error,
    terminal_numerical_defect_from_mapping,
)
from entropic_substrate.d1_resource_activation import D1ResourceActivation
from entropic_substrate.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.d1_resume_trainer import D1ResumeRunFailure, run_d1_resume
from entropic_substrate.d1_trainer_handoff import D1FreshRunFailure
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.resource_ledger import (
    ResourceLedgerState,
    ResourceMeterSnapshot,
    UnavailableMeterStrategy,
)
from entropic_substrate.evidence.resource_meter_models import ResourcePhase
from entropic_substrate.evidence.resource_provenance import (
    RunStartProvenance,
    unavailable_resource_activation_registration,
)
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


class _Lifecycle:
    registration = None
    sample_source = None

    def __enter__(self) -> _Lifecycle:
        return self

    def __exit__(self, *_: object) -> None:
        return None


class _RefreshFailMeter:
    def __init__(self, *, successful_snapshots: int) -> None:
        self.successful_snapshots = successful_snapshots
        self.snapshot_calls = 0

    @contextmanager
    def measure(self, phase: ResourcePhase) -> Generator[None]:
        assert type(phase) is ResourcePhase
        yield

    def record_positions(
        self, phase: ResourcePhase, *, online_positions: int, target_positions: int
    ) -> None:
        assert type(phase) is ResourcePhase
        assert online_positions >= 0 and target_positions >= 0

    def snapshot(self, *, trainer_wall_seconds: float) -> ResourceMeterSnapshot:
        self.snapshot_calls += 1
        if self.snapshot_calls > self.successful_snapshots:
            raise RuntimeError("terminal meter refresh failed")
        return UnavailableMeterStrategy().snapshot(trainer_wall_seconds=trainer_wall_seconds)


class _Factory:
    registration = unavailable_resource_activation_registration()

    def __init__(self, meter: _RefreshFailMeter) -> None:
        self.meter = meter

    def fresh(self, start: RunStartProvenance) -> D1ResourceActivation:
        assert start.resource_activation == self.registration
        return D1ResourceActivation(self.meter, _Lifecycle())

    def resume(self, start: RunStartProvenance, state: ResourceLedgerState) -> D1ResourceActivation:
        assert start.resource_activation == self.registration
        assert type(state) is ResourceLedgerState
        return D1ResourceActivation(self.meter, _Lifecycle())


def _nonfinite() -> D1NonfiniteMetricError:
    return d1_nonfinite_metric_error(
        metric_name="nats_per_byte",
        value=float("nan"),
        source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
    )


def _assert_dual(
    cause: BaseException | None,
    payload: object,
    *,
    primary_type: str,
) -> None:
    assert type(cause) is D1PrimarySecondaryFailure
    assert type(cause.primary).__name__ == primary_type  # type: ignore[union-attr]
    assert type(cause.secondary) is RuntimeError  # type: ignore[union-attr]
    assert isinstance(payload, Mapping)
    assert payload["error_type"] == primary_type
    secondary = secondary_failure_from_mapping(payload["secondary_failure"])
    assert secondary.error_type == "RuntimeError"
    assert secondary.error_message == "terminal meter refresh failed"


def test_fresh_generic_dual_failure_is_journaled_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = micro_plan(D1Arm.NONE)
    journal, checkpoint = output_paths(tmp_path, D1Arm.NONE)
    meter = _RefreshFailMeter(successful_snapshots=1)

    def fail_step(**_kwargs: object) -> object:
        raise RuntimeError("primary train failure")

    monkeypatch.setattr(fresh_module, "apply_d1_training_step", fail_step)
    with pytest.raises(D1FreshRunFailure) as captured:
        run_d1_fresh(
            plan=plan,
            start=run_start(plan),
            origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
            locations=D1FreshRunLocations(journal, checkpoint),
            clock=DeterministicTrainerClock(),
            resource_activation_factory=_Factory(meter),
        )

    event = read_journal(journal)[-1]
    _assert_dual(captured.value.__cause__, event.payload, primary_type="RuntimeError")
    assert captured.value.handoff.secondary_failure is not None


def test_fresh_terminal_numerical_defect_survives_secondary_meter_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = micro_plan(D1Arm.BOTH)
    journal, checkpoint = output_paths(tmp_path, D1Arm.BOTH)
    meter = _RefreshFailMeter(successful_snapshots=2)
    real_eval = fresh_module._record_evaluation  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def fail_terminal(**kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise _nonfinite()
        return real_eval(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fresh_module, "_record_evaluation", fail_terminal)
    with pytest.raises(D1FreshRunFailure) as captured:
        run_d1_fresh(
            plan=plan,
            start=run_start(plan),
            origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
            locations=D1FreshRunLocations(journal, checkpoint),
            clock=DeterministicTrainerClock(),
            resource_activation_factory=_Factory(meter),
        )

    event = read_journal(journal)[-1]
    _assert_dual(captured.value.__cause__, event.payload, primary_type="D1NonfiniteMetricError")
    terminal_numerical_defect_from_mapping(event.payload["numerical_defect"])


@pytest.mark.parametrize("numerical", [False, True])
def test_resumed_dual_failure_preserves_primary_and_secondary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    numerical: bool,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    meter = _RefreshFailMeter(successful_snapshots=0)
    if numerical:
        monkeypatch.setattr(
            resume_module, "_evaluate", lambda *_args: (_ for _ in ()).throw(_nonfinite())
        )
        primary_type = "D1NonfiniteMetricError"
    else:
        monkeypatch.setattr(
            resume_module,
            "apply_d1_training_step",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("primary resumed failure")),
        )
        primary_type = "RuntimeError"

    with pytest.raises(D1ResumeRunFailure) as captured:
        run_d1_resume(
            plan=case.plan,
            start=resume_start(case.plan, case.checkpoint_sha256),
            origin=origin(),
            locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
            observer=RecordingObserver(),
            clock=DeterministicTrainerClock(),
            resource_activation_factory=_Factory(meter),
        )

    event = read_journal(case.journal_path)[-1]
    _assert_dual(captured.value.__cause__, event.payload, primary_type=primary_type)
    assert captured.value.handoff.secondary_failure is not None
    if numerical:
        terminal_numerical_defect_from_mapping(event.payload["numerical_defect"])
