"""Fresh and resumed trainers durably bind typed terminal numerical defects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
from entropic_substrate.d1_evaluation import D1Evaluator
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_numerical_defect import (
    D1NonfiniteClassification,
    D1NonfiniteMetricError,
    d1_nonfinite_metric_error,
    terminal_numerical_defect_from_mapping,
)
from entropic_substrate.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.d1_resume_trainer import D1ResumeRunFailure, run_d1_resume
from entropic_substrate.d1_trainer_handoff import D1FreshRunFailure
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.data.batching import SequenceSampler, TokenSplit
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.experiments.d1_family import build_d1_experiment_family
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.predictive_state.initialization import SeededD1BranchInitializer
from entropic_substrate.predictive_state.runtime import D1PredictiveSystem

from .d1_runtime_fixtures import fit_runtime_baseline, runtime_family

pytestmark = pytest.mark.unit


def _typed_error(value: float = float("nan")) -> D1NonfiniteMetricError:
    return d1_nonfinite_metric_error(
        metric_name="nats_per_byte",
        value=value,
        source_exception_code="d1.eval.nats_per_byte.nonfinite.v1",
    )


def test_evaluator_raises_typed_metric_defect_at_the_aggregate_guard() -> None:
    template = runtime_family()
    draft = template.drafts[0]
    family = build_d1_experiment_family(
        codec=draft.codec,
        encoder_pair=draft.encoder_pair,
        execution_protocol=draft.execution_protocol,
    )
    system = D1PredictiveSystem.from_family(
        family, D1Arm.BOTH, initializer=SeededD1BranchInitializer(seed=7)
    )
    fit_runtime_baseline(system)
    protocol = draft.execution_protocol.batching
    sampler = SequenceSampler(
        TokenSplit(np.arange(4096, dtype=np.int32) % 11),
        batch_size=protocol.batch_size,
        seq_len=protocol.sequence_length,
        train=False,
    )

    def poison_language(_module: object, _args: object, output: object) -> object:
        output.objective.language.fill_(float("nan"))  # type: ignore[attr-defined]
        return output

    handle = system.register_forward_hook(poison_language)
    try:
        with pytest.raises(D1NonfiniteMetricError) as captured:
            D1Evaluator().evaluate(system=system, sampler=sampler)
    finally:
        handle.remove()
    assert captured.value.metric_name == "nats_per_byte"
    assert captured.value.classification is D1NonfiniteClassification.NAN
    assert captured.value.source_exception_code == "d1.eval.nats_per_byte.nonfinite.v1"


def test_fresh_terminal_defect_is_nested_without_nonfinite_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = micro_plan(D1Arm.BOTH)
    journal, checkpoint = output_paths(tmp_path, D1Arm.BOTH)
    real_eval = fresh_module._record_evaluation  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def fail_terminal(**kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise _typed_error(float("inf"))
        return real_eval(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fresh_module, "_record_evaluation", fail_terminal)
    with pytest.raises(D1FreshRunFailure) as captured:
        run_d1_fresh(
            plan=plan,
            start=run_start(plan),
            origin=D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS),
            locations=D1FreshRunLocations(journal, checkpoint),
            clock=DeterministicTrainerClock(),
        )
    events = read_journal(journal)
    assert type(captured.value.__cause__) is D1NonfiniteMetricError
    assert events[-1].event_kind is JournalEventKind.TRAINER_ABORTED
    record = terminal_numerical_defect_from_mapping(events[-1].payload["numerical_defect"])
    prior_receipt = [event for event in events if event.event_kind is JournalEventKind.CHECKPOINT][
        -1
    ]
    assert record.successful_step == record.max_steps == 2
    assert record.eval_index == 2
    assert record.checkpoint.sha256 == prior_receipt.payload["checkpoint_sha256"]
    assert record.execution_segment_index == 0
    assert b"NaN" not in journal.read_bytes() and b"Infinity" not in journal.read_bytes()


def test_intermediate_nonfinite_error_does_not_claim_terminal_defect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = micro_plan(D1Arm.NONE)
    journal, checkpoint = output_paths(tmp_path, D1Arm.NONE)
    real_eval = fresh_module._record_evaluation  # pyright: ignore[reportPrivateUsage]
    calls = 0

    def fail_intermediate(**kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise _typed_error()
        return real_eval(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(fresh_module, "_record_evaluation", fail_intermediate)
    with pytest.raises(D1FreshRunFailure):
        run_d1_fresh(
            plan=plan,
            start=run_start(plan),
            origin=D1TrainerWallOrigin(started_at=STARTED_AT, monotonic_ns=ORIGIN_NS),
            locations=D1FreshRunLocations(journal, checkpoint),
            clock=DeterministicTrainerClock(),
        )
    assert "numerical_defect" not in read_journal(journal)[-1].payload


def test_resumed_terminal_defect_binds_admission_checkpoint_and_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)

    def fail_terminal(*_args: object, **_kwargs: object) -> None:
        raise _typed_error(float("-inf"))

    monkeypatch.setattr(resume_module, "_evaluate", fail_terminal)
    with pytest.raises(D1ResumeRunFailure):
        run_d1_resume(
            plan=case.plan,
            start=resume_start(case.plan, case.checkpoint_sha256),
            origin=origin(),
            locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
            observer=RecordingObserver(),
            clock=DeterministicTrainerClock(),
        )
    event = read_journal(case.journal_path)[-1]
    record = terminal_numerical_defect_from_mapping(event.payload["numerical_defect"])
    assert record.classification is D1NonfiniteClassification.NEGATIVE_INFINITY
    assert record.checkpoint.sha256 == case.checkpoint_sha256
    assert record.execution_segment_index == 1
    assert record.eval_index == 2
