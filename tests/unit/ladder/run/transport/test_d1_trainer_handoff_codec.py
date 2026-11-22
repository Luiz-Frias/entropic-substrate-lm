"""Fresh, resume, and durable failure cross the trainer process boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from d1_billing_finalizer_fixtures import CompletedD1Case, completed_case
from d1_resume_fixtures import (
    RecordingObserver,
    captured_step_one_case,
    origin,
    resume_start,
)
from d1_resume_hard_crash_fixture import hard_crash_tail_case
from d1_trainer_fixtures import DeterministicTrainerClock
from entropic_substrate.ladder.run.trainer.d1_dual_failure import (
    D1SecondaryFailureEvidence,
    D1SecondaryFailureStage,
)
from entropic_substrate.ladder.run.trainer.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.ladder.run.trainer.d1_resume_trainer import run_d1_resume
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import D1TrainerFailureHandoff
from entropic_substrate.ladder.run.transport.d1_trainer_handoff_codec import (
    parse_d1_trainer_handoff,
    serialize_d1_trainer_handoff,
    trainer_handoff_from_mapping,
    trainer_handoff_to_mapping,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def case(tmp_path: Path) -> CompletedD1Case:
    return completed_case(tmp_path)


def test_fresh_completion_round_trips_every_value_and_rebinds_checkpoint(
    case: CompletedD1Case,
    tmp_path: Path,
) -> None:
    payload = serialize_d1_trainer_handoff(case.completion, artifact_root=tmp_path)
    rebound_root = tmp_path / "rebound"
    rebound_root.mkdir()
    rebound_checkpoint = rebound_root / case.completion.final_checkpoint.path.name
    rebound_checkpoint.write_bytes(case.completion.final_checkpoint.path.read_bytes())

    observed = parse_d1_trainer_handoff(payload, artifact_root=rebound_root)

    expected_receipt = replace(case.completion.final_checkpoint, path=rebound_checkpoint)
    assert observed == replace(case.completion, final_checkpoint=expected_receipt)
    assert observed.final_checkpoint.path == rebound_checkpoint
    mapping = json.loads(payload)
    assert mapping["kind"] == "fresh_completion"
    assert mapping["termination"]["model_successful_step"] == 2
    assert mapping["termination"]["schema_version"] == "d1-trainer-termination-v2"
    assert mapping["final_checkpoint"]["path"] == "none.pt"
    assert mapping["final_evaluation"]["schema_version"] == "d1-evaluation-wire-v2"
    assert mapping["final_evaluation"]["state_sample_covariance"]
    assert mapping["final_resource_work"]["event"] == "resource.work_snapshot"
    assert (
        trainer_handoff_from_mapping(
            trainer_handoff_to_mapping(case.completion, artifact_root=tmp_path),
            artifact_root=tmp_path,
        )
        == case.completion
    )


def test_failure_round_trips_primary_secondary_and_termination_authority(
    case: CompletedD1Case,
    tmp_path: Path,
) -> None:
    secondary = D1SecondaryFailureEvidence.from_error(
        D1SecondaryFailureStage.TERMINAL_METER_REFRESH,
        RuntimeError("meter stopped"),
    )
    failure = D1TrainerFailureHandoff.from_error(
        case.completion.termination,
        RuntimeError("\ntraining stopped\n"),
        secondary_failure=secondary,
    )

    payload = serialize_d1_trainer_handoff(failure, artifact_root=tmp_path)
    observed = parse_d1_trainer_handoff(payload, artifact_root=tmp_path)

    assert observed == failure
    assert observed.error_message == "\ntraining stopped\n"
    assert observed.secondary_failure == secondary
    assert json.loads(payload)["kind"] == "trainer_failure"


def test_failure_codec_refuses_abort_append_stage_as_durable_handoff(
    case: CompletedD1Case,
    tmp_path: Path,
) -> None:
    secondary = D1SecondaryFailureEvidence.from_error(
        D1SecondaryFailureStage.TERMINAL_METER_REFRESH,
        RuntimeError("meter stopped"),
    )
    failure = D1TrainerFailureHandoff.from_error(
        case.completion.termination,
        RuntimeError("training stopped"),
        secondary_failure=secondary,
    )
    value = trainer_handoff_to_mapping(failure, artifact_root=tmp_path)
    cast(dict[str, object], value["secondary_failure"])["stage"] = "trainer-abort-append"

    with pytest.raises(ValueError, match="durable TRAINER_ABORTED"):
        trainer_handoff_from_mapping(value, artifact_root=tmp_path)


def test_failure_codec_refuses_eval_ledger_ahead_of_durable_cursor(
    case: CompletedD1Case,
    tmp_path: Path,
) -> None:
    failure = D1TrainerFailureHandoff.from_error(
        case.completion.termination,
        RuntimeError("training stopped"),
    )
    value = trainer_handoff_to_mapping(failure, artifact_root=tmp_path)
    termination = cast(dict[str, object], value["termination"])
    termination["successful_step"] = 1
    cast(dict[str, object], termination["journal_state"])["successful_step"] = 1

    with pytest.raises(ValueError, match="durable successful_step"):
        trainer_handoff_from_mapping(value, artifact_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"extra": True}), "keys"),
        (lambda value: value.update({"kind": "invented"}), "kind"),
        (
            lambda value: cast(dict[str, object], value["final_checkpoint"]).update(
                {"path": "../escape.pt"}
            ),
            "path",
        ),
        (
            lambda value: cast(dict[str, object], value["final_checkpoint"]).update(
                {"size_bytes": 1}
            ),
            "checkpoint bytes",
        ),
        (
            lambda value: cast(dict[str, object], value["final_checkpoint"]).update(
                {"sha256": "0" * 64}
            ),
            "checkpoint bytes",
        ),
        (
            lambda value: cast(dict[str, object], value["termination"]).update(
                {"successful_step": 1}
            ),
            "termination",
        ),
        (
            lambda value: cast(dict[str, object], value["termination"]).pop(
                "model_successful_step"
            ),
            "keys",
        ),
        (
            lambda value: cast(dict[str, object], value["termination"]).update(
                {"model_successful_step": 4}
            ),
            "termination",
        ),
        (
            lambda value: cast(
                dict[str, object],
                cast(dict[str, object], value["termination"])["resource_ledger_state"],
            ).update({"event": "wrong"}),
            "termination",
        ),
    ],
)
def test_mapping_refuses_union_schema_path_and_termination_drift(
    case: CompletedD1Case,
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    value = trainer_handoff_to_mapping(case.completion, artifact_root=tmp_path)
    mutation(value)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        trainer_handoff_from_mapping(value, artifact_root=tmp_path)


def test_parser_refuses_duplicate_nonfinite_and_noncanonical_json(
    case: CompletedD1Case,
    tmp_path: Path,
) -> None:
    canonical = serialize_d1_trainer_handoff(case.completion, artifact_root=tmp_path)
    duplicate = canonical.replace(b'{"data_order":', b'{"kind":"x","data_order":', 1)
    nonfinite_value = json.loads(canonical)
    nonfinite_value["final_evaluation"]["future_nll"] = float("nan")
    nonfinite = json.dumps(
        nonfinite_value,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    indented = json.dumps(json.loads(canonical), indent=2).encode()

    with pytest.raises(ValueError, match="duplicate"):
        parse_d1_trainer_handoff(duplicate, artifact_root=tmp_path)
    with pytest.raises(ValueError, match="non-finite"):
        parse_d1_trainer_handoff(nonfinite, artifact_root=tmp_path)
    with pytest.raises(ValueError, match="canonical"):
        parse_d1_trainer_handoff(indented, artifact_root=tmp_path)


def test_checkpoint_byte_drift_fails_encode_and_decode(
    case: CompletedD1Case,
    tmp_path: Path,
) -> None:
    checkpoint = case.completion.final_checkpoint.path
    original = checkpoint.read_bytes()
    mapping = trainer_handoff_to_mapping(case.completion, artifact_root=tmp_path)
    checkpoint.write_bytes(b"x" * len(original))

    with pytest.raises(ValueError, match="checkpoint bytes"):
        trainer_handoff_from_mapping(mapping, artifact_root=tmp_path)
    with pytest.raises(ValueError, match="checkpoint bytes"):
        serialize_d1_trainer_handoff(case.completion, artifact_root=tmp_path)


def test_hydration_refuses_symlinked_root_and_checkpoint(
    case: CompletedD1Case,
    tmp_path: Path,
) -> None:
    mapping = trainer_handoff_to_mapping(case.completion, artifact_root=tmp_path)
    checkpoint_link = tmp_path / "checkpoint-link.pt"
    checkpoint_link.symlink_to(case.completion.final_checkpoint.path)
    cast(dict[str, object], mapping["final_checkpoint"])["path"] = checkpoint_link.name

    with pytest.raises(ValueError, match="symlink"):
        trainer_handoff_from_mapping(mapping, artifact_root=tmp_path)

    root_link = tmp_path.parent / f"{tmp_path.name}-link"
    root_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="root"):
        parse_d1_trainer_handoff(
            serialize_d1_trainer_handoff(case.completion, artifact_root=tmp_path),
            artifact_root=root_link,
        )


def test_resume_completion_without_tail_round_trips_under_a_new_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted = captured_step_one_case(tmp_path, monkeypatch)
    completion = run_d1_resume(
        plan=interrupted.plan,
        start=resume_start(interrupted.plan, interrupted.checkpoint_sha256),
        origin=origin(),
        locations=D1ResumeRunLocations(
            interrupted.journal_path,
            interrupted.checkpoint_path,
        ),
        observer=RecordingObserver(),
        clock=DeterministicTrainerClock(),
    )
    payload = serialize_d1_trainer_handoff(completion, artifact_root=tmp_path)
    rebound_root = tmp_path / "rebound-resume"
    rebound_root.mkdir()
    rebound_checkpoint = rebound_root / completion.final_checkpoint.path.name
    rebound_checkpoint.write_bytes(completion.final_checkpoint.path.read_bytes())

    observed = parse_d1_trainer_handoff(payload, artifact_root=rebound_root)

    expected_receipt = replace(completion.final_checkpoint, path=rebound_checkpoint)
    assert observed == replace(completion, final_checkpoint=expected_receipt)
    mapping = json.loads(payload)
    assert mapping["kind"] == "resume_completion"
    assert mapping["resume"]["tail_quarantine"] is None
    assert mapping["resume"]["resumed_from_sha256"] == interrupted.checkpoint_sha256


def test_resume_quarantine_binds_artifact_prefix_and_tail_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interrupted = hard_crash_tail_case(tmp_path, monkeypatch)
    quarantine_path = tmp_path / "crashed-tail.jsonl"
    completion = run_d1_resume(
        plan=interrupted.plan,
        start=resume_start(interrupted.plan, interrupted.checkpoint_sha256),
        origin=origin(),
        locations=D1ResumeRunLocations(
            interrupted.journal_path,
            interrupted.checkpoint_path,
            quarantine_path,
        ),
        observer=RecordingObserver(),
        clock=DeterministicTrainerClock(),
    )
    payload = serialize_d1_trainer_handoff(completion, artifact_root=tmp_path)
    rebound_root = tmp_path / "rebound-quarantine"
    rebound_root.mkdir()
    rebound_checkpoint = rebound_root / completion.final_checkpoint.path.name
    rebound_checkpoint.write_bytes(completion.final_checkpoint.path.read_bytes())
    rebound_quarantine = rebound_root / quarantine_path.name
    rebound_quarantine.write_bytes(quarantine_path.read_bytes())

    observed = parse_d1_trainer_handoff(payload, artifact_root=rebound_root)

    assert completion.tail_quarantine is not None
    expected = replace(
        completion,
        final_checkpoint=replace(completion.final_checkpoint, path=rebound_checkpoint),
        tail_quarantine=replace(
            completion.tail_quarantine,
            quarantine_path=rebound_quarantine,
        ),
    )
    assert observed == expected
    tail = json.loads(payload)["resume"]["tail_quarantine"]
    assert [event["event_kind"] for event in tail["tail_events"]] == ["train", "eval"]

    mapping = trainer_handoff_to_mapping(completion, artifact_root=tmp_path)
    original = quarantine_path.read_bytes()
    quarantine_path.write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match="quarantine"):
        trainer_handoff_from_mapping(mapping, artifact_root=tmp_path)
