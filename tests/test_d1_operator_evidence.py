"""TDD contract for pre-termination D1 trainer evidence custody."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.d1_operator_evidence as operator_module
import entropic_substrate.d1_segment_engine as segment_engine
from d1_billing_finalizer_fixtures import completed_case
from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    run_start,
)
from entropic_substrate.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1TrainerExportReceipt,
    verify_d1_trainer_export,
)
from entropic_substrate.d1_trainer_handoff import (
    D1FreshRunFailure,
    D1TrainerFailureHandoff,
)
from entropic_substrate.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.experiments.d1_primitives import D1Arm

pytestmark = pytest.mark.unit


def test_duplicate_operator_trainer_input_wire_is_retired() -> None:
    source_root = Path(__file__).parents[1] / "src/entropic_substrate"
    production = "\n".join(path.read_text() for path in source_root.rglob("*.py"))

    assert "D1TrainerInputEnvelope" not in production
    assert "d1-operator-trainer-input-v1" not in production
    assert "observe_runtime" not in production


def test_successful_export_rechecks_journal_prefix_and_checkpoint_bytes(
    tmp_path: Path,
) -> None:
    case = completed_case(tmp_path)
    candidate = D1ExportedTrainerHandoff(
        handoff=case.completion,
        journal_path=case.journal_path,
    )

    receipt = verify_d1_trainer_export(candidate)

    assert type(receipt) is D1TrainerExportReceipt
    assert receipt.quality_export_ready is True
    assert receipt.journal.content_hash.digest == case.completion.journal_state.prefix_sha256
    assert isinstance(receipt.checkpoint, ArtifactIdentity)
    assert receipt.checkpoint.content_hash.digest == case.completion.final_checkpoint.sha256
    assert receipt.manifest["handoff_kind"] == "fresh_completion"
    with pytest.raises(TypeError, match="factory-only"):
        D1TrainerExportReceipt(  # type: ignore[call-arg]
            handoff=case.completion,
            journal=receipt.journal,
            checkpoint=receipt.checkpoint,
            quality_export_ready=True,
        )


def test_checkpoint_or_journal_drift_is_refused_before_provider_termination(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_root.mkdir()
    checkpoint_case = completed_case(checkpoint_root)
    checkpoint_case.completion.final_checkpoint.path.write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="checkpoint"):
        verify_d1_trainer_export(
            D1ExportedTrainerHandoff(
                checkpoint_case.completion,
                checkpoint_case.journal_path,
            )
        )

    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    journal_case = completed_case(journal_root)
    with journal_case.journal_path.open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(ValueError, match="journal"):
        verify_d1_trainer_export(
            D1ExportedTrainerHandoff(journal_case.completion, journal_case.journal_path)
        )


def test_failure_handoff_without_matching_aborted_journal_is_not_export_ready(
    tmp_path: Path,
) -> None:
    case = completed_case(tmp_path)
    failure = D1TrainerFailureHandoff.from_error(
        case.completion.termination,
        RuntimeError("simulated trainer defect"),
    )

    with pytest.raises(ValueError, match="trainer_aborted"):
        verify_d1_trainer_export(D1ExportedTrainerHandoff(failure, case.journal_path))


def test_failure_export_binds_model_cursor_to_authoritative_abort_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = micro_plan(D1Arm.BOTH)
    journal_path = tmp_path / "failed.jsonl"

    real_step = segment_engine.apply_d1_training_step
    calls = 0

    def fail_step(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("trainer sentinel")
        return real_step(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_step)
    with pytest.raises(D1FreshRunFailure) as caught:
        run_d1_fresh(
            plan=plan,
            start=run_start(plan),
            origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
            locations=D1FreshRunLocations(journal_path, tmp_path / "failed.pt"),
            clock=DeterministicTrainerClock(),
        )
    handoff = caught.value.handoff
    drifted = replace(
        handoff,
        termination=replace(handoff.termination, model_successful_step=2),
    )

    with pytest.raises(ValueError, match="model_successful_step"):
        verify_d1_trainer_export(D1ExportedTrainerHandoff(drifted, journal_path))

    events = operator_module.read_journal(journal_path)
    for root, key in (
        (None, "model_successful_step"),
        ("journal_cursor", "successful_step"),
        ("resource_ledger_state", "last_training_step"),
    ):
        payload = json.loads(json.dumps(events[-1].manifest["payload"]))
        target = payload if root is None else payload[root]
        target[key] = True
        mutated = replace(events[-1], payload=payload)
        monkeypatch.setattr(
            operator_module, "read_journal", lambda _path, item=mutated: (*events[:-1], item)
        )
        with pytest.raises(ValueError, match=root or key):
            verify_d1_trainer_export(D1ExportedTrainerHandoff(handoff, journal_path))


def test_export_candidate_requires_regular_non_symlink_journal(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()
    case = completed_case(case_root)
    link = tmp_path / "journal-link.jsonl"
    link.symlink_to(case.journal_path)

    with pytest.raises(ValueError, match="non-symlink"):
        D1ExportedTrainerHandoff(case.completion, link)
