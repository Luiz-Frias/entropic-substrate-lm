"""TDD contract for pre-termination D1 trainer evidence custody."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import d1_launch_admission_fixtures as launch_fixtures
from d1_billing_finalizer_fixtures import completed_case
from entropic_substrate.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1TrainerExportReceipt,
    D1TrainerInputEnvelope,
    verify_d1_trainer_export,
)
from entropic_substrate.d1_trainer_handoff import D1TrainerFailureHandoff
from entropic_substrate.evidence.lifecycle import ArtifactIdentity
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping

pytestmark = pytest.mark.unit


def test_trainer_input_envelope_is_canonical_and_binds_plan_to_start(tmp_path: Path) -> None:
    del tmp_path
    run = launch_fixtures.plan(launch_fixtures.profiles()[0], 101)
    start = launch_fixtures.start(run)

    envelope = D1TrainerInputEnvelope(plan=run, start=start)

    assert envelope.payload == {
        "command": "d1-run-fresh",
        "plan": run.plan_manifest,
        "run_start": run_start_to_mapping(start),
        "schema_version": "d1-operator-trainer-input-v1",
    }
    assert envelope.canonical_json == envelope.canonical_json.strip()
    assert "secret" not in envelope.canonical_json.lower()
    with pytest.raises(ValueError, match="image"):
        D1TrainerInputEnvelope(
            plan=run,
            start=replace(start, image_digest="sha256:" + "9" * 64),
        )


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


def test_export_candidate_requires_regular_non_symlink_journal(tmp_path: Path) -> None:
    case_root = tmp_path / "case"
    case_root.mkdir()
    case = completed_case(case_root)
    link = tmp_path / "journal-link.jsonl"
    link.symlink_to(case.journal_path)

    with pytest.raises(ValueError, match="non-symlink"):
        D1ExportedTrainerHandoff(case.completion, link)
