"""TDD contract for pre-termination D1 trainer evidence custody."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import entropic_substrate.ladder.run.campaign.d1_operator_evidence as operator_module
import entropic_substrate.ladder.run.trainer.d1_segment_engine as segment_engine
from d1_billing_finalizer_fixtures import completed_case
from d1_resume_fixtures import (
    InterruptedD1Case,
    RecordingObserver,
    captured_step_one_case,
    origin,
    resume_start,
)
from d1_resume_hard_crash_fixture import hard_crash_tail_case
from d1_trainer_fixtures import (
    ORIGIN_NS,
    STARTED_AT,
    DeterministicTrainerClock,
    micro_plan,
    run_start,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.kernel.lifecycle import ArtifactIdentity
from entropic_substrate.ladder.capture.records.d1_journal import read_journal
from entropic_substrate.ladder.run.campaign.d1_operator_evidence import (
    D1ExportedTrainerHandoff,
    D1TrainerExportReceipt,
    verify_d1_trainer_export,
    verify_d1_trainer_export_for_request,
)
from entropic_substrate.ladder.run.trainer.d1_fresh_trainer import D1FreshRunLocations, run_d1_fresh
from entropic_substrate.ladder.run.trainer.d1_resume_runtime import D1ResumeRunLocations
from entropic_substrate.ladder.run.trainer.d1_resume_trainer import (
    D1ResumeRunFailure,
    run_d1_resume,
)
from entropic_substrate.ladder.run.trainer.d1_trainer_runtime import D1TrainerWallOrigin
from entropic_substrate.ladder.run.transport.d1_trainer_handoff import (
    D1FreshRunFailure,
    D1TrainerFailureHandoff,
)
from entropic_substrate.ladder.run.transport.d1_trainer_request_codec import (
    D1FreshTrainerRequest,
    D1ResumeTrainerRequest,
    D1TrainerLocationSpec,
)

pytestmark = pytest.mark.unit


class _LaterClock(DeterministicTrainerClock):
    def utc_now(self) -> datetime:
        return super().utc_now() + timedelta(minutes=1)


def _resume_request(
    root: Path,
    case: InterruptedD1Case,
    *,
    quarantine: Path | None = None,
) -> tuple[D1ResumeTrainerRequest, D1ResumeRunLocations]:
    start = resume_start(case.plan, case.checkpoint_sha256)
    start = replace(
        start,
        pod_created_at=start.pod_created_at + timedelta(minutes=1),
        trainer_started_at=start.trainer_started_at + timedelta(minutes=1),
    )
    locations = D1ResumeRunLocations(case.journal_path, case.checkpoint_path, quarantine)
    request = D1ResumeTrainerRequest(
        case.plan,
        start,
        replace(origin(), started_at=start.trainer_started_at),
        D1TrainerLocationSpec(
            str(case.journal_path.relative_to(root)),
            str(case.checkpoint_path.relative_to(root)),
            None if quarantine is None else str(quarantine.relative_to(root)),
        ),
    )
    return request, locations


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
    dummy = tmp_path / "dummy.jsonl"
    dummy.write_bytes(b"not the admitted journal\n")
    with pytest.raises(TypeError):
        verify_d1_trainer_export(
            D1ExportedTrainerHandoff(case.completion, dummy),
            journal_payload=case.journal_path.read_bytes(),  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError, match="factory-only"):
        D1TrainerExportReceipt(  # type: ignore[call-arg]
            handoff=case.completion,
            journal=receipt.journal,
            checkpoint=receipt.checkpoint,
            quality_export_ready=True,
        )


def test_export_binding_requires_exact_request_provenance_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = completed_case(tmp_path)
    candidate = D1ExportedTrainerHandoff(case.completion, case.journal_path)
    request = D1FreshTrainerRequest(
        case.plan,
        case.start,
        D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
        D1TrainerLocationSpec(
            case.journal_path.name,
            case.completion.final_checkpoint.path.name,
        ),
    )
    real_read = operator_module.read_d1_artifact_bytes
    reads = 0

    def read_once(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal reads
        reads += 1
        return real_read(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(operator_module, "read_d1_artifact_bytes", read_once)

    receipt = verify_d1_trainer_export_for_request(
        candidate,
        request=request,
        artifact_root=tmp_path,
    )

    assert receipt.handoff is case.completion
    assert reads == 1
    drifted = (
        replace(
            request,
            origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS + 1),
        ),
        replace(
            request,
            locations=D1TrainerLocationSpec("foreign.jsonl", request.locations.checkpoint_path),
        ),
        replace(
            request,
            locations=D1TrainerLocationSpec(request.locations.journal_path, "foreign.pt"),
        ),
    )
    for item in drifted:
        prior_reads = reads
        with pytest.raises(ValueError, match="request"):
            verify_d1_trainer_export_for_request(
                candidate,
                request=item,
                artifact_root=tmp_path,
            )
        assert reads == prior_reads + 1


def test_request_binding_accepts_real_resumed_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    request, locations = _resume_request(tmp_path, case)
    completion = run_d1_resume(
        plan=case.plan,
        start=request.start,
        origin=request.origin,
        locations=locations,
        observer=RecordingObserver(),
        clock=_LaterClock(),
    )

    receipt = verify_d1_trainer_export_for_request(
        D1ExportedTrainerHandoff(completion, case.journal_path),
        request=request,
        artifact_root=tmp_path,
    )

    assert receipt.handoff is completion
    assert completion.run_start_provenance == request.start


def test_request_binding_accepts_resumed_failure_with_quarantined_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = hard_crash_tail_case(tmp_path, monkeypatch)

    def fail_step(**_kwargs: object) -> object:
        raise RuntimeError("resumed trainer sentinel")

    monkeypatch.setattr(segment_engine, "apply_d1_training_step", fail_step)
    quarantine = tmp_path / "quarantined-tail.jsonl"
    request, locations = _resume_request(tmp_path, case, quarantine=quarantine)
    with pytest.raises(D1ResumeRunFailure) as caught:
        run_d1_resume(
            plan=case.plan,
            start=request.start,
            origin=request.origin,
            locations=locations,
            observer=RecordingObserver(),
            clock=_LaterClock(),
        )

    receipt = verify_d1_trainer_export_for_request(
        D1ExportedTrainerHandoff(caught.value.handoff, case.journal_path),
        request=request,
        artifact_root=tmp_path,
    )

    assert receipt.handoff is caught.value.handoff
    assert quarantine.is_file()


def test_request_binding_joins_successful_resume_quarantine_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = hard_crash_tail_case(tmp_path, monkeypatch)
    quarantine = tmp_path / "quarantined-tail.jsonl"
    request, locations = _resume_request(tmp_path, case, quarantine=quarantine)
    completion = run_d1_resume(
        plan=case.plan,
        start=request.start,
        origin=request.origin,
        locations=locations,
        observer=RecordingObserver(),
        clock=_LaterClock(),
    )
    candidate = D1ExportedTrainerHandoff(completion, case.journal_path)

    assert (
        verify_d1_trainer_export_for_request(
            candidate,
            request=request,
            artifact_root=tmp_path,
        ).handoff
        is completion
    )
    assert completion.tail_quarantine is not None
    drifted = (
        replace(completion, tail_quarantine=None),
        replace(
            completion,
            tail_quarantine=replace(
                completion.tail_quarantine,
                quarantine_path=tmp_path / "foreign-tail.jsonl",
            ),
        ),
        replace(
            completion,
            tail_quarantine=replace(
                completion.tail_quarantine,
                quarantined_journal_sha256="f" * 64,
            ),
        ),
    )
    for handoff in drifted:
        with pytest.raises(ValueError, match="quarantine"):
            verify_d1_trainer_export_for_request(
                D1ExportedTrainerHandoff(handoff, case.journal_path),
                request=request,
                artifact_root=tmp_path,
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
    checkpoint_path = tmp_path / "failed.pt"
    start = run_start(plan)

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
            start=start,
            origin=D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
            locations=D1FreshRunLocations(journal_path, checkpoint_path),
            clock=DeterministicTrainerClock(),
        )
    handoff = caught.value.handoff
    request = D1FreshTrainerRequest(
        plan,
        start,
        D1TrainerWallOrigin(STARTED_AT, ORIGIN_NS),
        D1TrainerLocationSpec(journal_path.name, checkpoint_path.name),
    )
    assert (
        verify_d1_trainer_export_for_request(
            D1ExportedTrainerHandoff(handoff, journal_path),
            request=request,
            artifact_root=tmp_path,
        ).handoff
        is handoff
    )
    drifted = replace(
        handoff,
        termination=replace(handoff.termination, model_successful_step=2),
    )

    with pytest.raises(ValueError, match="model_successful_step"):
        verify_d1_trainer_export(D1ExportedTrainerHandoff(drifted, journal_path))

    events = read_journal(journal_path)
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
            operator_module,
            "read_journal_bytes",
            lambda _payload, *, journal_path, item=mutated: (*events[:-1], item),
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
