"""D1 resume admission precedes and constrains heavyweight runtime restoration."""

from __future__ import annotations

from pathlib import Path

import pytest

from d1_resume_fixtures import RecordingObserver, captured_step_one_case, origin, resume_start
from entropic_substrate.d1_resume_runtime import (
    D1ResumeRunLocations,
    open_d1_resume,
    restore_d1_resume_runtime,
)
from entropic_substrate.evidence.d1_journal import read_journal
from entropic_substrate.evidence.d1_journal_values import JournalEventKind
from entropic_substrate.evidence.resource_provenance_codec import run_start_to_mapping

pytestmark = pytest.mark.unit


def test_open_records_actual_segment_before_runtime_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    observer = RecordingObserver()
    start = resume_start(case.plan, case.checkpoint_sha256)
    admission = open_d1_resume(
        plan=case.plan,
        start=start,
        origin=origin(),
        locations=D1ResumeRunLocations(case.journal_path, case.checkpoint_path),
        observer=observer,
    )

    event = read_journal(case.journal_path)[-1]
    assert event.event_kind is JournalEventKind.RUN_RESUME
    assert event.payload["run_start_provenance"] == run_start_to_mapping(start)
    assert observer.events == [event]
    restored = restore_d1_resume_runtime(admission)
    assert restored.successful_step == 1
    assert restored.components.resource_ledger.current_state().execution_segment_index == 1
    admission.journal.close()


def test_bad_receipt_and_wrong_segment_refuse_without_journal_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured_step_one_case(tmp_path, monkeypatch)
    original = case.journal_path.read_bytes()
    locations = D1ResumeRunLocations(case.journal_path, case.checkpoint_path)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        open_d1_resume(
            plan=case.plan,
            start=resume_start(case.plan, "0" * 64),
            origin=origin(),
            locations=locations,
            observer=RecordingObserver(),
        )
    assert case.journal_path.read_bytes() == original

    with pytest.raises(ValueError, match="exact next execution segment"):
        open_d1_resume(
            plan=case.plan,
            start=resume_start(case.plan, case.checkpoint_sha256, segment=2),
            origin=origin(),
            locations=locations,
            observer=RecordingObserver(),
        )
    assert case.journal_path.read_bytes() == original
