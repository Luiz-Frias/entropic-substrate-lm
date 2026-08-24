"""D1 resume receipts bind quarantined bytes into the authoritative journal."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from entropic_substrate.evidence.d1_journal import (
    AuthoritativeRunJournal,
    JournalTailQuarantineReceipt,
    read_journal,
)
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalError,
    D1JournalEvent,
    JournalCheckpointState,
    JournalEventKind,
    canonical_event_line,
    parse_event_line,
)
from entropic_substrate.experiments.d1_primitives import RunClass

pytestmark = pytest.mark.unit
_CHECKPOINT_SHA256 = "a" * 64


def _event(kind: JournalEventKind, sequence: int, step: int) -> D1JournalEvent:
    return D1JournalEvent(
        event_kind=kind,
        event_seq=sequence,
        logical_run_id="d1-confirmatory-both-seed-7",
        execution_digest="1" * 64,
        aggregation_digest="2" * 64,
        construction_digest="3" * 64,
        family_digest="4" * 64,
        run_class=RunClass.CONFIRMATORY,
        successful_step=step,
        payload={"kind": kind.value},
    )


def _segment_provenance() -> dict[str, object]:
    return {
        "run_start_provenance": {
            "execution_segment_id": "d1-confirmatory-both-seed-7:segment:1",
            "execution_segment_index": 1,
            "logical_run_id": "d1-confirmatory-both-seed-7",
            "resume_checkpoint_sha256": _CHECKPOINT_SHA256,
            "segment_start_mode": "resume",
        },
        "trainer_wall_origin": {
            "monotonic_ns": 10_000,
            "started_at": "2026-08-24T00:00:00+00:00",
        },
    }


def _crashed_tail(
    path: Path,
) -> tuple[JournalCheckpointState, bytes, bytes]:
    journal = AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0))
    journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
    journal.append(_event(JournalEventKind.EVAL, 2, 0))
    checkpoint = journal.state
    journal.append_checkpoint(_CHECKPOINT_SHA256)
    retained_prefix = path.read_bytes()
    journal.append(_event(JournalEventKind.TRAIN, 4, 1))
    journal.append(_event(JournalEventKind.EVAL, 5, 1))
    journal.close()
    return checkpoint, retained_prefix, path.read_bytes()


def _resume_with_quarantine(
    tmp_path: Path,
) -> tuple[Path, Path, JournalTailQuarantineReceipt, bytes, bytes]:
    path = tmp_path / "run.jsonl"
    checkpoint, retained_prefix, original = _crashed_tail(path)
    artifact = tmp_path / "run.pre-resume.jsonl"
    resumed = AuthoritativeRunJournal.resume(
        path,
        checkpoint,
        checkpoint_sha256=_CHECKPOINT_SHA256,
        segment_provenance=_segment_provenance(),
        quarantine_path=artifact,
    )
    receipt = resumed.tail_quarantine
    assert type(receipt) is JournalTailQuarantineReceipt
    resumed.close()
    return path, artifact, receipt, retained_prefix, original


def test_quarantine_receipt_is_canonical_and_self_verifying(tmp_path: Path) -> None:
    path, artifact, receipt, retained_prefix, original = _resume_with_quarantine(tmp_path)

    resume_event = read_journal(path)[-1]
    assert resume_event.payload["tail_quarantine"] == receipt.manifest
    assert receipt.manifest == {
        "artifact": {
            "byte_count": len(original),
            "file_name": artifact.name,
            "role": "pre_resume_authoritative_journal",
            "sha256": receipt.quarantined_journal_sha256,
        },
        "kind": "quarantined_tail",
        "retained_prefix": {
            "byte_count": len(retained_prefix),
            "sha256": receipt.recovered_prefix_sha256,
        },
        "schema_version": "d1-tail-quarantine/v1",
        "tail": {
            "event_count": 2,
            "first_event_sequence": 4,
            "last_event_sequence": 5,
        },
    }
    assert artifact.read_bytes() == original


def test_read_rejects_quarantine_artifact_content_drift(tmp_path: Path) -> None:
    path, artifact, _, _, original = _resume_with_quarantine(tmp_path)
    drifted = bytearray(original)
    drifted[0] ^= 1
    artifact.write_bytes(drifted)

    with pytest.raises(D1JournalError, match=r"artifact.*sha256|sha256.*artifact"):
        read_journal(path)


def test_read_rejects_receipt_drift_against_retained_prefix(tmp_path: Path) -> None:
    path, _, _, _, _ = _resume_with_quarantine(tmp_path)
    lines = path.read_bytes().splitlines(keepends=True)
    resume = parse_event_line(lines[-1])
    payload = cast(dict[str, object], resume.manifest["payload"])
    evidence = cast(dict[str, object], payload["tail_quarantine"])
    retained = cast(dict[str, object], evidence["retained_prefix"])
    retained["sha256"] = "f" * 64
    path.write_bytes(b"".join(lines[:-1]) + canonical_event_line(replace(resume, payload=payload)))

    with pytest.raises(D1JournalError, match=r"retained_prefix.*sha256|sha256.*retained"):
        read_journal(path)


def test_resume_rejects_non_sibling_quarantine_identity_before_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    checkpoint, _, original = _crashed_tail(path)
    other = tmp_path / "other"
    other.mkdir()
    artifact = other / "unresolved-after-capture.jsonl"

    with pytest.raises(D1JournalError, match=r"sibling|artifact"):
        AuthoritativeRunJournal.resume(
            path,
            checkpoint,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            segment_provenance=_segment_provenance(),
            quarantine_path=artifact,
        )

    assert path.read_bytes() == original
    assert not artifact.exists()


def test_read_rejects_implicit_legacy_no_tail_resume(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    journal = AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0))
    journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
    journal.append(_event(JournalEventKind.EVAL, 2, 0))
    checkpoint = journal.state
    journal.append_checkpoint(_CHECKPOINT_SHA256)
    journal.close()
    resumed = AuthoritativeRunJournal.resume(
        path,
        checkpoint,
        checkpoint_sha256=_CHECKPOINT_SHA256,
        segment_provenance=_segment_provenance(),
    )
    resumed.close()
    lines = path.read_bytes().splitlines(keepends=True)
    event = parse_event_line(lines[-1])
    payload = cast(dict[str, object], event.manifest["payload"])
    del payload["tail_quarantine"]
    path.write_bytes(b"".join(lines[:-1]) + canonical_event_line(replace(event, payload=payload)))

    with pytest.raises(D1JournalError, match=r"run_resume|resume_payload"):
        read_journal(path)


def test_failed_receipt_persist_keeps_the_original_tail_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "run.jsonl"
    checkpoint, _, original = _crashed_tail(path)
    artifact = tmp_path / "run.pre-resume.jsonl"

    def fail_receipt_persist(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("deliberate receipt persist failure")

    monkeypatch.setattr(AuthoritativeRunJournal, "_persist", fail_receipt_persist)
    with pytest.raises(RuntimeError, match="deliberate receipt persist failure"):
        AuthoritativeRunJournal.resume(
            path,
            checkpoint,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            segment_provenance=_segment_provenance(),
            quarantine_path=artifact,
        )

    assert artifact.read_bytes() == original
    assert path.read_bytes() == original
