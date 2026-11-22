"""D1 journal finalization is exact, non-resuming, and terminal-only."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.kernel.identity.d1_primitives import RunClass
from entropic_substrate.ladder.capture.records.d1_journal import (
    AuthoritativeRunJournal,
    read_journal,
)
from entropic_substrate.ladder.capture.records.d1_journal_values import (
    D1JournalError,
    D1JournalEvent,
    JournalCheckpointState,
    JournalEventKind,
)

pytestmark = pytest.mark.unit
_CHECKPOINT_SHA256 = "a" * 64


def _event(
    kind: JournalEventKind,
    sequence: int,
    step: int,
    **overrides: object,
) -> D1JournalEvent:
    values: dict[str, object] = {
        "event_kind": kind,
        "event_seq": sequence,
        "logical_run_id": "d1-confirmatory-both-seed-7",
        "execution_digest": "1" * 64,
        "aggregation_digest": "2" * 64,
        "construction_digest": "3" * 64,
        "family_digest": "4" * 64,
        "run_class": RunClass.CONFIRMATORY,
        "successful_step": step,
        "payload": {"kind": kind.value},
    }
    values.update(overrides)
    return D1JournalEvent(**values)  # type: ignore[arg-type]


def _checkpoint_tail(path: Path) -> JournalCheckpointState:
    with AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0)) as journal:
        journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
        journal.append(_event(JournalEventKind.EVAL, 2, 0))
        journal.append_checkpoint(_CHECKPOINT_SHA256)
        return journal.state


def _aborted_tail(path: Path) -> JournalCheckpointState:
    with AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0)) as journal:
        journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
        journal.append(_event(JournalEventKind.TRAINER_ABORTED, 2, 0))
        return journal.state


@pytest.mark.parametrize("tail_factory", [_checkpoint_tail, _aborted_tail])
def test_open_for_finalization_preserves_prefix_and_appends_no_resume(
    tmp_path: Path,
    tail_factory: object,
) -> None:
    assert callable(tail_factory)
    path = tmp_path / "run.jsonl"
    expected = tail_factory(path)
    before = path.read_bytes()

    finalizer = AuthoritativeRunJournal.open_for_finalization(path, expected)

    assert path.read_bytes() == before
    assert finalizer.state == expected
    assert JournalEventKind.RUN_RESUME not in {event.event_kind for event in read_journal(path)}
    finalizer.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("logical_run_id", "another-run"),
        ("execution_digest", "f" * 64),
        ("last_event_sequence", 1),
        ("prefix_sha256", "f" * 64),
        ("successful_step", 1),
    ],
)
def test_finalization_refuses_every_full_prefix_mismatch_without_mutation(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "run.jsonl"
    expected = _checkpoint_tail(path)
    before = path.read_bytes()

    with pytest.raises(D1JournalError, match=field):
        AuthoritativeRunJournal.open_for_finalization(
            path,
            replace(expected, **{field: value}),
        )

    assert path.read_bytes() == before


def test_finalization_prefix_guard_is_load_bearing_against_a_stale_valid_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    journal = AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0))
    journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
    journal.append(_event(JournalEventKind.EVAL, 2, 0))
    stale_but_valid = journal.state
    journal.append_checkpoint(_CHECKPOINT_SHA256)
    journal.close()
    before = path.read_bytes()

    with pytest.raises(D1JournalError, match="last_event_sequence"):
        AuthoritativeRunJournal.open_for_finalization(path, stale_but_valid)

    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "candidate",
    [
        JournalEventKind.TRAIN,
        JournalEventKind.EVAL,
        JournalEventKind.RUN_RESUME,
        JournalEventKind.BASELINE_CALIBRATED,
        JournalEventKind.RUN_ABORT,
    ],
)
def test_trainer_aborted_allows_only_same_step_run_end(
    tmp_path: Path,
    candidate: JournalEventKind,
) -> None:
    path = tmp_path / f"{candidate.value}.jsonl"
    journal = AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0))
    journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
    journal.append(_event(JournalEventKind.TRAINER_ABORTED, 2, 0))
    before = path.read_bytes()

    with pytest.raises(D1JournalError, match="trainer_aborted"):
        journal.append(_event(candidate, 3, 1 if candidate is JournalEventKind.TRAIN else 0))

    assert path.read_bytes() == before
    journal.close()


def test_trainer_aborted_refuses_checkpoint_and_step_advance(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    journal = AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0))
    journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
    journal.append(_event(JournalEventKind.TRAINER_ABORTED, 2, 0))
    before = path.read_bytes()

    with pytest.raises(D1JournalError, match="trainer_aborted"):
        journal.append_checkpoint(_CHECKPOINT_SHA256)
    with pytest.raises(D1JournalError, match=r"successful_step|advance"):
        journal.append(_event(JournalEventKind.RUN_END, 3, 1))

    assert path.read_bytes() == before
    journal.close()


def test_finalization_handle_rejects_training_after_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    expected = _checkpoint_tail(path)
    before = path.read_bytes()
    finalizer = AuthoritativeRunJournal.open_for_finalization(path, expected)

    with pytest.raises(D1JournalError, match="finalization"):
        finalizer.append(_event(JournalEventKind.TRAIN, 4, 1))

    assert path.read_bytes() == before
    finalizer.close()


@pytest.mark.parametrize("tail_factory", [_checkpoint_tail, _aborted_tail])
def test_finalizer_permits_exactly_one_same_step_run_end(
    tmp_path: Path,
    tail_factory: object,
) -> None:
    assert callable(tail_factory)
    path = tmp_path / "run.jsonl"
    expected = tail_factory(path)
    finalizer = AuthoritativeRunJournal.open_for_finalization(path, expected)
    sequence = expected.last_event_sequence + 1

    finalizer.append(
        _event(
            JournalEventKind.RUN_END,
            sequence,
            expected.successful_step,
            payload={"outcome": "success"},
        )
    )

    assert read_journal(path)[-1].event_kind is JournalEventKind.RUN_END
    with pytest.raises(D1JournalError, match="terminal"):
        finalizer.append(_event(JournalEventKind.RUN_END, sequence + 1, expected.successful_step))
    finalizer.close()


@pytest.mark.parametrize(
    "tail_kind",
    [
        JournalEventKind.RUN_START,
        JournalEventKind.BASELINE_CALIBRATED,
        JournalEventKind.EVAL,
        JournalEventKind.TRAIN,
        JournalEventKind.RUN_RESUME,
    ],
)
def test_finalization_refuses_nonfinalizable_tail_without_mutation(
    tmp_path: Path,
    tail_kind: JournalEventKind,
) -> None:
    path = tmp_path / f"{tail_kind.value}.jsonl"
    journal = AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0))
    if tail_kind is not JournalEventKind.RUN_START:
        journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
    if tail_kind is JournalEventKind.EVAL:
        journal.append(_event(JournalEventKind.EVAL, 2, 0))
    elif tail_kind is JournalEventKind.TRAIN:
        journal.append(_event(JournalEventKind.TRAIN, 2, 1))
    elif tail_kind is JournalEventKind.RUN_RESUME:
        journal.append(_event(JournalEventKind.EVAL, 2, 0))
        before_checkpoint = journal.state
        journal.append_checkpoint(_CHECKPOINT_SHA256)
        journal.close()
        resumed = AuthoritativeRunJournal.resume(
            path,
            before_checkpoint,
            checkpoint_sha256=_CHECKPOINT_SHA256,
            segment_provenance={
                "run_start_provenance": {"segment": 1},
                "trainer_wall_origin": {"monotonic_ns": 1},
            },
        )
        expected = resumed.state
        resumed.close()
    if tail_kind is not JournalEventKind.RUN_RESUME:
        expected = journal.state
        journal.close()
    before = path.read_bytes()

    with pytest.raises(D1JournalError, match="finalizable"):
        AuthoritativeRunJournal.open_for_finalization(path, expected)

    assert path.read_bytes() == before


@pytest.mark.parametrize("terminal_kind", [JournalEventKind.RUN_END, JournalEventKind.RUN_ABORT])
def test_finalization_refuses_terminal_legacy_or_current_tail_without_mutation(
    tmp_path: Path,
    terminal_kind: JournalEventKind,
) -> None:
    path = tmp_path / f"{terminal_kind.value}.jsonl"
    journal = AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0))
    journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
    journal.append(_event(JournalEventKind.EVAL, 2, 0))
    journal.append_checkpoint(_CHECKPOINT_SHA256)
    journal.append(_event(terminal_kind, 4, 0))
    expected = journal.state
    journal.close()
    before = path.read_bytes()

    with pytest.raises(D1JournalError, match="terminal"):
        AuthoritativeRunJournal.open_for_finalization(path, expected)

    assert path.read_bytes() == before


def test_finalization_state_hash_covers_the_entire_prefix(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    expected = _aborted_tail(path)

    assert expected.prefix_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
