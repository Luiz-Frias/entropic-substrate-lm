"""D1 authoritative journal durability, ordering, and exact-resume contract."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_journal import (
    AuthoritativeRunJournal,
    read_journal,
)
from entropic_substrate.evidence.d1_journal_values import (
    D1JournalError,
    D1JournalEvent,
    JournalEventKind,
    canonical_event_line,
)
from entropic_substrate.experiments.d1_primitives import RunClass

pytestmark = pytest.mark.unit


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


def _checkpointed(path: Path) -> AuthoritativeRunJournal:
    journal = AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0))
    journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
    journal.append(_event(JournalEventKind.EVAL, 2, 0))
    journal.append(_event(JournalEventKind.CHECKPOINT, 3, 0))
    return journal


def test_fresh_journal_writes_and_syncs_canonical_start_to_missing_or_empty_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import entropic_substrate.evidence.d1_journal as journal_module

    calls: list[int] = []
    monkeypatch.setattr(journal_module.os, "fsync", calls.append)
    for name in ("missing.jsonl", "empty.jsonl"):
        path = tmp_path / name
        if name.startswith("empty"):
            path.touch()
        with AuthoritativeRunJournal.fresh(
            path, _event(JournalEventKind.RUN_START, 0, 0)
        ) as journal:
            line = canonical_event_line(_event(JournalEventKind.RUN_START, 0, 0))
            assert path.read_bytes() == line
            assert journal.state.prefix_sha256 == hashlib.sha256(line).hexdigest()
            assert journal.state.last_event_sequence == 0
            assert journal.state.successful_step == 0
    assert len(calls) == 2


def test_fresh_refuses_nonempty_path_without_modifying_it(tmp_path: Path) -> None:
    path = tmp_path / "occupied.jsonl"
    path.write_bytes(b"owned-by-another-run\n")
    before = path.read_bytes()
    with pytest.raises(D1JournalError, match="empty"):
        AuthoritativeRunJournal.fresh(path, _event(JournalEventKind.RUN_START, 0, 0))
    assert path.read_bytes() == before


def test_valid_execution_orders_baseline_eval_checkpoint_train_and_terminal(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    journal = _checkpointed(path)
    journal.append(_event(JournalEventKind.TRAIN, 4, 1))
    journal.append(_event(JournalEventKind.EVAL, 5, 1))
    journal.append(_event(JournalEventKind.CHECKPOINT, 6, 1))
    journal.append(_event(JournalEventKind.RUN_END, 7, 1))
    assert [event.event_kind for event in read_journal(path)] == [
        JournalEventKind.RUN_START,
        JournalEventKind.BASELINE_CALIBRATED,
        JournalEventKind.EVAL,
        JournalEventKind.CHECKPOINT,
        JournalEventKind.TRAIN,
        JournalEventKind.EVAL,
        JournalEventKind.CHECKPOINT,
        JournalEventKind.RUN_END,
    ]
    with pytest.raises(D1JournalError, match="terminal"):
        journal.append(_event(JournalEventKind.TRAIN, 8, 2))
    journal.close()


@pytest.mark.parametrize(
    ("prefix", "candidate", "message"),
    [
        ((), _event(JournalEventKind.TRAIN, 1, 1), "baseline"),
        (
            (_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0),),
            _event(JournalEventKind.BASELINE_CALIBRATED, 2, 0),
            "once",
        ),
        ((), _event(JournalEventKind.EVAL, 1, 0), "baseline"),
        (
            (_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0),),
            _event(JournalEventKind.CHECKPOINT, 2, 0),
            "evaluation",
        ),
        (
            (
                _event(JournalEventKind.BASELINE_CALIBRATED, 1, 0),
                _event(JournalEventKind.EVAL, 2, 0),
            ),
            _event(JournalEventKind.CHECKPOINT, 3, 1),
            "same successful_step",
        ),
        ((), _event(JournalEventKind.TRAIN, 2, 1), "consecutive"),
        (
            (_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0),),
            _event(JournalEventKind.TRAIN, 2, 2),
            "exactly one",
        ),
        ((), _event(JournalEventKind.RUN_START, 1, 0), "exactly once"),
        ((), _event(JournalEventKind.RUN_RESUME, 1, 0), "resume"),
    ],
)
def test_append_refuses_illegal_event_transitions(
    tmp_path: Path,
    prefix: tuple[D1JournalEvent, ...],
    candidate: D1JournalEvent,
    message: str,
) -> None:
    journal = AuthoritativeRunJournal.fresh(
        tmp_path / "run.jsonl", _event(JournalEventKind.RUN_START, 0, 0)
    )
    for event in prefix:
        journal.append(event)
    with pytest.raises(D1JournalError, match=message):
        journal.append(candidate)
    journal.close()


def test_append_refuses_decreasing_step_and_mixed_identity(tmp_path: Path) -> None:
    journal = _checkpointed(tmp_path / "run.jsonl")
    journal.append(_event(JournalEventKind.TRAIN, 4, 1))
    with pytest.raises(D1JournalError, match="successful_step"):
        journal.append(_event(JournalEventKind.EVAL, 5, 0))
    with pytest.raises(D1JournalError, match="identity"):
        journal.append(_event(JournalEventKind.TRAIN, 5, 2, aggregation_digest="f" * 64))
    journal.close()


def test_read_replays_the_whole_prefix_and_rejects_partial_or_illegal_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    start = canonical_event_line(_event(JournalEventKind.RUN_START, 0, 0))
    path.write_bytes(start.rstrip(b"\n"))
    with pytest.raises(D1JournalError, match="partial"):
        read_journal(path)

    second_start = canonical_event_line(_event(JournalEventKind.RUN_START, 1, 0))
    path.write_bytes(start + second_start)
    with pytest.raises(D1JournalError, match="exactly once"):
        read_journal(path)

    mixed = canonical_event_line(
        _event(JournalEventKind.BASELINE_CALIBRATED, 1, 0, execution_digest="f" * 64)
    )
    path.write_bytes(start + mixed)
    with pytest.raises(D1JournalError, match="identity"):
        read_journal(path)


def test_resume_matches_checkpoint_prefix_then_appends_resume_automatically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "run.jsonl"
    original = _checkpointed(path)
    checkpoint = original.state
    original.close()
    before = path.read_bytes()

    resumed = AuthoritativeRunJournal.resume(
        path, checkpoint, payload={"checkpoint_sha256": "a" * 64}
    )
    events = read_journal(path)
    assert events[-1].event_kind is JournalEventKind.RUN_RESUME
    assert events[-1].event_seq == checkpoint.last_event_sequence + 1
    assert events[-1].successful_step == checkpoint.successful_step
    assert events[-1].payload["checkpoint_sha256"] == "a" * 64
    assert resumed.state.prefix_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes().startswith(before)
    resumed.append(_event(JournalEventKind.TRAIN, 5, 1))
    resumed.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("logical_run_id", "other-run", "logical_run_id"),
        ("execution_digest", "f" * 64, "execution_digest"),
        ("last_event_sequence", 2, "last_event_sequence"),
        ("prefix_sha256", "f" * 64, "prefix_sha256"),
        ("successful_step", 1, "successful_step"),
    ],
)
def test_resume_refuses_every_checkpoint_prefix_mismatch_without_appending(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path = tmp_path / "run.jsonl"
    journal = _checkpointed(path)
    checkpoint = journal.state
    journal.close()
    before = path.read_bytes()
    with pytest.raises(D1JournalError, match=message):
        AuthoritativeRunJournal.resume(path, replace(checkpoint, **{field: value}), payload={})
    assert path.read_bytes() == before


def test_resume_requires_latest_checkpoint_and_refuses_terminal_prefix(tmp_path: Path) -> None:
    no_checkpoint = tmp_path / "no-checkpoint.jsonl"
    journal = AuthoritativeRunJournal.fresh(no_checkpoint, _event(JournalEventKind.RUN_START, 0, 0))
    state = journal.state
    journal.close()
    with pytest.raises(D1JournalError, match="checkpoint"):
        AuthoritativeRunJournal.resume(no_checkpoint, state, payload={})

    terminal = tmp_path / "terminal.jsonl"
    journal = _checkpointed(terminal)
    journal.append(_event(JournalEventKind.RUN_ABORT, 4, 0))
    state = journal.state
    journal.close()
    with pytest.raises(D1JournalError, match="terminal"):
        AuthoritativeRunJournal.resume(terminal, state, payload={})


def test_closed_journal_refuses_append(tmp_path: Path) -> None:
    journal = AuthoritativeRunJournal.fresh(
        tmp_path / "run.jsonl", _event(JournalEventKind.RUN_START, 0, 0)
    )
    journal.close()
    with pytest.raises(D1JournalError, match="closed"):
        journal.append(_event(JournalEventKind.BASELINE_CALIBRATED, 1, 0))
