"""Durable single-writer journal for one D1 logical run."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from entropic_substrate.evidence.d1_journal_values import (
    D1JournalError,
    D1JournalEvent,
    JournalCheckpointState,
    JournalEventKind,
    JournalIdentity,
    canonical_event_line,
    parse_event_line,
)

_TERMINAL = frozenset({JournalEventKind.RUN_END, JournalEventKind.RUN_ABORT})


def _error(code: str, path: str, detail: str) -> D1JournalError:
    return D1JournalError(code=code, path=path, detail=detail)


def _require_sha256(value: object, *, path: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error("invalid_sha256", path, "must be 64 lowercase hexadecimal characters")
    return value


def _checkpoint_payload(event: D1JournalEvent) -> str:
    if set(event.payload) != {"checkpoint_sha256"}:
        raise _error(
            "checkpoint_payload",
            "payload",
            "checkpoint event requires exact checkpoint_sha256 payload",
        )
    return _require_sha256(event.payload["checkpoint_sha256"], path="checkpoint_sha256")


class _PrefixHasher(Protocol):
    def update(self, data: bytes, /) -> None: ...

    def hexdigest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _History:
    identity: JournalIdentity
    events: tuple[D1JournalEvent, ...]
    baseline_seen: bool
    terminal: bool

    @property
    def last(self) -> D1JournalEvent:
        return self.events[-1]


def _begin(event: D1JournalEvent) -> _History:
    if event.event_kind is not JournalEventKind.RUN_START:
        raise _error("missing_start", "event_kind", "the first event must be run_start")
    if event.event_seq != 0 or event.successful_step != 0:
        raise _error("invalid_start", "run_start", "run_start requires sequence and step zero")
    return _History(event.identity, (event,), baseline_seen=False, terminal=False)


def _advance(
    history: _History,
    event: D1JournalEvent,
    *,
    allow_resume: bool,
) -> _History:
    prior = history.last
    kind = event.event_kind
    if history.terminal:
        raise _error("post_terminal", "event_kind", "no event may follow a terminal event")
    if event.identity != history.identity:
        raise _error("mixed_identity", "identity", "all events must carry one exact run identity")
    if event.event_seq != prior.event_seq + 1:
        raise _error("invalid_sequence", "event_seq", "event sequences must be consecutive")
    if event.successful_step < prior.successful_step:
        raise _error("decreasing_step", "successful_step", "successful_step must be nondecreasing")
    if kind is JournalEventKind.RUN_START:
        raise _error("duplicate_start", "event_kind", "run_start must appear exactly once")
    if kind is JournalEventKind.RUN_RESUME:
        if not allow_resume:
            raise _error(
                "resume_factory_only", "event_kind", "run_resume requires resume admission"
            )
        _checkpoint_payload(event)
        if prior.event_kind is not JournalEventKind.CHECKPOINT:
            raise _error("resume_without_checkpoint", "event_kind", "resume requires a checkpoint")
        if event.successful_step != prior.successful_step:
            raise _error("resume_step", "successful_step", "resume must preserve checkpoint step")
    elif kind is JournalEventKind.BASELINE_CALIBRATED:
        if history.baseline_seen:
            raise _error("duplicate_baseline", "event_kind", "baseline may be calibrated once")
        if event.successful_step != 0:
            raise _error(
                "baseline_step", "successful_step", "baseline calibration precedes training"
            )
    elif kind is JournalEventKind.TRAIN:
        if not history.baseline_seen:
            raise _error("missing_baseline", "event_kind", "training requires baseline calibration")
        if event.successful_step != prior.successful_step + 1:
            raise _error(
                "train_step",
                "successful_step",
                "training must advance successful_step by exactly one",
            )
    elif kind is JournalEventKind.EVAL:
        if not history.baseline_seen:
            raise _error(
                "missing_baseline", "event_kind", "evaluation requires baseline calibration"
            )
    elif kind is JournalEventKind.CHECKPOINT:
        _checkpoint_payload(event)
        if prior.event_kind is not JournalEventKind.EVAL:
            raise _error("checkpoint_order", "event_kind", "checkpoint requires a prior evaluation")
        if event.successful_step != prior.successful_step:
            raise _error(
                "checkpoint_step",
                "successful_step",
                "checkpoint and evaluation require the same successful_step",
            )
    elif kind is JournalEventKind.RUN_END:
        if prior.event_kind not in {JournalEventKind.EVAL, JournalEventKind.CHECKPOINT}:
            raise _error("terminal_order", "event_kind", "run_end requires terminal evaluation")
        if event.successful_step != prior.successful_step:
            raise _error("terminal_step", "successful_step", "run_end cannot advance training")
    elif kind is JournalEventKind.RUN_ABORT and event.successful_step != prior.successful_step:
        raise _error("abort_step", "successful_step", "run_abort records the last successful step")
    return _History(
        history.identity,
        (*history.events, event),
        baseline_seen=history.baseline_seen or kind is JournalEventKind.BASELINE_CALIBRATED,
        terminal=kind in _TERMINAL,
    )


def _replay(data: bytes) -> tuple[_History, _PrefixHasher]:
    if not data:
        raise _error("empty_journal", "path", "journal must contain exactly one run_start")
    if not data.endswith(b"\n"):
        raise _error("partial_line", "path", "journal has a partial final line")
    history: _History | None = None
    for line in data.splitlines(keepends=True):
        event = parse_event_line(line)
        history = _begin(event) if history is None else _advance(history, event, allow_resume=True)
    if history is None:  # pragma: no cover - guarded by nonempty data
        raise _error("empty_journal", "path", "journal must contain a run_start")
    hasher: _PrefixHasher = hashlib.sha256()
    hasher.update(data)
    return history, hasher


def _checkpoint_state(history: _History, prefix_sha256: str) -> JournalCheckpointState:
    return JournalCheckpointState(
        logical_run_id=history.identity.logical_run_id,
        execution_digest=history.identity.execution_digest,
        last_event_sequence=history.last.event_seq,
        prefix_sha256=prefix_sha256,
        successful_step=history.last.successful_step,
    )


def read_journal(path: str | Path) -> tuple[D1JournalEvent, ...]:
    """Read and validate every canonical record in an authoritative prefix."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise _error("journal_io", "path", f"cannot read journal: {exc}") from exc
    history, _ = _replay(data)
    return history.events


class AuthoritativeRunJournal:
    """Append-only journal whose state never advances ahead of synchronized bytes."""

    def __init__(
        self,
        path: Path,
        handle: BinaryIO,
        history: _History,
        hasher: _PrefixHasher,
    ) -> None:
        self._path = path
        self._handle = handle
        self._history = history
        self._hasher = hasher
        self._closed = False

    @classmethod
    def fresh(cls, path: str | Path, start: D1JournalEvent) -> AuthoritativeRunJournal:
        history = _begin(start)
        target = Path(path)
        try:
            if target.exists():
                if not target.is_file() or target.stat().st_size != 0:
                    raise _error(
                        "occupied_path", "path", "fresh journal path must be missing or empty"
                    )
                handle = target.open("r+b")
            else:
                handle = target.open("x+b")
        except D1JournalError:
            raise
        except OSError as exc:
            raise _error("journal_io", "path", f"cannot create journal: {exc}") from exc
        hasher: _PrefixHasher = hashlib.sha256()
        journal = cls(target, handle, history, hasher)
        journal._persist(start, history)
        return journal

    @classmethod
    def resume(
        cls,
        path: str | Path,
        checkpoint: JournalCheckpointState,
        *,
        checkpoint_sha256: str,
    ) -> AuthoritativeRunJournal:
        if type(checkpoint) is not JournalCheckpointState:
            raise TypeError("checkpoint must be a JournalCheckpointState")
        actual_checkpoint_sha256 = _require_sha256(checkpoint_sha256, path="checkpoint_sha256")
        target = Path(path)
        handle: BinaryIO | None = None
        try:
            handle = target.open("r+b")
            data = handle.read()
            history, hasher = _replay(data)
            if history.terminal:
                raise _error("terminal_resume", "event_kind", "terminal journal cannot resume")
            if history.last.event_kind is not JournalEventKind.CHECKPOINT:
                raise _error(
                    "resume_without_checkpoint",
                    "event_kind",
                    "latest event must be checkpoint",
                )
            lines = data.splitlines(keepends=True)
            pre_history, pre_hasher = _replay(b"".join(lines[:-1]))
            if pre_history.last.event_kind is not JournalEventKind.EVAL:
                raise _error(
                    "checkpoint_prestate",
                    "event_kind",
                    "checkpoint prestate must end in evaluation",
                )
            expected = _checkpoint_state(pre_history, pre_hasher.hexdigest())
            for field in (
                "logical_run_id",
                "execution_digest",
                "last_event_sequence",
                "prefix_sha256",
                "successful_step",
            ):
                if getattr(checkpoint, field) != getattr(expected, field):
                    raise _error("checkpoint_mismatch", field, f"checkpoint {field} differs")
            if _checkpoint_payload(history.last) != actual_checkpoint_sha256:
                raise _error(
                    "checkpoint_hash_mismatch",
                    "checkpoint_sha256",
                    "journal checkpoint_sha256 differs from the checkpoint file",
                )
            resume = D1JournalEvent(
                event_kind=JournalEventKind.RUN_RESUME,
                event_seq=history.last.event_seq + 1,
                logical_run_id=history.identity.logical_run_id,
                execution_digest=history.identity.execution_digest,
                aggregation_digest=history.identity.aggregation_digest,
                construction_digest=history.identity.construction_digest,
                family_digest=history.identity.family_digest,
                run_class=history.identity.run_class,
                successful_step=history.last.successful_step,
                payload={"checkpoint_sha256": actual_checkpoint_sha256},
            )
            advanced = _advance(history, resume, allow_resume=True)
            handle.seek(0, os.SEEK_END)
            journal = cls(target, handle, history, hasher)
            journal._persist(resume, advanced)
            return journal
        except BaseException:
            if handle is not None:
                handle.close()
            raise

    @property
    def state(self) -> JournalCheckpointState:
        return _checkpoint_state(self._history, self._hasher.hexdigest())

    def append(self, event: D1JournalEvent) -> JournalCheckpointState:
        if self._closed:
            raise _error("closed_journal", "path", "journal is closed")
        if event.event_kind is JournalEventKind.CHECKPOINT:
            raise _error(
                "checkpoint_factory_only",
                "event_kind",
                "use append_checkpoint after the checkpoint file is hashed",
            )
        advanced = _advance(self._history, event, allow_resume=False)
        self._persist(event, advanced)
        return self.state

    def append_checkpoint(self, checkpoint_sha256: str) -> JournalCheckpointState:
        """Bind an evaluated prefix to an already-written checkpoint artifact."""
        if self._closed:
            raise _error("closed_journal", "path", "journal is closed")
        digest = _require_sha256(checkpoint_sha256, path="checkpoint_sha256")
        identity = self._history.identity
        event = D1JournalEvent(
            event_kind=JournalEventKind.CHECKPOINT,
            event_seq=self._history.last.event_seq + 1,
            logical_run_id=identity.logical_run_id,
            execution_digest=identity.execution_digest,
            aggregation_digest=identity.aggregation_digest,
            construction_digest=identity.construction_digest,
            family_digest=identity.family_digest,
            run_class=identity.run_class,
            successful_step=self._history.last.successful_step,
            payload={"checkpoint_sha256": digest},
        )
        advanced = _advance(self._history, event, allow_resume=False)
        self._persist(event, advanced)
        return self.state

    def _persist(self, event: D1JournalEvent, advanced: _History) -> None:
        line = canonical_event_line(event)
        try:
            written = self._handle.write(line)
            if written != len(line):
                raise OSError(f"short journal write: {written}/{len(line)} bytes")
            self._handle.flush()
            os.fsync(self._handle.fileno())
        except OSError as exc:
            self.close()
            raise _error("journal_io", "path", f"cannot synchronize journal: {exc}") from exc
        self._hasher.update(line)
        self._history = advanced

    def close(self) -> None:
        if not self._closed:
            self._handle.close()
            self._closed = True

    def __enter__(self) -> AuthoritativeRunJournal:
        if self._closed:
            raise _error("closed_journal", "path", "journal is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
