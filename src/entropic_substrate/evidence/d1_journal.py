"""Durable single-writer journal for one D1 logical run."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from entropic_substrate.evidence.d1_journal_values import (
    D1JournalError,
    D1JournalEvent,
    FrozenJson,
    JournalCheckpointState,
    JournalEventKind,
    JournalIdentity,
    canonical_event_line,
    parse_event_line,
)

_TERMINAL = frozenset({JournalEventKind.RUN_END, JournalEventKind.RUN_ABORT})
_FINALIZABLE_TAIL = frozenset({JournalEventKind.CHECKPOINT, JournalEventKind.TRAINER_ABORTED})
_RECOVERABLE_TAIL = frozenset(
    {JournalEventKind.RUN_RESUME, JournalEventKind.TRAIN, JournalEventKind.EVAL}
)
_STATE_FIELDS = (
    "logical_run_id",
    "execution_digest",
    "last_event_sequence",
    "prefix_sha256",
    "successful_step",
)
_TAIL_QUARANTINE_SCHEMA = "d1-tail-quarantine/v1"
_QUARANTINE_ARTIFACT_ROLE = "pre_resume_authoritative_journal"


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


def _exact_mapping(
    value: object,
    *,
    fields: frozenset[str],
    path: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _error("invalid_type", path, "must be a JSON object")
    actual = set(value)
    missing = sorted(fields - actual)
    unknown = sorted(actual - fields)
    if missing:
        raise _error("missing_field", path, f"missing required field(s): {', '.join(missing)}")
    if unknown:
        raise _error("unknown_field", path, f"unknown field(s): {', '.join(unknown)}")
    return cast(Mapping[str, object], value)


def _nonnegative_int(value: object, *, path: str) -> int:
    if type(value) is not int or value < 0:
        raise _error("invalid_integer", path, "must be a nonnegative integer")
    return value


def _positive_int(value: object, *, path: str) -> int:
    result = _nonnegative_int(value, path=path)
    if result == 0:
        raise _error("invalid_integer", path, "must be a positive integer")
    return result


def _artifact_file_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or value.strip() != value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
    ):
        raise _error(
            "artifact_identity",
            "tail_quarantine.artifact.file_name",
            "must be one safe sibling artifact file name",
        )
    return value


@dataclass(frozen=True, slots=True)
class _TailQuarantineEvidence:
    artifact_file_name: str
    artifact_sha256: str
    artifact_byte_count: int
    retained_prefix_sha256: str
    retained_prefix_byte_count: int
    tail_event_count: int
    first_tail_event_sequence: int
    last_tail_event_sequence: int


def _tail_quarantine_evidence(value: object) -> _TailQuarantineEvidence | None:
    root = _exact_mapping(
        value,
        fields=frozenset({"schema_version", "kind"})
        if isinstance(value, Mapping) and value.get("kind") == "no_tail"
        else frozenset({"schema_version", "kind", "artifact", "retained_prefix", "tail"}),
        path="tail_quarantine",
    )
    if root["schema_version"] != _TAIL_QUARANTINE_SCHEMA:
        raise _error(
            "unsupported_schema",
            "tail_quarantine.schema_version",
            "unsupported tail-quarantine schema",
        )
    if root["kind"] == "no_tail":
        return None
    if root["kind"] != "quarantined_tail":
        raise _error("invalid_enum", "tail_quarantine.kind", "unsupported quarantine kind")
    artifact = _exact_mapping(
        root["artifact"],
        fields=frozenset({"role", "file_name", "sha256", "byte_count"}),
        path="tail_quarantine.artifact",
    )
    if artifact["role"] != _QUARANTINE_ARTIFACT_ROLE:
        raise _error(
            "artifact_identity",
            "tail_quarantine.artifact.role",
            "unsupported quarantine artifact role",
        )
    retained = _exact_mapping(
        root["retained_prefix"],
        fields=frozenset({"sha256", "byte_count"}),
        path="tail_quarantine.retained_prefix",
    )
    tail = _exact_mapping(
        root["tail"],
        fields=frozenset({"event_count", "first_event_sequence", "last_event_sequence"}),
        path="tail_quarantine.tail",
    )
    evidence = _TailQuarantineEvidence(
        artifact_file_name=_artifact_file_name(artifact["file_name"]),
        artifact_sha256=_require_sha256(artifact["sha256"], path="tail_quarantine.artifact.sha256"),
        artifact_byte_count=_positive_int(
            artifact["byte_count"], path="tail_quarantine.artifact.byte_count"
        ),
        retained_prefix_sha256=_require_sha256(
            retained["sha256"], path="tail_quarantine.retained_prefix.sha256"
        ),
        retained_prefix_byte_count=_positive_int(
            retained["byte_count"], path="tail_quarantine.retained_prefix.byte_count"
        ),
        tail_event_count=_positive_int(
            tail["event_count"], path="tail_quarantine.tail.event_count"
        ),
        first_tail_event_sequence=_nonnegative_int(
            tail["first_event_sequence"],
            path="tail_quarantine.tail.first_event_sequence",
        ),
        last_tail_event_sequence=_nonnegative_int(
            tail["last_event_sequence"],
            path="tail_quarantine.tail.last_event_sequence",
        ),
    )
    if evidence.artifact_byte_count <= evidence.retained_prefix_byte_count:
        raise _error(
            "quarantine_size",
            "tail_quarantine.artifact.byte_count",
            "quarantined journal must contain bytes beyond the retained prefix",
        )
    expected_count = evidence.last_tail_event_sequence - evidence.first_tail_event_sequence + 1
    if evidence.tail_event_count != expected_count:
        raise _error(
            "tail_range",
            "tail_quarantine.tail",
            "tail event count and sequence range differ",
        )
    return evidence


def _no_tail_evidence() -> dict[str, str]:
    return {"kind": "no_tail", "schema_version": _TAIL_QUARANTINE_SCHEMA}


def _checkpoint_payload(event: D1JournalEvent) -> str:
    if set(event.payload) != {"checkpoint_sha256"}:
        raise _error(
            "checkpoint_payload",
            "payload",
            "checkpoint event requires exact checkpoint_sha256 payload",
        )
    return _require_sha256(event.payload["checkpoint_sha256"], path="checkpoint_sha256")


def _resume_checkpoint_payload(event: D1JournalEvent) -> str:
    expected = {
        "checkpoint_sha256",
        "run_start_provenance",
        "tail_quarantine",
        "trainer_wall_origin",
    }
    if set(event.payload) != expected:
        raise _error(
            "resume_payload",
            "payload",
            "run_resume requires checkpoint, run-start, and wall-origin evidence",
        )
    for field in ("run_start_provenance", "trainer_wall_origin"):
        value = event.payload[field]
        if not isinstance(value, Mapping) or not value:
            raise _error("resume_payload", field, "must be a nonempty JSON object")
    _tail_quarantine_evidence(event.payload["tail_quarantine"])
    return _require_sha256(event.payload["checkpoint_sha256"], path="checkpoint_sha256")


@dataclass(frozen=True, slots=True)
class JournalTailQuarantineReceipt:
    """Durable identity of a post-checkpoint journal tail preserved before rollback."""

    quarantine_path: Path
    quarantined_journal_sha256: str
    quarantined_journal_byte_count: int
    recovered_prefix_sha256: str
    recovered_prefix_byte_count: int
    tail_event_count: int
    first_tail_event_sequence: int
    last_tail_event_sequence: int
    tail_events: tuple[D1JournalEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.quarantine_path), Path):
            raise TypeError("journal quarantine_path must be a Path")
        _artifact_file_name(self.quarantine_path.name)
        _require_sha256(
            self.quarantined_journal_sha256,
            path="quarantined_journal_sha256",
        )
        _require_sha256(self.recovered_prefix_sha256, path="recovered_prefix_sha256")
        if (
            type(self.quarantined_journal_byte_count) is not int
            or type(self.recovered_prefix_byte_count) is not int
            or self.recovered_prefix_byte_count < 1
            or self.quarantined_journal_byte_count <= self.recovered_prefix_byte_count
        ):
            raise ValueError("journal quarantine byte counts are invalid")
        if type(self.tail_event_count) is not int or self.tail_event_count < 1:
            raise ValueError("journal quarantine must contain at least one tail event")
        if (
            type(self.first_tail_event_sequence) is not int
            or type(self.last_tail_event_sequence) is not int
            or self.first_tail_event_sequence < 0
            or self.last_tail_event_sequence < self.first_tail_event_sequence
        ):
            raise ValueError("journal quarantine event sequence range is invalid")
        if (
            type(self.tail_events) is not tuple
            or len(self.tail_events) != self.tail_event_count
            or self.tail_events[0].event_seq != self.first_tail_event_sequence
            or self.tail_events[-1].event_seq != self.last_tail_event_sequence
        ):
            raise ValueError("journal quarantine tail events differ from its receipt")
        if self.tail_event_count != (
            self.last_tail_event_sequence - self.first_tail_event_sequence + 1
        ):
            raise ValueError("journal quarantine tail sequence range is not consecutive")

    @property
    def manifest(self) -> dict[str, FrozenJson]:
        return {
            "artifact": {
                "byte_count": self.quarantined_journal_byte_count,
                "file_name": self.quarantine_path.name,
                "role": _QUARANTINE_ARTIFACT_ROLE,
                "sha256": self.quarantined_journal_sha256,
            },
            "kind": "quarantined_tail",
            "retained_prefix": {
                "byte_count": self.recovered_prefix_byte_count,
                "sha256": self.recovered_prefix_sha256,
            },
            "schema_version": _TAIL_QUARANTINE_SCHEMA,
            "tail": {
                "event_count": self.tail_event_count,
                "first_event_sequence": self.first_tail_event_sequence,
                "last_event_sequence": self.last_tail_event_sequence,
            },
        }


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
    if (
        prior.event_kind is JournalEventKind.TRAINER_ABORTED
        and kind is not JournalEventKind.RUN_END
    ):
        raise _error(
            "post_trainer_abort",
            "event_kind",
            "only run_end may follow trainer_aborted",
        )
    if kind is JournalEventKind.RUN_RESUME:
        if not allow_resume:
            raise _error(
                "resume_factory_only", "event_kind", "run_resume requires resume admission"
            )
        _resume_checkpoint_payload(event)
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
    elif kind is JournalEventKind.TRAINER_ABORTED:
        if event.successful_step != prior.successful_step:
            raise _error(
                "trainer_abort_step",
                "successful_step",
                "trainer_aborted records the last successful step",
            )
    elif kind is JournalEventKind.RUN_END:
        if prior.event_kind not in _FINALIZABLE_TAIL | {JournalEventKind.EVAL}:
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


def _validate_quarantine_artifact(
    evidence: _TailQuarantineEvidence,
    *,
    retained_prefix: bytes,
    retained_history: _History,
    journal_path: Path,
) -> None:
    if len(retained_prefix) != evidence.retained_prefix_byte_count:
        raise _error(
            "retained_prefix_size",
            "tail_quarantine.retained_prefix.byte_count",
            "retained prefix byte count differs from the authoritative journal",
        )
    if hashlib.sha256(retained_prefix).hexdigest() != evidence.retained_prefix_sha256:
        raise _error(
            "retained_prefix_hash",
            "tail_quarantine.retained_prefix.sha256",
            "retained prefix sha256 differs from the authoritative journal",
        )
    artifact_path = journal_path.parent / evidence.artifact_file_name
    if artifact_path == journal_path or artifact_path.is_symlink() or not artifact_path.is_file():
        raise _error(
            "quarantine_artifact_missing",
            "tail_quarantine.artifact.file_name",
            "quarantine artifact must be a distinct regular sibling file",
        )
    try:
        artifact = artifact_path.read_bytes()
    except OSError as exc:
        raise _error(
            "quarantine_artifact_io",
            "tail_quarantine.artifact.file_name",
            f"cannot read quarantine artifact: {exc}",
        ) from exc
    if len(artifact) != evidence.artifact_byte_count:
        raise _error(
            "quarantine_artifact_size",
            "tail_quarantine.artifact.byte_count",
            "quarantine artifact byte count differs",
        )
    if hashlib.sha256(artifact).hexdigest() != evidence.artifact_sha256:
        raise _error(
            "quarantine_artifact_hash",
            "tail_quarantine.artifact.sha256",
            "quarantine artifact sha256 differs",
        )
    if not artifact.startswith(retained_prefix):
        raise _error(
            "quarantine_prefix",
            "tail_quarantine.retained_prefix",
            "quarantine artifact does not begin with the retained prefix",
        )
    tail_bytes = artifact[len(retained_prefix) :]
    if not tail_bytes.endswith(b"\n"):
        raise _error(
            "quarantine_tail",
            "tail_quarantine.tail",
            "quarantined tail must end in a complete event line",
        )
    tail_events = tuple(parse_event_line(line) for line in tail_bytes.splitlines(keepends=True))
    if len(tail_events) != evidence.tail_event_count:
        raise _error(
            "tail_event_count",
            "tail_quarantine.tail.event_count",
            "quarantined artifact tail event count differs",
        )
    expected_sequences = tuple(
        range(evidence.first_tail_event_sequence, evidence.last_tail_event_sequence + 1)
    )
    if tuple(event.event_seq for event in tail_events) != expected_sequences:
        raise _error(
            "tail_event_range",
            "tail_quarantine.tail",
            "quarantined artifact tail sequence range differs",
        )
    replayed = retained_history
    for event in tail_events:
        if event.event_kind not in _RECOVERABLE_TAIL:
            raise _error(
                "unsafe_tail",
                "tail_quarantine.tail",
                "quarantined artifact contains a nonrecoverable event",
            )
        replayed = _advance(replayed, event, allow_resume=True)


def _replay(data: bytes, *, journal_path: Path) -> tuple[_History, _PrefixHasher]:
    if not data:
        raise _error("empty_journal", "path", "journal must contain exactly one run_start")
    if not data.endswith(b"\n"):
        raise _error("partial_line", "path", "journal has a partial final line")
    history: _History | None = None
    prefix_bytes = 0
    for line in data.splitlines(keepends=True):
        event = parse_event_line(line)
        retained_history = history
        history = _begin(event) if history is None else _advance(history, event, allow_resume=True)
        if event.event_kind is JournalEventKind.RUN_RESUME:
            if retained_history is None:  # pragma: no cover - run_resume cannot begin a journal
                raise _error("missing_start", "event_kind", "run_resume cannot begin a journal")
            evidence = _tail_quarantine_evidence(event.payload["tail_quarantine"])
            if evidence is not None:
                _validate_quarantine_artifact(
                    evidence,
                    retained_prefix=data[:prefix_bytes],
                    retained_history=retained_history,
                    journal_path=journal_path,
                )
        prefix_bytes += len(line)
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
    history, _ = _replay(data, journal_path=Path(path))
    return history.events


def _fsync_parent(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _quarantine_tail(
    *,
    original: bytes,
    recovered_prefix: bytes,
    tail: tuple[D1JournalEvent, ...],
    quarantine_path: Path,
    journal_path: Path,
) -> JournalTailQuarantineReceipt:
    parent = quarantine_path.parent
    _artifact_file_name(quarantine_path.name)
    try:
        if parent.resolve(strict=True) != journal_path.parent.resolve(strict=True):
            raise _error(
                "quarantine_artifact_location",
                "quarantine_path",
                "quarantine artifact must be a sibling of its authoritative journal",
            )
    except OSError as exc:
        raise _error(
            "tail_quarantine_path",
            "quarantine_path",
            f"cannot resolve quarantine artifact parent: {exc}",
        ) from exc
    if not parent.is_dir() or parent.is_symlink() or quarantine_path.exists():
        raise _error(
            "tail_quarantine_path",
            "quarantine_path",
            "tail quarantine target must be missing under a real directory",
        )
    durable = False
    try:
        with quarantine_path.open("xb") as quarantine:
            written = quarantine.write(original)
            if written != len(original):
                raise OSError(f"short quarantine write: {written}/{len(original)} bytes")
            quarantine.flush()
            os.fsync(quarantine.fileno())
        _fsync_parent(parent)
        durable = True
    except OSError as exc:
        if not durable:
            quarantine_path.unlink(missing_ok=True)
        detail = f"cannot durably quarantine journal tail: {exc}"
        if durable:
            detail += f"; original is preserved at {quarantine_path}"
        raise _error("tail_quarantine_io", "quarantine_path", detail) from exc
    return JournalTailQuarantineReceipt(
        quarantine_path=quarantine_path,
        quarantined_journal_sha256=hashlib.sha256(original).hexdigest(),
        quarantined_journal_byte_count=len(original),
        recovered_prefix_sha256=hashlib.sha256(recovered_prefix).hexdigest(),
        recovered_prefix_byte_count=len(recovered_prefix),
        tail_event_count=len(tail),
        first_tail_event_sequence=tail[0].event_seq,
        last_tail_event_sequence=tail[-1].event_seq,
        tail_events=tail,
    )


class AuthoritativeRunJournal:
    """Append-only journal whose state never advances ahead of synchronized bytes."""

    def __init__(
        self,
        path: Path,
        handle: BinaryIO,
        history: _History,
        hasher: _PrefixHasher,
        tail_quarantine: JournalTailQuarantineReceipt | None = None,
        *,
        finalization_only: bool = False,
    ) -> None:
        self._path = path
        self._handle = handle
        self._history = history
        self._hasher = hasher
        self._tail_quarantine = tail_quarantine
        self._finalization_only = finalization_only
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
    def open_for_finalization(
        cls,
        path: str | Path,
        expected_current_state: JournalCheckpointState,
    ) -> AuthoritativeRunJournal:
        """Open an exact full journal prefix for its sole terminal ``run_end``."""
        if type(expected_current_state) is not JournalCheckpointState:
            raise TypeError("expected_current_state must be a JournalCheckpointState")
        target = Path(path)
        handle: BinaryIO | None = None
        try:
            handle = target.open("r+b")
            history, hasher = _replay(handle.read(), journal_path=target)
            actual = _checkpoint_state(history, hasher.hexdigest())
            for field in _STATE_FIELDS:
                if getattr(expected_current_state, field) != getattr(actual, field):
                    raise _error(
                        "finalization_state_mismatch",
                        field,
                        f"current journal {field} differs from expected_current_state",
                    )
            if history.terminal:
                raise _error(
                    "terminal_finalization",
                    "event_kind",
                    "terminal journal cannot be finalized again",
                )
            if history.last.event_kind not in _FINALIZABLE_TAIL:
                raise _error(
                    "nonfinalizable_tail",
                    "event_kind",
                    "finalizable journal must end in checkpoint or trainer_aborted",
                )
            handle.seek(0, os.SEEK_END)
            return cls(
                target,
                handle,
                history,
                hasher,
                finalization_only=True,
            )
        except BaseException:
            if handle is not None:
                handle.close()
            raise

    @classmethod
    def resume(
        cls,
        path: str | Path,
        checkpoint: JournalCheckpointState,
        *,
        checkpoint_sha256: str,
        segment_provenance: Mapping[str, FrozenJson],
        quarantine_path: Path | None = None,
    ) -> AuthoritativeRunJournal:
        if type(checkpoint) is not JournalCheckpointState:
            raise TypeError("checkpoint must be a JournalCheckpointState")
        if set(segment_provenance) != {"run_start_provenance", "trainer_wall_origin"}:
            raise _error(
                "resume_payload",
                "segment_provenance",
                "resume segment provenance keys must be exact",
            )
        actual_checkpoint_sha256 = _require_sha256(checkpoint_sha256, path="checkpoint_sha256")
        target = Path(path)
        handle: BinaryIO | None = None
        try:
            handle = target.open("r+b")
            data = handle.read()
            history, hasher = _replay(data, journal_path=target)
            if history.terminal:
                raise _error("terminal_resume", "event_kind", "terminal journal cannot resume")
            lines = data.splitlines(keepends=True)
            matching = [
                index
                for index, event in enumerate(history.events)
                if event.event_kind is JournalEventKind.CHECKPOINT
                and _checkpoint_payload(event) == actual_checkpoint_sha256
            ]
            if len(matching) != 1:
                raise _error(
                    "checkpoint_hash_mismatch",
                    "checkpoint_sha256",
                    "journal must contain exactly one matching checkpoint event",
                )
            checkpoint_index = matching[0]
            pre_history, pre_hasher = _replay(
                b"".join(lines[:checkpoint_index]), journal_path=target
            )
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
            checkpoint_event = history.events[checkpoint_index]
            if checkpoint_event.event_kind is not JournalEventKind.CHECKPOINT:
                raise _error(
                    "resume_without_checkpoint",
                    "event_kind",
                    "checkpoint event does not follow the checkpoint prefix",
                )
            if _checkpoint_payload(checkpoint_event) != actual_checkpoint_sha256:
                raise _error(
                    "checkpoint_hash_mismatch",
                    "checkpoint_sha256",
                    "journal checkpoint_sha256 differs from the checkpoint file",
                )
            tail = history.events[checkpoint_index + 1 :]
            if any(event.event_kind is JournalEventKind.CHECKPOINT for event in tail):
                raise _error(
                    "later_checkpoint",
                    "event_kind",
                    "selected checkpoint has a later checkpoint in the journal",
                )
            if any(event.event_kind not in _RECOVERABLE_TAIL for event in tail):
                raise _error(
                    "unsafe_tail",
                    "event_kind",
                    "post-checkpoint tail contains a nonrecoverable event",
                )
            recovered_prefix = b"".join(lines[: checkpoint_index + 1])
            quarantine: JournalTailQuarantineReceipt | None = None
            if tail:
                if quarantine_path is None:
                    raise _error(
                        "tail_quarantine_required",
                        "quarantine_path",
                        "post-checkpoint journal tail requires an explicit quarantine path",
                    )
                quarantine = _quarantine_tail(
                    original=data,
                    recovered_prefix=recovered_prefix,
                    tail=tail,
                    quarantine_path=quarantine_path,
                    journal_path=target,
                )
                history, hasher = _replay(recovered_prefix, journal_path=target)
            quarantine_evidence: FrozenJson = (
                quarantine.manifest if quarantine is not None else _no_tail_evidence()
            )
            resume_payload: dict[str, FrozenJson] = {
                "checkpoint_sha256": actual_checkpoint_sha256,
                "tail_quarantine": quarantine_evidence,
                **segment_provenance,
            }
            resume = D1JournalEvent(
                event_kind=JournalEventKind.RUN_RESUME,
                event_seq=checkpoint_event.event_seq + 1,
                logical_run_id=history.identity.logical_run_id,
                execution_digest=history.identity.execution_digest,
                aggregation_digest=history.identity.aggregation_digest,
                construction_digest=history.identity.construction_digest,
                family_digest=history.identity.family_digest,
                run_class=history.identity.run_class,
                successful_step=checkpoint_event.successful_step,
                payload=resume_payload,
            )
            advanced = _advance(history, resume, allow_resume=True)
            if quarantine is None:
                handle.seek(0, os.SEEK_END)
            else:
                handle.seek(len(recovered_prefix))
            journal = cls(target, handle, history, hasher, quarantine)
            journal._persist(resume, advanced, replace_tail=quarantine is not None)
            return journal
        except BaseException:
            if handle is not None:
                handle.close()
            raise

    @property
    def state(self) -> JournalCheckpointState:
        return _checkpoint_state(self._history, self._hasher.hexdigest())

    @property
    def tail_quarantine(self) -> JournalTailQuarantineReceipt | None:
        return self._tail_quarantine

    def append(self, event: D1JournalEvent) -> JournalCheckpointState:
        if self._closed:
            raise _error("closed_journal", "path", "journal is closed")
        if self._finalization_only and event.event_kind is not JournalEventKind.RUN_END:
            raise _error(
                "finalization_only",
                "event_kind",
                "finalization journal accepts only run_end",
            )
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
        if self._finalization_only:
            raise _error(
                "finalization_only",
                "event_kind",
                "finalization journal accepts only run_end",
            )
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

    def _persist(
        self,
        event: D1JournalEvent,
        advanced: _History,
        *,
        replace_tail: bool = False,
    ) -> None:
        line = canonical_event_line(event)
        try:
            written = self._handle.write(line)
            if written != len(line):
                raise OSError(f"short journal write: {written}/{len(line)} bytes")
            if replace_tail:
                self._handle.truncate()
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
