"""Durable append-only repository for canonical D1 lifecycle events."""

from __future__ import annotations

import fcntl
import hashlib
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from entropic_substrate.evidence.d1_lifecycle_values import (
    D1LifecycleEvent,
    D1LifecycleValueError,
    canonical_lifecycle_event_line,
    parse_lifecycle_event_line,
)
from entropic_substrate.evidence.lifecycle import ContentHash


class D1LifecycleGapCode(StrEnum):
    """Expected absence states, distinct from a damaged repository."""

    MISSING_RECORD = "missing_record"
    EMPTY_RECORD = "empty_record"


@dataclass(frozen=True, slots=True)
class D1LifecycleRepositoryGap:
    code: D1LifecycleGapCode
    path: Path

    def __post_init__(self) -> None:
        if type(self.code) is not D1LifecycleGapCode:
            raise TypeError("code must be a D1LifecycleGapCode")
        if not isinstance(cast(object, self.path), Path):
            raise TypeError("path must be a Path")


class D1LifecycleCorruptionError(RuntimeError):
    """Unexpected persisted-history corruption with a stable error code."""

    def __init__(self, *, code: str, path: Path, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = {"code": code, "path": str(path)}
        super().__init__(f"D1 lifecycle repository {code} at {path}: {detail}")


@dataclass(frozen=True, slots=True)
class D1LifecyclePrefix:
    """Exact byte and logical cursor for one durable lifecycle prefix."""

    event_count: int
    last_event_sequence: int
    byte_count: int
    prefix_sha256: str

    def __post_init__(self) -> None:
        if type(self.event_count) is not int or self.event_count < 1:
            raise ValueError("event_count must be a positive integer")
        if self.last_event_sequence != self.event_count - 1:
            raise ValueError("last_event_sequence must equal event_count minus one")
        if type(self.byte_count) is not int or self.byte_count < 1:
            raise ValueError("byte_count must be a positive integer")
        try:
            ContentHash("sha256", self.prefix_sha256)
        except ValueError as exc:
            raise ValueError("prefix_sha256 must be lowercase SHA-256") from exc


@dataclass(frozen=True, slots=True)
class D1LifecycleHistory:
    events: tuple[D1LifecycleEvent, ...]
    prefix: D1LifecyclePrefix

    def __post_init__(self) -> None:
        if len(self.events) != self.prefix.event_count:
            raise ValueError("history events must match the prefix event count")


def _corruption(code: str, path: Path, detail: str) -> D1LifecycleCorruptionError:
    return D1LifecycleCorruptionError(code=code, path=path, detail=detail)


def _parse_history(data: bytes, *, path: Path) -> D1LifecycleHistory:
    if not data:
        raise _corruption("empty_record", path, "a lifecycle history must contain an event")
    events: list[D1LifecycleEvent] = []
    identities: set[str] = set()
    artifacts: dict[str, ContentHash] = {}
    for index, line in enumerate(data.splitlines(keepends=True)):
        try:
            event = parse_lifecycle_event_line(line)
        except D1LifecycleValueError as exc:
            raise _corruption("invalid_event", path, f"event {index}: {exc}") from exc
        if event.event_seq != index:
            raise _corruption(
                "invalid_sequence",
                path,
                f"event {index} carries sequence {event.event_seq}",
            )
        if event.event_id in identities:
            raise _corruption("duplicate_event", path, f"event_id {event.event_id!r} repeats")
        identities.add(event.event_id)
        for artifact in event.artifacts:
            prior = artifacts.get(artifact.path)
            if prior is not None and prior != artifact.content_hash:
                raise _corruption(
                    "artifact_hash_drift",
                    path,
                    f"artifact {artifact.path!r} changed content identity",
                )
            artifacts[artifact.path] = artifact.content_hash
        events.append(event)
    prefix = D1LifecyclePrefix(
        event_count=len(events),
        last_event_sequence=events[-1].event_seq,
        byte_count=len(data),
        prefix_sha256=hashlib.sha256(data).hexdigest(),
    )
    return D1LifecycleHistory(events=tuple(events), prefix=prefix)


def _verify_prefix(data: bytes, expected: D1LifecyclePrefix, *, path: Path) -> bytes:
    if len(data) < expected.byte_count:
        raise _corruption("prefix_length_mismatch", path, "stored history is shorter than cursor")
    prefix_bytes = data[: expected.byte_count]
    if hashlib.sha256(prefix_bytes).hexdigest() != expected.prefix_sha256:
        raise _corruption("prefix_hash_mismatch", path, "stored prefix SHA-256 differs")
    history = _parse_history(prefix_bytes, path=path)
    if history.prefix != expected:
        raise _corruption("prefix_cursor_mismatch", path, "logical cursor differs from bytes")
    return prefix_bytes


class D1LifecycleRepository:
    """Serialize appends and prove every replay against canonical bytes."""

    def __init__(self, path: Path) -> None:
        if not isinstance(cast(object, path), Path):
            raise TypeError("D1 lifecycle repository path must be a Path")
        self.path = path

    def try_load(self) -> D1LifecycleHistory | D1LifecycleRepositoryGap:
        if not self.path.exists():
            return D1LifecycleRepositoryGap(D1LifecycleGapCode.MISSING_RECORD, self.path)
        if self.path.stat().st_size == 0:
            return D1LifecycleRepositoryGap(D1LifecycleGapCode.EMPTY_RECORD, self.path)
        return self.load()

    def load(self) -> D1LifecycleHistory:
        try:
            with self.path.open("rb") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                data = handle.read()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except FileNotFoundError as exc:
            raise _corruption("missing_record", self.path, "repository file is absent") from exc
        return _parse_history(data, path=self.path)

    def replay_prefix(self, expected: D1LifecyclePrefix) -> D1LifecycleHistory:
        if type(expected) is not D1LifecyclePrefix:
            raise TypeError("expected must be a D1LifecyclePrefix")
        try:
            with self.path.open("rb") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                data = handle.read()
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except FileNotFoundError as exc:
            raise _corruption("missing_record", self.path, "repository file is absent") from exc
        return _parse_history(_verify_prefix(data, expected, path=self.path), path=self.path)

    def append(
        self,
        event: D1LifecycleEvent,
        *,
        expected_prefix: D1LifecyclePrefix | None,
    ) -> D1LifecycleHistory:
        if type(event) is not D1LifecycleEvent:
            raise TypeError("event must be a D1LifecycleEvent")
        if expected_prefix is not None and type(expected_prefix) is not D1LifecyclePrefix:
            raise TypeError("expected_prefix must be a D1LifecyclePrefix or None")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        created = not self.path.exists()
        with self.path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            data = handle.read()
            self._admit_append(data, event=event, expected_prefix=expected_prefix)
            line = canonical_lifecycle_event_line(event)
            history = _parse_history(data + line, path=self.path)
            handle.seek(0, os.SEEK_END)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        if created:
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return history

    def _admit_append(
        self,
        data: bytes,
        *,
        event: D1LifecycleEvent,
        expected_prefix: D1LifecyclePrefix | None,
    ) -> None:
        if expected_prefix is None:
            if data:
                raise _corruption("prefix_mismatch", self.path, "existing history needs a cursor")
            if event.event_seq != 0:
                raise _corruption(
                    "invalid_sequence", self.path, "first event requires sequence zero"
                )
            return
        current = _parse_history(data, path=self.path)
        if current.prefix != expected_prefix:
            raise _corruption("prefix_mismatch", self.path, "history differs from append cursor")
        if event.event_seq != expected_prefix.event_count:
            raise _corruption(
                "invalid_sequence",
                self.path,
                "appended event sequence must follow the cursor",
            )
