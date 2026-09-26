"""Durability and replay tests for the D1 lifecycle repository."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.kernel.lifecycle import (
    ActorIdentity,
    ActorRole,
    ArtifactIdentity,
    ContentHash,
)
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_repository import (
    D1LifecycleCorruptionError,
    D1LifecycleGapCode,
    D1LifecyclePrefix,
    D1LifecycleRepository,
    D1LifecycleRepositoryGap,
)
from entropic_substrate.ladder.capture.ledger.d1_lifecycle_values import (
    D1LifecycleEvent,
    D1ProcessEventKind,
)


def _event(sequence: int, *, artifact_index: int | None = None) -> D1LifecycleEvent:
    index = sequence if artifact_index is None else artifact_index
    return D1LifecycleEvent(
        event_seq=sequence,
        event_id=f"event-{sequence}",
        event_kind=(
            D1ProcessEventKind.CONTRACT_VERIFIED
            if sequence == 0
            else D1ProcessEventKind.PILOT_FILED
        ),
        actor=ActorIdentity("operator", "session", ActorRole.OPERATOR),
        artifacts=(
            ArtifactIdentity(
                path=f"artifacts/{index}.json",
                content_hash=ContentHash("sha256", f"{index + 1:064x}"),
            ),
        ),
    )


def test_missing_repository_is_a_typed_gap(tmp_path: Path) -> None:
    repository = D1LifecycleRepository(tmp_path / "state" / "d1.jsonl")

    result = repository.try_load()

    assert result == D1LifecycleRepositoryGap(
        code=D1LifecycleGapCode.MISSING_RECORD,
        path=repository.path,
    )


def test_append_durably_builds_canonical_history(tmp_path: Path) -> None:
    repository = D1LifecycleRepository(tmp_path / "d1.jsonl")

    first = repository.append(_event(0), expected_prefix=None)
    second = repository.append(_event(1), expected_prefix=first.prefix)

    assert second.events == (_event(0), _event(1))
    assert second.prefix.event_count == 2
    assert second.prefix.last_event_sequence == 1
    assert second.prefix.byte_count == repository.path.stat().st_size
    assert len(second.prefix.prefix_sha256) == 64
    assert repository.load() == second


def test_append_rejects_stale_or_forged_prefix(tmp_path: Path) -> None:
    repository = D1LifecycleRepository(tmp_path / "d1.jsonl")
    first = repository.append(_event(0), expected_prefix=None)
    repository.append(_event(1), expected_prefix=first.prefix)

    with pytest.raises(D1LifecycleCorruptionError) as error:
        repository.append(_event(2), expected_prefix=first.prefix)

    assert error.value.code == "prefix_mismatch"


def test_replay_prefix_verifies_exact_historical_bytes_despite_tail(tmp_path: Path) -> None:
    repository = D1LifecycleRepository(tmp_path / "d1.jsonl")
    first = repository.append(_event(0), expected_prefix=None)
    repository.append(_event(1), expected_prefix=first.prefix)

    replay = repository.replay_prefix(first.prefix)

    assert replay.events == (_event(0),)
    assert replay.prefix == first.prefix


def test_replay_prefix_fails_on_byte_replacement(tmp_path: Path) -> None:
    repository = D1LifecycleRepository(tmp_path / "d1.jsonl")
    first = repository.append(_event(0), expected_prefix=None)
    payload = repository.path.read_bytes()
    repository.path.write_bytes(payload.replace(b"operator", b"operat0r", 1))

    with pytest.raises(D1LifecycleCorruptionError) as error:
        repository.replay_prefix(first.prefix)

    assert error.value.code == "prefix_hash_mismatch"


def test_load_fails_hard_on_partial_or_noncanonical_event(tmp_path: Path) -> None:
    repository = D1LifecycleRepository(tmp_path / "d1.jsonl")
    repository.path.write_bytes(b'{"partial":true}')

    with pytest.raises(D1LifecycleCorruptionError) as error:
        repository.load()

    assert error.value.code == "invalid_event"


def test_load_fails_hard_on_sequence_fork(tmp_path: Path) -> None:
    repository = D1LifecycleRepository(tmp_path / "d1.jsonl")
    first = repository.append(_event(0), expected_prefix=None)
    from entropic_substrate.ladder.capture.ledger.d1_lifecycle_values import (
        canonical_lifecycle_event_line,
    )

    repository.path.write_bytes(
        repository.path.read_bytes() + canonical_lifecycle_event_line(_event(0, artifact_index=9))
    )

    with pytest.raises(D1LifecycleCorruptionError) as error:
        repository.load()

    assert error.value.code == "invalid_sequence"
    assert repository.replay_prefix(first.prefix).events == (_event(0),)


def test_empty_existing_repository_is_a_typed_gap(tmp_path: Path) -> None:
    path = tmp_path / "d1.jsonl"
    path.touch()
    repository = D1LifecycleRepository(path)

    result = repository.try_load()

    assert isinstance(result, D1LifecycleRepositoryGap)
    assert result.code is D1LifecycleGapCode.EMPTY_RECORD


def test_prefix_value_rejects_impossible_cursor() -> None:
    with pytest.raises(ValueError, match="event_count"):
        D1LifecyclePrefix(
            event_count=0,
            last_event_sequence=0,
            byte_count=1,
            prefix_sha256="0" * 64,
        )
