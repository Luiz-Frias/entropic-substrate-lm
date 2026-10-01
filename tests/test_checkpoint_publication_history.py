"""FOR-1567: checkpoint publication independent of run history.

Cross-cutting acceptance for the pre-for-1567 rung (brief: checkpoint
publication cost must be O(new objects), not O(k); a bounded retry covers
R2 verification reads only, never a PUT). Per-module unit tests for each
mechanism live beside the module they change:

- ``tests/test_sentence_worker_snapshot.py`` -- the publication ledger (L3).
- ``tests/test_sentence_worker_transport.py`` -- the bounded verification
  read retry, including the real adapter's own post-PUT readback (C6/C7).
- ``tests/test_sentence_seals.py`` -- the seal memo (L1/L2), its constant
  per-checkpoint ancestor-read count, its fresh-process control arm, and
  the tampered-ancestor-at-resume guard.
- ``tests/test_h_probe_events.py`` -- the ``publication`` block's read-old/
  write-new reader door.

This file holds the ACCEPTANCE bullet that names a counting fake object
port swept over k = 1, 2, 3, 5 checkpoints, independent of k.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_request import request

from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectIdentity,
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.worker_snapshot import (
    materialize_worker_snapshot,
    publish_worker_snapshot,
)
from entropic_substrate.ladder.run.worker_transport import VerificationReadRetry

pytestmark = pytest.mark.unit


@dataclass
class _CountingObjectStore(_ObjectStore):
    """An ``_ObjectStore`` that also counts ``read`` calls, per key."""

    reads: list[str] = field(default_factory=list)

    def read(self, key: str) -> D1TrainerMailboxObjectRead | None:
        self.reads.append(key)
        return super().read(key)


def _grow_and_publish(
    tmp_path: Path, objects: _CountingObjectStore, ledger: dict[str, tuple[str, int]], step: int
) -> tuple[int, int, D1TrainerMailboxObjectIdentity]:
    """Add ONE new accumulated output file, publish that checkpoint's
    snapshot, and return (new_puts, new_reads, marker)."""
    (tmp_path / f"run/step-{step:02d}.pt").write_bytes(f"checkpoint payload {step}".encode())
    paths = tuple(f"run/step-{i:02d}.pt" for i in range(1, step + 1))
    value = request()
    puts_before, reads_before = len(objects.publishes), len(objects.reads)
    retry = VerificationReadRetry()
    marker = publish_worker_snapshot(
        objects,
        value,
        tmp_path,
        paths,
        name=f"checkpoint-{step:08d}",
        ledger=ledger,
        retry=retry,
    )
    return len(objects.publishes) - puts_before, len(objects.reads) - reads_before, marker


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_puts_at_checkpoint_k_equal_new_files_plus_marker_independent_of_k(
    tmp_path: Path, k: int
) -> None:
    (tmp_path / "run").mkdir()
    objects = _CountingObjectStore()
    ledger: dict[str, tuple[str, int]] = {}
    put_counts: list[int] = []
    read_counts: list[int] = []
    for step in range(1, k + 1):
        puts, reads, _ = _grow_and_publish(tmp_path, objects, ledger, step)
        put_counts.append(puts)
        read_counts.append(reads)
    # Exactly ONE new output file plus ONE marker are PUT at every
    # checkpoint, regardless of how much history (k) precedes it.
    assert put_counts == [2] * k
    # Verification reads (bounded by new files: the new output + the
    # marker) are likewise independent of k.
    assert read_counts == [2] * k


def test_puts_are_republished_when_the_ledger_hit_is_dropped_mutation_control(
    tmp_path: Path,
) -> None:
    """Same scenario as above WITHOUT a ledger: the historical, O(k)
    behaviour this rung replaces. Pins the contrast the acceptance bullet's
    mutation ("restore republish-all") targets."""
    (tmp_path / "run").mkdir()
    objects = _CountingObjectStore()
    put_counts: list[int] = []
    for step in range(1, 6):
        (tmp_path / f"run/step-{step:02d}.pt").write_bytes(f"checkpoint payload {step}".encode())
        paths = tuple(f"run/step-{i:02d}.pt" for i in range(1, step + 1))
        value = request()
        before = len(objects.publishes)
        publish_worker_snapshot(
            objects, value, tmp_path, paths, name=f"checkpoint-{step:08d}"
        )  # no ledger: today's behaviour
        put_counts.append(len(objects.publishes) - before)
    # WITHOUT a ledger, checkpoint k republishes every accumulated file:
    # k new-PUT files plus the marker.
    assert put_counts == [2, 3, 4, 5, 6]


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_the_manifest_at_checkpoint_k_still_lists_every_file(tmp_path: Path, k: int) -> None:
    (tmp_path / "run").mkdir()
    objects = _CountingObjectStore()
    ledger: dict[str, tuple[str, int]] = {}
    marker = None
    for step in range(1, k + 1):
        _, _, marker = _grow_and_publish(tmp_path, objects, ledger, step)
    assert marker is not None
    value = request()
    target = tmp_path / "materialized"
    target.mkdir()
    result = materialize_worker_snapshot(objects, value, marker, target, max_marker_bytes=1_000_000)
    assert result["artifact_count"] == k
    for step in range(1, k + 1):
        assert (
            target / f"run/step-{step:02d}.pt"
        ).read_bytes() == f"checkpoint payload {step}".encode()


def test_a_path_whose_bytes_change_after_publication_raises_before_any_put(
    tmp_path: Path,
) -> None:
    (tmp_path / "run").mkdir()
    objects = _CountingObjectStore()
    ledger: dict[str, tuple[str, int]] = {}
    for step in (1, 2):
        _grow_and_publish(tmp_path, objects, ledger, step)
    # Now mutate an EARLIER, already-ledgered file's bytes in place.
    (tmp_path / "run/step-01.pt").write_bytes(b"tampered after the fact")
    puts_before = len(objects.publishes)
    with pytest.raises(ValueError, match="changed"):
        _grow_and_publish(tmp_path, objects, ledger, 3)
    # No PUT for the new step-03 path or the new marker happened either:
    # the ledger mismatch is caught in pass 1, before any network call.
    assert len(objects.publishes) == puts_before
