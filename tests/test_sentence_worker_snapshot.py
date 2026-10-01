"""Output snapshots prove bounded byte custody, not workload completion."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_request import request

from entropic_substrate.ladder.run.worker_snapshot import (
    materialize_worker_snapshot,
    publish_worker_snapshot,
)

pytestmark = pytest.mark.unit


def test_a_ledgered_path_is_not_reread_or_republished_on_a_later_snapshot(
    tmp_path: Path,
) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run/a.pt").write_bytes(b"alpha")
    (tmp_path / "run/b.pt").write_bytes(b"beta")
    value, objects = request(), _ObjectStore()
    ledger: dict[str, tuple[str, int]] = {}
    first = publish_worker_snapshot(
        objects, value, tmp_path, ("run/a.pt",), name="checkpoint-1", ledger=ledger
    )
    assert objects.publishes == [f"{value.prefix}/outputs/run/a.pt", first.key]
    puts_before = list(objects.publishes)
    second = publish_worker_snapshot(
        objects,
        value,
        tmp_path,
        ("run/a.pt", "run/b.pt"),
        name="checkpoint-2",
        ledger=ledger,
    )
    # Only the NEW path (b.pt) and the new marker were published; a.pt's key
    # from checkpoint-1 does not reappear in the publish log.
    assert objects.publishes == [
        *puts_before,
        f"{value.prefix}/outputs/run/b.pt",
        second.key,
    ]
    manifest = json.loads(objects.values[second.key])
    assert sorted(item["path"] for item in manifest["files"]) == ["run/a.pt", "run/b.pt"]
    assert set(ledger) == {f"{value.prefix}/outputs/run/a.pt", f"{value.prefix}/outputs/run/b.pt"}


def test_ledger_manifest_bytes_are_identical_with_and_without_a_hit(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run/a.pt").write_bytes(b"alpha")
    value = request()
    without = publish_worker_snapshot(
        _ObjectStore(), value, tmp_path, ("run/a.pt",), name="checkpoint-1"
    )
    ledger: dict[str, tuple[str, int]] = {}
    fresh_store = _ObjectStore()
    publish_worker_snapshot(
        fresh_store, value, tmp_path, ("run/a.pt",), name="checkpoint-1", ledger=ledger
    )
    with_hit_store = _ObjectStore()
    with_hit = publish_worker_snapshot(
        with_hit_store, value, tmp_path, ("run/a.pt",), name="checkpoint-1", ledger=ledger
    )
    assert with_hit == without
    assert with_hit_store.values[with_hit.key] == fresh_store.values[without.key]


def test_a_ledger_entry_mismatch_raises_before_any_network_call(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run/a.pt").write_bytes(b"alpha")
    value, objects = request(), _ObjectStore()
    ledger: dict[str, tuple[str, int]] = {
        f"{value.prefix}/outputs/run/a.pt": ("0" * 64, 999),
    }
    with pytest.raises(ValueError, match="changed"):
        publish_worker_snapshot(
            objects, value, tmp_path, ("run/a.pt",), name="checkpoint-1", ledger=ledger
        )
    assert objects.publishes == []


def test_ledger_entry_is_written_only_after_verification_succeeds(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run/a.pt").write_bytes(b"alpha")
    value = request()

    class DroppingStore(_ObjectStore):
        def read(self, key):  # type: ignore[override]
            if key == f"{value.prefix}/outputs/run/a.pt":
                return None
            return super().read(key)

    ledger: dict[str, tuple[str, int]] = {}
    with pytest.raises(ValueError, match="readback"):
        publish_worker_snapshot(
            DroppingStore(), value, tmp_path, ("run/a.pt",), name="checkpoint-1", ledger=ledger
        )
    assert ledger == {}


def test_snapshot_roundtrip_and_marker_last(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run/checkpoint.pt").write_bytes(b"transport fixture")
    value, objects = request(), _ObjectStore()
    marker = publish_worker_snapshot(
        objects, value, tmp_path, ("run/checkpoint.pt",), name="checkpoint-2"
    )
    assert objects.publishes[-1] == marker.key
    target = tmp_path / "host"
    target.mkdir()
    result = materialize_worker_snapshot(objects, value, marker, target, max_marker_bytes=1_000_000)
    assert result["authority"] == "byte_custody_only"
    assert (target / "run/checkpoint.pt").read_bytes() == b"transport fixture"
    with pytest.raises(ValueError, match="empty"):
        materialize_worker_snapshot(objects, value, marker, target, max_marker_bytes=1_000_000)


def test_corrupt_object_never_materializes_ready_marker(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run/checkpoint.pt").write_bytes(b"transport fixture")
    value, objects = request(), _ObjectStore()
    marker = publish_worker_snapshot(
        objects, value, tmp_path, ("run/checkpoint.pt",), name="checkpoint-2"
    )
    objects.values[f"{value.prefix}/outputs/run/checkpoint.pt"] = b"changed"
    target = tmp_path / "host"
    target.mkdir()
    with pytest.raises(ValueError, match="identity"):
        materialize_worker_snapshot(objects, value, marker, target, max_marker_bytes=1_000_000)
    assert not (target / "snapshot.json").exists()


def test_snapshot_budget_and_path_guards_precede_publication(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run/checkpoint.pt").write_bytes(b"transport fixture")
    value, objects = request(), _ObjectStore()
    tiny = replace(value, limits=replace(value.limits, max_total_bytes=1))
    with pytest.raises(ValueError, match="budget"):
        publish_worker_snapshot(
            objects, tiny, tmp_path, ("run/checkpoint.pt",), name="checkpoint-2"
        )
    for paths in (
        ("../escape",),
        ("source/prepared-corpus.json",),
        ("run/checkpoint.pt", "run/checkpoint.pt"),
    ):
        with pytest.raises(ValueError):
            publish_worker_snapshot(objects, value, tmp_path, paths, name="checkpoint-2")
    assert objects.publishes == []


@pytest.mark.parametrize("field", ["artifact_bytes", "artifact_count", "request_sha256"])
def test_rehashed_manifest_cannot_change_inventory_or_request(tmp_path: Path, field: str) -> None:
    (tmp_path / "run").mkdir()
    (tmp_path / "run/checkpoint.pt").write_bytes(b"transport fixture")
    value, objects = request(), _ObjectStore()
    marker = publish_worker_snapshot(
        objects, value, tmp_path, ("run/checkpoint.pt",), name="checkpoint-2"
    )
    row = json.loads(objects.values[marker.key])
    row[field] = "0" * 64 if field == "request_sha256" else row[field] + 1
    payload = json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
    objects.values[marker.key] = payload
    forged = objects._identity(marker.key, payload)
    target = tmp_path / "host"
    target.mkdir()
    with pytest.raises(ValueError, match=r"identity|inventory"):
        materialize_worker_snapshot(objects, value, forged, target, max_marker_bytes=1_000_000)
    assert not list(target.iterdir())


def test_distinct_snapshots_share_immutable_outputs_but_cannot_replace_them(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    path = tmp_path / "run/checkpoint.pt"
    path.write_bytes(b"original")
    value, objects = request(), _ObjectStore()
    first = publish_worker_snapshot(objects, value, tmp_path, ("run/checkpoint.pt",), name="first")
    second = publish_worker_snapshot(
        objects, value, tmp_path, ("run/checkpoint.pt",), name="second"
    )
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="immutable"):
        publish_worker_snapshot(objects, value, tmp_path, ("run/checkpoint.pt",), name="changed")
    assert f"{value.prefix}/snapshots/changed.json" not in objects.values
    for index, marker in enumerate((first, second)):
        target = tmp_path / f"host-{index}"
        target.mkdir()
        materialize_worker_snapshot(objects, value, marker, target, max_marker_bytes=1_000_000)
        assert (target / "run/checkpoint.pt").read_bytes() == b"original"


def test_late_corruption_retains_unadmitted_partial_files_and_requires_new_workspace(
    tmp_path: Path,
) -> None:
    (tmp_path / "run").mkdir()
    for name in ("a", "b"):
        (tmp_path / f"run/{name}.pt").write_bytes(name.encode())
    value, objects = request(), _ObjectStore()
    marker = publish_worker_snapshot(
        objects, value, tmp_path, ("run/a.pt", "run/b.pt"), name="pair"
    )
    key = f"{value.prefix}/outputs/run/b.pt"
    original = objects.values[key]
    objects.values[key] = b"corrupt"
    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(ValueError, match="identity"):
        materialize_worker_snapshot(objects, value, marker, partial, max_marker_bytes=1_000_000)
    assert (partial / "run/a.pt").read_bytes() == b"a"
    assert not (partial / "snapshot.json").exists()
    objects.values[key] = original
    with pytest.raises(ValueError, match="empty"):
        materialize_worker_snapshot(objects, value, marker, partial, max_marker_bytes=1_000_000)
    complete = tmp_path / "complete"
    complete.mkdir()
    materialize_worker_snapshot(objects, value, marker, complete, max_marker_bytes=1_000_000)
    assert (complete / "snapshot.json").is_file()
