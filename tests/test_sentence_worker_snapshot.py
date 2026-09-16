"""Output snapshots prove bounded byte custody, not workload completion."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_request import request

from entropic_substrate.sentence_prediction.worker_snapshot import (
    materialize_worker_snapshot,
    publish_worker_snapshot,
)

pytestmark = pytest.mark.unit


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
