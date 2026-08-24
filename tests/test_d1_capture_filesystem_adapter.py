"""Direct-custody filesystem adapter for D1 evidence export."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactRole,
    identify_d1_artifact,
    verify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_filesystem import (
    LocalD1EvidenceExportSource,
)
from entropic_substrate.evidence.d1_capture_ports import D1CaptureTransportError

pytestmark = pytest.mark.unit

_POINTER = "results/captured.dvc"


def _identity(root: Path, relative: str, payload: bytes = b'{"event":"run_end"}\n'):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return identify_d1_artifact(
        root,
        path,
        role=D1ArtifactRole.JOURNAL,
        dvc_pointer_path=_POINTER,
    )


def test_local_export_copies_exact_bytes_and_rewrites_an_identical_existing_file(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "runpod-export"
    source_root.mkdir()
    identity = _identity(source_root, "journals/d1-none-s7.jsonl")
    destination_root = tmp_path / "repo" / "results" / "captured"
    destination = destination_root / identity.normalized_path
    destination.parent.mkdir(parents=True)
    destination.write_bytes((source_root / identity.normalized_path).read_bytes())
    previous_inode = destination.stat().st_ino

    exported = LocalD1EvidenceExportSource(source_root).export(
        artifacts=(identity,),
        destination_root=destination_root,
    )

    assert exported == (identity,)
    assert destination.read_bytes() == (source_root / identity.normalized_path).read_bytes()
    assert destination.stat().st_ino != previous_inode
    assert verify_d1_artifact(destination_root, identity) == identity


def test_local_export_refuses_existing_destination_drift_without_overwriting_it(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "local-export"
    source_root.mkdir()
    identity = _identity(source_root, "d1-sg-s7.jsonl")
    destination_root = tmp_path / "repo" / "results" / "captured"
    destination_root.mkdir(parents=True)
    destination = destination_root / identity.normalized_path
    destination.write_bytes(b"different-existing-evidence\n")

    with pytest.raises(D1CaptureTransportError) as captured:
        LocalD1EvidenceExportSource(source_root).export(
            artifacts=(identity,),
            destination_root=destination_root,
        )

    assert captured.value.code == "destination_drift"
    assert captured.value.path == identity.normalized_path
    assert destination.read_bytes() == b"different-existing-evidence\n"


@pytest.mark.parametrize("mutation", ["changed", "missing"])
def test_local_export_refuses_source_identity_drift_or_disappearance(
    tmp_path: Path, mutation: str
) -> None:
    source_root = tmp_path / "runpod-export"
    source_root.mkdir()
    identity = _identity(source_root, "d1-ac-s7.jsonl")
    source = source_root / identity.normalized_path
    if mutation == "changed":
        source.write_bytes(b"post-manifest-drift\n")
    else:
        source.unlink()

    with pytest.raises(D1CaptureTransportError) as captured:
        LocalD1EvidenceExportSource(source_root).export(
            artifacts=(identity,),
            destination_root=tmp_path / "repo" / "results" / "captured",
        )

    assert captured.value.code == "source_artifact"
    assert captured.value.path == identity.normalized_path


def test_local_export_refuses_a_symlinked_destination_parent(tmp_path: Path) -> None:
    source_root = tmp_path / "runpod-export"
    source_root.mkdir()
    identity = _identity(source_root, "journals/d1-both-s7.jsonl")
    destination_root = tmp_path / "repo" / "results" / "captured"
    destination_root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination_root / "journals").symlink_to(outside, target_is_directory=True)

    with pytest.raises(D1CaptureTransportError) as captured:
        LocalD1EvidenceExportSource(source_root).export(
            artifacts=(identity,),
            destination_root=destination_root,
        )

    assert captured.value.code == "destination_boundary"
    assert not (outside / "d1-both-s7.jsonl").exists()
