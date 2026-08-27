"""Atomic repository adapter for D1 capture-preparation custody records."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_capture_preparation_custody import (
    capture_preparation_custody_from_preparation,
)
from entropic_substrate.evidence.d1_capture_preparation_custody_repository import (
    D1CapturePreparationCustodyRepository,
    D1CapturePreparationCustodyRepositoryError,
)
from entropic_substrate.evidence.lifecycle import ArtifactIdentity, ContentHash

from .test_d1_capture_preparation_custody import _case

pytestmark = pytest.mark.unit


def test_repository_saves_loads_and_reuses_exact_canonical_record(tmp_path: Path) -> None:
    preparation, _ = _case(tmp_path / "case")
    record = capture_preparation_custody_from_preparation(preparation)
    root = tmp_path / "custody"
    root.mkdir()
    repository = D1CapturePreparationCustodyRepository(root)

    first = repository.save(record)
    second = repository.save(record)
    restored = repository.load(first)

    assert first == second
    assert first.path == str(root / f"{record.record_sha256}.json")
    assert first.content_hash == ContentHash("sha256", record.record_sha256)
    assert hashlib.sha256(Path(first.path).read_bytes()).hexdigest() == record.record_sha256
    assert restored == record


def test_repository_refuses_substituted_artifact_path_or_hash(tmp_path: Path) -> None:
    preparation, _ = _case(tmp_path / "case")
    record = capture_preparation_custody_from_preparation(preparation)
    root = tmp_path / "custody"
    root.mkdir()
    repository = D1CapturePreparationCustodyRepository(root)
    artifact = repository.save(record)

    for candidate in (
        replace(artifact, content_hash=ContentHash("sha256", "0" * 64)),
        ArtifactIdentity(str(tmp_path / "elsewhere.json"), artifact.content_hash),
    ):
        with pytest.raises(D1CapturePreparationCustodyRepositoryError, match="artifact"):
            repository.load(candidate)


def test_repository_refuses_symlink_root_and_existing_collision(tmp_path: Path) -> None:
    preparation, _ = _case(tmp_path / "case")
    record = capture_preparation_custody_from_preparation(preparation)
    root = tmp_path / "custody"
    root.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(root, target_is_directory=True)

    with pytest.raises(D1CapturePreparationCustodyRepositoryError, match="root"):
        D1CapturePreparationCustodyRepository(linked)

    target = root / f"{record.record_sha256}.json"
    target.write_bytes(b"wrong bytes")
    before = target.read_bytes()
    with pytest.raises(D1CapturePreparationCustodyRepositoryError, match="collision"):
        D1CapturePreparationCustodyRepository(root).save(record)
    assert target.read_bytes() == before


def test_repository_refuses_wrong_value_type_and_closed_store(tmp_path: Path) -> None:
    root = tmp_path / "custody"
    root.mkdir()
    repository = D1CapturePreparationCustodyRepository(root)
    with pytest.raises(TypeError, match="nominal"):
        repository.save(object())  # type: ignore[arg-type]
    repository.close()
    with pytest.raises(D1CapturePreparationCustodyRepositoryError, match="closed"):
        repository.load(
            ArtifactIdentity(str(root / f"{'0' * 64}.json"), ContentHash("sha256", "0" * 64))
        )
