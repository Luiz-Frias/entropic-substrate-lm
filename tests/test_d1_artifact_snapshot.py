"""D1 artifact snapshots bind root containment and bytes to one descriptor."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from entropic_substrate.kernel.artifacts import (
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    publish_d1_atomic_bytes,
    relative_d1_artifact_path,
    snapshot_d1_artifact,
)

pytestmark = pytest.mark.unit


def test_nested_regular_file_snapshot_binds_path_size_and_digest(tmp_path: Path) -> None:
    root = tmp_path / "exports"
    nested = root / "run"
    nested.mkdir(parents=True)
    artifact = nested / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint-bytes")

    normalized = normalize_d1_artifact_root(root)
    relative = relative_d1_artifact_path(normalized, artifact, label="checkpoint")
    path, size, digest = snapshot_d1_artifact(
        normalized,
        relative,
        label="checkpoint",
    )

    assert relative == "run/checkpoint.pt"
    assert path == artifact
    assert size == len(b"checkpoint-bytes")
    assert digest == hashlib.sha256(b"checkpoint-bytes").hexdigest()


@pytest.mark.parametrize("kind", ["root", "directory", "file"])
def test_snapshot_refuses_symlinked_root_directory_and_file(
    tmp_path: Path,
    kind: str,
) -> None:
    real_root = tmp_path / "real"
    nested = real_root / "run"
    nested.mkdir(parents=True)
    artifact = nested / "checkpoint.pt"
    artifact.write_bytes(b"checkpoint")
    root = real_root
    relative = "run/checkpoint.pt"
    if kind == "root":
        root = tmp_path / "root-link"
        root.symlink_to(real_root, target_is_directory=True)
    elif kind == "directory":
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "checkpoint.pt").write_bytes(b"outside")
        (real_root / "linked").symlink_to(outside, target_is_directory=True)
        relative = "linked/checkpoint.pt"
    else:
        link = nested / "checkpoint-link.pt"
        link.symlink_to(artifact)
        relative = "run/checkpoint-link.pt"

    with pytest.raises(ValueError, match=r"symlink|open"):
        snapshot_d1_artifact(root, relative, label="checkpoint")


@pytest.mark.parametrize(
    "value",
    ["", ".", "../escape", "/absolute", "nested\\windows", "nested/../escape"],
)
def test_relative_artifact_path_refuses_empty_noncanonical_and_escape(value: str) -> None:
    with pytest.raises(ValueError, match=r"relative|path"):
        normalize_d1_relative_artifact_path(value, label="checkpoint")


def test_relative_derivation_refuses_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="escapes"):
        relative_d1_artifact_path(root, outside, label="checkpoint")


def test_atomic_publication_writes_once_and_returns_exact_artifact_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exports"
    (root / "run").mkdir(parents=True)
    payload = b'{"schema_version":"d1-output-v1"}'

    artifact = publish_d1_atomic_bytes(
        root,
        "run/handoff.json",
        payload,
        label="handoff",
    )

    target = root / "run/handoff.json"
    assert target.read_bytes() == payload
    assert artifact.path == str(target)
    assert artifact.content_hash.algorithm == "sha256"
    assert artifact.content_hash.digest == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="occupied"):
        publish_d1_atomic_bytes(root, "run/handoff.json", b"different", label="handoff")
    assert target.read_bytes() == payload


def test_partial_publication_is_removed_before_target_becomes_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "exports"
    root.mkdir()
    real_write = os.write
    calls = 0

    def short_then_stall(descriptor: int, data: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, memoryview(data)[:2])
        return 0

    monkeypatch.setattr(os, "write", short_then_stall)

    with pytest.raises(ValueError, match="partial"):
        publish_d1_atomic_bytes(root, "handoff.json", b"payload", label="handoff")

    assert not (root / "handoff.json").exists()
    assert list(root.iterdir()) == []
