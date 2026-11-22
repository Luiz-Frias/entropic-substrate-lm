"""One strict DVC pointer reader serves cadence sync and final capture."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from entropic_substrate.ladder.capture.custody.d1_capture_ports import D1CaptureTransportError
from entropic_substrate.ladder.capture.custody.d1_capture_service import read_d1_dvc_pointer

pytestmark = pytest.mark.unit

_OBJECT = "a" * 32 + ".dir"


def _case(tmp_path: Path) -> tuple[Path, Path, Path, bytes]:
    repo = tmp_path / "repo"
    tracked = repo / "results/d1-run-abc"
    tracked.mkdir(parents=True)
    pointer = repo / "results/d1-run-abc.dvc"
    payload = f"outs:\n- md5: {_OBJECT}\n  size: 42\n  nfiles: 2\n  path: d1-run-abc\n".encode()
    pointer.write_bytes(payload)
    return repo, tracked, pointer, payload


def test_generic_pointer_reader_returns_exact_bytes_and_object_identity(tmp_path: Path) -> None:
    repo, tracked, pointer, payload = _case(tmp_path)

    identity, observed = read_d1_dvc_pointer(
        repo_root=repo,
        tracked_path=tracked,
        pointer_path=pointer,
    )

    assert observed == payload
    assert identity.pointer_path == "results/d1-run-abc.dvc"
    assert identity.pointer_sha256 == hashlib.sha256(payload).hexdigest()
    assert identity.object_id == _OBJECT


@pytest.mark.parametrize(
    "payload",
    [
        b"[]\n",
        b"outs: []\n",
        b"outs:\n- md5: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.dir\n  path: foreign\n",
        b"outs:\n- path: d1-run-abc\n",
        b"outs:\n- md5: invalid\n  path: d1-run-abc\n",
    ],
)
def test_generic_pointer_reader_rejects_malformed_or_substituted_pointer(
    tmp_path: Path,
    payload: bytes,
) -> None:
    repo, tracked, pointer, _ = _case(tmp_path)
    pointer.write_bytes(payload)

    with pytest.raises(D1CaptureTransportError) as raised:
        read_d1_dvc_pointer(repo_root=repo, tracked_path=tracked, pointer_path=pointer)

    assert raised.value.code == "pointer_identity"


def test_generic_pointer_reader_refuses_symlink_and_wrong_pointer_relationship(
    tmp_path: Path,
) -> None:
    repo, tracked, pointer, _ = _case(tmp_path)
    outside = tmp_path / "outside.dvc"
    outside.write_bytes(pointer.read_bytes())
    pointer.unlink()
    pointer.symlink_to(outside)
    with pytest.raises(D1CaptureTransportError):
        read_d1_dvc_pointer(repo_root=repo, tracked_path=tracked, pointer_path=pointer)

    with pytest.raises(D1CaptureTransportError) as relationship:
        read_d1_dvc_pointer(
            repo_root=repo,
            tracked_path=tracked,
            pointer_path=repo / "results/other.dvc",
        )
    assert relationship.value.code == "pointer_path"


def test_generic_pointer_reader_refuses_paths_outside_repository(tmp_path: Path) -> None:
    repo, _tracked, pointer, _ = _case(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(D1CaptureTransportError) as raised:
        read_d1_dvc_pointer(
            repo_root=repo,
            tracked_path=outside,
            pointer_path=pointer,
        )
    assert raised.value.code == "pointer_path"
