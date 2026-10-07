"""D1 artifact snapshots bind root containment and bytes to one descriptor."""

from __future__ import annotations

import errno
import hashlib
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from entropic_substrate.kernel.artifacts import (
    D1ArtifactSnapshotError,
    normalize_d1_artifact_directory,
    normalize_d1_artifact_root,
    normalize_d1_relative_artifact_path,
    publish_d1_atomic_bytes,
    read_d1_artifact_bytes,
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


@dataclass
class _Ledger:
    """Per-descriptor open/closed state machine over one guarded call."""

    live: set[int] = field(default_factory=set[int])
    opened: int = 0
    double_closes: list[int] = field(default_factory=list[int])


@contextmanager
def _accounted(fail_on: str | None) -> Iterator[_Ledger]:
    """Count every os.open/os.close; fail the dir_fd-relative open of one named component.

    A close of a non-live descriptor is recorded and NOT forwarded, because the next open in the
    walk reuses the number. Descriptors still live at exit are reported, then really closed.
    """
    ledger = _Ledger()
    real_open, real_close = os.open, os.close

    def tracked_open(path: str, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if fail_on is not None and dir_fd is not None and path == fail_on:
            raise OSError(errno.EACCES, "injected", path)
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        ledger.live.add(descriptor)
        ledger.opened += 1
        return descriptor

    def tracked_close(descriptor: int) -> None:
        if descriptor not in ledger.live:
            ledger.double_closes.append(descriptor)
            return
        ledger.live.remove(descriptor)
        real_close(descriptor)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "open", tracked_open)
        patch.setattr(os, "close", tracked_close)
        try:
            yield ledger
        finally:
            patch.undo()
            leaked = sorted(ledger.live)
            for descriptor in leaked:
                real_close(descriptor)
            ledger.live.update(leaked)


_COMPONENTS = ("dir-one", "dir-two", "dir-three")
# Step 2 and the last step of a walk over three directory components; neither is the first.
_FAULT_STEPS = [pytest.param("dir-two", id="step-2"), pytest.param("dir-three", id="last-step")]


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "exports"
    root.joinpath(*_COMPONENTS).mkdir(parents=True)
    return root


def _injected(name: str) -> str:
    return str(OSError(errno.EACCES, "injected", name))


def _injected_cause(error: BaseException) -> None:
    cause = error.__cause__
    assert isinstance(cause, OSError)
    assert cause.errno == errno.EACCES
    assert cause.strerror == "injected"


def _balanced(ledger: _Ledger) -> None:
    assert ledger.double_closes == []
    assert ledger.live == set()


def _site_root(tmp_path: Path) -> tuple[Callable[[], object], str]:
    root = _tree(tmp_path) / _COMPONENTS[0] / _COMPONENTS[1] / _COMPONENTS[2]
    return lambda: normalize_d1_artifact_root(
        root
    ), "D1 artifact root could not be opened without a symlink"


def _site_read(tmp_path: Path) -> tuple[Callable[[], object], str]:
    root = _tree(tmp_path)
    (root.joinpath(*_COMPONENTS) / "file.bin").write_bytes(b"payload")
    relative = "/".join((*_COMPONENTS, "file.bin"))
    return (
        lambda: read_d1_artifact_bytes(root, relative, label="probe"),
        "probe artifact could not be opened beneath its root without a symlink",
    )


def _site_publish(tmp_path: Path) -> tuple[Callable[[], object], str]:
    root = _tree(tmp_path)
    relative = "/".join((*_COMPONENTS, "out.bin"))
    return (
        lambda: publish_d1_atomic_bytes(root, relative, b"payload", label="handoff"),
        "handoff atomic publication failed: {cause}",
    )


_SITES = {"root-open": _site_root, "read": _site_read, "publish": _site_publish}


@pytest.mark.parametrize("site", _SITES)
def test_descent_without_a_fault_balances_every_descriptor(site: str, tmp_path: Path) -> None:
    call, _ = _SITES[site](tmp_path)
    with _accounted(None) as ledger:
        call()
    assert ledger.opened >= len(_COMPONENTS)
    _balanced(ledger)


@pytest.mark.parametrize("name", _FAULT_STEPS)
@pytest.mark.parametrize("site", _SITES)
def test_a_mid_descent_open_failure_closes_each_descriptor_once_and_raises_the_sites_error(
    site: str, name: str, tmp_path: Path
) -> None:
    call, template = _SITES[site](tmp_path)
    with _accounted(name) as ledger, pytest.raises(D1ArtifactSnapshotError) as caught:
        call()
    assert str(caught.value) == template.format(cause=_injected(name))
    _injected_cause(caught.value)
    assert ledger.opened >= 2
    _balanced(ledger)


def _nul_site_root(tmp_path: Path) -> Callable[[], object]:
    root = _tree(tmp_path) / _COMPONENTS[0] / _COMPONENTS[1] / "dir\x00three"
    return lambda: normalize_d1_artifact_root(root)


def _nul_site_read(tmp_path: Path) -> Callable[[], object]:
    root = _tree(tmp_path)
    relative = "/".join((*_COMPONENTS[:2], "dir\x00three", "file.bin"))
    return lambda: read_d1_artifact_bytes(root, relative, label="probe")


def _nul_site_publish(tmp_path: Path) -> Callable[[], object]:
    root = _tree(tmp_path)
    relative = "/".join((*_COMPONENTS[:2], "dir\x00three", "out.bin"))
    return lambda: publish_d1_atomic_bytes(root, relative, b"payload", label="handoff")


_NUL_SITES = {"root-open": _nul_site_root, "read": _nul_site_read, "publish": _nul_site_publish}


@pytest.mark.parametrize("site", _NUL_SITES)
def test_a_non_oserror_raise_mid_descent_still_closes_every_descriptor(
    site: str, tmp_path: Path
) -> None:
    """An embedded NUL passes path normalisation, so os.open raises ValueError mid-walk.

    Two directory opens have succeeded by then; the helper must close the held descriptor for
    a raise that is not an OSError. The surfaced type is the site's own concern (publication
    lets the ValueError through unwrapped), so only the underlying ValueError is pinned here.
    """
    call = _NUL_SITES[site](tmp_path)
    with _accounted(None) as ledger, pytest.raises((D1ArtifactSnapshotError, ValueError)) as caught:
        call()
    assert isinstance(caught.value.__cause__ or caught.value, ValueError)
    assert ledger.opened >= 2
    _balanced(ledger)


def test_publication_passes_a_root_failure_through_unwrapped(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with _accounted(root.name) as ledger, pytest.raises(D1ArtifactSnapshotError) as caught:
        publish_d1_atomic_bytes(root, "dir-one/out.bin", b"payload", label="handoff")
    assert str(caught.value) == "D1 artifact root could not be opened without a symlink"
    _injected_cause(caught.value)
    _balanced(ledger)


def test_read_reports_a_root_failure_with_the_root_error_as_its_cause(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with _accounted(root.name) as ledger, pytest.raises(D1ArtifactSnapshotError) as caught:
        read_d1_artifact_bytes(root, "dir-one/file.bin", label="probe")
    assert (
        str(caught.value) == "probe artifact could not be opened beneath its root without a symlink"
    )
    assert type(caught.value.__cause__) is D1ArtifactSnapshotError
    _balanced(ledger)


def test_artifact_directory_joins_the_relative_path_then_admits_the_directory(
    tmp_path: Path,
) -> None:
    root = _tree(tmp_path)
    assert normalize_d1_artifact_directory(
        root, "dir-one/dir-two", label="span command directory"
    ) == (root / "dir-one" / "dir-two")


@pytest.mark.parametrize("label", ["H command directory", "source output directory"])
def test_artifact_directory_refusals_carry_the_callers_label(label: str, tmp_path: Path) -> None:
    root = _tree(tmp_path)
    for value in ("", "../escape", "/absolute", 7):
        with pytest.raises(D1ArtifactSnapshotError, match=f"^{re.escape(label)} path must"):
            normalize_d1_artifact_directory(root, value, label=label)


def test_artifact_directory_validates_the_relative_path_before_touching_the_root(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent"
    with pytest.raises(D1ArtifactSnapshotError, match=r"^capture command directory path must"):
        normalize_d1_artifact_directory(missing, "../escape", label="capture command directory")
    with pytest.raises(
        D1ArtifactSnapshotError,
        match=r"^D1 artifact root could not be opened without a symlink$",
    ):
        normalize_d1_artifact_directory(missing, "child", label="capture command directory")
    with pytest.raises(
        D1ArtifactSnapshotError,
        match=r"^D1 artifact root could not be opened without a symlink$",
    ):
        normalize_d1_artifact_directory(
            _tree(tmp_path), "dir-one/dir-two/dir-three/missing", label="capture command directory"
        )
