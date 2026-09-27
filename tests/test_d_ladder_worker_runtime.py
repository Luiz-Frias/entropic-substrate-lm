"""Actual worker provenance rejects unavailable CUDA and mismatched source trees."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from entropic_substrate.historical.d_ladder.d_ladder_worker_runtime import (
    SOURCE_PREFIXES,
    build_worker_factory,
    measure_hardware,
    verify_source_tree,
    worker_resource_registration,
)

pytestmark = pytest.mark.unit


def test_hardware_is_measured_and_cpu_cannot_masquerade_as_gpu() -> None:
    cuda = SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 2, get_device_name=lambda _: "Measured GPU"
    )
    assert measure_hardware(cuda) == ("Measured GPU", 2)
    cuda.is_available = lambda: False
    with pytest.raises(ValueError, match="CUDA"):
        measure_hardware(cuda)


def test_resource_registration_requires_the_existing_cuda_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import entropic_substrate.historical.d_ladder.d_ladder_worker_runtime as module

    monkeypatch.setattr(module, "build_torch_cuda_adapters", lambda **_: (None, None))
    with pytest.raises(ValueError, match="CUDA"):
        build_worker_factory(worker_resource_registration())


def test_source_tree_proof_rejects_changed_extra_and_missing_files(tmp_path: Path) -> None:
    class Git:
        def tree_paths(self, commit: str, prefixes: tuple[str, ...]) -> tuple[str, ...]:
            assert commit == "a" * 40 and prefixes == SOURCE_PREFIXES
            return ("src/entropic_substrate/a.py",)

        def read_file(self, commit: str, path: str) -> bytes:
            return b"verified source\n"

    path = tmp_path / "src/entropic_substrate/a.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"verified source\n")
    verify_source_tree(tmp_path, "a" * 40, Git())
    path.write_bytes(b"drift")
    with pytest.raises(ValueError, match="bytes"):
        verify_source_tree(tmp_path, "a" * 40, Git())
    path.write_bytes(b"verified source\n")
    extra = path.parent / "extra.py"
    extra.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="inventory"):
        verify_source_tree(tmp_path, "a" * 40, Git())
    extra.unlink()
    path.unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_source_tree(tmp_path, "a" * 40, Git())
