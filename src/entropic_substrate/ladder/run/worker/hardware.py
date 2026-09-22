"""Measured pod hardware and source-tree provenance.

Split from `entropic_substrate.__legacy.d_ladder_worker_runtime` (design-gate
RULING D-E, conductor 2026-09-22, wf_6d1caf7e-2e6): this is exactly the two
names p4's own moved importers consume, `measure_hardware` and
`verify_source_tree`, plus the `SOURCE_PREFIXES` constant and the two
Protocol helpers each one needs. `build_worker_factory`,
`worker_resource_registration` and `verify_worker_source` stay in the
`__legacy` remainder, which re-exports the three names below so its own
module-level consumers resolve unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

from entropic_substrate.__legacy.d1_artifact_snapshot import read_d1_artifact_bytes

SOURCE_PREFIXES = (
    "src/entropic_substrate",
    "configs",
    "scripts",
    "pyproject.toml",
    "uv.lock",
    ".dvc/config",
)


class _Hardware(Protocol):
    def is_available(self) -> bool: ...
    def device_count(self) -> int: ...
    def get_device_name(self, device: int) -> str: ...


class _GitTree(Protocol):
    def tree_paths(self, revision: str, prefixes: tuple[str, ...]) -> tuple[str, ...]: ...
    def read_file(self, revision: str, path: str) -> bytes | None: ...


def measure_hardware(cuda: object | None = None) -> tuple[str, int]:
    if cuda is None:
        import torch

        cuda = torch.cuda
    runtime = cast(_Hardware, cuda)
    if not runtime.is_available():
        raise ValueError("pod worker requires available CUDA")
    count = runtime.device_count()
    if type(count) is not int or count < 1:
        raise ValueError("CUDA device count must be measured positive")
    names = {runtime.get_device_name(index) for index in range(count)}
    if len(names) != 1 or any(type(name) is not str or not name for name in names):
        raise ValueError("pod worker requires measured homogeneous accelerator models")
    return names.pop(), count


def verify_source_tree(root: Path, commit: str, git: _GitTree) -> None:
    """Compare the image's entire runtime source inventory against committed bytes."""
    expected = set(git.tree_paths(commit, SOURCE_PREFIXES))
    actual: set[str] = set()
    for prefix in SOURCE_PREFIXES:
        selected = root / prefix
        paths = selected.rglob("*") if selected.is_dir() else (selected,)
        for path in paths:
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError("image runtime source inventory contains a symlink")
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError("image runtime source inventory differs from source commit")
    for path in sorted(actual):
        _, payload, _ = read_d1_artifact_bytes(root, path, label="image source")
        if payload != git.read_file(commit, path):
            raise ValueError(f"image source bytes differ from source commit: {path}")
