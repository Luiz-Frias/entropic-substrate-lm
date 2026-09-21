"""Measured pod hardware, source provenance, and existing CUDA resource adapters."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

from entropic_substrate.d1_artifact_snapshot import publish_d1_atomic_bytes, read_d1_artifact_bytes
from entropic_substrate.d1_predecessor_loader import verified_git_bundle
from entropic_substrate.d1_resource_activation import D1ProductionResourceActivationFactory
from entropic_substrate.d1_trainer_mailbox_contracts import D1TrainerMailboxObjectRead
from entropic_substrate.d1_trainer_object_transport import D1TrainerMailboxObjectPort
from entropic_substrate.d_ladder_remote_request import RemoteLadderRequest
from entropic_substrate.evidence.resource_meter_runtime import build_torch_cuda_adapters
from entropic_substrate.evidence.resource_provenance import (
    ResourceActivationMode,
    ResourceActivationRegistration,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest
from entropic_substrate.paths import source_root

SOURCE_PREFIXES = (
    "src/entropic_substrate",
    "configs",
    "scripts",
    "pyproject.toml",
    "uv.lock",
    ".dvc/config",
)
_PROVIDER = "torch-cuda-events-and-peak-memory-v1"
_FACTORY_DIGEST = canonical_digest(
    {
        "schema": "d-ladder-cuda-resource-factory-v1",
        "provider": _PROVIDER,
        "device_scope": "cuda:0",
        "flops": None,
        "power": None,
    }
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


def worker_resource_registration() -> ResourceActivationRegistration:
    """This tracer measures CUDA duration/memory; FLOPs/power remain explicit gaps."""
    return ResourceActivationRegistration(
        _FACTORY_DIGEST, ResourceActivationMode.PRODUCTION, _PROVIDER, "cuda:0", None, None
    )


def build_worker_factory(
    registration: ResourceActivationRegistration,
) -> D1ProductionResourceActivationFactory:
    if registration != worker_resource_registration():
        raise ValueError("worker cannot execute a different resource factory registration")
    timer, memory = build_torch_cuda_adapters(device=0)
    if timer is None or memory is None:
        raise ValueError("worker resource factory requires available CUDA timing and memory")
    return D1ProductionResourceActivationFactory(
        registration_digest=_FACTORY_DIGEST,
        meter_provider=_PROVIDER,
        device_scope="cuda:0",
        phase_timer_factory=lambda _: timer,
        peak_memory_factory=lambda _: memory,
    )


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


def verify_worker_source(
    objects: D1TrainerMailboxObjectPort, request: RemoteLadderRequest, repo_root: Path
) -> None:
    """Use bound Git evidence even for pilots; environment labels alone are insufficient.

    The comparison is layout-invariant (E-3/DN-8): the "actual" side is the
    installed ``entropic_substrate`` package's own resolved location
    (``entropic_substrate.paths.source_root()``), not this module's own
    ``__file__`` depth, so relocating a copy of this module elsewhere does
    not by itself make a legitimately bound tree read as unbound.
    """
    if (source_root() / "entropic_substrate").resolve() != (
        repo_root / "src/entropic_substrate"
    ).resolve():
        raise ValueError("worker is not executing the source tree selected for verification")
    bundle = request.registration_git_bundle
    if bundle is None:
        raise ValueError("pod worker requires a bound Git bundle to verify image source")
    key = f"{request.prefix}/inputs/{bundle.path}"
    observed = objects.read(key)
    if (
        type(observed) is not D1TrainerMailboxObjectRead
        or observed.identity.key != key
        or observed.identity.sha256 != bundle.content_hash.digest
    ):
        raise ValueError("worker source Git bundle object identity mismatch")
    with TemporaryDirectory(prefix="d-ladder-source-") as temporary:
        root = Path(temporary).resolve()
        (root / bundle.path).parent.mkdir(parents=True, exist_ok=True)
        publish_d1_atomic_bytes(root, bundle.path, observed.payload, label="source proof")
        with verified_git_bundle(root, bundle) as git:
            verify_source_tree(repo_root, request.code_commit, git)
