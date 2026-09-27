"""Measured pod hardware, source provenance, and existing CUDA resource adapters.

`measure_hardware`, `verify_source_tree` and `SOURCE_PREFIXES` moved to
`entropic_substrate.ladder.run.worker.hardware` (design-gate RULING D-E,
conductor 2026-09-22, wf_6d1caf7e-2e6: this module's own live importers
consume only those two names). Re-imported below so this module's other
consumers -- `__legacy/d_ladder_worker_command.py` and its tests -- keep
resolving `measure_hardware`/`verify_source_tree`/`SOURCE_PREFIXES` off this
same dotted name unchanged. p10 (ruling P10-WORKER-SPLIT, 2026-09-24) moved
`worker_resource_registration` to
`entropic_substrate.ladder.run.worker.resource_registration`, re-exported here
by identity; `build_worker_factory` and `verify_worker_source` stay until p12
because `tests/test_d_ladder_worker_runtime.py` monkeypatches
`build_torch_cuda_adapters` on this module.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from entropic_substrate.historical.d_ladder.d_ladder_remote_request import RemoteLadderRequest
from entropic_substrate.kernel.artifacts import publish_d1_atomic_bytes
from entropic_substrate.kernel.evidence_class.resource_provenance import (
    ResourceActivationRegistration,
)
from entropic_substrate.kernel.runtime.paths import source_root
from entropic_substrate.ladder.capture.resource.resource_meter_runtime import (
    build_torch_cuda_adapters,
)
from entropic_substrate.ladder.run.resource.d1_resource_activation import (
    D1ProductionResourceActivationFactory,
)
from entropic_substrate.ladder.run.transport.d1_trainer_mailbox_contracts import (
    D1TrainerMailboxObjectRead,
)
from entropic_substrate.ladder.run.transport.d1_trainer_object_transport import (
    D1TrainerMailboxObjectPort,
)
from entropic_substrate.ladder.run.worker.d1_predecessor_loader import verified_git_bundle
from entropic_substrate.ladder.run.worker.hardware import (
    SOURCE_PREFIXES as SOURCE_PREFIXES,
)
from entropic_substrate.ladder.run.worker.hardware import (
    measure_hardware as measure_hardware,
)
from entropic_substrate.ladder.run.worker.hardware import verify_source_tree
from entropic_substrate.ladder.run.worker.resource_registration import (
    _FACTORY_DIGEST,  # pyright: ignore[reportPrivateUsage]
    _PROVIDER,  # pyright: ignore[reportPrivateUsage]
)
from entropic_substrate.ladder.run.worker.resource_registration import (
    worker_resource_registration as worker_resource_registration,
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
