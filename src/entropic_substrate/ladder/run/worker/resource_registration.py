"""Worker CUDA-resource registration (split from d_ladder_worker_runtime,
RULING D-E; P12-WORKER-REMAINDER, conductor 2026-09-25).

``_FACTORY_DIGEST``/``_PROVIDER`` and their ``canonical_digest(...)``
computation MIGRATE here (this module's own top level) now that the
remainder lands at ``historical.d_ladder`` -- keeping the function-local
back-import would otherwise become a ``ladder -> historical`` inversion.
The landed remainder imports both constants BACK from this module
(``historical -> ladder``, downward, allowed). Baseline value at p10,
unaffected by this migration: ``_FACTORY_DIGEST ==
fc57ff862df4b279b8e87fbc56c59a8efe49796bf662c66c03c4a1faf92c56ef``.
"""

from __future__ import annotations

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    ResourceActivationMode,
    ResourceActivationRegistration,
)
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest

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


def worker_resource_registration() -> ResourceActivationRegistration:
    """This tracer measures CUDA duration/memory; FLOPs/power remain explicit gaps."""
    return ResourceActivationRegistration(
        _FACTORY_DIGEST, ResourceActivationMode.PRODUCTION, _PROVIDER, "cuda:0", None, None
    )
