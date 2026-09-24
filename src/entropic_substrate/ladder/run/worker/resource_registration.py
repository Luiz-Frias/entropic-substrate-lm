"""Worker CUDA-resource registration, split from d_ladder_worker_runtime (RULING D-E)."""

from __future__ import annotations

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    ResourceActivationMode,
    ResourceActivationRegistration,
)


def worker_resource_registration() -> ResourceActivationRegistration:
    """This tracer measures CUDA duration/memory; FLOPs/power remain explicit gaps."""
    from entropic_substrate.__legacy.d_ladder_worker_runtime import (
        _FACTORY_DIGEST,  # pyright: ignore[reportPrivateUsage]
        _PROVIDER,  # pyright: ignore[reportPrivateUsage]
    )

    return ResourceActivationRegistration(
        _FACTORY_DIGEST, ResourceActivationMode.PRODUCTION, _PROVIDER, "cuda:0", None, None
    )
