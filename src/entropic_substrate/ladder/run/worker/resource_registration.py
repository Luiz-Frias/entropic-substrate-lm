"""Worker CUDA-resource registration, split from d_ladder_worker_runtime (RULING D-E).

Cycle note (ruling P10-WORKER-SPLIT, 2026-09-24): ``_FACTORY_DIGEST`` and
``_PROVIDER`` are imported inside the function body because the ``__legacy``
remainder imports THIS module at module scope to re-export
``worker_resource_registration`` by identity; a module-level back-import would
be a real cycle. p12 migrates the two constants here and flips the direction
(ruling P12-WORKER-REMAINDER), which also answers CodeRabbit's two nitpicks on
PR #302.
"""

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
