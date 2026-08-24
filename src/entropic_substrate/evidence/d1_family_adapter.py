"""Adapter from the registered D1 experiment family to verdict validation."""

from __future__ import annotations

from entropic_substrate.evidence.d1_types import D1FamilyIdentity
from entropic_substrate.experiments.d1 import D1Arm, D1ExperimentFamily


def family_identity_from_experiment(family: D1ExperimentFamily) -> D1FamilyIdentity:
    """Project the experiment layer's authoritative full construction identity.

    The evidence domain receives the resulting immutable value object; it
    never recomputes an arm recipe from abbreviated treatment labels.
    """

    return D1FamilyIdentity(
        family_digest=family.family_digest,
        arm_construction_digests={
            arm.value: family.draft_for(arm).construction_digest for arm in D1Arm
        },
    )
