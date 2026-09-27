"""Public compatibility façade for the D1 experiment contracts.

Feature-owned modules hold scalar construction, stabilization, provenance, and
family lifecycle concerns. Existing callers keep importing this module.
"""

from __future__ import annotations

from entropic_substrate.kernel.identity.d1_primitives import (
    CausalEncoderSpec,
    CollapseStrategy,
    CovarianceNormalization,
    D1Arm,
    D1BoundaryStateSpec,
    EMAKind,
    EncoderInitializationIdentity,
    EncoderShape,
    EvidenceClass,
    FutureActivation,
    FutureLawFamily,
    GeometryConsumer,
    LatentDynamicsPosture,
    MatchedEncoderPairSpec,
    PredictiveMetric,
    RunClass,
    StateConstructionSpec,
    StateLiftPosture,
    TargetUpdateStrategy,
)
from entropic_substrate.ladder.register.d1_family import (
    D1DraftSpec,
    D1ExperimentFamily,
    build_d1_experiment_family,
    build_four_arm_drafts,
    strategies_for_arm,
)
from entropic_substrate.ladder.register.d1_provenance import (
    CodeIdentity,
    ContainerIdentity,
    DataIdentity,
    DataOrderAttestation,
    EvidenceEnvelope,
    ExecutionIdentity,
    PreregistrationIdentity,
)
from entropic_substrate.ladder.register.d1_runs import D1RunSpec
from entropic_substrate.ladder.register.d1_stabilization import (
    CollapseParameters,
    D1GeometrySpec,
    D1TargetSpec,
    EMAParameters,
    FutureLawSpec,
)

__all__ = [
    "CausalEncoderSpec",
    "CodeIdentity",
    "CollapseParameters",
    "CollapseStrategy",
    "ContainerIdentity",
    "CovarianceNormalization",
    "D1Arm",
    "D1BoundaryStateSpec",
    "D1DraftSpec",
    "D1ExperimentFamily",
    "D1GeometrySpec",
    "D1RunSpec",
    "D1TargetSpec",
    "DataIdentity",
    "DataOrderAttestation",
    "EMAKind",
    "EMAParameters",
    "EncoderInitializationIdentity",
    "EncoderShape",
    "EvidenceClass",
    "EvidenceEnvelope",
    "ExecutionIdentity",
    "FutureActivation",
    "FutureLawFamily",
    "FutureLawSpec",
    "GeometryConsumer",
    "LatentDynamicsPosture",
    "MatchedEncoderPairSpec",
    "PredictiveMetric",
    "PreregistrationIdentity",
    "RunClass",
    "StateConstructionSpec",
    "StateLiftPosture",
    "TargetUpdateStrategy",
    "build_d1_experiment_family",
    "build_four_arm_drafts",
    "strategies_for_arm",
]
