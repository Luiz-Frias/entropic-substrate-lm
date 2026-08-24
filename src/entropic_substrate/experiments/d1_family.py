"""Four-cell D1 family construction, lifecycle transitions, and digests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from entropic_substrate.experiments.d1_execution import (
    D1ExecutionProtocol,
    default_d1_execution_protocol,
)
from entropic_substrate.experiments.d1_primitives import (
    CONSTRUCTION_SCHEMA,
    FAMILY_SCHEMA,
    CausalEncoderSpec,
    CollapseStrategy,
    D1Arm,
    D1BoundaryStateSpec,
    GeometryConsumer,
    MatchedEncoderPairSpec,
    PredictiveMetric,
    RunClass,
    StateConstructionSpec,
    TargetUpdateStrategy,
    canonical_digest,
    default_d1_encoder_initialization_identity,
)
from entropic_substrate.experiments.d1_provenance import (
    DataIdentity,
    EvidenceEnvelope,
    ExecutionIdentity,
    PreregistrationIdentity,
)
from entropic_substrate.experiments.d1_stabilization import (
    CollapseParameters,
    D1GeometrySpec,
    D1TargetSpec,
    EMAParameters,
    FutureLawSpec,
)
from entropic_substrate.observation import CodecIdentity

if TYPE_CHECKING:
    from entropic_substrate.experiments.d1_runs import D1RunSpec


@dataclass(frozen=True, slots=True)
class _StrategyPair:
    target_update: TargetUpdateStrategy
    collapse: CollapseStrategy


_ARM_STRATEGIES: Final = MappingProxyType(
    {
        D1Arm.NONE: _StrategyPair(TargetUpdateStrategy.ONLINE, CollapseStrategy.NONE),
        D1Arm.SG: _StrategyPair(TargetUpdateStrategy.STOP_GRADIENT_EMA, CollapseStrategy.NONE),
        D1Arm.AC: _StrategyPair(TargetUpdateStrategy.ONLINE, CollapseStrategy.MOMENT),
        D1Arm.BOTH: _StrategyPair(TargetUpdateStrategy.STOP_GRADIENT_EMA, CollapseStrategy.MOMENT),
    }
)


def strategies_for_arm(arm: D1Arm) -> tuple[TargetUpdateStrategy, CollapseStrategy]:
    """Return the one registered target/collapse strategy pair for ``arm``."""
    pair = _ARM_STRATEGIES[arm]
    return pair.target_update, pair.collapse


@dataclass(frozen=True, slots=True)
class D1DraftSpec:
    """Scientifically complete construction of one D1 family cell."""

    arm: D1Arm
    codec: CodecIdentity
    encoder_pair: MatchedEncoderPairSpec
    boundary_state: D1BoundaryStateSpec
    state_construction: StateConstructionSpec
    target: D1TargetSpec
    future_law: FutureLawSpec
    geometry: D1GeometrySpec
    execution_protocol: D1ExecutionProtocol
    target_update: TargetUpdateStrategy
    collapse_strategy: CollapseStrategy
    ema: EMAParameters | None
    collapse: CollapseParameters | None

    def __post_init__(self) -> None:
        expected_target, expected_collapse = strategies_for_arm(self.arm)
        if self.target_update is not expected_target:
            msg = f"D1 arm {self.arm.value} requires target_update={expected_target.value}"
            raise ValueError(msg)
        if self.collapse_strategy is not expected_collapse:
            msg = f"D1 arm {self.arm.value} requires collapse_strategy={expected_collapse.value}"
            raise ValueError(msg)
        if self.target_update is TargetUpdateStrategy.ONLINE and self.ema is not None:
            msg = "online target update forbids EMA parameters"
            raise ValueError(msg)
        if self.target_update is TargetUpdateStrategy.STOP_GRADIENT_EMA and self.ema is None:
            msg = "stop-gradient target update requires EMA parameters"
            raise ValueError(msg)
        if self.collapse_strategy is CollapseStrategy.NONE and self.collapse is not None:
            msg = "collapse-off strategy forbids collapse parameters"
            raise ValueError(msg)
        if self.collapse_strategy is CollapseStrategy.MOMENT and self.collapse is None:
            msg = "moment collapse strategy requires collapse parameters"
            raise ValueError(msg)
        if (
            self.geometry.metric is not PredictiveMetric.EUCLIDEAN
            or self.geometry.consumer is not GeometryConsumer.OFF
        ):
            msg = "D1 requires Euclidean metric and geometry off"
            raise ValueError(msg)
        if type(self.execution_protocol) is not D1ExecutionProtocol:
            msg = "D1 draft requires a validated execution protocol"
            raise TypeError(msg)
        if self.execution_protocol.batching.horizon != self.target.horizon:
            msg = "D1 execution batching horizon must match the target horizon"
            raise ValueError(msg)
        if self.execution_protocol.batching.sequence_length > self.encoder_pair.online.block_size:
            msg = "D1 execution sequence length cannot exceed the encoder block size"
            raise ValueError(msg)
        if (
            self.execution_protocol.baseline.num_batches
            != self.future_law.mean_baseline_calibration_batches
            or self.execution_protocol.baseline.sampling.value
            != self.future_law.mean_baseline_sampling
        ):
            msg = "D1 execution baseline protocol must match the future-law calibration"
            raise ValueError(msg)

    @property
    def construction_manifest(self) -> dict[str, object]:
        """Canonical per-arm identity, including its poolable execution protocol."""
        return {
            "arm": self.arm.value,
            "boundary_state": self.boundary_state.manifest,
            "codec": {
                "cardinality": self.codec.cardinality,
                "construction_digest": self.codec.construction_digest,
                "name": self.codec.name,
            },
            "collapse": None if self.collapse is None else self.collapse.manifest,
            "collapse_strategy": self.collapse_strategy.value,
            "ema": None if self.ema is None else self.ema.manifest,
            "encoder_pair": self.encoder_pair.manifest,
            "execution_protocol": self.execution_protocol.manifest,
            "future_law": self.future_law.manifest,
            "geometry": self.geometry.manifest,
            "schema_version": CONSTRUCTION_SCHEMA,
            "state_construction": self.state_construction.manifest,
            "target": self.target.manifest,
            "target_update": self.target_update.value,
        }

    @property
    def construction_digest(self) -> str:
        """SHA-256 binding this individual cell's complete construction."""
        return canonical_digest(self.construction_manifest)

    def validate_smoke(
        self,
        *,
        execution: ExecutionIdentity,
        evidence: EvidenceEnvelope,
        data: DataIdentity | None = None,
    ) -> D1RunSpec:
        """Construct a validated smoke run; it cannot be confirmatory evidence."""
        from entropic_substrate.experiments.d1_runs import D1RunSpec

        return D1RunSpec(
            draft=self,
            execution=execution,
            data=data,
            evidence=evidence,
            preregistration=None,
            family=None,
            expected_run_class=RunClass.SMOKE,
        )

    def validate_pilot(
        self,
        *,
        execution: ExecutionIdentity,
        evidence: EvidenceEnvelope,
        data: DataIdentity | None = None,
    ) -> D1RunSpec:
        """Construct a validated, explicitly excluded pilot run."""
        from entropic_substrate.experiments.d1_runs import D1RunSpec

        return D1RunSpec(
            draft=self,
            execution=execution,
            data=data,
            evidence=evidence,
            preregistration=None,
            family=None,
            expected_run_class=RunClass.PILOT,
        )

    def validate_confirmatory(
        self,
        *,
        family: D1ExperimentFamily,
        execution: ExecutionIdentity,
        data: DataIdentity,
        evidence: EvidenceEnvelope,
        preregistration: PreregistrationIdentity,
    ) -> D1RunSpec:
        """Validate this cell against its one preregistered four-arm family."""
        return family.validate_confirmatory_arm(
            arm=self.arm,
            execution=execution,
            data=data,
            evidence=evidence,
            preregistration=preregistration,
        )


@dataclass(frozen=True, slots=True)
class D1ExperimentFamily:
    """The one immutable, capacity-matched object a D1 preregistration binds."""

    drafts: tuple[D1DraftSpec, ...]

    def __post_init__(self) -> None:
        if type(self.drafts) is not tuple:
            msg = "D1 experiment family drafts must be an immutable tuple"
            raise ValueError(msg)
        if len(self.drafts) != len(D1Arm):
            msg = "D1 experiment family requires exactly four arms"
            raise ValueError(msg)
        seen = [draft.arm for draft in self.drafts]
        if len(set(seen)) != len(seen):
            msg = "D1 experiment family refuses duplicate arms"
            raise ValueError(msg)
        if set(seen) != set(D1Arm):
            msg = "D1 experiment family requires every registered arm"
            raise ValueError(msg)

        baseline = self.drafts[0]
        shared_fields = (
            "codec",
            "encoder_pair",
            "boundary_state",
            "state_construction",
            "target",
            "future_law",
            "geometry",
            "execution_protocol",
        )
        for draft in self.drafts[1:]:
            if any(getattr(draft, field) != getattr(baseline, field) for field in shared_fields):
                msg = "D1 family arms must be capacity-matched outside stabilization strategies"
                raise ValueError(msg)

        allowed_to_vary = {
            "arm",
            "collapse",
            "collapse_strategy",
            "ema",
            "target_update",
        }
        baseline_manifest = baseline.construction_manifest
        for draft in self.drafts[1:]:
            changed = {
                key
                for key, value in draft.construction_manifest.items()
                if value != baseline_manifest[key]
            }
            if not changed <= allowed_to_vary:
                msg = "D1 family cells may differ only on registered stabilization strategy axes"
                raise ValueError(msg)

    @property
    def arms(self) -> Mapping[D1Arm, D1DraftSpec]:
        """Read-only lookup for the four validated family cells."""
        return MappingProxyType({draft.arm: draft for draft in self.drafts})

    def draft_for(self, arm: D1Arm) -> D1DraftSpec:
        """Return the exact registered cell for ``arm``."""
        for draft in self.drafts:
            if draft.arm is arm:
                return draft
        msg = f"D1 family does not contain arm {arm!r}"
        raise ValueError(msg)

    @property
    def family_manifest(self) -> dict[str, object]:
        """Canonical whole-design identity, ordered by the registered arm enum."""
        baseline = self.drafts[0]
        return {
            "arms": [
                {
                    "arm": arm.value,
                    "construction_digest": self.draft_for(arm).construction_digest,
                }
                for arm in D1Arm
            ],
            "schema_version": FAMILY_SCHEMA,
            "shared_construction": {
                "boundary_state": baseline.boundary_state.manifest,
                "codec": {
                    "cardinality": baseline.codec.cardinality,
                    "construction_digest": baseline.codec.construction_digest,
                    "name": baseline.codec.name,
                },
                "encoder_pair": baseline.encoder_pair.manifest,
                "execution_protocol": baseline.execution_protocol.manifest,
                "future_law": baseline.future_law.manifest,
                "geometry": baseline.geometry.manifest,
                "state_construction": baseline.state_construction.manifest,
                "target": baseline.target.manifest,
            },
        }

    @property
    def family_digest(self) -> str:
        """SHA-256 identity of the entire four-cell factorial design."""
        return canonical_digest(self.family_manifest)

    def validate_confirmatory_arm(
        self,
        *,
        arm: D1Arm,
        execution: ExecutionIdentity,
        data: DataIdentity,
        evidence: EvidenceEnvelope,
        preregistration: PreregistrationIdentity,
    ) -> D1RunSpec:
        """Construct one confirmatory run bound to this preregistered family."""
        from entropic_substrate.experiments.d1_runs import D1RunSpec

        return D1RunSpec(
            draft=self.draft_for(arm),
            execution=execution,
            data=data,
            evidence=evidence,
            preregistration=preregistration,
            family=self,
            expected_run_class=RunClass.CONFIRMATORY,
        )


def build_d1_experiment_family(
    *,
    codec: CodecIdentity,
    encoder_pair: MatchedEncoderPairSpec,
    boundary_state: D1BoundaryStateSpec | None = None,
    state_construction: StateConstructionSpec | None = None,
    target: D1TargetSpec | None = None,
    future_law: FutureLawSpec | None = None,
    geometry: D1GeometrySpec | None = None,
    execution_protocol: D1ExecutionProtocol | None = None,
    ema: EMAParameters | None = None,
    collapse: CollapseParameters | None = None,
) -> D1ExperimentFamily:
    """Build the capacity-matched D1 family from one shared construction."""
    shared_boundary = boundary_state if boundary_state is not None else D1BoundaryStateSpec()
    shared_state = state_construction if state_construction is not None else StateConstructionSpec()
    shared_target = target if target is not None else D1TargetSpec()
    shared_law = (
        future_law
        if future_law is not None
        else FutureLawSpec(predictor_hidden_width=encoder_pair.latent_width)
    )
    shared_geometry = geometry if geometry is not None else D1GeometrySpec()
    shared_execution = (
        execution_protocol
        if execution_protocol is not None
        else default_d1_execution_protocol(sequence_length=encoder_pair.online.block_size)
    )
    shared_ema = ema if ema is not None else EMAParameters()
    shared_collapse = collapse if collapse is not None else CollapseParameters()

    drafts = tuple(
        D1DraftSpec(
            arm=arm,
            codec=codec,
            encoder_pair=encoder_pair,
            boundary_state=shared_boundary,
            state_construction=shared_state,
            target=shared_target,
            future_law=shared_law,
            geometry=shared_geometry,
            execution_protocol=shared_execution,
            target_update=strategies_for_arm(arm)[0],
            collapse_strategy=strategies_for_arm(arm)[1],
            ema=(
                shared_ema
                if strategies_for_arm(arm)[0] is TargetUpdateStrategy.STOP_GRADIENT_EMA
                else None
            ),
            collapse=(
                shared_collapse if strategies_for_arm(arm)[1] is CollapseStrategy.MOMENT else None
            ),
        )
        for arm in D1Arm
    )
    return D1ExperimentFamily(drafts=drafts)


def build_four_arm_drafts(*, codec: CodecIdentity, latent_width: int) -> dict[D1Arm, D1DraftSpec]:
    """Compatibility factory yielding a default complete D1 family as four drafts."""
    initialization = default_d1_encoder_initialization_identity()
    encoder = CausalEncoderSpec(
        latent_width=latent_width,
        n_layers=2,
        n_heads=4,
        mlp_ratio=4.0,
        block_size=256,
        dropout=0.0,
        bias=False,
        rms_epsilon=1e-5,
        initialization=initialization,
    )
    family = build_d1_experiment_family(
        codec=codec,
        encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
    )
    return dict(family.arms)
