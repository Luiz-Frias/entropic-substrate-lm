"""Feature-owned tests for D1 four-arm family construction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.core.observation.contracts import CodecIdentity
from entropic_substrate.kernel.identity.d1_primitives import (
    CausalEncoderSpec,
    CollapseStrategy,
    D1Arm,
    D1BoundaryStateSpec,
    EMAKind,
    EncoderInitializationIdentity,
    MatchedEncoderPairSpec,
    StateConstructionSpec,
    TargetUpdateStrategy,
)
from entropic_substrate.ladder.register.d1_execution import default_d1_execution_protocol
from entropic_substrate.ladder.register.d1_family import (
    D1ExperimentFamily,
    build_d1_experiment_family,
)
from entropic_substrate.ladder.register.d1_stabilization import (
    CollapseParameters,
    D1GeometrySpec,
    D1TargetSpec,
    EMAParameters,
    FutureLawSpec,
)

pytestmark = pytest.mark.unit


def _codec() -> CodecIdentity:
    return CodecIdentity(
        name="identity-byte",
        construction_digest="a" * 64,
        cardinality=256,
    )


def _encoder(*, n_layers: int = 2) -> CausalEncoderSpec:
    return CausalEncoderSpec(
        latent_width=64,
        n_layers=n_layers,
        n_heads=4,
        mlp_ratio=4.0,
        block_size=256,
        dropout=0.0,
        bias=False,
        rms_epsilon=1e-5,
        initialization=EncoderInitializationIdentity(
            name="d1-causal-random-v1",
            construction_digest="5" * 64,
        ),
    )


def _family() -> D1ExperimentFamily:
    value = _encoder()
    return build_d1_experiment_family(
        codec=_codec(),
        encoder_pair=MatchedEncoderPairSpec(online=value, target=value),
    )


def test_four_arm_drafts_are_capacity_matched_and_factorial() -> None:
    arms = _family().arms

    assert set(arms) == set(D1Arm)
    assert arms[D1Arm.NONE].target_update is TargetUpdateStrategy.ONLINE
    assert arms[D1Arm.NONE].collapse_strategy is CollapseStrategy.NONE
    assert arms[D1Arm.SG].target_update is TargetUpdateStrategy.STOP_GRADIENT_EMA
    assert arms[D1Arm.SG].collapse_strategy is CollapseStrategy.NONE
    assert arms[D1Arm.AC].target_update is TargetUpdateStrategy.ONLINE
    assert arms[D1Arm.AC].collapse_strategy is CollapseStrategy.MOMENT
    assert arms[D1Arm.BOTH].target_update is TargetUpdateStrategy.STOP_GRADIENT_EMA
    assert arms[D1Arm.BOTH].collapse_strategy is CollapseStrategy.MOMENT

    shared = tuple(arms.values())
    assert {draft.codec for draft in shared} == {_codec()}
    assert {draft.encoder_pair for draft in shared} == {_family().drafts[0].encoder_pair}
    assert {draft.target for draft in shared} == {D1TargetSpec()}
    assert {draft.future_law for draft in shared} == {FutureLawSpec()}
    assert {draft.geometry for draft in shared} == {D1GeometrySpec()}
    assert {draft.state_construction for draft in shared} == {StateConstructionSpec()}
    assert {draft.boundary_state for draft in shared} == {D1BoundaryStateSpec()}
    assert {draft.execution_protocol for draft in shared} == {
        default_d1_execution_protocol(sequence_length=256)
    }

    assert arms[D1Arm.NONE].ema is None
    assert arms[D1Arm.AC].ema is None
    assert arms[D1Arm.SG].ema is not None
    assert arms[D1Arm.BOTH].ema is not None
    assert arms[D1Arm.NONE].collapse is None
    assert arms[D1Arm.SG].collapse is None
    assert isinstance(arms[D1Arm.AC].collapse, CollapseParameters)
    assert isinstance(arms[D1Arm.BOTH].collapse, CollapseParameters)


def test_family_refuses_missing_duplicate_or_asymmetric_cells() -> None:
    value = _family()
    drafts = value.arms

    with pytest.raises(ValueError, match="immutable tuple"):
        D1ExperimentFamily(drafts=list(value.drafts))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly four"):
        D1ExperimentFamily(drafts=tuple(drafts[arm] for arm in (D1Arm.NONE, D1Arm.SG, D1Arm.AC)))
    with pytest.raises(ValueError, match="duplicate"):
        D1ExperimentFamily(
            drafts=(
                drafts[D1Arm.NONE],
                drafts[D1Arm.NONE],
                drafts[D1Arm.AC],
                drafts[D1Arm.BOTH],
            )
        )
    asymmetric = replace(
        drafts[D1Arm.AC],
        encoder_pair=MatchedEncoderPairSpec(
            online=_encoder(n_layers=3),
            target=_encoder(n_layers=3),
        ),
    )
    with pytest.raises(ValueError, match="capacity-matched"):
        D1ExperimentFamily(
            drafts=(
                drafts[D1Arm.NONE],
                drafts[D1Arm.SG],
                asymmetric,
                drafts[D1Arm.BOTH],
            )
        )


def test_execution_protocol_is_bound_into_cell_and_family_identity() -> None:
    baseline = _family()
    changed_protocol = replace(
        baseline.drafts[0].execution_protocol,
        optimization=replace(
            baseline.drafts[0].execution_protocol.optimization,
            learning_rate=3e-4,
        ),
    )
    encoder = _encoder()
    changed = build_d1_experiment_family(
        codec=_codec(),
        encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
        execution_protocol=changed_protocol,
    )

    assert changed.family_digest != baseline.family_digest
    assert changed.drafts[0].construction_digest != baseline.drafts[0].construction_digest
    assert changed.family_manifest["shared_construction"]["execution_protocol"] == (
        changed_protocol.manifest
    )


def test_family_refuses_execution_sequences_wider_than_the_encoder_block() -> None:
    encoder = _encoder()
    with pytest.raises(ValueError, match=r"sequence length.*block size"):
        build_d1_experiment_family(
            codec=_codec(),
            encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
            execution_protocol=default_d1_execution_protocol(sequence_length=257),
        )


def test_cosine_ema_schedule_must_cover_the_registered_optimizer_budget() -> None:
    encoder = _encoder()
    with pytest.raises(ValueError, match=r"EMA schedule_steps.*max_steps"):
        build_d1_experiment_family(
            codec=_codec(),
            encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
            ema=EMAParameters(
                kind=EMAKind.COSINE_SCHEDULE,
                decay=0.9,
                final_decay=0.99,
                schedule_steps=3,
            ),
        )
