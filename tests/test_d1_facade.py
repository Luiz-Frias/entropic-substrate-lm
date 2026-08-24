"""Public-import compatibility and cross-feature D1 contract tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.experiments.d1 import (
    CausalEncoderSpec,
    CodeIdentity,
    CollapseParameters,
    ContainerIdentity,
    D1Arm,
    D1GeometrySpec,
    DataIdentity,
    EMAKind,
    EMAParameters,
    EncoderInitializationIdentity,
    EvidenceClass,
    EvidenceEnvelope,
    ExecutionIdentity,
    FutureLawSpec,
    GeometryConsumer,
    MatchedEncoderPairSpec,
    PredictiveMetric,
    PreregistrationIdentity,
    RunClass,
    TargetUpdateStrategy,
    build_d1_experiment_family,
)
from entropic_substrate.observation import CodecIdentity

pytestmark = pytest.mark.unit


def _codec() -> CodecIdentity:
    return CodecIdentity(
        name="identity-byte",
        construction_digest="a" * 64,
        cardinality=256,
    )


def _family():
    encoder = CausalEncoderSpec(
        latent_width=64,
        n_layers=2,
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
    return build_d1_experiment_family(
        codec=_codec(),
        encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
    )


def _evidence() -> EvidenceEnvelope:
    return EvidenceEnvelope(
        schema_version="d1-evidence-v1",
        required_classes=frozenset(
            {
                EvidenceClass.MEASURED,
                EvidenceClass.DERIVED_FROM_MEASURED,
            }
        ),
    )


def _execution(seed: int) -> ExecutionIdentity:
    return ExecutionIdentity(
        run_class=RunClass.CONFIRMATORY,
        seed=seed,
        pairing_id="d1-stabilization-pair-a",
        code=CodeIdentity(commit="b" * 40, repository_digest="c" * 64),
        container=ContainerIdentity(image_digest="sha256:" + "d" * 64),
    )


def _data() -> DataIdentity:
    return DataIdentity(
        dataset_name="enwik8",
        dataset_digest="1" * 64,
        split_digest="2" * 64,
        sampler_construction_digest="3" * 64,
    )


def test_d1_rejects_strategy_constants_that_do_not_match_the_arm() -> None:
    arms = _family().arms
    draft = arms[D1Arm.NONE]

    with pytest.raises(ValueError, match="EMA parameters"):
        replace(draft, ema=arms[D1Arm.SG].ema)
    with pytest.raises(ValueError, match="collapse parameters"):
        replace(draft, collapse=arms[D1Arm.AC].collapse)
    with pytest.raises(ValueError, match="requires Euclidean metric and geometry off"):
        replace(
            draft,
            geometry=D1GeometrySpec(
                metric=PredictiveMetric.DECODER_FISHER_PULLBACK,
                consumer=GeometryConsumer.OFF,
            ),
        )
    with pytest.raises(ValueError, match="requires target_update"):
        replace(draft, target_update=TargetUpdateStrategy.STOP_GRADIENT_EMA)


def test_ema_and_moment_conventions_are_bound_into_the_family_digest() -> None:
    fixed = EMAParameters()
    scheduled = EMAParameters(
        kind=EMAKind.COSINE_SCHEDULE,
        decay=0.99,
        final_decay=0.999,
        schedule_steps=100,
    )
    collapse = CollapseParameters()

    assert fixed.kind is EMAKind.FIXED
    assert scheduled.kind is EMAKind.COSINE_SCHEDULE
    assert collapse.std_floor > 0.0
    assert collapse.variance_epsilon > 0.0
    assert collapse.covariance_normalization.value == "off_diagonal_sum_per_dimension"
    baseline = _family()
    scheduled_family = build_d1_experiment_family(
        codec=_codec(),
        encoder_pair=baseline.drafts[0].encoder_pair,
        execution_protocol=replace(
            baseline.drafts[0].execution_protocol,
            optimization=replace(
                baseline.drafts[0].execution_protocol.optimization,
                max_steps=100,
                warmup_steps=50,
            ),
        ),
        ema=scheduled,
    )
    changed_moment_convention = build_d1_experiment_family(
        codec=_codec(),
        encoder_pair=baseline.drafts[0].encoder_pair,
        collapse=replace(collapse, variance_epsilon=1e-3),
    )
    changed_law = build_d1_experiment_family(
        codec=_codec(),
        encoder_pair=baseline.drafts[0].encoder_pair,
        future_law=FutureLawSpec(predictor_hidden_width=128),
    )
    changed_encoder = build_d1_experiment_family(
        codec=_codec(),
        encoder_pair=MatchedEncoderPairSpec(
            online=replace(baseline.drafts[0].encoder_pair.online, n_layers=3),
            target=replace(baseline.drafts[0].encoder_pair.target, n_layers=3),
        ),
    )
    assert baseline.family_digest != scheduled_family.family_digest
    assert baseline.family_digest != changed_moment_convention.family_digest
    assert baseline.family_digest != changed_law.family_digest
    assert baseline.family_digest != changed_encoder.family_digest


def test_one_preregistration_family_digest_validates_every_d1_arm() -> None:
    value = _family()
    preregistration = PreregistrationIdentity(
        path="docs/d1-preregistration.md",
        commit="e" * 40,
        family_digest=value.family_digest,
    )
    construction_digests = {draft.construction_digest for draft in value.drafts}

    assert len(value.drafts) == 4
    assert set(value.arms) == set(D1Arm)
    assert len(construction_digests) == 4
    baseline_manifest = value.draft_for(D1Arm.NONE).construction_manifest
    allowed_strategy_axes = {
        "arm",
        "collapse",
        "collapse_strategy",
        "ema",
        "target_update",
    }
    for arm in D1Arm:
        manifest = value.draft_for(arm).construction_manifest
        changed = {key for key, item in manifest.items() if item != baseline_manifest[key]}
        assert changed <= allowed_strategy_axes
    for seed, arm in enumerate(D1Arm, start=1):
        run = value.validate_confirmatory_arm(
            arm=arm,
            execution=_execution(seed),
            data=_data(),
            evidence=_evidence(),
            preregistration=preregistration,
        )
        assert run.preregistration == preregistration
        assert run.family_digest == value.family_digest
