"""Factory translation from registered D1 drafts to runtime strategies."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.experiments.d1_family import (
    build_d1_experiment_family,
    build_four_arm_drafts,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, EMAKind
from entropic_substrate.experiments.d1_stabilization import EMAParameters
from entropic_substrate.observation import CodecIdentity
from entropic_substrate.predictive_state.ema_schedules import CosineEmaDecaySchedule
from entropic_substrate.predictive_state.runtime_construction import (
    collapse_regularizer_from_draft,
    encoder_config_from_draft,
    target_policy_from_draft,
)
from entropic_substrate.predictive_state.stabilization import (
    MomentCollapseRegularizer,
    NoCollapseRegularizer,
)
from entropic_substrate.predictive_state.target_updates import (
    OnlineTargetUpdate,
    StopGradientEmaUpdate,
)

pytestmark = pytest.mark.unit


def _drafts():
    return build_four_arm_drafts(
        codec=CodecIdentity(name="construction-test", construction_digest="c" * 64, cardinality=19),
        latent_width=8,
    )


def test_encoder_config_is_an_exact_translation_of_the_registered_pair() -> None:
    draft = _drafts()[D1Arm.NONE]

    cfg = encoder_config_from_draft(draft)

    spec = draft.encoder_pair.online
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads) == (
        spec.latent_width,
        spec.n_layers,
        spec.n_heads,
    )
    assert (cfg.mlp_ratio, cfg.block_size, cfg.dropout, cfg.bias) == (
        spec.mlp_ratio,
        spec.block_size,
        spec.dropout,
        spec.bias,
    )
    assert cfg.vocab_size == draft.codec.cardinality
    assert cfg.tokenizer == "byte"
    assert cfg.tie_embeddings is False
    assert cfg.block == "baseline"
    assert cfg.norm == "layernorm"


@pytest.mark.parametrize(
    ("arm", "policy_type"),
    [
        (D1Arm.NONE, OnlineTargetUpdate),
        (D1Arm.AC, OnlineTargetUpdate),
        (D1Arm.SG, StopGradientEmaUpdate),
        (D1Arm.BOTH, StopGradientEmaUpdate),
    ],
)
def test_target_policy_translation_preserves_the_factorial_cells(arm, policy_type) -> None:
    policy = target_policy_from_draft(_drafts()[arm])

    assert type(policy) is policy_type
    if isinstance(policy, StopGradientEmaUpdate):
        assert policy.state_dict()["schedule"] == {"kind": "fixed", "tau": 0.99}


def test_cosine_ema_translation_preserves_every_registered_constant() -> None:
    baseline = _drafts()[D1Arm.NONE]
    family = build_d1_experiment_family(
        codec=baseline.codec,
        encoder_pair=baseline.encoder_pair,
        execution_protocol=replace(
            baseline.execution_protocol,
            optimization=replace(
                baseline.execution_protocol.optimization,
                max_steps=3,
                warmup_steps=1,
            ),
        ),
        ema=EMAParameters(
            kind=EMAKind.COSINE_SCHEDULE,
            decay=0.5,
            final_decay=0.9,
            schedule_steps=3,
        ),
    )

    policy = target_policy_from_draft(family.draft_for(D1Arm.SG))

    assert isinstance(policy, StopGradientEmaUpdate)
    assert isinstance(policy.schedule, CosineEmaDecaySchedule)
    assert policy.state_dict()["schedule"] == {
        "kind": "cosine",
        "start_decay": 0.5,
        "final_decay": 0.9,
        "total_successful_commits": 3,
    }


@pytest.mark.parametrize("arm", [D1Arm.NONE, D1Arm.SG])
def test_collapse_off_builds_the_explicit_control(arm: D1Arm) -> None:
    assert type(collapse_regularizer_from_draft(_drafts()[arm])) is NoCollapseRegularizer


@pytest.mark.parametrize("arm", [D1Arm.AC, D1Arm.BOTH])
def test_moment_translation_preserves_every_registered_constant(arm: D1Arm) -> None:
    regularizer = collapse_regularizer_from_draft(_drafts()[arm])

    assert isinstance(regularizer, MomentCollapseRegularizer)
    assert regularizer.gamma == 1.0
    assert regularizer.variance_weight == 1.0
    assert regularizer.covariance_weight == 0.04
    assert regularizer.variance_eps == 1e-4
