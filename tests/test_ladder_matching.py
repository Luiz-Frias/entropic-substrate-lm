"""Matched-total selection uses instantiated inventories, never count formulas."""

from dataclasses import replace

import pytest

from d1_runtime_fixtures import runtime_family
from entropic_substrate.experiments.d1 import D1Arm, MatchedEncoderPairSpec
from entropic_substrate.experiments.ladder import (
    MatchingLane,
    ParameterMatchReference,
    Rung,
    RungSpec,
    Treatment,
)
from entropic_substrate.experiments.ladder_matching import (
    MatchedRuntime,
    NoAdmissibleMatch,
    measured_inventory,
    select_total_match,
)
from entropic_substrate.predictive_state.ladder_runtime import LadderPredictiveSystem

pytestmark = pytest.mark.unit


def reference() -> LadderPredictiveSystem:
    base = runtime_family().draft_for(D1Arm.BOTH)
    return LadderPredictiveSystem(RungSpec(base, Rung.D2, Treatment.S, seed=7))


def candidate(anchor: LadderPredictiveSystem, width: int, gap: float = 0.01) -> RungSpec:
    encoder = replace(anchor.spec.base.encoder_pair.online, latent_width=width)
    base = replace(anchor.spec.base, encoder_pair=MatchedEncoderPairSpec(encoder, encoder))
    return RungSpec(
        base,
        Rung.D3,
        Treatment.S,
        seed=7,
        predecessor=anchor.spec,
        matching_lane=MatchingLane.MATCHED_TOTAL,
        parameter_reference=ParameterMatchReference(
            anchor.construction_identity, measured_inventory(anchor).total, gap
        ),
    )


def test_exact_match_counts_trainable_and_frozen_tensors() -> None:
    anchor = reference()
    selected = select_total_match(anchor, [candidate(anchor, 8)])
    assert isinstance(selected, MatchedRuntime)
    assert selected.inventory.total == sum(p.numel() for p in selected.runtime.parameters())
    assert selected.inventory.trainable + selected.inventory.frozen == selected.inventory.total
    assert selected.relative_gap == 0


def test_approximate_match_requires_explicit_tolerance_and_keeps_all_inventories() -> None:
    anchor = reference()
    rejected = select_total_match(anchor, [candidate(anchor, 4)])
    assert isinstance(rejected, NoAdmissibleMatch)
    accepted = select_total_match(anchor, [candidate(anchor, 4, 0.9), candidate(anchor, 8, 0.9)])
    assert isinstance(accepted, MatchedRuntime)
    assert accepted.runtime.spec.base.encoder_pair.latent_width == 8
    assert len(accepted.candidates) == 2


def test_reference_count_and_nonshape_changes_are_rejected() -> None:
    anchor = reference()
    selected = candidate(anchor, 8)
    false_reference = replace(selected.parameter_reference, total_parameters=1)
    with pytest.raises(ValueError, match="measured reference"):
        select_total_match(anchor, [replace(selected, parameter_reference=false_reference)])
    changed = replace(selected.base.encoder_pair.online, dropout=0.1)
    with pytest.raises(ValueError, match="shape"):
        replace(
            selected,
            base=replace(selected.base, encoder_pair=MatchedEncoderPairSpec(changed, changed)),
        )
