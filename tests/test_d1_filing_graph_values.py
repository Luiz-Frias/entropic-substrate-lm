"""Factory-only filing authority and recomputed capture-integrity values."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from d1_verdict_v2_fixtures import decision_contract
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.kernel.verdict.d1_types import D1FamilyIdentity
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.ladder.capture.custody.d1_capture_aggregate_validation import (
    build_d1_capture_aggregate,
)
from entropic_substrate.ladder.capture.custody.d1_capture_roster import (
    D1PairExclusion,
    D1PairExclusionReason,
)
from entropic_substrate.ladder.capture.custody.d1_capture_validation import build_d1_capture_bundle
from entropic_substrate.ladder.file.d1_filing_graph import (
    ValidatedD1Filing,
    recompute_capture_integrity,
)

from .d1_capture_aggregate_fixtures import heterogeneous_family
from .d1_capture_fixtures import capture_family

pytestmark = pytest.mark.unit


def _one_seed_contract(
    aggregate: D1HeterogeneousCaptureAggregate,
) -> D1DecisionContract:
    base = decision_contract()
    return replace(
        base,
        family_identity=D1FamilyIdentity(
            family_digest=aggregate.roster.family_digest,
            arm_construction_digests={
                item.arm.value: item.construction_digest
                for item in aggregate.roster.expected_confirmatory
            },
        ),
        registered_constants={"seeds": [7], "primary": {"alpha": 0.025}},
        contrast_rules={
            name: replace(rule, registered_n=1, df_rule="NOT_APPLICABLE")
            for name, rule in base.contrast_rules.items()
        },
        exclusion_rules=(replace(base.exclusion_rules[0], measurement="nats_per_byte"),),
    )


def test_capture_integrity_is_recomputed_from_heterogeneous_aggregate(tmp_path: Path) -> None:
    roster, inputs, root = capture_family(tmp_path)
    aggregate = build_d1_capture_aggregate(
        root,
        roster=roster,
        complete_inputs=inputs,
        terminal_defect_inputs=(),
    )

    decision = _one_seed_contract(aggregate)
    integrity = recompute_capture_integrity(aggregate, decision)

    assert integrity.roster_complete
    assert integrity.pairing_verified
    assert integrity.data_order_verified
    assert integrity.completion_verified
    assert integrity.exclusions_verified
    assert integrity.applied_exclusions == ()
    assert integrity.effective_sample_size == 1
    assert len(integrity.evidence_digest) == 64

    duplicate_seed_contract = replace(
        decision,
        registered_constants={"seeds": [7, 7], "primary": {"alpha": 0.025}},
    )
    assert not recompute_capture_integrity(aggregate, duplicate_seed_contract).roster_complete


def test_terminal_defect_pair_is_measured_but_not_an_effective_sample(tmp_path: Path) -> None:
    roster, complete, defects, attempts, root = heterogeneous_family(tmp_path)
    aggregate = build_d1_capture_aggregate(
        root,
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )

    decision = _one_seed_contract(aggregate)
    integrity = recompute_capture_integrity(aggregate, decision)

    assert integrity.passed
    assert integrity.exclusions_verified
    assert integrity.effective_sample_size == 0
    assert integrity.applied_exclusions is not None
    _, reason_id, evidence_digest = integrity.applied_exclusions[0]
    assert reason_id == "nonfinite-final-loss"
    assert len(evidence_digest) == 64

    moved = next(iter(aggregate.partition_proof.complete_run_ids))
    drifted_partition = replace(
        aggregate.partition_proof,
        complete_run_ids=aggregate.partition_proof.complete_run_ids - {moved},
        terminal_defect_run_ids=aggregate.partition_proof.terminal_defect_run_ids | {moved},
    )
    assert not recompute_capture_integrity(
        replace(aggregate, partition_proof=drifted_partition), decision
    ).roster_complete

    first = aggregate.complete_runs[0]
    drifted_order = replace(
        aggregate,
        complete_runs=(
            replace(first, data_order_segments=((0, "different-order", 1),)),
            *aggregate.complete_runs[1:],
        ),
    )
    assert not recompute_capture_integrity(drifted_order, decision).data_order_verified

    unmatched_rule = replace(decision.exclusion_rules[0], measurement="unregistered_metric")
    unmatched_decision = replace(
        decision,
        registered_constants={"seeds": [7], "primary": {"alpha": 0.025}},
        exclusion_rules=(unmatched_rule,),
    )
    assert not recompute_capture_integrity(aggregate, unmatched_decision).exclusions_verified


def test_mixed_aggregate_counts_only_complete_nonexcluded_blocks(tmp_path: Path) -> None:
    seeds = (7, 8, 9)
    roster, complete, defects, attempts, root = heterogeneous_family(tmp_path, seeds=seeds)
    aggregate = build_d1_capture_aggregate(
        root,
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )
    base = _one_seed_contract(aggregate)
    decision = replace(
        base,
        registered_constants={"seeds": list(seeds), "primary": {"alpha": 0.025}},
        contrast_rules={
            name: replace(
                rule,
                n_rule="NONEXCLUDED_SEED_COUNT",
                registered_n=None,
                df_rule="N_MINUS_ONE",
            )
            for name, rule in base.contrast_rules.items()
        },
    )

    integrity = recompute_capture_integrity(aggregate, decision)

    assert integrity.passed
    assert integrity.effective_sample_size == 2


def test_unsupported_exclusion_cannot_reduce_effective_sample_size(tmp_path: Path) -> None:
    roster, inputs, root = capture_family(tmp_path)
    aggregate = build_d1_capture_aggregate(
        root,
        roster=roster,
        complete_inputs=inputs,
        terminal_defect_inputs=(),
    )
    pairing_id = roster.pairing_ids[0]
    claim = D1PairExclusion(
        pairing_id=pairing_id,
        member_run_ids=tuple(
            item.logical_run_id
            for item in roster.expected_confirmatory
            if item.pairing_id == pairing_id
        ),
        reason=D1PairExclusionReason.NONFINITE_FINAL_LOSS,
        evidence_paths=("results/captured/forged-defect.jsonl",),
    )

    integrity = recompute_capture_integrity(
        replace(aggregate, pair_exclusions=(claim,)), _one_seed_contract(aggregate)
    )

    assert not integrity.exclusions_verified
    assert not integrity.passed
    assert integrity.effective_sample_size == 1


def test_legacy_complete_bundle_is_not_filing_authority(tmp_path: Path) -> None:
    roster, inputs, root = capture_family(tmp_path)
    bundle = build_d1_capture_bundle(root, roster=roster, capture_inputs=inputs)
    with pytest.raises(TypeError, match="heterogeneous capture aggregate"):
        recompute_capture_integrity(bundle, decision_contract())  # type: ignore[arg-type]


def test_gate_authority_value_is_factory_only() -> None:
    with pytest.raises(TypeError, match="factory-only"):
        ValidatedD1Filing()  # type: ignore[call-arg]
