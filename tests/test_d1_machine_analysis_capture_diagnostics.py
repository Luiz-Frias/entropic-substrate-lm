"""Authoritative capture and registered diagnostics complete D1 output verification."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from d1_launch_admission_fixtures import SEEDS, contract
from entropic_substrate.analysis.d1_capture import derive_d1_analysis_exclusions
from entropic_substrate.analysis.d1_external_analysis_control import (
    D1ExternalAnalysisInputControl,
)
from entropic_substrate.analysis.d1_machine_analysis import D1AliasedMachineAnalysis
from entropic_substrate.analysis.d1_machine_analysis_binding import (
    bind_d1_machine_analysis_contract,
)
from entropic_substrate.analysis.d1_machine_analysis_verification import (
    D1MachineAnalysisVerificationError,
    D1VerifiedMachineAnalysis,
    verify_d1_machine_analysis,
)
from entropic_substrate.evidence.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.evidence.d1_capture_aggregate_validation import (
    build_d1_capture_aggregate,
)
from entropic_substrate.experiments.d1_primitives import canonical_digest

from .d1_capture_aggregate_fixtures import heterogeneous_family
from .test_d1_external_analysis_input import _case as input_case
from .test_d1_machine_analysis_expected_bindings import _binding_case, _changed

pytestmark = pytest.mark.unit


def _case(tmp_path: Path):
    control, packet = _binding_case(tmp_path / "packet")
    *_, aggregate = input_case(tmp_path / "authority")
    assert aggregate.manifest_digest == control.aggregate.manifest_digest
    bound = bind_d1_machine_analysis_contract(packet.canonical_json.encode(), expected=control)
    return control, packet, bound, aggregate


def test_authoritative_capture_and_policy_issue_final_verified_value(tmp_path: Path) -> None:
    control, packet, bound, aggregate = _case(tmp_path)

    verified = verify_d1_machine_analysis(binding=bound, aggregate=aggregate)

    assert verified.binding is bound
    assert verified.analysis == packet
    assert verified.expected is control
    assert verified.aggregate is aggregate
    assert verified.canonical_bytes == packet.canonical_json.encode()
    assert verified.analysis_digest == packet.analysis_digest
    assert verified.input_manifest_sha256 == control.input_manifest_sha256
    assert verified.diagnostic_policy.target_metric == "nats_per_byte"
    target = verified.manifest["diagnostics"][0]["quality_target"]  # type: ignore[index]
    assert {
        field: target[field]  # type: ignore[index]
        for field in ("metric", "operator", "threshold")
    } == {
        "metric": verified.diagnostic_policy.target_metric,
        "operator": verified.diagnostic_policy.target_operator.value,
        "threshold": verified.diagnostic_policy.target_threshold,
    }


def test_final_value_is_factory_only_and_every_nested_identity_is_rebound(tmp_path: Path) -> None:
    control, packet, bound, aggregate = _case(tmp_path)
    verified = verify_d1_machine_analysis(binding=bound, aggregate=aggregate)
    with pytest.raises(TypeError, match="factory-only"):
        D1VerifiedMachineAnalysis(
            binding=bound,
            aggregate=aggregate,
            canonical_bytes=verified.canonical_bytes,
            analysis=packet,
            expected=control,
            input_manifest_sha256=control.input_manifest_sha256,
            analysis_digest=packet.analysis_digest,
            diagnostic_policy=verified.diagnostic_policy,
        )
    changed = _changed(
        packet,
        lambda root: root["endpoint"]["values_by_alias"]["A"].__setitem__(0, 0.7),  # type: ignore[index,union-attr]
    )
    other_binding = bind_d1_machine_analysis_contract(changed, expected=control)
    with pytest.raises(ValueError, match="binding"):
        replace(verified, binding=other_binding)
    with pytest.raises(ValueError, match="declaration"):
        replace(verified, analysis=D1AliasedMachineAnalysis("{}"))
    with pytest.raises(TypeError, match="input"):
        replace(verified, expected=cast(D1ExternalAnalysisInputControl, object()))
    with pytest.raises(ValueError, match="digest"):
        replace(verified, analysis_digest="0" * 64)
    with pytest.raises(ValueError, match="input"):
        replace(verified, input_manifest_sha256="0" * 64)


def test_aggregate_digest_and_exact_exclusion_rows_are_authoritative(tmp_path: Path) -> None:
    control, packet, bound, aggregate = _case(tmp_path)
    reordered = replace(aggregate, complete_runs=tuple(reversed(aggregate.complete_runs)))
    with pytest.raises(D1MachineAnalysisVerificationError, match="aggregate"):
        verify_d1_machine_analysis(binding=bound, aggregate=reordered)

    extra = _changed(
        packet,
        lambda root: root.update(
            {
                "exclusions": [
                    {
                        "evidence_digest": "0" * 64,
                        "exclusion_alias": "X01",
                        "reason_id": "nonfinite-final-loss",
                    }
                ]
            }
        ),
    )
    extra_bound = bind_d1_machine_analysis_contract(extra, expected=control)
    with pytest.raises(D1MachineAnalysisVerificationError, match="exclusion"):
        verify_d1_machine_analysis(binding=extra_bound, aggregate=aggregate)


@pytest.mark.parametrize(
    ("field", "value"),
    [("aulc_metric", "other_metric"), ("aulc_normalization", "OTHER")],
)
def test_diagnostic_metric_normalization_and_grid_bind_registered_policy(
    tmp_path: Path, field: str, value: object
) -> None:
    control, packet, _, aggregate = _case(tmp_path)
    changed = _changed(
        packet,
        lambda root: root["diagnostics"][0].update({field: value}),  # type: ignore[index,union-attr]
    )
    bound = bind_d1_machine_analysis_contract(changed, expected=control)
    with pytest.raises(D1MachineAnalysisVerificationError, match="diagnostic"):
        verify_d1_machine_analysis(binding=bound, aggregate=aggregate)


@pytest.mark.parametrize("metric_value", [9.0, 0.7])
def test_reached_target_must_satisfy_policy_and_same_cursor_metric(
    tmp_path: Path, metric_value: float
) -> None:
    control, packet, _, aggregate = _case(tmp_path)

    def reach(root: dict[str, object]) -> None:
        target = root["diagnostics"][0]["quality_target"]  # type: ignore[index]
        target.update(  # type: ignore[union-attr]
            {
                "eval_index": 0,
                "metric_value": metric_value,
                "model_tokens": 1,
                "reached": True,
                "source_bytes": 1,
                "step": 0,
                "trainer_wall_seconds": 1.0,
            }
        )

    bound = bind_d1_machine_analysis_contract(_changed(packet, reach), expected=control)
    with pytest.raises(D1MachineAnalysisVerificationError, match="quality target"):
        verify_d1_machine_analysis(binding=bound, aggregate=aggregate)


@pytest.mark.parametrize(
    ("field", "value"),
    [("metric", "other_metric"), ("operator", "AT_LEAST"), ("threshold", 0.5)],
)
def test_quality_target_policy_fields_bind_the_registered_diagnostic(
    tmp_path: Path, field: str, value: object
) -> None:
    control, packet, _, aggregate = _case(tmp_path)
    changed = _changed(
        packet,
        lambda root: root["diagnostics"][0]["quality_target"].update({field: value}),  # type: ignore[index,union-attr]
    )
    bound = bind_d1_machine_analysis_contract(changed, expected=control)

    with pytest.raises(D1MachineAnalysisVerificationError, match="quality target"):
        verify_d1_machine_analysis(binding=bound, aggregate=aggregate)


def test_quality_target_cursor_and_unreached_state_match_trajectory(tmp_path: Path) -> None:
    control, packet, _, aggregate = _case(tmp_path)

    def absent_cursor(root: dict[str, object]) -> None:
        root["diagnostics"][0]["quality_target"].update(  # type: ignore[index,union-attr]
            {
                "eval_index": 99,
                "metric_value": 0.7,
                "model_tokens": 1,
                "reached": True,
                "source_bytes": 1,
                "step": 99,
                "trainer_wall_seconds": 1.0,
            }
        )

    for changed in (
        _changed(packet, absent_cursor),
        _changed(
            packet,
            lambda root: root["diagnostics"][0]["trajectory"][0].update({"value": 0.7}),  # type: ignore[index,union-attr]
        ),
    ):
        bound = bind_d1_machine_analysis_contract(changed, expected=control)
        with pytest.raises(D1MachineAnalysisVerificationError, match="quality target"):
            verify_d1_machine_analysis(binding=bound, aggregate=aggregate)


def test_pure_exclusion_derivation_is_shared_with_analysis_admission(tmp_path: Path) -> None:
    roster, complete, defects, attempts, root = heterogeneous_family(tmp_path)
    aggregate = build_d1_capture_aggregate(
        root,
        roster=roster,
        complete_inputs=complete,
        terminal_defect_inputs=defects,
        attempt_evidence=attempts,
    )
    decision = contract()
    rule = replace(decision.exclusion_rules[0], measurement="nats_per_byte")
    decision = replace(
        decision,
        registered_constants={"primary": {"alpha": 0.025}, "seeds": list(SEEDS)},
        exclusion_rules=(rule,),
    )

    (exclusion,) = derive_d1_analysis_exclusions(
        aggregate=aggregate,
        decision_contract=decision,
    )

    assert exclusion.reason_id == rule.reason_id
    assert exclusion.evidence_digest == canonical_digest(
        {"defects": [aggregate.terminal_defects[0].manifest]}
    )
    assert exclusion.seed == aggregate.terminal_defects[0].expected.seed
    assert exclusion.pairing_id == aggregate.pair_exclusions[0].pairing_id
    assert len(exclusion.member_run_ids) == 4


def test_final_verifier_requires_factory_attested_aggregate(tmp_path: Path) -> None:
    _, _, bound, _ = _case(tmp_path)
    with pytest.raises(TypeError, match="aggregate"):
        verify_d1_machine_analysis(
            binding=bound,
            aggregate=cast(D1HeterogeneousCaptureAggregate, object()),
        )
