"""Blind D1 machine analysis removes every construction-bearing arm label."""

from __future__ import annotations

import json

import pytest

from entropic_substrate.analysis.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.analysis.d1_clause_policy import (
    evaluate_d1_clauses,
    parse_d1_clause_policy,
)
from entropic_substrate.analysis.d1_diagnostics import (
    D1QualityTargetResult,
    D1ReportRunDiagnostic,
    D1TrajectoryPoint,
)
from entropic_substrate.analysis.d1_factorial import compute_d1_factorial
from entropic_substrate.analysis.d1_machine_analysis import (
    D1AliasError,
    D1ArmAliasMap,
    build_d1_aliased_machine_analysis,
)
from entropic_substrate.experiments.d1_primitives import D1Arm

from .test_d1_clause_policy import _contract_with_policy, _policy
from .test_d1_factorial_analysis import _authority, _cells, _design
from .test_d1_report_diagnostics import _policy as diagnostic_policy

pytestmark = pytest.mark.unit


def _aliases(**overrides: str) -> D1ArmAliasMap:
    values = {
        D1Arm.NONE: "A",
        D1Arm.SG: "B",
        D1Arm.AC: "C",
        D1Arm.BOTH: "D",
    }
    for arm_name, alias in overrides.items():
        values[D1Arm(arm_name)] = alias
    return D1ArmAliasMap(values, sealed_commitment="f" * 64)


def _diagnostics() -> tuple[D1ReportRunDiagnostic, ...]:
    return tuple(
        D1ReportRunDiagnostic(
            arm=cell.arm,
            seed=cell.seed,
            aulc_metric="nats_per_byte",
            aulc=float(cell.seed),
            aulc_normalization="RAW_TRAPEZOID",
            aulc_deciding=False,
            trajectory=(D1TrajectoryPoint(0, 0, cell.metrics["nats_per_byte"][0].value),),
            quality_target=D1QualityTargetResult(False, None, None, None, None, None, None),
        )
        for cell in _cells()
    )


def _packet(*, future_threshold: float = 15.0):
    decision = _contract_with_policy(
        _policy(future_threshold=future_threshold),
        diagnostic_policy=diagnostic_policy(),
    )
    design = _design(decision)
    cells = _cells()
    factorial = compute_d1_factorial(
        design=design,
        decision_contract=decision,
        cells=cells,
        exclusion_authority=_authority(design),
    )
    clauses = evaluate_d1_clauses(
        policy=parse_d1_clause_policy(decision),
        decision_contract=decision,
        factorial=factorial,
        cells=cells,
    )
    admitted = D1AdmittedAnalysisCapture(
        design_digest=design.design_digest,
        decision_contract_digest=design.decision_contract_digest,
        capture_digest="e" * 64,
        cells=cells,
        exclusions=(),
        runs=(),
        input_artifacts=(),
    )
    return build_d1_aliased_machine_analysis(
        design=design,
        decision_contract=decision,
        admitted=admitted,
        factorial=factorial,
        clauses=clauses,
        diagnostics=_diagnostics(),
        aliases=_aliases(),
    )


def _strings(value: object) -> set[str]:
    if isinstance(value, str):
        return {value}
    if isinstance(value, dict):
        return {item for key, child in value.items() for item in ({key} | _strings(child))}
    if isinstance(value, list):
        return {item for child in value for item in _strings(child)}
    return set()


def test_machine_packet_uses_only_opaque_arm_contrast_and_effect_aliases() -> None:
    packet = _packet()
    manifest = packet.manifest
    encoded = json.dumps(manifest, allow_nan=False, sort_keys=True)
    strings = _strings(manifest)

    assert packet.analysis_digest == packet.analysis_digest
    assert manifest["alias_commitment"] == "f" * 64
    assert {"none", "sg", "ac", "both"}.isdisjoint(strings)
    assert {"sg-main", "ac-main", "interaction"}.isdisjoint(strings)
    assert "sg_at_ac_off" not in encoded
    assert [item["contrast_alias"] for item in manifest["contrasts"]] == [  # type: ignore[index]
        "C01",
        "C02",
        "C03",
    ]
    assert [item["effect_alias"] for item in manifest["factorial_effects"]] == [  # type: ignore[index]
        "F01",
        "F02",
        "F03",
        "F04",
        "F05",
        "F06",
        "F07",
    ]
    assert manifest["feasible_aliases"] == ["D"]
    assert len(manifest["diagnostics"]) == 16  # type: ignore[arg-type]


def test_empty_feasible_set_and_all_clause_riders_survive_serialization() -> None:
    manifest = _packet(future_threshold=30.0).manifest

    assert manifest["feasible_aliases"] == []
    assert manifest["clause_precedence"] == ["noncollapse", "future_grounding", "language"]
    by_alias = manifest["clauses_by_alias"]
    assert isinstance(by_alias, dict)
    assert set(by_alias["B"]) == {"noncollapse", "future_grounding", "language"}
    assert by_alias["B"]["future_grounding"]["status"] == "FAIL"
    assert by_alias["B"]["language"]["status"] == "PASS"


def test_alias_map_requires_exact_unique_opaque_labels_and_sealed_commitment() -> None:
    with pytest.raises(D1AliasError, match="unique"):
        _aliases(sg="A")
    with pytest.raises(D1AliasError, match="opaque"):
        _aliases(sg="sg")
    with pytest.raises(D1AliasError, match="commitment"):
        D1ArmAliasMap(
            {D1Arm.NONE: "A", D1Arm.SG: "B", D1Arm.AC: "C", D1Arm.BOTH: "D"},
            sealed_commitment="not-a-hash",
        )
