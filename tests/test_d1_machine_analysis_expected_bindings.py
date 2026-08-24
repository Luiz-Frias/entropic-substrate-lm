"""Registered controls bind structurally valid D1 machine-analysis bytes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from d1_launch_admission_fixtures import family
from entropic_substrate.analysis.d1_capture import D1AdmittedAnalysisCapture
from entropic_substrate.analysis.d1_clause_policy import (
    evaluate_d1_clauses,
    parse_d1_clause_policy,
)
from entropic_substrate.analysis.d1_diagnostics import (
    D1QualityTargetResult,
    D1ReportRunDiagnostic,
    D1TrajectoryPoint,
    parse_d1_diagnostic_policy,
)
from entropic_substrate.analysis.d1_external_analysis_control import (
    decode_d1_external_analysis_input_control,
)
from entropic_substrate.analysis.d1_factorial import (
    D1AnalysisCell,
    D1MetricPoint,
    compute_d1_factorial,
)
from entropic_substrate.analysis.d1_machine_analysis import (
    D1AliasedMachineAnalysis,
    build_d1_aliased_machine_analysis,
)
from entropic_substrate.analysis.d1_machine_analysis_binding import (
    D1ContractBoundMachineAnalysis,
    D1MachineAnalysisBindingError,
    bind_d1_machine_analysis_contract,
)
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.experiments.d1_registered_design import (
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)
from entropic_substrate.experiments.d1_registration_codec import (
    serialize_decision_contract,
    serialize_registered_design,
)

from .test_d1_analysis_api import _decision
from .test_d1_external_analysis_input import _b64, _canonical
from .test_d1_external_analysis_input import _case as input_case

pytestmark = pytest.mark.unit


def _binding_case(tmp_path: Path):
    manifest, evidence, old_design, base, aggregate = input_case(tmp_path)
    decision = _decision(base)
    design = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=family(),
            decision_contract=decision,
            registered_seeds=(7,),
            preregistration=old_design.preregistration,
        )
    )
    assert type(design) is D1RegisteredDesign
    manifest["decision_contract_base64"] = _b64(
        serialize_decision_contract(decision).encode("utf-8")
    )
    manifest["registered_design_base64"] = _b64(serialize_registered_design(design).encode("utf-8"))
    manifest["expected_design_digest"] = design.design_digest
    control = decode_d1_external_analysis_input_control(
        _canonical(manifest), preregistration_evidence=evidence
    )
    points = {
        metric: tuple(D1MetricPoint(index, index, value) for index in range(3))
        for metric, value in (
            ("effective_rank", 2.0),
            ("mean_baseline_improvement", 0.1),
            ("nats_per_byte", 0.8),
        )
    }
    cells = tuple(D1AnalysisCell(arm, 7, "d1-pair-s7", points) for arm in D1Arm)
    factorial = compute_d1_factorial(
        design=design,
        decision_contract=decision,
        cells=cells,
    )
    clauses = evaluate_d1_clauses(
        policy=parse_d1_clause_policy(decision),
        decision_contract=decision,
        factorial=factorial,
        cells=cells,
    )
    diagnostic_policy = parse_d1_diagnostic_policy(decision)
    diagnostics = tuple(
        D1ReportRunDiagnostic(
            arm=arm,
            seed=7,
            aulc_metric=diagnostic_policy.aulc_metric,
            aulc=0.0,
            aulc_normalization=diagnostic_policy.aulc_normalization,
            aulc_deciding=False,
            trajectory=(D1TrajectoryPoint(0, 0, 0.8),),
            quality_target=D1QualityTargetResult(False, None, None, None, None, None, None),
        )
        for arm in D1Arm
    )
    admitted = D1AdmittedAnalysisCapture(
        design_digest=design.design_digest,
        decision_contract_digest=design.decision_contract_digest,
        capture_digest=aggregate.manifest_digest,
        cells=cells,
        exclusions=(),
        runs=(),
        input_artifacts=(),
    )
    packet = build_d1_aliased_machine_analysis(
        design=design,
        decision_contract=decision,
        admitted=admitted,
        factorial=factorial,
        clauses=clauses,
        diagnostics=diagnostics,
        aliases=control.alias_application.aliases,
    )
    return control, packet


def _changed(
    packet: D1AliasedMachineAnalysis, mutate: Callable[[dict[str, object]], None]
) -> bytes:
    manifest = packet.manifest
    mutate(manifest)
    return _canonical(manifest)


def _change_seed(root: dict[str, object]) -> None:
    root["included_seeds"] = [8]
    for diagnostic in root["diagnostics"]:  # type: ignore[union-attr]
        diagnostic["seed"] = 8


def test_registered_control_binds_the_generated_machine_analysis(tmp_path: Path) -> None:
    control, packet = _binding_case(tmp_path)
    payload = packet.canonical_json.encode("utf-8")

    bound = bind_d1_machine_analysis_contract(payload, expected=control)

    assert bound.canonical_bytes == payload
    assert bound.analysis == packet
    assert bound.input_manifest_sha256 == control.input_manifest_sha256
    assert bound.analysis_digest == packet.analysis_digest


def test_contract_bound_value_is_factory_only_and_copy_stays_bound(tmp_path: Path) -> None:
    control, packet = _binding_case(tmp_path)
    bound = bind_d1_machine_analysis_contract(packet.canonical_json.encode(), expected=control)
    with pytest.raises(TypeError, match="factory-only"):
        D1ContractBoundMachineAnalysis(
            canonical_bytes=bound.canonical_bytes,
            analysis=bound.analysis,
            expected=control,
            input_manifest_sha256=control.input_manifest_sha256,
        )
    with pytest.raises(ValueError, match="canonical"):
        replace(bound, analysis=D1AliasedMachineAnalysis("{}"))
    with pytest.raises(ValueError, match="input"):
        replace(bound, input_manifest_sha256="0" * 64)


@pytest.mark.parametrize(
    "field",
    ["capture_digest", "decision_contract_digest", "design_digest"],
)
def test_top_bindings_and_alias_commitment_must_match_input(tmp_path: Path, field: str) -> None:
    control, packet = _binding_case(tmp_path)
    with pytest.raises(D1MachineAnalysisBindingError, match="binding"):
        bind_d1_machine_analysis_contract(
            _changed(packet, lambda root: root["bindings"].update({field: "0" * 64})),  # type: ignore[union-attr]
            expected=control,
        )
    with pytest.raises(D1MachineAnalysisBindingError, match="alias"):
        bind_d1_machine_analysis_contract(
            _changed(packet, lambda root: root.update({"alias_commitment": "0" * 64})),
            expected=control,
        )


def test_registered_seed_partition_and_endpoint_are_exact(tmp_path: Path) -> None:
    control, packet = _binding_case(tmp_path)
    with pytest.raises(D1MachineAnalysisBindingError, match="seed"):
        bind_d1_machine_analysis_contract(
            _changed(packet, _change_seed),
            expected=control,
        )
    with pytest.raises(D1MachineAnalysisBindingError, match="endpoint"):
        bind_d1_machine_analysis_contract(
            _changed(packet, lambda root: root["endpoint"].update({"smoothing_window": 2})),  # type: ignore[union-attr]
            expected=control,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confidence_level", 0.5),
        ("multiplicity", "UNREGISTERED"),
        ("margins", {"practical": 9.0}),
        ("terms", [{"arm_alias": "A", "coefficient": 1.0}]),
    ],
)
def test_each_registered_contrast_field_is_bound(tmp_path: Path, field: str, value: object) -> None:
    control, packet = _binding_case(tmp_path)
    with pytest.raises(D1MachineAnalysisBindingError, match="contrast"):
        bind_d1_machine_analysis_contract(
            _changed(packet, lambda root: root["contrasts"][0].update({field: value})),  # type: ignore[index,union-attr]
            expected=control,
        )


def test_factorial_effects_and_clause_construction_are_exact(tmp_path: Path) -> None:
    control, packet = _binding_case(tmp_path)
    with pytest.raises(D1MachineAnalysisBindingError, match="effect"):
        bind_d1_machine_analysis_contract(
            _changed(packet, lambda root: root["factorial_effects"].pop()),  # type: ignore[union-attr]
            expected=control,
        )
    with pytest.raises(D1MachineAnalysisBindingError, match="clause"):
        bind_d1_machine_analysis_contract(
            _changed(packet, lambda root: root["clause_precedence"].reverse()),  # type: ignore[union-attr]
            expected=control,
        )
    with pytest.raises(D1MachineAnalysisBindingError, match="condition"):
        bind_d1_machine_analysis_contract(
            _changed(
                packet,
                lambda root: root["clauses_by_alias"][next(iter(root["clauses_by_alias"]))][  # type: ignore[index,union-attr]
                    "noncollapse"
                ]["conditions"][0].update({"threshold": 99.0}),
            ),
            expected=control,
        )
