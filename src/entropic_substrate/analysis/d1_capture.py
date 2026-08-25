"""Fresh-analysis admission for the heterogeneous D1 capture handoff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import NoReturn, cast

import entropic_substrate.telemetry as telemetry_api
from entropic_substrate.analysis.d1_factorial import (
    D1AnalysisCell,
    D1AppliedExclusion,
    D1FactorialExclusionAuthority,
    D1MetricPoint,
    seal_d1_factorial_exclusion_authority,
)
from entropic_substrate.evidence.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
    D1ValidatedTerminalDefect,
)
from entropic_substrate.evidence.d1_capture_artifacts import (
    D1ArtifactIdentity,
    D1CaptureArtifactError,
    verify_d1_artifact,
)
from entropic_substrate.evidence.d1_capture_roster import (
    D1PairExclusion,
    D1PairExclusionReason,
)
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.evidence.d1_telemetry_values import (
    D1_SCALAR_METRICS,
    D1RunEvidence,
    D1RunStatus,
)
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass, canonical_digest
from entropic_substrate.experiments.d1_registered_design import (
    D1RegisteredDesign,
    decision_contract_digest,
)

__all__ = [
    "D1AdmittedAnalysisCapture",
    "D1AnalysisAdmissionError",
    "D1AppliedExclusion",
    "D1FactorialExclusionAuthority",
    "admit_d1_analysis_capture",
    "derive_d1_analysis_exclusions",
    "issue_d1_factorial_exclusion_authority",
]


class D1AnalysisAdmissionError(ValueError):
    """A capture cannot enter confirmatory D1 arithmetic."""

    def __init__(self, *, code: str, path: str, detail: str) -> None:
        self.code = code
        self.path = path
        self.context = MappingProxyType({"code": code, "path": path})
        super().__init__(f"D1 analysis admission {code} at {path}: {detail}")


def _fail(code: str, path: str, detail: str, cause: Exception | None = None) -> NoReturn:
    error = D1AnalysisAdmissionError(code=code, path=path, detail=detail)
    if cause is None:
        raise error
    raise error from cause


@dataclass(frozen=True, slots=True)
class D1AdmittedAnalysisCapture:
    design_digest: str
    decision_contract_digest: str
    capture_digest: str
    cells: tuple[D1AnalysisCell, ...]
    exclusions: tuple[D1AppliedExclusion, ...]
    runs: tuple[D1RunEvidence, ...]
    input_artifacts: tuple[D1ArtifactIdentity, ...]


def _validate_design(
    design: D1RegisteredDesign,
    contract: D1DecisionContract,
    aggregate: D1HeterogeneousCaptureAggregate,
) -> None:
    if decision_contract_digest(contract) != design.decision_contract_digest:
        _fail("design", "decision_contract", "differs from the registered design")
    roster = aggregate.roster
    if (
        roster.family_digest != design.family_digest
        or roster.preregistration_path != design.preregistration.path
        or roster.preregistration_commit != design.preregistration.commit
    ):
        _fail("design", "roster", "family or preregistration identity differs")
    expected = {(arm, seed) for seed in design.registered_seeds for arm in D1Arm}
    observed = {(item.arm, item.seed) for item in roster.expected_confirmatory}
    if observed != expected or len(observed) != len(roster.expected_confirmatory):
        _fail("design", "roster", "does not equal the registered seed-by-arm matrix")
    for item in roster.expected_confirmatory:
        if item.construction_digest != design.arm_construction_digests[item.arm.value]:
            _fail("design", item.logical_run_id, "construction digest differs")


def derive_d1_analysis_exclusions(
    *,
    aggregate: D1HeterogeneousCaptureAggregate,
    decision_contract: D1DecisionContract,
) -> tuple[D1AppliedExclusion, ...]:
    """Purely derive registered whole-block exclusions from validated defects."""
    if (
        type(aggregate) is not D1HeterogeneousCaptureAggregate
        or type(decision_contract) is not D1DecisionContract
    ):
        raise TypeError("D1 exclusion derivation requires validated aggregate and contract")
    defects_by_pair: dict[str, list[D1ValidatedTerminalDefect]] = {}
    for defect in aggregate.terminal_defects:
        defects_by_pair.setdefault(defect.expected.pairing_id, []).append(defect)
    claims = tuple(
        D1PairExclusion(
            pairing_id=pairing_id,
            member_run_ids=tuple(
                item.logical_run_id
                for item in aggregate.roster.expected_confirmatory
                if item.pairing_id == pairing_id
            ),
            reason=D1PairExclusionReason.NONFINITE_FINAL_LOSS,
            evidence_paths=tuple(
                sorted(item.capture_input.journal.normalized_path for item in defects)
            ),
        )
        for pairing_id, defects in sorted(defects_by_pair.items())
    )
    if claims != aggregate.pair_exclusions:
        _fail("exclusion", "pair_exclusions", "does not equal recomputed exclusions")
    applied: list[D1AppliedExclusion] = []
    for claim in claims:
        defects = defects_by_pair[claim.pairing_id]
        matches = {
            rule.reason_id
            for defect in defects
            for rule in decision_contract.exclusion_rules
            if rule.measurement == defect.record.metric_name
            and rule.predicate == "NONFINITE"
            and rule.scope == "WHOLE_SEED_BLOCK"
        }
        if len(matches) != 1:
            _fail(
                "exclusion",
                claim.pairing_id,
                "terminal defect lacks one exact registered whole-block rule",
            )
        applied.append(
            D1AppliedExclusion(
                seed=defects[0].expected.seed,
                pairing_id=claim.pairing_id,
                member_run_ids=claim.member_run_ids,
                reason_id=matches.pop(),
                evidence_paths=claim.evidence_paths,
                evidence_digest=canonical_digest(
                    {
                        "defects": [
                            item.manifest
                            for item in sorted(
                                defects, key=lambda value: value.expected.logical_run_id
                            )
                        ]
                    }
                ),
            )
        )
    return tuple(applied)


def issue_d1_factorial_exclusion_authority(
    *,
    design: D1RegisteredDesign,
    decision_contract: D1DecisionContract,
    aggregate: D1HeterogeneousCaptureAggregate,
    admitted: D1AdmittedAnalysisCapture,
) -> D1FactorialExclusionAuthority:
    """Seal the exact exclusions rederived from one admitted capture."""
    if type(admitted) is not D1AdmittedAnalysisCapture:
        raise TypeError("D1 factorial authority requires admitted analysis capture")
    _validate_design(design, decision_contract, aggregate)
    exclusions = derive_d1_analysis_exclusions(
        aggregate=aggregate,
        decision_contract=decision_contract,
    )
    if (
        admitted.design_digest != design.design_digest
        or admitted.decision_contract_digest != design.decision_contract_digest
        or admitted.capture_digest != aggregate.manifest_digest
        or admitted.exclusions != exclusions
    ):
        _fail("factorial_authority", "admitted", "differs from authoritative capture")
    return seal_d1_factorial_exclusion_authority(
        design_digest=design.design_digest,
        decision_contract_digest=design.decision_contract_digest,
        capture_digest=aggregate.manifest_digest,
        exclusions=exclusions,
    )


def _verify_inputs(root: Path, aggregate: D1HeterogeneousCaptureAggregate) -> None:
    if not aggregate.dvc_handoff.analysis_ready:
        _fail("dvc", "capture", "DVC handoff is not analysis-ready")
    for artifact in aggregate.dvc_handoff.artifacts:
        try:
            verify_d1_artifact(root, artifact)
        except D1CaptureArtifactError as error:
            _fail("artifact", artifact.normalized_path, str(error), error)


def _read_complete(root: Path, capture: object) -> D1RunEvidence:
    from entropic_substrate.evidence.d1_capture_bundle import D1ValidatedRunCapture

    if type(capture) is not D1ValidatedRunCapture:
        raise TypeError("complete analysis lane requires D1ValidatedRunCapture")
    expected = capture.expected
    path = root / capture.capture_input.journal.normalized_path
    try:
        run = telemetry_api.read_d1_run_evidence(
            path,
            expected_logical_run_id=expected.logical_run_id,
            expected_family_digest=expected.family_digest,
            expected_run_class=RunClass.CONFIRMATORY,
            expected_plan_digest=expected.plan_digest,
        )
    except Exception as error:
        _fail("public_reader", capture.capture_input.journal.normalized_path, str(error), error)
    if type(run) is not D1RunEvidence or run.status is not D1RunStatus.COMPLETE:
        _fail("public_reader", expected.logical_run_id, "did not return complete D1RunEvidence")
    identity = run.identity
    if (
        identity.logical_run_id != expected.logical_run_id
        or identity.family_digest != expected.family_digest
        or identity.construction_digest != expected.construction_digest
        or identity.execution_digest != expected.execution_digest
        or identity.aggregation_digest != expected.aggregation_digest
        or run.data_order_segments != capture.data_order_segments
        or run.checkpoint_receipts != capture.checkpoint_receipts
        or run.final_checkpoint_sha256 != capture.capture_input.checkpoint.sha256
    ):
        _fail("public_reader", expected.logical_run_id, "evidence differs from capture identity")
    plan = cast(Mapping[str, object], run.plan_manifest)
    if (plan.get("arm"), plan.get("seed"), plan.get("pairing_id")) != (
        expected.arm.value,
        expected.seed,
        expected.pairing_id,
    ):
        _fail("public_reader", expected.logical_run_id, "plan cell identity differs")
    return run


def _cell(expected: object, run: D1RunEvidence) -> D1AnalysisCell:
    from entropic_substrate.evidence.d1_capture_roster import D1ExpectedRun

    if type(expected) is not D1ExpectedRun:
        raise TypeError("analysis extraction requires D1ExpectedRun")
    metrics = {
        name: tuple(
            D1MetricPoint(item.step, item.eval_index, item.quality.scalar_metric(name))
            for item in run.evals
        )
        for name in sorted(D1_SCALAR_METRICS)
    }
    return D1AnalysisCell(expected.arm, expected.seed, expected.pairing_id, metrics)


def admit_d1_analysis_capture(
    *,
    root: str | Path,
    design: D1RegisteredDesign,
    decision_contract: D1DecisionContract,
    aggregate: D1HeterogeneousCaptureAggregate,
) -> D1AdmittedAnalysisCapture:
    """Re-admit an exact DVC capture without consulting MLflow or live state."""
    if (
        type(design) is not D1RegisteredDesign
        or type(decision_contract) is not D1DecisionContract
        or type(aggregate) is not D1HeterogeneousCaptureAggregate
    ):
        raise TypeError("D1 analysis admission requires nominal design, contract, and aggregate")
    capture_root = Path(root)
    _validate_design(design, decision_contract, aggregate)
    _verify_inputs(capture_root, aggregate)
    exclusions = derive_d1_analysis_exclusions(
        aggregate=aggregate,
        decision_contract=decision_contract,
    )
    excluded = {item.pairing_id for item in exclusions}
    complete = {
        (item.expected.arm, item.expected.seed): item
        for item in aggregate.complete_runs
        if item.expected.pairing_id not in excluded
    }
    runs: list[D1RunEvidence] = []
    cells: list[D1AnalysisCell] = []
    for seed in design.registered_seeds:
        if any(
            item.expected.seed == seed and item.expected.pairing_id in excluded
            for item in aggregate.terminal_defects
        ):
            continue
        for arm in D1Arm:
            capture = complete.get((arm, seed))
            if capture is None:
                _fail("capture_partition", f"seed={seed},arm={arm.value}", "finite cell is absent")
            run = _read_complete(capture_root, capture)
            runs.append(run)
            cells.append(_cell(capture.expected, run))
    return D1AdmittedAnalysisCapture(
        design_digest=design.design_digest,
        decision_contract_digest=design.decision_contract_digest,
        capture_digest=aggregate.manifest_digest,
        cells=tuple(cells),
        exclusions=exclusions,
        runs=tuple(runs),
        input_artifacts=aggregate.dvc_handoff.artifacts,
    )
