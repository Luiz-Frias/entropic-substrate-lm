"""The D1 composition root runs the whole capture-to-receipt analysis chain."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import entropic_substrate.telemetry as telemetry_api
from d1_launch_admission_fixtures import (
    PREREG_BLOB,
    PREREG_COMMIT,
    PREREG_PATH,
    family,
)
from entropic_substrate.analysis.d1_analysis_api import (
    D1AnalysisCompositionRequest,
    D1InProcessCompositionRunnerFake,
    execute_d1_analysis,
)
from entropic_substrate.analysis.d1_execution_receipt import (
    D1IsolationRequest,
)
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.experiments.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)

from .test_d1_aliased_analysis import _aliases
from .test_d1_analysis_capture import _analysis_case
from .test_d1_analysis_receipt import _implementation
from .test_d1_clause_policy import _policy
from .test_d1_report_diagnostics import _policy as diagnostic_policy

pytestmark = pytest.mark.unit


def _decision(base: D1DecisionContract) -> D1DecisionContract:
    analysis_policy = _policy(future_threshold=0.0)
    clauses = analysis_policy["clauses"]
    assert isinstance(clauses, dict)
    replacements = {
        "noncollapse": ("effective_rank", "AT_LEAST", 1.0),
        "future_grounding": ("mean_baseline_improvement", "AT_LEAST", 0.0),
        "language": ("nats_per_byte", "AT_MOST", 1.0),
    }
    for clause_id, (metric, operator, threshold) in replacements.items():
        clause = clauses[clause_id]
        assert isinstance(clause, dict)
        by_arm = clause["conditions_by_arm"]
        assert isinstance(by_arm, dict)
        for arm in by_arm:
            by_arm[arm] = [
                {
                    "contrast_id": None,
                    "estimator": "last_k_mean",
                    "metric": metric,
                    "operator": operator,
                    "smoothing_window": 3,
                    "source": "ARM_ENDPOINT_MEAN",
                    "threshold": threshold,
                }
            ]
    contrast_rules = {
        name: replace(
            rule,
            endpoint=replace(rule.endpoint, smoothing_window=3),
            registered_n=1,
            df_rule="NOT_APPLICABLE",
        )
        for name, rule in base.contrast_rules.items()
    }
    return D1DecisionContract(
        family_identity=base.family_identity,
        registered_constants={
            "analysis_policy": analysis_policy,
            "diagnostic_policy": diagnostic_policy(target=0.79),
            "primary": {"alpha": 0.025},
            "seeds": [7],
        },
        arm_ids=base.arm_ids,
        contrast_ids=base.contrast_ids,
        clause_ids=base.clause_ids,
        contrast_rules=contrast_rules,
        clause_rules=base.clause_rules,
        exclusion_rules=base.exclusion_rules,
        clause_precedence=base.clause_precedence,
        verdict_vocabulary=base.verdict_vocabulary,
        decision_rules=base.decision_rules,
    )


def _design(decision: D1DecisionContract) -> D1RegisteredDesign:
    admitted = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=family(),
            decision_contract=decision,
            registered_seeds=(7,),
            preregistration=D1PreregistrationArtifact(
                path=PREREG_PATH,
                blob_sha256=PREREG_BLOB,
                commit=PREREG_COMMIT,
            ),
        )
    )
    assert type(admitted) is D1RegisteredDesign
    return admitted


def test_composition_executes_capture_through_receipt_inside_runner_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, base, aggregate, runs = _analysis_case(tmp_path)
    decision = _decision(base)
    design = _design(decision)
    calls: list[str] = []

    def read(_path: Path, **expected: object):
        logical_run_id = str(expected["expected_logical_run_id"])
        calls.append(logical_run_id)
        return runs[logical_run_id]

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", read)
    request = D1AnalysisCompositionRequest(
        root=root,
        design=design,
        decision_contract=decision,
        aggregate=aggregate,
        aliases=_aliases(),
        isolation=D1IsolationRequest(),
    )
    execution = execute_d1_analysis(
        request=request,
        runner=D1InProcessCompositionRunnerFake(),
        implementation=_implementation(),
    )

    assert calls == [item.expected.logical_run_id for item in aggregate.complete_runs]
    assert execution.composition.factorial is not None
    assert execution.composition.factorial.included_seeds == (7,)
    assert execution.composition.clauses.feasible_arms
    assert len(execution.composition.diagnostics) == 4
    assert len(execution.composition.analysis.manifest["diagnostics"]) == 4  # type: ignore[arg-type]
    assert execution.receipt.manifest["filing_eligible"] is False
    assert execution.receipt.manifest["analysis_digest"] == (
        execution.composition.analysis.analysis_digest
    )
    assert request.request_digest == request.request_digest


def test_request_digest_changes_with_capture_or_alias_commitment(tmp_path: Path) -> None:
    root, _, base, aggregate, _ = _analysis_case(tmp_path)
    decision = _decision(base)
    design = _design(decision)
    request = D1AnalysisCompositionRequest(
        root=root,
        design=design,
        decision_contract=decision,
        aggregate=aggregate,
        aliases=_aliases(),
        isolation=D1IsolationRequest(),
    )
    changed_alias = replace(request, aliases=replace(_aliases(), sealed_commitment="a" * 64))
    changed_capture = replace(
        request,
        aggregate=replace(
            aggregate,
            dvc_handoff=replace(
                aggregate.dvc_handoff, artifacts=tuple(reversed(aggregate.dvc_handoff.artifacts))
            ),
        ),
    )

    assert changed_alias.request_digest != request.request_digest
    assert changed_capture.request_digest != request.request_digest


def test_execution_requires_nominal_implementation_identity(tmp_path: Path) -> None:
    root, _, base, aggregate, _ = _analysis_case(tmp_path)
    request = D1AnalysisCompositionRequest(
        root=root,
        design=_design(_decision(base)),
        decision_contract=_decision(base),
        aggregate=aggregate,
        aliases=_aliases(),
        isolation=D1IsolationRequest(),
    )
    with pytest.raises(TypeError, match="implementation"):
        execute_d1_analysis(
            request=request,
            runner=D1InProcessCompositionRunnerFake(),
            implementation=object(),  # type: ignore[arg-type]
        )
