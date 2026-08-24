"""D1 report probes use registered targets and same-cursor measured work."""

from __future__ import annotations

from pathlib import Path

import pytest

import entropic_substrate.telemetry as telemetry_api
from d1_launch_admission_fixtures import (
    PREREG_BLOB,
    PREREG_COMMIT,
    PREREG_PATH,
    family,
)
from d1_launch_admission_fixtures import (
    contract as base_contract,
)
from entropic_substrate.analysis.d1_capture import admit_d1_analysis_capture
from entropic_substrate.analysis.d1_diagnostics import (
    D1DiagnosticPolicyError,
    build_d1_report_diagnostics,
    parse_d1_diagnostic_policy,
)
from entropic_substrate.evidence.d1_contract import D1DecisionContract
from entropic_substrate.experiments.d1_family import D1ExperimentFamily
from entropic_substrate.experiments.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
)

from .test_d1_analysis_capture import _analysis_case

pytestmark = pytest.mark.unit


def _policy(*, target: float = 0.79) -> dict[str, object]:
    return {
        "aulc": {"metric": "nats_per_byte", "normalization": "RAW_TRAPEZOID"},
        "quality_target": {
            "metric": "nats_per_byte",
            "operator": "AT_MOST",
            "threshold": target,
        },
        "schema_version": "d1-diagnostic-policy-v1",
    }


def _contract(base: D1DecisionContract, policy: dict[str, object]) -> D1DecisionContract:
    return D1DecisionContract(
        family_identity=base.family_identity,
        registered_constants={
            "diagnostic_policy": policy,
            "primary": {"alpha": 0.025},
            "seeds": [7],
        },
        arm_ids=base.arm_ids,
        contrast_ids=base.contrast_ids,
        clause_ids=base.clause_ids,
        contrast_rules=base.contrast_rules,
        clause_rules=base.clause_rules,
        exclusion_rules=base.exclusion_rules,
        clause_precedence=base.clause_precedence,
        verdict_vocabulary=base.verdict_vocabulary,
        decision_rules=base.decision_rules,
    )


def _design(
    decision: D1DecisionContract, experiment_family: D1ExperimentFamily
) -> D1RegisteredDesign:
    admitted = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=experiment_family,
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


def _admitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, target: float = 0.79):
    root, old_design, base, aggregate, runs = _analysis_case(tmp_path)
    decision = _contract(base, _policy(target=target))
    design = _design(decision, family())

    def read(_path: Path, **expected: object):
        return runs[str(expected["expected_logical_run_id"])]

    monkeypatch.setattr(telemetry_api, "read_d1_run_evidence", read)
    admitted = admit_d1_analysis_capture(
        root=root,
        design=design,
        decision_contract=decision,
        aggregate=aggregate,
    )
    assert old_design.family_digest == design.family_digest
    return admitted, decision


def test_aulc_and_first_quality_crossing_use_the_measured_eval_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted, decision = _admitted(tmp_path, monkeypatch)
    diagnostics = build_d1_report_diagnostics(
        admitted=admitted,
        policy=parse_d1_diagnostic_policy(decision),
        decision_contract=decision,
    )

    assert len(diagnostics) == 4
    first = diagnostics[0]
    assert tuple((point.step, point.value) for point in first.trajectory) == (
        (0, 0.8),
        (1, 0.79),
        (2, 0.78),
    )
    assert first.aulc == pytest.approx(1.58)
    assert first.aulc_deciding is False
    assert first.quality_target.reached is True
    assert first.quality_target.step == 1
    assert first.quality_target.trainer_wall_seconds == 2.0
    assert first.quality_target.source_bytes == 4
    assert first.quality_target.model_tokens == 4
    assert first.evidence_class == "DERIVED_FROM_MEASURED"


def test_unreached_target_is_null_not_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    admitted, decision = _admitted(tmp_path, monkeypatch, target=0.1)
    (first, *_) = build_d1_report_diagnostics(
        admitted=admitted,
        policy=parse_d1_diagnostic_policy(decision),
        decision_contract=decision,
    )

    assert first.quality_target.reached is False
    assert first.quality_target.step is None
    assert first.quality_target.trainer_wall_seconds is None
    assert first.quality_target.source_bytes is None
    assert first.quality_target.model_tokens is None


def test_missing_or_unknown_diagnostic_policy_has_no_fallback() -> None:
    with pytest.raises(D1DiagnosticPolicyError, match="diagnostic_policy"):
        parse_d1_diagnostic_policy(base_contract(seeds=(7,)))
    with pytest.raises(D1DiagnosticPolicyError, match="normalization"):
        parse_d1_diagnostic_policy(
            _contract(
                base_contract(seeds=(7,)),
                {
                    **_policy(),
                    "aulc": {"metric": "nats_per_byte", "normalization": "INFER_IT"},
                },
            )
        )
