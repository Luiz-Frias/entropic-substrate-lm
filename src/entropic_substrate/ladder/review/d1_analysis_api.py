"""Composition root for the complete capture-to-receipt D1 analysis flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from entropic_substrate.__legacy.evidence.d1_capture_aggregate import (
    D1HeterogeneousCaptureAggregate,
)
from entropic_substrate.kernel.identity.d1_primitives import canonical_digest
from entropic_substrate.kernel.verdict.d1_contract import D1DecisionContract
from entropic_substrate.ladder.analyze.d1_capture import (
    D1AdmittedAnalysisCapture,
    admit_d1_analysis_capture,
    issue_d1_factorial_exclusion_authority,
)
from entropic_substrate.ladder.analyze.d1_clause_policy import (
    D1ClauseAnalysis,
    evaluate_d1_clauses,
    parse_d1_clause_policy,
)
from entropic_substrate.ladder.analyze.d1_diagnostics import (
    D1ReportRunDiagnostic,
    build_d1_report_diagnostics,
    parse_d1_diagnostic_policy,
)
from entropic_substrate.ladder.analyze.d1_factorial import (
    D1FactorialAnalysis,
    compute_d1_factorial,
)
from entropic_substrate.ladder.analyze.d1_machine_analysis import (
    D1AliasedMachineAnalysis,
    D1ArmAliasMap,
    build_d1_aliased_machine_analysis,
)
from entropic_substrate.ladder.register.d1_registered_design import (
    D1RegisteredDesign,
    decision_contract_digest,
)
from entropic_substrate.ladder.review.d1_execution_receipt import (
    D1AnalysisExecutionReceipt,
    D1AnalysisImplementationIdentity,
    D1ExternalIsolationAttestation,
    D1IsolationRequest,
    issue_d1_analysis_receipt,
)

__all__ = [
    "D1AnalysisComposition",
    "D1AnalysisCompositionRequest",
    "D1AnalysisExecution",
    "D1CompositionRunnerPort",
    "D1ExecutedComposition",
    "D1InProcessCompositionRunnerFake",
    "compose_d1_analysis",
    "execute_d1_analysis",
]


@dataclass(frozen=True, slots=True)
class D1AnalysisCompositionRequest:
    root: Path
    design: D1RegisteredDesign
    decision_contract: D1DecisionContract
    aggregate: D1HeterogeneousCaptureAggregate
    aliases: D1ArmAliasMap
    isolation: D1IsolationRequest

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.root), Path):
            raise TypeError("D1 analysis request root must be Path")
        if (
            type(self.design) is not D1RegisteredDesign
            or type(self.decision_contract) is not D1DecisionContract
            or type(self.aggregate) is not D1HeterogeneousCaptureAggregate
            or type(self.aliases) is not D1ArmAliasMap
            or type(self.isolation) is not D1IsolationRequest
        ):
            raise TypeError("D1 analysis request requires nominal evidence values")
        if decision_contract_digest(self.decision_contract) != self.design.decision_contract_digest:
            raise ValueError("D1 analysis request decision contract differs from design")

    @property
    def request_digest(self) -> str:
        return canonical_digest(
            {
                "alias_commitment": self.aliases.sealed_commitment,
                "capture_digest": self.aggregate.manifest_digest,
                "decision_contract_digest": self.design.decision_contract_digest,
                "design_digest": self.design.design_digest,
                "isolation_request_digest": self.isolation.request_digest,
                "schema_version": "d1-analysis-composition-request-v1",
            }
        )


@dataclass(frozen=True, slots=True)
class D1AnalysisComposition:
    admitted: D1AdmittedAnalysisCapture
    factorial: D1FactorialAnalysis | None
    clauses: D1ClauseAnalysis
    diagnostics: tuple[D1ReportRunDiagnostic, ...]
    analysis: D1AliasedMachineAnalysis


@dataclass(frozen=True, slots=True)
class D1ExecutedComposition:
    request_digest: str
    composition: D1AnalysisComposition
    isolation: D1ExternalIsolationAttestation


class D1CompositionRunnerPort(Protocol):
    def run(self, request: D1AnalysisCompositionRequest) -> D1ExecutedComposition: ...


def compose_d1_analysis(request: D1AnalysisCompositionRequest) -> D1AnalysisComposition:
    """Execute all analysis transforms; intended to run behind the isolation port."""
    if type(request) is not D1AnalysisCompositionRequest:
        raise TypeError("D1 composition requires D1AnalysisCompositionRequest")
    admitted = admit_d1_analysis_capture(
        root=request.root,
        design=request.design,
        decision_contract=request.decision_contract,
        aggregate=request.aggregate,
    )
    factorial = (
        compute_d1_factorial(
            design=request.design,
            decision_contract=request.decision_contract,
            cells=admitted.cells,
            exclusion_authority=issue_d1_factorial_exclusion_authority(
                design=request.design,
                decision_contract=request.decision_contract,
                aggregate=request.aggregate,
                admitted=admitted,
            ),
        )
        if admitted.cells
        else None
    )
    clauses = evaluate_d1_clauses(
        policy=parse_d1_clause_policy(request.decision_contract),
        decision_contract=request.decision_contract,
        factorial=factorial,
        cells=admitted.cells,
    )
    diagnostics = build_d1_report_diagnostics(
        admitted=admitted,
        policy=parse_d1_diagnostic_policy(request.decision_contract),
        decision_contract=request.decision_contract,
    )
    analysis = build_d1_aliased_machine_analysis(
        design=request.design,
        decision_contract=request.decision_contract,
        admitted=admitted,
        factorial=factorial,
        clauses=clauses,
        diagnostics=diagnostics,
        aliases=request.aliases,
    )
    return D1AnalysisComposition(admitted, factorial, clauses, diagnostics, analysis)


class D1InProcessCompositionRunnerFake:
    """Contract fake: executes composition in-process and attests non-filing posture."""

    def run(self, request: D1AnalysisCompositionRequest) -> D1ExecutedComposition:
        if type(request) is not D1AnalysisCompositionRequest:
            raise TypeError("D1 composition runner fake requires typed request")
        return D1ExecutedComposition(
            request.request_digest,
            compose_d1_analysis(request),
            D1ExternalIsolationAttestation.test_fake(request.isolation),
        )


@dataclass(frozen=True, slots=True)
class D1AnalysisExecution:
    composition: D1AnalysisComposition
    receipt: D1AnalysisExecutionReceipt


def execute_d1_analysis(
    *,
    request: D1AnalysisCompositionRequest,
    runner: D1CompositionRunnerPort,
    implementation: D1AnalysisImplementationIdentity,
) -> D1AnalysisExecution:
    """Run through an isolation adapter and issue a receipt over its returned evidence."""
    if type(request) is not D1AnalysisCompositionRequest:
        raise TypeError("D1 analysis execution requires typed request")
    if type(implementation) is not D1AnalysisImplementationIdentity:
        raise TypeError("D1 analysis execution requires implementation identity")
    result = runner.run(request)
    if type(result) is not D1ExecutedComposition:
        raise TypeError("D1 composition runner returned an unsupported result")
    if result.request_digest != request.request_digest:
        raise ValueError("D1 composition runner result differs from request")
    composition = result.composition
    receipt = issue_d1_analysis_receipt(
        analysis=composition.analysis,
        admitted=composition.admitted,
        aliases=request.aliases,
        implementation=implementation,
        isolation=result.isolation,
    )
    return D1AnalysisExecution(composition, receipt)
