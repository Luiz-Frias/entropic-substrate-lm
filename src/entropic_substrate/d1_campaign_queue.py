"""Strictly sequential execution queue for one registered D1 campaign."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Protocol, cast

from entropic_substrate.d1_operator_controller import (
    D1OperatorQueueHalt,
    D1OperatorRunResult,
)
from entropic_substrate.d1_operator_values import D1OperatorLaunchIntent
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.experiments.d1_confirmatory_matrix import (
    D1ConfirmatoryLaunch,
    D1ConfirmatoryMatrix,
    D1GitProvenancePort,
    attest_d1_confirmatory_matrix,
)
from entropic_substrate.experiments.d1_launch_attempts import D1LaunchAttemptLedger
from entropic_substrate.experiments.d1_plan import ConfirmatoryD1RunPlan
from entropic_substrate.experiments.d1_primitives import D1Arm
from entropic_substrate.experiments.d1_registered_design import D1RegisteredDesign

_CAMPAIGN_TOKEN: Final = object()


class D1FreshAttemptControllerPort(Protocol):
    def execute_fresh(self, intent: D1OperatorLaunchIntent) -> D1OperatorRunResult: ...


class D1CampaignQueueHalt(RuntimeError):
    """One cell stopped the campaign; later cells were never invoked."""

    def __init__(
        self,
        logical_run_id: str,
        completed: tuple[D1OperatorRunResult, ...],
        detail: str,
    ) -> None:
        self.logical_run_id = logical_run_id
        self.completed = completed
        super().__init__(f"D1 campaign halted at {logical_run_id}: {detail}")


@dataclass(frozen=True, slots=True)
class D1CompletedCampaign:
    """Factory-attested complete wave, ready for evidence assembly and capture."""

    design_digest: str
    results: tuple[D1OperatorRunResult, ...]
    matrix: D1ConfirmatoryMatrix
    attempt_ledger: D1LaunchAttemptLedger
    process_manager: D1ProcessManager
    recovered_prefix_count: int
    _token: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _CAMPAIGN_TOKEN:
            raise TypeError("D1CompletedCampaign is factory-only")
        if any(not item.queue_admissible for item in self.results):
            raise ValueError("completed D1 campaign requires only queue-admissible results")
        if self.results and self.attempt_ledger != self.results[-1].attempt_ledger:
            raise ValueError("completed D1 campaign ledger differs from its final result")
        if self.results and self.process_manager != self.results[-1].process_manager:
            raise ValueError("completed D1 campaign process differs from its final result")
        if (
            type(self.recovered_prefix_count) is not int
            or self.recovered_prefix_count < 0
            or self.recovered_prefix_count + len(self.results) != len(self.matrix.launches)
        ):
            raise ValueError("completed D1 campaign recovered/executed partition is inconsistent")

    @classmethod
    def from_attested(
        cls,
        *,
        design_digest: str,
        results: tuple[D1OperatorRunResult, ...],
        matrix: D1ConfirmatoryMatrix,
        attempt_ledger: D1LaunchAttemptLedger,
        process_manager: D1ProcessManager,
        recovered_prefix_count: int,
    ) -> D1CompletedCampaign:
        return cls(
            design_digest,
            results,
            matrix,
            attempt_ledger,
            process_manager,
            recovered_prefix_count,
            _token=_CAMPAIGN_TOKEN,
        )


def _validate_roster(
    design: D1RegisteredDesign,
    intents: tuple[D1OperatorLaunchIntent, ...],
) -> None:
    if type(intents) is not tuple:
        raise TypeError("D1 campaign intents must be an immutable tuple")
    expected = tuple((arm, seed) for seed in design.registered_seeds for arm in D1Arm)
    actual = tuple((intent.plan.profile.arm, intent.plan.seed) for intent in intents)
    if actual != expected:
        raise ValueError("D1 intents differ from the exact registered campaign roster/order")
    if any(
        intent.registered_design_digest != design.design_digest
        or intent.execution_segment_index != 0
        for intent in intents
    ):
        raise ValueError("D1 intents differ from the exact registered campaign roster/order")


class D1CampaignQueue:
    """Invoke one complete attempt at a time and attest the post-wave matrix."""

    def __init__(
        self,
        *,
        design: D1RegisteredDesign,
        controller: D1FreshAttemptControllerPort,
        git: D1GitProvenancePort,
    ) -> None:
        if type(design) is not D1RegisteredDesign:
            raise TypeError("D1 campaign queue requires a registered design")
        self._design, self._controller, self._git = design, controller, git

    def run(self, intents: tuple[D1OperatorLaunchIntent, ...]) -> D1CompletedCampaign:
        _validate_roster(self._design, intents)
        results: list[D1OperatorRunResult] = []
        launches: list[D1ConfirmatoryLaunch] = []
        prior = D1LaunchAttemptLedger()
        for intent in intents:
            subject = intent.plan.logical_run_id
            try:
                result = self._controller.execute_fresh(intent)
            except (D1OperatorQueueHalt, Exception) as error:
                raise D1CampaignQueueHalt(subject, tuple(results), str(error)) from error
            if type(result) is not D1OperatorRunResult:
                raise TypeError("D1 campaign controller returned a non-nominal result")
            if result.start.logical_run_id != subject or result.start.execution_segment_index != 0:
                raise D1CampaignQueueHalt(
                    subject, tuple(results), "controller result differs from the queued cell"
                )
            prefix = result.attempt_ledger.events[: len(prior.events)]
            if prefix != prior.events:
                raise D1CampaignQueueHalt(
                    subject, tuple(results), "attempt ledger is not an append-only prefix"
                )
            results.append(result)
            prior = result.attempt_ledger
            if not result.queue_admissible:
                raise D1CampaignQueueHalt(
                    subject, tuple(results), "cell returned an evidence gap or trainer failure"
                )
            launches.append(
                D1ConfirmatoryLaunch(
                    cast(ConfirmatoryD1RunPlan, intent.plan),
                    result.start,
                    self._design.design_digest,
                    self._design.decision_contract_digest,
                )
            )
        matrix = attest_d1_confirmatory_matrix(
            design=self._design,
            launches=tuple(launches),
            git=self._git,
            ledger=prior,
        )
        return D1CompletedCampaign.from_attested(
            design_digest=self._design.design_digest,
            results=tuple(results),
            matrix=matrix,
            attempt_ledger=prior,
            process_manager=results[-1].process_manager,
            recovered_prefix_count=0,
        )
