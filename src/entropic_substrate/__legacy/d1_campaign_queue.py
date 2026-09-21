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
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptIdentity,
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.experiments.d1_plan import ConfirmatoryD1RunPlan
from entropic_substrate.experiments.d1_primitives import D1Arm, RunClass, canonical_digest
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
        if any(not item.campaign_continuable for item in self.results):
            raise ValueError("completed D1 campaign contains a non-continuable result")
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
    pairing_ids: set[str] = set()
    for seed in design.registered_seeds:
        block = tuple(intent for intent in intents if intent.plan.seed == seed)
        pairings = {intent.plan.run.execution.pairing_id for intent in block}
        provenance = {
            canonical_digest(
                {
                    "code": (
                        None
                        if intent.plan.run.execution.code is None
                        else intent.plan.run.execution.code.manifest
                    ),
                    "container": (
                        None
                        if intent.plan.run.execution.container is None
                        else intent.plan.run.execution.container.manifest
                    ),
                    "data": None if intent.plan.run.data is None else intent.plan.run.data.manifest,
                    "image": {
                        "digest": intent.image.digest,
                        "platform": intent.image.platform,
                        "repository": intent.image.repository,
                        "source_commit": intent.image.source_commit,
                    },
                }
            )
            for intent in block
        }
        if len(pairings) != 1 or pairing_ids & pairings or len(provenance) != 1:
            raise ValueError("D1 campaign paired provenance differs before provider launch")
        pairing_ids.update(pairings)
        for intent in block:
            plan = intent.plan
            preregistration = plan.run.preregistration
            if (
                plan.profile.family.family_digest != design.family_digest
                or plan.run.construction_digest
                != design.arm_construction_digests[plan.profile.arm.value]
                or preregistration is None
                or (
                    preregistration.path,
                    preregistration.commit,
                    preregistration.family_digest,
                )
                != (
                    design.preregistration.path,
                    design.preregistration.commit,
                    design.family_digest,
                )
            ):
                raise ValueError("D1 campaign registered provenance differs before provider launch")


def validate_d1_campaign_ledger_extension(
    result: D1OperatorRunResult,
    intent: D1OperatorLaunchIntent,
    prior: D1LaunchAttemptLedger,
) -> None:
    if (
        type(result) is not D1OperatorRunResult
        or type(intent) is not D1OperatorLaunchIntent
        or type(prior) is not D1LaunchAttemptLedger
    ):
        raise TypeError("D1 campaign ledger validation requires nominal values")
    terminal_status = (
        D1LaunchAttemptStatus.FAILED
        if result.terminal_numerical_defect is not None
        else D1LaunchAttemptStatus.COMPLETED
    )
    prior_events = prior.events
    extension = result.attempt_ledger.events[len(prior_events) :]
    expected_identity = D1LaunchAttemptIdentity(
        RunClass.CONFIRMATORY,
        intent.plan.profile.family.family_digest,
        intent.plan.profile.arm,
        intent.plan.seed,
        intent.plan.logical_run_id,
        intent.registered_design_digest,
    )
    if (
        result.attempt_ledger.events[: len(prior_events)] != prior_events
        or tuple(event.status for event in extension)
        != (
            D1LaunchAttemptStatus.RECORDED,
            D1LaunchAttemptStatus.TRAINER_STARTED,
            terminal_status,
        )
        or any(event.identity != expected_identity for event in extension)
    ):
        raise ValueError("controller attempt ledger does not extend campaign authority")


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
            try:
                validate_d1_campaign_ledger_extension(result, intent, prior)
            except (TypeError, ValueError) as error:
                raise D1CampaignQueueHalt(subject, tuple(results), str(error)) from error
            results.append(result)
            prior = result.attempt_ledger
            if not result.campaign_continuable:
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
            terminal_defect_run_ids=tuple(
                result.start.logical_run_id
                for result in results
                if result.terminal_numerical_defect is not None
            ),
        )
        return D1CompletedCampaign.from_attested(
            design_digest=self._design.design_digest,
            results=tuple(results),
            matrix=matrix,
            attempt_ledger=prior,
            process_manager=results[-1].process_manager,
            recovered_prefix_count=0,
        )
