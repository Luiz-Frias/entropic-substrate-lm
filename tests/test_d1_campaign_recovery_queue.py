from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from entropic_substrate.d1_campaign_checkpoint import D1CampaignDisposition
from entropic_substrate.d1_campaign_checkpoint_repository import (
    D1CampaignCheckpointReceipt,
)
from entropic_substrate.d1_campaign_queue import D1CampaignQueueHalt
from entropic_substrate.d1_campaign_recovery import (
    D1CampaignRecoveryHalt,
    D1CampaignRecoveryStatus,
    D1HydratedCampaignRecovery,
    hydrate_d1_campaign_recovery,
)
from entropic_substrate.d1_campaign_recovery_queue import D1RecoverableCampaignQueue
from entropic_substrate.d1_operator_controller import D1OperatorRunResult
from entropic_substrate.evidence.d1_lifecycle_values import D1ProcessState
from entropic_substrate.evidence.d1_process_manager import D1ProcessManager
from entropic_substrate.experiments.d1_launch_attempts import (
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)

from .test_d1_campaign_queue import FakeGit, _results
from .test_d1_campaign_recovery import _prefix
from .test_d1_process_manager import _execute

pytestmark = pytest.mark.unit


@dataclass
class _RecoveryController:
    results: tuple[D1OperatorRunResult, ...]
    calls: list[str]
    process_manager: D1ProcessManager
    attempt_ledger: D1LaunchAttemptLedger
    closed: bool = False

    def execute_fresh(self, intent):  # type: ignore[no-untyped-def]
        self.calls.append(intent.plan.logical_run_id)
        result = self.results[len(self.calls) - 1]
        self.process_manager = result.process_manager
        self.attempt_ledger = result.attempt_ledger
        return result

    def close(self) -> None:
        self.closed = True


@dataclass
class _Repository:
    checkpoint: object
    receipt: D1CampaignCheckpointReceipt
    saves: list[tuple[object, str | None]] = field(default_factory=list)

    def load(self, *, expected_sha256: str | None = None):  # type: ignore[no-untyped-def]
        return self.checkpoint, self.receipt

    def save(
        self, checkpoint: object, *, expected_previous_sha256: str | None
    ) -> D1CampaignCheckpointReceipt:
        self.saves.append((checkpoint, expected_previous_sha256))
        self.checkpoint = checkpoint
        self.receipt = D1CampaignCheckpointReceipt(
            self.receipt.path,
            100 + len(self.saves),
            f"{len(self.saves):064x}",
        )
        return self.receipt


def _generation(tmp_path: Path, checkpoint: object) -> tuple[_Repository, object]:
    receipt = D1CampaignCheckpointReceipt(tmp_path / "campaign.json", 100, "9" * 64)
    return _Repository(checkpoint, receipt), receipt


def _ready(tmp_path: Path, count: int):  # type: ignore[no-untyped-def]
    design, intents, starts, manager, _ledger, checkpoint = _prefix(tmp_path, count)
    recovery = hydrate_d1_campaign_recovery(
        design=design,
        intents=intents,
        checkpoint=checkpoint,
        process_manager=manager,
    )
    assert type(recovery) is D1HydratedCampaignRecovery
    all_results = _results(intents, starts)
    suffix: list[D1OperatorRunResult] = []
    for index in range(count, len(intents)):
        manager = _execute(manager, intents[index].plan.logical_run_id, 100 + index * 10)
        suffix.append(replace(all_results[index], process_manager=manager))
    return design, intents, recovery, tuple(suffix)


def _controller(
    recovery: D1HydratedCampaignRecovery | D1CampaignRecoveryHalt,
    results: tuple[D1OperatorRunResult, ...],
) -> _RecoveryController:
    return _RecoveryController(results, [], recovery.process_manager, recovery.attempt_ledger)


def test_recovery_executes_only_suffix_and_cas_saves_each_finalized_cell(
    tmp_path: Path,
) -> None:
    design, intents, recovery, suffix = _ready(tmp_path, 2)
    controller = _controller(recovery, suffix)
    repository, generation = _generation(tmp_path, recovery.checkpoint)

    completed = D1RecoverableCampaignQueue(
        design=design,
        intents=intents,
        controller=controller,
        git=FakeGit(),
        repository=repository,
    ).run(recovery, generation=generation)

    assert controller.calls == [intent.plan.logical_run_id for intent in intents[2:]]
    assert completed.recovered_prefix_count == 2
    assert len(completed.results) == 2
    assert tuple(launch.plan.logical_run_id for launch in completed.matrix.launches) == tuple(
        intent.plan.logical_run_id for intent in intents
    )
    assert [expected for _checkpoint, expected in repository.saves] == [
        "9" * 64,
        "1".zfill(64),
    ]
    assert [item.next_index for item, _expected in repository.saves] == [3, 4]
    assert all(
        item.process_state is D1ProcessState.OPERATOR_FINALIZED for item, _ in repository.saves
    )
    assert completed.process_manager == suffix[-1].process_manager


def test_recovery_preserves_defect_partition_from_checkpoint_prefix(tmp_path: Path) -> None:
    design, intents, recovery, suffix = _ready(tmp_path, 2)
    checkpoint = recovery.checkpoint
    cell = checkpoint.completed_cells[1]
    defect_id = cell.logical_run_id
    events = tuple(
        replace(
            event,
            status=D1LaunchAttemptStatus.FAILED,
            reason="terminal numerical defect",
        )
        if event.identity.logical_run_id == defect_id
        and event.status is D1LaunchAttemptStatus.COMPLETED
        else event
        for event in checkpoint.attempt_ledger.events
    )
    ledger = D1LaunchAttemptLedger(events)
    failed_checkpoint = replace(
        checkpoint,
        completed_cells=(
            checkpoint.completed_cells[0],
            replace(
                cell,
                terminal_outcome="numerical_defect",
                ledger_digest_after=ledger.ledger_digest,
            ),
        ),
        attempt_ledger=ledger,
    )
    failed_recovery = hydrate_d1_campaign_recovery(
        design=design,
        intents=intents,
        checkpoint=failed_checkpoint,
        process_manager=recovery.process_manager,
    )
    assert type(failed_recovery) is D1HydratedCampaignRecovery
    changed_suffix = tuple(
        replace(
            result,
            attempt_ledger=D1LaunchAttemptLedger(
                tuple(
                    replace(
                        event,
                        status=D1LaunchAttemptStatus.FAILED,
                        reason="terminal numerical defect",
                    )
                    if event.identity.logical_run_id == defect_id
                    and event.status is D1LaunchAttemptStatus.COMPLETED
                    else event
                    for event in result.attempt_ledger.events
                )
            ),
        )
        for result in suffix
    )
    controller = _RecoveryController(
        changed_suffix,
        [],
        failed_recovery.process_manager,
        failed_recovery.attempt_ledger,
    )
    repository, generation = _generation(tmp_path, failed_checkpoint)

    completed = D1RecoverableCampaignQueue(
        design=design,
        intents=intents,
        controller=controller,
        git=FakeGit(),
        repository=repository,
    ).run(failed_recovery, generation=generation)

    assert completed.matrix.terminal_defect_run_ids == (defect_id,)
    assert completed.recovered_prefix_count == 2


def test_hard_halt_returns_queue_halt_before_controller_or_save(tmp_path: Path) -> None:
    design, intents, recovery, suffix = _ready(tmp_path, 0)
    halted_checkpoint = replace(
        recovery.checkpoint,
        disposition=D1CampaignDisposition.HARD_HALT,
        halt_reason="provider state requires adjudication",
    )
    halted = hydrate_d1_campaign_recovery(
        design=design,
        intents=intents,
        checkpoint=halted_checkpoint,
        process_manager=recovery.process_manager,
    )
    assert type(halted) is D1CampaignRecoveryHalt
    controller = _controller(halted, suffix)
    repository, generation = _generation(tmp_path, halted_checkpoint)

    with pytest.raises(D1CampaignQueueHalt, match="adjudication"):
        D1RecoverableCampaignQueue(
            design=design,
            intents=intents,
            controller=controller,
            git=FakeGit(),
            repository=repository,
        ).run(halted, generation=generation)

    assert controller.calls == []
    assert repository.saves == []


def test_complete_pre_capture_recovery_attests_matrix_without_controller_calls(
    tmp_path: Path,
) -> None:
    design, intents, recovery, suffix = _ready(tmp_path, 4)
    assert recovery.status is D1CampaignRecoveryStatus.COMPLETE
    controller = _controller(recovery, suffix)
    repository, generation = _generation(tmp_path, recovery.checkpoint)

    completed = D1RecoverableCampaignQueue(
        design=design,
        intents=intents,
        controller=controller,
        git=FakeGit(),
        repository=repository,
    ).run(recovery, generation=generation)

    assert controller.calls == []
    assert repository.saves == []
    assert completed.results == ()
    assert completed.recovered_prefix_count == len(intents)
    assert len(completed.matrix.launches) == len(intents)
    assert completed.process_manager is recovery.process_manager


@pytest.mark.parametrize("defect", ["manager", "ledger", "identity", "start", "inadmissible"])
def test_result_must_extend_hydrated_authorities_before_checkpoint_save(
    tmp_path: Path, defect: str
) -> None:
    design, intents, recovery, suffix = _ready(tmp_path, 1)
    first = suffix[0]
    if defect == "manager":
        first = replace(first, process_manager=recovery.process_manager)
    elif defect == "ledger":
        first = replace(first, attempt_ledger=recovery.attempt_ledger)
    elif defect == "identity":
        prefix = len(recovery.attempt_ledger.events)
        events = first.attempt_ledger.events
        drifted = replace(events[prefix].identity, seed=events[prefix].identity.seed + 1000)
        first = replace(
            first,
            attempt_ledger=D1LaunchAttemptLedger(
                (*events[:prefix], *(replace(event, identity=drifted) for event in events[prefix:]))
            ),
        )
    elif defect == "start":
        first = replace(first, start=replace(first.start, provider_sku="wrong-sku"))
    else:
        first = replace(first, queue_admissible=False)
    controller = _controller(recovery, (first,))
    repository, generation = _generation(tmp_path, recovery.checkpoint)

    with pytest.raises(D1CampaignQueueHalt):
        D1RecoverableCampaignQueue(
            design=design,
            intents=intents,
            controller=controller,
            git=FakeGit(),
            repository=repository,
        ).run(recovery, generation=generation)

    assert controller.calls == [intents[1].plan.logical_run_id]
    assert repository.saves == []


@pytest.mark.parametrize("authority", ["process_manager", "attempt_ledger"])
def test_controller_authority_drift_halts_before_execute_or_save(
    tmp_path: Path, authority: str
) -> None:
    design, intents, recovery, suffix = _ready(tmp_path, 1)
    controller = _controller(recovery, suffix)
    drift = suffix[0].process_manager if authority == "process_manager" else D1LaunchAttemptLedger()
    setattr(controller, authority, drift)
    repository, generation = _generation(tmp_path, recovery.checkpoint)

    with pytest.raises(D1CampaignQueueHalt, match="controller authority"):
        D1RecoverableCampaignQueue(
            design=design,
            intents=intents,
            controller=controller,
            git=FakeGit(),
            repository=repository,
        ).run(recovery, generation=generation)

    assert controller.calls == []
    assert repository.saves == []


def test_generation_must_reload_the_same_hydrated_checkpoint_before_launch(
    tmp_path: Path,
) -> None:
    design, intents, recovery, suffix = _ready(tmp_path, 1)
    controller = _controller(recovery, suffix)
    repository, generation = _generation(tmp_path, replace(recovery.checkpoint, halt_reason=None))
    repository.checkpoint = _ready(tmp_path / "different", 0)[2].checkpoint

    with pytest.raises(D1CampaignQueueHalt, match="generation"):
        D1RecoverableCampaignQueue(
            design=design,
            intents=intents,
            controller=controller,
            git=FakeGit(),
            repository=repository,
        ).run(recovery, generation=generation)

    assert controller.calls == []
    assert repository.saves == []
