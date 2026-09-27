"""TDD contract for all-or-nothing D1 confirmatory matrix admission."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from d1_launch_admission_fixtures import (
    CODE_COMMIT,
    PREREG_BLOB,
    PREREG_COMMIT,
    PREREG_PATH,
    SEEDS,
    STARTED_AT,
    contract,
    family,
    plan,
    profiles,
    start,
)
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.register.d1_confirmatory_matrix import (
    D1ConfirmatoryLaunch,
    D1ConfirmatoryMatrix,
    D1GitProvenancePort,
    admit_d1_confirmatory_matrix,
    attest_d1_confirmatory_matrix,
)
from entropic_substrate.ladder.register.d1_launch_attempts import (
    D1LaunchAttemptLedger,
    D1LaunchAttemptStatus,
)
from entropic_substrate.ladder.register.d1_registered_design import (
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
    decision_contract_digest,
)

pytestmark = pytest.mark.unit


class FakeGit(D1GitProvenancePort):
    def __init__(self) -> None:
        self.commits = {
            PREREG_COMMIT: STARTED_AT - timedelta(days=2),
            CODE_COMMIT: STARTED_AT - timedelta(days=1),
        }
        self.blobs = {(PREREG_COMMIT, PREREG_PATH): PREREG_BLOB}
        self.ancestry = {(PREREG_COMMIT, CODE_COMMIT)}

    def commit_time(self, commit: str):  # type: ignore[no-untyped-def]
        return self.commits.get(commit)

    def blob_sha256(self, commit: str, path: str) -> str | None:
        return self.blobs.get((commit, path))

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        return (ancestor, descendant) in self.ancestry


def design() -> D1RegisteredDesign:
    admitted = admit_d1_registered_design(
        D1RegisteredDesignCandidate(
            family=family(),
            decision_contract=contract(),
            registered_seeds=SEEDS,
            preregistration=D1PreregistrationArtifact(
                path=PREREG_PATH,
                blob_sha256=PREREG_BLOB,
                commit=PREREG_COMMIT,
            ),
        )
    )
    assert type(admitted) is D1RegisteredDesign
    return admitted


def launches() -> tuple[D1ConfirmatoryLaunch, ...]:
    registered = design()
    decision_digest = decision_contract_digest(contract())
    return tuple(
        D1ConfirmatoryLaunch(
            plan=run_plan,
            start=start(run_plan),
            registered_design_digest=registered.design_digest,
            decision_contract_digest=decision_digest,
        )
        for seed in SEEDS
        for profile in profiles()
        for run_plan in (plan(profile, seed),)
    )


def admit(
    cells: tuple[D1ConfirmatoryLaunch, ...] | None = None,
    *,
    git: FakeGit | None = None,
    ledger: D1LaunchAttemptLedger | None = None,
):  # type: ignore[no-untyped-def]
    return admit_d1_confirmatory_matrix(
        design=design(),
        launches=launches() if cells is None else cells,
        git=FakeGit() if git is None else git,
        ledger=D1LaunchAttemptLedger() if ledger is None else ledger,
    )


def test_complete_registered_four_by_seed_matrix_is_admitted_once() -> None:
    admission = admit()
    assert type(admission.matrix) is D1ConfirmatoryMatrix
    assert len(admission.matrix.launches) == len(D1Arm) * len(SEEDS)
    assert admission.matrix.registered_seeds == SEEDS
    assert len(admission.ledger.events) == 2 * len(admission.matrix.launches)
    assert {event.status.value for event in admission.ledger.events} == {
        "recorded",
        "trainer_started",
    }


def test_missing_duplicate_and_unregistered_seed_cells_are_refused() -> None:
    cells = launches()
    for bad, message in (
        (cells[:-1], "missing"),
        ((*cells, cells[0]), "duplicate"),
        (
            (replace(cells[0], plan=plan(profiles()[0], 999)), *cells[1:]),
            "unregistered seed",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            admit(bad)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("registered_design_digest", "0" * 64, "registered-design"),
        ("decision_contract_digest", "0" * 64, "decision-contract"),
    ],
)
def test_every_cell_binds_exact_design_and_decision_contract(
    field: str, replacement: str, message: str
) -> None:
    cells = list(launches())
    cells[0] = replace(cells[0], **{field: replacement})
    with pytest.raises(ValueError, match=message):
        admit(tuple(cells))


def test_paired_seed_refuses_data_code_image_pairing_or_resource_drift() -> None:
    cells = list(launches())
    victim = cells[1]
    profile = victim.plan.profile
    seed = victim.plan.seed

    changed_data = plan(
        profile,
        seed,
        data={
            "dataset_name": "enwik8",
            "dataset_digest": "9" * 64,
            "split_digest": "2" * 64,
            "sampler_construction_digest": "3" * 64,
        },
    )
    variants = (
        replace(victim, plan=changed_data, start=start(changed_data)),
        replace(victim, start=replace(victim.start, provider_sku="different-sku")),
        replace(
            victim,
            start=replace(
                victim.start,
                resource_activation=replace(
                    victim.start.resource_activation,
                    factory_registration_digest="8" * 64,
                ),
            ),
        ),
    )
    for changed in variants:
        mutated = list(cells)
        mutated[1] = changed
        with pytest.raises(ValueError, match="paired seed"):
            admit(tuple(mutated))

    wrong_image = replace(victim.start, image_digest="sha256:" + "9" * 64)
    mutated = list(cells)
    mutated[1] = replace(victim, start=wrong_image)
    with pytest.raises(ValueError, match="image"):
        admit(tuple(mutated))


def test_preregistration_blob_commit_ancestry_and_time_precedence_are_proved_by_port() -> None:
    cases: list[tuple[FakeGit, str]] = []
    missing = FakeGit()
    missing.commits.pop(PREREG_COMMIT)
    cases.append((missing, "preregistration commit"))
    blob = FakeGit()
    blob.blobs[(PREREG_COMMIT, PREREG_PATH)] = "0" * 64
    cases.append((blob, "blob"))
    ancestry = FakeGit()
    ancestry.ancestry.clear()
    cases.append((ancestry, "ancestor"))
    late = FakeGit()
    late.commits[PREREG_COMMIT] = STARTED_AT + timedelta(seconds=1)
    cases.append((late, "trainer_start"))
    late_code = FakeGit()
    late_code.commits[CODE_COMMIT] = STARTED_AT + timedelta(seconds=1)
    cases.append((late_code, "code commit"))

    for fake, message in cases:
        with pytest.raises(ValueError, match=message):
            admit(git=fake)


def test_second_matrix_with_same_optional_stopping_identities_is_not_a_rerun() -> None:
    first = admit()
    with pytest.raises(ValueError, match="optional-stopping"):
        admit(ledger=first.ledger)


def test_post_wave_attestation_consumes_completed_controller_ledger_without_new_events() -> None:
    launched = launches()
    started = admit(launched)
    completed = started.ledger
    for cell in launched:
        identity = next(
            event.identity
            for event in completed.events
            if event.identity.logical_run_id == cell.plan.logical_run_id
        )
        completed = completed.transition(
            identity,
            status=D1LaunchAttemptStatus.COMPLETED,
            observed_at=cell.start.trainer_started_at + timedelta(seconds=1),
        )

    matrix = attest_d1_confirmatory_matrix(
        design=design(),
        launches=launched,
        git=FakeGit(),
        ledger=completed,
    )

    assert matrix.manifest == started.matrix.manifest
    assert len(completed.events) == 3 * len(launched)


def test_post_wave_attestation_refuses_incomplete_or_extra_attempt_history() -> None:
    launched = launches()
    started = admit(launched)
    with pytest.raises(ValueError, match="completed"):
        attest_d1_confirmatory_matrix(
            design=design(), launches=launched, git=FakeGit(), ledger=started.ledger
        )
