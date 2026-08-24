"""TDD contract for the canonical D1 registered-design value."""

from __future__ import annotations

from dataclasses import replace

import pytest

from d1_launch_admission_fixtures import (
    PREREG_BLOB,
    PREREG_COMMIT,
    PREREG_PATH,
    SEEDS,
    contract,
    family,
)
from entropic_substrate.experiments.d1_registered_design import (
    D1AdmissionBlocked,
    D1AdmissionField,
    D1AdmissionGap,
    D1PreregistrationArtifact,
    D1RegisteredDesign,
    D1RegisteredDesignCandidate,
    admit_d1_registered_design,
    decision_contract_digest,
)

pytestmark = pytest.mark.unit


def preregistration() -> D1PreregistrationArtifact:
    return D1PreregistrationArtifact(
        path=PREREG_PATH,
        blob_sha256=PREREG_BLOB,
        commit=PREREG_COMMIT,
    )


def candidate(**overrides: object) -> D1RegisteredDesignCandidate:
    payload: dict[str, object] = {
        "family": family(),
        "decision_contract": contract(),
        "registered_seeds": SEEDS,
        "preregistration": preregistration(),
    }
    payload.update(overrides)
    return D1RegisteredDesignCandidate(**payload)  # type: ignore[arg-type]


def test_registered_design_is_factory_attested_and_canonical() -> None:
    admitted = admit_d1_registered_design(candidate())
    assert type(admitted) is D1RegisteredDesign
    assert admitted.registered_seeds == SEEDS
    assert admitted.family_digest == family().family_digest
    assert admitted.decision_contract_digest == decision_contract_digest(contract())
    assert admitted.manifest["preregistration"] == preregistration().manifest
    assert len(admitted.design_digest) == 64
    with pytest.raises(TypeError, match="factory-only"):
        D1RegisteredDesign(  # type: ignore[call-arg]
            family_digest=admitted.family_digest,
            arm_construction_digests=admitted.arm_construction_digests,
            decision_contract_digest=admitted.decision_contract_digest,
            registered_seeds=SEEDS,
            preregistration=preregistration(),
        )


def test_unresolved_scientific_values_return_typed_gaps_not_defaults() -> None:
    seed_gap = D1AdmissionGap(
        field=D1AdmissionField.REGISTERED_SEEDS,
        reason="seed roster awaits excluded-pilot variance",
    )
    contract_gap = D1AdmissionGap(
        field=D1AdmissionField.DECISION_CONTRACT,
        reason="margins and multiplicity await a scientific ruling",
    )
    blocked = admit_d1_registered_design(
        candidate(registered_seeds=seed_gap, decision_contract=contract_gap)
    )
    assert type(blocked) is D1AdmissionBlocked
    assert blocked.gaps == (contract_gap, seed_gap)
    assert "default" not in repr(blocked)


@pytest.mark.parametrize("seeds", [(), (1, 1), (True,), (1, -1)])
def test_registered_seed_roster_must_be_explicit_unique_nonnegative_integers(
    seeds: tuple[object, ...],
) -> None:
    with pytest.raises((TypeError, ValueError), match="registered seed"):
        admit_d1_registered_design(candidate(registered_seeds=seeds))


def test_contract_must_bind_the_exact_family_and_seed_roster() -> None:
    wrong_family = replace(
        contract(),
        family_identity=replace(contract().family_identity, family_digest="0" * 64),
        registered_constants={"seeds": list(SEEDS), "primary": {"alpha": 0.025}},
    )
    with pytest.raises(ValueError, match="family"):
        admit_d1_registered_design(candidate(decision_contract=wrong_family))

    with pytest.raises(ValueError, match="seed"):
        admit_d1_registered_design(candidate(registered_seeds=(1, 2, 3, 4)))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"path": "/absolute/prereg.md"}, "repository-relative"),
        ({"path": "docs/../prereg.md"}, "parent traversal"),
        ({"blob_sha256": "A" * 64}, "lowercase"),
        ({"commit": "A" * 40}, "lowercase"),
    ],
)
def test_preregistration_artifact_identity_is_canonical(
    overrides: dict[str, str], message: str
) -> None:
    values = {
        "path": PREREG_PATH,
        "blob_sha256": PREREG_BLOB,
        "commit": PREREG_COMMIT,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        D1PreregistrationArtifact(**values)


def test_decision_contract_digest_changes_when_a_registered_constant_changes() -> None:
    baseline = contract()
    changed = replace(
        baseline,
        registered_constants={"seeds": list(SEEDS), "primary": {"alpha": 0.01}},
    )
    assert decision_contract_digest(changed) != decision_contract_digest(baseline)
