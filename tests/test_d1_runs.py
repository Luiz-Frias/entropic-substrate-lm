"""Feature-owned tests for D1 execution lifecycle and evidence identities."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.experiments.d1_family import (
    D1ExperimentFamily,
    build_d1_experiment_family,
)
from entropic_substrate.experiments.d1_primitives import (
    CausalEncoderSpec,
    D1Arm,
    EncoderInitializationIdentity,
    EvidenceClass,
    MatchedEncoderPairSpec,
    RunClass,
)
from entropic_substrate.experiments.d1_provenance import (
    CodeIdentity,
    ContainerIdentity,
    DataIdentity,
    EvidenceEnvelope,
    ExecutionIdentity,
    PreregistrationIdentity,
)
from entropic_substrate.observation import CodecIdentity

pytestmark = pytest.mark.unit


def _family() -> D1ExperimentFamily:
    codec = CodecIdentity(
        name="identity-byte",
        construction_digest="a" * 64,
        cardinality=256,
    )
    initialization = EncoderInitializationIdentity(
        name="d1-causal-random-v1",
        construction_digest="5" * 64,
    )
    encoder = CausalEncoderSpec(
        latent_width=64,
        n_layers=2,
        n_heads=4,
        mlp_ratio=4.0,
        block_size=256,
        dropout=0.0,
        bias=False,
        rms_epsilon=1e-5,
        initialization=initialization,
    )
    return build_d1_experiment_family(
        codec=codec,
        encoder_pair=MatchedEncoderPairSpec(online=encoder, target=encoder),
    )


def _evidence() -> EvidenceEnvelope:
    return EvidenceEnvelope(
        schema_version="d1-evidence-v1",
        required_classes=frozenset(
            {
                EvidenceClass.MEASURED,
                EvidenceClass.DERIVED_FROM_MEASURED,
            }
        ),
    )


def _data() -> DataIdentity:
    return DataIdentity(
        dataset_name="enwik8",
        dataset_digest="1" * 64,
        split_digest="2" * 64,
        sampler_construction_digest="3" * 64,
    )


def _confirmatory_execution(seed: int = 11) -> ExecutionIdentity:
    return ExecutionIdentity(
        run_class=RunClass.CONFIRMATORY,
        seed=seed,
        pairing_id="d1-stabilization-pair-a",
        code=CodeIdentity(commit="b" * 40, repository_digest="c" * 64),
        container=ContainerIdentity(image_digest="sha256:" + "d" * 64),
    )


def _preregistration(value: D1ExperimentFamily) -> PreregistrationIdentity:
    return PreregistrationIdentity(
        path="docs/d1-preregistration.md",
        commit="e" * 40,
        family_digest=value.family_digest,
    )


def test_draft_transitions_to_excluded_pilot_and_registered_confirmatory() -> None:
    value = _family()
    draft = value.draft_for(D1Arm.BOTH)
    pilot = draft.validate_pilot(
        execution=ExecutionIdentity(
            run_class=RunClass.PILOT,
            seed=3,
            pairing_id="pilot-a",
        ),
        evidence=_evidence(),
    )
    assert pilot.run_class is RunClass.PILOT
    assert pilot.preregistration is None
    assert pilot.execution_digest != pilot.construction_digest

    preregistration = _preregistration(value)
    confirmatory = value.validate_confirmatory_arm(
        arm=D1Arm.BOTH,
        execution=_confirmatory_execution(),
        data=_data(),
        evidence=_evidence(),
        preregistration=preregistration,
    )
    assert confirmatory.run_class is RunClass.CONFIRMATORY
    assert confirmatory.preregistration == preregistration
    assert confirmatory.execution_manifest["preregistration"] == preregistration.manifest


def test_confirmatory_rejects_missing_or_mismatched_provenance() -> None:
    value = _family()
    bad_digest = PreregistrationIdentity(
        path="docs/d1-preregistration.md",
        commit="e" * 40,
        family_digest="f" * 64,
    )
    with pytest.raises(ValueError, match="does not match registered family digest"):
        value.validate_confirmatory_arm(
            arm=D1Arm.SG,
            execution=_confirmatory_execution(),
            data=_data(),
            evidence=_evidence(),
            preregistration=bad_digest,
        )
    with pytest.raises(ValueError, match="code identity"):
        value.validate_confirmatory_arm(
            arm=D1Arm.SG,
            execution=ExecutionIdentity(
                run_class=RunClass.CONFIRMATORY,
                seed=11,
                pairing_id="d1-stabilization-pair-a",
                container=ContainerIdentity(image_digest="sha256:" + "d" * 64),
            ),
            data=_data(),
            evidence=_evidence(),
            preregistration=_preregistration(value),
        )


def test_canonical_digests_are_deterministic_seed_scoped_and_sensitive() -> None:
    value = _family()
    draft = value.draft_for(D1Arm.BOTH)
    preregistration = _preregistration(value)
    first = value.validate_confirmatory_arm(
        arm=D1Arm.BOTH,
        execution=_confirmatory_execution(seed=11),
        data=_data(),
        evidence=_evidence(),
        preregistration=preregistration,
    )
    repeated = value.validate_confirmatory_arm(
        arm=D1Arm.BOTH,
        execution=_confirmatory_execution(seed=11),
        data=_data(),
        evidence=_evidence(),
        preregistration=preregistration,
    )
    other_seed = value.validate_confirmatory_arm(
        arm=D1Arm.BOTH,
        execution=_confirmatory_execution(seed=12),
        data=_data(),
        evidence=_evidence(),
        preregistration=preregistration,
    )
    changed_codec = replace(
        draft,
        codec=CodecIdentity(
            name="identity-byte-revision",
            construction_digest="f" * 64,
            cardinality=256,
        ),
    )

    assert first.construction_manifest == repeated.construction_manifest
    assert first.construction_digest == repeated.construction_digest
    assert first.execution_digest == repeated.execution_digest
    assert first.aggregation_digest == other_seed.aggregation_digest
    assert first.execution_digest != other_seed.execution_digest
    assert first.construction_digest != changed_codec.construction_digest
    assert list(first.construction_manifest) == sorted(first.construction_manifest)
    assert first.execution_manifest["execution"]["seed"] == 11
    assert first.execution_manifest["execution"]["run_class"] == "confirmatory"
