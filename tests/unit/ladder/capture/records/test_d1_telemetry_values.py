"""Immutable D1 telemetry values enforce registered metric semantics."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from entropic_substrate.kernel.verdict.d1_evaluation_values import D1StateMomentAccessors
from entropic_substrate.ladder.capture.records.d1_telemetry_values import (
    D1QualityEvidence,
    D1RunStatus,
    D1TailQuarantineEvidence,
    D1TailQuarantineKind,
    D1TelemetryGap,
)
from entropic_substrate.ladder.register.d1_execution import PAIRWISE_COSINE_SEMANTIC
from entropic_substrate.ladder.run.trainer.d1_evaluation import D1ChannelMoments

pytestmark = pytest.mark.unit


def _quality() -> D1QualityEvidence:
    language = 0.8
    future = 0.9
    zero = 1.0
    mean = 0.95
    return D1QualityEvidence(
        language_loss=language,
        nats_per_byte=language,
        bits_per_byte=language / math.log(2.0),
        perplexity=math.exp(language),
        future_nll=future,
        zero_displacement_nll=zero,
        mean_displacement_nll=mean,
        zero_baseline_improvement=zero - future,
        mean_baseline_improvement=mean - future,
        residual_mean=0.015,
        residual_rms=1.0,
        residual_variance=0.999775,
        coverage_50=0.5,
        coverage_80=0.8,
        coverage_95=0.95,
        state_moments=D1ChannelMoments(
            (0.1, -0.2),
            (0.75, 1.25),
            (0.5, 1.5),
            1.7547653506033232,
            2,
            1.2449899597988732,
        ),
        target_moments=D1ChannelMoments(
            (0.02, 0.01), (0.2, 0.3), (0.2, 0.3), 1.9601317042077895, 2, 0.6127805479941412
        ),
        predicted_moments=D1ChannelMoments(
            (0.01, -0.01),
            (0.18, 0.22),
            (0.18, 0.22),
            1.9900083082219462,
            2,
            0.547905101272109,
        ),
        target_predicted_exact_match_count=0,
        mean_pairwise_cosine_distance=0.8,
        pairwise_cosine_semantic=PAIRWISE_COSINE_SEMANTIC,
        raw_variance_penalty=0.01,
        weighted_variance_penalty=0.001,
        raw_covariance_penalty=0.02,
        weighted_covariance_penalty=0.002,
        batch_count=1,
        online_positions=4,
        residual_elements=8,
    )


def test_quality_value_is_immutable_and_exposes_only_registered_scalars() -> None:
    evidence = _quality()
    assert evidence.latent_width == 2
    assert evidence.scalar_metric("nats_per_byte") == 0.8
    assert evidence.scalar_metric("numerical_rank") == 2.0
    with pytest.raises(D1TelemetryGap, match="registered scalar quality metric"):
        evidence.scalar_metric("state_variance")
    with pytest.raises(FrozenInstanceError):
        evidence.nats_per_byte = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("coverage_80", 1.1, "coverage_80"),
        ("coverage_50", 0.9, "coverage"),
        ("pairwise_cosine_semantic", "different", "meaning"),
        ("nats_per_byte", 0.7, "derivation"),
    ],
)
def test_quality_value_refuses_domain_shape_semantic_and_identity_mutations(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(D1TelemetryGap, match=message):
        replace(_quality(), **{field: value})


def test_run_status_vocabulary_is_closed() -> None:
    assert tuple(status.value for status in D1RunStatus) == (
        "complete",
        "partial",
        "trainer_aborted",
        "run_aborted",
    )


def test_no_tail_quarantine_evidence_is_typed_and_immutable() -> None:
    manifest = MappingProxyType({"kind": "no_tail", "schema_version": "d1-tail-quarantine/v1"})
    evidence = D1TailQuarantineEvidence.no_tail(manifest)
    assert evidence.kind is D1TailQuarantineKind.NO_TAIL
    assert evidence.artifact_sha256 is None
    with pytest.raises(TypeError):
        evidence.manifest["kind"] = "quarantined_tail"  # type: ignore[index]


def test_state_accessors_come_from_the_shared_mixin_and_stay_slotted() -> None:
    evidence = _quality()
    moments = evidence.state_moments
    assert isinstance(evidence, D1StateMomentAccessors)
    assert evidence.state_mean == moments.mean
    assert evidence.state_variance == moments.variance
    assert evidence.state_covariance_eigenvalues == moments.covariance_eigenvalues
    assert evidence.effective_rank == moments.effective_rank
    assert evidence.state_variance != evidence.state_mean
    assert not hasattr(evidence, "__dict__")
    for name in ("state_mean", "state_variance", "state_covariance_eigenvalues", "effective_rank"):
        assert name not in vars(D1QualityEvidence)
        assert name in vars(D1StateMomentAccessors)
