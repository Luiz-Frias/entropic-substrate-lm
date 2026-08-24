"""Immutable D1 telemetry values enforce registered metric semantics."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from entropic_substrate.evidence.d1_telemetry_values import (
    D1QualityEvidence,
    D1RunStatus,
    D1TailQuarantineEvidence,
    D1TailQuarantineKind,
    D1TelemetryGap,
)
from entropic_substrate.experiments.d1_execution import PAIRWISE_COSINE_SEMANTIC

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
        residual_mean=0.0,
        residual_rms=1.0,
        residual_variance=1.0,
        coverage_50=0.5,
        coverage_80=0.8,
        coverage_95=0.95,
        state_variance=(0.75, 1.25),
        state_covariance_eigenvalues=(0.5, 1.5),
        effective_rank=2.0,
        numerical_rank=2,
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
        ("state_variance", (1.0,), "width"),
        ("state_covariance_eigenvalues", (1.5, 0.5), "ascending"),
        ("effective_rank", 3.0, "latent_width"),
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
