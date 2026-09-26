"""Canonical run-start methodology values for optional D1 resource activation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from entropic_substrate.kernel.evidence_class.resource_provenance import (
    PowerIntegrationMethod,
    PowerSamplingRegistration,
    ResourceActivationMode,
    ResourceActivationRegistration,
    unavailable_resource_activation_registration,
)

pytestmark = pytest.mark.unit


def _power() -> PowerSamplingRegistration:
    return PowerSamplingRegistration(
        cadence_seconds=1.0,
        max_gap_seconds=2.0,
        integration_method=PowerIntegrationMethod.TRAPEZOIDAL_MONOTONIC_NS,
        provider="nvml",
        device_scope="cuda:0",
    )


def _production() -> ResourceActivationRegistration:
    return ResourceActivationRegistration(
        factory_registration_digest="a" * 64,
        mode=ResourceActivationMode.PRODUCTION,
        meter_provider="torch-cuda+nvml",
        device_scope="cuda:0",
        flop_model_identity="b" * 64,
        power_sampling=_power(),
    )


def test_production_registration_manifest_binds_every_methodology_choice() -> None:
    registration = _production()

    assert registration.manifest == {
        "device_scope": "cuda:0",
        "factory_registration_digest": "a" * 64,
        "flop_model_identity": "b" * 64,
        "meter_provider": "torch-cuda+nvml",
        "mode": "production",
        "power_sampling": {
            "cadence_seconds": 1.0,
            "device_scope": "cuda:0",
            "integration_method": "trapezoidal-monotonic-ns-v1",
            "max_gap_seconds": 2.0,
            "provider": "nvml",
            "schema_version": "resource-power-sampling/v1",
        },
        "schema_version": "resource-activation-registration/v1",
    }
    assert len(registration.identity_digest) == 64


def test_unavailable_registration_is_explicit_and_stable() -> None:
    first = unavailable_resource_activation_registration()
    second = unavailable_resource_activation_registration()

    assert first == second
    assert first.mode is ResourceActivationMode.GAPS_ONLY
    assert first.meter_provider == first.device_scope == "unavailable"
    assert first.flop_model_identity is None
    assert first.power_sampling is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("factory_registration_digest", "not-a-digest", "factory_registration_digest"),
        ("flop_model_identity", "not-a-digest", "flop_model_identity"),
        ("mode", ResourceActivationMode.GAPS_ONLY, "gaps-only"),
        ("meter_provider", "unavailable", "production"),
    ],
)
def test_registration_refuses_incoherent_or_unbound_identity(
    field: str, value: object, message: str
) -> None:
    values: dict[str, object] = {
        "factory_registration_digest": "a" * 64,
        "mode": ResourceActivationMode.PRODUCTION,
        "meter_provider": "torch-cuda+nvml",
        "device_scope": "cuda:0",
        "flop_model_identity": "b" * 64,
        "power_sampling": _power(),
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        ResourceActivationRegistration(**values)  # type: ignore[arg-type]


def test_power_sampling_requires_explicit_method_provider_scope_and_gap_budget() -> None:
    with pytest.raises(ValueError, match="max_gap_seconds"):
        replace(_power(), cadence_seconds=3.0, max_gap_seconds=2.0)
    with pytest.raises(ValueError, match="provider"):
        replace(_power(), provider="")
    with pytest.raises(TypeError, match="integration_method"):
        replace(_power(), integration_method="trapezoid")  # type: ignore[arg-type]
