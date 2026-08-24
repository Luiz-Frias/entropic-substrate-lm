"""Feature-owned tests for D1 preflight and capture provenance boundaries."""

from __future__ import annotations

import pytest

from entropic_substrate.experiments.d1_provenance import DataIdentity, DataOrderAttestation

pytestmark = pytest.mark.unit


def test_data_order_is_runtime_attestation_not_preflight_identity() -> None:
    preflight_data = DataIdentity(
        dataset_name="enwik8",
        dataset_digest="1" * 64,
        split_digest="2" * 64,
        sampler_construction_digest="3" * 64,
    )
    observed_order = DataOrderAttestation(data_order_digest="4" * 64)

    assert "data_order_digest" not in preflight_data.manifest
    assert observed_order.manifest == {"data_order_digest": "4" * 64}
