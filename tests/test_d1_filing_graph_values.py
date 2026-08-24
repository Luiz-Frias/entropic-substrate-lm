"""Factory-only filing authority and recomputed capture-integrity values."""

from __future__ import annotations

from pathlib import Path

import pytest

from entropic_substrate.evidence.d1_capture_validation import build_d1_capture_bundle
from entropic_substrate.evidence.d1_filing_graph import (
    ValidatedD1Filing,
    recompute_capture_integrity,
)

from .d1_capture_fixtures import capture_family

pytestmark = pytest.mark.unit


def test_capture_integrity_is_recomputed_from_factory_attested_bundle(tmp_path: Path) -> None:
    roster, inputs, root = capture_family(tmp_path)
    bundle = build_d1_capture_bundle(root, roster=roster, capture_inputs=inputs)

    integrity = recompute_capture_integrity(bundle)

    assert integrity.roster_complete
    assert integrity.pairing_verified
    assert integrity.data_order_verified
    assert integrity.completion_verified
    assert len(integrity.evidence_digest) == 64


def test_gate_authority_value_is_factory_only() -> None:
    with pytest.raises(TypeError, match="factory-only"):
        ValidatedD1Filing()  # type: ignore[call-arg]
