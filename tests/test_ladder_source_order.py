"""Checkpoint source-order evidence preserves observations across process boundaries."""

import pytest

from entropic_substrate.data.ladder_source_order import SourceOrderAudit

pytestmark = pytest.mark.unit


def test_audit_restores_actual_prefix_and_continues_without_replaying_sampler() -> None:
    uninterrupted = SourceOrderAudit(2)
    first = SourceOrderAudit(2)
    for index in range(40):
        uninterrupted.record((index, index + 2))
        if index < 7:
            first.record((index, index + 2))
    resumed = SourceOrderAudit.restore(first.manifest, step=7, batch_size=2)
    for index in range(7, 40):
        resumed.record((index, index + 2))
    assert resumed.attestation == uninterrupted.attestation
    assert resumed.attestation[1] == 32
    assert len(resumed.manifest["offsets"]) == 32


def test_missing_truncated_and_invalid_offsets_are_refused() -> None:
    audit = SourceOrderAudit(2)
    assert audit.attestation is None
    for invalid in ((True, 2), (-1, 2), (1,), (1, 2.0)):
        with pytest.raises(ValueError):
            audit.record(invalid)
    audit.record((1, 2))
    with pytest.raises(ValueError):
        SourceOrderAudit.restore(audit.manifest, step=2, batch_size=2)
    damaged = {**audit.manifest, "sha256": "0" * 64}
    with pytest.raises(ValueError):
        SourceOrderAudit.restore(damaged, step=1, batch_size=2)
