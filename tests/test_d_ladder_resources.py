"""The ladder consumes existing resource activation and ledger contracts."""

import pytest

from d1_telemetry_fixtures import start_provenance
from entropic_substrate.d1_resource_activation import UnavailableResourceActivationFactory
from entropic_substrate.d_ladder_resources import LadderResources

pytestmark = pytest.mark.unit


def test_resource_session_retains_real_positions_and_explicit_meter_gaps() -> None:
    start = start_provenance()
    with LadderResources(start, UnavailableResourceActivationFactory()) as resources:
        resources.activate(None)
        resources.baseline(32)
        resources.train(online=16, target=18, committed=True)
        snapshot = resources.evaluate(online=16, target=18, step=1)
        assert snapshot["committed_quality_source_bytes"] == 16
        assert snapshot["baseline_calibration_positions"] == 32
        assert resources.state()["online_train_token_positions"] == 16
    with pytest.raises(RuntimeError, match="active"):
        resources.state()
