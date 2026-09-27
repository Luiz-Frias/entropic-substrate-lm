"""D1 delegates its irreversible phase ordering to the shared attempt flow."""

import pytest

from entropic_substrate.kernel.runtime.shared_attempt import AttemptFlow

from .test_d1_operator_controller import _case, _controller

pytestmark = pytest.mark.unit


def test_fresh_controller_reaches_shared_flow_before_allocation(tmp_path, monkeypatch):
    case, events = _case(tmp_path), []
    controller, _ = _controller(tmp_path, case, events)

    def reached(_flow):
        assert not events
        assert callable(_flow.prepare) and callable(_flow.allocate)
        raise RuntimeError("shared flow reached")

    monkeypatch.setattr(AttemptFlow, "run", reached)
    with pytest.raises(RuntimeError, match="shared flow reached"):
        controller.execute_fresh(case.intent)
