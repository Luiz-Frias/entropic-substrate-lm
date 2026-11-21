"""The single attempt flow owns ordering and exactly one failure dispatch."""

import pytest

from entropic_substrate.kernel.runtime.shared_attempt import AttemptFlow

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "failure", [None, "prepare", "allocate", "start", "collect", "terminate", "bill", "finalize"]
)
def test_attempt_order_and_failure_dispatch(failure):
    observed = []
    phases = ("prepare", "allocate", "start", "collect", "terminate", "bill", "finalize")

    def stage(name):
        def run():
            observed.append(name)
            if failure == name:
                raise ValueError(name)
            return "complete" if name == "finalize" else None

        return run

    def failed(error):
        observed.append("cleanup:" + str(error))
        raise RuntimeError("halted") from error

    flow = AttemptFlow(*(stage(name) for name in phases), on_failure=failed)
    if failure is None:
        assert flow.run() == "complete"
        assert observed == list(phases)
    else:
        with pytest.raises(RuntimeError, match="halted"):
            flow.run()
        assert observed == [*phases[: phases.index(failure) + 1], "cleanup:" + failure]


def test_failure_handler_cannot_return_successfully():
    def fail():
        raise ValueError("original failure")

    def noop():
        return None

    flow = AttemptFlow(fail, noop, noop, noop, noop, noop, noop, lambda error: None)
    with pytest.raises(RuntimeError, match="must not return") as error:
        flow.run()
    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(), GeneratorExit()])
def test_interruptions_reach_cleanup_before_propagating(interruption):
    observed = []

    def fail():
        raise interruption

    def cleanup(error):
        observed.append(error)
        raise error

    def noop():
        return None

    flow = AttemptFlow(fail, noop, noop, noop, noop, noop, noop, cleanup)
    with pytest.raises(type(interruption)) as caught:
        flow.run()
    assert caught.value is interruption and observed == [interruption]
