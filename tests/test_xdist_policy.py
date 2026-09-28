"""FOR-1618 P4b: the torch thread policy under xdist, both arms plus guard teeth."""

from __future__ import annotations

import os

import pytest
import torch

from xdist_policy import (
    THREADS_PER_XDIST_WORKER,
    THREADS_SERIAL,
    XDIST_WORKER_ENV,
    torch_threads_for,
)

pytestmark = pytest.mark.unit


def test_serial_session_keeps_two_threads() -> None:
    assert torch_threads_for({}) == THREADS_SERIAL == 2


def test_unrelated_env_is_still_serial() -> None:
    assert torch_threads_for({"PYTEST_CURRENT_TEST": "x", "OMP_NUM_THREADS": "8"}) == 2


def test_xdist_worker_gets_one_thread() -> None:
    assert torch_threads_for({XDIST_WORKER_ENV: "gw0"}) == THREADS_PER_XDIST_WORKER == 1


def test_presence_not_value_decides() -> None:
    # xdist always sets a non-empty id, but the policy must not depend on it:
    # an empty string is still a worker that set the variable.
    assert torch_threads_for({XDIST_WORKER_ENV: ""}) == 1


def test_xdist_worker_env_name_matches_xdist() -> None:
    # Pin the variable name xdist actually exports (xdist/remote.py setup_config);
    # a rename upstream must fail here, not silently fall back to 2 threads.
    assert XDIST_WORKER_ENV == "PYTEST_XDIST_WORKER"


def test_conftest_applies_the_policy_to_this_process() -> None:
    """Guard teeth: RED under `-n 2` while conftest hardcodes set_num_threads(2).

    Under `-n 0` both sides are 2 and this cannot distinguish anything; the
    load-bearing arm is the xdist one, which is the regime CI runs in.
    """
    assert torch.get_num_threads() == torch_threads_for(os.environ)
