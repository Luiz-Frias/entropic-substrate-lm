"""Characterization of every H run-class gate, captured from the code before the port.

Rung A (operator ruling 2026-10-27) consolidates the scattered class gates into one
admission port without changing a single verdict. This table was written against
the pre-port code at e73d1230 and is the parity oracle for every refactor that
routes a site through ``ladder/run/admission.py``: a site probe reaches the gate
itself (profiles are built without validation, so classes and stages the profile
validator would refuse upstream still reach it), and stubs the first call after
the gate to raise ``_Passed`` so an admitted case runs no training and touches no
disk. A refusal is recorded as its verbatim message.
"""

from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest
from tests.test_h_probe_launch import launch, remote_request
from tests.test_sentence_worker_request import request

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.ladder.run import (
    h_probe_launch,
    runner,
    worker_command,
    worker_completion,
    worker_execution,
)
from entropic_substrate.ladder.run.worker_request import ProbeRequest

pytestmark = pytest.mark.unit

CLASSES = ("local_tiny", "remote_tiny", "pilot", "confirmatory", "unknown_class")
STAGES = ("H0", "H1", "H2", "H3", "H4", "H5", "H9")

ADMITTED = "admitted"
NO_EXECUTION = "probe worker has no scientific execution authority"
NO_SUCCESSOR = "remote successor requires its scientific gate adapter"
NO_RECEIPT = "probe receipt has no scientific authority"
NO_DRIVER = "probe driver lacks scientific or successor authority"
NO_SCOPE = "worker command scope differs from the request"
NO_LAUNCH = "deployment is restricted to remote H0 tiny probes"
NO_GUARD = "H0 deployment requires the declared tiny-only watchdog exception"
LOCAL = "local_branch"
REMOTE = "remote_branch"


class _Passed(Exception):
    """Raised by a stub placed immediately after a gate: the gate admitted."""


def _raise_passed(*_: object, **__: object) -> Any:
    raise _Passed


def _outcome(call: Callable[[], object]) -> str:
    try:
        call()
    except _Passed:
        return ADMITTED
    except ValueError as error:
        return str(error)
    return ADMITTED


def _gate(refusal: str, call: Callable[[], object]) -> str:
    """The gate's verdict when a later check may also raise: only ``refusal`` refuses."""
    return refusal if _outcome(call) == refusal else ADMITTED


def _unvalidated(template: Any, **changes: object) -> Any:
    """A frozen slots dataclass copy that skips ``__post_init__``."""
    value = object.__new__(type(template))
    for field in fields(template):
        object.__setattr__(value, field.name, changes.get(field.name, getattr(template, field.name)))
    return value


def _profile(run_class: str, stage: str) -> HProfile:
    return _unvalidated(remote_request().profile, run_class=run_class, stage=stage)


def _request(run_class: str, stage: str) -> ProbeRequest:
    base = request() if run_class == "local_tiny" else remote_request()
    return _unvalidated(base, profile=_profile(run_class, stage))


def _expected_request(run_class: str, stage: str) -> str:
    if run_class not in ("local_tiny", "remote_tiny"):
        return NO_EXECUTION
    if run_class == "remote_tiny" and stage != "H0":
        return NO_SUCCESSOR
    return ADMITTED


def _expected_driver(run_class: str, stage: str) -> str:
    return ADMITTED if _expected_request(run_class, stage) == ADMITTED else NO_DRIVER


GRID = [(c, s) for c in CLASSES for s in STAGES]


# ---------------------------------------------------------------- worker request
@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_worker_request_gate(run_class: str, stage: str) -> None:
    base = request() if run_class == "local_tiny" else remote_request()
    kwargs = {f.name: getattr(base, f.name) for f in fields(base)}
    kwargs["profile"] = _profile(run_class, stage)
    observed = _outcome(lambda: ProbeRequest(**kwargs))
    assert observed == _expected_request(run_class, stage)


@pytest.mark.parametrize(
    ("run_class", "remote_proof", "expected"),
    [
        ("local_tiny", False, ADMITTED),
        ("local_tiny", True, "local tiny cannot claim unverified remote source proof"),
        ("remote_tiny", True, ADMITTED),
        ("remote_tiny", False, "remote worker requires immutable image and source proof"),
    ],
)
def test_worker_request_input_rules_follow_the_class(
    run_class: str, remote_proof: bool, expected: str
) -> None:
    base = remote_request() if remote_proof else request()
    profile = replace(base.profile, run_class=run_class, device="cuda", precision="bf16")
    if run_class == "local_tiny":
        profile = replace(base.profile, run_class="local_tiny")
    kwargs = {f.name: getattr(base, f.name) for f in fields(base)}
    kwargs["profile"] = profile
    assert _outcome(lambda: ProbeRequest(**kwargs)) == expected


# ------------------------------------------------------------------------ runner
@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_runner_receipt_gate_is_class_only(
    run_class: str, stage: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # recovery_record's own arguments read the artifact before the stub can run, so a
    # later read error is also "past the gate"; only the gate's message is a refusal.
    monkeypatch.setattr(runner, "recovery_record", _raise_passed)
    observed = _gate(
        NO_RECEIPT,
        lambda: runner._read_training_progress(  # pyright: ignore[reportPrivateUsage]
            tmp_path,
            request().inputs[0],
            _profile(run_class, stage),
            tmp_path,
            max_source_bytes=1,
            max_artifact_bytes=1,
            memo=None,
        ),
    )
    assert observed == (ADMITTED if run_class in ("local_tiny", "remote_tiny") else NO_RECEIPT)


@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_runner_driver_gate(
    run_class: str, stage: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "normalize_d1_artifact_root", _raise_passed)
    observed = _outcome(
        lambda: runner.run_probe_training(
            _profile(run_class, stage),
            tmp_path,
            tmp_path,
            max_source_bytes=1,
            max_artifact_bytes=1,
        )
    )
    assert observed == _expected_driver(run_class, stage)


# -------------------------------------------------------------- worker command
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_worker_command_scope(run_class: str, stage: str, local: bool) -> None:
    observed = _outcome(
        lambda: worker_command._scope(  # pyright: ignore[reportPrivateUsage]
            _request(run_class, stage), local, 10**12, 10**6
        )
    )
    expected = "local_tiny" if local else "remote_tiny"
    assert observed == (ADMITTED if run_class == expected else NO_SCOPE)


# ------------------------------------------------- execution and completion kind
@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_execution_runtime_kind(
    run_class: str, stage: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(worker_execution, "require_observed_execution_precision", lambda _: {})
    runtime: dict[str, object] = {}

    def call() -> None:
        runtime.update(
            worker_execution._runtime(  # pyright: ignore[reportPrivateUsage]
                _request(run_class, stage), tmp_path, None
            )
        )

    observed = _outcome(call)
    if run_class == "local_tiny":
        assert observed == ADMITTED and runtime["kind"] == "local_tiny"
    else:
        assert observed == "remote worker must execute its bound source tree"


@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_completion_runtime_branch(run_class: str, stage: str) -> None:
    local_runtime = {"kind": "local_tiny", "pod_id": None}
    observed = _outcome(
        lambda: worker_completion.admit_worker_runtime(
            local_runtime, _request(run_class, stage), None
        )
    )
    branch = LOCAL if observed == ADMITTED else REMOTE
    assert branch == (LOCAL if run_class == "local_tiny" else REMOTE)
    if run_class != "local_tiny":
        remote_runtime = {"kind": "local_tiny", "pod_id": "pod-x"}
        refused = _outcome(
            lambda: worker_completion.admit_worker_runtime(
                remote_runtime, _request(run_class, stage), "pod-x"
            )
        )
        assert refused == "worker runtime differs from the allocated construction"


# ------------------------------------------------------------------------ launch
@pytest.mark.parametrize(
    "cost_guard", ["host_watchdog_tiny_only", "budgeted", "", "HOST_WATCHDOG_TINY_ONLY"]
)
@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_launch_gate(
    run_class: str, stage: str, cost_guard: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = launch()
    probe = _request(run_class, stage)
    monkeypatch.setattr(h_probe_launch, "parse_probe_request", lambda *_, **__: probe)
    kwargs = {f.name: getattr(template, f.name) for f in fields(template)}
    kwargs["cost_guard"] = cost_guard
    observed = _outcome(lambda: h_probe_launch.H0ProbeLaunch(**kwargs))
    if (run_class, stage) != ("remote_tiny", "H0"):
        assert observed == NO_LAUNCH
    elif cost_guard != "host_watchdog_tiny_only":
        assert observed == NO_GUARD
    else:
        assert observed == ADMITTED
