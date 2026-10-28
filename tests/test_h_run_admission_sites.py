"""Characterization of every H run-class gate, captured from the code before the port.

Rung A (operator ruling 2026-10-27) consolidates the scattered class gates into one
admission port without changing a single verdict. This table was written against
the pre-port code at e73d1230 and is the parity oracle for every refactor that
routes a site through ``ladder/run/admission.py``: a site probe reaches the gate
itself (profiles are built without validation, so classes and stages the profile
validator would refuse upstream still reach it), and stubs the first call after
the gate to raise ``_Passed`` so an admitted case runs no training and touches no
disk. A refusal is recorded as its verbatim message.

Rung B (operator ruling 2026-10-27, "option B") adds the authority axis: every site
is probed with no authority, a matching sealed ``RunAuthority``, one sealed for another
stage, one sealed for another scientific class, and one tampered after sealing. The
smoke classes keep every rung-A verdict and message byte-identical when they carry no
authority; ``pilot`` and ``confirmatory`` are admitted at H1-H5 on the remote worker and
the launch only under a matching authority, and each defect is refused by name.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest
from tests.test_h_probe_launch import launch, remote_request
from tests.test_run_authority import authority as sealed_authority
from tests.test_sentence_worker_request import request

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.ladder.run import (
    h_probe_launch,
    runner,
    worker_command,
    worker_completion,
    worker_execution,
)
from entropic_substrate.ladder.run.run_authority import RunAuthority
from entropic_substrate.ladder.run.worker_request import ProbeRequest

pytestmark = pytest.mark.unit

CLASSES = ("local_tiny", "remote_tiny", "pilot", "confirmatory", "unknown_class")
# The six stages a validated profile can carry. MEASURED 2026-10-27: an unvalidated
# unknown stage ("H9") is ADMITTED for local_tiny by the request and driver gates,
# which never read the stage of a local run; the port refuses it. The divergence is
# unreachable because HProfile refuses an unknown stage first, which
# test_the_profile_fences_unknown_stages_and_classes_before_any_gate pins.
STAGES = ("H0", "H1", "H2", "H3", "H4", "H5")

ADMITTED = "admitted"
NO_EXECUTION = "probe worker has no scientific execution authority"
NO_SUCCESSOR = "remote successor requires its scientific gate adapter"
NO_RECEIPT = "probe receipt has no scientific authority"
NO_DRIVER = "probe driver lacks scientific or successor authority"
NO_SCOPE = "worker command scope differs from the request"
NO_LAUNCH = "deployment is restricted to remote H0 tiny probes"
NO_GUARD = "H0 deployment requires the declared tiny-only watchdog exception"
# Rung B: authority refusals, named by the port, raised verbatim by every site.
SMOKE_AUTHORITY = "smoke run classes carry no run authority"
NO_AUTHORITY = "scientific run requires a sealed run authority"
UNSEALED = "run authority fails its seal"
OTHER_CLASS = "run authority names another run class"
OTHER_STAGE = "run authority names another stage"
AUTHORITY_REFUSALS = (SMOKE_AUTHORITY, NO_AUTHORITY, UNSEALED, OTHER_CLASS, OTHER_STAGE)
NO_BUDGET = "scientific deployment requires the budgeted host watchdog"
SCIENTIFIC = ("pilot", "confirmatory")
SCIENTIFIC_STAGES = ("H1", "H2", "H3", "H4", "H5")
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
        object.__setattr__(
            value, field.name, changes.get(field.name, getattr(template, field.name))
        )
    return value


def _profile(run_class: str, stage: str) -> HProfile:
    return _unvalidated(remote_request().profile, run_class=run_class, stage=stage)


AUTHORITIES = ("none", "matching", "other_stage", "other_class", "tampered")


def _authority(kind: str, run_class: str, stage: str) -> RunAuthority | None:
    """A sealed authority of ``kind`` for a run of ``run_class`` at ``stage``.

    A smoke or unknown class has no matching scientific class, so ``matching`` and
    ``other_stage`` seal for ``pilot``; every non-``none`` kind is a defect there.
    ``tampered`` is sealed for another stage and then edited to this one, so only
    the seal check can catch it.
    """
    if kind == "none":
        return None
    cls = run_class if run_class in SCIENTIFIC else "pilot"
    other_stage = "H2" if stage == "H1" else "H1"
    if kind == "matching":
        return sealed_authority(run_class=cls, stage=stage)
    if kind == "other_stage":
        return sealed_authority(run_class=cls, stage=other_stage)
    if kind == "other_class":
        return sealed_authority(
            run_class="confirmatory" if cls == "pilot" else "pilot", stage=stage
        )
    forged = sealed_authority(run_class=cls, stage=other_stage)
    object.__setattr__(forged, "stage", stage)
    return forged


def _request(run_class: str, stage: str, kind: str = "none") -> ProbeRequest:
    base = request() if run_class == "local_tiny" else remote_request()
    return _unvalidated(
        base, profile=_profile(run_class, stage), authority=_authority(kind, run_class, stage)
    )


def _expected_authority(run_class: str, stage: str, kind: str) -> str | None:
    """The named authority refusal, or ``None`` when the authority axis admits.

    An unknown class is refused by its class or table gate, never by authority.
    """
    if run_class == "unknown_class":
        return None
    if run_class not in SCIENTIFIC:
        return None if kind == "none" else SMOKE_AUTHORITY
    return {
        "none": NO_AUTHORITY,
        "matching": None,
        "other_stage": OTHER_STAGE,
        "other_class": OTHER_CLASS,
        "tampered": UNSEALED,
    }[kind]


def _expected_request(run_class: str, stage: str, kind: str = "none") -> str:
    if run_class not in ("local_tiny", "remote_tiny", *SCIENTIFIC):
        return NO_EXECUTION
    refusal = _expected_authority(run_class, stage, kind)
    if refusal is not None:
        return refusal
    if run_class == "remote_tiny" and stage != "H0":
        return NO_SUCCESSOR
    if run_class in SCIENTIFIC and stage not in SCIENTIFIC_STAGES:
        return NO_SUCCESSOR
    return ADMITTED


def _expected_driver(run_class: str, stage: str, kind: str = "none") -> str:
    verdict = _expected_request(run_class, stage, kind)
    return verdict if verdict in (ADMITTED, *AUTHORITY_REFUSALS) else NO_DRIVER


GRID = [(c, s) for c in CLASSES for s in STAGES]
AUTHORITY_GRID = [(c, s, k) for c in CLASSES for s in STAGES for k in AUTHORITIES]


@pytest.mark.parametrize(
    ("change", "message"),
    [({"stage": "H9"}, "unknown H stage"), ({"run_class": "unknown_class"}, "unknown run class")],
)
def test_the_profile_fences_unknown_stages_and_classes_before_any_gate(
    change: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(request().profile, **change)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------- worker request
@pytest.mark.parametrize(("run_class", "stage", "kind"), AUTHORITY_GRID)
def test_worker_request_gate(run_class: str, stage: str, kind: str) -> None:
    base = request() if run_class == "local_tiny" else remote_request()
    kwargs = {f.name: getattr(base, f.name) for f in fields(base)}
    kwargs["profile"] = _profile(run_class, stage)
    kwargs["authority"] = _authority(kind, run_class, stage)
    observed = _outcome(lambda: ProbeRequest(**kwargs))
    assert observed == _expected_request(run_class, stage, kind)


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
    assert observed == (ADMITTED if run_class != "unknown_class" else NO_RECEIPT)


@pytest.mark.parametrize(("run_class", "stage", "kind"), AUTHORITY_GRID)
def test_runner_driver_gate(
    run_class: str, stage: str, kind: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "normalize_d1_artifact_root", _raise_passed)
    observed = _outcome(
        lambda: runner.run_probe_training(
            _profile(run_class, stage),
            tmp_path,
            tmp_path,
            max_source_bytes=1,
            max_artifact_bytes=1,
            authority=_authority(kind, run_class, stage),
        )
    )
    assert observed == _expected_driver(run_class, stage, kind)


# -------------------------------------------------------------- worker command
@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_worker_command_scope(run_class: str, stage: str, local: bool) -> None:
    observed = _outcome(
        lambda: worker_command._scope(  # pyright: ignore[reportPrivateUsage]
            _request(run_class, stage), local, 10**12, 10**6
        )
    )
    expected = ("local_tiny",) if local else ("remote_tiny", *SCIENTIFIC)
    assert observed == (ADMITTED if run_class in expected else NO_SCOPE)


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


def _attested_remote_runtime(
    probe: ProbeRequest, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, object]:
    """The remote branch with its source, bundle and hardware attestations stubbed."""

    @contextmanager
    def bundle(*_: object) -> Iterator[object]:
        yield object()

    monkeypatch.setattr(worker_execution, "require_observed_execution_precision", lambda _: {})
    monkeypatch.setattr(worker_execution, "source_root", lambda: tmp_path / "src")
    monkeypatch.setattr(worker_execution, "verified_git_bundle", bundle)
    monkeypatch.setattr(worker_execution, "verify_source_tree", lambda *_: None)
    monkeypatch.setattr(worker_execution, "measure_hardware", lambda: ("Measured GPU", 1))
    monkeypatch.setenv("RUNPOD_POD_ID", "podfixture1234")
    return worker_execution._runtime(  # pyright: ignore[reportPrivateUsage]
        probe, tmp_path, tmp_path
    )


@pytest.mark.parametrize("run_class", ["remote_tiny", *SCIENTIFIC])
def test_remote_runtime_kind_and_authority_echo(
    run_class: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage = "H0" if run_class == "remote_tiny" else "H1"
    kind = "none" if run_class == "remote_tiny" else "matching"
    probe = _request(run_class, stage, kind)
    runtime = _attested_remote_runtime(probe, monkeypatch, tmp_path)
    if run_class == "remote_tiny":
        assert runtime["kind"] == "remote_tiny" and "authority" not in runtime
    else:
        assert probe.authority is not None
        assert runtime["kind"] == f"remote_{run_class}"
        assert runtime["authority"] == probe.authority.digest
    admitted = worker_completion.admit_worker_runtime(runtime, probe, "podfixture1234")
    assert admitted == runtime


@pytest.mark.parametrize(
    "defect", ["kind_tiny", "authority_missing", "authority_other", "authority_on_smoke"]
)
def test_completion_refuses_a_runtime_that_does_not_echo_its_authority(
    defect: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_class = "remote_tiny" if defect == "authority_on_smoke" else "pilot"
    stage = "H0" if run_class == "remote_tiny" else "H1"
    probe = _request(run_class, stage, "none" if run_class == "remote_tiny" else "matching")
    runtime = dict(_attested_remote_runtime(probe, monkeypatch, tmp_path))
    if defect == "kind_tiny":
        runtime["kind"] = "remote_tiny"
    elif defect == "authority_missing":
        del runtime["authority"]
    else:
        runtime["authority"] = sealed_authority(stage="H2").digest
    refused = _outcome(
        lambda: worker_completion.admit_worker_runtime(runtime, probe, "podfixture1234")
    )
    assert refused == "worker runtime differs from the allocated construction"


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
    if run_class == "local_tiny":
        echoed = {**local_runtime, "authority": "0" * 64}
        refused_echo = _outcome(
            lambda: worker_completion.admit_worker_runtime(echoed, _request(run_class, stage), None)
        )
        assert refused_echo == "local completion cannot claim a provider allocation"
    if run_class != "local_tiny":
        remote_runtime = {"kind": "local_tiny", "pod_id": "pod-x"}
        refused = _outcome(
            lambda: worker_completion.admit_worker_runtime(
                remote_runtime, _request(run_class, stage), "pod-x"
            )
        )
        assert refused == "worker runtime differs from the allocated construction"


# ------------------------------------------------------------------------ launch
GUARD_CASES = [
    "host_watchdog_tiny_only",
    "host_watchdog_budgeted",
    "budgeted",
    "",
    "HOST_WATCHDOG_TINY_ONLY",
    None,
    ["host_watchdog_tiny_only"],
]


def _expected_launch(run_class: str, stage: str, kind: str, cost_guard: object) -> str:
    refusal = _expected_authority(run_class, stage, kind)
    if refusal is not None:
        return refusal
    scientific = run_class in SCIENTIFIC and stage in SCIENTIFIC_STAGES
    if (run_class, stage) != ("remote_tiny", "H0") and not scientific:
        return NO_LAUNCH
    if scientific:
        return ADMITTED if cost_guard == "host_watchdog_budgeted" else NO_BUDGET
    return ADMITTED if cost_guard == "host_watchdog_tiny_only" else NO_GUARD


@pytest.mark.parametrize("cost_guard", GUARD_CASES)
@pytest.mark.parametrize(("run_class", "stage", "kind"), AUTHORITY_GRID)
def test_launch_gate(
    run_class: str, stage: str, kind: str, cost_guard: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = launch()
    probe = _request(run_class, stage, kind)
    monkeypatch.setattr(h_probe_launch, "parse_probe_request", lambda *_, **__: probe)
    kwargs = {f.name: getattr(template, f.name) for f in fields(template)}
    kwargs["cost_guard"] = cost_guard
    observed = _outcome(lambda: h_probe_launch.H0ProbeLaunch(**kwargs))
    assert observed == _expected_launch(run_class, stage, kind, cost_guard)


def test_the_smoke_verdicts_with_no_authority_are_rung_a_verbatim() -> None:
    """With no authority every smoke verdict is the rung-A characterization."""
    for run_class, stage in GRID:
        if run_class in SCIENTIFIC:
            continue
        legacy_request = (
            NO_EXECUTION
            if run_class == "unknown_class"
            else NO_SUCCESSOR
            if run_class == "remote_tiny" and stage != "H0"
            else ADMITTED
        )
        assert _expected_request(run_class, stage) == legacy_request
        legacy_driver = ADMITTED if legacy_request == ADMITTED else NO_DRIVER
        assert _expected_driver(run_class, stage) == legacy_driver
