"""The H run admission port: boundaries, table validation and parity with every gate site."""

import ast
from pathlib import Path
from typing import cast

import pytest
from tests.test_h_run_admission_sites import (
    ADMITTED,
    CLASSES,
    GRID,
    NO_DRIVER,
    NO_GUARD,
    NO_LAUNCH,
    NO_RECEIPT,
    _expected_driver,  # pyright: ignore[reportPrivateUsage]
    _expected_request,  # pyright: ignore[reportPrivateUsage]
)

from entropic_substrate.ladder.run import admission
from entropic_substrate.ladder.run.admission import (
    RUN_ADMISSION,
    RUNTIME_KIND,
    RunAdmission,
    Surface,
)

pytestmark = pytest.mark.unit

GUARDS = ("host_watchdog_tiny_only", "budgeted", "", "HOST_WATCHDOG_TINY_ONLY")


# ------------------------------------------------------------------- boundaries
@pytest.mark.parametrize("surface", ["local_worker", "remote_worker", "launch", None])
def test_unknown_class_and_unknown_stage_are_refused_everywhere(surface: Surface | None) -> None:
    assert not RUN_ADMISSION.admits("unknown_class", "H0", surface)
    assert not RUN_ADMISSION.admits("local_tiny", "H9", surface)
    assert not RUN_ADMISSION.admits("remote_tiny", "h0", surface)
    assert RUN_ADMISSION.worker_surface("unknown_class") is None
    assert not RUN_ADMISSION.admits_class("unknown_class")


def test_no_surface_is_never_admitted() -> None:
    assert not any(RUN_ADMISSION.admits(c, s, None) for c, s in GRID)


def test_local_and_remote_worker_surfaces_do_not_cross() -> None:
    assert RUN_ADMISSION.admits("local_tiny", "H3", "local_worker")
    assert not RUN_ADMISSION.admits("local_tiny", "H0", "remote_worker")
    assert not RUN_ADMISSION.admits("local_tiny", "H0", "launch")
    assert RUN_ADMISSION.admits("remote_tiny", "H0", "remote_worker")
    assert not RUN_ADMISSION.admits("remote_tiny", "H0", "local_worker")
    assert not RUN_ADMISSION.admits("remote_tiny", "H1", "remote_worker")


@pytest.mark.parametrize("run_class", CLASSES)
@pytest.mark.parametrize("cost_guard", GUARDS)
def test_only_the_tiny_watchdog_guard_is_admitted_and_only_for_remote_tiny(
    run_class: str, cost_guard: str
) -> None:
    expected = (run_class, cost_guard) == ("remote_tiny", "host_watchdog_tiny_only")
    assert RUN_ADMISSION.admits_cost_guard(run_class, cost_guard) is expected


@pytest.mark.parametrize("cost_guard", [None, ["host_watchdog_tiny_only"], 1])
def test_a_non_string_cost_guard_is_refused_not_raised(cost_guard: object) -> None:
    assert not RUN_ADMISSION.admits_cost_guard("remote_tiny", cost_guard)  # type: ignore[arg-type]


def test_scientific_classes_are_admitted_nowhere_today() -> None:
    for run_class in ("pilot", "confirmatory"):
        assert not RUN_ADMISSION.admits_class(run_class)
        assert not any(
            RUN_ADMISSION.admits(run_class, s, surface)
            for _, s in GRID
            for surface in ("local_worker", "remote_worker", "launch")
        )


def test_runtime_kinds_are_the_receipt_wire_values() -> None:
    assert dict(RUNTIME_KIND) == {"local_worker": "local_tiny", "remote_worker": "remote_tiny"}


# ----------------------------------------------------------------- table checks
def test_the_policy_is_frozen() -> None:
    with pytest.raises(TypeError):
        RUN_ADMISSION.surfaces[("pilot", "H1")] = frozenset({"launch"})  # type: ignore[index]
    with pytest.raises(TypeError):
        RUN_ADMISSION.cost_guards["pilot"] = frozenset({"budgeted"})  # type: ignore[index]


@pytest.mark.parametrize(
    ("surfaces", "workers", "guards", "message"),
    [
        (
            {("local_tiny", "H9"): frozenset({"local_worker"})},
            {"local_tiny": "local_worker"},
            {},
            "unknown stage or surface",
        ),
        (
            {("local_tiny", "H0"): frozenset({"cloud"})},
            {"local_tiny": "local_worker"},
            {},
            "unknown stage or surface",
        ),
        (
            {("local_tiny", "H0"): frozenset({"remote_worker"})},
            {"local_tiny": "local_worker"},
            {},
            "exactly its one worker surface",
        ),
        ({("pilot", "H1"): frozenset({"remote_worker"})}, {}, {}, "exactly its one worker surface"),
        ({}, {"local_tiny": "gpu"}, {}, "unknown worker surface"),
        (
            {("local_tiny", "H0"): frozenset({"local_worker"})},
            {"local_tiny": "local_worker"},
            {"local_tiny": frozenset({"x"})},
            "only to a class admitted at launch",
        ),
    ],
)
def test_an_inconsistent_table_is_refused(
    surfaces: object, workers: object, guards: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        RunAdmission(surfaces, workers, guards)  # type: ignore[arg-type]


# ----------------------------------------------------- parity with the gate sites
# Each expression below is the site's check rewritten through the port; it must
# reproduce the characterized verdict for every class and stage.
def _request_verdict(run_class: str, stage: str) -> str:
    if not RUN_ADMISSION.admits_class(run_class):
        return "probe worker has no scientific execution authority"
    if not RUN_ADMISSION.admits(run_class, stage, RUN_ADMISSION.worker_surface(run_class)):
        return "remote successor requires its scientific gate adapter"
    return ADMITTED


@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_port_reproduces_the_request_receipt_and_driver_gates(run_class: str, stage: str) -> None:
    assert _request_verdict(run_class, stage) == _expected_request(run_class, stage)
    driver = RUN_ADMISSION.admits(run_class, stage, RUN_ADMISSION.worker_surface(run_class))
    assert (ADMITTED if driver else NO_DRIVER) == _expected_driver(run_class, stage)
    receipt = ADMITTED if RUN_ADMISSION.admits_class(run_class) else NO_RECEIPT
    assert receipt == (ADMITTED if run_class in ("local_tiny", "remote_tiny") else NO_RECEIPT)


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("run_class", CLASSES)
def test_port_reproduces_the_command_scope_and_runtime_branch(run_class: str, local: bool) -> None:
    surface = RUN_ADMISSION.worker_surface(run_class)
    assert (surface == ("local_worker" if local else "remote_worker")) is (
        run_class == ("local_tiny" if local else "remote_tiny")
    )
    is_local = surface == "local_worker"
    assert is_local is (run_class == "local_tiny")
    assert RUNTIME_KIND["local_worker" if is_local else "remote_worker"] == (
        "local_tiny" if run_class == "local_tiny" else "remote_tiny"
    )


@pytest.mark.parametrize("cost_guard", GUARDS)
@pytest.mark.parametrize(("run_class", "stage"), GRID)
def test_port_reproduces_the_launch_gate(run_class: str, stage: str, cost_guard: str) -> None:
    if not RUN_ADMISSION.admits(run_class, stage, "launch"):
        verdict = NO_LAUNCH
    elif not RUN_ADMISSION.admits_cost_guard(run_class, cost_guard):
        verdict = NO_GUARD
    else:
        verdict = ADMITTED
    if (run_class, stage) != ("remote_tiny", "H0"):
        assert verdict == NO_LAUNCH
    elif cost_guard != "host_watchdog_tiny_only":
        assert verdict == NO_GUARD
    else:
        assert verdict == ADMITTED


# ------------------------------------------------------------------------- leaf
def test_the_port_is_a_stdlib_only_leaf() -> None:
    tree = ast.parse(Path(cast(str, admission.__file__)).read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"collections.abc", "dataclasses", "types", "typing"}
