"""The H run admission port: boundaries, table validation and parity with every gate site.

Rung B (operator ruling 2026-10-27, "option B"): the port also reads the sealed
``RunAuthority``; the authority checks are pinned here at the port and, per site, in
``tests/test_h_run_admission_sites.py``.
"""

import ast
from pathlib import Path
from typing import cast

import pytest
from tests.test_h_run_admission_sites import (
    ADMITTED,
    AUTHORITY_GRID,
    CLASSES,
    GRID,
    NO_BUDGET,
    NO_DRIVER,
    NO_GUARD,
    NO_LAUNCH,
    NO_RECEIPT,
    NO_SUCCESSOR,
    OTHER_CLASS,
    OTHER_STAGE,
    SCIENTIFIC,
    SCIENTIFIC_STAGES,
    SMOKE_AUTHORITY,
    UNSEALED,
    _authority,  # pyright: ignore[reportPrivateUsage]
    _expected_driver,  # pyright: ignore[reportPrivateUsage]
    _expected_launch,  # pyright: ignore[reportPrivateUsage]
    _expected_request,  # pyright: ignore[reportPrivateUsage]
)
from tests.test_run_authority import authority as sealed_authority

from entropic_substrate.ladder.run import admission
from entropic_substrate.ladder.run.admission import (
    BUDGETED_GUARD,
    RUN_ADMISSION,
    RUNTIME_KIND,
    TINY_GUARD,
    RunAdmission,
    Surface,
)
from entropic_substrate.ladder.run.run_authority import RunAuthority

pytestmark = pytest.mark.unit

GUARDS = (
    "host_watchdog_tiny_only",
    "host_watchdog_budgeted",
    "budgeted",
    "",
    "HOST_WATCHDOG_TINY_ONLY",
)


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
def test_each_class_family_has_exactly_its_own_watchdog_guard(
    run_class: str, cost_guard: str
) -> None:
    expected = (run_class, cost_guard) == ("remote_tiny", TINY_GUARD) or (
        run_class in SCIENTIFIC and cost_guard == BUDGETED_GUARD
    )
    assert RUN_ADMISSION.admits_cost_guard(run_class, cost_guard) is expected


def test_the_guard_names_are_the_wire_values() -> None:
    assert (TINY_GUARD, BUDGETED_GUARD) == ("host_watchdog_tiny_only", "host_watchdog_budgeted")


@pytest.mark.parametrize("cost_guard", [None, ["host_watchdog_tiny_only"], 1])
def test_a_non_string_cost_guard_is_refused_not_raised(cost_guard: object) -> None:
    assert not RUN_ADMISSION.admits_cost_guard("remote_tiny", cost_guard)  # type: ignore[arg-type]
    assert not RUN_ADMISSION.admits_cost_guard("pilot", cost_guard)  # type: ignore[arg-type]


@pytest.mark.parametrize("run_class", SCIENTIFIC)
def test_scientific_classes_are_admitted_only_under_a_matching_authority(run_class: str) -> None:
    for stage in ("H0", *SCIENTIFIC_STAGES):
        sealed = sealed_authority(run_class=run_class, stage=stage)
        for surface in ("local_worker", "remote_worker", "launch"):
            expected = surface != "local_worker" and stage in SCIENTIFIC_STAGES
            assert RUN_ADMISSION.admits(run_class, stage, surface, sealed) is expected
            assert not RUN_ADMISSION.admits(run_class, stage, surface)
            assert not RUN_ADMISSION.admits(run_class, stage, surface, None)
        assert RUN_ADMISSION.admits_class(run_class)


@pytest.mark.parametrize(("run_class", "stage", "kind"), AUTHORITY_GRID)
def test_authority_refusals_are_named_by_the_port(run_class: str, stage: str, kind: str) -> None:
    refusal = RUN_ADMISSION.authority_refusal(run_class, stage, _authority(kind, run_class, stage))
    verdict = _expected_request(run_class, stage, kind)
    if run_class == "unknown_class":
        assert refusal is None  # the class gate refuses an unknown class first
    elif verdict in (ADMITTED, NO_SUCCESSOR):
        assert refusal is None
    else:
        assert refusal == verdict


def test_a_forgery_that_is_not_a_run_authority_is_refused() -> None:
    sealed = sealed_authority()
    impostor = object.__new__(RunAuthority)
    for name in ("stage", "run_class", "ruling", "spend_cap", "gpu_class", "registration_sha"):
        object.__setattr__(impostor, name, getattr(sealed, name))
    object.__setattr__(impostor, "digest", "0" * 64)
    assert RUN_ADMISSION.authority_refusal("pilot", "H1", impostor) == UNSEALED
    assert RUN_ADMISSION.authority_refusal("pilot", "H1", sealed.manifest) == (
        "scientific run requires a sealed run authority"
    )
    assert not RUN_ADMISSION.admits("pilot", "H1", "launch", impostor)


@pytest.mark.parametrize(
    ("name", "value", "refusal"),
    [
        ("stage", "H2", UNSEALED),
        ("run_class", "confirmatory", UNSEALED),
        ("registration_sha", "b" * 40, UNSEALED),
        ("digest", "0" * 64, UNSEALED),
    ],
)
def test_tamper_after_sealing_is_refused_at_the_port(name: str, value: str, refusal: str) -> None:
    sealed = sealed_authority()
    assert RUN_ADMISSION.authority_refusal("pilot", "H1", sealed) is None
    object.__setattr__(sealed, name, value)
    assert RUN_ADMISSION.authority_refusal("pilot", "H1", sealed) == refusal
    assert not RUN_ADMISSION.admits("pilot", "H1", "remote_worker", sealed)


def test_mismatches_are_refused_by_name() -> None:
    assert RUN_ADMISSION.authority_refusal("pilot", "H1", sealed_authority(stage="H2")) == (
        OTHER_STAGE
    )
    assert RUN_ADMISSION.authority_refusal(
        "pilot", "H1", sealed_authority(run_class="confirmatory")
    ) == (OTHER_CLASS)
    assert RUN_ADMISSION.authority_refusal("remote_tiny", "H0", sealed_authority()) == (
        SMOKE_AUTHORITY
    )


def test_runtime_kinds_are_the_receipt_wire_values() -> None:
    assert dict(RUNTIME_KIND) == {
        "local_tiny": "local_tiny",
        "remote_tiny": "remote_tiny",
        "pilot": "remote_pilot",
        "confirmatory": "remote_confirmatory",
    }


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
        (
            {("remote_tiny", "H0"): frozenset({"remote_worker", "launch"})},
            {"remote_tiny": "remote_worker"},
            {"remote_tiny": frozenset({"host_watchdog_budgeted"})},
            "smoke classes keep the tiny-only guard",
        ),
        (
            {("pilot", "H1"): frozenset({"remote_worker", "launch"})},
            {"pilot": "remote_worker"},
            {"pilot": frozenset({"host_watchdog_tiny_only"})},
            "scientific classes take only the budgeted guard",
        ),
        (
            {("pilot", "H1"): frozenset({"local_worker"})},
            {"pilot": "local_worker"},
            {},
            "scientific classes run only on the remote worker",
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
# reproduce the characterized verdict for every class, stage and authority.
def _request_verdict(run_class: str, stage: str, authority: RunAuthority | None) -> str:
    if not RUN_ADMISSION.admits_class(run_class):
        return "probe worker has no scientific execution authority"
    refusal = RUN_ADMISSION.authority_refusal(run_class, stage, authority)
    if refusal is not None:
        return refusal
    surface = RUN_ADMISSION.worker_surface(run_class)
    if not RUN_ADMISSION.admits(run_class, stage, surface, authority):
        return "remote successor requires its scientific gate adapter"
    return ADMITTED


@pytest.mark.parametrize(("run_class", "stage", "kind"), AUTHORITY_GRID)
def test_port_reproduces_the_request_receipt_and_driver_gates(
    run_class: str, stage: str, kind: str
) -> None:
    authority = _authority(kind, run_class, stage)
    request = _request_verdict(run_class, stage, authority)
    assert request == _expected_request(run_class, stage, kind)
    refusal = RUN_ADMISSION.authority_refusal(run_class, stage, authority)
    surface = RUN_ADMISSION.worker_surface(run_class)
    driver = (
        refusal
        if refusal is not None
        else ADMITTED
        if RUN_ADMISSION.admits(run_class, stage, surface, authority)
        else NO_DRIVER
    )
    assert driver == _expected_driver(run_class, stage, kind)
    receipt = ADMITTED if RUN_ADMISSION.admits_class(run_class) else NO_RECEIPT
    assert receipt == (ADMITTED if run_class != "unknown_class" else NO_RECEIPT)


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("run_class", CLASSES)
def test_port_reproduces_the_command_scope_and_runtime_branch(run_class: str, local: bool) -> None:
    surface = RUN_ADMISSION.worker_surface(run_class)
    remote = ("remote_tiny", *SCIENTIFIC)
    assert (surface == ("local_worker" if local else "remote_worker")) is (
        run_class in (("local_tiny",) if local else remote)
    )
    assert (surface == "local_worker") is (run_class == "local_tiny")
    assert RUNTIME_KIND.get(run_class) == (
        None
        if run_class == "unknown_class"
        else run_class
        if run_class in ("local_tiny", "remote_tiny")
        else f"remote_{run_class}"
    )


@pytest.mark.parametrize("cost_guard", GUARDS)
@pytest.mark.parametrize(("run_class", "stage", "kind"), AUTHORITY_GRID)
def test_port_reproduces_the_launch_gate(
    run_class: str, stage: str, kind: str, cost_guard: str
) -> None:
    authority = _authority(kind, run_class, stage)
    refusal = RUN_ADMISSION.authority_refusal(run_class, stage, authority)
    if refusal is not None:
        verdict = refusal
    elif not RUN_ADMISSION.admits(run_class, stage, "launch", authority):
        verdict = NO_LAUNCH
    elif not RUN_ADMISSION.admits_cost_guard(run_class, cost_guard):
        verdict = NO_BUDGET if run_class in SCIENTIFIC else NO_GUARD
    else:
        verdict = ADMITTED
    assert verdict == _expected_launch(run_class, stage, kind, cost_guard)


# ------------------------------------------------------------------------- leaf
def test_the_port_is_a_stdlib_only_leaf() -> None:
    tree = ast.parse(Path(cast(str, admission.__file__)).read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    # Rung B: the port reads the sealed authority, itself a stdlib-only leaf.
    assert imported <= {
        "collections.abc",
        "dataclasses",
        "types",
        "typing",
        "entropic_substrate.ladder.run.run_authority",
    }


def test_the_authority_module_is_a_stdlib_only_leaf() -> None:
    from entropic_substrate.ladder.run import run_authority

    tree = ast.parse(Path(cast(str, run_authority.__file__)).read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"dataclasses", "datetime", "hashlib", "json", "math", "re", "typing"}
