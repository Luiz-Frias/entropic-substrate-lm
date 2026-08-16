"""Contracts for the live-gate shell boundary (``scripts/run_container_test.sh``)."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]

# The allowlist knobs the gate must neutralize for its OWN stack. Both are
# operator-facing names; compose.yaml maps them onto MLflow's native
# MLFLOW_SERVER_* variables.
ALLOWLIST_VARS = ("MLFLOW_ALLOWED_HOSTS", "MLFLOW_CORS_ALLOWED_ORIGINS")


@pytest.mark.parametrize("var_name", ALLOWLIST_VARS)
def test_live_gate_neutralizes_allowlist_vars_with_empty_not_unset(var_name: str) -> None:
    """A developer's ``.env`` must not be able to make the live gate 403 itself.

    THE HAZARD. A set allowlist REPLACES mlflow's compiled-in localhost default
    rather than extending it. This gate reaches its own mlflow container at
    ``localhost:<ephemeral port>``, so a Railway-domain value in ``.env`` — set
    for perfectly good reasons, e.g. "match the deploy" — makes every
    integration test 403 on that developer's machine and nobody else's.

    WHY ``export VAR=""`` AND NOT ``unset VAR``. This is the whole point of the
    test, and the intuitive fix is the wrong one. podman-compose reads the
    ``.env`` sitting beside compose.yaml ITSELF, independently of this process's
    environment, so removing the variable from the process env does not remove
    it from the render. MEASURED against podman-compose 1.6.0 with a scratch
    stack whose ``.env`` contained ``MLFLOW_ALLOWED_HOSTS=FROM_DOTENV_FILE`` and
    a compose file interpolating ``${MLFLOW_ALLOWED_HOSTS:-UNSET_DEFAULT}``::

        process env UNSET               -> FROM_DOTENV_FILE   (.env wins!)
        process env = FROM_PROCESS_ENV  -> FROM_PROCESS_ENV
        process env = "" (empty)        -> UNSET_DEFAULT      (empty wins)

    So ``unset`` leaves the .env value fully in force, while an EMPTY exported
    value takes precedence over .env and lets the ``:-`` default apply. Only the
    latter actually isolates the gate.
    """
    script = (REPO_ROOT / "scripts" / "run_container_test.sh").read_text(encoding="utf-8")

    assert f'export {var_name}=""' in script, (
        f"run_container_test.sh must neutralize {var_name} with an EMPTY EXPORTED "
        "value before invoking podman-compose. A developer's .env would otherwise "
        "reach the gate's own stack and 403 every integration test (see this test's "
        "docstring for the measured precedence order)."
    )

    # `unset` is not merely weaker here, it is INEFFECTIVE — .env still wins.
    # Pinned so a future "simplification" to the more idiomatic-looking form
    # fails loudly instead of silently reopening the hole.
    assert f"unset {var_name}" not in script, (
        f"run_container_test.sh uses `unset {var_name}`, which does NOT neutralize a "
        "value set in .env — podman-compose reads that file directly. Use "
        f'`export {var_name}=""` instead.'
    )


def test_live_gate_reaps_orphaned_healthcheck_timers_on_both_exit_paths() -> None:
    """Every gate run must clean up the healthcheck timers it caused.

    A podman healthcheck timer can outlive its container. Orphans re-fire every
    3s against an id that no longer exists, and this suite's bounds are
    load-sensitive (MEASURED: 29.64s to healthy on a settled VM versus 135.71s at
    load1 4.4, against a 120s bound), so an orphan left by one run RATCHETS the
    floor for the next. That makes the baseline move, which is what disguises the
    problem as flakiness rather than as accumulation.

    Pinned inside ``cleanup_main_stack`` specifically, because that function runs
    on BOTH exit paths — a red run is if anything the more likely to have left an
    orphan behind, so reaping only on success would miss the common case.
    """
    script = (REPO_ROOT / "scripts" / "run_container_test.sh").read_text(encoding="utf-8")

    assert "reap_healthcheck_orphans.sh" in script, (
        "run_container_test.sh must invoke scripts/reap_healthcheck_orphans.sh so a "
        "gate run cleans up the healthcheck timers it caused"
    )

    cleanup = script.partition("cleanup_main_stack()")[2].partition("\n}")[0]
    assert "reap_healthcheck_orphans.sh" in cleanup, (
        "the reap must live in cleanup_main_stack (the EXIT trap), which runs on BOTH "
        "the success and failure paths — not somewhere only a green run reaches"
    )


def test_reaper_script_exists_and_is_executable_and_never_fails_its_caller() -> None:
    """The reaper is cleanup, so it must not be able to fail the run it cleans up.

    Two properties: it ends with an explicit ``exit 0``, and the call site tolerates
    a failure anyway (``|| true``). Belt and braces is deliberate here — this script
    mutates machine state inside the podman VM, and a future edit that introduces a
    non-zero path must not be able to turn cleanup into a gate failure.
    """
    reaper = REPO_ROOT / "scripts" / "reap_healthcheck_orphans.sh"
    assert reaper.exists(), "scripts/reap_healthcheck_orphans.sh is missing"

    body = reaper.read_text(encoding="utf-8")
    assert body.rstrip().endswith("exit 0"), "the reaper must end with an explicit `exit 0`"
    assert "podman inspect" in body, (
        "the reaper must check each unit's container INDIVIDUALLY before stopping its "
        "timer — otherwise it would disturb a live container's healthcheck"
    )

    gate = (REPO_ROOT / "scripts" / "run_container_test.sh").read_text(encoding="utf-8")
    assert 'reap_healthcheck_orphans.sh" || true' in gate.replace("'", '"'), (
        "the reap call must be `|| true`-guarded so cleanup can never fail the gate"
    )


def test_gate_waits_for_the_vm_to_settle_between_phases() -> None:
    """Phase 2 must not start on the machine phase 1's teardown just made busy.

    This is a CAUSE fix, not a bigger number, and the distinction is the point.
    Phase 2's 120s health bound is fine on a quiet VM — MEASURED time from ``up``
    to healthy on this 4-vCPU machine, one stack at a time: 29.64s at load1 0.86
    versus 135.71s at load1 4.4. The bound only fails because phase 2 begins
    immediately after phase 1's teardown, i.e. at the exact moment the VM is
    least settled. Raising the bound would move the cliff rather than remove it
    (House rule 3; see sinks.py for the same lesson learned the expensive way).

    Pinned properties: the settle happens AFTER the main stack is torn down and
    BEFORE phase 2 runs, and it can never fail the gate — a machine that refuses
    to settle must degrade to the old behaviour with a warning, not to a hang.
    """
    script = (REPO_ROOT / "scripts" / "run_container_test.sh").read_text(encoding="utf-8")

    assert "settle_podman_vm" in script, (
        "run_container_test.sh must wait for the podman VM to settle between phases"
    )

    teardown_at = script.index("cleanup_main_stack 0")
    settle_at = script.index("settle_podman_vm()")
    phase2_at = script.index("phase 2/3")
    assert teardown_at < settle_at < phase2_at, (
        "the settle must sit between the phase-1 teardown and the phase-2 run — "
        "settling before the teardown would wait for the wrong thing"
    )

    settle_body = script.partition("settle_podman_vm()")[2].partition("\n}")[0]
    assert "return 0" in settle_body, (
        "settle_podman_vm must always return 0 — a machine that will not settle has "
        "to degrade to a warning, never fail the gate or hang it"
    )
    assert "WARNING" in settle_body, (
        "a give-up path that stays silent would make a machine-caused timing failure "
        "downstream look like a code failure"
    )

    # It must poll the HOST, not the guest. The podman VM's vCPUs are
    # time-sliced by the hypervisor, so a quiet guest on a saturated host is
    # still starved — MEASURED while this gate was failing: guest showed 1
    # runnable process of 193 and a completely clean container/network/volume
    # state, while the 8-core host sat at load 42.86 under an unrelated Rust
    # build. Polling the guest would have called that machine settled.
    assert "vm.loadavg" in settle_body, (
        "settle_podman_vm must read the HOST load (sysctl vm.loadavg); the guest's "
        "own load average measures the STARVATION rather than its cause"
    )
    assert "/proc/loadavg" not in settle_body, (
        "settle_podman_vm must not gate on the GUEST's /proc/loadavg — that was the "
        "first version of this fix and it would pass on a saturated host"
    )
    assert "hw.logicalcpu" in settle_body, (
        "the threshold must be normalised per core — a load of 8 means something "
        "very different on 8 cores than on 2"
    )


# --------------------------------------------------------------- p4: phase 3


def test_phase_1_ignores_phase_2_and_phase_3_modules() -> None:
    """`-m integration` in phase 1's pytest invocation must exclude BOTH
    tests/test_mlflow_allowed_hosts.py (phase 2) and tests/test_caddy_auth.py
    (phase 3) — MEASURED failure mode: adding tests/test_caddy_auth.py
    without this ignore let phase 1's own `-m integration` sweep it in,
    so its fixture tried to bring up a THIRD live stack while phase 1's was
    still up, and assert_no_foreign_mlflow_stack() correctly refused it.
    """
    script = (REPO_ROOT / "scripts" / "run_container_test.sh").read_text(encoding="utf-8")
    phase1_call = script.partition("pytest -m integration -q --disable-warnings")[2].partition(
        "\n\n"
    )[0]
    assert "--ignore=tests/test_mlflow_allowed_hosts.py" in phase1_call, phase1_call
    assert "--ignore=tests/test_caddy_auth.py" in phase1_call, phase1_call


def test_gate_waits_for_the_vm_to_settle_before_phase_3_too() -> None:
    """Same seam, same cause, one phase boundary later: phase 2's own
    fixture tears its stack down (down -v --remove-orphans, inside its
    `finally`) immediately before phase 3 starts, which is exactly what
    makes the VM busiest at that moment — see the phase-1/phase-2 test
    above for the underlying measurement.
    """
    script = (REPO_ROOT / "scripts" / "run_container_test.sh").read_text(encoding="utf-8")

    phase2_at = script.index("phase 2/3")
    second_settle_at = script.index("settle_podman_vm", phase2_at)
    phase3_at = script.index("phase 3/3")
    assert phase2_at < second_settle_at < phase3_at, (
        "a second settle_podman_vm call must sit between the phase-2 run and the phase-3 run"
    )


def test_backup_teardown_trap_covers_both_the_allowlist_and_caddy_projects() -> None:
    """The re-armed EXIT trap (a SIGTERM backup net, not either phase's
    primary cleanup — each fixture's own `finally` already owns that) must
    cover phase 3's project too, not just phase 2's, or a SIGTERM mid-phase-3
    would leak that stack.
    """
    script = (REPO_ROOT / "scripts" / "run_container_test.sh").read_text(encoding="utf-8")

    assert "CADDY_PROJECT=" in script, "run_container_test.sh must derive phase 3's project name"
    backup = script.partition("cleanup_backup_stacks()")[2].partition("\ntrap")[0]
    assert "$ALLOWLIST_PROJECT" in backup, backup
    assert "$CADDY_PROJECT" in backup, backup
    assert "--profile proxy" in backup, (
        "the caddy project's teardown must pass --profile proxy, mirroring the `up` "
        "invocation that created it"
    )
