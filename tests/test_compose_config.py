"""Static config checks for ``compose.yaml`` — the live-gate stack definition.

These are unit tests (no podman, no network): they parse the YAML and assert
on its SHAPE, pinning the two structural properties FIX 2 (concurrency-safe
live gate) depends on. The rest of FIX 2 — that podman actually assigns an
ephemeral port, that ``podman port`` discovers it, that the named volume
survives a failure-path ``down`` without ``-v`` — was verified empirically
against a scratch compose stack (not this one; touching the real stack here
would violate the "don't run container-test" boundary) and is proven for
real by the conductor's live gate, not by these tests. See
``scripts/run_container_test.sh`` for the orchestration these pins protect.
"""

from __future__ import annotations

import pytest

from helpers import load_compose, mlflow_command_script, mlflow_service_config

pytestmark = pytest.mark.unit


def test_mlflow_service_publishes_no_fixed_host_port() -> None:
    """FIX 2(b): a "host:container" mapping (e.g. the old "5001:5000") is a
    concurrency hazard — two worktrees running ``just container-test`` at
    once would collide on the same host port. The compose short-syntax for
    "publish this container port to an EPHEMERAL host port" is the bare
    container port with no colon (verified against the installed
    podman-compose: a scratch stack with ``ports: ["5432"]`` came up as
    ``0.0.0.0:<random>->5432/tcp``, discoverable via ``podman port``).
    """
    ports = mlflow_service_config()["ports"]
    assert ports, "mlflow service must publish its port somehow"
    for entry in ports:
        assert ":" not in str(entry), (
            f"port entry {entry!r} looks like a fixed host:container mapping — "
            "this must stay a bare container port so podman assigns an ephemeral "
            "host port (see compose.yaml's module-level comment)"
        )


def test_postgres_backing_store_uses_a_named_not_anonymous_volume() -> None:
    """FIX 2(c): a failure-path teardown (`down` without `-v`) must leave
    something an operator can actually find. An anonymous volume survives
    too, but as an unlabelled dangling 64-hex ID; a NAMED volume (declared
    once at the top level, referenced by the service) is what makes
    "survives for diagnosis" actually diagnosable — podman names it
    ``${PROJECT}_pgdata`` (verified empirically against this exact shape).
    """
    compose = load_compose()
    postgres_volumes = compose["services"]["postgres"].get("volumes")
    assert postgres_volumes, "postgres service must declare a data volume"
    # Anonymous volumes are bare container-path strings ("/var/lib/..."); a
    # named volume is "name:/container/path" where "name" is also declared
    # under the top-level `volumes:` key.
    mount = postgres_volumes[0]
    assert ":" in mount, f"expected a NAMED 'volume:/path' mount, got {mount!r}"
    volume_name = mount.split(":", 1)[0]
    assert volume_name in (compose.get("volumes") or {}), (
        f"volume {volume_name!r} is mounted but not declared under the top-level "
        "`volumes:` key — that makes it anonymous, not named"
    )


@pytest.mark.parametrize("var_name", ["MLFLOW_ALLOWED_HOSTS", "MLFLOW_CORS_ALLOWED_ORIGINS"])
def test_allowlist_vars_flow_through_environment_not_spliced_into_command(
    var_name: str,
) -> None:
    """P3 host-allowlist injection fix (L2 blocker).

    MEASURED (throwaway probe, /tmp — never touched this repo): with the
    OLD single-``$`` shape (``${MLFLOW_ALLOWED_HOSTS:+--allowed-hosts
    "$MLFLOW_ALLOWED_HOSTS"}`` directly in ``command:``), podman-compose
    1.6.0's interpolator resolves it HOST-SIDE, at ``config``/``up`` parse
    time, using whatever string is in this process's environment — before
    the container's own shell ever runs. Setting
    ``MLFLOW_ALLOWED_HOSTS='foo.test"; touch /tmp/PWNED ; echo "'`` and
    running ``podman compose -f compose.yaml config`` rendered the literal
    text ``--allowed-hosts "foo.test"; touch /tmp/PWNED ; echo ""`` into the
    ``sh -c`` script — i.e. an arbitrary shell command spliced into
    container-startup text, executed on every ``up``, gated only by who can
    set an env var. Every other externally-sourced value in this file
    (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``,
    ``MLFLOW_S3_ENDPOINT_URL``) passes through the ``environment:`` map — a
    pure data channel podman hands to the container's ``exec()`` environ
    array, never re-parsed as shell text. These two vars are the only
    exception in the whole compose/Containerfile surface, which is what
    makes this a blocker and not a style nit.

    Two properties pin the fix:
      1. the var is declared under ``services.mlflow.environment`` (data
         channel), so its value never touches podman-compose's host-side
         string interpolator at all; and
      2. every reference to the var inside ``command:``'s ``sh -c`` script
         is ``$``-escaped (``$$VAR`` / ``$${VAR:+...}``) so podman-compose's
         interpolator leaves it as a LITERAL single ``$`` for the
         CONTAINER's own shell to expand at runtime — never as something
         podman-compose itself substitutes a value into.
    """
    mlflow_service = mlflow_service_config()

    env_block = mlflow_service.get("environment") or {}
    assert var_name in env_block, (
        f"{var_name} must be declared under services.mlflow.environment (a pure "
        "data channel) — not read only inside command: via host-side ${VAR} "
        "interpolation, which splices its value into shell-command text at "
        "podman-compose parse time (see this test's docstring for the measured "
        "injection)."
    )

    # mlflow_command_script() also pins that command: stays an ["sh", "-c",
    # script] wrapper — reading an unrelated element instead of the actual
    # script if the command ever became a flat argv list is exactly the
    # false-pass this shared helper exists to rule out.
    script = mlflow_command_script()

    # UNESCAPED forms podman-compose's host-side interpolator would resolve
    # at parse time — splicing the raw value into the command string. Must
    # not appear.
    for unescaped in (f"${{{var_name}", f"${var_name}"):
        # A double-escaped occurrence ("$${VAR}") contains the single-escaped
        # substring too, so exclude matches that are actually part of a
        # "$$"-prefixed (escaped) token before failing.
        idx = script.find(unescaped)
        while idx != -1:
            preceding = script[max(0, idx - 1) : idx]
            assert preceding == "$", (
                f"command: references {unescaped!r} without podman-compose "
                f"$-escaping — this splices {var_name}'s value into shell-command "
                "text at HOST-SIDE parse time instead of letting the container's "
                "own shell expand an environment-block variable at runtime "
                "(see this test's docstring for the measured injection)."
            )
            idx = script.find(unescaped, idx + 1)

    # ESCAPED forms must actually be present — the property above is
    # satisfied just as well by omitting the var from command: entirely,
    # which would silently drop the --allowed-hosts/--cors-allowed-origins
    # flag no matter what the operator sets the var to.
    assert f"$${{{var_name}:+" in script, (
        f"command: must reference {var_name} via the $-escaped optional-flag "
        f"form ($${{{var_name}:+...}}) so the CONTAINER's shell (not "
        "podman-compose) decides whether to pass the flag at all"
    )
    assert f"$${var_name}" in script, (
        f"command: must reference {var_name}'s value via the $-escaped form "
        f"($${var_name}) inside the quoted flag argument"
    )
