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

from helpers import load_compose, mlflow_command, mlflow_service_config

# MLflow's OWN env var names (click `envvar=`, mlflow/utils/cli_args.py:256,270
# in the pinned ghcr.io/mlflow/mlflow:v3.15.0). Setting these is equivalent to
# passing --allowed-hosts / --cors-allowed-origins, with no flag and no shell.
NATIVE_ALLOWLIST_VARS = {
    "MLFLOW_SERVER_ALLOWED_HOSTS": "MLFLOW_ALLOWED_HOSTS",
    "MLFLOW_SERVER_CORS_ALLOWED_ORIGINS": "MLFLOW_CORS_ALLOWED_ORIGINS",
}

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


def test_allowlist_uses_mlflows_native_env_vars_not_command_flags() -> None:
    """The allowlist reaches MLflow as ENVIRONMENT, never as command text.

    MLflow binds ``--allowed-hosts`` / ``--cors-allowed-origins`` to
    ``MLFLOW_SERVER_ALLOWED_HOSTS`` / ``MLFLOW_SERVER_CORS_ALLOWED_ORIGINS``
    through click's native ``envvar=`` (mlflow/utils/cli_args.py:256,270,
    read inside the pinned ghcr.io/mlflow/mlflow:v3.15.0 image). So the
    flags are unnecessary, and with them goes the ``sh -c`` wrapper that
    existed only to make them optional.

    That deletion is a SECURITY improvement, not merely a simplification.
    The wrapper was the one place in the entire compose/Containerfile
    surface where an externally-sourced value was spliced into shell-command
    TEXT. With the old single-``$`` shape that was a MEASURED injection:
    ``MLFLOW_ALLOWED_HOSTS='foo.test"; touch /tmp/PWNED ; echo "'`` rendered
    ``--allowed-hosts "foo.test"; touch /tmp/PWNED ; echo ""`` into the
    startup script at ``podman compose config`` time. The escaped
    (``$$VAR``) form fixed that by deferring expansion to the container's
    shell — but a correctly-escaped interpolation site is still an
    interpolation site. Passing these as ``environment:`` entries removes
    the site itself: podman hands the value to the container's ``exec()``
    environ array, where nothing ever re-parses it as shell.

    Empty is SAFE and means "unset". Measured against the pinned image with
    both vars set to "": the server boots in ~7s logging "Security
    middleware enabled with default settings (localhost-only)" — identical
    to omitting them — so the ``:-`` default below preserves local
    behaviour exactly, with no conditional-flag machinery.
    """
    env_block = mlflow_service_config().get("environment") or {}

    for native, operator_facing in NATIVE_ALLOWLIST_VARS.items():
        assert native in env_block, (
            f"{native} must be declared under services.mlflow.environment — that is "
            "how the allowlist reaches MLflow now that no CLI flag is passed"
        )
        # The OPERATOR-facing name is what gets interpolated into it, so the
        # documented interface (and .env, and the live-test fixtures) keep
        # working. A hardcoded value here would also silently ignore whatever
        # the operator set at deploy time.
        assert f"${{{operator_facing}" in str(env_block[native]), (
            f"{native} must take its value from ${{{operator_facing}}} so the "
            f"operator-facing variable name stays {operator_facing}; got "
            f"{env_block[native]!r}"
        )


def test_mlflow_command_has_no_interpolation_site_at_all() -> None:
    """No ``$`` anywhere in ``command:`` — the injection vector is GONE.

    Stronger than the escaping check it replaces. That one asserted every
    reference was ``$``-escaped correctly, which is a property you can get
    wrong in a new way each time someone edits the script. This asserts
    there is nothing to escape: the command is a flat argv list of literal
    arguments, so no host-side interpolation and no container-side shell
    expansion can occur, correctly or otherwise.

    ``mlflow_command()`` separately rejects a regression to ``sh -c``.
    """
    command = mlflow_command()

    for entry in command:
        assert "$" not in entry, (
            f"command entry {entry!r} contains a '$'. compose.yaml's command: must "
            "hold only literal arguments — any variable reference here is a "
            "host-side interpolation site (podman-compose substitutes at `config`/"
            "`up` parse time) or a container-side shell expansion, and splicing an "
            "externally-set value into command text is the measured injection this "
            "shape exists to make impossible. Pass values via `environment:`."
        )

    # The flags themselves must be absent: passing them AND the env vars would
    # reintroduce the empty-value edge cases the native binding avoids.
    for flag in ("--allowed-hosts", "--cors-allowed-origins"):
        assert flag not in command, (
            f"{flag} is passed on the command line again. It is redundant with the "
            "MLFLOW_SERVER_* environment variables and reintroduces the "
            "optional-flag problem that required a shell wrapper."
        )
