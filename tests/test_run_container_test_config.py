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
