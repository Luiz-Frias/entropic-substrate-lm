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

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]


def _compose() -> dict[str, Any]:
    text = (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_mlflow_service_publishes_no_fixed_host_port() -> None:
    """FIX 2(b): a "host:container" mapping (e.g. the old "5001:5000") is a
    concurrency hazard — two worktrees running ``just container-test`` at
    once would collide on the same host port. The compose short-syntax for
    "publish this container port to an EPHEMERAL host port" is the bare
    container port with no colon (verified against the installed
    podman-compose: a scratch stack with ``ports: ["5432"]`` came up as
    ``0.0.0.0:<random>->5432/tcp``, discoverable via ``podman port``).
    """
    ports = _compose()["services"]["mlflow"]["ports"]
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
    compose = _compose()
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
