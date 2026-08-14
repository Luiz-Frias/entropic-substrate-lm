"""LIVE integration tests for the MLflow host allowlist (phase p3).

Boots a SECOND podman-compose stack from the repo's REAL ``compose.yaml`` —
not a reduced test-only fixture — with ``MLFLOW_ALLOWED_HOSTS`` /
``MLFLOW_CORS_ALLOWED_ORIGINS`` set in the subprocess environment. Reusing
the actual compose file means these tests exercise the real wiring: those
operator-facing names are interpolated into ``services.mlflow.environment``
as MLflow's own ``MLFLOW_SERVER_ALLOWED_HOSTS`` /
``MLFLOW_SERVER_CORS_ALLOWED_ORIGINS`` (click ``envvar=``,
mlflow/utils/cli_args.py:256,270 in the pinned image), which is what makes
the allowlist take effect with no CLI flag and no shell anywhere in
``command:``. Delete those two environment entries and this suite goes red
(see the phase's GUARD-TEETH requirement), because there would be no way to
pass a configured allowlist into the server at all.

Gated behind ``just container-test`` (skipped unless
``MLFLOW_COMPOSE_PROJECT`` is set — this module's own stack lookup needs the
project name itself, not just a tracking URL, so it gates on a different
env var than ``test_mlflow_live.py`` does; both are exported together by
run_container_test.sh, so there is no behavioral gap between the two
gates in practice. See that module for why a hardcoded fallback project
name is refused).
scripts/run_container_test.sh runs this module as its OWN phase, AFTER the
main postgres+mlflow stack has been torn down — never concurrently with it.
MEASURED: two full postgres+mlflow pairs on this repo's 4-vCPU podman VM
put the second stack's healthcheck within a few failed probes of its own
retry budget (interval 3s x retries 40) after 5+ minutes, i.e. a coin flip
between "eventually healthy" and permanently "unhealthy" — not a slow but
bounded delay a bigger client-side timeout could paper over. Running this
module solo (this module's default) keeps every timing bound below
measured solo behavior, with margin.

Hostnames used throughout are RFC 2606 reserved (``*.test``, ``*.example``)
— never anything railway.app-shaped, per this phase's NO PLACEHOLDER
DOMAINS invariant. The allowlist value is never committed anywhere; it is
constructed here, in the test process, and passed only as an environment
variable to a throwaway compose stack this module owns start-to-finish.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

from helpers import (
    assert_no_foreign_mlflow_stack,
    capture_container_logs,
    discover_host_port,
    reap_healthcheck_orphans,
    teardown_compose_stack,
    wait_container_healthy,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("MLFLOW_COMPOSE_PROJECT", "").strip(),
        reason="MLFLOW_COMPOSE_PROJECT not set — run via `just container-test`",
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[1]

# RFC 2606 reserved — never resolvable, never a real domain, never
# railway.app-shaped. ALLOWED_HOST is what MLFLOW_ALLOWED_HOSTS is set to
# for this stack; FOREIGN_HOST is deliberately absent from it.
ALLOWED_HOST = "allowlisted.host.test"
FOREIGN_HOST = "evil.example"
ALLOWED_ORIGIN = f"https://{ALLOWED_HOST}"
# Derived from FOREIGN_HOST, not a second literal: a hardcoded duplicate
# would silently stop matching the Host-header identity the moment someone
# edits FOREIGN_HOST alone, splitting "foreign" into two different domains
# across the Host-guard and CORS tests.
FOREIGN_ORIGIN = f"https://{FOREIGN_HOST}"

_SEARCH_PATH = "/api/2.0/mlflow/experiments/search"


def _request(
    base_url: str,
    path: str,
    *,
    host: str,
    origin: str | None = None,
    method: str = "GET",
    json_body: dict[str, object] | None = None,
    timeout: float = 15.0,
) -> requests.Response:
    headers = {"Host": host}
    if origin is not None:
        headers["Origin"] = origin
    return requests.request(
        method, base_url + path, headers=headers, json=json_body, timeout=timeout
    )


def _expect_ok(
    base_url: str,
    path: str,
    *,
    host: str,
    origin: str | None = None,
    timeout_s: float = 90.0,
) -> requests.Response:
    """POST, retrying until 200 — for calls this test module EXPECTS to
    succeed only (never for the 403/refused side: those are asserted with a
    single ``_request`` call because the security middleware short-circuits
    them before the DB-backed handler, so they're fast and deterministic —
    see the module docstring's cold-start measurement).

    Necessary, not cosmetic: MEASURED against this exact compose.yaml shape
    (a fresh Postgres + mlflow pair, cold — even running SOLO, this
    module's default shape, see the module docstring on why it never runs
    alongside the main stack), the first POST .../search after the
    container reports "healthy" took OVER 15s to respond, the second
    ~10.6s, the third ~2.5s — a monotonic decay matching the SAME cold-start
    tax scripts/run_container_test.sh's own warm-up already documents for
    the main stack (7.27s / 1.22s / 0.11s there): SQLAlchemy engine /
    connection-pool init and first query-shape compile, paid once per fresh
    Postgres. A single fixed-timeout request — including the fixture's own
    one-shot warm-up call — flakes for exactly this reason. This is the
    SAME retry-on-cold-start pattern already established in this codebase
    (run_container_test.sh's own 45-attempt warm-up loop), applied to every
    expected-200 call in this module rather than only the fixture's first
    one. Not "raise the timeout" (forbidden — see sinks.py/compose.yaml on
    why bigger numbers are not a fix): the per-attempt timeout stays fixed;
    only the RETRY budget is generous, and only for calls proven to
    eventually settle.
    """
    deadline = time.monotonic() + timeout_s
    last: str | int = "cold"
    while time.monotonic() < deadline:
        try:
            resp = _request(
                base_url,
                path,
                host=host,
                origin=origin,
                method="POST",
                json_body={"max_results": 1},
            )
            last = resp.status_code
            if resp.status_code == 200:
                return resp
        except requests.RequestException as exc:  # pragma: no cover - transient startup
            last = str(exc)
        time.sleep(2)
    msg = f"expected-200 call to {path} (host={host!r}, origin={origin!r}) never settled (last observed: {last})"
    raise RuntimeError(msg)


@pytest.fixture(scope="module")
def allowlisted_mlflow_url() -> Iterator[str]:
    """Bring up a second, independent compose stack with a configured
    allowlist; discover its ephemeral port; tear it down unconditionally.

    Independent PROJECT NAME (suffixed off the main stack's, itself derived
    per-worktree by run_container_test.sh — see that script's own comment
    on why a fixed name is a cross-run hazard) so this stack's containers,
    volumes, and postgres data can never fuse with the main
    `just container-test` stack or with another worktree's run.
    """
    base_project = os.environ.get("MLFLOW_COMPOSE_PROJECT", "")
    if not base_project:
        msg = (
            "MLFLOW_COMPOSE_PROJECT is not set — this fixture must run via "
            "`just container-test` (scripts/run_container_test.sh exports the "
            "per-worktree compose project name before pytest starts)."
        )
        raise RuntimeError(msg)

    project = f"{base_project}-allowlist"
    compose = ["podman", "compose", "-p", project, "-f", str(REPO_ROOT / "compose.yaml")]
    container = f"{project}_mlflow_1"

    env = os.environ.copy()
    # THE point of this fixture: configure a real allowlist through the
    # SAME env-var-driven path an operator would use for a Railway deploy,
    # then prove both halves — configured host admitted, everything else
    # still refused. Never '*': see compose.yaml's own comment on why a
    # wildcard default is forbidden regardless of convenience.
    env["MLFLOW_ALLOWED_HOSTS"] = ALLOWED_HOST
    env["MLFLOW_CORS_ALLOWED_ORIGINS"] = ALLOWED_ORIGIN

    # PRECONDITION, checked before anything is created: this module's whole
    # timing model assumes it is the only mlflow stack on the machine (see the
    # module docstring). That was documented but UNENFORCED, and a run where it
    # silently did not hold cost five minutes and produced no usable diagnosis.
    # Raising here, before `up`, means nothing was created and there is nothing
    # for the outer `finally` to tear down.
    # Reap BEFORE the contention check, not after: an orphaned timer is not a
    # running stack, so the check would pass while the machine stays slow.
    reap_healthcheck_orphans()
    assert_no_foreign_mlflow_stack(project)

    try:
        try:
            # `up` itself belongs INSIDE this try, not before it: `check=True`
            # raises CalledProcessError and a hung build raises
            # TimeoutExpired — either one, before this fixture reworked to
            # put `up` here, skipped the outer `finally`'s `down -v
            # --remove-orphans` entirely, leaking a partially-started stack
            # (containers/network/volume) that starves the NEXT run exactly
            # the way the module docstring's contention measurement
            # describes. `down -v --remove-orphans` is safe to run against a
            # stack that never fully came up (check=False below already
            # tolerates a stack that never existed at all).
            subprocess.run(  # nosec B603 B607
                [*compose, "up", "-d", "--build"], check=True, cwd=REPO_ROOT, env=env, timeout=240
            )
            wait_container_healthy(container)
            port = discover_host_port(container)
            base_url = f"http://127.0.0.1:{port}"
            _expect_ok(base_url, _SEARCH_PATH, host=ALLOWED_HOST)
        except Exception:
            # PRESERVE EVIDENCE ON SETUP FAILURE: a red run that never
            # yields tears itself down in the outer `finally` before any
            # test body gets to inspect anything — the exact blindness that
            # made an earlier round's dual-stack-contention timeout
            # undiagnosable until re-measured by hand. Dump the mlflow
            # container's full log to stdout (pytest attaches captured
            # stdout to whichever test triggered fixture setup) so a future
            # red run carries its own diagnosis.
            #
            # A RETRACTION, because this file's comments are treated as
            # evidence. The previous revision called `subprocess.run([*compose,
            # "logs", "mlflow"], ..., timeout=30)` inline here and reasoned that
            # `check=False` was enough to keep it from masking the real
            # exception. `check=False` suppresses a NON-ZERO EXIT; it does
            # nothing about a TIMEOUT. MEASURED on the 2026-08-13 live gate:
            # that call raised `subprocess.TimeoutExpired` from inside this
            # `except` block, so the `print` below never ran, the genuine setup
            # error was demoted to `__context__`, and the outer `finally`'s
            # `down -v` then destroyed the container and volume. The diagnostic
            # became the reported failure and deleted the evidence for the real
            # one. capture_container_logs() cannot raise at all — see its
            # docstring for the measured cost figures behind that shape.
            print(
                f"[allowlist-fixture] SETUP FAILED — mlflow container logs "
                f"({container}):\n{capture_container_logs(container)}"
            )
            # Recovery path, load-bearing rather than decorative: the outer
            # `finally` below removes the container and its volume, but podman's
            # log driver on this machine is journald (`podman info --format
            # '{{.Host.LogDriver}}'`), so the container's own lines OUTLIVE that
            # teardown. This is how the 2026-08-13 failure was diagnosed after
            # the fact; printing it means the next operator does not have to
            # rediscover it.
            print(
                "[allowlist-fixture] container logs survive the teardown below in "
                f"the podman VM's journal — recover with:\n"
                f'  podman machine ssh "sudo journalctl --no-pager CONTAINER_NAME={container}"'
            )
            raise
        # Evidence, not just a side effect: print the server's own boot line
        # confirming the allowlist/CORS config actually landed. pytest shows
        # captured stdout on any failing test in this module, so this is a
        # cheap, always-available anchor for "was the flag even applied" —
        # separate from whether the HTTP-level assertions below pass.
        # Same never-raising helper as the failure path above, for the same
        # reason one level less obvious: this call sits on the GREEN path, so a
        # stall here would fail a run in which nothing is actually wrong. It is
        # an evidence nicety; it must never be able to decide the gate.
        for line in capture_container_logs(container).splitlines():
            if "Security middleware enabled" in line:
                print(f"[allowlist-fixture] {line}")
        yield base_url
    finally:
        # Never raises — see teardown_compose_stack's docstring. Raising from a
        # `finally` replaces whatever error the run was actually reporting, and
        # this exact call did so: a genuine "did not report healthy within
        # 120.0s" reached the operator as a TimeoutExpired on `podman compose`.
        teardown_compose_stack(compose, cwd=REPO_ROOT, env=env)
        # Drop this stack's healthcheck timer if podman left it behind. Done
        # AFTER the down, because that is when the container disappears and the
        # timer is orphaned; done at all, because an orphan re-fires every 3s
        # and slows every subsequent run (see the helper for the load figures).
        reap_healthcheck_orphans()


# --------------------------------------------------------------- acceptance #3
def test_configured_host_allowed_and_foreign_host_still_refused(
    allowlisted_mlflow_url: str,
) -> None:
    """THE POINT, both halves in one run: a test that only checks the
    allowed side would pass just as happily with the guard disabled."""
    allowed = _expect_ok(allowlisted_mlflow_url, _SEARCH_PATH, host=ALLOWED_HOST)
    assert allowed.status_code == 200, allowed.text

    foreign = _request(
        allowlisted_mlflow_url,
        _SEARCH_PATH,
        host=FOREIGN_HOST,
        method="POST",
        json_body={"max_results": 1},
    )
    assert foreign.status_code == 403, foreign.text
    assert "DNS rebinding" in foreign.text, foreign.text


# --------------------------------------------------------------- acceptance #4
def test_health_endpoint_exempt_from_allowlist_for_foreign_host(
    allowlisted_mlflow_url: str,
) -> None:
    """/health is EXEMPT from the Host guard even with a custom allowlist
    configured — this is the exact asymmetry that makes a Railway
    healthcheck report HEALTHY while the API and UI both 403."""
    resp = _request(allowlisted_mlflow_url, "/health", host=FOREIGN_HOST)
    assert resp.status_code == 200, resp.text

    # Sanity anchor for the asymmetry itself, in the SAME run: the identical
    # foreign Host header that /health accepts must still be refused by the
    # real API — otherwise this "exemption" test would be indistinguishable
    # from the guard being off entirely.
    api_with_foreign_host = _request(
        allowlisted_mlflow_url,
        _SEARCH_PATH,
        host=FOREIGN_HOST,
        method="POST",
        json_body={"max_results": 1},
    )
    assert api_with_foreign_host.status_code == 403, api_with_foreign_host.text


# ------------------------------------------------------- CORS (measured, p3)
def test_cors_allowed_origins_permits_configured_origin_and_blocks_foreign(
    allowlisted_mlflow_url: str,
) -> None:
    """Host header alone is not sufficient for a real browser: an Origin
    header MLflow's CORS middleware doesn't recognize is blocked even when
    the Host is on the allowlist (measured — a real browser hitting the UI
    sends Origin on POST/fetch calls, same-origin or not). Pins that
    MLFLOW_CORS_ALLOWED_ORIGINS actually does something, not just that it's
    wired through to the CLI.
    """
    allowed_origin = _expect_ok(
        allowlisted_mlflow_url, _SEARCH_PATH, host=ALLOWED_HOST, origin=ALLOWED_ORIGIN
    )
    assert allowed_origin.status_code == 200, allowed_origin.text

    foreign_origin = _request(
        allowlisted_mlflow_url,
        _SEARCH_PATH,
        host=ALLOWED_HOST,
        origin=FOREIGN_ORIGIN,
        method="POST",
        json_body={"max_results": 1},
    )
    assert foreign_origin.status_code == 403, foreign_origin.text
