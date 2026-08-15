"""LIVE integration tests for the Caddy-fronted auth topology (phase p4).

Boots a THIRD podman-compose stack from the repo's REAL ``compose.yaml``
(``--profile proxy``, so ``caddy`` — otherwise gated off the default set —
is included alongside postgres and mlflow). Own project name
(``${MLFLOW_COMPOSE_PROJECT}-caddy``), own credentials, own teardown — never
concurrent with phase 1 (main stack) or phase 2 (allowlist stack), same
reasoning as ``test_mlflow_allowed_hosts.py``'s own module docstring: this
4-vCPU podman VM makes two live mlflow stacks a contention hazard, not
merely a resource nicety.

TWO-STEP BOOT ORDER, not a single ``up --profile proxy``. MLflow's
``MLFLOW_SERVER_ALLOWED_HOSTS``/``...CORS_ALLOWED_ORIGINS`` do an EXACT
``host[:port]`` STRING match (MEASURED against the pinned image: an
allowlist of ``localhost`` admitted ``Host: localhost`` but refused
``Host: localhost:15000``; an allowlist of ``localhost:15001`` admitted
only that exact string). ``build_sink()``'s real HTTP client
(mlflow-skinny's own requests session) derives its Host header from
``MLFLOW_TRACKING_URI`` automatically — this module has no hook to
override that per-request the way ``_request()`` below does for raw HTTP
calls — so acceptance #5 (the sink authenticating through the proxy)
needs mlflow's allowlist to already contain the EXACT ``localhost:<port>``
Caddy will be reachable on. That port is assigned by podman only once the
``caddy`` container is created, and mlflow's own allowlist is baked into
ITS environment at ITS OWN container creation — so this fixture:

  1. brings up postgres + caddy only (mlflow untouched — compose.yaml's
     caddy service declares no ``depends_on`` for exactly this reason);
  2. discovers caddy's ephemeral host port (``podman port``, same idiom
     ``run_container_test.sh`` already uses for mlflow's own bare port);
  3. THEN brings up mlflow, with ``MLFLOW_ALLOWED_HOSTS`` set to the fixed
     RFC 2606 public test domain (used by the raw-HTTP Host-override
     proofs below) PLUS the just-discovered ``localhost:<caddy-port>``
     (used by the sink e2e proof, which cannot override its own Host
     header).

Hostnames are RFC 2606 reserved (``*.example``) — never anything
railway.app-shaped, per this phase's NO PLACEHOLDER DOMAINS invariant.
Credentials (a fresh random username/password per run, hashed with the
SAME pinned ``caddy`` image's own ``caddy hash-password``) are generated
in-process and never written to disk or committed anywhere — NO
CREDENTIALS COMMITTED, hashed or otherwise.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import requests

from entropic_substrate.sinks import MlflowStartupError, build_sink
from entropic_substrate.train import Trainer
from helpers import (
    assert_no_foreign_mlflow_stack,
    capture_container_logs,
    discover_host_port,
    reap_healthcheck_orphans,
    teardown_compose_stack,
    tiny_config,
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

# The image compose.yaml pins for the caddy service — reused here to hash
# the throwaway password with the EXACT SAME tool/version Caddy itself will
# verify against at request time.
_CADDY_IMAGE = "docker.io/library/caddy:2.11-alpine"

# RFC 2606 reserved — never resolvable, never a real domain, never
# railway.app-shaped. PUBLIC_TEST_DOMAIN is what MLFLOW_ALLOWED_HOSTS is
# configured to admit for the "public domain" proof; FOREIGN_TEST_DOMAIN is
# deliberately absent from it. Distinct literals from
# tests/test_mlflow_allowed_hosts.py's own (allowlisted.host.test /
# evil.example) so a failure in one module's fixture can never be misread
# as touching the other's.
PUBLIC_TEST_DOMAIN = "mlflow-proxy.entropic-substrate-lm.example"
FOREIGN_TEST_DOMAIN = "evil-proxy.example"

_SEARCH_PATH = "/api/2.0/mlflow/experiments/search"


def _hash_password(password: str) -> str:
    """Hash ``password`` with the PINNED caddy image's own ``hash-password``
    subcommand — the same tool Caddy's ``basic_auth`` directive verifies
    against, so this can never drift from what the running proxy expects
    (verify-first #2: "whether that tooling exists in the image you
    choose" — confirmed present, default algorithm bcrypt).
    """
    result = subprocess.run(  # nosec B603 B607
        ["podman", "run", "--rm", _CADDY_IMAGE, "caddy", "hash-password", "--plaintext", password],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    hashed = result.stdout.strip()
    if not hashed:
        msg = f"`caddy hash-password` produced no output (stderr: {result.stderr.strip()!r})"
        raise RuntimeError(msg)
    return hashed


def _request(
    base_url: str,
    path: str,
    *,
    host: str | None = None,
    auth: tuple[str, str] | None = None,
    method: str = "GET",
    json_body: dict[str, object] | None = None,
    timeout: float = 15.0,
) -> requests.Response:
    headers = {"Host": host} if host is not None else {}
    return requests.request(
        method,
        base_url + path,
        headers=headers,
        auth=auth,
        json=json_body,
        timeout=timeout,
    )


def _expect_ok(
    base_url: str,
    path: str,
    *,
    host: str,
    auth: tuple[str, str],
    timeout_s: float = 90.0,
) -> requests.Response:
    """POST, retrying until 200 — same cold-start rationale as
    ``test_mlflow_allowed_hosts.py``'s own ``_expect_ok`` (this is a FRESH
    mlflow container behind its own fresh Postgres, so it pays the same
    one-time SQLAlchemy engine / first-query-shape compile tax). Not
    reused directly from that module: this one additionally carries HTTP
    Basic auth, which p3's mlflow has none of.
    """
    deadline = time.monotonic() + timeout_s
    last: str | int = "cold"
    while time.monotonic() < deadline:
        try:
            resp = _request(
                base_url,
                path,
                host=host,
                auth=auth,
                method="POST",
                json_body={"max_results": 1},
            )
            last = resp.status_code
            if resp.status_code == 200:
                return resp
        except requests.RequestException as exc:  # pragma: no cover - transient startup
            last = str(exc)
        time.sleep(2)
    msg = f"expected-200 call to {path} (host={host!r}) never settled (last observed: {last})"
    raise RuntimeError(msg)


@pytest.fixture(scope="module")
def caddy_stack() -> Iterator[tuple[str, int, str, str]]:
    """Bring up postgres + caddy + mlflow (in that boot order — see module
    docstring), a fresh basic-auth username/password, and the discovered
    Caddy host port. Yields ``(base_url, caddy_port, username, password)``.
    """
    base_project = os.environ.get("MLFLOW_COMPOSE_PROJECT", "")
    if not base_project:
        msg = (
            "MLFLOW_COMPOSE_PROJECT is not set — this fixture must run via "
            "`just container-test` (scripts/run_container_test.sh exports the "
            "per-worktree compose project name before pytest starts)."
        )
        raise RuntimeError(msg)
    # MLFLOW_COMPOSE_PROJECT_FAMILY (p5): see
    # tests/test_mlflow_allowed_hosts.py's identical fixture-setup comment —
    # same mechanism, this phase's own stack.
    family = os.environ.get("MLFLOW_COMPOSE_PROJECT_FAMILY") or None

    project = f"{base_project}-caddy"
    compose_file = str(REPO_ROOT / "compose.yaml")
    compose = ["podman", "compose", "-p", project, "-f", compose_file, "--profile", "proxy"]
    mlflow_container = f"{project}_mlflow_1"
    caddy_container = f"{project}_caddy_1"

    reap_healthcheck_orphans()
    assert_no_foreign_mlflow_stack(project, family=family)

    # Fresh per run, never committed, never printed. secrets.token_urlsafe
    # (CSPRNG-backed) — this is a throwaway credential for a throwaway
    # stack that is torn down (with its volume) at the end of this fixture,
    # but "throwaway" does not mean "predictable".
    username = f"probe-{secrets.token_hex(4)}"
    password = secrets.token_urlsafe(24)
    password_hash = _hash_password(password)

    env = os.environ.copy()
    env["CADDY_BASIC_AUTH_USER"] = username
    env["CADDY_BASIC_AUTH_HASH"] = password_hash

    try:
        try:
            # STEP 1: postgres + caddy only. mlflow is deliberately absent —
            # see the module docstring on why its allowlist can't be fixed
            # yet. Caddy's own healthcheck proxies to mlflow's /health and
            # will report unhealthy until step 3; that's fine, this step
            # only needs the container CREATED so its ephemeral port is
            # assigned (podman assigns the host-side port at container
            # creation, independent of the app inside ever becoming ready).
            subprocess.run(  # nosec B603 B607
                [*compose, "up", "-d", "--build", "postgres", "caddy"],
                check=True,
                cwd=REPO_ROOT,
                env=env,
                timeout=240,
            )
            caddy_port = discover_host_port(caddy_container, 8080)

            # STEP 2: now that caddy's port is known, fold BOTH the fixed
            # public test domain (used by the raw-HTTP Host-override proofs)
            # and the just-discovered localhost:<port> (used by the sink
            # e2e proof, which cannot override its own Host header) into
            # mlflow's allowlist before ITS container is created.
            env["MLFLOW_ALLOWED_HOSTS"] = f"{PUBLIC_TEST_DOMAIN},localhost:{caddy_port}"
            env["MLFLOW_CORS_ALLOWED_ORIGINS"] = f"https://{PUBLIC_TEST_DOMAIN}"
            subprocess.run(  # nosec B603 B607
                [*compose, "up", "-d", "--build", "mlflow"],
                check=True,
                cwd=REPO_ROOT,
                env=env,
                timeout=240,
            )
            wait_container_healthy(mlflow_container)
            # Caddy's own healthcheck proxies to mlflow/health — it can only
            # go healthy now that mlflow genuinely exists.
            wait_container_healthy(caddy_container)

            base_url = f"http://127.0.0.1:{caddy_port}"
            # Warm the REST API through Caddy with a real, authenticated,
            # DB-backed write before any test body runs — same cold-start
            # rationale as run_container_test.sh's own warm-up and p3's
            # _expect_ok, applied here through the proxy+auth path too.
            _expect_ok(base_url, _SEARCH_PATH, host=PUBLIC_TEST_DOMAIN, auth=(username, password))
        except Exception:
            # PRESERVE EVIDENCE ON SETUP FAILURE — same pattern as
            # test_mlflow_allowed_hosts.py's fixture, same reason: a
            # `finally`'s `down -v` below would otherwise destroy the only
            # diagnosis of why setup failed. capture_container_logs() never
            # raises, so it cannot replace this exception with its own.
            for container in (caddy_container, mlflow_container):
                print(
                    f"[caddy-fixture] SETUP FAILED — {container} logs:\n"
                    f"{capture_container_logs(container)}"
                )
            raise
        yield base_url, caddy_port, username, password
    finally:
        teardown_compose_stack(compose, cwd=REPO_ROOT, env=env)
        reap_healthcheck_orphans()


# ----------------------------------------------------------- acceptance #3
def test_unauthenticated_refused_authenticated_with_public_host_admitted(
    caddy_stack: tuple[str, int, str, str],
) -> None:
    """THE ONE-RUN PROOF (acceptance #3, all three legs).

    (a) unauthenticated -> 401: proves Caddy's basic_auth is in front of the
        API at all. Sent WITH the public-domain Host already attached
        (deliberately admitted by mlflow's OWN allowlist) so this isolates
        Caddy's auth layer specifically — GUARD-TEETH VERIFIED (2026-08-14):
        an earlier version of this call sent no Host override at all, so it
        landed on ``127.0.0.1:<port>`` — a host mlflow's OWN allowlist ALSO
        refuses regardless of Caddy auth. Removing Caddy's basic_auth block
        against THAT shape still 403'd (mlflow's guard, not Caddy's), so
        the perturbation looked like a pass. Pinning the Host to the
        admitted domain here means Caddy's guard is the ONLY thing that can
        produce this 401; removing it now genuinely flips this to 200 (RED
        pasted in the phase's evidence — see the commit).
    (b)+(c) ONE authenticated request that ALSO carries the PUBLIC-domain
        Host -> 200: THE POINT. Testing (a) and an authenticated request
        SEPARATELY would pass just as happily with MLFLOW_ALLOWED_HOSTS
        misconfigured (e.g. missing the public domain entirely) — only a
        single request that is BOTH authenticated AND carries the exact
        Host mlflow's allowlist must admit proves Caddy's auth and
        mlflow's allowlist work TOGETHER, not past each other.

    A same-run foreign-Host sanity anchor closes the loop the way
    test_mlflow_allowed_hosts.py's own "both halves" tests do: the
    identical credentials that succeed against the allowed domain must
    still be refused against a domain that was never added — otherwise
    this "proof" would be indistinguishable from the allowlist being off.
    """
    base_url, _port, username, password = caddy_stack

    unauthenticated = _request(
        base_url, _SEARCH_PATH, host=PUBLIC_TEST_DOMAIN, method="POST", json_body={"max_results": 1}
    )
    assert unauthenticated.status_code == 401, unauthenticated.text

    authenticated_public_host = _expect_ok(
        base_url, _SEARCH_PATH, host=PUBLIC_TEST_DOMAIN, auth=(username, password)
    )
    assert authenticated_public_host.status_code == 200, authenticated_public_host.text

    authenticated_foreign_host = _request(
        base_url,
        _SEARCH_PATH,
        host=FOREIGN_TEST_DOMAIN,
        auth=(username, password),
        method="POST",
        json_body={"max_results": 1},
    )
    assert authenticated_foreign_host.status_code == 403, authenticated_foreign_host.text
    assert "DNS rebinding" in authenticated_foreign_host.text, authenticated_foreign_host.text


# ----------------------------------------------------------- acceptance #4
def test_health_exempt_from_auth_but_api_still_requires_it(
    caddy_stack: tuple[str, int, str, str],
) -> None:
    """/health through Caddy answers UNAUTHENTICATED (a platform healthcheck
    cannot send credentials), while the API one route over still demands
    them — same asymmetry MLflow's own DNS-rebinding guard has for /health
    vs the API (test_mlflow_allowed_hosts.py), now proven one layer up, at
    Caddy, where it is Caddy's OWN basic_auth doing the exempting rather
    than mlflow's allowlist. THE FALSE-GREEN THIS PHASE EXISTS TO PREVENT:
    asserting only that /health answers would pass even if Caddy forwarded
    every request unauthenticated.
    """
    base_url, _port, _username, _password = caddy_stack

    health = _request(base_url, "/health")
    assert health.status_code == 200, health.text

    # Host pinned to the admitted public domain so this isolates Caddy's
    # auth layer specifically, same reasoning as the sibling test above.
    api_unauthenticated = _request(
        base_url, _SEARCH_PATH, host=PUBLIC_TEST_DOMAIN, method="POST", json_body={"max_results": 1}
    )
    assert api_unauthenticated.status_code == 401, api_unauthenticated.text


# ----------------------------------------------------------- acceptance #5
def test_sink_authenticates_through_caddy_and_fails_loud_without_credentials(
    caddy_stack: tuple[str, int, str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``src/entropic_substrate/sinks.py`` authenticating through the proxy
    end to end — not a stub client, the REAL ``mlflow.tracking.MlflowClient``
    path ``build_sink()`` constructs from ``MLFLOW_TRACKING_URI`` plus
    ``MLFLOW_TRACKING_USERNAME``/``MLFLOW_TRACKING_PASSWORD`` (the exact
    pair mlflow's own ``mlflow/utils/credentials.py`` reads — confirmed by
    grepping the INSTALLED package, verify-first #5 — and attaches as an
    HTTP Basic ``Authorization`` header in ``mlflow/utils/rest_utils.py``).

    AUTH MUST BE PROVEN BY ITS ABSENCE on the client side too: a second
    construction with the credential env vars deliberately unset must raise
    ``MlflowStartupError`` — fast, not after the 90s cold-start retry
    budget, because a 401 is a rejected REST call, not a read-timeout
    (``_is_cold_start`` only recognises transport-level timeout exceptions
    by name; an ``MlflowException`` wrapping a 401 is neither, so it aborts
    on the very first attempt).
    """
    base_url, caddy_port, username, password = caddy_stack
    tracking_uri = f"http://localhost:{caddy_port}"

    # --- credentials present: real write, through Caddy, for real ---
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", username)
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", password)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("ENTROPIC_MLFLOW_TINY", "1")

    cfg = tiny_config(tmp_path)
    cfg.name = "live-caddy-auth-e2e"
    sink = build_sink(cfg, tiny=True, results_path=tmp_path / "out.jsonl")
    trainer = Trainer(cfg, results_path=tmp_path / "out.jsonl", sink=sink)
    trainer.fit(verbose=False)

    mlflow_sink = sink.best_effort[0]  # type: ignore[attr-defined]
    assert mlflow_sink.failed_writes == 0, (
        "the sink authenticated through Caddy but still recorded failed writes — "
        "auth succeeded yet something else broke the run"
    )

    # Read the run back the SAME authenticated way an operator would, proving
    # this was not merely "the write didn't raise" but that MLflow actually
    # persisted it behind the proxy.
    run_lookup = _request(
        base_url,
        f"/api/2.0/mlflow/runs/get?run_id={mlflow_sink.run_id}",
        host=PUBLIC_TEST_DOMAIN,
        auth=(username, password),
    )
    assert run_lookup.status_code == 200, run_lookup.text

    # --- credentials ABSENT: must fail loud, at construction, before any
    # training work happens ---
    monkeypatch.delenv("MLFLOW_TRACKING_USERNAME", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_PASSWORD", raising=False)
    with pytest.raises(MlflowStartupError):
        build_sink(cfg, tiny=True, results_path=tmp_path / "out-noauth.jsonl")
