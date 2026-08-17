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

import re
import subprocess
from pathlib import Path

import pytest

from helpers import load_compose, mlflow_command, mlflow_service_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# A bcrypt hash's own format ($2a$/$2b$/$2y$ + cost + 53-char salt+digest) —
# used to grep this WHOLE TREE for an accidentally committed credential, not
# just compose.yaml. Hashed or not, a credential belongs only in the
# environment (this phase's NO CREDENTIALS COMMITTED invariant).
_BCRYPT_HASH_RE = re.compile(r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}")

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


# --------------------------------------------------------------- p4: caddy


def _caddy_service_config() -> dict[str, object]:
    return load_compose()["services"]["caddy"]


def test_caddy_service_exists_and_is_profile_gated() -> None:
    """The default `up -d --build` (phases 1 and 2, no service names) must
    NOT build/start caddy — VERIFIED against podman-compose 1.6.0 with a
    scratch two-service file: an unprofiled `config --services` listed only
    the unprofiled service; `--profile proxy config --services` listed
    both. Un-profiled, phases 1/2 would pay caddy's build+boot cost for
    nothing they exercise.
    """
    compose = load_compose()
    assert "caddy" in compose["services"], "compose.yaml must declare a caddy service (p4)"
    profiles = _caddy_service_config().get("profiles")
    assert profiles == ["proxy"], (
        f"caddy service must be gated behind profiles: ['proxy'] so the default "
        f"`up` (phases 1/2) never starts it; got {profiles!r}"
    )


def test_caddy_service_publishes_no_fixed_host_port() -> None:
    """Same concurrency hazard as mlflow's own port — see that test above."""
    ports = _caddy_service_config()["ports"]
    assert ports, "caddy service must publish its port somehow"
    for entry in ports:
        assert ":" not in str(entry), (
            f"port entry {entry!r} looks like a fixed host:container mapping — must "
            "stay a bare container port so podman assigns an ephemeral host port"
        )


def test_caddy_service_has_no_depends_on_mlflow() -> None:
    """MLflow's allowlist does an EXACT host:port string match (measured — see
    compose.yaml's own comment on this service), so the fixture must bring
    caddy up BEFORE mlflow to discover caddy's ephemeral port first. A
    compose-level `depends_on: mlflow` would force mlflow into the SAME `up`
    call as caddy and defeat that ordering.
    """
    depends_on = _caddy_service_config().get("depends_on")
    assert not depends_on, (
        f"caddy must not declare depends_on (found {depends_on!r}) — the live-gate "
        "fixture owns the boot order explicitly; see compose.yaml's comment"
    )


def test_caddy_credentials_come_from_environment_only() -> None:
    """The basic-auth user/hash must be sourced from ${VAR}, never a literal."""
    env_block = _caddy_service_config().get("environment") or {}
    for var in ("CADDY_BASIC_AUTH_USER", "CADDY_BASIC_AUTH_HASH"):
        assert var in env_block, f"caddy service must declare {var} under environment:"
        assert str(env_block[var]) == f"${{{var}}}", (
            f"{var} must be a bare ${{{var}}} interpolation with no default/literal "
            f"fallback; got {env_block[var]!r}"
        )


def _caddyfile_code_only(caddyfile: str) -> str:
    """Same-length copy with comments and ``{$VAR}`` placeholders blanked out.

    Offsets are preserved so positions found here index straight back into the
    original text. Two things must go before any structural analysis:

    - COMMENTS. This Caddyfile documents the basic_auth hazard at length in its
      own header, so a naive search finds ``basic_auth`` five times in prose
      before reaching the one occurrence that is actually configuration. MEASURED
      while writing this: offsets 349, 815, 1570, 1609 and 1984 are all comment
      text; only 3539 is the directive.
    - ``{$VAR}`` PLACEHOLDERS, because they contain braces. Left in, the depth
      counter closes the handle block early on
      ``{$CADDY_BASIC_AUTH_USER}`` and reports a span ending before the very
      directive it exists to contain.
    """
    masked = re.sub(r"\{\$[^}]*\}", lambda m: "x" * len(m.group(0)), caddyfile)
    return re.sub(r"#[^\n]*", lambda m: " " * len(m.group(0)), masked)


def _catch_all_handle_span(code: str) -> tuple[int, int]:
    """Byte span of the catch-all ``handle {`` block, found by brace matching.

    "Catch-all" is the ``handle`` with NO matcher; ``handle @health`` is the
    exempt one. Matching ``handle`` followed directly by ``{`` distinguishes the
    two without needing a Caddyfile parser. Takes the masked text from
    :func:`_caddyfile_code_only`.
    """
    match = re.search(r"handle\s*\{", code)
    assert match is not None, (
        "Caddyfile must contain a catch-all `handle {` block — a named-matcher "
        "`handle @health` alone cannot scope basic_auth away from /health"
    )
    depth = 0
    for index in range(match.end() - 1, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return match.start(), index + 1
    raise AssertionError("Caddyfile has unbalanced braces — cannot locate the handle block")


def test_caddyfile_exempts_health_from_basic_auth_via_matcher_not_ordering() -> None:
    """A Caddyfile's directives are NOT ordered by textual position — Caddy
    sorts by internal precedence, so a bare `basic_auth` anywhere in the
    file applies globally regardless of where it sits relative to a
    `handle /health` block (VERIFIED: `caddy adapt` on that naive shape put
    the authentication handler in routes[0] with NO path matcher at all).
    Only a named-matcher `handle`/`handle` pair (mutually exclusive, like a
    switch) actually scopes basic_auth away from /health.
    """
    caddyfile = (REPO_ROOT / "Caddyfile").read_text(encoding="utf-8")
    assert "@health path /health" in caddyfile, (
        "Caddyfile must define a named matcher for /health — a bare textual "
        "ordering does not scope basic_auth away from it (see this test's docstring)"
    )

    # STRUCTURAL, not a text partition. An earlier version of this test scanned
    # only the span BETWEEN "@health path /health" and the next "handle {", and
    # was MEASURED toothless by a guard-teeth flip: moving basic_auth to a bare
    # top-level directive — the exact historical regression this docstring
    # describes — sits BEFORE that span and was therefore invisible, leaving the
    # test green while `caddy adapt` showed routes[0] carrying an authentication
    # handler with no match key (i.e. /health authenticated too). The invariant
    # that actually holds is positional in the BLOCK sense: every basic_auth must
    # live inside the catch-all handle. That fails closed on placements nobody
    # has thought of yet, including ones above the matcher.
    code = _caddyfile_code_only(caddyfile)
    start, end = _catch_all_handle_span(code)
    occurrences = [m.start() for m in re.finditer(r"\bbasic_auth\b", code)]
    assert occurrences, "Caddyfile must configure basic_auth somewhere"
    for index in occurrences:
        assert start < index < end, (
            "basic_auth must appear ONLY inside the catch-all `handle` block. One at "
            f"offset {index} falls outside it (block spans {start}..{end}). A "
            "basic_auth outside that block is applied GLOBALLY by Caddy's directive "
            "sorting — including to /health, which a platform healthcheck cannot "
            "authenticate — regardless of where it sits textually."
        )


def test_caddyfile_never_overrides_the_forwarded_host_header() -> None:
    """reverse_proxy forwards the ORIGINAL Host header by default (VERIFIED
    against the pinned caddy:2.11-alpine image — see Caddyfile's own header
    comment). An explicit `header_up Host ...` override would substitute a
    fixed value for whatever the client actually sent, making
    MLFLOW_ALLOWED_HOSTS's public-domain entry a lie about what MLflow
    receives.
    """
    caddyfile = (REPO_ROOT / "Caddyfile").read_text(encoding="utf-8")
    assert "header_up Host" not in caddyfile, (
        "Caddyfile must not override the forwarded Host header — the default "
        "pass-through is what makes MLFLOW_ALLOWED_HOSTS's public-domain entry "
        "correspond to what MLflow actually receives"
    )


def test_caddyfile_upstream_is_env_parameterized_with_the_compose_default() -> None:
    """Both reverse_proxy sites use ``{$MLFLOW_UPSTREAM:mlflow:5000}`` — Caddy's
    native env placeholder, resolved from the CONTAINER environment at config
    load, with the local compose service name as the fallback.

    A deploy venue's private network names the tracking server something only
    the deploy environment knows (Railway: ``<service>.railway.internal:<port>``)
    while the local stack reaches it as ``mlflow:5000``. The placeholder is the
    same pure-data channel as the basic-auth credentials — no shell, no
    host-side interpolation site — and its parser splits on the FIRST colon
    only, so a default that itself contains ``:5000`` survives intact. MEASURED
    against the pinned caddy:2.11-alpine image (2026-08-17, ``caddy adapt``,
    one container at a time): variable unset -> upstream dial ``mlflow:5000``;
    MLFLOW_UPSTREAM=mlflow.railway.internal:5000 -> dial
    ``mlflow.railway.internal:5000``. Unset is therefore byte-identical to the
    pre-parameterized local behavior.

    BOTH sites must carry the placeholder — the @health handle and the
    catch-all. If they diverged, the auth-exempt health route and the
    authenticated API would proxy to DIFFERENT backends, and a platform
    healthcheck would attest liveness of a server no real request reaches.
    """
    caddyfile = (REPO_ROOT / "Caddyfile").read_text(encoding="utf-8")
    code = re.sub(r"#[^\n]*", "", caddyfile)  # strip comments, KEEP placeholders
    upstreams = re.findall(r"reverse_proxy\s+(\S+)", code)
    assert len(upstreams) == 2, (
        f"expected exactly two reverse_proxy sites (the @health handle and the "
        f"catch-all); found {len(upstreams)}: {upstreams!r}"
    )
    for upstream in upstreams:
        assert upstream == "{$MLFLOW_UPSTREAM:mlflow:5000}", (
            f"every reverse_proxy upstream must be the env placeholder "
            f"{{$MLFLOW_UPSTREAM:mlflow:5000}}; got {upstream!r}"
        )


def test_no_bcrypt_hash_literal_anywhere_in_the_tracked_tree() -> None:
    """NO CREDENTIALS COMMITTED, hashed or otherwise. Greps every git-tracked
    text file for a bcrypt-hash-shaped string — not just compose.yaml/Caddyfile,
    so a hash pasted into a test, a doc, or a stray fixture is caught too.
    """
    result = subprocess.run(  # nosec B603 B607
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        timeout=15,
    )
    offenders: list[str] = []
    for rel_path in result.stdout.splitlines():
        path = REPO_ROOT / rel_path
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if _BCRYPT_HASH_RE.search(text):
            offenders.append(rel_path)
    assert not offenders, (
        f"bcrypt-hash-shaped literal found in tracked file(s): {offenders} — "
        "credentials (hashed or not) must come from the environment only"
    )


def test_caddyfile_binds_a_bare_port_with_no_domain_literal() -> None:
    """NO PLACEHOLDER DOMAINS anywhere in the tree — the Caddyfile's site
    address must be a bare port, not a hardcoded hostname (real or
    railway.app-shaped). A real deploy's public domain is a Railway-supplied
    runtime value, out of scope for this phase to hardcode or invent.
    """
    caddyfile = (REPO_ROOT / "Caddyfile").read_text(encoding="utf-8")
    first_code_line = next(
        line for line in caddyfile.splitlines() if line.strip() and not line.strip().startswith("#")
    )
    assert first_code_line.strip().startswith(":"), (
        f"Caddyfile's site address ({first_code_line!r}) must be a bare :port, not a "
        "hardcoded hostname"
    )
