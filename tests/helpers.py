"""Test helpers shared across modules (importable as `helpers`)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from entropic_substrate.config import Config, DataConfig, ModelConfig, TrainConfig

__all__ = [
    "assert_no_foreign_mlflow_stack",
    "capture_container_logs",
    "load_compose",
    "mlflow_command",
    "mlflow_service_config",
    "tiny_config",
    "tiny_model_config",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The label podman/podman-compose stamps on every container of a compose
# service. Verified empirically (2026-08-14, podman 6.0.1 client / 5.8.3 server,
# podman-compose 1.6.0): `podman ps --filter
# label=com.docker.compose.service=mlflow --format '{{.Names}}'` matched exactly
# the one mlflow container of a live stack (0.31s with none up, 3.59s with one
# up) and matched NEITHER the stack's own postgres NOR the unrelated
# `agx-qdrant` container resident on this machine — i.e. the filter is specific
# enough to be a precondition and cheap enough to run on every fixture entry.
_MLFLOW_SERVICE_FILTER = "label=com.docker.compose.service=mlflow"


def tiny_model_config(**overrides: object) -> ModelConfig:
    """A CPU-sized :class:`ModelConfig`."""
    cfg = ModelConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        block_size=64,
        tokenizer="byte",
        tie_embeddings=True,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def load_compose() -> dict[str, Any]:
    """Parse ``compose.yaml`` fresh on every call (no caching — these are
    cheap, infrequent static-config checks, not a hot path).
    """
    text = (_REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        msg = (
            f"compose.yaml did not parse to a mapping (got {type(parsed).__name__}) — "
            "an empty or malformed compose.yaml would otherwise surface as an opaque "
            "KeyError/TypeError several calls downstream in mlflow_service_config()"
        )
        raise ValueError(msg)
    return parsed


def mlflow_service_config() -> dict[str, Any]:
    """``compose.yaml``'s ``services.mlflow`` block, parsed."""
    return load_compose()["services"]["mlflow"]


def mlflow_command() -> list[str]:
    """``compose.yaml``'s ``services.mlflow.command`` as a FLAT argv list.

    Parsed with a real YAML loader rather than string-sliced, so this stays
    correct regardless of how the surrounding comment block is reflowed.

    The list shape is itself the contract. An earlier revision wrapped this
    command in ``sh -c`` so the container's own shell could expand
    ``${VAR:+--flag "$VAR"}`` and make the allowlist flags OPTIONAL. That
    wrapper is gone: MLflow binds ``--allowed-hosts`` /
    ``--cors-allowed-origins`` to ``MLFLOW_SERVER_ALLOWED_HOSTS`` /
    ``MLFLOW_SERVER_CORS_ALLOWED_ORIGINS`` natively (click ``envvar=``,
    mlflow/utils/cli_args.py:256,270 in the pinned image), so the flags are
    never passed and nothing needs a shell. Asserting the flat shape here
    means a regression to any shell wrapper fails LOUDLY in one place
    instead of silently reintroducing a command-text interpolation site.
    """
    service = mlflow_service_config()
    command = service.get("command")
    assert command, "mlflow service must declare a command"
    assert isinstance(command, list), (
        f"mlflow's command must be a flat argv LIST, not a shell string; got {command!r}"
    )
    assert command[:2] != ["sh", "-c"], (
        "mlflow's command is wrapped in `sh -c` again. That wrapper existed only to "
        "make the allowlist flags optional, and it is exactly the command-TEXT "
        "interpolation site that allowed a measured shell injection. MLflow reads "
        "MLFLOW_SERVER_ALLOWED_HOSTS/MLFLOW_SERVER_CORS_ALLOWED_ORIGINS natively — "
        f"pass them via `environment:` instead; got {command!r}"
    )
    return [str(entry) for entry in command]


def _as_text(raw: str | bytes | None) -> str:
    """Normalise subprocess output that may arrive as bytes even in text mode.

    NOT defensive padding: ``subprocess.run(..., text=True, timeout=N)`` returns
    ``str`` on the success path but attaches BYTES to ``TimeoutExpired.stdout`` /
    ``.stderr`` on POSIX (CPython's ``Popen._communicate`` builds the partial
    output with ``b"".join(...)`` before the text-mode newline translation ever
    runs). capture_container_logs() reads both paths, so both types reach here.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def _render_streams(stdout: str, stderr: str) -> str:
    """Label the two streams rather than silently concatenating them.

    ``podman logs`` replays the container's stdout and stderr as SEPARATE pipes,
    so gluing them together would present a single chronological log that the
    data does not support. When only one stream has content — the common case for
    mlflow, which writes essentially everything to stderr — no heading is added
    and the output reads exactly as it does in a terminal.
    """
    out, err = stdout.strip(), stderr.strip()
    if out and err:
        return f"--- stdout ---\n{out}\n--- stderr ---\n{err}"
    return out or err


def capture_container_logs(container: str, *, timeout_s: float = 30.0) -> str:
    """Read a container's logs for DIAGNOSIS ONLY. **This never raises.**

    WHY A NEVER-RAISING HELPER, not a plain ``subprocess.run``. A diagnostic
    fires exactly when something already went wrong, so it runs at the machine's
    WORST moment — and if it can fail, it replaces the real exception with its
    own and the operator loses the evidence entirely. MEASURED (2026-08-13 live
    gate, worktree ``observability-sprint``): the previous inline
    ``subprocess.run([*compose, "logs", "mlflow"], timeout=30)`` inside an
    ``except`` block raised ``TimeoutExpired`` from within that block; the
    ``print`` of the captured log never executed, the chained original error was
    demoted to ``__context__``, and the fixture's ``finally`` then ran
    ``down -v`` — deleting the containers AND the volume. The only reason that
    failure was diagnosable at all is that podman's log driver on this machine is
    journald, so the container's lines survived outside the container.

    A timeout here is therefore TRUNCATION, not failure: CPython attaches the
    output captured so far to ``TimeoutExpired``, and this returns it with a
    marker naming what was lost.

    WHY ``podman logs <container>`` AND NOT ``podman compose logs <service>``.
    The container name is already deterministic for these stacks, and the
    compose wrapper costs a Python interpreter start plus a full compose-file
    re-parse for the same bytes. MEASURED back-to-back against one live stack,
    ONE CONTAINER AT A TIME (2026-08-14, 13-line mlflow log):

        settled VM (load1 0.86)   podman logs 0.41s   podman compose logs 2.14s
        VM at load1 4.4           podman logs 1.05s   podman compose logs 5.39s

    ``podman logs`` also splits the container's streams (stdout 214B / stderr
    882B for that same log, where ``podman compose logs`` merged all 1360B onto
    stdout — 1096B of content plus podman-compose's own ``[mlflow] |`` per-line
    prefix), so BOTH are captured here; reading only ``.stdout``, as the previous
    call site did, would silently drop most of an mlflow boot log. The price of
    the direct form is that the two streams arrive SEPARATELY rather than
    interleaved in real time, so they are emitted under explicit headings — an
    unlabelled concatenation would assert a chronological order that the data
    does not actually carry.

    NOT bounded with ``--tail``: cost does scale with volume (MEASURED: 11.47s
    for a 4896-line / 863KB container log on this VM) and the failures this
    fires on are the ones with the biggest logs — but truncating the HEAD would
    drop the boot/config lines the allowlist fixture greps for. The timeout is
    the bound instead, and it is no longer allowed to fail anything.
    """
    argv = ["podman", "logs", container]
    try:
        result = subprocess.run(  # nosec B603 B607
            argv, capture_output=True, text=True, check=False, timeout=timeout_s
        )
    except subprocess.TimeoutExpired as exc:
        partial = _render_streams(_as_text(exc.stdout), _as_text(exc.stderr))
        marker = (
            f"<`podman logs {container}` did not finish within {timeout_s}s — "
            "anything below is TRUNCATED>"
        )
        return f"{marker}\n{partial}" if partial else marker
    except OSError as exc:  # podman missing / not executable
        return f"<`podman logs {container}` could not run: {exc!r}>"
    rendered = _render_streams(_as_text(result.stdout), _as_text(result.stderr))
    if not rendered:
        return f"<`podman logs {container}` returned no output (exit {result.returncode})>"
    return rendered


def assert_no_foreign_mlflow_stack(project: str, *, timeout_s: float = 15.0) -> None:
    """Refuse to bring a live-gate stack up while ANOTHER one is running.

    THE INVARIANT THIS ENFORCES was previously asserted only in prose. Both
    scripts/run_container_test.sh's header and tests/test_mlflow_allowed_hosts.py's
    docstring state that the two stacks never run concurrently, and the script's
    control flow is correct as written — but nothing DETECTED a violation, so
    when one happened the symptom was a five-minute unexplained timeout.

    MEASURED violation (2026-08-13, from the podman VM's own journal + event log,
    project ``entropic-substrate-lm-mlflow-test-observability-sprint-676710117``):
    the phase-1 ``…_mlflow_1`` container ran 16:09:09–16:16:18 while the phase-2
    ``…-allowlist_mlflow_1`` container ran 16:11:04–16:13:46 — fully nested. Under
    that overlap the allowlist container took 77s to emit its first line (4s in the
    green re-run), logged "Security middleware enabled" and then NEVER reached
    "Application startup complete", so it could not report healthy and the
    fixture's 120s wait expired legitimately.

    That the budget is contention-bound and not simply too small is a controlled
    A/B on this machine, ONE STACK AT A TIME (2026-08-14, identical compose file,
    identical image, agx-qdrant resident for both arms):

        VM settled (load1 0.86)     up 8.64s   ->  healthy 29.64s
        VM at load1 4.4             up 10.39s  ->  healthy 135.71s

    4.6x on background load alone — so a co-resident second stack pushes a 30s
    boot past the 120s bound with room to spare. The fix is to refuse the
    condition immediately and say so, not to widen any budget.

    Fails LOUD, because this is a gate: a run that proceeds under contention can
    report red for reasons that have nothing to do with the code under test.
    A probe that cannot be RUN is reported and tolerated — an unverifiable
    precondition must not become a new flake source of its own.
    """
    try:
        result = subprocess.run(  # nosec B603 B607
            ["podman", "ps", "--filter", _MLFLOW_SERVICE_FILTER, "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"[live-gate] WARNING: could not verify no other mlflow stack is up: {exc!r}")
        return
    if result.returncode != 0:
        print(
            "[live-gate] WARNING: could not verify no other mlflow stack is up — "
            f"`podman ps` exited {result.returncode}: {result.stderr.strip()!r}"
        )
        return
    running = [name.strip() for name in result.stdout.splitlines() if name.strip()]
    foreign = sorted(name for name in running if not name.startswith(f"{project}_"))
    if foreign:
        msg = (
            f"REFUSING to start compose project {project!r}: another mlflow stack is "
            f"already running ({', '.join(foreign)}). This machine's podman VM is "
            "4 vCPU / 4 GB; two mlflow stacks contend and the second one's boot "
            "stretches past every timing bound in this suite (MEASURED: 29.64s to "
            "healthy on a settled VM vs 135.71s at load1 4.4). A run started here "
            "would fail for reasons unrelated to the code under test. Stop the other "
            "stack first, e.g.:\n"
            f"  podman compose -p <its-project> -f compose.yaml down -v --remove-orphans"
        )
        raise RuntimeError(msg)


def tiny_config(tmp_path: Path, **model_overrides: object) -> Config:
    """A complete CPU smoke-test :class:`Config` rooted at ``tmp_path``."""
    return Config(
        name="pytest-tiny",
        model=tiny_model_config(**model_overrides),
        data=DataConfig(
            source="synthetic",
            synthetic_bytes=80_000,
            batch_size=4,
            seq_len=32,
            val_fraction=0.1,
            num_eval_batches=2,
        ),
        train=TrainConfig(
            max_steps=5,
            lr=3e-3,
            warmup_steps=2,
            eval_interval=0,
            log_interval=1,
            save_interval=0,
            amp_dtype="off",
            device="cpu",
            results_dir=str(tmp_path / "results"),
            models_dir=str(tmp_path / "models"),
        ),
    )
