#!/usr/bin/env python3
"""FOR-754 port: the deterministic exit-gate runner — AGX's ``run-exit-gate.ts``
(``/tmp/entropic-port-ref-04a8fc9c/run-exit-gate.ts``) ported to Python.

The agent is a COURIER, never a witness: this tool executes the gate command
itself, captures the exit code from the process (not from prose), reads the
git tip before AND after the run (mid-gate tip movement -> STALE_TIP), checks
tree cleanliness (unfolded auto-fixes -> DIRTY_TREE), and mints the
certificate via :mod:`exit_gate`'s typestate core. Output is EXACTLY one JSON
object on stdout; process exit is 0 iff a certificate was minted, 1 otherwise.

Round-2 review note: this file is what closes the round-1/round-2 finding that
``scripts/exit_gate.py`` alone leaves "nothing in this repo can PRODUCE
evidence, only judge it." ``exit_gate.py``'s own docstring has been updated to
stop claiming this CLI is out of scope / a future follow-on — it exists now,
here, ported test-first against the same 14 cases (11 CLI + 3 isolation) that
``exit-gate.test.ts`` pins for ``run-exit-gate.ts``. See
``tests/test_run_exit_gate.py``.

ENV QUARANTINE (core.worktree incident, 2026-07-13): every child this runner
spawns -- its own git reads AND the gate command -- gets a sanitized env with
all ``GIT_*`` variables stripped. Hook runners (prek) hand hooks
``GIT_DIR`` + ``GIT_WORK_TREE`` + ``GIT_INDEX_FILE``; inherited, they
re-target git at the HOST repo (a ``git init`` under them writes
``core.worktree`` into the SHARED ``.git/config`` of a linked worktree's
host -- the exact poisoning that burned the conductor). The repo this runner
certifies is ALWAYS the one resolved from its invoking directory (Python, like
Node, defaults every subprocess call to the parent process's cwd when no
``cwd=`` is passed -- we rely on that rather than tracking cwd ourselves,
matching ``run-exit-gate.ts``).

Usage::

    python scripts/run_exit_gate.py --cmd "RUN_LIVE_LLM_TESTS= prek run --all-files" \\
        [--expect-tip <sha>] [--out <path>]

PORTABILITY: no AGX facts -- the gate command always arrives via ``--cmd``.

DOCUMENTED DIVERGENCE from ``run-exit-gate.ts`` (not silent, flagged per
round-2 review instruction 5 -- "no unforced divergence without a name"): the
TS reference's ``spawnSync`` maps a killed-by-timeout gate child (``status:
null``) to evidence with ``exitCode: -1`` and lets it flow through the normal
certify path, where it surfaces as a plain ``NONZERO_EXIT``. This port instead
lets Python's ``subprocess.TimeoutExpired`` propagate to :func:`_run`'s
top-level handler, which reports it as ``RUNNER_ERROR`` -- a deliberately
LOUDER failure for an infrastructure-level timeout (the gate command hung for
15 minutes) than folding it into "your gate command failed", per this
project's own fail-loud-on-measurement-paths rule. Untested by the ported
contract (``exit-gate.test.ts`` has no timeout case either), so this is a free
design choice, not a fidelity gap.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess  # nosec B404 -- gate cmd via sh -c by design; see _git/_collect_evidence
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

from exit_gate import (
    CertifyError,
    DirtyTree,
    EmptyTail,
    ExitGateEvidence,
    MissingEvidence,
    NonzeroExit,
    RunnerCertShape,
    StaleTip,
    certify_exit_gate,
)

_GATE_TIMEOUT_S: Final = 900  # matches run-exit-gate.ts's GATE_TIMEOUT_MS = 900_000
_TAIL_MAX_CHARS: Final = 4000

# Sanitized child env: everything except GIT_* (subtraction is the isolation --
# additive overrides proved insufficient twice in the FOR-754 incident). Computed
# once at import time, mirroring run-exit-gate.ts's module-level SAFE_ENV --
# os.environ (like process.env) doesn't change during this script's lifetime.
_SAFE_ENV: Final[dict[str, str]] = {
    key: value for key, value in os.environ.items() if not key.startswith("GIT_")
}


class _RunnerError(Exception):
    """A structured runner-level failure -- reported as ``RUNNER_ERROR``, never
    a raw stack trace. Mirrors ``run-exit-gate.ts``'s top-level try/catch:
    every crash (bad flags, non-git cwd, fs failure) becomes exactly one JSON
    object on stdout, never a Python traceback."""


@dataclass(frozen=True, slots=True)
class _CliArgs:
    cmd: str | None
    expect_tip: str | None
    out: str | None
    timeout_s: int


_KNOWN_FLAGS: Final = ("--cmd", "--expect-tip", "--out", "--timeout-s")


def _parse_args(argv: list[str]) -> _CliArgs:
    """Deliberately NOT ``argparse``: ``ArgumentParser.error()`` calls
    ``sys.exit(2)`` directly for an unrecognized argument regardless of the
    ``exit_on_error`` constructor flag (that flag only governs errors raised
    *inside* actions, not the unrecognized-argument path) -- which would print
    to stderr and hard-exit with code 2, never reaching this module's
    structured-JSON contract. A hand-rolled parser over 3 known ``--flag
    value`` pairs raises a catchable exception instead, exactly like
    ``run-exit-gate.ts`` relying on ``node:util``'s ``parseArgs`` throwing on
    an unknown option and letting the top-level try/catch turn it into
    ``RUNNER_ERROR``."""
    values: dict[str, str] = {}
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag not in _KNOWN_FLAGS:
            msg = f"unrecognized option '{flag}'"
            raise _RunnerError(msg)
        if i + 1 >= len(argv):
            msg = f"option '{flag}' expects a value"
            raise _RunnerError(msg)
        values[flag] = argv[i + 1]
        i += 2
    return _CliArgs(
        cmd=values.get("--cmd"),
        expect_tip=values.get("--expect-tip"),
        out=values.get("--out"),
        timeout_s=_parse_timeout(values.get("--timeout-s")),
    )


def _parse_timeout(raw: str | None) -> int:
    """``--timeout-s`` overrides :data:`_GATE_TIMEOUT_S` (a positive integer of
    seconds). It exists so the timeout path is black-box testable in seconds
    rather than the production 900 s; the harness never passes it."""
    if raw is None:
        return _GATE_TIMEOUT_S
    try:
        value = int(raw)
    except ValueError as exc:
        msg = f"--timeout-s expects a positive integer of seconds, got {raw!r}"
        raise _RunnerError(msg) from exc
    if value <= 0:
        msg = f"--timeout-s expects a positive integer of seconds, got {raw!r}"
        raise _RunnerError(msg)
    return value


def _git(*args: str) -> str:
    # nosec B603 B607 -- "git" is deliberately resolved from PATH, not an
    # absolute path: the quarantined _SAFE_ENV (GIT_* stripped) is what must
    # govern which repo this resolves against, not a hardcoded interpreter
    # location. shell=True is never used; args are a fixed argv list, never
    # interpolated into a shell string.
    proc = subprocess.run(  # nosec B603 B607
        ["git", *args],
        capture_output=True,
        text=True,
        env=_SAFE_ENV,
    )
    if proc.returncode != 0:
        msg = f"git {' '.join(args)} failed: {proc.stderr}"
        raise _RunnerError(msg)
    return proc.stdout.strip()


_GROUP_KILL_GRACE_S: Final = 2.0


def _terminate_process_group(proc: subprocess.Popen[str]) -> list[int]:
    """FOR-1528 (measured 2026-09-21): ``subprocess.run(timeout=)`` kills only
    the direct ``sh`` child, so ``just -> prek -> pytest -> xdist`` survived a
    900 s timeout as 11 orphans that raced the retry. The gate is spawned as
    its own session leader (``start_new_session=True``), so its pgid is
    ``proc.pid`` and one signal reaches every descendant: SIGTERM, a grace
    period, then SIGKILL. Returns the pids still alive in the group afterwards
    (empty when the group is gone), measured with ``pgrep -g`` so the caller
    names survivors instead of assuming."""
    pgid = os.getpgid(proc.pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=_GROUP_KILL_GRACE_S)
        except subprocess.TimeoutExpired:
            continue
        if not _group_survivors(pgid):
            break
    proc.wait()
    return _group_survivors(pgid)


def _group_survivors(pgid: int) -> list[int]:
    # nosec B603 B607 -- fixed argv, no shell; pgrep resolved from PATH under
    # the same quarantined env as every other subprocess here.
    scan = subprocess.run(  # nosec B603 B607
        ["pgrep", "-g", str(pgid)], capture_output=True, text=True, env=_SAFE_ENV
    )
    return [int(line) for line in scan.stdout.split() if line.isdigit()]


def _collect_evidence(cmd: str, timeout_s: int = _GATE_TIMEOUT_S) -> ExitGateEvidence:
    """Execute the gate and assemble evidence -- every field machine-captured,
    never transcribed by an agent."""
    # cmd is the operator-supplied gate command, run via a shell by design
    # (mirrors run-exit-gate.ts's spawnSync("sh", ["-c", cmd])) -- "sh" is
    # deliberately resolved from PATH, not an absolute path, so the
    # quarantined _SAFE_ENV governs resolution the same way it does for
    # _git() above. shell=True is never used; args are a fixed 3-element
    # argv list, never an interpolated shell string.
    proc = subprocess.Popen(  # nosec B603 B607
        ["sh", "-c", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_SAFE_ENV,
        start_new_session=True,  # FOR-1528: the gate is its own process group
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        survivors = _terminate_process_group(proc)
        detail = f" (survivors after group kill: {survivors})" if survivors else ""
        msg = f"Command {['sh', '-c', cmd]!r} timed out after {timeout_s} seconds{detail}"
        raise _RunnerError(msg) from exc
    combined = f"{stdout}{stderr}"
    tail = combined[-_TAIL_MAX_CHARS:] if len(combined) > _TAIL_MAX_CHARS else combined
    return ExitGateEvidence(
        cmd=cmd,
        exit_code=proc.returncode,
        tested_tip=_git("rev-parse", "HEAD"),
        tail=tail,
        dirty=_git("status", "--porcelain") != "",
    )


def _evidence_json(evidence: ExitGateEvidence) -> dict[str, object]:
    """camelCase wire shape -- the harness parses ``testedTip``/``tailDigest``
    literally (see exit_gate.py's module docstring)."""
    return {
        "cmd": evidence.cmd,
        "exitCode": evidence.exit_code,
        "testedTip": evidence.tested_tip,
        "tail": evidence.tail,
        "dirty": evidence.dirty,
    }


def _certify_error_json(error: CertifyError) -> dict[str, object]:
    match error:
        case MissingEvidence():
            return {"kind": "MISSING_EVIDENCE"}
        case NonzeroExit(exit_code=code):
            return {"kind": "NONZERO_EXIT", "exitCode": code}
        case StaleTip(expected=expected, got=got):
            return {"kind": "STALE_TIP", "expected": expected, "got": got}
        case DirtyTree():
            return {"kind": "DIRTY_TREE"}
        case EmptyTail():
            return {"kind": "EMPTY_TAIL"}
    # No `case _:` fallback: CertifyError is a closed 5-member union (pinned by
    # test_certify_errors_form_a_closed_five_member_union) and basedpyright
    # proves this match exhaustive over it -- a stronger, compile-time version
    # of the "unreachable" guard a runtime `case _: raise` would only assert.
    # Adding one back would itself be a basedpyright error (the arm becomes
    # unreachable: reportUnnecessaryComparison on a `Never`-typed subject).


def _report_json(
    *,
    ok: bool,
    cert: RunnerCertShape | None = None,
    error: dict[str, object] | None = None,
    evidence: ExitGateEvidence | None = None,
) -> dict[str, object]:
    """Builds the RunnerReportShape wire object -- optional keys are OMITTED
    entirely when not provided (never set to ``null``), matching JS
    ``JSON.stringify`` dropping ``undefined``-valued keys. This is instruction
    6's contract: a fail-fast report (MISSING_EVIDENCE, pre-run STALE_TIP)
    carries no ``evidence`` key at all."""
    report: dict[str, object] = {"ok": ok}
    if cert is not None:
        report["cert"] = {
            "cmd": cert.cmd,
            "certifiedTip": cert.certified_tip,
            "tailDigest": cert.tail_digest,
        }
    if error is not None:
        report["error"] = error
    if evidence is not None:
        report["evidence"] = _evidence_json(evidence)
    return report


def _emit(ok: bool, report: dict[str, object], out_path: str | None) -> NoReturn:
    text = json.dumps(report, indent=2)
    if out_path is not None:
        try:
            Path(out_path).write_text(text)
        except OSError:
            pass  # stdout stays the contract; a failed --out write must not mask the report
    sys.stdout.write(f"{text}\n")
    sys.stdout.flush()
    sys.exit(0 if ok else 1)


def main(out_path_holder: dict[str, str | None]) -> NoReturn:
    args = _parse_args(sys.argv[1:])
    out_path_holder["out_path"] = args.out

    if args.cmd is None or args.cmd.strip() == "":
        _emit(False, _report_json(ok=False, error={"kind": "MISSING_EVIDENCE"}), args.out)

    cmd = args.cmd  # narrowed to str by the NoReturn branch above
    tip_before = _git("rev-parse", "HEAD")
    expected_tip = args.expect_tip if args.expect_tip is not None else tip_before

    # Fail fast BEFORE running the gate: certifying the wrong tip would be
    # waste at best and a stale-evidence false-green at worst.
    if tip_before != expected_tip:
        _emit(
            False,
            _report_json(
                ok=False,
                error={"kind": "STALE_TIP", "expected": expected_tip, "got": tip_before},
            ),
            args.out,
        )

    evidence = _collect_evidence(cmd, args.timeout_s)
    result = certify_exit_gate(evidence, expected_tip)
    # `is True`, not bare truthy: basedpyright only performs discriminated-union
    # narrowing on this Literal[True]/Literal[False] `ok` field for an identity/
    # equality comparison, not for `if result.ok:` -- verified empirically
    # against a scratch fixture (bare truthy left both branches type-checking
    # as the full Ok | Err union; `is True` narrowed cleanly). See exit_gate.py's
    # Ok/Err docstrings.
    if result.ok is True:
        cert = result.data
        _emit(
            True,
            _report_json(
                ok=True,
                cert=RunnerCertShape(
                    cmd=cert.cmd, certified_tip=cert.certified_tip, tail_digest=cert.tail_digest
                ),
                evidence=evidence,
            ),
            args.out,
        )
    _emit(
        False,
        _report_json(ok=False, error=_certify_error_json(result.error), evidence=evidence),
        args.out,
    )


def _run() -> NoReturn:
    # Every execution emits exactly one structured JSON report -- a crash (not
    # a git repo, bad flags, fs failure) becomes a typed RUNNER_ERROR, never a
    # stack trace.
    out_path_holder: dict[str, str | None] = {"out_path": None}
    try:
        main(out_path_holder)
    except (
        Exception
    ) as exc:  # every crash becomes structured JSON, on purpose - see _RunnerError docstring
        _emit(
            False,
            _report_json(ok=False, error={"kind": "RUNNER_ERROR", "message": str(exc)}),
            out_path_holder["out_path"],
        )


if __name__ == "__main__":
    _run()
