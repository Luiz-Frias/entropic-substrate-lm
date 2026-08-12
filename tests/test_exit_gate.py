"""Tests for ``scripts/exit_gate.py`` — the Python port of AGX's ``exit-gate.ts``.

Ported test-first from
``/tmp/entropic-port-ref-04a8fc9c/exit-gate.test.ts``. Scope, matching
``scripts/exit_gate.py``'s module docstring:

* ``describe("certifyExitGate (typestate core)")`` and
  ``describe("typestate: SUBMITTED is unconstructible without a certificate")``
  port directly below — every ``it()`` case in those two blocks has a Python
  counterpart.
* ``describe("run-exit-gate CLI ...")`` and
  ``describe("run-exit-gate isolation ...")`` pin ``run-exit-gate.ts``, the
  deterministic-evidence-producer CLI. Round-2 review finding: this file
  previously deferred those two blocks with "there is deliberately no Python
  counterpart" — that was wrong per the review's own scope reading (no
  describe-block carve-out was granted) and is retracted here. Both blocks now
  have a Python counterpart: ``tests/test_run_exit_gate.py``, ported against
  ``scripts/run_exit_gate.py`` (the now-existing sibling port of
  ``run-exit-gate.ts``). See that file's module docstring for the 14-case
  breakdown (11 CLI + 3 isolation).
* ``describe("parity: harness exitGateBlockReason ≡ scripts/exit-gate.ts")``
  PARTIALLY ports: its case-matrix table (below, "runner_report_block_reason
  case matrix") is transcribed verbatim, with expected reason strings
  verified against the AGX harness source read during porting. What does NOT
  port is the "parity" half proper — a live cross-check against a second copy
  of the predicate living in a real workflow harness. This repo does not
  currently carry one: a ``.claude/workflows/td-phase-lifecycle.js`` is
  present in the porting worktree's filesystem, but ``git show
  main:.claude/workflows/td-phase-lifecycle.js`` fails (path does not exist
  in that tree, no history on any ref), and its content is unmistakably
  sales-collateral-agx's own (FOR-2xx ticket references, ``web/app``, Zod
  codegen) — contamination from another repo's worktree/template scaffolding,
  not a committed part of entropic-substrate-lm. Depending on it would make
  this suite pass in the one contaminated worktree and fail on every clean
  checkout, including CI. See the case-matrix section's own comment for what
  re-adding real parity would require.
"""

from __future__ import annotations

import json
import re
import typing
from dataclasses import dataclass, replace

import exit_gate
import pytest
from exit_gate import (
    DirtyTree,
    EmptyTail,
    ExitGateCertificate,
    ExitGateEvidence,
    MissingEvidence,
    NonzeroExit,
    ReadyToSubmit,
    RunnerCertShape,
    RunnerReportShape,
    StaleTip,
    Submitted,
    certify_exit_gate,
    runner_report_block_reason,
    to_submitted,
)

pytestmark = pytest.mark.unit

_SHA_HEX_LEN = 40
TIP = "a" * _SHA_HEX_LEN
OTHER_TIP = "b" * _SHA_HEX_LEN
EXIT_CMD = "RUN_LIVE_LLM_TESTS= prek run --all-files"
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _evidence(
    *,
    cmd: str = EXIT_CMD,
    exit_code: int = 0,
    tested_tip: str = TIP,
    tail: str = "check passed\nall hooks green",
    dirty: bool = False,
) -> ExitGateEvidence:
    return ExitGateEvidence(
        cmd=cmd, exit_code=exit_code, tested_tip=tested_tip, tail=tail, dirty=dirty
    )


# ─────────────────────────── certifyExitGate (typestate core) ────────────────────────────


def test_mints_a_certificate_for_clean_passing_evidence_on_the_expected_tip() -> None:
    result = certify_exit_gate(_evidence(), TIP)
    assert result.ok is True
    if result.ok:
        assert result.data.certified_tip == TIP
        assert "prek" in result.data.cmd
        # digest ties the cert to the exact tail bytes it certified
        assert _SHA256_HEX_RE.match(result.data.tail_digest)


def test_errs_missing_evidence_on_none_evidence() -> None:
    result = certify_exit_gate(None, TIP)
    assert result.ok is False
    if not result.ok:
        assert result.error == MissingEvidence()


def test_errs_missing_evidence_on_a_blank_command() -> None:
    """Nothing identifiable ran."""
    result = certify_exit_gate(_evidence(cmd="   "), TIP)
    assert result.ok is False
    if not result.ok:
        assert result.error == MissingEvidence()


def test_errs_nonzero_exit_carrying_the_exit_code() -> None:
    result = certify_exit_gate(_evidence(exit_code=3), TIP)
    assert result.ok is False
    if not result.ok:
        assert result.error == NonzeroExit(3)


def test_errs_stale_tip_when_the_tested_tip_differs_from_the_expected_tip() -> None:
    result = certify_exit_gate(_evidence(tested_tip=OTHER_TIP), TIP)
    assert result.ok is False
    if not result.ok:
        assert result.error == StaleTip(TIP, OTHER_TIP)


def test_errs_dirty_tree_when_the_post_run_tree_is_not_clean() -> None:
    result = certify_exit_gate(_evidence(dirty=True), TIP)
    assert result.ok is False
    if not result.ok:
        assert result.error == DirtyTree()


@pytest.mark.parametrize("tail", ["", "   \n  "])
def test_errs_empty_tail_on_empty_or_whitespace_only_output(tail: str) -> None:
    """Silence is not evidence."""
    result = certify_exit_gate(_evidence(tail=tail), TIP)
    assert result.ok is False
    if not result.ok:
        assert result.error == EmptyTail()


def test_pins_error_precedence_missing_evidence_over_nonzero_exit() -> None:
    """The top link in the precedence chain, previously unexercised: a blank
    command paired with a nonzero exit code must still resolve to
    MISSING_EVIDENCE, not NONZERO_EXIT — round-1 review finding."""
    result = certify_exit_gate(_evidence(cmd="   ", exit_code=3), TIP)
    assert not result.ok and result.error.kind == "MISSING_EVIDENCE"


def test_pins_error_precedence_nonzero_exit_over_stale_tip_over_dirty_tree_over_empty_tail() -> (
    None
):
    worst = certify_exit_gate(
        _evidence(exit_code=1, tested_tip=OTHER_TIP, dirty=True, tail=""), TIP
    )
    assert not worst.ok and worst.error.kind == "NONZERO_EXIT"

    stale = certify_exit_gate(_evidence(tested_tip=OTHER_TIP, dirty=True, tail=""), TIP)
    assert not stale.ok and stale.error.kind == "STALE_TIP"

    dirty = certify_exit_gate(_evidence(dirty=True, tail=""), TIP)
    assert not dirty.ok and dirty.error.kind == "DIRTY_TREE"


def test_certify_exit_gate_precedes_dirty_tree_over_empty_tail() -> None:
    """The fifth precedence step, isolated: with STALE_TIP and NONZERO_EXIT
    both ruled out, a tree that is both dirty AND silent certifies as
    DIRTY_TREE here — the opposite ordering from the report-path check below
    (see test_report_path_precedes_empty_tail_over_dirty_tree_unlike_certify).
    This is the exact precedence exit-gate.ts's own certifyExitGate pins."""
    result = certify_exit_gate(_evidence(dirty=True, tail=""), TIP)
    assert not result.ok
    assert result.error.kind == "DIRTY_TREE"


# ──────────── typestate: SUBMITTED is unconstructible without a certificate ─────────────


def test_to_submitted_consumes_a_ready_to_submit_state_and_certificate() -> None:
    cert_result = certify_exit_gate(_evidence(), TIP)
    assert cert_result.ok is True
    if not cert_result.ok:
        return
    ready = ReadyToSubmit(branch="feature/x", tip=TIP)
    submitted = to_submitted(ready, cert_result.data)
    assert isinstance(submitted, Submitted)
    assert submitted.tag == "Submitted"
    assert submitted.cert.certified_tip == TIP


def test_certificate_cannot_be_constructed_directly_without_the_brand() -> None:
    """Runtime analog of the TS forge-proof pin (``expectTypeOf(...).not
    .toExtend<ExitGateCertificate>()``). TS's guarantee is compile-time only
    (a ``unique symbol`` has no runtime representation at all); this Python
    port adds a REAL runtime check on top. Mechanism, precisely (round-2
    review found the PREVIOUS mechanism — a required, identity-checked kwarg —
    genuinely forgeable via ``dataclasses.replace()``; see
    test_replace_can_no_longer_forge_a_certificate below and
    scripts/exit_gate.py's module docstring for the before/after): the brand
    field is ``init=False``, excluded from the constructor entirely, so THIS
    call — a shape-identical construction simply omitting it, since it isn't a
    parameter to omit or supply — is fully valid to the type checker (no
    ``# type: ignore`` needed here anymore) and still raises ``TypeError`` at
    construction time, because ``__post_init__`` now refuses unconditionally.
    See scripts/exit_gate.py's ``ExitGateCertificate`` docstring + this task's
    designNotes for the precise (weaker) static-only half of the story:
    basedpyright's ``reportPrivateUsage`` on the private brand class/instance,
    verified empirically against a scratch fixture, not merely asserted."""
    with pytest.raises(TypeError):
        ExitGateCertificate(cmd=EXIT_CMD, certified_tip=TIP, tail_digest="d" * 64)


def test_certificate_rejects_a_forged_brand() -> None:
    """Even explicitly NAMING the private kwarg is refused — and now for a
    stronger reason than before: ``_brand`` isn't a constructor parameter at
    all post-hardening (``init=False``), so basedpyright itself flags this
    call (``reportCallIssue``: "No parameter named _brand", verified against a
    scratch fixture placed inside scripts/ so the sibling import resolves),
    and at runtime it's an immediate ``__init__``-level ``TypeError`` — an
    unexpected-keyword-argument rejection that happens BEFORE
    ``__post_init__`` even runs, not the identity check that test used to
    exercise pre-hardening."""
    with pytest.raises(TypeError):
        ExitGateCertificate(
            cmd=EXIT_CMD,
            certified_tip=TIP,
            tail_digest="d" * 64,
            _brand=object(),  # type: ignore[call-arg]
        )


def test_replace_can_no_longer_forge_a_certificate() -> None:
    """THE round-2 finding, pinned directly: ``dataclasses.replace()`` on a
    GENUINE certificate must no longer mint a certificate for a different
    tip/command. Pre-hardening repro (round-2 review, verbatim): starting from
    a real cert minted by certify_exit_gate(), ``replace(genuine_cert,
    certified_tip=OTHER_TIP, cmd="echo weaker-gate")`` re-passed the ORIGINAL
    ``_brand`` (untouched by the override) straight back into ``__init__``,
    and since it WAS the genuine singleton, the old identity-only check in
    ``__post_init__`` passed — minting a certificate for an arbitrary
    tip/command pair from a legitimate one, no private import required. This
    is the "pin the actual ceiling" test the round-2 mustFix asked for,
    upgraded: because the brand is now ``init=False`` (see
    scripts/exit_gate.py's module docstring), ``replace()`` can no longer
    carry it forward through ``__init__`` at all, so this now pins CLOSURE,
    not just documentation of the limitation."""
    cert_result = certify_exit_gate(_evidence(), TIP)
    assert cert_result.ok is True
    if not cert_result.ok:
        return
    genuine_cert = cert_result.data

    with pytest.raises(TypeError):
        replace(genuine_cert, certified_tip=OTHER_TIP, cmd="echo weaker-gate")


def test_certify_errors_form_a_closed_five_member_union() -> None:
    """Type-level pin (``expectTypeOf<CertifyError["kind"]>()...``), approximated
    at runtime via ``typing.get_args`` over the ``type CertifyError = ...``
    alias itself — not by enumerating reachable cases. Round-1 review finding:
    a case-enumeration test only pins REACHABILITY (it would stay green if a
    sixth ``CertifyError`` member were added) and this test's name claims
    CLOSEDNESS. ``get_args(CertifyError.__value__)`` reads the union's actual
    member list, so adding a member without adding it here goes red."""
    member_kinds = {
        member.__dataclass_fields__["kind"].default
        for member in typing.get_args(exit_gate.CertifyError.__value__)
    }
    assert member_kinds == {
        "MISSING_EVIDENCE",
        "NONZERO_EXIT",
        "STALE_TIP",
        "DIRTY_TREE",
        "EMPTY_TAIL",
    }

    # Cross-check: every kind the runtime actually produces is a subset of the
    # declared union (reachability), which combined with the closedness check
    # above pins both directions.
    kinds = {
        certify_exit_gate(None, TIP),
        certify_exit_gate(_evidence(exit_code=1), TIP),
        certify_exit_gate(_evidence(tested_tip=OTHER_TIP), TIP),
        certify_exit_gate(_evidence(dirty=True), TIP),
        certify_exit_gate(_evidence(tail=""), TIP),
    }
    observed = {result.error.kind for result in kinds if not result.ok}
    assert observed == member_kinds


# ──────────── runner_report_block_reason case matrix (from exit-gate.ts's own
# "parity: harness exitGateBlockReason" describe block) ────────────────────────
# NOTE ON SCOPE: the TS suite runs this table against BOTH `runnerReportBlock
# Reason` and a second copy extracted from AGX's own workflow harness JS, then
# asserts the two agree. This repo has no such WORKFLOW HARNESS wired in yet —
# narrowly true even though scripts/run_exit_gate.py now consumes exit_gate.py
# at runtime (it imports certify_exit_gate; see that module's docstring), a
# deterministic evidence-producer CLI is not a harness that gates a SUBMITTED
# transition. There is still no second copy of runner_report_block_reason to
# check agreement against — porting a "parity" test against a harness that
# doesn't exist here would just be a test that always trivially "agrees with
# itself".
# What DOES port, and is exactly as load-bearing as it was in the TS suite: the
# case matrix itself, with expected reason strings transcribed verbatim from
# exit-gate.ts (the task's stated contract) via the identical AGX harness
# source read during porting. This pins runner_report_block_reason against the
# REFERENCE, independent of whether any local harness exists to double-check.
# If/when this repo grows its own JS or Python harness copy of this predicate,
# the reusable half is exactly this table — re-add a second parity assertion
# against that copy's output at that point, not before.
def _passing_report() -> RunnerReportShape:
    return RunnerReportShape(
        ok=True,
        cert=RunnerCertShape(cmd=EXIT_CMD, certified_tip=TIP, tail_digest="d" * 64),
        evidence=ExitGateEvidence(
            cmd=EXIT_CMD, exit_code=0, tested_tip=TIP, tail="all hooks green", dirty=False
        ),
    )


@dataclass(frozen=True, slots=True)
class _Case:
    name: str
    xg: RunnerReportShape | None
    tip: str | None
    expected_reason: str | None  # None means the case must certify (pass)


_QUOTED_EXIT_CMD = json.dumps(EXIT_CMD, separators=(",", ":"), ensure_ascii=False)

# Round-1 review regression fixtures: pin the END boundary of the canonical-SHA
# guard so `_FULL_SHA_RE`'s `\Z` anchor cannot be silently deleted (or weakened
# back to `$`, which — unlike JS — also matches just before a trailing "\n")
# again without three tests going red. See scripts/exit_gate.py's
# `_FULL_SHA_RE` comment for the Python-`$`-vs-`\Z` semantics this closes.
_TIP_WITH_TRAILING_NEWLINE = TIP + "\n"
_TIP_PLUS_ONE_HEX_CHAR = TIP + "a"  # 41 hex chars — one over the boundary

_CASES: list[_Case] = [
    _Case(
        "null report",
        None,
        TIP,
        'exit gate did not certify (missing or ok!==true): "no exitGate relayed"',
    ),
    _Case(
        "ok:false with error",
        RunnerReportShape(ok=False, error={"kind": "NONZERO_EXIT"}),
        TIP,
        'exit gate did not certify (missing or ok!==true): {"kind":"NONZERO_EXIT"}',
    ),
    _Case(
        "ok:true missing cert",
        replace(_passing_report(), cert=None),
        TIP,
        "ok:true without cert+evidence — non-canonical runner output",
    ),
    _Case(
        "ok:true missing evidence",
        replace(_passing_report(), evidence=None),
        TIP,
        "ok:true without cert+evidence — non-canonical runner output",
    ),
    _Case(
        "evidence cmd mismatch",
        replace(
            _passing_report(),
            evidence=replace(_passing_report().evidence, cmd="echo weaker-gate"),  # type: ignore[union-attr]
        ),
        TIP,
        f"exit-gate command mismatch: expected {_QUOTED_EXIT_CMD}",
    ),
    _Case(
        "cert cmd mismatch",
        replace(
            _passing_report(),
            cert=RunnerCertShape(cmd="echo weaker-gate", certified_tip=TIP, tail_digest="d" * 64),
        ),
        TIP,
        f"exit-gate command mismatch: expected {_QUOTED_EXIT_CMD}",
    ),
    _Case(
        "nonzero exit",
        replace(
            _passing_report(),
            evidence=replace(_passing_report().evidence, exit_code=3),  # type: ignore[union-attr]
        ),
        TIP,
        "exit-gate evidence nonzero exit code",
    ),
    _Case(
        "empty tail",
        replace(
            _passing_report(),
            evidence=replace(_passing_report().evidence, tail="  "),  # type: ignore[union-attr]
        ),
        TIP,
        "exit-gate evidence tail is empty (silence is not evidence)",
    ),
    _Case(
        "dirty tree",
        replace(
            _passing_report(),
            evidence=replace(_passing_report().evidence, dirty=True),  # type: ignore[union-attr]
        ),
        TIP,
        "exit-gate ran on a dirty tree (unfolded auto-fixes)",
    ),
    _Case(
        "short evidence tip",
        replace(
            _passing_report(),
            evidence=replace(_passing_report().evidence, tested_tip=TIP[:8]),  # type: ignore[union-attr]
        ),
        TIP,
        "exit-gate tip not a full canonical SHA or cert/evidence tip mismatch",
    ),
    _Case(
        "cert/evidence tip disagreement",
        replace(
            _passing_report(),
            cert=RunnerCertShape(cmd=EXIT_CMD, certified_tip=OTHER_TIP, tail_digest="d" * 64),
        ),
        TIP,
        "exit-gate tip not a full canonical SHA or cert/evidence tip mismatch",
    ),
    _Case(
        "short phase tip",
        _passing_report(),
        TIP[:8],
        f"phase tip is not a full 40-hex SHA: {TIP[:8]}",
    ),
    _Case(
        "missing phase tip",
        _passing_report(),
        None,
        "phase tip is not a full 40-hex SHA: missing",
    ),
    _Case(
        "phase tip mismatch",
        _passing_report(),
        OTHER_TIP,
        f"exit-gate tip mismatch: certified {TIP} but phase tip is {OTHER_TIP}",
    ),
    _Case("fully canonical pass", _passing_report(), TIP, None),
    # --- Round-1 review regression cases: the SHA end-anchor boundary ---------
    _Case(
        # (a) evidence tip carries a trailing newline — exactly what a raw,
        # un-stripped `git rev-parse HEAD` produces. cert.certified_tip and the
        # phase tip are set to the SAME value so the case isolates the anchor
        # check itself rather than tripping the separate tip-mismatch branch.
        "evidence tip with trailing newline (SHA end-anchor)",
        replace(
            _passing_report(),
            cert=RunnerCertShape(
                cmd=EXIT_CMD, certified_tip=_TIP_WITH_TRAILING_NEWLINE, tail_digest="d" * 64
            ),
            evidence=replace(
                _passing_report().evidence,  # type: ignore[union-attr]
                tested_tip=_TIP_WITH_TRAILING_NEWLINE,
            ),
        ),
        _TIP_WITH_TRAILING_NEWLINE,
        "exit-gate tip not a full canonical SHA or cert/evidence tip mismatch",
    ),
    _Case(
        # (b) an otherwise fully-passing report, but the PHASE tip (not the
        # evidence tip) carries the trailing newline — isolates the second
        # `_FULL_SHA_RE` call site (runner_report_block_reason's own check).
        "phase tip with trailing newline (SHA end-anchor)",
        _passing_report(),
        _TIP_WITH_TRAILING_NEWLINE,
        f"phase tip is not a full 40-hex SHA: {_TIP_WITH_TRAILING_NEWLINE}",
    ),
    _Case(
        # (c) a 41-hex-character tip — one char over the boundary, no
        # non-hex character involved at all. Kills a mutation that only
        # widens the match to "at least 40 hex chars" (e.g. a stray `+`
        # instead of `{40}`, or a bare prefix match) in a way (a)/(b) do not,
        # since every character here stays inside the `[0-9a-f]` class.
        "41-hex tip, one char over the boundary (SHA end-anchor)",
        replace(
            _passing_report(),
            cert=RunnerCertShape(
                cmd=EXIT_CMD, certified_tip=_TIP_PLUS_ONE_HEX_CHAR, tail_digest="d" * 64
            ),
            evidence=replace(
                _passing_report().evidence,  # type: ignore[union-attr]
                tested_tip=_TIP_PLUS_ONE_HEX_CHAR,
            ),
        ),
        _TIP_PLUS_ONE_HEX_CHAR,
        "exit-gate tip not a full canonical SHA or cert/evidence tip mismatch",
    ),
    # Extra case (not in the TS Cases table): pins the deliberate divergence
    # from certify_exit_gate's own precedence — evidenceIssue (which this
    # report path calls) checks EMPTY_TAIL *before* DIRTY_TREE, the opposite
    # order from certify_exit_gate. Ported verbatim from exit-gate.ts, not
    # normalized away — see test_certify_exit_gate_precedes_dirty_tree_over_
    # empty_tail for the certify-side half of this pair.
    _Case(
        "dirty AND empty tail together — report path resolves EMPTY_TAIL first",
        replace(
            _passing_report(),
            evidence=replace(_passing_report().evidence, dirty=True, tail=""),  # type: ignore[union-attr]
        ),
        TIP,
        "exit-gate evidence tail is empty (silence is not evidence)",
    ),
    # Round-2 review finding: `evidence` here is RunnerReportShape's untrusted
    # relay copy (never a genuine ExitGateEvidence instance, per that class's
    # own docstring), so despite the `tested_tip: str` annotation a real JSON
    # relay can hand this a `None` for an omitted `testedTip`. Pre-fix, this
    # crashed `_evidence_issue` with a raw TypeError instead of returning a
    # block reason (`_FULL_SHA_RE.match(None)`); the fix guards with `or ""`,
    # matching both the TS reference (`tip || ''`) and the harness's
    # PARITY-PIN copy, which block cleanly on this input.
    _Case(
        "evidence tip is null (untrusted relay omitted testedTip)",
        replace(
            _passing_report(),
            evidence=replace(_passing_report().evidence, tested_tip=None),  # type: ignore[arg-type,union-attr]
        ),
        TIP,
        "exit-gate tip not a full canonical SHA or cert/evidence tip mismatch",
    ),
]


@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_runner_report_block_reason_case_matrix(case: _Case) -> None:
    reason = runner_report_block_reason(case.xg, case.tip, EXIT_CMD)
    assert reason == case.expected_reason


def test_report_path_precedes_empty_tail_over_dirty_tree_unlike_certify() -> None:
    """The report-path twin of
    test_certify_exit_gate_precedes_dirty_tree_over_empty_tail: same shape of
    input (dirty AND silent), opposite winner, because runner_report_block_
    reason's evidenceIssue check chain orders these two checks the other way
    round. Both orderings are inherited verbatim from exit-gate.ts and are
    pinned here so neither can be silently "cleaned up" into agreement."""
    reason = runner_report_block_reason(
        replace(
            _passing_report(),
            evidence=replace(_passing_report().evidence, dirty=True, tail=""),  # type: ignore[union-attr]
        ),
        TIP,
        EXIT_CMD,
    )
    assert reason == "exit-gate evidence tail is empty (silence is not evidence)"
