"""FOR-754 port: the exit-gate typestate certificate core — AGX ``exit-gate.ts``
ported to Python (``/tmp/entropic-port-ref-04a8fc9c/exit-gate.ts``).

AGX's td-phase-lifecycle workflow harness refuses to mark a phase SUBMITTED
without a certifying ``exitGate`` JSON produced by a deterministic runner: the
agent is a COURIER, never a witness — a runner executes the gate itself, reads
the git tip before and after, and checks tree cleanliness, so a GREEN claim
cannot be forged or transcribed. That runner is TypeScript/bun
(``run-exit-gate.ts``); this repo is Python/uv. This module ports the
certificate core; the deterministic evidence-producer CLI that actually
executes a gate command and reads git state is a separate, now-existing
sibling port, :mod:`run_exit_gate` (``scripts/run_exit_gate.py``, ported from
``run-exit-gate.ts``) — see that module's docstring. This repo still does not
wire either into a *harness* (a Python-side td-phase-lifecycle equivalent that
consumes :func:`runner_report_block_reason` to gate a SUBMITTED transition);
see the NOT-A-DEPENDENCY note below for why that harness step is out of scope
here regardless.

NOT A DEPENDENCY: a ``.claude/workflows/td-phase-lifecycle.js`` may be present
in a given porting worktree's filesystem, but it is not part of
entropic-substrate-lm — ``git show main:.claude/workflows/td-phase-lifecycle.js``
fails on every ref this repo has, and its content (FOR-2xx tickets, ``web/app``,
Zod codegen) is unmistakably sales-collateral-agx's own, bled in via worktree
scaffolding. Nothing in this module reads, imports, or asserts the existence of
that file; :func:`runner_report_block_reason` below is a standalone port of
``exit-gate.ts``'s predicate, not a live parity check against it.

SCOPE (this file): the typestate certificate core only —
:func:`certify_exit_gate`, :class:`ExitGateCertificate`, the closed
:data:`CertifyError` union, the :data:`Phase` FSM + :func:`to_submitted`, and
:func:`runner_report_block_reason` (the deterministic BLOCKED predicate a
future harness would need — ported here as a standalone function pinned by
its own case-matrix test; see :mod:`tests.test_exit_gate`'s module docstring
for exactly what "parity" does and doesn't mean without a wired-in harness).
Nothing in THIS file performs I/O or subprocess execution — that lives
entirely in :mod:`run_exit_gate` (``scripts/run_exit_gate.py``), which imports
this module's :func:`certify_exit_gate` rather than duplicating it, exactly as
``run-exit-gate.ts`` imports from ``exit-gate.ts``.

TYPESTATE, and where the Python guarantee is WEAKER than the TS original:
in TS, :class:`ExitGateCertificate` is branded with ``declare const Passed:
unique symbol`` — a symbol with **no runtime existence at all**, used purely
as a compile-time tag, so ``certifyExitGate()`` is the only door and a
structural object literal fails ``tsc``. That guarantee is 100% static: it
vanishes entirely once JS actually runs (a raw ``.js`` caller, or a ``ts``
caller behind ``as unknown as ExitGateCertificate``, faces zero runtime
resistance).

Python has no true encapsulation, so this port uses a **module-private brand
class** (:class:`_CertBrand`, a leading-underscore name never re-exported)
instead of a symbol with no runtime form at all — and gets two guarantees of
a different (not strictly weaker) shape:

* **Static**: importing ``_CertBrand`` or its singleton from outside this
  module is a ``reportPrivateUsage`` error under strict basedpyright — and a
  structural literal (e.g. a plain ``dict``) fails ``ExitGateCertificate``
  typing too, because pyright treats dataclasses nominally, not
  structurally. Empirically verified against a scratch fixture before
  writing this module (not merely asserted): both the private-import and the
  structural-literal-as-return-type cases produced real diagnostics. Basis
  for this guarantee: ``basedpyright src scripts`` runs as a blocking gate in
  both ``just check`` and CI, so an in-tree forgery attempt fails the same
  gate any other strict-mode violation would. NOTE the trade this makes
  (round-2 review, see below): ``_brand`` is deliberately *not* a constructor
  parameter at all, so ``ExitGateCertificate(cmd=..., certified_tip=...,
  tail_digest=...)`` is now **statically clean** — the "missing required
  keyword-only argument" diagnostic that used to flag a brand-less call is
  gone. The static half of this guarantee is nominal-dataclass typing (a
  plain ``dict`` still fails ``ExitGateCertificate`` typing) plus
  ``reportPrivateUsage`` on importing ``_CertBrand``/``_THE_BRAND`` by name —
  not "the brand kwarg is required".
* **Runtime** (a layer the TS original does not have at all, and the one a
  round-2 review found genuinely broken — see below): the brand field is
  excluded from ``__init__`` entirely (``init=False``), so every path that
  reaches Python's normal dataclass construction — a direct call, AND
  ``dataclasses.replace()`` on a genuine certificate — always sees ``_brand``
  at its unset default and :meth:`ExitGateCertificate.__post_init__` refuses
  unconditionally. The one legitimate mint, :func:`_mint_certificate`, never
  calls ``__init__`` at all: it builds the instance via
  ``object.__new__``/``object.__setattr__`` (the same low-level mechanism
  CPython's own generated ``__init__`` uses to populate a frozen dataclass's
  slots) and is called from exactly one place, :func:`certify_exit_gate`.

ROUND-2 FINDING AND FIX, stated precisely (no overclaiming — this is the exact
class of error round-2 review failed on): the PREVIOUS runtime design kept
``_brand`` as a required, keyword-only, ``init=True`` field checked by
identity in ``__post_init__``. That defeats a caller who tries to construct a
certificate FROM SCRATCH, but not one who starts from a **genuine** one:
``dataclasses.replace(genuine_cert, certified_tip=OTHER, cmd="echo
weaker-gate")`` re-passes the ORIGINAL, real ``_brand`` for every field not
listed in the override — since it IS the genuine singleton, the identity
check trivially passed, minting a certificate for an arbitrary tip/command
pair with zero private imports and zero type-checker complaints. Repro was
empirical, not theoretical (see this task's designNotes). Moving ``_brand``
out of ``__init__`` closes exactly that hole: ``replace()`` cannot carry a
field forward that ``__init__`` never accepted from it in the first place.
Pinned by ``test_replace_can_no_longer_forge_a_certificate`` in
``tests/test_exit_gate.py``.

THE CEILING THIS DOES **NOT** CLOSE, named rather than left implicit: anyone
willing to replicate :func:`_mint_certificate`'s own technique —
``object.__new__(ExitGateCertificate)`` plus four ``object.__setattr__``
calls, using only builtins, no private import required — can still mint an
arbitrary certificate from scratch. Nothing downstream (:func:`to_submitted`
included) re-checks a certificate's provenance beyond "is this an
``ExitGateCertificate`` instance", so this was ALREADY the ceiling before
this fix too, in both languages: Python's is "reach past ``__init__`` with
builtins", TS's is "lie to the compiler with ``as unknown as
ExitGateCertificate``". Claim precisely: this port refuses every certificate
that reaches ``__init__`` genuinely and not carrying a legitimate brand —
direct construction AND ``dataclasses.replace()`` alike — never "unforgeable
at runtime" in the absolute sense.

Field names are snake_case (PEP 8), unlike the TS source's camelCase:
``exitCode`` -> ``exit_code``, ``testedTip`` -> ``tested_tip``,
``certifiedTip`` -> ``certified_tip``, ``tailDigest`` -> ``tail_digest``.
:mod:`run_exit_gate` re-camelCases these back on the JSON wire it emits (AGX's
own harness parses ``testedTip``/``certifiedTip``/``tailDigest`` literally) —
see that module's ``_evidence_json`` / ``_report_json`` helpers for the one
place in this repo that relay now happens.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Final, Literal

__all__ = [
    "CertifyError",
    "DirtyTree",
    "EmptyTail",
    "Err",
    "ExitGateCertificate",
    "ExitGateEvidence",
    "MissingEvidence",
    "NonzeroExit",
    "Ok",
    "Phase",
    "ReadyToSubmit",
    "Result",
    "RunnerCertShape",
    "RunnerReportShape",
    "StaleTip",
    "Submitted",
    "certify_exit_gate",
    "err",
    "ok",
    "runner_report_block_reason",
    "to_submitted",
]

# `\Z` (not `$`) is the exact analog of JavaScript's non-multiline `$`: Python's
# `$` also matches immediately before a trailing "\n", which would let a raw,
# un-stripped `git rev-parse HEAD` (which ends in "\n") pass this guard while
# the TS reference (`^[0-9a-f]{40}$` under JS semantics, where `$` matches only
# the true end of input) rejects it — a fail-OPEN divergence in the exact
# canonical-SHA check this gate exists to enforce. Round-1 review finding;
# regression cases pinning this boundary live in tests/test_exit_gate.py.
_FULL_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}\Z")


# ───────────────────────────────────────────────────────────────── Result ────
# Minimal local port of result.ts's Ok/Err/Result/ok/err — exit-gate.ts's only
# import, kept dependency-free per its own portability constraint ("its only
# import is src/shared/result, itself dependency-free by design. Extraction
# later is a file move, not a refactor."). Deliberately NOT the full result.ts
# surface (map/match/partition/fromThrowable/fromPromise): exit-gate.ts uses
# only ok/err/Result, and porting unused surface would be scope creep here.
@dataclass(frozen=True, slots=True)
class Ok[T]:
    """Successful result. ``ok`` is a ``Literal[True]`` discriminant field —
    pyright/basedpyright narrow ``Result[T, E]`` on it like a TS discriminated
    union, but ONLY for an identity/equality check (``if result.ok is True:``),
    not a bare truthy test (``if result.ok:``) — verified empirically against a
    scratch fixture (``scripts/run_exit_gate.py`` uses the ``is True`` form for
    exactly this reason; a bare truthy check left both branches type-checking
    as the full ``Ok | Err`` union with 0 diagnostics, a silent narrowing
    failure rather than a caught one, since it degrades to an ordinary runtime
    ``bool`` check that still happens to work at runtime)."""

    data: T
    ok: Literal[True] = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class Err[E]:
    """Failed result. ``ok`` is a ``Literal[False]`` discriminant field."""

    error: E
    ok: Literal[False] = field(default=False, init=False)


type Result[T, E] = Ok[T] | Err[E]


def ok[T](data: T) -> Ok[T]:
    return Ok(data)


def err[E](error: E) -> Err[E]:
    return Err(error)


# ─────────────────────────────────────────────────────────────── Evidence ────
@dataclass(frozen=True, slots=True)
class ExitGateEvidence:
    """Raw evidence from one gate execution — produced by a deterministic
    runner, never typed by hand."""

    cmd: str
    exit_code: int
    tested_tip: str
    tail: str
    dirty: bool


# ──────────────────────────────────────────────────────────── Certificate ────
class _CertBrand:
    """Module-private marker type — the closest Python analog of TS's
    ``declare const Passed: unique symbol``. See this module's docstring for
    the precise, non-overclaiming comparison between the two guarantees."""

    __slots__ = ()


_THE_BRAND: Final = _CertBrand()


@dataclass(frozen=True, slots=True)
class ExitGateCertificate:
    """Proof that one exact tip passed the exit gate.

    Unconstructible outside :func:`certify_exit_gate` in the two senses this
    module's docstring spells out: statically (the brand type/instance is
    private; basedpyright strict flags any outside import) and, uniquely to
    this port, at runtime.

    The runtime half, precisely (round-2 review closed a real hole here — see
    this module's docstring for the before/after): ``_brand`` is ``init=False``,
    so it is not a constructor parameter at all — every call that reaches
    ``__init__`` (a direct call, or ``dataclasses.replace()`` on a genuine
    certificate) always sees it at its unset default, and
    :meth:`__post_init__` refuses unconditionally. The only legitimate mint is
    :func:`_mint_certificate`, called from exactly :func:`certify_exit_gate`,
    which never invokes ``__init__`` at all.
    """

    cmd: str
    certified_tip: str
    # sha256 hex of the evidence tail — ties the cert to the exact bytes it
    # certified.
    tail_digest: str
    # init=False (deliberately NOT a constructor parameter — see the class
    # docstring and this module's docstring for why): the unset default is
    # `None`, which every path through `__init__`, replace() included, always
    # produces, so __post_init__ below can refuse unconditionally rather than
    # needing to distinguish "no brand" from "wrong brand".
    _brand: _CertBrand | None = field(default=None, repr=False, compare=False, init=False)

    def __post_init__(self) -> None:
        msg = (
            "ExitGateCertificate can only be minted by certify_exit_gate() — "
            "every construction path that reaches __init__ (direct "
            "construction, or dataclasses.replace() on a genuine certificate) "
            "is refused at runtime, not just flagged by the type checker."
        )
        raise TypeError(msg)


def _mint_certificate(*, cmd: str, certified_tip: str, tail_digest: str) -> ExitGateCertificate:
    """The one legitimate constructor. Deliberately bypasses ``__init__`` (and
    therefore ``__post_init__``) via ``object.__new__`` + ``object.__setattr__``
    — the same low-level mechanism CPython's own generated ``__init__`` uses to
    populate a frozen dataclass's slots, since a frozen dataclass's normal
    ``self.x = value`` is disallowed by its overridden ``__setattr__``. Called
    from exactly one place: :func:`certify_exit_gate`."""
    cert = object.__new__(ExitGateCertificate)
    object.__setattr__(cert, "cmd", cmd)
    object.__setattr__(cert, "certified_tip", certified_tip)
    object.__setattr__(cert, "tail_digest", tail_digest)
    object.__setattr__(cert, "_brand", _THE_BRAND)
    return cert


# ───────────────────────────────────────────────────────────── CertifyError ───
# Closed union — every way certification can fail, in precedence order:
# MISSING_EVIDENCE > NONZERO_EXIT > STALE_TIP > DIRTY_TREE > EMPTY_TAIL (pinned
# in tests/test_exit_gate.py). This is certify_exit_gate's OWN precedence; the
# report-path check below (runner_report_block_reason / _evidence_issue) is a
# SEPARATE function with a different order between DIRTY_TREE and EMPTY_TAIL —
# inherited verbatim from exit-gate.ts (certifyExitGate vs evidenceIssue) and
# preserved deliberately, not normalized into agreement. See both precedence
# tests in tests/test_exit_gate.py for the pinned pair.
@dataclass(frozen=True, slots=True)
class MissingEvidence:
    kind: Literal["MISSING_EVIDENCE"] = field(default="MISSING_EVIDENCE", init=False)


@dataclass(frozen=True, slots=True)
class NonzeroExit:
    exit_code: int
    kind: Literal["NONZERO_EXIT"] = field(default="NONZERO_EXIT", init=False)


@dataclass(frozen=True, slots=True)
class StaleTip:
    expected: str
    got: str
    kind: Literal["STALE_TIP"] = field(default="STALE_TIP", init=False)


@dataclass(frozen=True, slots=True)
class DirtyTree:
    kind: Literal["DIRTY_TREE"] = field(default="DIRTY_TREE", init=False)


@dataclass(frozen=True, slots=True)
class EmptyTail:
    kind: Literal["EMPTY_TAIL"] = field(default="EMPTY_TAIL", init=False)


type CertifyError = MissingEvidence | NonzeroExit | StaleTip | DirtyTree | EmptyTail


def certify_exit_gate(
    evidence: ExitGateEvidence | None, expected_tip: str
) -> Result[ExitGateCertificate, CertifyError]:
    """The single certificate constructor. Pure; Result-typed (expected
    failures are data, per result.ts's own conventions). Precedence:
    MISSING_EVIDENCE > NONZERO_EXIT > STALE_TIP > DIRTY_TREE > EMPTY_TAIL.
    """
    if evidence is None or evidence.cmd.strip() == "":
        return err(MissingEvidence())
    if evidence.exit_code != 0:
        return err(NonzeroExit(evidence.exit_code))
    if evidence.tested_tip != expected_tip:
        return err(StaleTip(expected_tip, evidence.tested_tip))
    if evidence.dirty:
        return err(DirtyTree())
    if evidence.tail.strip() == "":
        return err(EmptyTail())
    return ok(
        _mint_certificate(
            cmd=evidence.cmd,
            certified_tip=evidence.tested_tip,
            tail_digest=hashlib.sha256(evidence.tail.encode("utf-8")).hexdigest(),
        )
    )


# ─────────────────────────────────────────────────────────────────── Phase ────
# Phase FSM (typestate): Submitted structurally requires the certificate, so
# the transition cannot be written without one.
@dataclass(frozen=True, slots=True)
class ReadyToSubmit:
    branch: str
    tip: str
    tag: Literal["ReadyToSubmit"] = field(default="ReadyToSubmit", init=False)


@dataclass(frozen=True, slots=True)
class Submitted:
    branch: str
    tip: str
    cert: ExitGateCertificate
    tag: Literal["Submitted"] = field(default="Submitted", init=False)


type Phase = ReadyToSubmit | Submitted


def to_submitted(state: ReadyToSubmit, cert: ExitGateCertificate) -> Submitted:
    """The only transition into Submitted — consumes the certificate by type."""
    return Submitted(branch=state.branch, tip=state.tip, cert=cert)


# ──────────────────────────────────────────────────────────── Runner report ───
@dataclass(frozen=True, slots=True)
class RunnerCertShape:
    """The cert as RELAYED in untrusted JSON — deliberately unbranded and
    NOT an :class:`ExitGateCertificate` (that type can only come from
    :func:`certify_exit_gate`). This is what an agent's runner output claims,
    nothing more; see :class:`RunnerReportShape`."""

    cmd: str
    certified_tip: str
    tail_digest: str


@dataclass(frozen=True, slots=True)
class RunnerReportShape:
    """The serialized runner report — what :mod:`run_exit_gate`
    (``scripts/run_exit_gate.py``) emits and agents relay verbatim. Untrusted
    data:
    ``cert``/``evidence`` here are plain shapes, never genuine
    :class:`ExitGateCertificate`/:class:`ExitGateEvidence` instances,
    which is exactly why :func:`runner_report_block_reason` re-derives every
    check from raw fields instead of trusting the shape.
    """

    ok: bool
    cert: RunnerCertShape | None = None
    error: dict[str, object] | None = None
    evidence: ExitGateEvidence | None = None


_NO_RELAY_MSG: Final = "no exitGate relayed"


def _json_compact(value: object) -> str:
    """``json.dumps`` with JS ``JSON.stringify``-equivalent formatting: no
    space after ``:``/``,`` (Python's default inserts one) and no ASCII
    escaping of non-ASCII text (Python's default does, JS's does not)."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _evidence_issue(evidence: ExitGateEvidence, cert: RunnerCertShape, exit_cmd: str) -> str | None:
    """Mirrors exit-gate.ts's private ``evidenceIssue`` helper. NOTE the
    order here — NONZERO_EXIT, then EMPTY_TAIL, then DIRTY_TREE — is NOT the
    same as certify_exit_gate's (DIRTY_TREE before EMPTY_TAIL). Both orders
    are ported verbatim from their respective TS functions; see this
    module's CertifyError comment."""
    if evidence.cmd != exit_cmd or cert.cmd != exit_cmd:
        return f"exit-gate command mismatch: expected {_json_compact(exit_cmd)}"
    if evidence.exit_code != 0:
        return "exit-gate evidence nonzero exit code"
    if evidence.tail.strip() == "":
        return "exit-gate evidence tail is empty (silence is not evidence)"
    if evidence.dirty:
        return "exit-gate ran on a dirty tree (unfolded auto-fixes)"
    # `evidence.tested_tip or ""`, not a bare `evidence.tested_tip`: `evidence`
    # here is RunnerReportShape's untrusted-relay copy (see that class's
    # docstring — "never genuine ExitGateEvidence instances"), so despite the
    # `str` annotation a real JSON relay can hand this a `None` for an omitted
    # `testedTip`. Round-2 review finding: without the `or ""` guard,
    # `_FULL_SHA_RE.match(None)` raises TypeError instead of returning a block
    # reason — a crash where both the TS reference (`FULL_SHA.test(tip || '')`,
    # JS coerces `undefined`/`null` to `''`) and the harness's PARITY-PIN copy
    # block cleanly. Pinned by the "evidence tip is null" case in
    # tests/test_exit_gate.py's case matrix.
    if (
        not _FULL_SHA_RE.match(evidence.tested_tip or "")
        or cert.certified_tip != evidence.tested_tip
    ):
        return "exit-gate tip not a full canonical SHA or cert/evidence tip mismatch"
    return None


def runner_report_block_reason(
    xg: RunnerReportShape | None, tip: str | None, exit_cmd: str
) -> str | None:
    """Deterministic BLOCKED predicate over a relayed runner report — the
    single source of truth this repo would need for "may this phase reach
    SUBMITTED?" once a harness consumes it. Ported standalone from
    ``exit-gate.ts``'s ``runnerReportBlockReason``; AGX's own JS-side harness
    carries a behaviorally identical copy between ``PARITY-PIN`` markers, but
    that harness is not part of this repo (see this module's docstring), so
    there is no in-repo second copy to check live agreement against yet.
    ``tests/test_exit_gate.py`` pins THIS function's behavior against a case
    matrix transcribed from ``exit-gate.ts``'s own test — see that file's
    module docstring for the exact, non-overclaiming scope of what "parity"
    means today versus what it would mean once a harness exists here.
    """
    if xg is None or xg.ok is not True:
        error_payload: object = (
            xg.error if (xg is not None and xg.error is not None) else _NO_RELAY_MSG
        )
        return f"exit gate did not certify (missing or ok!==true): {_json_compact(error_payload)}"
    if xg.cert is None or xg.evidence is None:
        return "ok:true without cert+evidence — non-canonical runner output"
    issue = _evidence_issue(xg.evidence, xg.cert, exit_cmd)
    if issue is not None:
        return issue
    if not _FULL_SHA_RE.match(tip or ""):
        return f"phase tip is not a full 40-hex SHA: {tip or 'missing'}"
    if xg.evidence.tested_tip != tip:
        return f"exit-gate tip mismatch: certified {xg.evidence.tested_tip} but phase tip is {tip}"
    return None
