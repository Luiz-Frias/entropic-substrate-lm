"""Assemble a double-blind review packet for a rung (docs/peer-review-protocol.md).

Operator-side tool. It relabels the arms, SCRUBS arm-identifying fields from
every record of the captured JSONLs (filename relabeling alone is not
blinding: ``run_start`` carries the arm axis and every record carries
``run``), emits a self-contained reader with all prose stripped
(docstrings/comments carry project vocabulary), and writes ``manifest.txt``.
The arm mapping is written OUTSIDE the packet (``--sealed-out``) and
disclosed only after the review is filed.

Per-rung differences live in a ``RungSpec`` (the ``RUNGS`` registry below) —
arms, which config keys carry the arm axis, which records fields must be
dropped, the banned vocabulary, and the packet prose. The engine is shared;
selecting a rung is ``--rung``.

Fail-loud gates, all of which abort assembly:

1. **Banned-substring scan** over every emitted file. A packet that leaks arm
   identity is worse than no packet — the review it produces looks blind and
   is not.
2. **Completeness** — the captures must yield exactly one file per (arm,
   seed) named by the spec. A missing or duplicated pair is never silently
   dropped.
3. **Parity** — post-scrub, the *blinded* arms' ``run_start`` records must be
   identical outside the declared exemptions. This is the gate that catches a
   NEW arm-identifying field appearing in the schema: E1's arms were
   parameter-identical, but E1.5a's differ in ``params`` (85.33M untied vs
   85.13M tied), which alone would unblind the packet.
4. **Attestation presence** — for a rung declaring ``attestation_required``,
   every ``run_start`` must carry all three freeze-attestation keys. NOT a
   uniformity rule: the E1.5a A/C extension mixes pre- and post-attestation
   captures inside one arm, which is legitimate.

The freeze attestation (review finding MAJOR-1) is the one channel that is
DELIBERATELY not equalized: the scrub removes every freeze-axis config field,
so without it a reviewer cannot verify that the arms froze different parameter
subsets at all. ``trainable_param_count``/``frozen_param_count`` pass through
raw; ``frozen_mask_digest`` is rekeyed under a per-assembly HMAC salt written
only to ``--sealed-out``, because a raw sha256 over parameter names is
dictionary-attackable.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import json
import re
import secrets
import shutil
import subprocess  # nosec B404 -- ruff-normalize the packet only, fixed argv below
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, TypeVar

# Substrings that must not survive into any packet file (case-insensitive).
# Shared across rungs; a spec adds its own axis vocabulary via banned_extra.
BANNED_BASE = ("entropic", "fisher", "random_sphere", "sqrt", "e1-", "e3", "ladder", "anchor")


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the comparison.

    ``run_name`` is the captured run name, which is also the capture
    filename's prefix (``<run_name>-<run_id8>-<pod_name>.jsonl``). Matching on
    that prefix — longest first — is what lets ``e1-5c-field`` and
    ``e1-5c-field-matched-total`` coexist in one capture directory.

    ``blinded=False`` marks an arm whose ROLE is disclosed in the packet prose
    (E1.5a's control: the paired gaps are signed against it, so its role
    cannot be hidden). Its arm-axis values are still scrubbed — disclosure is
    of role, never of vocabulary.
    """

    label: str
    run_name: str
    blinded: bool = True


@dataclass(frozen=True)
class RungSpec:
    """Everything that varies between rungs. The engine reads only this."""

    # Unblinded freeze attestation (review finding MAJOR-1): the run_start
    # RECORD keys that are EXPECTED to differ between arms, because
    # distinguishing the arms is exactly what they are for. Structural, not
    # per-rung — a rung cannot opt out of the finding's resolution, so this is
    # a class constant rather than an entry in ``parity_exempt``.
    ATTESTATION_KEYS: ClassVar[tuple[str, ...]] = (
        "trainable_param_count",
        "frozen_param_count",
        "frozen_mask_digest",
    )

    name: str
    arms: tuple[ArmSpec, ...]
    seeds: tuple[str, ...]
    abstract: str
    preregistration: str
    sealed_slug: str
    sealed_mapping_note: str
    manifest_title: str
    # config.model key -> replacement token; "{arm}" interpolates the label.
    # A token WITHOUT "{arm}" is deliberate: the value is shared across arms,
    # so a per-arm token would manufacture a difference the parity gate then
    # reports as a leak.
    value_scrub: Mapping[str, str] = field(default_factory=dict)
    model_drop: tuple[str, ...] = ()
    record_drop: tuple[str, ...] = ()
    banned_extra: tuple[str, ...] = ()
    # config keys (top level) exempt from the blinded-arm parity assertion.
    parity_exempt: tuple[str, ...] = ("name",)
    # True only for a rung whose runs were ALL captured after the freeze
    # attestation landed. Not a uniformity rule: the E1.5a A/C extension
    # legitimately mixes pre- and post-attestation captures inside ONE arm,
    # so presence can be required per-rung but never inferred per-packet.
    attestation_required: bool = False
    extension_registration: str | None = None

    @property
    def banned(self) -> tuple[str, ...]:
        return BANNED_BASE + self.banned_extra

    def resolve_arm(self, capture_name: str) -> ArmSpec | None:
        """Longest-prefix match, so a shorter run name cannot shadow a longer."""
        for arm in sorted(self.arms, key=lambda a: -len(a.run_name)):
            if capture_name.startswith(arm.run_name + "-"):
                return arm
        return None


# --------------------------------------------------------------------------
# E1 — paired two-arm initialization comparison. Filed 2026-08-18/19; its
# prose is reproduced verbatim so the committed packets regenerate byte-identically.
# --------------------------------------------------------------------------

_E1_ABSTRACT = """\
# Abstract

Paired comparison of two embedding initializations (A, B) for a byte-level
language model trained on enwik8 (~85M parameters, 20,000 steps). Five seed
pairs (seeds 1-5); within a pair both arms share the seed, data order, and
all non-embedding initialization. Primary metric: final nats/byte on a held
out split, lower is better. The registered hypothesis is that arm A attains
lower nats/byte than arm B. This packet contains the pre-registration, the
captured per-run JSONL series, a self-contained validated reader, and a
manifest of file hashes and registration timestamps.
"""

_E1_PREREGISTRATION = """\
# Pre-registration — paired two-arm comparison (arms A, B)

Committed before any confirmatory run; binding. Amendment 1 (epsilon) was
committed after the runs completed but before any analysis or arm-level
readout (timestamps in manifest.txt).

## Hypothesis

H1: arm A's embedding initialization improves final nats/byte over arm B's
(lower is better). With d_s = y_B,s - y_A,s, H1 predicts mean(d) > 0.
Architecture, data, and per-seed conditions are identical between arms; only
the embedding initialization differs.

## Design

- Pairing: same seed s implies identical data order and all non-embedding
  init; evidenced per run by the data_order_digest field (Appendix A). A
  digest mismatch within a pair INVALIDATES the pair (reported, excluded).
- Stage 1 seeds: {1, 2, 3, 4, 5} (df = 4), analyzed once.
- Stage 2 seeds: {6, 7, 8, 9} (cumulative n = 9, df = 8) — run ONLY if the
  stage-1 CI straddles 0.
- Primary metric: final nats/byte, read exclusively via reader.py's
  final_eval_metric(path, "nats_per_byte").
- Estimator: CI = mean(d) +/- t_{0.9875, n-1} * sd(d)/√n — the total 5%
  alpha is split across the two looks (Bonferroni, per-look 2.5%):
  t_{0.9875,4} ~= 3.50 (stage 1), t_{0.9875,8} ~= 2.75 (stage 2, cumulative).
- Verdict rule: CI entirely above 0 -> H1 supported. CI entirely below
  +epsilon -> no practically relevant improvement. Otherwise -> inconclusive
  (stage 2 if not yet run; else reported as inconclusive).
- Exclusion rule (only this one): a run is excluded solely for a non-finite
  final loss; an exclusion always drops the whole PAIR and is reported.
- Pilot declaration: one declared pilot run of arm A (2026-08-17) established
  cost/duration/plumbing. Pilots are never part of the confirmatory set.

## Amendment 1 — epsilon fixed (2026-08-18, post-run / pre-analysis)

epsilon = 0.01 nats/byte. Basis, chosen with NO access to any confirmatory
outcome of this comparison (the amender had seen baseline runs outside this
comparison and the declared pilot): one enwik8 reporting step (0.01 bits/byte
~= 0.0069 nats/byte) rounded up to ~=1.3% of a byte-level baseline outside
this comparison (0.7738 nats/byte). Every other registration term unchanged.

## Appendix A — data_order_digest

SHA-256 over the source offsets of the first 32 training batches, emitted
ONCE as the additive fields data_order_digest + data_order_batches on the
first logged train record at/after the window — or on the final record of a
run shorter than 32 steps, which then attests its actual batch count. Fresh
runs only: a resumed run never attests (absence = not attested, never "same
order").
"""

_E1_EXTENSION = """\

## Extension registration — seeds 6-9 (outcome-aware; committed before any extension run)

The stage-1 confirmatory phase closed with its filed verdict. This extension
was registered AFTER that outcome was known and BEFORE any extension run
launched: seeds {6, 7, 8, 9}, same paired design, same container image
digest, same GPU class, same corpus and configs.

- Primary: cumulative paired analysis over seeds {1..9}. STRONGLY_SUPPORTED
  iff mean(d) / (sd(d)/√9) > t_{0.99865, 8} = 4.277 (the one-sided 99.865%
  lower confidence bound of mean(d) exceeds 0). Otherwise
  NOT_STRONGLY_SUPPORTED. This label is reported ALONGSIDE — never
  replacing — the stage-1 verdict.
- Practical relevance is read against the ratified epsilon = 0.01 nats/byte,
  unchanged.
- Clause precedence (declared after extension runs launched, before any
  completed or was analyzed): the direction label is primary; the
  practical-relevance rider is mandatory on every quotation of the verdict.
- Exclusion rule unchanged (non-finite final loss drops the whole pair);
  pairing rule unchanged (a within-pair data_order_digest mismatch
  invalidates the pair).
"""

_E1_SEALED_NOTE = """\
Mapping: **A = e1-fisher, B = e1-random** (A = registered-favored arm by
protocol construction; the blind hides which initialization A is).

Scrub transform applied to every record: run -> arm-<X>; config.name ->
arm-<X>; config.model.embedding_init -> init-<X>; config.model.
fisher_init_path and config.model.anchor_mode dropped. reader.py =
telemetry.py with all docstrings/comments stripped (ast.unparse), the
package logging import shimmed, and the operator CLI removed.
"""

# --------------------------------------------------------------------------
# E1.5a — frozen-table sufficiency. THREE arms: two blinded freeze variants
# paired against a disclosed control.
#
# What is blindable here is NOT "which arm is the control" — the gaps are
# signed against it, so its role is structural and disclosed. What is blinded
# is WHICH PHYSICAL FREEZE MODE each variant is. Arm A is defined by ROLE, as
# in E1: A := the variant the registration expects to rank >= the other. That
# is exactly the prior a reviewer would otherwise confirm by expectation, so
# sealing it is what gives this blind teeth.
# --------------------------------------------------------------------------

_E15A_ABSTRACT = """\
# Abstract

Paired sufficiency comparison of two embedding-parameter freezing variants
(A, B) against a shared trainable control (C), for a byte-level language
model trained on enwik8 (~85M parameters, 20,000 steps). Four seeds
(1-4); within a seed all three arms share the seed, the data order, and
every non-embedding initialization. Each variant freezes a registered subset
of the embedding parameters at their initialized values; WHICH subset each
of A and B freezes is sealed. C trains all parameters.

Primary metric: the mean of the final five held-out eval readings of
nats/byte, lower is better. The registered question is SUFFICIENCY — whether
freezing costs less than a pre-registered practical-relevance threshold —
not direction. This packet contains the pre-registration, the captured
per-run JSONL series, a self-contained validated reader, and a manifest of
file hashes and registration timestamps.
"""

_E15A_PREREGISTRATION = """\
# Pre-registration — paired sufficiency, two variants (A, B) vs control (C)

Committed BEFORE any confirmatory run of this rung; binding. Any analysis
deviating from it must say so explicitly and is exploratory by definition.

## Question

A prior filed comparison established that this model's embedding
initialization carries real, confidently-signed predictive structure, with a
mandatory co-equal rider that the terminal gap was below the practical
threshold. A subsequent endpoint readout measured 0.840 Pearson retention of
the initialized relational structure (Gram matrix) in the TRAINED embedding
table — unconstrained training keeps 84% of the induced structure on its own.

This rung asks the sharpest follow-up: **was the remaining 16% of adaptation
necessary?** If a frozen table performs on par with a trainable one, the
initialized geometry carries the durable predictive structure. If freezing
fails, the initialization is a launch ramp rather than a substrate.

## Arms and seeds

| Arm | Description | Seeds |
|---|---|---|
| A | freezes a registered subset of the embedding parameters (sealed) | {1,2,3,4} |
| B | freezes a different registered subset (sealed) | {1,2,3,4} |
| C | control: all parameters trainable; PREVIOUSLY CAPTURED, no new runs | {1,2,3,4} |

Arm A is defined by ROLE: it is the variant the registration expects to rank
at least as high as B on the verdict ordering below (it leaves strictly more
capacity free to adapt). Which physical freezing subset A is, is sealed.

Seeds {1,2,3,4} are fixed by an ordinal rule (the lowest-numbered seeds of
the prior stage-1 set). **Outcome-awareness disclosure:** the designer had
read the per-seed control outcomes, since the prior comparison was already
filed. The ordinal rule removes seed-level discretion, and the seed choice
cannot bias a paired gap in any case: the variant arms were unrun at
registration, and under pairing the estimand is the within-seed difference,
not the control's level.

A third variant (freezing a further subset) is DEFERRED-TO-LAST, not
excluded: it runs only if the Trigger below fires. It is not part of this
packet.

## Primary metric — last-k(=5) eval mean

Per run: the mean of nats_per_byte over the FINAL FIVE eval records (steps
18000-20000 at eval_interval 500), read exclusively via reader.py's
read_eval_series on the captured JSONLs — never a live tracking server. The
k = 1 terminal reading (the prior rung's endpoint) is reported alongside for
continuity.

This endpoint was adopted from the prior rung's blind review, which found
single-eval jitter to be roughly 40% of pair variance, and from that rung's
k-sensitivity table, in which the margin WIDENED under smoothing.

## Estimator and decision rule

Paired gaps per variant, per seed: g_s = y_variant,s - y_C,s (positive =
freezing costs nats/byte). n = 4, df = 3. Two registered variants, so the
total 5% alpha is Bonferroni-split: per-variant TWO-SIDED CI at 97.5%
(t_{0.9875,3} = 4.177), ONE look (no stage 2).

**Clause precedence (declared here):** this rung's registered question is
SUFFICIENCY, not direction, so — deliberately INVERTING the prior rung's
precedence — the epsilon-clause is the PRIMARY verdict label and the
direction-vs-0 clause is the MANDATORY CO-EQUAL rider. Both are always
reported; quoting either without the other misquotes this registration.
epsilon = 0.01 nats/byte, unchanged from the prior rung and ratified there.

Per-variant verdict vocabulary:

- **GEOMETRY-SUFFICIENT** — CI upper bound < epsilon: freezing costs less
  than the practical-relevance threshold; the frozen geometry carries the
  structure.
- **GEOMETRY-INSUFFICIENT** — CI entirely above epsilon: freezing
  practically hurts.
- **INCONCLUSIVE** — otherwise.
- Rider (direction): the CI's position versus 0, reported verbatim (entirely
  below 0 = frozen strictly better; contains 0 = no detectable cost; entirely
  within (0, epsilon) = detectable cost below practical relevance).

**Power context** [projected from the prior rung's measured cumulative
sd(d) = 0.00300307 at n = 9; DECLARED ASSUMPTION: the frozen-vs-trainable
within-pair sd is of the same order as that measured between-arm sd — no
direct measurement of it existed at registration]: se = sd/√4 ~= 0.0015, CI
half-width ~= 4.177 x 0.0015 ~= 0.0063. If freezing truly costs ~= 0, the
expected upper bound ~= +0.0063 < epsilon, readable as SUFFICIENT. If the sd
doubles, half-width ~= 0.0125, giving INCONCLUSIVE — an informative
heterogeneity result, not a nullity.

## Trigger — when the deferred third variant runs

Order the labels GEOMETRY-SUFFICIENT > INCONCLUSIVE > GEOMETRY-INSUFFICIENT.
The expected pattern is label(A) >= label(B). The trigger is the STRANGE
DIVERGENCE label(A) < label(B): the variant with strictly more free capacity
reading WORSE than the more constrained one. Only then does the deferred
variant run, to isolate which part of the registered difference drives the
inversion. Under any other outcome it does not run under this registration.

## Regime and hyperparameter scope (declared)

All arms run at the SHARED configuration of the prior rung (lr 6e-4, batch
16 x seq 1024, cosine to 20k steps), modulo the freeze axis. The registered
estimand is therefore the ARM EFFECT AT THIS FIXED CONFIGURATION, not at each
arm's individually tuned frontier. Published work (arXiv:2608.11859) shows
that at <=100M parameters comparisons can be confounded by per-arm
hyperparameter sensitivity — an arm can lose at a shared configuration purely
by being harder to tune. The exposure here is bounded: all arms share the
backbone architecture and differ only in which parameters receive gradients,
so their tuning response is expected to be close. But freezing does change
effective trainable capacity, so this remains a DECLARED LIMITATION of the
estimand, not an assumption of equivalence. Verdicts are scoped to this
configuration and this ~85M/enwik8 regime.

## Pairing evidence — data_order_digest

SHA-256 over the source offsets of the first 32 training batches, emitted
ONCE as the additive fields data_order_digest + data_order_batches on the
first logged train record at/after the window. A within-seed mismatch across
arms INVALIDATES that seed (reported, excluded). Fresh runs only: a resumed
run never attests (absence = not attested, never "same order").

## Exclusion rule

A run is excluded solely for a non-finite final loss; an exclusion always
drops the whole SEED across all arms, and is reported.

## Note on the control arm's schema

Arm C was captured under an earlier revision of the training code, before the
freeze axis existed. Its records may therefore lack fields that A and B
carry. This is expected and is not evidence about any arm: the fields
distinguishing A from B have been removed from ALL arms by the packet's
scrub, and the packet asserts, as a build-time gate, that A's and B's
run_start configurations are IDENTICAL after scrubbing.

The scrub additionally removes configuration keys belonging to OTHER,
unrelated comparisons that share this codebase. Every arm here is identical
on those keys and all of them are INACTIVE in this rung (they configure
machinery none of these runs use), so their removal withholds nothing
relevant to the registered comparison; they are removed because their default
values name vocabulary that would identify the project.
"""

_E15A_SEALED_NOTE = """\
Mapping: **A = e1-5a-input (input-only freeze; untie entailed), B =
e1-5a-both (freeze-both, tied), C = e1-fisher (trainable control)**.

A is the registered expected-`>=` variant by protocol construction (it frees
strictly more capacity to adapt — a trainable output classifier); the blind
hides WHICH physical freeze subset A is, which is exactly the prior a
reviewer would otherwise confirm by expectation. C's ROLE is disclosed in the
packet prose — the paired gaps are signed against it, so it cannot be
hidden — but its arm vocabulary is scrubbed like every other arm's.

Scrub transform applied to every record: run -> arm-<X>; config.name ->
arm-<X>; config.model.embedding_init -> the SHARED token init-shared (all
three arms share one initialization here, so a per-arm token would
manufacture a difference); config.model.embedding_freeze,
config.model.tie_embeddings, config.model.fisher_init_path and
config.model.anchor_mode DROPPED; the top-level run_start field `params`
DROPPED.

`params` is the field E1's packet never had to scrub: E1's arms were
parameter-identical, whereas untying doubles the embedding table (85.33M
untied vs 85.13M tied) and that single integer would have unblinded the
packet on its own. reader.py = telemetry.py with all docstrings/comments
stripped (ast.unparse), the package logging import shimmed, and the operator
CLI removed.
"""


RUNGS: dict[str, RungSpec] = {
    "e1": RungSpec(
        name="E1",
        arms=(ArmSpec("A", "e1-fisher"), ArmSpec("B", "e1-random")),
        seeds=("1", "2", "3", "4", "5", "6", "7", "8", "9"),
        abstract=_E1_ABSTRACT,
        preregistration=_E1_PREREGISTRATION,
        extension_registration=_E1_EXTENSION,
        sealed_slug="e1-review-sealed-mapping",
        sealed_mapping_note=_E1_SEALED_NOTE,
        manifest_title="E1 review packet manifest",
        value_scrub={"embedding_init": "init-{arm}"},
        model_drop=("fisher_init_path", "anchor_mode"),
    ),
    "e1-5a": RungSpec(
        name="E1.5a",
        arms=(
            ArmSpec("A", "e1-5a-input"),
            ArmSpec("B", "e1-5a-both"),
            ArmSpec("C", "e1-fisher", blinded=False),
        ),
        seeds=("1", "2", "3", "4"),
        abstract=_E15A_ABSTRACT,
        preregistration=_E15A_PREREGISTRATION,
        sealed_slug="e1-5a-review-sealed-mapping",
        sealed_mapping_note=_E15A_SEALED_NOTE,
        manifest_title="E1.5a review packet manifest",
        # Shared token, NOT per-arm: all three arms use one initialization, so
        # "init-{arm}" would fabricate a difference the parity gate then flags.
        value_scrub={"embedding_init": "init-shared"},
        model_drop=(
            "embedding_freeze",
            "tie_embeddings",
            "fisher_init_path",
            "anchor_mode",
            # INERT cross-rung keys. Every E1.5a arm is table-sourced, so the
            # field lane is entirely inactive here — but its DEFAULT value
            # carries banned vocabulary ("sqrt_prob"), and the control was
            # captured before these keys existed, so they are also a schema
            # asymmetry. Caught by the banned scan on first assembly
            # (2026-08-22): a later rung's default leaking into an earlier
            # rung's packet is a standing hazard, not a one-off.
            "state_source",
            "field_state",
            "field_mode",
            "field_loss_lambda",
            "field_ckpt_path",
            "field_estimator",
            "field_n_layers",
            "field_d_model",
            "field_n_heads",
            "gram_lambda",
            "gram_reference",
        ),
        record_drop=("params",),
        banned_extra=("embedding_freeze", "tie_embeddings", "input-only", "freeze-both"),
    ),
}


_READER_HEADER = '"""Validated reader for the packet\'s JSONL run series.\n\nProse (docstrings/comments) was stripped from the original for blinding;\nlogic is unmodified. Parse the data ONLY through these functions.\n"""\n'
_LOGGING_SHIM = (
    "import logging\n"
    'logger = logging.getLogger("reader")\n'
    "def configure_logging() -> None:\n"
    "    logging.basicConfig(level=logging.INFO)\n"
)


def md5(path: Path) -> str:
    # integrity manifest, not crypto — declared via usedforsecurity
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def scrub_record(record: dict[str, Any], arm: str, spec: RungSpec) -> dict[str, Any]:
    """Map arm-identifying values to blind labels; drop arm-axis keys."""
    out = dict(record)
    if "run" in out:
        out["run"] = f"arm-{arm}"
    for key in spec.record_drop:
        out.pop(key, None)
    config = out.get("config")
    if isinstance(config, dict):
        config = json.loads(json.dumps(config))  # deep copy
        if "name" in config:
            config["name"] = f"arm-{arm}"
        model = config.get("model", {})
        for key, token in spec.value_scrub.items():
            if key in model:
                model[key] = token.format(arm=arm)
        for key in spec.model_drop:
            model.pop(key, None)
        out["config"] = config
    return out


def rekey_attestation(record: Mapping[str, Any], salt: str) -> dict[str, Any]:
    """Replace the raw frozen-parameter digest with an HMAC under ``salt``.

    The train-time digest is a sha256 over frozen parameter names and shapes
    (backbone.py ``parameter_attestation``). Passed through raw it is
    dictionary-attackable: a reviewer who guesses the architecture enumerates
    candidate frozen masks, hashes them, and reads the arm off the packet.
    HMAC under a per-assembly salt kills preimage enumeration while keeping
    the only structure the reviewer needs — equality: same arm and axis maps
    to the same value, different masks to different values.

    The salt lives ONLY in the sealed mapping file, outside the packet, so it
    is disclosed on the same schedule as the arm mapping itself.

    The counts pass through unmodified: the reviewer asked for counts, and
    coarse parameter integers identify no project. Absence of the digest is a
    no-op — pre-attestation captures (E1, E1.5a) carry none, and a rung whose
    arms MUST carry one says so via ``attestation_required``
    (:func:`assert_attestation_present`).
    """
    out = dict(record)
    if "frozen_mask_digest" not in out:
        return out
    raw = out["frozen_mask_digest"]
    if not isinstance(raw, str):
        # Present-but-malformed is the one shape that would reach the packet
        # unrekeyed: assert_uniform_attestation counts it as attested and the
        # parity gate exempts it, so nothing downstream would catch it.
        msg = f"non-string frozen_mask_digest {raw!r} — corrupt attestation, refusing to assemble"
        raise AssertionError(msg)
    out["frozen_mask_digest"] = hmac.new(
        salt.encode("utf-8"), raw.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return out


def assert_attestation_present(records: Mapping[str, Mapping[str, Any]], spec: RungSpec) -> None:
    """Under ``attestation_required``, every ``run_start`` must carry all three keys.

    Deliberately NOT a whole-packet uniformity rule. The E1.5a A/C extension
    mixes pre-attestation captures (seeds 1-4) with post-attestation reruns
    INSIDE one arm, so neither whole-packet nor per-arm uniformity is
    satisfiable there — requiring it would refuse a legitimate packet.

    A rung whose runs were ALL captured after the attestation landed declares
    ``attestation_required=True``, and a missing key is then an assembly
    error naming the offending run, never a silent omission.
    """
    if not spec.attestation_required:
        return
    incomplete = {
        name: sorted(set(RungSpec.ATTESTATION_KEYS) - set(record))
        for name, record in records.items()
        if not set(RungSpec.ATTESTATION_KEYS) <= set(record)
    }
    if incomplete:
        msg = (
            f"{spec.name}: attestation_required, but these runs are missing "
            f"attestation keys: {dict(sorted(incomplete.items()))}"
        )
        raise AssertionError(msg)


_StripNode = TypeVar("_StripNode", ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class _DocstringStripper(ast.NodeTransformer):
    def _strip(self, node: _StripNode) -> _StripNode:
        self.generic_visit(node)
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:] or [ast.Pass()]
        node.body = body
        return node

    visit_Module = _strip
    visit_FunctionDef = _strip
    visit_AsyncFunctionDef = _strip
    visit_ClassDef = _strip


_PACKET_READER_EXCLUDED_FUNCTIONS = {
    "_build_parser",
    "main",
    "read_d1_terminal_defect",
}
_PACKET_READER_EXCLUDED_IMPORT_PREFIXES = (
    # p13-sweep, B2 row 7 (P12-D5 cut, E-4 door): the two entries this tuple
    # carried at the old prefix are rewritten to their landed addresses.
    # They are pinned verbatim by tests/test_cli_surface.py's
    # _FROZEN_LITERAL_SITES_PATH.
    "entropic_substrate.ladder.capture.reader.d1_",
    "entropic_substrate.ladder.capture.records.d1_",
)


def _is_shim_source(source: str) -> bool:
    """True if ``source`` is (structurally) a generated alias shim: the
    ``import <target> as _target`` / ``sys.modules[__name__] = _target``
    rebind ``tools/gen_shims.py`` emits, with or without its leading
    entrypoint ``if __name__ == "__main__":`` guard.

    A reader built from a source this shallow degrades to a no-op that
    still "succeeds" (FOR-1524, MEASURED 2026-09-20: p3-quarantine's move
    made ``src/entropic_substrate/telemetry.py`` exactly this shape, and
    the bare ``import entropic_substrate.__legacy.telemetry as _target``
    line survives ``build_reader``'s ImportFrom-only filter, leaking the
    banned ``entropic`` token into the packet instead of failing loud).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    body = tree.body
    if body and isinstance(body[0], ast.If):
        body = body[1:]
    if len(body) != 3:
        return False
    s0, s1, s2 = body
    imports_sys = (
        isinstance(s0, ast.Import)
        and len(s0.names) == 1
        and s0.names[0].name == "sys"
        and s0.names[0].asname is None
    )
    imports_target = (
        isinstance(s1, ast.Import) and len(s1.names) == 1 and s1.names[0].asname == "_target"
    )
    rebinds = (
        isinstance(s2, ast.Assign)
        and len(s2.targets) == 1
        and isinstance(s2.targets[0], ast.Subscript)
        and isinstance(s2.targets[0].value, ast.Attribute)
        and s2.targets[0].value.attr == "modules"
        and isinstance(s2.targets[0].value.value, ast.Name)
        and s2.targets[0].value.value.id == "sys"
        and isinstance(s2.value, ast.Name)
        and s2.value.id == "_target"
    )
    return imports_sys and imports_target and rebinds


def build_reader(telemetry_source: str, *, source_path: Path | None = None) -> str:
    """Strip prose and D1-only imports from the legacy packet reader.

    Also drops the operator CLI (``main``/``_build_parser``/``__main__``
    block) — its argparse prog/help strings carry repository names, and the
    reviewer parses through the functions, never argv. D1 evidence readers
    live in separate modules and are not dependencies of the legacy E-series
    packet reader; copying their package imports would both break standalone
    execution and leak repository identity into a blinded packet.

    Refuses a shim-form ``telemetry_source`` outright (FOR-1524): a literal
    alias shim can never become a packet reader, whether reached through the
    default resolution or an explicit ``--telemetry`` override.
    """
    if _is_shim_source(telemetry_source):
        where = f" ({source_path})" if source_path is not None else ""
        msg = (
            f"refusing to build a review-packet reader from a generated alias "
            f"shim{where} -- it has no real telemetry code to read; point "
            "--telemetry at the real module (FOR-1524)"
        )
        raise ValueError(msg)
    tree = _DocstringStripper().visit(ast.parse(telemetry_source))
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.FunctionDef) and node.name in _PACKET_READER_EXCLUDED_FUNCTIONS
        )
        and not (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith(_PACKET_READER_EXCLUDED_IMPORT_PREFIXES)
        )
        and not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        )
        # __all__ would dangle names of the dropped CLI (ruff F822)
        and not (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        )
    ]
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    # p13-sweep, B2 row 18: the old-spelling strip
    # ("from entropic_substrate.logging import configure_logging, logger")
    # retired as dead code -- no source under the retired namespace exists
    # to produce that string any more. The landed spelling below is the
    # only one any reader source can carry now.
    code = code.replace(
        "from entropic_substrate.kernel.runtime.logging import configure_logging, logger",
        _LOGGING_SHIM,
    )
    return _READER_HEADER + code + "\n"


# Long pure-hex runs are masked before the substring scan. Every token in
# the banned vocabulary except "e3" carries a non-hex character and cannot
# occur inside one; "e3" inside a digest is two coincidental nibbles, not a
# lexical reference — and at 64 hex chars it appears with p ≈ 22% (63
# positions × 1/256), so digest-bearing packets (data_order_digest, the
# rekeyed frozen_mask_digest) would spuriously abort more often than not.
_HEX_RUN = re.compile(r"[0-9a-f]{32,}")


def banned_scan(path: Path, banned: Sequence[str]) -> list[str]:
    # The "#" placeholder keeps masked runs from splicing their neighbours
    # into a banned word ("e" + masked-run + "3" stays "e#3", never "e3").
    text = _HEX_RUN.sub("#", path.read_text(encoding="utf-8").lower())
    return [word for word in banned if word in text]


def select_captures(captured: Path, spec: RungSpec) -> list[tuple[Path, ArmSpec, str]]:
    """Resolve capture files to (path, arm, seed), fail-loud on incompleteness.

    A capture directory accumulates every rung's runs, so files that match no
    arm of THIS rung are skipped — but the selection is then required to be
    exactly one file per (arm, seed). Silent under-selection is the failure
    mode this gate exists to prevent: a packet missing a seed still assembles,
    still looks complete, and quietly changes n.
    """
    chosen: dict[tuple[str, str], tuple[Path, ArmSpec, str]] = {}
    for src in sorted(captured.glob("*.jsonl")):
        arm = spec.resolve_arm(src.name)
        if arm is None:
            continue
        seed = src.stem.rsplit("-s", 1)[-1]
        if seed not in spec.seeds:
            continue
        key = (arm.label, seed)
        if key in chosen:
            msg = f"duplicate capture for arm {arm.label} seed {seed}: {chosen[key][0].name}, {src.name}"
            raise AssertionError(msg)
        chosen[key] = (src, arm, seed)

    missing = [
        f"{arm.label}-seed{seed}"
        for arm in spec.arms
        for seed in spec.seeds
        if (arm.label, seed) not in chosen
    ]
    if missing:
        msg = f"{spec.name}: no capture for {missing}"
        raise AssertionError(msg)
    return [chosen[(arm.label, seed)] for arm in spec.arms for seed in spec.seeds]


def assert_blinded_parity(records: Mapping[str, dict[str, Any]], spec: RungSpec) -> None:
    """Post-scrub, blinded arms must carry identical run_start RECORDS.

    This is the gate that catches a NEW arm-identifying field entering the
    schema between rungs. Comparing whole records (minus declared exemptions)
    rather than an allow-list means a field nobody thought about is a FAILURE,
    not an omission. The comparison covers the RECORD, not only its ``config``
    — the freeze attestation (MAJOR-1) enters at record top level, and so did
    ``params``, the one field E1's packet never had to scrub.

    A key whose ``value_scrub`` token interpolates ``{arm}`` is a DECLARED
    difference — that token IS the blind label (E1's ``embedding_init`` ->
    ``init-A``/``init-B``) — so it is exempt. A key mapped to a shared token,
    or not mapped at all, must match. That asymmetry is the whole rule: the
    per-arm tokens are the only differences anyone declared, and every other
    difference is a leak by definition.

    Two record-level exemptions are structural, applying to every rung:
    ``run`` (scrubbed to ``arm-<label>``, the record-level analogue of
    ``config.name``) and :attr:`RungSpec.ATTESTATION_KEYS`, whose whole
    purpose is to differ between arms.
    """
    blinded = [arm.label for arm in spec.arms if arm.blinded]
    if len(blinded) < 2:
        return

    per_arm = {key for key, token in spec.value_scrub.items() if "{arm}" in token}
    record_exempt = ("run", *RungSpec.ATTESTATION_KEYS)

    def strip(record: dict[str, Any]) -> dict[str, Any]:
        out = {k: v for k, v in record.items() if k not in record_exempt}
        cfg = out.get("config")
        if isinstance(cfg, dict):
            cfg = {k: v for k, v in cfg.items() if k not in spec.parity_exempt}
            model = cfg.get("model")
            if isinstance(model, dict) and per_arm:
                cfg["model"] = {k: v for k, v in model.items() if k not in per_arm}
            out["config"] = cfg
        return out

    reference_label = blinded[0]
    reference = strip(records[reference_label])
    for label in blinded[1:]:
        other = strip(records[label])
        if other != reference:
            diff = sorted(
                k
                for k in set(reference) | set(other)
                if reference.get(k, "<absent>") != other.get(k, "<absent>")
            )
            detail = {k: (reference.get(k, "<absent>"), other.get(k, "<absent>")) for k in diff}
            msg = (
                f"blinded arms {reference_label} and {label} differ post-scrub "
                f"at {diff} -> {detail}; these fields identify the arm"
            )
            raise AssertionError(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=sorted(RUNGS), default="e1")
    parser.add_argument("--captured", type=Path, default=Path("results/captured"))
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=None,
        help="defaults to the real module behind "
        "entropic_substrate.ladder.capture.reader.telemetry, "
        "resolved THROUGH the module (not a literal repo path) so it follows "
        "the sys.modules rebind to wherever the module actually lives "
        "(FOR-1524)",
    )
    parser.add_argument("--packet-out", type=Path, required=True)
    parser.add_argument("--sealed-out", type=Path, required=True)
    parser.add_argument("--run-times", type=Path, required=True, help="run_id -> times JSON")
    parser.add_argument("--prereg-commit", required=True, help="'<sha> <iso-utc>'")
    parser.add_argument("--amendment-commit", default=None, help="'<sha> <iso-utc>'")
    parser.add_argument(
        "--ext-prereg-commit",
        default=None,
        help="'<sha> <iso-utc>' — include the extension registration (n=9 packet)",
    )
    parser.add_argument(
        "--precedence-commit",
        default=None,
        help="'<sha> <iso-utc>' of the clause-precedence declaration",
    )
    parser.add_argument(
        "--abandoned-note",
        action="append",
        default=[],
        help="operator disclosure line(s) for abandoned/incomplete runs "
        "(the complete-run-ledger requirement)",
    )
    args = parser.parse_args()
    spec = RUNGS[args.rung]

    telemetry_path = args.telemetry
    if telemetry_path is None:
        # Resolve THROUGH the module, never a literal repo path: the
        # sys.modules rebind (tools/gen_shims.py) makes __file__ the real
        # target module wherever it actually lives, and stays correct
        # across a later structural move (FOR-1524).
        import entropic_substrate.ladder.capture.reader.telemetry as _telemetry

        telemetry_path = Path(_telemetry.__file__)

    packet = args.packet_out
    if packet.exists():
        shutil.rmtree(packet)
    (packet / "data").mkdir(parents=True)

    run_times = json.loads(args.run_times.read_text(encoding="utf-8"))

    (packet / "abstract.md").write_text(spec.abstract, encoding="utf-8")
    prereg_text = spec.preregistration
    if args.ext_prereg_commit and spec.extension_registration:
        prereg_text += spec.extension_registration
    (packet / "preregistration.md").write_text(prereg_text, encoding="utf-8")
    (packet / "reader.py").write_text(
        build_reader(telemetry_path.read_text(encoding="utf-8"), source_path=telemetry_path),
        encoding="utf-8",
    )

    # One fresh salt per assembly: two packets over the same runs must not
    # share rekeyed digests, or disclosing one sealed mapping de-blinds the
    # other. Written only to --sealed-out, never into the packet.
    packet_salt = secrets.token_hex(32)

    sealed_rows: list[str] = []
    time_rows: list[str] = []
    start_records: dict[str, dict[str, Any]] = {}
    attested_starts: dict[str, dict[str, Any]] = {}
    for src, arm, seed in select_captures(args.captured, spec):
        dst = packet / "data" / f"{arm.label}-seed{seed}.jsonl"
        with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line in fin:
                record = json.loads(line)
                scrubbed = scrub_record(record, arm.label, spec)
                if record.get("event") == "run_start":
                    scrubbed = rekey_attestation(scrubbed, packet_salt)
                    attested_starts[f"{arm.label}-seed{seed}"] = scrubbed
                    start_records.setdefault(arm.label, scrubbed)
                fout.write(json.dumps(scrubbed) + "\n")
        run_id8 = src.name[len(arm.run_name) + 1 :][:8]
        full = next((k for k in run_times if k.startswith(run_id8)), None)
        if full is None:
            msg = f"no tracking-server times for {src.name} (id prefix {run_id8})"
            raise LookupError(msg)
        for stamp_key in ("start", "end"):
            stamp = run_times[full][stamp_key]
            iso = datetime.fromtimestamp(stamp / 1000, tz=UTC).isoformat()
            time_rows.append(f"# {dst.name}  {stamp_key}={iso}")
        sealed_rows.append(f"| {src.name} | {md5(src)} | {dst.name} | {md5(dst)} | run_id {full} |")

    assert_attestation_present(attested_starts, spec)
    assert_blinded_parity(start_records, spec)

    # Hook-stability: the repo's lint hooks rewrite any committed .py — if
    # that happened AFTER hashing, the manifest would invalidate the packet
    # (measured on the first commit attempt, 2026-08-18: ruff applied 9
    # autofixes to reader.py post-manifest). Normalize BEFORE hashing.
    # nosec B603 B607 -- fixed argv, packet path from our own CLI arg
    subprocess.run(["ruff", "format", "--quiet", str(packet)], check=True)  # nosec B603 B607
    subprocess.run(  # nosec B603 B607
        ["ruff", "check", "--fix-only", "--quiet", str(packet)], check=True
    )

    files = sorted(p for p in packet.rglob("*") if p.is_file())
    leaks = {str(p): banned_scan(p, spec.banned) for p in files if banned_scan(p, spec.banned)}
    if leaks:
        msg = f"identity leak(s) in packet files: {leaks}"
        raise AssertionError(msg)

    manifest_lines = [f"# {spec.manifest_title}", "#", "# md5  file"]
    manifest_lines += [f"{md5(p)}  {p.relative_to(packet)}" for p in files]
    manifest_lines += [
        "#",
        "# Registration commits (UTC):",
        f"# preregistration committed: {args.prereg_commit}",
        *(
            [f"# amendment 1 (epsilon) committed: {args.amendment_commit}"]
            if args.amendment_commit
            else []
        ),
        *(
            [f"# extension registration committed: {args.ext_prereg_commit}"]
            if args.ext_prereg_commit
            else []
        ),
        *(
            [f"# clause precedence committed: {args.precedence_commit}"]
            if args.precedence_commit
            else []
        ),
        "#",
        "# Run times, operator-supplied from the tracking server (UTC).",
        "# These are asserted metadata, not raw record content:",
        *time_rows,
        *(
            ["#", "# Complete run ledger — abandoned/incomplete runs (operator-declared):"]
            + [f"# {note}" for note in args.abandoned_note]
            if args.abandoned_note
            else []
        ),
    ]
    (packet / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    args.sealed_out.parent.mkdir(parents=True, exist_ok=True)
    args.sealed_out.write_text(
        "---\n"
        f"name: {spec.sealed_slug}\n"
        f'description: "SEALED arm mapping for the {spec.name} review packet — '
        'disclose only after the review is filed (docs/peer-review-protocol.md)"\n'
        "metadata:\n  type: project\n---\n\n"
        f"# SEALED — {spec.name} review packet arm mapping\n\n"
        f"{spec.sealed_mapping_note}\n"
        "## Freeze-attestation salt\n\n"
        "HMAC-SHA256 key the packet's `frozen_mask_digest` values were rekeyed\n"
        "under; recomputing a digest from a candidate frozen-parameter mask "
        "needs it.\nFresh per assembly, so it de-blinds THIS packet and no "
        "other.\n\n"
        f"    {packet_salt}\n\n"
        "| capture file | capture md5 | packet file | packet md5 | run |\n"
        "|---|---|---|---|---|\n" + "\n".join(sealed_rows) + "\n",
        encoding="utf-8",
    )
    print(f"packet: {packet} ({len(files)} files) | sealed: {args.sealed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
