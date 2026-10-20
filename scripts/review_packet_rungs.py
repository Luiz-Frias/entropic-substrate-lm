"""Rung registry for ``assemble_review_packet.py``: specs, prose, and ``RUNGS``.

Pure data and declarations, split out of the entry script by FOR-1609 because
the refactor pushed that script to 1038 lines (948 before it, measured with
``wc -l``). Nothing here touches the filesystem or assembles a packet: the
engine (scrub, parity, reader build, ``main``) stays in the entry script,
which re-exports every public name below so the file-path-loaded module keeps
exposing ``RUNGS``, ``ArmSpec``, ``RungSpec`` and ``BANNED_BASE`` unchanged.

The preregistration prose blocks are reproduced verbatim from the filed
packets; editing a character of them changes a regenerated packet.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

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
