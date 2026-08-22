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
