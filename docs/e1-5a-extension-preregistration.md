# E1.5a extension pre-registration — input-only seeds 5-7 (outcome-aware)

Committed BEFORE any extension run. This is NOT a stage 2 of
`docs/e1-5a-preregistration.md` — that registration declared **one look, no
stage 2**, and its filed verdict (`docs/verdicts/e1-5a-verdict.md`) stands
untouched: both variants `INCONCLUSIVE` with their riders. This extension is
**outcome-aware by construction**, exactly as `docs/e1-extension-preregistration.md`
was for E1: it is motivated by the blind review's own highest-value item
(`docs/reviews/e1-5a-review-cloud.md`, "What would change my verdict" #1 —
"Two to three additional seeds on A and C … flips A to GEOMETRY-SUFFICIENT
at n = 6") and by its MAJOR-2 finding that the stage-1 design was
underpowered because the assumed within-pair sd (0.00300307, borrowed from
E1's between-arm measurement) came in 1.47-1.59× low. It therefore carries
its OWN registered decision rule — deliberately stricter than stage 1's —
and reports under it, alongside the stage-1 labels, never replacing them.

## Design

- **Arm: `e1-5a-input` only**, seeds **{5, 6, 7}**, config unchanged
  (`configs/e1-5a-input.yaml`: untied entailed, `embedding_freeze: input`),
  one run per pod, A100-SXM4-80GB secure cloud (same class as stage 1 and as
  every capture being paired against). Three pods total.
- **Controls: zero new pods.** The E1 extension already captured
  `e1-fisher` seeds 5-7 at this exact shared configuration
  (`results/captured/e1-fisher-1fca0b15-e1-fisher-s5.jsonl`,
  `…-52658db6-…-s6.jsonl`, `…-44c9e748-…-s7.jsonl`, in DVC). Data order is a
  deterministic function of the seed and the sampler config, which the
  freeze axis never touches, so these captures are the registered control
  side of the new pairs. The pairing claim is verified, not assumed — see
  §Pairing.
- **Seed rule (ordinal):** {5, 6, 7} are the lowest-numbered captured
  `e1-fisher` seeds not used by stage 1. **Outcome-awareness disclosure:**
  the designer has read the per-seed control outcomes AND the stage-1 review,
  including its MINOR-5 observation that A's gap correlates −0.955 with the
  control's own level — which is precisely why seed selection must carry no
  discretion. The ordinal rule fixes the set before any look at which
  controls sit high or low.
- **freeze-both is CLOSED, no new runs.** Its stage-1 mean gap (0.010604)
  sits above ε, so a sufficiency finding is arithmetically unreachable at
  any n (the CI upper bound cannot fall below the mean), and the review
  computed n ≈ 273 for GEOMETRY-INSUFFICIENT. Seeds spent on B answer
  nothing; its stage-1 label and rider stand as filed.
- **The `e1-5a-output` contingency stays closed.** Its trigger was evaluated
  once, at stage 1, as that registration specified; this extension varies
  only n for input-only and does not re-arm it.

## Primary metric and estimator

Unchanged instrument: last-k(=5) eval mean of `nats_per_byte` (steps
18000-20000), read exclusively via `entropic_substrate.telemetry` on
DVC-captured JSONLs, never the live tracking server. Paired gap
`g_s = y_input,s − y_fisher,s`, positive = freezing costs nats/byte.

**Cumulative analysis over seeds {1..7}:** the four stage-1 gaps plus the
three extension gaps, n = 7, df = 6.

## Registered decision rule

- Two-sided **99% CI** on mean(g), critical value `t_{0.995,6} = 3.707`.
  This is deliberately stricter than stage 1's per-variant 97.5%: a
  sufficiency claim from an outcome-aware extension has to clear a higher
  bar than the look it extends, mirroring the E1 extension's 3σ escalation.
- **Clause precedence unchanged from stage 1** (sufficiency primary,
  direction the mandatory co-equal rider) and the same label vocabulary:
  GEOMETRY-SUFFICIENT (CI upper < ε), GEOMETRY-INSUFFICIENT (CI entirely
  above ε), INCONCLUSIVE otherwise. ε = 0.01 nats/byte, unchanged.
- **Rider vocabulary closed over all interval positions** (review MINOR-1:
  stage 1's enumeration missed the region B landed in). The rider reports
  exactly one of: CI entirely below 0; CI contains 0; CI entirely within
  (0, ε); CI entirely above 0 and straddling ε; CI entirely above ε.
- **Power context** [projected from the stage-1 MEASURED A statistics:
  mean 0.0026743, sd 0.0047656]: se = sd/√7 ≈ 0.0018013, half-width ≈
  3.707 × 0.0018013 ≈ 0.00668. At the observed mean the projected upper
  bound is ≈ 0.00935 < ε ⇒ readable as GEOMETRY-SUFFICIENT even at the
  stricter bar. The margin is thin and declared so: the upper bound crosses
  ε if the cumulative sd exceeds ≈ 0.00523 (only ~10% above the stage-1
  observation) or if the cumulative mean drifts up by ≈ 0.0006. An
  INCONCLUSIVE here is therefore a live outcome, and it would be the
  informative result that this design is structurally underpowered at
  ε = 0.01 and must be re-registered on a lower-variance endpoint (review
  item #4), not extended again seed by seed.

## Pairing, exclusion

- **Pairing:** `data_order_digest` equality between each new
  `e1-5a-input-sN` run and the captured `e1-fisher-sN` control, N ∈ {5,6,7}
  — the same check that returned 8/8 at stage 1. A mismatch INVALIDATES the
  pair (reported, excluded). The stage-1 pairs are not re-verified; their
  verification is filed.
- **Exclusion rule (only this one):** a non-finite final loss drops the
  whole pair, always reported.

## Integrity additions (review-driven)

1. **Arm attestation (MAJOR-1).** The extension runs are the first to carry
   the frozen-mask attestation in `run_start`: `trainable_param_count`,
   `frozen_param_count`, and `frozen_mask_digest` (a hash over the
   frozen-parameter mask — names and shapes, never weights — constant
   across seeds within an arm). At packet assembly the digest is re-keyed
   under a packet-scoped salt kept with the sealed mapping, so equality
   structure survives blinding while dictionary enumeration does not.
   Stage-1 captures and the controls predate the attestation and carry no
   such keys — a declared, visible asymmetry, not an error.
2. **Launch-record retention (MINOR-3).** Pod-create timestamps and the
   launcher output for the extension wave are retained as evidence and
   enter the review packet manifest, so an off-schedule launch is
   explainable from the record rather than left as a coincidence for the
   reviewer to weigh.
3. **Chain unchanged:** enwik8 DVC md5 → run configs in `run_start` →
   captured JSONL md5s. The stage-1 review noted `config_hash`/`git_sha`
   arrive null in these records; that gap is inherited, disclosed, and not
   repaired by inference here.

## Predeclared probes (report, never decide)

1. Leave-one-out CI sensitivity at n = 7 (stage 1's rider fragility check,
   review MINOR-6).
2. MINOR-5 replication: is A's gap still monotone in seed index, and does
   the gap-vs-control-level correlation persist at n = 7? n = 4 could not
   test either; n = 7 can, weakly.
3. Within-pair sd, measured at n = 7 — the quantity the stage-1 power
   projection had to assume (review item #4). Reported regardless of label.
4. k-sensitivity of the primary metric, k ∈ {1, 3, 5}.
5. Steps to L* = 0.80 nats/byte for the three new runs (directional only).

## Execution gate, cost

The `embedding_freeze` axis is already registered in
`config.E15_PREREGISTRATIONS` against the stage-1 file; this extension adds
seeds, not an axis, so no gate change is required. Cost: 3 pods × ~65 min ×
$1.59/hr ≈ $5.2 projected from the measured stage-1 wave; launch requires
explicit operator go-ahead against the current balance, per the standing
pod-launch rule.

## Analysis, review, verdict

Fresh analysis session (implementation sessions never analyze); inputs are
DVC captures read through `telemetry.py` only. The review packet is
re-assembled over all seven pairs with a fresh sealed mapping and the
attestation re-keying above; cloud lane per `docs/peer-review-protocol.md`.
Verdict artifact: `docs/verdicts/e1-5a-extension-verdict.{md,json}` carrying
this file's commit, reporting the cumulative n = 7 label + rider alongside
the untouched stage-1 verdict.
