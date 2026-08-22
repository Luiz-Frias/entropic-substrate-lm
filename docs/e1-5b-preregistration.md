# E1.5b pre-registration — Gram-preserving regularization (two-stage: sweep → confirm)

Committed BEFORE any E1.5b run, per `docs/rung-lifecycle-protocol.md`.
Binding design; deviations are exploratory by definition and must say so.

> **DECISION-NEEDED (operator, before any E1.5b pod launches):** the λ grid
> {0.01, 0.1, 1.0}, the retention threshold 0.95, and the stage-C seed set
> {3, 4, 5, 6} below are the executing agent's recommendation
> (2026-08-20) — FOR-904's grid ruling was never recorded, unlike
> E1.5a/E1.5c's (FOR-900/FOR-907). Ratify or adjust via a dated amendment
> to this file; every other term is fixed now. E1.5b pods follow the E1.5a
> wave and E1.5c pilots regardless, so this blocks nothing today.

## Question

E1.5-D measured **0.840 Pearson Gram retention** on the trained fisher
table at λ = 0 (`docs/verdicts/e1-5d-readout.md`): training keeps 84% of the
init's relational geometry unforced. E1.5b applies
`L = L_CE + λ‖G(E_t) − G(E_fisher)‖_F²` (merged, `geometry/gram.py`) and asks
the dose-response question: **does pushing retention from 0.84 toward 1.0
cost or gain nats/byte?** Three readings the sweep separates: helpful
inductive bias (gain), harmful overconstraint (cost), transient warm-start
information (flat loss, rising retention).

## Stage S — selection sweep (exploratory by design)

- λ ∈ {0.01, 0.1, 1.0} × seeds {1, 2} = 6 runs, `configs/e1-5b-gram.yaml`
  with λ via CLI override (the yaml's 0.1 is a PLACEHOLDER, not a registered
  value), full 20k steps.
- Readout per λ: mean paired gap vs the captured `e1-fisher` runs at
  matching seeds (last-5 metric, below), Gram retention from the captured
  final checkpoints (`embedding-geometry` lane, the E1.5-D estimator), and
  the `gram_frob_sq` / `gram_penalty` training trajectories.
- **No confirmatory claim is registered from stage S** (n = 2 per λ carries
  no usable CI). It exists to select λ*.
- **Selection rule (declared, mechanical):** λ* = the λ minimizing the mean
  paired gap, subject to measured retention(λ*) ≥ 0.95; ties break toward
  the smaller λ. If NO λ reaches retention 0.95 → **registered STOP**: the
  rung reports the dose-response as its outcome (characterization, no
  confirmatory stage) — retention that cannot be bought at any swept λ is
  itself the answer.

## Stage C — confirmatory (runs only if stage S selects a λ*)

- λ* × seeds {3, 4, 5, 6} — disjoint from stage S's seeds, so selection
  data never enters the confirmatory estimate.
- Paired gaps `d_s = y_gram(λ*),s − y_fisher,s` (negative = the penalty
  helps) vs the captured `e1-fisher` runs at matching seeds. n = 4, df = 3,
  single look, two-sided 95% CI (`t_{0.975,3} = 3.182`).
- **Clause precedence** (declared per the stage-1 cloud review's MAJOR-1):
  this is a superiority-shaped question, so E1's precedence carries
  unchanged — direction (CI vs 0) is the primary verdict label; practical
  relevance vs ε = 0.01 nats/byte is the mandatory co-equal rider. Quoting
  one without the other misquotes this registration.

## Primary metric — last-k(=5) eval mean

Identical to `docs/e1-5a-preregistration.md`: mean `nats_per_byte` over the
final five eval records (steps 18000–20000), via
`telemetry.read_eval_series` on DVC captures only; k = 1 reported alongside.

## Regime and hyperparameter scope (declared)

All runs use the SHARED E1 configuration (lr 6e-4, batch 16 × seq 1024,
cosine to 20k); the estimand is the λ effect at this fixed configuration
(Lourie et al. 2026, arXiv:2608.11859 — small-scale comparisons are
confounded by per-arm tuning response). Named interaction: the penalty's
gradient scale rides the shared lr, so the λ sweep IS a one-dimensional
hyperparameter search along the new axis at fixed lr — λ* is
lr-conditional by construction, and the registration claims nothing about
other learning rates. Terminal (fully decayed) readings only; no
mid-schedule budget estimand exists on this rung.

## Pairing, exclusion, pilots, probes

- Pairing, digest evidence, and the non-finite-pair-drop exclusion rule:
  identical to `docs/e1-5a-preregistration.md`.
- Pilots: none (cost/duration carry from the measured E1 fleet; the Gram
  seam's overhead is bounded — 256² fp32 Grams, guarded byte-only).
- Predeclared probes (report, never decide): retention-vs-λ curve including
  the λ = 0 anchor (measured 0.840); `gram_penalty` share of total loss over
  training; per-eval-step gap trajectory; k-sensitivity (k ∈ {1, 3, 5});
  steps to L* = 0.80 (directional).

## Execution gate, integrity, analysis

The commit carrying this file registers the `gram_lambda` axis in
`config.E15_PREREGISTRATIONS` (FOR-914 gate; `ENTROPIC_ALLOW_UNGATED=1` for
deliberate ungated runs). Integrity chain, fresh-session analysis, blind
review with sealed mapping, and verdict artifacts
(`docs/verdicts/e1-5b-verdict.{md,json}`) all as in
`docs/e1-5a-preregistration.md`.

## Cost (measured $1.59/hr A100-SXM4-80GB, ~65 min/run)

Stage S: 6 × ~$1.7 ≈ $10. Stage C: 4 × ~$1.7 ≈ $7.

## Amendment 1 — operator ratification (2026-08-21, pre-run)

The operator ratifies the three DECISION-NEEDED terms exactly as drafted:
λ grid **{0.01, 0.1, 1.0}**; retention threshold **0.95**; stage-C seed set
**{3, 4, 5, 6}**. Recorded before any E1.5b run exists (the arm has never
been launched — the execution gate held it), with no arm-level E1.5b
outcome in existence to condition on. Every other registration term is
unchanged; the banner above is resolved by this amendment and retained
verbatim for the record. Ratification also recorded on FOR-904
(2026-08-21).
