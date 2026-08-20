# E1.5a pre-registration — frozen Fisher table (paired noninferiority vs E1)

Committed BEFORE any E1.5a confirmatory run, per
`docs/rung-lifecycle-protocol.md`. This is the binding design; any analysis
deviating from it must say so explicitly and is exploratory by definition.
Confirmatory analysis happens in a fresh agent session under
`docs/peer-review-protocol.md`. Operator rulings baked in here were recorded
on FOR-900 (2026-08-20) before this document was drafted.

## Question

E1's verdict (filed, `docs/verdicts/e1-extension-verdict.md`): the Fisher
init advantage is real and confidently signed (cumulative n = 9
`STRONGLY_SUPPORTED`, t = 4.59063 > 4.277) with the mandatory co-equal rider
*no practically relevant improvement* (terminal gap below ε = 0.01). The
E1.5-D endpoint readout (`docs/verdicts/e1-5d-readout.md`) measured **0.840
Pearson Gram retention** on the trained fisher table (random arm: 0.017):
unconstrained training keeps 84% of the induced relational structure on its
own. E1.5a asks the sharpest follow-up: **was the 16% adaptation necessary?**

- Frozen table performs on par with trainable → the Fisher geometry itself
  carries the durable predictive structure (a substrate, not a launch ramp).
- Frozen fails where trainable-Fisher won → warm-start reading confirmed:
  good launch geometry, insufficient substrate.

## Arms and seeds (operator-ruled 2026-08-20)

| Arm | Config | Seeds | Pods |
|---|---|---|---|
| freeze-both | `configs/e1-5a-both.yaml` (tied, `embedding_freeze: both`) | {1, 2, 3, 4} | 4 |
| input-only | `configs/e1-5a-input.yaml` (untied entailed, `embedding_freeze: input`) | {1, 2, 3, 4} | 4 |
| control | the nine CAPTURED `e1-fisher` runs — **no new control pods** | pairing at {1, 2, 3, 4} | 0 |
| output-only | `configs/e1-5a-output.yaml` | {1, 2, 3, 4} | contingency only (§Trigger) |

Seed set {1, 2, 3, 4} is fixed by ordinal rule (the lowest-numbered E1
stage-1 seeds). **Outcome-awareness disclosure:** the designer has read the
per-seed `e1-fisher` outcomes (the E1 verdict is filed). The ordinal rule
removes seed-level discretion, and the seed choice cannot bias the paired
gap in any case: the frozen arms are unrun, and under pairing the estimand
is the within-seed difference, not the control's level.

`e1-5a-output` is **DEFERRED-TO-LAST, not excluded** (operator ruling): it
runs only if the §Trigger condition fires, as a disambiguation arm.

## Primary metric — last-k(=5) eval mean

Adopted from the n=9 blind review's endpoint-noise finding (single-eval
jitter ≈ 40% of pair variance; `docs/reviews/e1-n9-review-cloud.md`) and the
E1 extension verdict's k-sensitivity table (the 3σ margin *widens* under
smoothing: t 4.591 → 5.379 at k = 5, all nine paired differences positive at
every k). Per run: the mean of `nats_per_byte` over the **final five eval
records** (steps 18000–20000 at `eval_interval: 500`), read exclusively via
`entropic_substrate.telemetry.read_eval_series` on DVC-captured JSONLs
(`capture-results`) — never the live tracking server. The k = 1 terminal
reading (E1's endpoint) is reported alongside for continuity.

## Estimator and decision rule

Paired gaps per variant, per seed: `g_s = y_variant,s − y_fisher,s`
(positive = freezing costs nats/byte). n = 4, df = 3. Two registered
variants ⇒ the total 5% alpha is Bonferroni-split: per-variant two-sided CI
at 97.5% (`t_{0.9875,3} = 4.177`), one look (no stage 2).

**Clause precedence (declared here, per the stage-1 cloud review's MAJOR-1
requirement that every registration declare it):** this rung's registered
question is *sufficiency*, not direction, so — deliberately inverting E1's
precedence — the ε-clause is the **primary verdict label** and the
direction-vs-0 clause is the **mandatory co-equal rider**. Both are always
reported; quoting either without the other misquotes this registration.
ε = 0.01 nats/byte, unchanged from E1 (Amendments 1–2 there; ratified).

Per-variant verdict vocabulary:

- **GEOMETRY-SUFFICIENT** — CI upper bound < ε: freezing costs less than
  the practical-relevance threshold; the frozen geometry carries the
  structure.
- **GEOMETRY-INSUFFICIENT** — CI entirely above ε: freezing practically
  hurts.
- **INCONCLUSIVE** — otherwise.
- Rider (direction): the CI's position versus 0, reported verbatim
  (entirely below 0 = frozen strictly better; contains 0 = no detectable
  cost; entirely within (0, ε) = detectable cost below practical relevance).

**Power context** [projected from the measured E1 cumulative sd(d) =
0.00300307 (n = 9, `docs/verdicts/e1-extension-verdict.md`); declared
assumption: the frozen-vs-trainable within-pair sd is of the same order as
the measured between-arm sd — no direct measurement of it exists yet]:
se = sd/√4 ≈ 0.0015, CI half-width ≈ 4.177 × 0.0015 ≈ 0.0063. If freezing
truly costs ≈ 0, the expected upper bound ≈ +0.0063 < ε ⇒ readable as
SUFFICIENT. If the sd doubles, half-width ≈ 0.0125 ⇒ INCONCLUSIVE — an
informative heterogeneity result, not a nullity, exactly as in the E1
extension's power framing.

## Trigger — when the output-only contingency arm runs

Order the labels GEOMETRY-SUFFICIENT > INCONCLUSIVE > GEOMETRY-INSUFFICIENT.
The expected pattern is `label(input-only) ≥ label(freeze-both)`: input-only
frees strictly more capacity to adapt (a trainable output classifier).
**The strange divergence** — the operator ruling's trigger, made concrete —
is `label(input-only) < label(freeze-both)`: untying plus a trainable head
reading *worse* than a fully frozen tied table. Only then does
`e1-5a-output` × {1, 2, 3, 4} run, to isolate whether the untie itself or
head-freezing drives the inversion. Under any other outcome, output-only
does not run under this registration.

## Regime and hyperparameter scope (declared)

All arms run at the SHARED E1 configuration (`configs/e1-fisher.yaml` modulo
the freeze axis: lr 6e-4, batch 16 × seq 1024, cosine to 20k steps). The
registered estimand is therefore the **arm effect at this fixed
configuration**, not at each arm's individually tuned frontier. Lourie et
al. 2026 (arXiv:2608.11859) show that at ≤100M parameters comparisons can be
confounded by per-arm hyperparameter sensitivity — an arm can lose at a
shared configuration purely by being harder to tune. The exposure here is
bounded: both arms share the backbone architecture and differ only in which
parameters receive gradients, so their tuning response is expected to be
close — but freezing does change the effective trainable capacity, so this
stays a DECLARED limitation of the estimand, not an assumption of
equivalence. Verdicts are scoped to this configuration and the ~85M/enwik8
regime; no claim is registered about tuned-frontier or larger-scale
behavior.

## Pairing, exclusion, pilots

- **Pairing:** same seed ⇒ identical data order (`embedding_freeze` and
  `tie_embeddings` never touch the sampler); evidenced by
  `data_order_digest` equality between the new run and the CAPTURED
  `e1-fisher` run at the same seed. A mismatch INVALIDATES the pair
  (reported, excluded).
- **Exclusion rule (only this one):** a non-finite final loss drops the
  whole PAIR, always reported.
- **Pilots:** none. Cost, duration, and plumbing carry from the measured E1
  fleet (same configs modulo the freeze axis; ~65 min/run on A100-SXM4-80GB
  at $1.59/hr). No pilot runs precede or enter this registration.

## Predeclared probes (report, never decide)

1. Steps to L* = 0.80 nats/byte per arm per seed (the E1.5c co-primary's
   target; NOT the e0-byte terminal 0.7738, which right-censored 2/9 random
   runs in E1). Directional readout only.
2. Per-eval-step gap trajectory (mirrors the E1 verdict's §4b table).
3. Train-vs-eval nats/byte gap per arm (overfit probe).
4. k-sensitivity of the primary metric, k ∈ {1, 3, 5}.
5. E1.5-D endpoint diagnostics on the new checkpoints: angular drift on a
   frozen table must be exactly 0 (a plumbing check — nonzero drift on a
   frozen table is a DEFECT, cf. `tests/test_freeze.py`'s bitwise pin, not
   a finding), and Gram retention of the TRAINABLE half on `e1-5a-input`.

## Execution gate and integrity chain

The commit carrying this file registers the `embedding_freeze` axis in
`config.E15_PREREGISTRATIONS` (FOR-914): `train` refuses any non-tiny run
with `embedding_freeze != "none"` unless this registration is committed
(`ENTROPIC_ALLOW_UNGATED=1` for deliberate ungated runs;
`tests/test_execution_gate.py` pins the map entry to this file's existence).

Chain, unchanged from E1: `data/enwik8.dvc` md5
`7722de7ace6575299cd8ab788c2fd201.dir` → `config_hash` param/tag per run →
`git_sha` param per run → container image digest recorded in the launch
record at pod-create time (identical across all pods of the wave) → run
artifacts → DVC-captured JSONL md5s.

## Analysis, review, verdict

Fresh analysis session (implementation sessions never analyze); inputs are
DVC captures read through `telemetry.py` only. Blind review packet with arm
relabeling and a sealed mapping per `docs/peer-review-protocol.md`, cloud
lane (`scripts/blind_review_cloud.py`). Verdict artifact:
`docs/verdicts/e1-5a-verdict.{md,json}` carrying `preregistration_commit`.
