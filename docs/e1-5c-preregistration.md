# E1.5c v1 pre-registration — online contextual Fisher state (four-arm contract)

Committed BEFORE any E1.5c confirmatory run, per
`docs/rung-lifecycle-protocol.md`. Binding design; deviations are
exploratory by definition and must say so. Design record:
`docs/e1_5-instruction.md` §E1.5c (operator-ratified 2026-08-19, PR #73);
implementation delta FOR-923/924/925 merged (PRs #74, #77, #79, #80, #84).
Operator rulings recorded on FOR-907 2026-08-20 — λ_field pilot grid,
byte-budget estimand, and the production-dtype diagnostics channel — are
baked in below. Drafted after digesting Lourie et al. 2026
(arXiv:2608.11859, `docs/references.md`); the sections it reshaped are
§Regime, §Schedule, and the LR-sensitivity diagnostic in §Pilots.

## Thesis

A symbol enters the model as a causal predictive distribution, not a fixed
identity vector: `z_t = √p_φ(·|x_≤t) · Q` with `Q` E1's fixed
semi-orthogonal lift, so `z_t·z_sᵀ` is the Bhattacharyya coefficient
between predictions — angular separation IS distinguishability. The byte is
the observation consumed by `p_φ`, never the state supplied to the
backbone. Four separable value claims, never compressed into one "faster":
statistical efficiency (primary here), optimization efficiency
(co-primary), compute efficiency (reported, separate), architectural
efficiency (the matched-total lane).

## Arms (registered wave)

Every field arm loads the SAME frozen `p_φ` checkpoint (§Pilots) — the
cleanest attribution: identical field weights under all constructions, and
a frozen field structurally cannot become a private code negotiated with
the backbone.

| Arm | Config | Regime | Seeds | Pods |
|---|---|---|---|---|
| T (static Fisher table) | the nine CAPTURED `e1-fisher` runs | — | pairing at {1, 2, 3, 4} | 0 |
| F (Fisher state, `√p·Q`) | `configs/e1-5c-field.yaml` + `field_mode=frozen` | frozen | {1, 2, 3, 4} | 4 |
| H (hidden state, `W·h_φ`) | `configs/e1-5c-hidden.yaml` + `field_mode=frozen` | frozen | {1, 2, 3, 4} | 4 |
| F matched-total | `configs/e1-5c-field-matched-total.yaml` + `field_mode=frozen` | frozen | {1, 2, 3, 4} | 4 |

- **P (`p·Q`, the √-map control) is the DECLARED FOLLOW-UP** (budget-limited
  start T/H/F per the design record); its configs are registered and this
  same file gates it when it runs. H/P matched-total twins likewise.
- **Joint-calibrated regime = registered SECONDARY lane**: same arms and
  seeds with `field_mode=joint`, `field_loss_lambda = λ*` from the pilot —
  λ* applies UNIFORMLY to every joint-regime arm (FOR-907 ruling, ex CR
  PR #77). Runs after the frozen wave, budget permitting; reported
  separately, never pooled with the frozen lane.
- The commit carrying this file registers the three matched-total configs
  in `config.EXPERIMENTS` (they were code-ready, deliberately unregistered
  until now — execution was prereg-time scope).
- Critical contrasts, read in this order: **F−T** (headline), **H−T**
  (contextual at all), **F−H** (probability geometry beyond contextual
  preprocessing); **F−P** waits for P.

## Pilots (declared here; never confirmatory)

1. **Standalone `p_φ` pretrain** — the `field-pretrain` CLI lane (landed in
   this stack): e1-5c-field's estimator shape (2 layers, d=128, 4 heads,
   tied 256-way head), seed 42, 20k steps, same corpus/split machinery as
   the trainer. Produces (a) the frozen checkpoint artifact — DVC-tracked;
   md5 enters the integrity chain; delivered to credential-free pods baked
   into the wave image, so the image digest covers the field weights — and
   (b) the **standalone reference CE** (held-out, last-5 mean), the anchor
   for the λ selection rule and the calibration gate. Declared asymmetry:
   the frozen field saw the train split once before the backbone's own
   pass (the regime is already multi-epoch: 20k steps ≈ 3.45 passes).
2. **λ_field pilot** (operator-ruled grid): joint `e1-5c-field` at
   λ ∈ {0.1, 0.3, 0.5, 1.0}, seed 42, SHORT runs (3000 steps ≈ 49.2M
   bytes). **Selection rule (mechanical):** λ* = the λ whose held-out field
   CE lands closest to the standalone reference, subject to backbone
   nats/byte at step 3000 within +0.005 of the best pilot arm; ties break
   toward the smaller λ.
3. **LR-sensitivity diagnostic** (added from Lourie et al. 2026 — small
   models are hyperparameter-sensitive, and T vs F is an ARCHITECTURE
   comparison): T and F-frozen at lr ∈ {3e-4, 6e-4, 1.2e-3}, seed 42,
   3000 steps (6 short runs). Report each arm's short-run LR response.
   **Rider rule (declared):** if the two arms' best-LR choices differ AND
   the T−F gap changes sign across the grid, every quotation of this
   registration's verdict carries a mandatory hyperparameter-sensitivity
   rider. No mid-registration re-tuning: confirmatory arms run at the
   shared configuration regardless.

## Endpoints

All loss readings are **last-k(=5) eval means**: the mean of
`nats_per_byte` over the five consecutive eval records ending at the
reading's step (eval_interval 500), via `telemetry.read_eval_series` on
DVC captures only. Adopted per the n=9 review's endpoint-noise finding.

- **Scientific primary — learning-curve estimand** (FOR-907 ruling:
  BYTES). Source-byte budgets `B ∈ {49.152, 98.304, 196.608, 294.912}M` =
  steps {3000, 6000, 12000, 18000} at the measured 16,384 bytes/step
  (batch 16 × seq 1024). Per seed: `d_B,s = L_T,s(B) − L_F,s(B)` (positive
  = F better), primary statistic `d̄_s = (1/4)·Σ_B d_B,s`. Paired CI over
  the 4 seeds, single look, two-sided 95% (`t_{0.975,3} = 3.182`).
  **Clause precedence:** superiority-shaped, so E1's precedence carries —
  direction (CI vs 0) is the primary label; practical relevance vs
  ε_L = 0.01 nats/byte is the mandatory co-equal rider.
- **Engineering co-primary — steps/bytes-to-target.** L* = 0.80 nats/byte
  (NOT the e0-byte terminal 0.7738, which right-censored 2/9 E1 random
  runs). Crossing = the FIRST eval step whose last-5-ending-there mean
  ≤ L*. `S_s = (steps_T − steps_F)/steps_T`; bytes/step is constant across
  byte arms so steps- and bytes-to-target coincide (both reported).
  **Censoring rule:** a run that never crosses by step 20000 censors the
  pair — reported, excluded from the CI; decision labels require ≥3
  uncensored pairs, else the co-primary reads INCONCLUSIVE. Decision
  language (ruled): *convergence improvement* = one-sided 95% LCB of
  mean(S) > 0; *practically relevant* = point estimate ≥ 10% (= ε_S);
  *strong support* = LCB > 10%.
- **Guardrail — terminal noninferiority.** At step 20000 (327.68M bytes):
  mean over seeds of `L_F(term) − L_T(term)` must be < δ = 0.005 nats/byte
  (the design record's point-estimate form; the paired CI is co-reported).
  "Reaches useful quality sooner" cannot hide "plateaus worse".
- **Compute/systems readouts — reported, NEVER merged into any primary:**
  estimated training FLOPs (effective-parameter accounting per Porian et
  al. — the field encoder adds FLOPs beyond its parameter share) and
  FLOPs-to-target, measured GPU-seconds and GPU-seconds-to-target, peak
  VRAM, tokens/s, field share of params and FLOPs. Both lanes.
- The matched-total lane answers the fixed-budget question with the same
  estimators, F-matched-total vs T.

## Hierarchical decision rule (ruled)

- **Thesis supported** (ALL three): (1) learning-curve CI entirely above
  0; (2) terminal noninferiority within δ; (3) the field stays calibrated
  (§Calibration gate).
- **Practically valuable** (additionally, any ONE): ≥10% fewer bytes/steps
  to L*; ≥0.01 nats/byte lower at any registered budget (point estimate,
  CI co-reported); matched-total F matches/beats the 85M static baseline
  (terminal point gap < δ).
- **Compute-efficient** (separate claim, never merged): fewer FLOPs- or
  GPU-seconds-to-target.

## Calibration gate and diagnostics

Telemetry (FOR-925, merged): field nats/byte, Brier, ECE (15 bins), mean
entropy, top-1 accuracy in every eval record; `telemetry.py` refuses a
partial calibration group.

**Ratified channel (operator, 2026-08-20, FOR-907):** these diagnostics
are computed from the PRODUCTION-DTYPE encoder forward — ambient bf16 on
pods, CE reduction in FP32 (the post-dedupe channel from PR #80). Declared
ex-ante: they describe `p_φ` as the backbone actually consumes it, not an
FP32 counterfactual. Exact-equality on CPU stays test-pinned.

Gate readings (clause (3) of the decision rule):
- **Frozen arms:** held-out field CE must match the pretrain-time
  standalone reference within ±0.02 nats/byte. This is a plumbing/drift
  check — frozen weights whose measured CE moves beyond dtype/venue delta
  is a DEFECT to investigate, not a finding.
- **Joint arms (secondary lane):** terminal held-out field CE ≤ standalone
  reference + 0.05 nats/byte. Field CE degrading while backbone CE
  improves is the private-code signature; breaching fails clause (3) for
  that lane.

Registration pins (must be green at the launch commit): the behavioral
anti-leakage proofs (prefix-identical/suffix-divergent sequences share
`z_≤t` and diverge after; a forward hook proves targets never alter the
encoder's output), the geometry correspondence (`z_t·z_s` reproduces the
Bhattacharyya coefficient to fp32 tolerance under the semi-orthogonal
lift; θ and Jensen–Shannon move together along a mixture path), and the
context-sensitivity probe versus the static-table zero control. The
context-sensitivity probe is additionally REPORTED from the confirmatory
checkpoints: within-byte state-angle spread, field arms vs T's exact zero
— the clearest demonstration that the state is no longer "byte encoding".

## Schedule (declared) and the WSD branch-decay diagnostic

Confirmatory runs keep E1's cosine-to-20k schedule so the captured
`e1-fisher` runs remain valid paired controls. Consequence, declared per
Lourie et al. 2026 / Hägele et al. 2024: the budget-indexed readings are
**mid-schedule, shared-trajectory values** — both arms sit at the
identical LR trajectory at every budget, so the PAIRED gap is
interpretable, but readings are not budget-tuned frontier losses (the
terminal reading is: cosine is fully decayed at 20000).

**Registered schedule-sensitivity diagnostic (machinery-gated):** one
seed (s1), T-rerun + F-frozen under warmup-stable to 20k with decay
branches from the four budget steps (WSD branch-decay per Hägele et al.).
Compare `d_B^WSD` to `d_B^cosine`: agreement in sign and rough magnitude
validates the cosine curve estimand; disagreement is itself a registered
finding and E2's registration (unconstrained by legacy controls) adopts
WSD outright. Runs once the trainer grows stable-phase + decay-resume
support (its own Linear issue) — no later than the joint-secondary lane,
and never gating the frozen wave.

## Regime and hyperparameter scope (declared)

All confirmatory arms run at the SHARED E1 configuration (lr 6e-4, batch
16 × seq 1024). The registered estimand is the **arm effect at this fixed
configuration** — Lourie et al. 2026's central caution: at ≤100M
parameters, architecture comparisons are confounded by per-arm tuning
response, and an arm can lose at a shared configuration purely by being
harder to tune. Unlike E1.5a's freeze (same architecture), T-vs-F/H is a
genuine architecture comparison, so the exposure is real: it is bounded by
the LR-sensitivity diagnostic (§Pilots) and its declared rider rule, not
assumed away. All claims are scoped to the ~85M / enwik8 / byte-level
regime; nothing is registered about tuned-frontier or larger-scale
behavior. Fixed pretraining data across every arm means pretraining loss
is the sufficient readout (perplexity–capability correspondence,
Mayilvahanan et al. 2025, via Lourie et al. 2026) — no downstream
evaluations are registered or needed at this rung.

## Pairing, exclusion, integrity

- Pairing digest, non-finite-pair-drop exclusion, DVC-captures-only
  analysis inputs: identical to `docs/e1-5a-preregistration.md`.
- Integrity chain gains three links: the frozen `p_φ` checkpoint's md5
  (DVC pointer), the pretrain results JSONL md5, and the wave image digest
  (which covers the baked checkpoint bytes).
- Execution gate: the commit carrying this file registers the
  `state_source` axis in `config.E15_PREREGISTRATIONS` (FOR-914);
  `ENTROPIC_ALLOW_UNGATED=1` for deliberate ungated runs.

## Analysis, review, verdict

Fresh analysis session; blind review packet with arm relabeling and sealed
mapping per `docs/peer-review-protocol.md` (cloud lane). Verdict artifact:
`docs/verdicts/e1-5c-verdict.{md,json}` with `preregistration_commit`.
The verdict must quote the hierarchical rule's three clauses separately —
collapsing them into one label misquotes this registration.

## Costs (measured $1.59/hr A100-SXM4-80GB, ~65 min/full run)

Pilots: pretrain ≈ $0 (local) or ≤$1 (pod); λ grid 4 × ~$0.3 ≈ $1.2; LR
diagnostic 6 × ~$0.3 ≈ $1.8. Frozen wave: 12 pods ≈ $21. Joint secondary:
+12 ≈ $21 when budgeted. WSD diagnostic: ≈ $7 when machinery lands.

## Amendment 1 — matched-total width correction (2026-08-21, post-run, pre-analysis)

The matched-total configs as registered carried `d_model: 756` with
`n_heads: 12`, giving **head_dim 63 — odd**. RoPE rotates coordinate PAIRS,
so `models/blocks.py`'s `rotary_embeddings` refuses an odd head_dim: the
arm raised `ValueError: head_dim must be even for RoPE, got 63` on its
first forward pass and **could never have trained**, on any hardware. This
was not a tuning choice; it was an invalid configuration, and the first
attempt at this lane produced no data at all (four pods, zero metrics,
terminated — reported as a registered deviation on FOR-907).

The lane therefore ran at **`d_model: 744`** (head_dim 62, even), the
nearest valid width below 768 that preserves `n_heads: 12`. Heads were held
fixed deliberately: this lane's declared axis is `d_model`, so changing
`n_heads` instead (756/14 and 756/6 are also even) would have introduced a
SECOND axis into a one-variable comparison.

**Consequence, stated rather than buried.** Total parameters move
**82,940,244 → 80,344,872**, i.e. from 2.6% under `e1-fisher`'s 85,150,464
to **5.6% under**. With `n_heads: 12` there is no width between 744 and 768,
so "matched-total" is approximate either way. **The bias runs AGAINST the
field hypothesis**: the field arm now carries *less* total budget than the
baseline it must match, so a favourable result is conservative and an
unfavourable one is partly attributable to the shortfall. The analysis must
report the actual total, never the registered one.

**Timing and information access.** Recorded after the matched-total runs
completed but BEFORE any analysis session opened, and before any
matched-total outcome had been read by anyone. The width was chosen from
arithmetic constraints alone (divisibility and the 80–100M band) at a
moment when the lane had produced **zero** data — the prior attempt having
failed before its first metric. No endpoint, estimator, decision rule,
seed set or clause precedence is altered by this amendment.

**Not amended, and still open:** whether to prefer an exact-total variant at
a different head count (756 with `n_heads` 14 or 6 preserves 82,940,244
exactly, at the cost of a second axis) remains an operator preference. The
conservative setting stands until ruled otherwise.

**Related measurement, replacing a retracted claim.** The 744 arm ran at
470–480 steps/min against ~513 for the 768-wide F/H arms — a **~6%**
penalty for head_dim 62 vs 64, measured. An earlier note on FOR-907
asserting "3–5× slower, tensor-core artifact" was retracted: the wall-clock
gap it described belonged to the hung runs, which were not slow but dead.
Achieved throughput is ~63 TFLOP/s ≈ 20% MFU on the A100, the model being
occupancy-starved at the registered batch 16 × seq 1024 — a deliberate cost
of pairing with E1's captured controls, and a further reason GPU-seconds
stay "reported, never merged into the primary".
