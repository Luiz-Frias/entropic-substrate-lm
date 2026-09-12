# Predictive-State Geometry Pivot — Implementation Marathon

**Status:** approved forward program; D1 must be a complete tracer bullet, not a skeleton.

**Date:** 2026-08-23.

**Architecture:** `docs/architecture.md`.

**Linear initiative:** [Entropic Substrate — statistics-first geometry for language models](https://linear.app/form-fxn/initiative/entropic-substrate-statistics-first-geometry-for-language-models-9bd302c5925e).

**Canonical Linear project:** [Predictive-State Geometry Pivot](https://linear.app/form-fxn/project/predictive-state-geometry-pivot-f9dccf38344b).

**Team:** Form_fxn.

**Lead:** Luiz Frias.

**Schedule:** verdict-gated; no completion date is inferred.

## Implementation amendment — 2026-09-04

Operator-approved delivery order: implement D1 repairs and all D2–D5 mechanisms
with local tiny tests first; then execute the real scientific flywheel serially
through the eligible rungs. This supersedes any reading below that gates writing
successor code on a predecessor verdict. Remote successor probes and confirmatory
execution remain gated. Test-only gate fixtures cannot authorize a paid run.

- Each commit contains at most **599 added-plus-deleted lines**, including tests,
  documentation, and schema changes. RED tests precede production changes.
- Enwik8 remains the fixed-configuration regime. Each rung starts fresh paired
  seeds with the eligible predecessor construction, not pretrained checkpoints.
- Multiplicity-controlled 95% inference decides; a predeclared marginal 3-sigma
  analysis of the same sealed observations is report-only. Seed sizing targets
  90% projected power per primary contrast from excluded, representative pilots
  and a conservative variance bound. Tiny results cannot supply pilot variance.
- Probe, pilot, and confirmatory spending each require a separate phase quote.
  Source-byte observations, target-encoder exposure, and probe resources remain
  separate measured scopes. Publication remains an explicit operator action.
- D1 distinguishes missing/nonfinite gradients and ownership defects from finite
  measured zero gradients and registered collapse/grounding outcomes. A zero
  gradient does not skip a successful AdamW/EMA transaction. Zero-vector cosine
  is an explicit undefined diagnostic, never an invented numerical value.
- D2 uses one final-block inverse-Fisher residual intervention, fully
  differentiated through the metric and matrix-free solve. The initial damping
  is 1.0, iteration limit 32, relative tolerance 1e-5, absolute tolerance 1e-8.
  A failed solve is a defect; no Euclidean fallback is authorized.
  Recompute the metric at the current state; do not retain metrics or autograd
  graphs across steps. Older cache work items below do not authorize such reuse.
- Predictive score gain is an evaluation-only observable-byte probe, motivated
  by [IGPO v2](https://arxiv.org/html/2510.14967v2), not an RL training objective.
  Compare the same future byte at positions strictly after both scored states,
  with explicit remaining horizons, frozen checkpoint/probe identity, training-
  only probe fitting, held-out scoring, and uniform/shuffled-state controls.
  Diagnostic lags are {2,4,16,64}; fitting defaults are 65,536 training positions,
  200 AdamW updates, batch 256, learning rate 1e-3, weight decay 1e-4, seed 42;
  evaluation uses 8,192 positions. Tiny reductions are declared explicitly.
- Score gain cannot decide D1/D2. Any deciding D3/D4 use must be preregistered.
  Fixed-covariance Gaussian NLL improvement is not decreasing Gaussian entropy.
- D3 retains fixed-variance Gaussian horizons {1,4,16,64} plus the equal-weight
  composite; D4 starts at fixed W=8; D5 ends at bijective byte permutations.
  Alternate-codec and post-D5 topology work named below is deferred and outside
  this delivery contract.
- Reuse the D1 transport, resource and custody boundaries. Scientific rosters,
  exclusions, contrasts, and successor policies remain rung-specific. Preserve
  D1's strict four-arm validator behind its own adapter.

Implementation completion and scientific completion are distinct. Passing local
golden tests does not constitute an excluded remote pilot, a confirmatory run,
fresh analysis, independent review, or a filed scientific successor decision.

### Local implementation handoff — 2026-09-12

[PROGRESS.md](../PROGRESS.md) records the delivered commit checkpoint and validation
scope. Local D1 repairs, D2–D5 scientific mechanisms, predictive score probes,
worker transport, and shared D1 fresh/resume phase ordering are committed. The
host campaign/operator/capture integration remains an unfinished tail: its public
dispatch still needs `d_ladder_operator.py` and `d_ladder_application.py`, followed
by a production-shaped host-command golden lineage and integrated validation.
Do not treat the worker command or test-double-backed local flywheel as proof
that this host path is ready. Existing E-series verdicts remain historical; no
new scientific successor gate or remote spending authorization is implied.

## Program thesis

> The causal hidden state remains latent. Codec-independent future laws supervise that state
> and induce a Fisher semimetric on its dynamics; their outputs never become the state.
> Predictive horizon and temporal coarse-graining are earned from future consequences, and
> reversible external symbol labels do not determine latent geometry.

## Tracer-bullet commitment

D1 implements the first vertical slice **in full**:

```text
contract and RED tests
  → production implementation
  → guard-tooth negative proofs
  → production-shaped smoke
  → declared excluded pilot
  → committed preregistration
  → confirmatory run
  → authoritative capture
  → fresh-session analysis
  → arm-blind independent review
  → filed Markdown + JSON verdict
  → carry / narrow / retire gate
```

“Tracer bullet” therefore does not mean a toy model that stops after a forward pass. It means
the narrowest end-to-end scientific path that crosses every real boundary once: model,
configuration, checkpoint/resume, telemetry, DVC, analysis, review, verdict, and successor
authorization. Later rungs widen one axis at a time over that proven path.

The flywheel is one program but not one blended context. Implementation, run operation, fresh
analysis, and blind review use separate handoffs. An implementation session never sees
confirmatory metrics before the analysis packet is sealed, and an analysis session does not
change runtime behavior.

## Entry evidence: frozen-head audit

The newly filed `docs/e1-5c-head-audit-findings.md` is D1 design input with an explicit
**EXPLORATORY** label. It measures a full-rank centered head and production clamp-associated
reconstruction loss; it does not become a D1 endpoint or retroactively change E1.5c.

D1 uses the audit in three bounded ways:

1. retire “128→256 necessarily loses rank” as an unqualified explanation;
2. preserve clamp/conditioning and matched-interface controls in the D1 diagnostic packet; and
3. keep the new main state latent so neither a 256-way softmax nor its clamp is the state path.

The D1 preregistration may quote the audit as design provenance. It may not reuse its held-out
examples as confirmatory D1 evidence unless that reuse is declared before D1 execution.

## Linear hierarchy

The canonical project has five project milestones. Each has exactly **15 issues**:

```text
project milestone / scientific epic
  ├── parent workstream A
  │     └── four nested task issues
  ├── parent workstream B
  │     └── four nested task issues
  └── parent workstream C
        └── four nested task issues
```

The 3 × (1 parent + 4 children) shape gives fifteen issues per milestone, within the requested
15–20 range. Parent issues describe outcomes and boundaries; child issues are independently
verifiable units. Confirmatory work stays closed until the predecessor verdict authorizes it.

## Shared lifecycle contract

Every confirmatory rung follows `docs/rung-lifecycle-protocol.md` and
`docs/peer-review-protocol.md`:

1. **Design** — name one axis, controls, estimators, practical margins, exclusions, and kill
   conditions.
2. **TDD implementation** — failing test first for each new public contract; prove guard teeth
   by deliberately breaking the guarded behavior.
3. **Pilot** — excluded data only; characterize numerical ranges and choose constants the
   preregistration allows pilots to set.
4. **Preregistration** — commit the complete design before any confirmatory metric exists.
5. **Run** — execute paired arms sequentially under the registered data order and runtime
   envelope.
6. **Capture** — content-address results/checkpoints/manifests with DVC; validate through the
   production reader.
7. **Fresh analysis** — read DVC-captured evidence only through admissible readers; never use
   the live tracking server.
8. **Blind review** — seal arm mappings, give the reviewer a self-contained recomputation
   packet, and file the independent review before unblinding.
9. **File and gate** — write narrative and machine verdicts, adjudicate review findings, then
   authorize, narrow, or retire the successor.

Lifecycle problem states are first-class: `NumericalDefect`, `EvidenceGap`, `ReviewReopened`,
`OperatorDecisionNeeded`, `ValidNegative`, and `Retired`. A failed arm, collapsed state, or
empty feasible set is an outcome—not missing paperwork to be filled with an inference.

## Shared evidence contract

Evidence classes are never mixed without a row-level label:

- `MEASURED` — a counter or provider value read from the actual execution;
- `DERIVED_FROM_MEASURED` — deterministic arithmetic over measured values;
- `MODEL_ESTIMATE` — analytical/profiler work estimate;
- `RATE_ESTIMATE` — measured elapsed scope multiplied by an immutable quoted rate; and
- `PROVIDER_BILLED` — provider-reconciled duration or amount.

JSONL and DVC captures are authoritative. MLflow is a best-effort operational mirror and is
never a confirmatory analysis input. Missing, partial, nonfinite, nonmonotonic, mismatched, or
schema-incompatible evidence fails loudly.

## Cross-project measurement dependency

The existing `entropic-substrate-lm` milestone **Auditable compute-to-quality telemetry** owns
the implementation of resource provenance and counters:

- `FOR-938` — versioned measurement envelope;
- `FOR-939` — provider/SKU/region/price provenance;
- `FOR-940` — resolved train/eval FLOP accounting;
- `FOR-941` — accelerator/runtime/container provenance;
- `FOR-942` — trainer-wall, token, and source-byte counters;
- `FOR-943` — GPU-active train/eval time;
- `FOR-944` — optional energy integration;
- `FOR-945` — provider billing reconciliation;
- `FOR-946` — rate-estimated cost; and
- `FOR-947` — DVC capture and fail-loud cost-to-quality read side.

The D1 measurement workstream validates and consumes those outputs; it does not reimplement
them. Confirmatory efficiency claims are blocked on the applicable upstream issue. If a field
is unavailable, the corresponding claim and chart are omitted rather than silently estimated
under a measured heading.

## Program dependency graph

```text
E1.5c frozen verdict + exploratory frozen-head audit
                     │
                     ▼
       D1 — stabilized latent predictive state
                     │
          ┌──────────┴──────────┐
          │                     │
  no eligible state       ≥1 eligible state
          │                     │
     retire D2–D4               ▼
                  D2 — decoder-induced geometry
                     │
          ┌──────────┼─────────────────────┐
          │          │                     │
   no predictive  predictive only     predictive + Fisher
      signal         passes                passes
          │          │                     │
     retire D3/D4    └──────────┬──────────┘
                               ▼
                 D3 — future-law/horizon sweep
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
       no longer-horizon value       complementary value
                 │                           │
             skip D4                        ▼
                                  D4 — coarse-graining
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                      D5 — codec invariance
                               │
                               ▼
                    final synthesis and close
```

---

# D1 — Stabilized Latent Predictive State

**Linear milestone:** `D1 - Stabilized latent predictive state`.

**Outcome:** train and evaluate a codec-independent latent causal state without routing the
backbone state through a 256-way probability simplex; isolate stop-gradient and anti-collapse
mechanisms in a registered 2 × 2 design; complete the entire scientific flywheel.

**Issue count:** 3 parents + 12 nested tasks = 15.

## Scientific design

Fixed tracer:

- byte observations remain the external codec and nats/byte remains the language readout;
- the online encoder emits latent `h_t` whose width is independent of vocabulary size;
- byte observation and byte readout parameters are untied boundary adapters;
- the online and target encoders share one latent width and use RMS-style final normalization;
- from a contiguous `T + 1` source span, the target encoder constructs the normalized latent
  displacement `xi[t+1] - xi[t]`;
- a diagonal Gaussian future-law head predicts that displacement from `h_t`, with fixed
  `sigma^2 = 1`, per-dimension NLL, objective weight `1.0`, and the zero-displacement law as
  the trivial comparator;
- the future target never enters the causal online state path; and
- D1 does not consume a Fisher metric yet.

Registered stabilization matrix:

| Arm | Stop-gradient target | Anti-collapse penalty | Role |
|---|---:|---:|---|
| `D1-NONE` | off | off | positive collapse/failure control |
| `D1-SG` | on | off | target-encoder stabilization |
| `D1-AC` | off | on | moment anti-collapse stabilization |
| `D1-BOTH` | on | on | complementarity/interaction |

Primary scientific outputs must keep three claims separate:

1. **noncollapse** — variance/covariance/effective-rank criteria;
2. **future grounding** — held-out future-law NLL/calibration and observable probe quality; and
3. **language utility** — nats/byte learning and terminal behavior.

### Parent `FOR-948` — D1.1 Consume and validate the auditable measurement contract

**Purpose:** prove D1 can consume auditable resource evidence before making efficiency claims.
These tasks are integration acceptance over `FOR-938`–`FOR-947`, not duplicate telemetry
implementations.

1. **`FOR-951` — Validate D-run start provenance from the shared envelope.** Consume and validate the versioned
   provider, accelerator, runtime, container, price, pod-created, and trainer-start envelope.
   **Blocked by:** `FOR-938`, `FOR-939`, `FOR-941`.
2. **`FOR-952` — Validate cumulative resource counters on D-series evals.** Verify monotonic trainer wall,
   source bytes, tokens, GPU-active train/eval seconds, FLOPs, optional energy, and rate-cost
   counters at every eval. **Blocked by:** `FOR-940`, `FOR-942`, `FOR-943`, `FOR-944`,
   `FOR-946`.
3. **`FOR-953` — Validate D-run provider billing reconciliation.** Prove termination time, billed
   duration, billed amount, currency, and record identity survive capture without being
   commingled with rate estimates. **Blocked by:** `FOR-945`.
4. **`FOR-954` — Wire D-series cost-to-quality views to the admissible reader.** Build
   wall/bytes/tokens/FLOPs/cost-to-quality views only from evidence classes accepted by
   `FOR-947`. **Blocked by:** `FOR-947`.

### Parent `FOR-949` — D1.2 Latent predictive-state core

**Purpose:** build the production-shaped online/target/future-law path without making the
language classifier or target decoder the state.

5. **`FOR-955` — Introduce observation-codec and latent-state contracts.** Pin codec identity,
   source-span alignment, fixed latent width, and the rule that alphabet cardinality belongs at
   the boundary.
6. **`FOR-956` — Implement the latent online encoder trunk.** Emit causal `LatentTrajectory`,
   preserve existing language readout, and prove the main state path never receives a
   probability vector.
7. **`FOR-957` — Add future-target windows and decoder interfaces.** Build deterministic
   future windows, target encoding, diagonal Gaussian law, exact causal offsets, and no-leakage
   fixtures.
8. **`FOR-959` — Emit collapse and grounding telemetry.** Record variance, covariance,
   numerical/effective rank, pairwise uniformity, future-law NLL/calibration, observable probe
   quality, and gradient reachability.

### Parent `FOR-950` — D1.3 Stabilization strategies and complete tracer-bullet flywheel

**Purpose:** implement both requested mechanisms as independent strategies and carry D1 through
the filed gate.

9. **`FOR-962` — Implement stop-gradient target encoder.** Detach targets, keep teacher
   parameters outside the optimizer, apply EMA only after successful optimizer steps, and make
   checkpoint/resume restore the exact teacher cursor.
10. **`FOR-963` — Implement anti-collapse regularization.** Add FP32 variance-floor and
    covariance penalties with constant-state, rank-one, and full-rank guard fixtures.
11. **`FOR-964` — Add four-arm configs, tiny gates, and production smoke.** Validate complete
    construction identities; prove future target blindness, strategy identity, metric-off
    identity, checkpoint resume, nonzero gradient influence, and guard teeth.
12. **`FOR-965` — Run the complete prereg-to-gate D1 flywheel.** Run the excluded pilot; commit the D1
    preregistration; launch paired confirmatory arms; capture with DVC; hand off to a fresh
    analysis session; assemble sealed arm aliases; obtain independent blind review; file
    `d1-stabilized-latent-verdict.{md,json}`; adjudicate and gate D2.

`FOR-965` is hard-blocked by the D1 scientific implementation and core reproducibility
provenance acceptance (`FOR-951`). `FOR-952`–`FOR-954` form the efficiency annex: they stay in
the D1 milestone and gate only the corresponding wall/byte/token/FLOP/GPU/energy/cost claims.
An unavailable optional modality is a typed gap, not a blocker of a fixed-step scientific
verdict unless the committed preregistration explicitly promotes it to a deciding endpoint.

## D1 pilot and preregistration decisions

The excluded pilot may set only the quantities explicitly delegated to it:

- one shared online/target latent width within the documented envelope;
- EMA decay schedule;
- anti-collapse variance floor and covariance weight;
- collapse/effective-rank defect thresholds;
- smoke-scale memory/runtime envelope.

The pilot may not change the D1 horizon (`1` source byte), target displacement construction,
fixed Gaussian variance (`sigma^2 = 1`), per-dimension reduction, future-loss weight (`1.0`),
untied boundary adapters, or online-only placement of the collapse penalty.

The committed preregistration must freeze:

- arm identities, seeds, pairing/data-order contract, and exclusion rule;
- primary metric and smoothing/window rule;
- noncollapse, future-grounding, and language-utility estimators;
- practical margins and multiplicity handling;
- target/horizon/strategy constants;
- source-byte and step budgets;
- required telemetry fields and evidence classes;
- blind alias map procedure;
- trigger for D2 eligibility; and
- clause precedence and exact verdict vocabulary.

## D1 gate

D2 opens only if at least one stabilization arm:

- passes the registered noncollapse clause;
- predicts the registered future target better than the trivial control;
- passes causal/no-leakage and construction-integrity checks;
- has complete authoritative evidence; and
- is not rejected by the filed blind review.

If `SG` and `AC` disagree, both results are retained. If both independently pass and the
interaction is ambiguous, D2 remains stratified instead of selecting by narrative preference.
If every arm collapses or lacks future signal, D2–D4 are retired; D5 may still test a narrower
codec boundary only if a separate preregistration justifies it.

---

# D2 — Decoder-Induced Geometry

**Linear milestone:** `D2 - Decoder-induced geometry`.

**Outcome:** determine whether the future law’s Fisher pullback improves latent dynamics beyond
the same predictive system under Euclidean dynamics.

**Issue count:** 3 parents + 12 nested tasks = 15.

## Scientific design

Minimum arm family, instantiated inside every D1-eligible stabilization block:

```text
L — latent state + ordinary language objective + Euclidean dynamics
S — same latent state + future-law supervision + Euclidean dynamics
G — identical to S, but decoder-Fisher pullback is consumed at one residual site
```

`S−L` identifies predictive supervision. `G−S` is the primary geometry contrast. `G−L` is a
composite and never receives a mechanistic label by itself.

### Parent `FOR-969` — D2.1 Metric-induction contracts

1. **`FOR-991` — Define metric-induction and geometry-consumer ports.** Separate future-law
   estimation, metric action, caching, and consumer ownership.
2. **`FOR-989` — Implement decoder-Fisher pullback estimator.** Build a dense tiny-model oracle
   and matrix-free JVP/VJP production action with registered damping.
3. **`FOR-992` — Add metric cache and numerical-health telemetry.** Record rank/spectrum
   summaries, damping share, null directions, cache identity, and finite gradients.
4. **`FOR-994` — Prove metric read/write contracts.** Require dense/matrix-free agreement,
   shape/device/dtype integrity, checkpoint identity, and failure on a deliberately stale cache.

### Parent `FOR-968` — D2.2 Geometry-aware consumers

5. **`FOR-995` — Add explicit Euclidean control consumer.** Make the control a named strategy
   behind the same call site.
6. **`FOR-997` — Adapt existing sphere/kappa machinery as optional comparators.** Preserve
   their relational interpretation without treating constant curvature as Fisher evidence.
7. **`FOR-998` — Implement one pullback-aware residual consumer.** Use the local metric at one
   registered site and prove nonzero forward/gradient influence.
8. **`FOR-1000` — Add matched resource accounting.** Capture compute, memory, wall, byte,
   token, FLOP, energy, and cost evidence by class for every consumer.

### Parent `FOR-971` — D2.3 Complete confirmatory operations

9. **`FOR-1002` — Draft and commit D2 preregistration.** Freeze `L/S/G`, eligible stabilization
   blocks, metric estimator, damping, call site, estimators, margins, interactions, and gates.
10. **`FOR-1003` — Launch D2 and capture authoritative evidence.** Execute sequential paired
    runs, validate construction/metric-consumed attestations, DVC-capture, and reconcile
    resource provenance.
11. **`FOR-1009` — Produce fresh D2 analysis and verdict packet.** Recompute `S−L`, `G−S`,
    `G−L`, metric health, and cost-to-quality from captures only.
12. **`FOR-1008` — Blind-review and adjudicate D2.** Review strategy blocks before any pooled
    claim; file `d2-decoder-geometry-verdict.{md,json}` and authorize the D3 posture.

## D2 gate

- If `S−L` passes and `G−S` passes without a contradictory eligible-strategy interaction,
  carry predictive supervision and Fisher consumption.
- If `S−L` passes but `G−S` fails, carry predictive state and retire Fisher consumption; D3
  proceeds with Euclidean dynamics.
- If both fail, retire the D3/D4 paid scale-up.
- If kappa/sphere comparators win while pullback fails, record that as relational-kernel evidence,
  not decoder-Fisher evidence.

---

# D3 — Future-Law Family and Horizon Sweep

**Linear milestone:** `D3 - Future-law family and horizon sweep`.

**Outcome:** determine whether longer or multihorizon future targets add complementary
predictive information without introducing words, sentences, or tokenizer ontology.

**Issue count:** 3 parents + 12 nested tasks = 15.

### Parent `FOR-972` — D3.1 Future-law target-family framework

1. **`FOR-1010` — Add horizon-family schema and validated builder.** Make target family,
   source span, weighting, and output width complete construction identities.
2. **`FOR-1011` — Implement single-horizon targets `{1,4,16,64}`.** Use source-byte positions
   and exact causal alignment; do not enumerate `256^H` outcomes.
3. **`FOR-1012` — Implement one multihorizon composite law.** Add it only after the four fixed
   controls share a stable contract.
4. **`FOR-1013` — Add target-family tests and telemetry validation.** Pin equal state width,
   no leakage, calibration, target grounding, collapse statistics, and horizon identity.

### Parent `FOR-974` — D3.2 Horizon comparison lanes

5. **`FOR-1015` — Create matched-backbone configs.** Vary future-law family only while keeping
   the settled D2 system unchanged.
6. **`FOR-1017` — Create matched-total-parameter configs.** Separate a better target from a
   larger auxiliary system.
7. **`FOR-1018` — Extend horizon cost-to-quality analysis.** Compare wall, bytes, tokens,
   FLOPs, cost, and target-encoder compute share by evidence class.
8. **`FOR-1020` — Add horizon collapse/calibration/spectrum diagnostics.** Emit incremental
   effective rank, uncertainty, redundancy, and marginal predictive information.

### Parent `FOR-976` — D3.3 Complete confirmatory operations

9. **`FOR-1028` — Draft and commit D3 preregistration.** Freeze horizons, primary adjacent and
   composite contrasts, matching lanes, multiplicity, margin, and carry rule.
10. **`FOR-1026` — Launch D3 and verify captures.** Prove exact source-byte exposure,
    data-order pairing, target alignment, construction identity, and resource provenance.
11. **`FOR-1029` — Produce fresh D3 analysis, verdicts, and charts.** Report learning curves,
    fixed-quality crossings, predictive rank, redundancy, and resource efficiency from DVC.
12. **`FOR-1031` — Blind-review and adjudicate D3.** Alias horizons neutrally, prohibit
    linguistic scale names in the blind packet, file `d3-future-horizon-verdict.{md,json}`.

## D3 gate

D4 learned scale is earned only if a longer or composite horizon adds registered predictive or
language value without breaking the compute envelope. A null result retires learned-horizon
complexity; D5 can still test codec equivariance on the settled short-horizon state.

---

# D4 — Multiscale Coarse-Graining

**Linear milestone:** `D4 - Multiscale coarse-graining`.

**Outcome:** test fixed and learned temporal aggregation over latent predictive states without
assuming character, word, sentence, or historical compression-ratio ontology.

**Issue count:** 3 parents + 12 nested tasks = 15.

D4 is serial:

- **D4a:** fixed-window macro-state control;
- **D4b:** learned predictive boundaries, launched only if the filed D4 trigger fires.

### Parent `FOR-977` — D4.1 Fixed-window macro-state control

1. **`FOR-1032` — Implement fixed-window latent macro-state adapter.** Aggregate latent states,
   not byte IDs, behind the common coarse-graining contract.
2. **`FOR-1033` — Implement dechunk/upsample readout.** Pin exact alignment, shape, inverse
   mapping, causal availability, and language-eval semantics.
3. **`FOR-1034` — Normalize macro-step accounting.** Keep source bytes, tokens, wall,
   GPU-active time, and compute separate from macro-step count.
4. **`FOR-1035` — Add fixed-control configs and analysis helpers.** Produce identity and
   fixed-window baselines suitable for preregistration and capture.

### Parent `FOR-979` — D4.2 Learned coarse-graining

5. **`FOR-1036` — Implement predictive boundary router and smoothing.** Use pure Torch for the
   first tracer and preserve causal boundary decisions.
6. **`FOR-1039` — Implement ratio and anti-degeneracy losses.** Prevent always/never-boundary,
   fixed-duration, whitespace, and punctuation collapse without targeting a linguistic ratio.
7. **`FOR-1047` — Add production tracer and optional container hooks.** Keep H-Net/Mamba as a
   conditional acceleration lane after the pure-Torch path earns scale.
8. **`FOR-1046` — Add boundary inspection and failure telemetry.** Emit boundary entropy,
   duration, compression, gradient health, predictive-state change, and heuristic-control
   agreement.

### Parent `FOR-980` — D4.3 Complete confirmatory operations

9. **`FOR-1049` — Draft and commit serial D4 preregistration.** Freeze D4a, the exact D4b
   trigger, controls, degeneracy thresholds, resource margins, and no-retuning clause.
10. **`FOR-1051` — Run D4a and conditionally D4b; capture separately.** Never pool pilot,
    fixed-control, and learned-boundary evidence.
11. **`FOR-1053` — Produce fresh D4 analysis, verdicts, and boundary visuals.** Report quality,
    sufficiency, compression, boundary behavior, and resource cost per subrung.
12. **`FOR-1055` — Blind-review and adjudicate D4.** Give boundary excerpts neutral IDs, file
    separate D4a/D4b verdicts, and name the legitimate D5 carry-forward state.

## D4 gate

Retire learned coarse-graining if it collapses, merely reproduces a heuristic control, fails
the fixed-window comparator, or violates the resource envelope. If learned horizon passes but
boundaries fail, carry the horizon without chunking. Recursion/two-stage chunking is not opened
by default.

---

# D5 — Codec Quarantine and Interface Generalization

**Linear milestone:** `D5 - Codec quarantine and interface generalization`.

**Outcome:** test whether reversible observation relabeling changes the settled predictive
state, geometry, scale policy, function, or quality; close the D1–D5 synthesis.

**Issue count:** 3 parents + 12 nested tasks = 15.

The first confirmatory codec test is a bijective permutation of all 256 external byte labels.
That preserves the exact English byte process and source-byte accounting. It does not establish
equivalence across bytes, bits, Unicode code points, BPE tokens, speech, or images.

### Parent `FOR-983` — D5.1 Observation and readout codec ports

1. **`FOR-1056` — Formalize observation/readout codec ports.** Confine label cardinality and
   inverse decoding to boundary adapters.
2. **`FOR-1057` — Harden identity bytes and add deterministic byte permutation.** Prove exact
   round trip, matched data order, identical source bytes, and checkpoint codec identity.
3. **`FOR-1058` — Implement one alternate codec adapter.** Open only after the reversible
   permutation control passes implementation and pilot integrity.
4. **`FOR-1059` — Add cross-codec contract and resume tests.** Refuse mismatched codec digests;
   pin target/horizon/coarse-graining alignment across recodings.

### Parent `FOR-984` — D5.2 Equivalence diagnostics

5. **`FOR-1060` — Add codec-permutation invariance surfaces.** Compare paired quality and
   convergence under identity and permuted labels.
6. **`FOR-1061` — Add functional-equivalence diagnostics.** Compare inverse-decoded next-byte
   laws on identical contexts with KL/JS or registered equivalents.
7. **`FOR-1063` — Add relational representation diagnostics.** Compare Procrustes-aligned
   states, Gram/predictive-distance matrices, pullback spectra, and effective rank without
   demanding raw coordinate equality.
8. **`FOR-1064` — Build cross-codec cost-to-quality scoreboards.** Preserve evidence classes
   for wall, bytes, tokens, FLOPs, energy, estimated cost, and provider billing.

### Parent `FOR-985` — D5.3 Final synthesis and adjudication

9. **`FOR-1066` — Draft and commit D5 preregistration.** Freeze codec/permutation seeds,
   equivalence/noninferiority margins, multiplicity, alignment, and failure handling.
10. **`FOR-1068` — Launch D5, capture, and verify provenance.** Pair model seeds and source
    order; capture permutation manifests, latent snapshots, and inverse-decoded outputs.
11. **`FOR-1070` — Produce D1–D5 synthesis.** State separately what was supported, retired,
    inconclusive, or not measured; never let a later rung overwrite an earlier negative.
12. **`FOR-1072` — Run final blind review and archive verdict set.** Blind codec and
    permutation identities, adjudicate, file `d5-codec-equivariance-verdict.{md,json}`, and
    close the pivot program.

## D5 gate and permitted language

The phrase **codec-equivariant predictive substrate** is allowed only if all registered
performance, functional, relational-geometry, and scale-policy clauses pass. Any failure names
the leaking boundary and blocks broader codec claims.

“Predictive manifold” remains disallowed unless a separate, preregistered topology/regularity
capstone establishes the required rank and stability. Report-only TDA may follow D5, but is not
created as a sixth milestone until D5 evidence earns it.

## Cross-rung handoff artifacts

Every parent workstream produces a handoff that another context can verify:

### Implementation handoff

- exact branch/commit and dirty-state declaration;
- changed contracts and complete construction identity;
- failing-test-first evidence and guard-tooth proof;
- focused and full gate outputs;
- pilot-only knobs still open; and
- known defects or measurement gaps.

### Run-operator handoff

- preregistration commit;
- immutable run matrix and seed/data-order map;
- image/checkpoint/config digests;
- provider/SKU/price envelope;
- launch and capture commands;
- sequential-run requirement; and
- explicit abort conditions.

### Fresh-analysis handoff

- only admissible DVC capture paths and input hashes;
- telemetry/checkpoint reader contracts;
- exact estimators, constants, exclusions, and clause precedence;
- sealed arm aliases;
- required machine and narrative verdict schemas; and
- a prohibition on live MLflow and implementation changes.

### Blind-review handoff

- self-contained aliased packet;
- preregistration and analysis implementation sufficient to recompute;
- integrity manifest;
- exact reviewer prompt;
- no architecture labels capable of revealing arms; and
- adjudication procedure for MAJOR/MINOR findings.

## Program review checklist

Before closing any milestone:

- [ ] the implemented axis matches the registered axis;
- [ ] both sides of every quantitative comparison are measured under matched conditions;
- [ ] all evidence rows carry their evidence class;
- [ ] pilots are excluded and content-addressed separately;
- [ ] the preregistration predates every confirmatory metric;
- [ ] construction and data-order identities match across paired arms;
- [ ] authoritative captures validate through the production reader;
- [ ] analysis came from a fresh context and changed no runtime code;
- [ ] blind review used sealed aliases and a recomputable packet;
- [ ] review findings were adjudicated before unblinding or successor authorization;
- [ ] narrative and JSON verdicts agree exactly;
- [ ] the verdict names the precise carried artifact or retires the branch; and
- [ ] Linear state, dependencies, and links match the filed repo artifacts.

## Linear hygiene ruling still open

A concurrent worker also created a second project named
`Predictive Manifold Substrate Pivot`. It overlaps this program, has a different milestone
shape, and should not receive further work. The canonical recommendation is the project linked
at the top of this document because:

- “predictive-state geometry” does not presuppose a manifold result;
- it has the requested five D1–D5 milestones;
- it has exactly fifteen issues per milestone; and
- it already owns the project-level plan resource.

Archiving the duplicate is intentionally left for an explicit operator ruling. No issue or
document should be copied between the two projects in the meantime.
