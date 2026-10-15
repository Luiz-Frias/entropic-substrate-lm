# Architecture — Codec-Separated Hierarchical Latent Prediction

**Status:** approved forward north star; sentence-first generative design is not yet implemented.

**North-star update:** 2026-09-15. The retained D-series design dates to 2026-08-23.

**Prior Linear program:** `Predictive-State Geometry Pivot`, under the Entropic Substrate
initiative. This documentation change does not update external project status.

**Historical boundary:** E-series evidence and existing D-series construction and registration
identities remain bound to their original experiments. Nothing below upgrades their readiness
or reinterprets a filed result.

## North star

Build a codec-separated hierarchical latent predictor: a boundary encoder constructs
continuous sentence/chunk representations, the main spine predicts the next representation
in that sequence, and a local decoder generates text from predicted latents. The main
computational clock is the chunk sequence, not necessarily one step per byte or surface token.
Start with sentence boundaries; paragraph and document-context scales are controlled research
stages, not assumed semantic improvements. Latent slots/tensors may preserve detail beyond a
single vector; coarsening does not imply lossless compression.

Target roughly 500M total deployed parameters, with an upper design envelope of 1B. Count
encoder, spine, and decoder together; report training-only teachers separately. Fisher-based
dynamics remain a candidate mechanism whose signed predictive value and off-target effects
must be measured. The approved implementation contract is in
[the hierarchical predictor itinerary](hierarchical-latent-predictor.plan.md).

Sentence-vector normalization and spine-carrier normalization are independent 2×2 axes.
The magnitude-preserving control must retain its carrier without a final UnitNorm or RMS
overwrite. No probability-floor/square-root state substitution is permitted. The effective
directional/log-magnitude law induces an update operator, never an alphabet-sized carrier.
H1 tests preservation with isotropic dynamics before H2 supervision and H3 Fisher consumption.

## Horizon

First complete one causal sentence-level generation tracer with checkpoint/recovery and the
full reviewed scientific flywheel. Then retain or retire mechanisms through sealed paired
comparisons before increasing temporal resolution, changing backbone/codec, or scaling toward
the deployment envelope. Final success requires reproducible generation, appropriate held-out
quality and resource evidence, and filed support for the retained mechanisms—not merely a
latent-prediction loss or a working byte-level model. The itinerary is not a preregistration
or a remote-spending authorization.

## Retained D-series design boundary

All remaining sections preserve the earlier byte-clock D-series architecture for provenance
and component reuse. Their D1–D5 gates apply only to those constructions. They do not describe
the new sentence-level generator or authorize its execution.

### D-series north star (2026-08-23)

> The hidden state stays latent. Future laws determine which latent displacements matter, and
> geometry is consumed as an operator on latent dynamics—not substituted for latent state.

The observation entering the system may still be a byte. The representation carried by the
semantic core must not be a byte ID, a fixed table row, a 256-coordinate probability vector,
or a lift of that vector. Alphabet cardinality is allowed at observation and language-readout
adapters because the model must communicate with the measured dataset. It is not allowed to
define the width or ontology of the predictive-state core.

The internal implementation term is **predictive semimetric**. “Predictive manifold” remains
a research hypothesis until measured rank, regularity, stability, and codec-equivariance
support the stronger word.

### D-series completion contract

The pivot is done enough when D1–D5 have each completed the full scientific lifecycle:

1. a tested contract with load-bearing negative controls;
2. a declared, excluded pilot;
3. a committed preregistration before confirmatory execution;
4. paired confirmatory runs under the registered regime;
5. authoritative DVC capture and validation through the admissible reader;
6. fresh-session analysis outside the implementation context;
7. an independently recomputable, arm-blind review where technically possible;
8. filed narrative and machine-readable verdicts; and
9. a gate that carries, narrows, or retires the mechanism.

D1 must establish a noncollapsed latent predictive state and test stop-gradient and
anti-collapse stabilization independently. D2 must separate the value of predictive
supervision from the value of consuming decoder-induced Fisher geometry. D3 must earn or
retire longer predictive horizons. D4 must earn or retire learned temporal coarse-graining.
D5 must test whether the settled state survives reversible observation relabeling. Negative
results are terminally useful outcomes, not invitations to move the threshold.

## Why the pivot is evidence-led

The redesign is not a blank-slate reaction to one run. It preserves what the existing ladder
actually measured and rejects the inferences those measurements cannot support.

| Evidence | Class | Measured result | Architectural consequence |
|---|---|---|---|
| E1 extension | **MEASURED** | Fisher initialization beat random initialization at the registered endpoint by `0.00459533333333333` nats/byte over nine paired seeds, while the registered practical-relevance rider remained below `epsilon = 0.01`. It reached the E0-byte terminal target earlier in 9/9 pairs; the reported median step advantage was about 13.5%, with two random runs right-censored. | Corpus-induced geometry is a useful warm-start prior. E1 does not establish a static symbol table as the final representation, nor an asymptotic advantage. |
| E1.5-D | **MEASURED** | Fisher-initialized tables retained their own pairwise Gram geometry (`0.8401 ± 0.0026` Pearson) while random tables did not (`0.0170 ± 0.0009`). | Relational initialization leaves a persistent geometric trace, but persistence alone does not show that freezing or using the table as state is sufficient. |
| E1.5a | **MEASURED** | Freeze-both last-five gap was `0.010604000000000058` with registered 97.5% CI `[0.0013623925061083918, 0.019845607493891724]`; input-only was `0.0026743500000000753` with CI `[-0.007278657794102857, 0.012627357794103008]`. Both sufficiency clauses were inconclusive under the registered threshold; the freeze-both direction rider detected a cost. | Static-table sufficiency was not established. Input and output roles cannot be collapsed into one causal claim. |
| E1.5c frozen | **MEASURED** | The registered learning estimator gave `F−T = -0.11300426250000001` nats/byte with 95% CI `[-0.11800596590322686, -0.10800255909677316]`; the terminal loss gap was `L_F−L_T = 0.05196629999999999`. The decomposed learning contrasts gave `P−H = -0.05077926250000001` and `F−P = +0.01384328750000001`. | The square-root map repaired part of the probability-state loss, but the frozen `sqrt(p_phi)` state did not replace the latent hidden state successfully. A categorical predictive law can supervise or induce geometry without becoming the state. |
| Frozen-head audit | **EXPLORATORY / MEASURED WITHIN ITS OWN CAPTURE** | The current `field-head-audit` work measures the centered head’s rank, conditioning, production clamp loss, and realized hidden spectrum. It is not a registered D verdict. | It can refine D1 controls and kill an incorrect compression story; it cannot retroactively change E1.5c or decide the pivot by itself. |

The architectural inference is therefore explicit and separate from the measured rows:

> Preserve the predictive law as a source of supervision and local geometry. Stop forcing the
> backbone state to be the predictive law’s alphabet-sized output.

## System boundary

The complete forward dataflow is:

```text
external observations
        │
        ▼
ObservationCodecPort ──────────────── codec identity + source-span accounting
        │
        ▼
EncodedObservations
        │
        ▼
CausalStateEncoder ────────────────── h_t : LatentState
        │                                  │
        │                                  ├──────────────► LanguageReadoutPort
        │                                  │                    │
        │                                  │                    ▼
        │                                  │             observable logits / nats-byte
        │                                  │
        │                                  └──────────────► FutureLawDecoder
        │                                                       │
future observation window ─► TargetEncoder ─► FutureTarget      │
                                  │                              ▼
                                  ├─ TargetUpdatePolicy ─► q(Y_future | h_t)
                                  └─ CollapseRegularizer         │
                                                                 ▼
                                                  DecoderFisherPullbackMetric
                                                                 │
                                ┌────────────────────────────────┘
                                ▼
                      GeometryConsumerStrategy
                                │
                                ▼
                    metric-aware latent dynamics
                                │
                                ▼
                      HorizonPolicy / CoarseGrainer
                                │
                                ▼
                    multiscale latent trajectory
                                │
                   ┌────────────┴────────────┐
                   ▼                         ▼
            language readout          report-only observers
                                      (geometry, TDA, systems)
```

The future window is a training target, never an input to `h_t`. At inference, the online path
needs only past and current encoded observations. A target encoder may inspect future windows
to construct supervision, but gradients and dataflow must not create a future-to-online-state
shortcut.

## Mathematical object

Let the causal encoder produce `h_t ∈ R^d`. A future-law decoder defines a conditional law
`q_phi(Y | h_t)` over a codec-independent future target `Y`. The decoder induces a local
Fisher pullback on the latent state:

```text
G(h) = E_{Y ~ q_phi(. | h)}[
         grad_h log q_phi(Y | h) grad_h log q_phi(Y | h)^T
       ]
```

For a tangent displacement `v`, the locally meaningful quadratic form is
`v^T G(h) v`. The production path should compute this matrix-free with JVP/VJP products. A
dense matrix is allowed only in tiny-model oracle tests.

`G(h)` may be singular because two latent directions can induce the same future law. That is
why the initial contract says semimetric. Damping may make an operator numerically usable, but
damping does not turn an unmeasured rank claim into a manifold claim.

The first tracer future law is deliberately simple. The online and target encoders have the
same latent width. From a contiguous source span with one registered lookahead position, the
target encoder supplies RMS-normalized states and D1 constructs the displacement
`Y_t = xi_{t+1} - xi_t`. The future decoder predicts
`q(Y_t | h_t) = Normal(mu(h_t), I)`: fixed `sigma^2 = 1`, NLL averaged per position and latent
dimension, objective weight `1.0`. The zero-displacement law under the same variance is the
trivial comparator. A learned variance, target projection, wider future window, or longer
horizon is an additional experimental axis and waits until this diagonal law proves the
pipeline.

## Architectural shape: modular hexagonal monolith

The repository remains one process and one deployable training program. Ports separate
scientific responsibilities; adapters connect datasets, checkpoints, telemetry, and provider
metadata. No service split, broker, actor system, workflow engine, or general plugin registry
is justified by this ladder.

Proposed feature ownership:

```text
src/entropic_substrate/
  observation/
    codec.py                 # observation/readout boundary contracts
    byte_codec.py            # current identity byte adapter
    permuted_byte_codec.py   # D5 reversible relabeling control
  predictive_state/
    state.py                 # latent state value types and causal encoder port
    targets.py               # target encoder and update policy
    stabilization.py         # stop-gradient and collapse strategies
    future_law.py            # probabilistic decoder over future targets
    metric.py                # Euclidean and decoder-Fisher operators
    dynamics.py              # geometry consumers / retractions
    horizons.py              # fixed and learned horizon policies
    coarse_graining.py       # identity, fixed-window, learned-boundary strategies
  experiments/
    specs.py                 # rung-specific validated experiment specifications
    factory.py               # composition root only
  evidence/
    ...                      # existing sinks, telemetry, capture, checkpoint contracts
  analysis/
    ...                      # read-only diagnostics and verdict asset builders
```

The final directory names may follow the codebase’s local conventions, but the ownership
boundaries are fixed: the causal-state domain must not import MLflow, DVC, provider APIs, or an
alphabet-specific readout.

## Semantic value types

The pivot should make illegal substitutions difficult by using narrow typed wrappers rather
than undifferentiated tensors:

- `EncodedObservations`
- `ObservationAlignment`
- `CodecIdentity`
- `SourceSpan`
- `LatentState`
- `LatentTrajectory`
- `FutureObservationWindow`
- `FutureTarget`
- `StabilizedTargets`
- `FutureLaw`
- `TangentVector`
- `MetricVector`
- `MetricQuadraticForm`
- `Horizon`
- `ExperimentIdentity`
- `EvidenceClass`

These are semantic contracts, not tensor subclasses for their own sake. They should validate
shape, causal alignment, dtype, device, and identity at construction boundaries, then keep the
inner numerical loop direct.

## Ports and strategies

### Observation and communication

`ObservationCodecPort`

- encodes external observations into the model’s boundary representation;
- reports a stable codec digest and source-span mapping;
- supports exact inverse decoding where the codec is reversible; and
- owns vocabulary or alphabet cardinality.

`LanguageReadoutPort`

- maps latent state to the evaluation observation space;
- owns the alphabet-sized classifier when next-byte NLL is measured; and
- never exposes its logits or probabilities as the substrate state by default.

The byte codec remains the first adapter because enwik8 is the measured corpus. D5 adds a
deterministic bijective byte permutation before attempting any non-isomorphic codec. That
control preserves the exact information process and byte accounting while destroying accidental
privilege in label identity or order.

D1 keeps byte observation and byte readout parameters untied. Both are boundary objects, but
tying them would let output-classifier training silently rewrite the observation adapter and
would complicate D5 attribution.

### Causal state

`CausalStateEncoder`

- consumes `EncodedObservations` under a causal mask;
- emits `LatentTrajectory` with fixed latent width independent of codec cardinality; and
- exposes no future target to the online forward path.

The existing decoder backbone and `FieldEncoder` contain reusable causal machinery, but the
pivot is not another `field_state` enum member. The new seam separates causal state,
supervision, readout, and geometry rather than expanding one overloaded class.

D1 uses RMS-style final latent normalization rather than a mean-centering LayerNorm output.
This avoids manufacturing the known `d - 1` rank ceiling measured in the frozen-head audit.
Internal block normalization remains an implementation detail held identical across all four
D1 arms.

### Target stabilization

`TargetEncoder`

- constructs a latent target from a registered future window; and
- has a separately checkpointed identity from the online encoder.

`TargetUpdatePolicy`

- `OnlineTargetUpdate`: target and online parameters update normally;
- `StopGradientEmaUpdate`: targets are detached and the teacher updates by EMA only after a
  successful optimizer step.

`CollapseRegularizer`

- `NoCollapseRegularizer`: explicit off control;
- `MomentCollapseRegularizer`: FP32 variance-floor and off-diagonal covariance penalties,
  with effective-rank diagnostics.

Stop-gradient and anti-collapse are orthogonal axes. The D1 matrix is:

| Arm | Target update | Collapse penalty | Scientific role |
|---|---|---|---|
| `OFF/OFF` | online | none | positive collapse/failure control |
| `SG/OFF` | stop-gradient EMA target | none | tests target stabilization alone |
| `OFF/AC` | online | moment regularization | tests anti-collapse pressure alone |
| `SG/AC` | stop-gradient EMA target | moment regularization | tests complementarity and interaction |

The off/off arm is not excluded for collapsing; collapse is its informative outcome. Pooling SG
and AC is forbidden unless a preregistration defines an interaction criterion and the measured
interaction permits it.

Stop-gradient requirements:

- target tensors are detached at the strategy boundary;
- target parameters are absent from the online optimizer;
- EMA updates occur only after a successful optimizer step;
- skipped/nonfinite steps do not advance the EMA cursor;
- checkpoint/resume restores teacher weights, decay schedule, and update cursor; and
- a guard-tooth test must fail when gradient leakage is deliberately reintroduced.

Anti-collapse requirements:

- variance and covariance statistics run in FP32;
- constant-state and rank-one fixtures trigger the penalty;
- a full-rank synthetic state remains finite and trainable;
- effective rank, per-dimension variance, covariance spectrum, and pairwise uniformity are
  emitted alongside loss; and
- regularization weight is a registered constant, not tuned on confirmatory outcomes.

### Future law and metric

`FutureLawDecoder`

- consumes `LatentState` and returns a normalized probabilistic law over `FutureTarget`;
- reports NLL/calibration and target-grounding diagnostics; and
- keeps target width independent of vocabulary cardinality.

`PredictiveMetric`

- `EuclideanMetric`: explicit control, not an implied default;
- `DecoderFisherPullbackMetric`: matrix-free local Fisher action induced by the same future law.

`GeometryConsumerStrategy`

- `EuclideanResidualRetraction`: matched control;
- `PullbackResidualRetraction`: uses the local operator at one registered residual site first;
- later `KappaRelationalConsumer`: optional comparator only after the pullback tracer is
  characterized.

D2 isolates two questions:

1. Does predictive supervision help relative to a latent LM control?
2. Holding the predictive system fixed, does consuming its Fisher pullback help relative to
   Euclidean dynamics?

The primary geometry contrast is therefore geometry-aware minus geometry-off within the same
stabilization block. A composite geometry-versus-baseline contrast cannot identify the metric
mechanism. If D1 leaves multiple stabilization strategies eligible, D2 is stratified by
strategy; it does not silently average them.

Metric health is part of the contract:

- dense-oracle agreement on tiny models;
- finite JVP/VJP products and gradients;
- effective rank and spectrum summaries;
- damping share and null-direction diagnostics;
- “metric consumed” attestation proving that the registered call site changed; and
- matched wall, byte, token, FLOP, memory, and cost evidence classes.

### Horizon and coarse-graining

`HorizonPolicy`

- `FixedHorizonPolicy`: initial registered set `{1, 4, 16, 64}` source-byte positions;
- `LearnedHorizonPolicy`: introduced only if fixed horizons show complementary predictive
  signal worth the complexity.

`CoarseGrainingStrategy`

- `IdentityCoarseGrainer`;
- `FixedWindowCoarseGrainer`;
- `PredictiveBoundaryCoarseGrainer`.

Horizon is not a synonym for word or sentence. A 16-byte future is a measured source span, not
a linguistic ontology. D3 measures whether longer registered futures add predictive rank,
quality, or efficiency. D4 measures whether learned scale adds value over the winning fixed
horizon and a fixed-window macro-state control.

The old “4.5–5 bytes per chunk” expectation from the E4 design is retained only as historical
context. It is not a target ratio or semantic success criterion. Whitespace, punctuation,
fixed-duration, random, always-boundary, and never-boundary controls distinguish learned
predictive boundaries from trivial text heuristics.

## Experiment construction: typestate, not flag soup

The existing flat config surface permits semantically incomplete combinations—for example, an
E1.5c experiment name can resolve to `field_mode = joint`, `field_loss_lambda = 0.0`, and no
checkpoint even when a frozen registered arm requires the opposite. D-series construction must
make the complete experimental identity explicit.

Rung-specific specifications:

- `D1StabilizedLatentSpec`
- `D2PredictiveGeometrySpec`
- `D3FixedHorizonSpec`
- `D4CoarseGrainingSpec`
- `D5CodecEquivarianceSpec`

Each specification validates its own legal state and can be built only after required artifact
identities are present. The composition root returns one complete runtime; the trainer sees a
linear interface and does not branch on every arm.

Required identities include:

- observation codec and inverse codec digest;
- online and target encoder construction;
- target update and collapse strategies;
- future-law family and horizon policy;
- predictive metric and geometry consumer;
- coarse-graining policy;
- preregistration commit;
- checkpoint schema and resume cursor; and
- evidence-envelope version.

## Lifecycle state machine

The prose protocol in `docs/rung-lifecycle-protocol.md` remains binding and should become a
mechanically checked lifecycle:

```text
DraftDesign
    │
    ▼
Implemented ─► ContractVerified ─► ExploratoryPilot
                                      │
                                      ▼
                                Preregistered
                                      │
                                      ▼
                              ConfirmatoryRunning
                                      │
                                      ▼
                                 DvcCaptured
                                      │
                                      ▼
                                   Analyzed
                                      │
                                      ▼
                                BlindReviewed
                                      │
                                      ▼
                                    Filed
                                      │
                                      ▼
                              SuccessorAuthorized
```

Explicit problem states are `Invalidated`, `NumericalDefect`, `EvidenceGap`,
`ReviewReopened`, `OperatorDecisionNeeded`, `ValidNegative`, and `Retired`. A MAJOR blind-review
finding reopens the rung; it does not get summarized away in a comment.

Implementation sessions never analyze their own confirmatory runs. Analysis sessions never
change runtime behavior. Arm mappings stay sealed until the independent review is filed where
the design can support blinding.

## Evidence plane

The existing evidence architecture is retained:

- JSONL plus DVC capture is authoritative;
- `entropic_substrate.ladder.capture.reader.telemetry` is the only admissible read side;
- MLflow is a best-effort operational mirror and never a verdict input;
- checkpoints carry schema and construction identities; and
- missing, partial, nonfinite, nonmonotonic, or incompatible evidence fails loudly.

Capture runs two lanes, partitioned per run by the MLflow `schema` tag. A run with no tag, or a
tag not starting with `h-`, is the E lane: every `results/*.jsonl` goes through
`final_eval_record`. A run tagged `h-...` is the H lane: its `results/metrics.jsonl` must carry
exactly one `run_start`, exactly one `run_end` (schema `h-run-lifecycle-v1`, written by the H
worker through the run's metric sink and never projected to MLflow metrics) and at least one
committed step row, and any other `results/*.jsonl` must be well-formed JSON lines; an
inadmissible H run is refused with a named file, never skipped. A capture pass is atomic with
respect to its destination in both lanes: every file is downloaded and validated in a per-pass
staging directory `<dest>.pending/<pass-id>/`, a sibling of `dest` so `dvc add <dest>` never
tracks it, and only a fully admissible pass is moved into `dest`; a refused pass moves nothing
and keeps the refused file in staging as named evidence. The `run_end` field
`wall_seconds` is measured against a different end point on each arm, and the schema keeps both
meanings under the one name: for `status: ok` it is the worker's own execution time, copied from
the probe result before result admission and final-snapshot custody; for `status: failed` it is
the elapsed time since execution began, read after failure custody (snapshot publish, verify-read
retries, backoff). Consumers must not compare the two arms as one clock. A change to either
meaning is a schema version bump, not an additive field.

An H-tagged run that is FINISHED but has no `h-run-lifecycle-v1` records is refused whatever
produced it: a worker from before FOR-1639, and the local tracer (schema `h-tiny-trace-v1`,
run name `h-tiny-trace-<identity>`) when it mirrors to MLflow under `ENTROPIC_MLFLOW_TINY=1`.
The tracer's run name carries no `-tiny` suffix, so the `include_tiny` gate, which runs before
the lane partition, does not exclude it. Until FOR-1642 (slice 4) gives every local H producer
that bypasses `object_worker` the `-tiny` run-name suffix, capturing an experiment that holds
such a run makes `capture-results` exit 2. Capturing the shared experiment is therefore not yet
safe while such runs remain on the server.

Evidence classes remain separate in every table and chart:

- `MEASURED`
- `DERIVED_FROM_MEASURED`
- `MODEL_ESTIMATE`
- `RATE_ESTIMATE`
- `PROVIDER_BILLED`

The existing Linear milestone `Auditable compute-to-quality telemetry` (`FOR-938` through
`FOR-947`) is a cross-project dependency, not work to duplicate. D-series integration issues
validate and consume that envelope. They do not reimplement it under new issue numbers.

Until those counters land, the repo can report measured trainer wall time, source bytes, and
tokens only where captured. It cannot call estimated FLOPs, accelerator-active time, energy, or
provider cost measured. A missing counter removes the corresponding comparison; it does not
authorize a substitute.

The H worker narrates its mechanisms (every R2 publish, retry, stop-marker observation, completion
write and sink finish) through `kernel.runtime.logging`, using the closed event vocabulary in
`entropic_substrate.ladder.run.worker_log_events`. The structured copy, `worker-log.jsonl`, is uploaded
to the MLflow run at sink finish as an operational mirror that never feeds a verdict.
`sink.finish.end` reports `settled` and `stranded`; the names and the one deliberately event-less
warning are recorded in that module's docstring.

## D1–D5 ladder

| Rung | Isolated question | Primary comparison | Advancement rule |
|---|---|---|---|
| **D1 — Stabilized latent predictive state** | Can a codec-independent latent state learn future structure without collapsing? Which of SG and AC is necessary or complementary? | Four-arm `OFF/OFF`, `SG/OFF`, `OFF/AC`, `SG/AC` matrix under a fixed latent tracer. | At least one noncollapsed strategy must beat the preregistered trivial/latent control on future-target grounding without leakage. |
| **D2 — Decoder-induced geometry** | Does consuming the future decoder’s Fisher pullback help beyond predictive supervision? | Geometry-on versus Euclidean within the same predictive system; latent LM and supervision-only controls retained. | Carry Fisher consumption only if a registered within-strategy geometry contrast passes without a contradictory strategy interaction. Predictive supervision may advance without Fisher if its own clause passes. |
| **D3 — Future-law family and horizon sweep** | Do longer or multihorizon futures add complementary information at acceptable resource cost? | Fixed `{1,4,16,64}` source-byte horizons and a preregistered multihorizon composite, in matched-backbone and matched-total lanes. | Learned scale is earned only if a longer/composite future adds registered value beyond the settled short-horizon state. |
| **D4 — Multiscale coarse-graining** | Can predictive dynamics discover useful temporal boundaries beyond a fixed-window control? | Identity/fixed-window first; learned predictive boundaries only behind the filed trigger. | Kill learned chunking if it collapses, merely rediscovers a heuristic control, or fails the fixed-window/resource-normalized comparison. |
| **D5 — Codec quarantine and interface generalization** | Is the settled state equivariant to reversible relabeling, and where does codec dependence leak? | Identity bytes versus registered bijective byte permutations; alternate codec only after that control. | “Codec-equivariant” is allowed only if quality, functional behavior, latent geometry, and scale-policy clauses all pass their registered margins. |

No rung inherits a winner merely because it is newer. Each verdict names the exact artifact and
strategy carried forward, or closes the branch.

## Relationship to the existing E3 design

`docs/e-3-instruction.md` correctly established a durable principle: curvature is relational
and acts where points meet. The measured nullity of a radial single-point kappa normalization is
valuable and remains pinned.

What changes is causal order. The kappa attention family does not itself estimate the
decoder-induced Fisher pullback, and a constant-curvature kernel is not evidence that the
predictive state has that geometry. D2 begins with Euclidean versus pullback-aware consumers.
Existing nGPT/kappa blocks become later consumer candidates or comparators after the predictive
metric has a measured contract. They do not define D1 state construction.

## Relationship to the existing E4 design

`docs/e-4-instruction.md` supplies two useful implementation ideas:

- a fixed-window chunk-superposition control; and
- a quarantined learned-boundary/container lane.

The pivot reorders them around latent predictive state. Fixed-window macro-state construction
comes first and must preserve byte/source-span accounting. Learned boundaries come second and
are conditional. Pure Torch is the tracer-bullet path; the H-Net/Mamba container lane is an
optional scale implementation only after the tracer earns it.

The previous fixed compression-ratio expectation is retired as a semantic objective. The model
must earn its scale from predictive consequences, not from an English token-length heuristic.

## TDA and geometry diagnostics

Angular drift, Gram retention, Procrustes residuals, functional KL/JS divergence, effective
rank, pullback spectra, and persistent-homology summaries are observers. They report relational
and functional structure without pretending raw coordinates are identifiable.

TDA is not a D1–D5 loss and not a gate that rescues a failed quality result. After D5, a
separately registered capstone may test whether the settled predictive pseudometric has stable
topology across seeds, horizons, and codecs. Until then, “TDA-like” is design motivation, not a
measured property of the model.

## Existing component disposition

| Existing asset | Disposition in the pivot |
|---|---|
| E0/E1 configs, captures, and verdicts | Preserve unchanged as historical controls and evidence. |
| `core/substrate/fisher.py` | Preserve as the E1 warm-start control; do not promote it to the D-series semantic state. |
| E1.5-D geometry diagnostics | Reuse as report-only relational observers. |
| E1.5a/E1.5b freeze/Gram arms | Preserve for attribution history; no automatic D-series carry-forward. |
| `FieldEncoder` H/P/F paths | Preserve as bridge controls and possible target-encoder components. Do not add another enum member as the pivot architecture. |
| `ladder/analyze/field_head_audit.py` and the frozen-head findings | Complete as exploratory D-entry evidence; use it to refine controls, not as a confirmatory D verdict. |
| nGPT projection/retraction blocks | Candidate geometry consumer and matched control after D1. |
| kappa attention | Later relational comparator after the decoder-Fisher tracer; not a substitute for it. |
| E4 fixed-window code/reference | Reuse as D4 fixed control over latent states. |
| H-Net/Mamba references | Keep quarantined for conditional learned-boundary scale-up. |
| `MetricSink`, JSONL, MLflow fan-out, telemetry reader, DVC capture | Preserve as the evidence plane and extend only through the registered telemetry dependency. |

## Non-negotiable numerical and failure contracts

- Metric, normalization, covariance, spectrum, and collapse diagnostics run in FP32 or FP64
  islands with autocast disabled as specified by the operation.
- Matrix-free and dense-oracle metric actions must agree on tiny fixtures.
- Nonfinite loss, metric action, collapse statistic, or cumulative counter invalidates the
  affected run/pair according to the preregistration; no fail-open wrapper is permitted.
- Unexpected invariant violations fail hard with structured context.
- Expected registered negatives—collapse, low rank, no horizon value, trivial boundaries, or
  codec dependence—are typed outcomes and remain in the evidence.
- Resume refuses mismatched codec, target encoder, target strategy, future-law, metric,
  horizon, coarse-graining, or evidence-envelope identities.
- Every guard proves both arms: happy path and deliberate mutation/negative fixture.

## Deliberate exclusions

- No claim of universal language encoding from enwik8 or English-only byte experiments.
- No claim that a byte permutation establishes equivalence across bytes, bits, code points,
  BPE tokens, speech, or images.
- No alphabet-sized tensor inside the semantic core, while alphabet-sized observation/readout
  adapters remain necessary for the measured task.
- No learned tokenizer, word detector, sentence detector, or linguistic-boundary target in D1–D3.
- No TDA training objective in D1–D5.
- No Riemannian optimizer until a measured consumer result shows the forward operator is worth
  carrying.
- No distributed services or extension framework for a single experimental ladder.

## Migration sequence

1. File the current frozen-head audit as explicitly exploratory and preserve its inputs.
2. Seal the D-series semantic value types and rung-specific construction specs.
3. Consume—not duplicate—the auditable compute-to-quality envelope.
4. Build D1’s latent tracer, target encoder, four stabilization arms, and collapse telemetry.
5. Complete the D1 pilot, preregistration, confirmatory capture, fresh analysis, blind review,
   and verdict before fixing D2’s eligible strategy set.
6. Add the future law, Euclidean control, dense Fisher oracle, matrix-free pullback, and one
   geometry-consumer call site for D2.
7. Run the D3 fixed/multihorizon comparison under the settled D2 posture.
8. Run D4 fixed-window control; launch learned boundaries only if its registered trigger fires.
9. Run D5 identity/permuted-codec equivalence, then one alternate codec only if the reversible
   control is clean.
10. Close with a cross-rung synthesis and, only if separately preregistered, a report-only
    topology capstone.

The execution-level decomposition, issue hierarchy, gates, and Linear identifiers live in
`docs/pivoted-implementation.plan.md`.
