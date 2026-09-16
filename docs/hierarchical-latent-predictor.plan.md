# Codec-Separated Sentence Prediction: Preservation-Gated Route to 500M–1B

**Date:** 2026-09-15.
**Status:** approved implementation contract; implementation is in progress. This is not a
preregistration, completed implementation, scientific result, or spending authorization.

## North star

Generate text by predicting the next continuous sentence representation. A learned boundary
encoder consumes UTF-8 bytes, a causal Transformer advances on sentence states, and a local
byte decoder realizes sampled predictions. The future law supervises a latent state and
induces geometry for its dynamics; its alphabet-sized probabilities never replace that state.

The destination is approximately 500M total deployed parameters, up to 1B only behind a new
gate and quote. Count every unique deployed encoder/spine/decoder parameter, including frozen
modules; disclose training-only teachers and external pretraining separately. Transformer
comes first. Mamba-2, paragraph/document recursion, and learned boundaries remain later arms.

The E1.5c frozen-head audit and checkpoint replay found a full-rank centered head with
position-selective reconstruction loss where the probability floor activated. This establishes
numerical corruption, not the fraction of semantic information lost or the fraction of the
language-loss deficit caused by the floor. Preserve the historical code and verdicts.

**Binding correction:** no `softmax → probability floor → sqrt → lift` replacement of the main
latent state. Sphere-native states remain a candidate, not a preservation assumption.
Sentence normalization and spine normalization are separate, measured experimental axes.

## Horizon

Complete one causal sentence encoder → predicted latent → text decoder path, including
multi-sentence generation, checkpoint/resume, capture, fresh analysis, independent review,
filing, and an explicit gate. Teacher-forced reconstruction or a forward call is insufficient.

Engineering completion is the tested generator and its complete evidence path. Scientific
completion is the sequential, powered, reviewed programme. Deployment completion is the
retained approximately 500M Transformer meeting registered criteria, not reaching a parameter
count. Restrictions survive only where evidence supports them. No inferred new-stage authority
from E/D results, old registrations, local fixtures, account balance, or a running pod.

## Architecture and causal contract

```text
completed sentences → learned encoder → continuous sentence vectors → latent Transformer
  → conditional future law → sampled vector → local byte decoder → emitted sentence
  → re-encode emitted text for the next operative state
```

Retain the sampled vector as a shadow prediction, not an additional active feedback input.
A target encoder may read the next completed sentence for supervision, but that target must
not enter the context predicting itself. Prove causality with future-suffix mutations.

### Data and boundary

- UTF-8 bytes with versioned sentence/document control symbols; preserve the historical codec.
- Flat and hierarchical controls share the boundary protocol and account for boundary scores.
- Bounded, captured FineWeb-Edu documents at the existing pinned revision; enwik8 is regression
  and historical data. Add a document-preserving loader alongside the historical flat stream.
- Split and deduplicate documents before sentence construction. Preserve source IDs, original
  whitespace, byte offsets, order, and preprocessing identity.
- Pin English pySBD, disable cleaning, validate spans. Boundaries are preprocessing assumptions,
  not proof that every sentence constitutes a semantic concept.
- Initially accept prompts ending at explicit sentence boundaries.
- Development limits: 4,096 source bytes per sentence and 64 completed sentences of macro
  context. Reject and count oversize records; never silently truncate or relabel fragments.
- Partial-sentence prompting, multiple semantic slots, and recursive levels remain deferred.

### Learned codec and profiles

Train our own encoder and decoder from scratch without a pretrained training teacher. The
encoder may attend bidirectionally within a completed sentence and produces one pooled vector.
Warm up with original-sentence reconstruction; evaluation allows paraphrase but checks entities,
numbers, negation, relations, and order. Freeze the encoder for initial predictor studies,
training the spine and decoder. Unfreezing is a separate later arm.

The decoder receives the predicted representation and its current sentence's causal byte
prefix, never unrestricted raw history. Re-encode actual emitted text for operative feedback.
Development profile: sentence width 384, four encoder layers, 12-layer/768-wide spine, four
384-wide decoder layers. Tiny profiles shrink sizes without disabling the scientific axis.
Actual inventories, not architectural estimates, establish reported parameter count.

### Independent preservation controls

| Axis | Spherical candidate | Magnitude-preserving control |
|---|---|---|
| Sentence representation | Unit direction | Continuous vector retaining magnitude |
| Spine carrier | Unit retraction | Residual magnitude retained; normalized sublayer views allowed |

Implement the two switches independently and declare the full 2×2 comparison. Share attention,
sublayers, initialization, and parameter-normalization policy where structurally possible;
record unavoidable differences. A magnitude-preserving carrier has no hidden final UnitNorm
or RMS overwrite. The existing D encoder does not satisfy that condition unchanged.

### Effective directional/magnitude law

Start with `U ~ PowerSpherical(mu(h), kappa)`, `rho ~ Normal(m(h), sigma_rho²)`, and
`C = exp(rho) U`. This factorized family is a proposed baseline, not a sufficiency claim.
The magnitude-preserving interface consumes C. The spherical interface consumes U and
integrates out unused magnitude. Magnitude belongs to the same vector, not a second slot.

Declare the density base measure and change-of-variable terms. Never apply an ambient
Gaussian density formula directly to normalized points. Unused magnitude branches are
explicitly inactive, excluded from optimizer participation requirements, and disclosed in
active/deployed inventories. Dormant parameters cannot pad matching capacity.

Excluded-development defaults: mean resultant length 0.9 and log-magnitude standard deviation
0.1. No learned dispersion, mixture, or diffusion in this first construction. Initial latent
and surface sampling use temperature 1 without top-k/top-p. Failure to emit a boundary and
length truncation are explicit outcomes, never silently completed sentences.

### Law-correct dynamics

Use the final spine block as the single intervention site. For the magnitude-preserving
carrier, compare the ordinary proposal with its damped inverse-Fisher transformation. For
the spherical carrier, preserve the nGPT proposal's radial component, transform its tangent
component, and retract. Substituting the isotropic tangent metric must reproduce the original
spherical update within the declared tolerance.

Compute the pullback of the effective active law: directional for the spherical sentence
interface; directional plus magnitude otherwise. Do not transplant fixed-Gaussian JᵀJ without
law-specific factors. Matrix-free products and solves remain fully differentiated through
state, proposal, and active law parameters. Recompute at the current state: no stale graphs,
detached metrics, future-target leakage, or automatic isotropic fallback.

Development solver: damping 1, 32 iterations, relative tolerance 1e-5, absolute tolerance 1e-8.
Normalization, densities, and Fisher operations establish their own FP32 islands; historical
UnitNorm does not establish this stronger contract. Dense FP64 tiny oracles validate the
production action. Exact-zero right-hand sides retain their mathematical derivatives.

## Sequential experimental route

| Stage | Question | Required exit |
|---|---|---|
| H0: Complete tracer | Does generation and evidence custody work? | Both carriers, stochastic rollout, recovery, capture, analysis, review, filing, test-only gate share one lineage. |
| H1: Preservation | What do normalization and its interaction lose? | Paired 2×2 controls, independent detail checks, numerical and observable predictive evidence. |
| H2: Predictive value | Does hierarchy help; does future supervision help? | Flat byte, hierarchical language-only, and hierarchy with future-law supervision; separate conclusions. |
| H3: Geometry | Does Fisher consumption add value? | Isotropic versus differentiated Fisher within every eligible construction; measured resources. |
| H4: Interface/scale | Do conclusions transfer? | Coupled permutation correctness, separate paired recoded training, larger Transformer study. |
| H5: Deployment | Does the retained ~500M model generate useful text? | Registered prediction, fidelity, rollout, and resource criteria; independently reviewed verdict. |
| Later | 1B, Mamba-2, unfreezing, richer laws, recursion | Separate hypothesis, controls, registration, quote, approval; never bundled rescue changes. |

H1 uses isotropic dynamics and no auxiliary future-law objective, keeping preservation
separate from geometry consumption and predictive supervision. Combine fixed-state
interventions/matched diagnostic decoders with paired retraining. Recoverability, adaptation,
and usefulness are different tests. Retain every construction meeting registered eligibility;
if both fail, pause rather than automatically scale.

Bind shared codec artifacts within seed blocks only where the contrast calls for them. When
codec construction differs, warm it independently on the same paired source stream and
budget. Repeat warm-up across confirmatory blocks. Successor spines start from fresh paired
initializations, carrying constructions, not trained predecessor weights. H2's language-only
arm must still train its predictor through the observable decoder loss.

Planning anchors are approximately 100M → 250M → 500M deployed parameters. Admit profiles only
after inventory and representative resource measurement. Default matching tolerance is 1%;
otherwise do not label a comparison matched. Re-estimate variance at each regime. Separate
shared-configuration tests from equal-budget per-arm tuning. Account for warm-up, repeated
source presentations, encoder/decoder and diagnostic exposure. Tuned-frontier claims are
confined to measured sizes, data, competitors, and tuning envelopes.

## Evidence and statistical contract

### Preservation and usefulness

Measure around pooling, representation normalization, spine retraction, sampling, precision
conversion, and decoder conditioning: reconstruction/angular error where defined; norms,
covariance/effective rank, conditioning, saturation counts, actual computation dtype;
predictive-probe quality; detail preservation; shadow-versus-emitted state; matched
perturbation effects on output laws and signed predictive scores.

No norm ratio, correlation, or rank becomes a percentage of semantic information retained.
Use stable log-space calculations where appropriate. Probability floors, denominator
safeguards, parameter-domain constraints, and gradient clipping have distinct roles: audit
each rather than approving or banning all operations named clamp. Neither numerical
recoverability nor large Fisher sensitivity alone establishes useful prediction.

### Honest generative scores

Text likelihood marginalizes its latent variable. Decoding the true future sentence's
encoding is an oracle reconstruction diagnostic, never the generator's language likelihood.
Implement importance-weighted scoring with the frozen encoder proposal: development uses
64 samples plus a fixed 256-sample diagnostic subset. Capture weights, effective sample
size, Monte Carlo variation, and sample-count sensitivity. Integrate out unused magnitude.
Name the bound/estimate and boundary-augmented task. A better bound alone does not certify
better true likelihood.

Pair scores with controlled detail tests, a frozen revision-pinned evaluation-only model
(initial candidate `cross-encoder/nli-deberta-v3-base`), and a blinded human audit with fixed
rubric and sampling. No evaluator training signal. Measure free-running coherence,
contradiction, repetition, termination, and boundaries. A valid continuation need not
paraphrase the corpus's single continuation.

### Span-aware predictive score gain

Version sentence-update evidence separately; never weaken the adjacent-byte reader. Before
and after scores target the same canonical future byte under the same checkpoint, frozen
probe, document, split, codec, and target-offset identities. Record sentence-event and raw-byte
coordinates and remaining horizons. Source-byte lags are {2,4,16,64} after the completed span.

Fit equal-capacity horizon-conditioned probes independently per arm on detached training
states. Development: 65,536 training positions; 200 updates; batch 256; learning rate 1e-3;
weight decay 1e-4; seed 42; 8,192 held-out positions. Freeze before scoring; capture weights,
sampling identities, and paired scores. Include uniform and horizon-matched shuffled-state
controls. Report signed gain, dispersion, negative frequency, and predictive quality. This
remains diagnostic unless a later registration promotes it; it is not mutual information.

### Replication and registration

Retain 95% familywise deciding inference, Bonferroni over the complete deciding family,
predeclared secondary 3-sigma reports on the same sealed data, and 90% planned power per
primary contrast. Freeze proposed effects, margins, regimes, and pilot-selection rules before
pilot inspection. Eight complete paired pilot blocks are a separately quoted variance proposal.
Use conservative variance bounds and appropriate power calculations; projections are not
guaranteed achieved power. Do not transfer E/D or tiny-run variance.

Commit seed matrix, endpoints, multiplicity, reader identities, exclusions, budgets, and
successor clauses before confirmation. Seed blocks are model replicates; documents, probe
positions, and generated samples are nested measurements. Finite collapse stays in the
population. Missing/nonfinite/wrong-owner gradients remain defects; measured-zero and
structurally absent gradients remain distinct. No outcome-selected replacements or extra
looks. Missing margins or an unaffordable powered matrix block execution, not enforcement.

The historical 0.01 nats/byte reference does not automatically become a margin for a different
bound or semantic metric. Every new margin requires explicit units and prospective justification.

## Execution, delivery, and approval

Each stage: local validation → approved remote tiny → excluded pilot → committed
preregistration → confirmation → DVC capture → fresh analysis → independent review → filed
verdict → successor gate. Smoke, pilot, and confirmation are separate authorities.

Reuse the pinned project image and digest, R2 transport, DVC custody, and artifact-based
completion evidence. Before remote execution verify credentials and pin one verified CLI
build. Production CUDA/bf16 tiny profiles retain their mechanisms. Set provider termination
and controller deadlines. Recover checkpoint, optimizer, RNG streams, source order, and
operative/shadow feedback before resume. Verify capture externally before deleting the owned
pod; confirm termination and retain billing. Shared volumes are not disposable probe output.

Implement dependency-ordered commits, each at most 599 added-plus-deleted lines including
tests and documentation. RED-first contracts, guard-tooth mutations, normal hooks, repository
quality gates, integration tests, and applicable sequential live-container gates are required.
Do not modify hook configuration without separate approval.

1. Document/sentence identities, boundary codec, source-preserving capture.
2. Learned codec, directional/magnitude law, both sentence interfaces.
3. Both spine carriers, local decoder, rollout, shadow feedback.
4. Preservation diagnostics and independent normalization controls.
5. Law-correct Fisher actions, differentiated consumers, numerical oracles.
6. Generative scoring, fidelity evaluation, span-aware score gain.
7. Experiment adapters, inventories, resource accounting, pilot preparation.
8. Minimal shared custody/execution integration and one complete golden lineage.
9. Full local gates, then separately approved eligible remote validation.

Acceptance includes future-suffix/target mutation guards; exact source and inverse codec
mapping; no hidden normalization overwrite; deliberate clamp/retraction/precision-loss
mutations; isotropic equivalence; dense/matrix-free/score/local-KL Fisher agreement and full
derivatives; typed zero-collapse/undefined geometry; unambiguous oracle versus marginal
scores; RNG and feedback recovery; no pilot/smoke admission into confirmation; one lineage
covering supported, valid-negative, inconclusive, numerical-defect, missing-evidence, and
review-reopened outcomes.

Keep seams narrow: sentence records, codec, latent law, carrier/update convention, local
decoder, versioned evidence. No generalized recursive framework in advance. The unfinished
old D host tail remains a recoverable deferred snapshot, not implicitly restored or approved.

Approval authorizes implementation, not GPU spending, publication, or confirmation. Quotes
and approvals remain separate for probes, pilots, confirmation, evaluation, and publishing.

## Source preparation command

`python -m entropic_substrate.cli h-prepare-source --root ROOT --manifest FILE`
prepares and independently replays one bounded FineWeb source capture. Run it in
the intended reader runtime; the command does not relax Python/segmentation identity
checks or grant training authority. `ROOT` and its declared output directory must
already exist; the output directory must be empty. No corpus download occurs in
the unit tests, which substitute provider rows while retaining real preparation.

The manifest requires exactly `schema: h-prepare-source-command-v1`, `dataset_name`,
`revision` (the existing `FINEWEB_REVISION`), `output_root`, `max_artifact_bytes`,
`candidate_limit`, `admitted_byte_limit`, `max_sentence_bytes`,
`segmentation_version` (0.3.4), `split_seed`, `validation_buckets`, and `test_buckets`.
No omitted policy field is filled from a runtime default. Candidate/admitted-byte
limits do not claim to bound provider network traffic. Empty corpora and rejected
documents remain explicit source evidence, not an assertion of model readiness.

`h-inspect-source` uses the same root/manifest options with exactly
`schema: h-inspect-source-command-v1`, `source_root`, `source` (the returned
`path`/`sha256` identity), and `max_artifact_bytes`. Inspection verifies the source
hash and reader, then replays preparation without contacting the provider.

New source captures use `h-source-capture-v2`: dataset configuration participates
in preparation/corpus identity even for empty results. Synthetic or programmatic
sources without a declared provider configuration serialize explicit `null`; the
FineWeb command always supplies the selected name. V1 evidence remains unchanged
and requires its original reader; no automatic migration or cross-reader resume
is permitted. Preparation tests reproduce the former empty-configuration collision.

## Research provenance

- [Frozen-head audit](e1-5c-head-audit-findings.md): exploratory numerical preservation evidence,
  not a revised E1.5c verdict or a D/H successor gate.
- [NCP-ArchPreview](https://arxiv.org/abs/2609.10715): concept prediction feeding surface
  generation; fixed token groups/codebooks are not our sentence-first continuous construction.
- [Large Concept Models](https://arxiv.org/abs/2412.08821): sentence-representation prediction
  is prior art; external representations bring teacher/data provenance.
- [Power Spherical](https://arxiv.org/abs/2006.04437): directional density and sampling; not
  evidence that the proposed directional/magnitude law improves text generation.
- [Picozzi](https://arxiv.org/abs/2609.11063): output-Fisher and matched-effect interventions
  motivate diagnostics, not a universal codec or hierarchical generator.
