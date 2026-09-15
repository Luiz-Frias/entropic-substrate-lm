# Codec-Separated Hierarchical Latent Predictor

**Date:** 2026-09-14.

**Status:** approved destination and sentence-first posture; provisional experimental itinerary.
This is not a preregistration, a completed implementation, or spending authorization.

## North star

Generate text by predicting the next continuous chunk representation in a sequence. A
boundary encoder maps observed text into latent vectors or slots, a small Transformer or
Mamba-2/state-space spine models the chunk trajectory, and a local decoder realizes predicted
latents as text. Start with sentences. Paragraph, document-context, learned-boundary, and
recursive scales must earn their value through controlled comparisons.

The intended deployment envelope is roughly 500M total parameters, up to 1B. Count all unique
deployed encoder, spine, and decoder parameters, including frozen deployed modules. Report
training-only teachers, external pretraining, and their data/compute separately. No training
budget or parameter-matched result is inferred from this target.

## Horizon

The first complete tracer must generate a multi-sentence continuation from an observed prompt,
using predicted rather than future-target latents, with checkpoint/resume and one coherent
execution/capture/analysis/review/filing lineage. Its controllable paths must expose whether
concept predictions influence the output. Teacher-forced reconstruction alone is insufficient.

Programme completion requires a reproducibly evaluated generator in the deployment envelope
meeting preregistered generation-quality, noninferiority, and resource criteria, with explicit
support or retirement decisions for every tested mechanism. A successful final
model need not retain Fisher dynamics, recursive hierarchy, or a particular backbone if those
mechanisms fail their registered value tests.

## Architecture and causal contract

1. **Observation adapter:** bytes or BPE are boundary representations, not the core's ontology.
   Hold the initial codec fixed. Existing byte tooling is reusable; a different tokenizer is
   a separate construction and comparison, not a silent change to the enwik8 experiments.
2. **Chunk encoder:** map each completed sentence/span to `C_k`, one vector or a declared set
   of latent slots. Preserve source offsets, order, boundary policy, and document identity.
   Sentence boundaries are a controlled starting assumption, not a definition of a concept.
3. **Latent spine:** advance `s_k` from past chunks and predict `q(C_next | s_k)` or a declared
   deterministic next-vector forecast. The main sequence clock advances by chunks.
4. **Geometry consumer:** use a predictive-law pullback to transform latent
   dynamics. Keep a matched Euclidean consumer. The law, intervention site, precision,
   damping/reference metric, solver, and differentiation convention are explicit identities.
5. **Surface decoder:** realize predicted latents as text. A local next-byte/token objective
   may remain; moving the main spine to chunk prediction does not eliminate decoding.
6. **Higher levels:** aggregate and predict sentence trajectories at paragraph/document-context
   scales only after the lower-level signal and its utility are established.

A target encoder may read the next complete chunk to construct training supervision. That
target must not enter the context used to predict itself or earlier surface tokens. A summary
of a completed span becomes available only after its closing observation. Test this through
future-suffix mutations, not just function signatures.

Declare the free-running feedback rule: advancing on predicted latents and re-encoding emitted
text are different constructions. Test generated continuations, not only teacher-forced
latent histories. If a local detail/bypass path exists, measure its contribution and show that
the concept path is not ignored.

## Information, geometry, and resolution are different axes

- A fixed-size finite-precision vector cannot preserve arbitrary-length text losslessly.
  Temporal coarsening, reduced latent width, quantization, and numerical clipping are distinct.
  Test preservation of relevant detail, including entity identity, numbers, negation, order,
  and long-range dependencies. Extra latent slots or a detail path are explicit capacity axes.
- Fisher geometry requires a predictive law; it is not an encoder or a substitute for fitting
  that law. Keep full conditional structure rather than treating scalar entropy as meaning.
- A latent-future-law metric and an observable text-output metric are different instruments.
  Name and validate each. A cheap latent proxy does not automatically predict downstream
  decoder divergence; test that connection using matched perturbations and held-out outputs.
- Local Fisher cost measures unsigned output sensitivity. Signed predictive improvement and
  actual intervention outcomes determine whether a change is useful. Large effect is not
  necessarily positive effect, and movement is not the creation of new information.
- Damping is part of the geometry contract. An identity reference in one coordinate system is
  not automatically the same regularized problem after an arbitrary change of coordinates.
- Paragraph prediction has a natural within-document order. Whole-document prediction needs a
  coherent conditional task; adjacency of randomly shuffled documents is not semantic scale.
  Document planning/completion and section-level generation are candidate tasks to define.
- Quantum-state geometry is an analogy and a possible separate research direction, not a
  requirement for this classical implementation or evidence of codec independence.

## Provisional itinerary (not registered experiment IDs)

| Stage | Question | Controls and decision boundary |
|---|---|---|
| 0: Complete sentence tracer | Does the encoder → next-vector predictor → decoder actually generate causally? | Tiny end-to-end lineage, oracle target only as a labelled diagnostic, no future leakage, checkpoint/resume, observable concept-path influence. No scientific win from a smoke test. |
| 1: Sentence-signal utility | Does the architecture or explicit latent prediction help? | Flat surface baseline; same hierarchy with/without the explicit latent objective; matched budgets, readout, and data. Keep reconstruction, predictive grounding, and language utility separate. |
| 2: Predictive geometry | Does a specified metric improve useful dynamics beyond that same predictive system? | Euclidean versus the declared Fisher consumer; separately identify latent-law and text-output metrics. Compare achieved benefit, disturbance, and resources. |
| 3: Temporal hierarchy | Do paragraph/document-context or recursive levels add value? | Sentence-only and fixed grouping before learned grouping; preserve source exposure, context, and detail. Higher granularity may be retired. |
| 4: Sampling and rollout | Which conditional latent family and rollout policy work? | Change one sampling axis at a time; measure calibration, diversity, likelihood validity, and free-running stability, not attractive examples alone. |
| 5: Backbone and codec | Which conclusions survive implementation/representation changes? | Hold the latent interface fixed while comparing backbones; test bijective relabeling before non-isomorphic tokenizers. Do not confound both swaps. |
| 6: Deployment scale | Does the retained construction hold up near 500M, then at most 1B? | Measured parameter inventories, stated per-arm tuning budgets, source/compute accounting, powered confirmation, and resource quotes. No extrapolated gain or automatic scale-up. |

Stages are separable decisions, not a requirement to combine every candidate. Detailed arm
rosters, interactions, thresholds, and successor rules must be registered before execution.
The old D registrations cannot be reused as authority for these new constructions.

## Sampling: name the random object before choosing a distribution

- **Training exposure:** propose shared deterministic document/span streams and a uniform
  reference sampler first. Sentence-uniform and byte-uniform weighting differ. Split documents
  before making chunks; preserve contiguous context. Length/domain/rarity stratification,
  curriculum, or power-law reweighting changes the training distribution and needs its own arm.
- **Latent futures:** Gaussian, mixture, flow/diffusion, and deterministic forecasts are model
  family choices. Sampling arbitrary Gaussian vectors does not make them meaningful concepts.
- **Numerical probes:** Gaussian/Rademacher directions and power iteration estimate different
  operator properties; neither selects the training examples or establishes statistical power.
- **Surface decoding:** temperature, truncation, and other sampling choices must be fixed for
  comparisons. Report likelihood/calibration separately from sampled-generation behavior.
- **Replication:** pilot seed blocks estimate variability across trained models. More positions
  or samples from one model do not replace independent model seeds.

A stochastic latent generator requires an honest likelihood contract: text likelihood is
marginal over its latent variable. Do not present decoder likelihood conditional on the true
future chunk's encoding as language-model likelihood. A deterministic predicted-latent path
is a simpler control; marginal estimators or bounds must be named when stochastic latents are
introduced. The initial predictive family and feedback policy are still design decisions.

## Mechanism evidence and statistical contract

Build controllable wiring before running the full tracer. Then use two complementary tests:

1. **Inference interventions:** remove, shuffle, or replace a concept signal with matched
   controls to measure reliance, localization, and perturbation effects in a trained system.
2. **Paired retraining ablations:** train the corresponding controls from fresh paired starts
   to estimate the mechanism's incremental value and interactions. Removing a trained branch
   alone is not the same counterfactual as training without it.

Predeclare signed benefit in held-out observable units, latent grounding/calibration, detail
preservation, reference-output disturbance, and resource cost. Equal-norm interventions need
not have equal effects; where comparing disturbance efficiency, match achieved target effect
and use separate calibration/evaluation data. Keep perturbations within the declared local
regime and verify predictions against realized behavior.

Retain 95% familywise deciding inference, Bonferroni over the complete deciding family,
secondary predeclared 3-sigma reporting on the same sealed population, and 90% planned power
per primary contrast. Size confirmation from conservative variance bounds on excluded,
representative paired pilots; eight complete pilot blocks remain a separately quoted proposal,
not a guaranteed sufficient sample. Specify raw effect sizes and justified practical margins
before pilot inspection. Do not transfer old E/D variance or tiny-run variance to new stages.

Seed blocks are model replicates; documents, contexts, and samples have their own nested
measurement roles. Keep finite collapse in the declared population. Missing/nonfinite
gradients, wrong optimizer ownership, or invalid evidence remain structured defects. No
outcome-selected replacement seeds, extra looks, or post-seal margin changes.

Scientific completion remains pilot → committed preregistration → paired confirmation →
authoritative DVC capture → fresh analysis → independent review → filed verdict → successor
gate. Probe, pilot, confirmation, publication, and scale-up retain distinct approval boundaries.

## Relationship to prior work and required decisions

- [NCP-ArchPreview](https://arxiv.org/abs/2609.10715): a concept-prediction pathway feeds token
  generation; its quantized codebooks and fixed token groups are not this sentence-first
  continuous design or evidence for Fisher dynamics.
- [Large Concept Models](https://arxiv.org/abs/2412.08821): sentence-representation prediction
  is direct structural prior art. Borrowing pretrained representations adds teacher/data
  provenance that must be counted, not treated as free input.
- [Picozzi, information geometry](https://arxiv.org/abs/2609.11063): output-Fisher geometry and
  matched-effect interventions motivate behavioral diagnostics, not a sentence hierarchy or
  universal codec. This is a new preprint; its code and derived data are available on request,
  and its reported experiments have not been reproduced here.

Before building the first new scientific construction, settle the corpus/document boundaries,
encoder initialization and teacher, latent width/slots, predictor and decoder split, feedback
and likelihood conventions, first target family, held-out readouts, and pilot/tuning budgets.
The existing E/D code is reusable only where its contracts fit. Its historical verdicts and
uncompleted host tail are not silently promoted into new-stage evidence.
