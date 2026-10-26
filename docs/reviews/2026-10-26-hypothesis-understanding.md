# Hypothesis understanding — 2026-10-26

## Archival provenance

This is the complete substantive conceptual response requested by Luiz before the
source audit. Its historical claims and recommendations are preserved, not silently
updated to match the implementation. Conversation-only memory-citation metadata is
omitted. The subsequent systems checklist is a separate review, not a revision of
this snapshot or a scientific verdict.

Archive branch: `docs/2026-10-26-hypothesis-systems-review`, based on stack tip
`d6935e2830f8502f9e86151de90b9438f01b6bc7`. The original response explicitly did
not inspect that implementation. Recommendations below do not amend registrations.

## Original response

**My understanding is that you want the model’s main generative process to operate on learned, contextual units of information—not merely to give a conventional token-level model better starting embeddings.**

The proposed advantage has two layers:

1. **Representation and computation:** predict the next sentence-level latent state, realize it as text, and eventually compose those states into larger predictive structures.
2. **Geometry:** organize changes to those states according to their predictive consequences, so the model can make useful changes without destroying other information or paying unnecessary computational cost.

The destination is a useful generator around **500M total deployed parameters**, with larger models, Mamba-2, and recursive paragraph/document levels earned through experiments.

The central question is therefore:

> **Can a preservation-aware latent sequence become a materially better medium for generative prediction—and does predictive geometry make that medium more effective, selective, or efficient?**

That is stronger than “Fisher has a nonzero effect,” “the embeddings look structured,” or “the model can reconstruct a sentence.”

I reread **“fisher program direction,”** the September 14–15 design decisions, and selected later project transcripts, including the October evaluation discussion. I also read relevant methods, ablations, and limitations from the papers discussed below through alphaXiv and Firecrawl. **I have not inspected the current source code, Git state, or live execution readiness.** This is the conceptual alignment pass you requested.

## 1. The linear history of the idea

### 1. The original intuition: relationships should define the representation

Your substitution-cipher analogy was about arbitrary external labels: why should a symbol’s numerical identity determine the substrate on which learning occurs?

The durable idea was not literally “remove every encoding.” It was:

- construct representations through context and relationships;
- compare predictive distributions rather than rely on arbitrary coordinate distances;
- let meaningful larger-scale units emerge through composition;
- make the main computation operate at a useful information resolution.

The “encoding is a field” phrase captured a **living construction process**, not a static dictionary.

Some early explanations overstated the ASCII problem. Embedding lookup does not impose numerical distances between token IDs, and byte-level BPE can be reversible. The stronger objection survived that correction: **surface segmentation need not determine the main model’s computational clock or abstraction level.**

### 2. E0/E1: we first tested an initialization prior

The first implementation reduced that broad intuition to a clean, narrow experiment:

- ordinary byte/BPE baselines;
- random spherical byte initialization;
- corpus-derived Fisher initialization;
- otherwise conventional training.

The reported E1 evidence supported an initialization effect, while keeping its practical-relevance limitation explicit.

That was useful evidence, but it did not test your complete idea. **A corpus-informed initial table is not a live contextual encoder or a concept-level generator.** This was the first significant narrowing of your intent.

### 3. E1.5: separate geometry persistence, preservation, and contextual state construction

The subsequent controls asked different questions:

- **E1.5-D:** does the initial relational geometry survive training?
- **E1.5a:** is a frozen table sufficient?
- **E1.5b:** would preserving its Gram geometry help?
- **E1.5c:** should the backbone receive hidden states, probabilities, or square-root probabilities from a contextual predictor?

The historical findings discussed in this thread constrained the interpretation:

- Geometric retention meant a fading loss advantage was **not proof that Fisher geometry had been erased**.
- The freeze comparisons did not establish static-table sufficiency.
- The frozen contextual Fisher-state analysis did not support its headline substitution claim, although the square-root map improved on raw probabilities.
- The frozen-head audit identified a numerical loss mechanism associated with the probability floor, while ruling out the simplistic “128→256 necessarily loses rank” story.

The important correction was:

> **Numerical recoverability, downstream usability, and generative quality are different properties.**

Clamping explained a measured reconstruction problem. It did not establish that clamping explained the entire language-loss deficit, that every normalization is harmful, or that reconstruction error is a percentage of semantic information lost.

### 4. The D-pivot: keep state latent; derive geometry from future laws

The D-series disentangled representation from its predictive law:

- causal latent state;
- a separately defined future-target objective;
- stabilization and collapse controls;
- a Fisher pullback consumed by dynamics;
- later horizon, coarse-graining, and codec tests.

This was an important conceptual repair. A distribution could **supervise state and induce geometry without replacing state**.

But it remained an intermediate formulation. Its early stages still advanced on a byte sequence and used future-latent prediction alongside language modeling. That was closer to your intent, but **not yet the next-vector generative process you meant**.

### 5. September 14: you explicitly changed the operating unit

Your clarification made the destination unambiguous:

> Predict the next continuous chunk/vector in an ordered sequence, starting with sentences; later test paragraph and document-context recursion.

You also specified a practical destination: a small, usable text generator, roughly 500M parameters and at most 1B within a separately controlled expansion.

This was the second major correction to my understanding. **A latent auxiliary objective is not equivalent to making latent prediction the model’s principal generative path.**

The prior art helps locate the idea:

- **Large Concept Models** explicitly predict sentence representations and then decode them. Their limitations also demonstrate why embedding fidelity, distribution mismatch, and frozen-codec choices matter. [LCM, particularly §6](https://www.alphaxiv.org/abs/2412.08821v2)
- **NCP-ArchPreview** predicts concepts and feeds those predictions into generation, but uses fixed token grouping, learned codebooks, and joint token/concept objectives. It supports studying a *consumed predictive latent path*, not the claim that our continuous sentence/Fisher design is already validated. [NCP-ArchPreview, §§2–4](https://www.alphaxiv.org/abs/2609.10715v1)

### 6. The output-geometry discussion clarified “influence” versus “value”

Your “second-order derivative” analogy became more precise:

- how much does an internal change alter downstream predictions?
- does it improve the intended prediction?
- how much unrelated behavior does it disturb?

Those are different measurements.

Locally, Fisher geometry describes distributional change:

\[
\mathrm{KL}(p_h\|p_{h+\delta})
\approx \tfrac12\delta^\top G(h)\delta.
\]

But improvement of a chosen objective depends on its direction:

\[
\Delta \text{score}\approx \nabla_h\text{score}^{\top}\delta.
\]

**The first is not a substitute for the second.**

Picozzi’s work is relevant because it studies intervention cost at matched desired effects and measures collateral disturbance. Its minimum-disturbance construction requires an objective direction and a declared reference metric; it does not automatically prove that our learned forward residual transformation is beneficial. [Output-geometry paper, pp. 11, 15 and supplementary discussion](https://www.alphaxiv.org/abs/2609.11063v1)

### 7. September 15: the preservation-gated H-program became the binding design

You explicitly challenged the possibility that we were quietly replacing the spherical idea with an ordinary Euclidean model.

The resolution was **not to choose the winner in advance**. It was to test two independent axes:

- sentence representation: direction-only versus magnitude-preserving;
- spine carrier: spherical retraction versus magnitude-preserving.

The approved design also established:

- a learned sentence encoder/decoder trained from scratch;
- reconstruction warm-up, followed initially by a frozen encoder;
- a sentence-clock Transformer;
- a stochastic directional-plus-log-magnitude law;
- a local decoder conditioned on the predicted representation and its current sentence prefix;
- re-encoding the sentence actually emitted for the next operative context;
- retaining the originally sampled latent as a shadow diagnostic;
- no unrestricted raw-history bypass around the spine.

The directional distribution is a modeling choice, not a semantic theorem. Power Spherical supplies a tractable directional family; it does not establish that direction contains everything language generation needs. [Power Spherical](https://www.alphaxiv.org/abs/2006.04437)

### 8. September–October: the emphasis moved to trustworthy execution and measurement

The later conversations concentrated on running, observing, recovering, capturing, and interpreting that construction.

Two lessons matter conceptually:

- You corrected the tendency to make complete automation a prerequisite for an excluded development probe. Recording what an attempt actually used is different from sealing a confirmatory experiment.
- The October importance-sampling discussion distinguished a malfunctioning measurement path from poor prior–proposal overlap and from inadequate estimator precision.

That is a maturation of the **measurement apparatus**, not evidence that the central model hypothesis has passed. I am deliberately not turning those dated operational conversations into an October 26 readiness claim.

## 2. Where I have been conservative—and where that became too narrow

### Where I narrowed your intent too much

**First, I treated convenient experiments as if they were the destination.** E1 initialization and the D-series auxiliary future-state path were legitimate decompositions, but neither was the complete latent generator you intended.

**Second, I sometimes presented Euclidean dynamics as a recommended destination rather than an essential comparator.** Your actual position is: retain spherical restrictions where they earn their place, and expose their costs where they do not.

**Third, I leaned too far toward infrastructure completeness before obtaining decisive empirical contact.** The flywheel exists to produce interpretable experiments. Its value is not proportional to how much machinery we can build around it.

The correction is:

> **Conservatism should constrain our claims and experimental confounds—not shrink the architecture back into a different, easier research question.**

### Where I remain deliberately conservative

I would still refuse to infer that:

- a sentence is inherently one semantic atom;
- a higher-rank or more recoverable representation is automatically more useful;
- a strong intervention effect is automatically beneficial;
- a Fisher metric is automatically aligned with observable text quality;
- a codec-separated design is already codec-equivariant;
- a normalization restriction is harmful merely because it removes a coordinate;
- reaching 500M or 1B establishes a tuned-frontier result.

The nGPT literature is a useful reason to retain spherical candidates, but its architecture changes several coupled ingredients. It does not settle our independent sentence-normalization and carrier-normalization questions. [nGPT and its ablations](https://arxiv.org/html/2410.01131v2)

## 3. My current understanding of the intended model

The intended operative loop is:

\[
\text{completed text}
\rightarrow \text{sentence representation}
\rightarrow \text{latent sequence prediction}
\rightarrow \text{sampled next representation}
\rightarrow \text{generated sentence}
\rightarrow \text{re-encoded emitted sentence}.
\]

The main spine advances on **sentences**, while the local decoder realizes each prediction through bytes.

Three distinctions are essential:

1. **The representation is not its probability distribution.**
   A future law reads the state; the state is not replaced by an alphabet-sized simplex.

2. **The predicted representation is not the true future sentence’s encoding.**
   True-future encodings can be training targets or oracle reconstruction diagnostics. They cannot silently supply the answer to generation.

3. **Geometry describes consequences, not intentions.**
   It can shape an update, but the model’s objective and measured behavior must establish whether the update was useful.

The full hypothesis is a conjunction:

> **The codec preserves relevant distinctions; the sentence-clock predictor uses them to improve generation; and the claimed geometric mechanism adds meaningful value beyond that predictor.**

If hierarchy works but Fisher adds nothing, that is a useful, narrower result. It is **not full confirmation of the geometric hypothesis**.

## 4. The ablation ladder I would use

I would retain the approved **H0–H5 itinerary**. The following sharpens its questions; additional controls are recommendations to settle before registration, not retroactive changes.

| Stage | Decisive question | Minimum comparison or control |
|---|---|---|
| **H0 — Complete tracer** | Is this genuinely causal latent-mediated generation? | Free-running multi-sentence generation; future-target blindness; predicted versus oracle latents; recovery of stochastic and feedback state; one complete evidence lineage. |
| **H1 — Preservation** | Which restrictions preserve what generation needs? | Full **2×2 sentence normalization × spine normalization**, with isotropic dynamics and no auxiliary future-law objective. Separate fixed-state interventions from paired retraining. |
| **H2 — Predictive value** | Does hierarchy help, and does explicit next-latent supervision help beyond hierarchy? | Flat byte model; hierarchical model trained through observable decoder loss; same hierarchy with future-law supervision. Compare under declared data, context, capacity, and resource conditions. |
| **H3 — Predictive geometry** | Does the correct geometry help beyond the same predictive system? | Isotropic versus fully differentiated Fisher consumers within eligible constructions; predeclare controls for update magnitude/damping and metric–state correspondence where needed for attribution. |
| **H4 — Interface and scale transfer** | Does the conclusion survive recoding and increased scale? | Exactly coupled permutation correctness **and** separately paired recoded training; replicate the decisive contrasts at another measured scale. Do not change codec and scale together and attribute the result to one. |
| **H5 — Deployment-scale model** | Is the retained construction useful around 500M total deployed parameters? | Held-out predictive quality, fidelity, free-running behavior, and measured resource comparisons against a credible baseline. |

Mamba-2, encoder unfreezing, alternative density families, paragraph/document-context levels, and recursion remain **subsequent branches with their own claims**.

### H2 and H3 are the center

Using an admissible observable loss \(L\), the key effects are:

\[
\Delta_{\mathrm{hierarchy}}
= L_{\mathrm{flat}}-L_{\mathrm{hierarchical,\;LM}}
\]

\[
\Delta_{\mathrm{future}}
= L_{\mathrm{hierarchical,\;LM}}
-L_{\mathrm{hierarchical,\;future}}
\]

\[
\Delta_{\mathrm{geometry}}
= L_{\mathrm{predictive,\;isotropic}}
-L_{\mathrm{predictive,\;Fisher}}.
\]

These answer different questions. A positive final-system-versus-flat contrast alone does not identify which mechanism helped.

Nor should the ladder assume effects are additive. **If the theory predicts an interaction—such as future supervision becoming useful specifically when its geometry is consumed—we should register that interaction.** A weak standalone effect need not kill a predeclared conjunction; it also cannot become permission for unlimited post-hoc rescue experiments.

### Intervention and retraining are both necessary

Within a trained model, remove, replace, shuffle, or perturb the latent signal. This asks whether the system uses it.

Then train the corresponding controlled alternatives. This asks whether learning with that mechanism improves the outcome.

Those are not the same counterfactual. Zeroing a component can create an out-of-distribution shock, so large damage after ablation is not automatically evidence of unique value. Matched replacements and paired retraining are important.

### The geometry-to-text bridge must be tested

A future-latent-law Fisher metric could emphasize directions that the decoder barely uses.

Therefore H3 must ask:

> **Does the geometry of the internal law correspond to beneficial, selective changes in observable generation?**

Otherwise we could demonstrate elegant internal geometry without demonstrating the intended generative advantage.

## 5. What would make the effect material rather than merely present?

I would distinguish five evidence levels:

| Evidence | What it establishes |
|---|---|
| Gradients reach the component | It is connected. |
| Intervening on it changes output | It is used under that intervention. |
| Paired retraining improves held-out outcomes | It contributes predictive value under that comparison. |
| The effect clears a predeclared practical margin and preservation/resource guardrails | It is materially valuable. |
| The decisive result survives replication, fair tuning, and the declared transfer tests | It is robust within that scope. |

The practical margin must be defined **before confirmation**. The historical E-series `0.01 nats/byte` reference is not automatically the right margin for a new latent-variable likelihood estimate or semantic metric.

For a registered benefit \(d\) and materiality threshold \(\delta_{\mathrm{mat}}\):

- A simultaneous confidence interval entirely above \(\delta_{\mathrm{mat}}\) provides strong evidence of material benefit.
- An interval above zero but entirely below that threshold supports a detectable, practically small effect.
- An interval above zero but crossing the threshold leaves materiality unresolved.
- A wide interval crossing zero is inconclusive—not evidence of equivalence.

A resource-saving route is also legitimate: **noninferior quality with a material resource reduction**, if that route and its margins are preregistered.

One statistical detail matters: **90% power to reject zero at effect size \(\delta\) is not 90% power to establish that the effect exceeds \(\delta\).** The power calculation must match the actual deciding null, margin, and alternative.

## 6. Four protections against false confirmation

### A. Preservation must be tied to observable distinctions

Norms, spectra, reconstruction error, and rank are diagnostics. The meaningful tests include entities, numbers, negation, relations, order, and context-conditioned continuation.

A numerically invertible map can still be difficult for the downstream model to use. Conversely, retraining may compensate for a restriction that damages a frozen-state probe. LCM’s discussion of fragile sentence encodings makes this a real architectural concern, not just a numerical nicety. [LCM limitations](https://www.alphaxiv.org/abs/2412.08821v2)

### B. The likelihood must belong to the generator

The target is a marginal predictive likelihood:

\[
p(x_{k+1}\mid \text{history})
=
\int p_\omega(x_{k+1}\mid c)\,
q_\phi(dc\mid s_k).
\]

Scoring text conditioned on its **true future encoding** is a different quantity.

Importance-weighted scoring introduces another distinction: its lower-bound guarantee is in expectation, and its tightness depends on the proposal and sample count. A better estimate or tighter bound is not by itself proof of a better generator. The October ESS discussion makes estimator admission and sample-count sensitivity particularly important. [IWAE, §3](https://www.alphaxiv.org/abs/1509.00519v4)

### C. Score gain remains a diagnostic until promoted prospectively

The IGPO-inspired question is useful: does the update improve the score of the **same future target**?

But more observed context can help any model. We need matched before/after comparisons, appropriate horizon controls, probe quality, and corresponding control arms—not merely a positive gain. This is realized predictive scoring, not automatically mutual information or semantic information created. [IGPO, §3.2](https://www.alphaxiv.org/abs/2510.14967v2)

### D. Fair tuning and replication cannot be replaced by model size

We should preserve two separate lanes:

- **Shared-configuration mechanism comparisons:** what changed under a controlled configuration?
- **Equal-budget per-arm tuning:** what can each architecture attain under a fair search allowance?

The updated small-scale study reinforces that tuning sensitivity can obscure architectural comparisons, and extrapolation can magnify statistical error. More seeds do not repair an unfair tuning comparison, and a 500M model does not automatically establish a frontier result. [Lourie et al., v2, §§5–6](https://www.alphaxiv.org/abs/2608.11859v2)

The existing agreement remains appropriate: paired seed blocks, pilot-based variance, 95% familywise decisions, secondary 3σ reporting on the same sealed data, and no outcome-selected extensions. Codec warm-up variability must be represented across seed blocks; many documents or samples from one trained model are not independent model replications.

## My bottom line

**You are trying to establish a new working generative medium, then determine which geometric and hierarchical mechanisms make it genuinely better.**

My appropriate conservatism is about what the evidence licenses. My inappropriate conservatism was allowing narrow intermediate experiments—or infrastructure completeness—to substitute for your actual destination.

The scientific center I would carry into the next review is:

> **Preserve useful distinctions → predict the next latent state → prove that prediction drives generation → establish material incremental value → determine whether predictive geometry improves that value selectively and efficiently.**

H0 and H1 make that test interpretable. **H2 and H3 test its central value proposition.** H4 and H5 determine how far the result travels and whether it supports the intended small generator.

We should be willing to obtain a useful hierarchy without useful Fisher consumption, or a useful geometry only under one preservation regime—but name those narrower outcomes honestly. We should not declare the full hypothesis supported because an internal score moved, a branch was influential, or a run completed.

**That is the understanding I would now use to examine the codebase. Whether the machinery is actually ready to execute this contract—and whether anything has drifted from it—remains the next, separate pass.**
