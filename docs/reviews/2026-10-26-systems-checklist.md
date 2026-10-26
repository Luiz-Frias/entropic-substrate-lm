# Latent predictor: implementation and evidence checklist — 2026-10-26

## Finding in one paragraph

The implementation now contains the intended **continuous sentence-clock generator**,
not merely Fisher initialization or a byte-clock auxiliary latent objective. Its core
construction is aligned with the thesis. The programme is **not yet at “just run every
flywheel”**: H0 has a filed `tracer_partial` (October 22, HEpP7g; addendum, same day after this
audit: the October 26 final probe is filed `tracer_complete` in the stacked filing PR), H1 has
numerical and statistical declarations
but not an admitted pilot/confirmation executor, and later H stages still need their
own controlled experiment adapters and evidence. More infrastructure is not the immediate
goal; close the specific evidence and experimental seams below.

The [pre-audit understanding](2026-10-26-hypothesis-understanding.md) is preserved separately.
This checklist does not amend a preregistration, file a scientific verdict, authorize
spending, or confer successor eligibility.

## Snapshot, method, and limits

- Source snapshot: `d6935e2830f8502f9e86151de90b9438f01b6bc7`, the tip of
  `fix/h-probe-resumed-progress` when the documentation branch was created.
- Branch: `docs/2026-10-26-hypothesis-systems-review`, explicitly Graphite-tracked on
  that parent. The main workspace initially had an older detached HEAD; the review
  uses the requested current stack tip, not that older checkout.
- Scope: contract-surface audit across all five source layers, with direct numerical,
  causal, execution, custody, and statistical-path reads. This is not a claim that
  every line of every historical module received a line-by-line bug audit.
- Inventory: `rg --files src -g '*.py'` returned 392 Python files;
  `git ls-files 'src/**/*.py' | xargs wc -l` returned 95,307 physical lines, including
  comments and blanks. These are source inventory measurements, not evidence of value.
- Used `tree -L 4 -I '__pycache__' src/entropic_substrate`, `rscope 0.2.1` map/calls/
  callers/blast-radius, direct `nl -ba` reads, and targeted `rg` consumer searches.
  Source and test anchors below refer to the pinned source snapshot.
- Graph output is navigational, not authoritative. `rscope map` reported zero functions
  while listing functions; reverse lookup omitted imported callers. Direct source
  resolves the actual `SentenceExecution` → `importance_evaluate` call. Dynamic module
  invocation, injected callbacks, and imported functions require manual verification.
- Delegated reviews were read-only: sentence mathematics/generation; execution/evidence;
  historical mechanisms/statistics. Findings were reconciled against source reads.
- No new GPU run, provider termination, DVC download, or scientific analysis was performed.
  Existing repository verdicts are cited as filed evidence, not independently rerun outcomes.

Reproducible navigation examples (not complete call graphs):

```sh
rscope map src/entropic_substrate --depth 2 --budget 6500
rscope --root src/entropic_substrate calls --fn transition \
  --file kernel/lifecycle.py --depth 2 --budget 2800
rscope --root src/entropic_substrate callers --fn transition \
  --file kernel/lifecycle.py --depth 2 --tree --budget 2200
rscope --root src/entropic_substrate/core/sentence blast-radius \
  --fn predict_latents --file model.py --depth 2 --budget 2200
```

Status vocabulary: **implemented** means a concrete producer/consumer exists;
**partial** means a named contract or evidence link is missing; **unvalidated** means
implementation is not the required scientific result; **deferred** means intentionally
outside the current stage. A checked box below does not mean scientific confirmation.

## Topology and ownership

```text
composition/                    CLI dispatch and wiring
  entrypoints/                  operator commands; H analysis handlers
  wiring/                       retained D adapters; local H tracer
historical/                     E training and D-ladder execution adapters
ladder/                         register → run → capture → analyze → review → file → gate
  run/                          actual H training, generation, probes and remote H0 path
core/                           corpus, observation, latent mechanics, probes, sentence model
kernel/                         identities, immutable artifacts, lifecycle, runtime contracts
```

The actual dependency rule is `composition → historical → ladder → core → kernel`,
with historical optional to a given active path, not a mandatory processing stage.
[Import contracts](../../.importlinter#L41) enforce the layers and forbid core imports
of external storage/provider packages. Historical architecture diagrams naming old
`predictive_state/` or `experiments/` folders are provenance, not the current file map.

Two important flows now exist:

```text
prepared documents → completed sentence encoder → sentence-clock spine → future law
  → sampled continuous latent → local byte decoder → emitted text → re-encoded context

H0 request → worker → training/seals → generation → frozen span probes → trace capture
  → native verification → external collection → H analysis / independent review
```

The second flow's engineering result explicitly carries `scientific_authority: none`.
It must not be connected to a scientific successor by relabeling a successful probe.

## A. Representation, dynamics, and generation

| Contract | Implementation and test evidence | Status / remaining obligation |
|---|---|---|
| Continuous state is distinct from its law | [SentencePredictor](../../src/entropic_substrate/core/sentence/model.py#L40), [predict_latents](../../src/entropic_substrate/core/sentence/model.py#L96); [composition tests](../../tests/test_sentence_model.py#L40) | **Implemented.** No alphabet-sized probability carrier in this H path. This is the substantive advance beyond E1/E1.5c. |
| Independent sentence and spine normalization | [sentence interface](../../src/entropic_substrate/core/sentence/model.py#L87), [carrier residual](../../src/entropic_substrate/core/sentence/spine.py#L73), [no final overwrite](../../src/entropic_substrate/core/sentence/spine.py#L161); [H1 block](../../src/entropic_substrate/ladder/register/h_preservation.py#L29) | **Implemented numerically; unvalidated experimentally.** Both axes exist; the old H0 diagonal comparison is not the full factorial. |
| Learned sentence codec; warm-up then freeze | [encoder and decoder](../../src/entropic_substrate/core/sentence/codec.py#L141), [freeze](../../src/entropic_substrate/core/sentence/codec.py#L234), [warm-up loss](../../src/entropic_substrate/core/sentence/objective.py#L46) | **Implemented.** H1 still needs artifact-sharing execution within each paired block, not independent warm-ups masquerading as sharing. |
| Local decoder has no unrestricted history bypass | [model forward](../../src/entropic_substrate/core/sentence/model.py#L108), [decoder signature](../../src/entropic_substrate/core/sentence/codec.py#L186); [boundary guard](../../tests/test_sentence_model.py#L95) | **Implemented structural restriction.** Actual utility still requires signal interventions and paired retraining. |
| Language-only hierarchy trains the predictor | [prediction_loss](../../src/entropic_substrate/core/sentence/objective.py#L67); [decoder-loss gradient test](../../tests/test_sentence_model.py#L79) | **Implemented building block for H2.** `future_weight=0` skips future-target exposure, not the trainable prior/head. |
| Directional plus log-magnitude law | [law](../../src/entropic_substrate/core/sentence/law.py#L92), [active head parameters](../../src/entropic_substrate/core/sentence/head.py#L26), [Fisher weights](../../src/entropic_substrate/core/sentence/head.py#L67) | **Implemented.** Spherical sentence mode omits the magnitude head; fixed dispersion and one-vector factorization remain hypotheses, not universal semantic laws. |
| Fisher acts on updates at one site | [final-block intervention](../../src/entropic_substrate/core/sentence/spine.py#L139), [radial/tangent consumer](../../src/entropic_substrate/core/sentence/consumer.py#L50) | **Implemented.** Isotropic substitution returns the original proposal; Fisher weights belong to the active law, not an unweighted transplanted Gaussian metric. |
| Full differentiation, including zero RHS | [unrolled solve](../../src/entropic_substrate/core/latent/dynamics.py#L124), [zero inverse derivative](../../src/entropic_substrate/core/latent/dynamics.py#L49), [JVP/VJP action](../../src/entropic_substrate/core/latent/metric.py#L44); [dense derivative oracle](../../tests/test_sentence_consumer.py#L20), [detachment mutation](../../tests/test_sentence_consumer.py#L115) | **Implemented and locally tested.** Numerical validity is not proof of useful geometry or acceptable deployment cost. |
| Target-blind prior and causal macro history | [predict](../../src/entropic_substrate/core/sentence/model.py#L91), [causal attention](../../src/entropic_substrate/core/sentence/spine.py#L91); [future mutation](../../tests/test_sentence_model.py#L57), [target-proposal isolation](../../tests/test_sentence_scoring.py#L63) | **Implemented.** Offline declared sentence boundaries remain part of the task; this is not arbitrary partial-sentence streaming. |
| Free-running feedback uses emitted text | [advance_rollout](../../src/entropic_substrate/core/sentence/rollout.py#L72), [operative/shadow separation](../../src/entropic_substrate/core/sentence/rollout.py#L117); [shadow mutation test](../../tests/test_sentence_rollout.py#L53) | **Implemented.** One sampled latent persists across a sentence; emitted bytes are inverted, UTF-8 checked, and re-encoded. |
| Invalid generation stays an outcome | [rollout statuses](../../src/entropic_substrate/core/sentence/rollout.py#L12), [terminal handling](../../src/entropic_substrate/core/sentence/rollout.py#L102); [negative tests](../../tests/test_sentence_rollout.py#L86) | **Implemented.** `generation_complete` denotes completed execution, not coherent or even valid text. Inspect outcome, finished sentences, and retained pending bytes. |

**Interpretation update:** the state-construction correction is no longer just a plan.
The architecture really predicts absolute next-sentence latents and consumes them during
generation. The old D horizon displacement objective is a separate retained mechanism;
it is not the H model's main generation target.

## B. Data, objectives, and measurements

- [x] Bounded document preparation, source identity, deduplication, splits, and whole-document
  rejection exist in [prepare_documents](../../src/entropic_substrate/core/sentence/corpus.py#L173).
  [Segmentation](../../src/entropic_substrate/core/sentence/segmentation.py#L84) pins pySBD,
  preserves original whitespace/byte coverage, and rejects malformed spans. Tests cover
  [duplicate content](../../tests/test_sentence_corpus.py#L35) and
  [causal document-local sampling](../../tests/test_sentence_sampling.py#L39).
- [x] Codec framing is explicit: [SentenceByteCodec](../../src/entropic_substrate/core/sentence/boundary.py#L10)
  has 256 bytes plus two output controls; BOS/PAD are input-only. The byte permutation
  leaves framing unchanged. This is codec separation, not learned codec equivariance.
- [x] Training loss explicitly distinguishes reconstruction from prior-sampled decoder
  training and counts source, boundary, history, and encoder exposure separately:
  [SentenceLoss](../../src/entropic_substrate/core/sentence/objective.py#L17).
- [x] Generator scoring uses `decoder + prior - proposal` log weights, not oracle
  reconstruction: [importance_evaluate](../../src/entropic_substrate/core/sentence/scoring.py#L79).
  [Statistics](../../src/entropic_substrate/core/sentence/scoring.py#L22) retain ESS, relative
  likelihood MCSE, and typed zero-mass gaps; [importance-correction test](../../tests/test_sentence_scoring.py#L82).
- [x] Current [H profile v3](../../src/entropic_substrate/core/sentence/profile.py#L229)
  binds the development importance-sample curve, diagnostic count, and continuation count.
  It refuses old profile schemas instead of inventing omitted choices. Current defaults
  must not be attributed retrospectively to old H0 captures.
- [x] Span-aware score gain is a distinct schema with same target, checkpoint, probe,
  codec, document/split, and explicit horizon checks:
  [SpanScoreGain](../../src/entropic_substrate/core/probe/score_gain.py#L47).
  [State collection](../../src/entropic_substrate/ladder/run/span_states.py#L63) uses causal
  states and canonical labels; [fitting](../../src/entropic_substrate/ladder/run/span_probe.py#L221)
  accepts detached training states; [scoring](../../src/entropic_substrate/ladder/run/span_probe.py#L288)
  rejects split/artifact mismatches and includes uniform and horizon-matched shuffled controls.
- [x] Probe weights/positions and replay are captured, with collection and fit/score resource
  accounting separate from model training:
  [diagnostic orchestration](../../src/entropic_substrate/ladder/run/span_diagnostics.py#L251).
  Tests cover [identity mismatches](../../tests/test_span_score_gain.py#L44),
  [shuffling](../../tests/test_span_probe.py#L61), and
  [actual checkpoint lineage](../../tests/test_span_diagnostics.py#L41).
- [ ] Detail preservation, frozen NLI evaluator, fixed-rubric human audit, and matched
  perturbation/off-target measurements are **not implemented in the inspected H path**.
  Whole-word source searches plus direct evaluator reads found no such instruments.
  The [H1 skeleton](../h1/preregistration-skeleton.md#L93) independently lists E-DETAIL
  and E-PERT as missing. Norms, rank, and score-gain probes do not replace them.
- [ ] Scientific admission thresholds for ESS/sample-count sensitivity, fidelity,
  material effects, and eligible constructions remain to be sealed. An estimator
  executing correctly does not make its precision adequate for a deciding comparison.

**Interpretation update:** H losses are boundary-augmented latent-variable quantities.
Do not compare their training CE or oracle reconstruction to the old E-series nats/byte
as if they estimated the same generator likelihood. The flat H2 comparator needs the
same boundary protocol and declared exposure/resource accounting.

## C. Transactions, custody, and execution

| Contract | Source/test anchors | Status / boundary |
|---|---|---|
| Missing/nonfinite/wrong-owner versus zero/absent gradients | [apply_sentence_step](../../src/entropic_substrate/ladder/run/training.py#L106), [declared inactive roles](../../src/entropic_substrate/ladder/run/training.py#L36); [zero transaction](../../tests/test_sentence_training.py#L58), [disconnected graph](../../tests/test_sentence_training.py#L77) | **Implemented.** Zero still runs AdamW; missing participation is not called collapse. Warm-up/frozen roles have explicit reasons. |
| Zero-vector geometry versus execution defects | [UndefinedDirection](../../src/entropic_substrate/core/sentence/numerics.py#L12), [magnitude interface](../../src/entropic_substrate/core/sentence/numerics.py#L76); [diagnostic classification test](../../tests/test_span_diagnostics.py#L158) | **Typed distinction implemented.** A zero vector does not acquire a fabricated spherical direction. H scientific clauses must still decide collapse/grounding outcomes; diagnostic gaps cannot supply the absent scientific adapter. |
| Exact recovery state and version binding | [checkpoint reader identity](../../src/entropic_substrate/ladder/run/checkpoint.py#L120), [construction/optimizer binding](../../src/entropic_substrate/ladder/run/checkpoint.py#L134); [next transaction replay](../../tests/test_sentence_checkpoint.py#L56), [pending rollout replay](../../tests/test_sentence_checkpoint.py#L91) | **Implemented locally.** Model, optimizer, cursor, RNG, operative and shadow state are not interchangeable with a weights-only checkpoint. Remote same-lineage recovery remains an H0 evidence obligation. |
| Fail-loud artifact custody | [no-follow bounded read](../../src/entropic_substrate/kernel/artifacts.py#L182), [immutable atomic publication](../../src/entropic_substrate/kernel/artifacts.py#L251), [bound re-read](../../src/entropic_substrate/kernel/artifacts.py#L322) | **Implemented shared substrate.** A content-addressed file is not by itself a scientific registration. |
| Single H engineering lineage | [worker sequence](../../src/entropic_substrate/ladder/run/worker_execution.py#L182), [capture](../../src/entropic_substrate/ladder/run/worker_execution.py#L259), [native verification](../../src/entropic_substrate/ladder/run/worker_execution.py#L281); [end-to-end fixture](../../tests/test_sentence_worker_execution.py#L53) | **Implemented.** Training, generation, diagnostics, capture, and verification use the same profile/source/checkpoint lineage. Fixture success is not remote completion. |
| Remote tiny versus science authority | [ProbeRequest rejection](../../src/entropic_substrate/ladder/run/worker_request.py#L60), [H0 launch restriction](../../src/entropic_substrate/ladder/run/h_probe_launch.py#L176); [admission test](../../tests/test_sentence_worker_request.py#L75) | **Intentionally restricted.** No production H1 pilot/confirmatory path is admitted by these classes. Do not remove the rejection without the corresponding scientific adapter. |
| Measured resources versus gaps | [execution CUDA adapters](../../src/entropic_substrate/ladder/run/execution.py#L99), [shared meter](../../src/entropic_substrate/ladder/capture/resource/resource_meter_strategy.py#L93), [D worker resource declaration](../../src/entropic_substrate/ladder/run/worker/resource_registration.py#L34) | **Partial for the final compute claim.** CUDA time/memory and exposures have producers; FLOPs/power are not established by those counters. Retain gaps and distinguish estimates/billing from device measurements. |
| Fresh analysis and review authority | [pure lifecycle checks](../../src/entropic_substrate/kernel/lifecycle.py#L396), [filing](../../src/entropic_substrate/kernel/lifecycle.py#L421), [successor checks](../../src/entropic_substrate/kernel/lifecycle.py#L448) | **Reusable contracts exist.** They do not authenticate an H1–H5 run absent H-specific producer/admission/filing integration. |

## D. Where the experimental ladder actually stands

| Stage | Implemented or filed now | What still prevents its scientific exit |
|---|---|---|
| H0 complete tracer | Real generator, worker, custody/readers, H aggregation/charts, final-probe checks, and [filed October 22 verdict](../h0/verdict-hepp7g-2026-10-22.md#L3). | Verdict is **tracer_partial**, not a hypothesis win (addendum, same day after this audit: the October 26 final probe is filed `tracer_complete` in the stacked filing PR, closing the exits listed next). Close same-lineage stochastic rollout, recovery/resume, test-only gate, and remaining observation/schema exits with new admissible evidence. |
| H1 preservation | [Four-cell block](../../src/entropic_substrate/ladder/register/h_preservation.py#L29), factorial coefficients, codec-sharing declaration, strict profile exclusions; [draft skeleton](../h1/preregistration-skeleton.md#L12). | Bind actual shared codec artifacts; add missing detail/intervention instruments; admit H observations and pilot captures; seal margins/endpoints/seed matrix; execute/review/file. |
| H2 predictive value | Prior-sampled language-only and supervised objectives are connected; profile forbids Fisher. | No complete boundary-matched flat-versus-hierarchical campaign/registration/analysis adapter was found. Retained E flat training does not by itself supply that matched control. |
| H3 geometry | Isotropic/Fisher consumers, dense action and derivative oracles, per-solve diagnostics. | Registered paired experiment in each eligible construction, attribution controls and actual geometry-to-text usefulness/off-target evidence; measured resources. |
| H4 interface and scale | H reversible byte framing and canonical labels; historical D codec correctness/equivalence components. | H-specific coupled model test and separately trained recoding/scale comparisons with admitted evidence. Do not promote a D test or reversible map into learned H equivariance. |
| H5 deployment | Configurable Transformer/codec; [unique-parameter inventory test](../../tests/test_sentence_model.py#L140). | Inventoried and resource-admitted ~500M profile, credible comparator, data/tuning budget, registered generation/fidelity/resource criteria, and filed evidence. Parameter target alone establishes none of these. |

### The narrow H1 adapter that is actually missing

1. `H1PreservationBlock.codec_sharing_groups` declares the two required pairings
   ([line 80](../../src/entropic_substrate/ladder/register/h_preservation.py#L80)).
   Searches found only the declaration and tests, not an executor binding shared codec bytes.
2. `read_h_statistical_design` exists, but
   [paired_contrasts](../../src/entropic_substrate/ladder/analyze/ladder_registration.py#L271)
   explicitly rejects H observations pending their source-order adapter.
3. [prepare_registration](../../src/entropic_substrate/ladder/analyze/ladder_registration.py#L371)
   accepts finalized FULL D1 pilot captures, not H pilots. The reusable
   [seed planner](../../src/entropic_substrate/ladder/analyze/seed_sizing.py#L184) already
   supports declared bounds, conservative pilot variance and conditional power.
   New H evidence admission is needed; invented H pilot variance is not a substitute.
4. Existing [campaign admission](../../src/entropic_substrate/ladder/gate/ladder_campaign.py#L192)
   and [outcome filing](../../src/entropic_substrate/ladder/file/ladder_filing.py#L96)
   are D-specific. H stage strings in `HProfile` are not complete scientific designs.

Recommendation: implement this as one explicit H preservation adapter using existing
custody/recovery/statistical services. Do not weaken D validators, grant scientific
authority to tiny receipts, or create a second generic orchestration framework.

### H0 check utilities are not a complete successor gate

The [final-probe script](../../scripts/h0_final_probe_checks.py#L148) deliberately runs
any nonempty selected subset of checks. [K2](../../src/entropic_substrate/ladder/analyze/h_final_probe_checks.py#L318)
checks every supplied request and distinct carriers, but does not require exactly both
carriers: a singleton roster is not rejected by the roster condition. This is a
source-verified composition gap, **not a demonstrated scientific-authority escalation**.
An aggregate H0 exit must bind the complete required check set, both carriers, and one
shared lineage before calling the tracer complete. Keep individual check passes scoped.

The archived HEpP7g tests explicitly retain old-evidence limits:
[K4 fails on its older read behavior](../../tests/test_h_final_probe_checks_hepp7g.py#L66),
and [current K settings are rejected on the old capture](../../tests/test_h_final_probe_checks_hepp7g.py#L94).
New code cannot retroactively supply an old run's missing observations.

## E. Historical machinery to retain, not relabel

- E init/block/kernel implementations remain under `core/substrate` with execution in
  `historical/e_series`. E1 initialization and E3 curvature results keep their original
  semantics; they are not H generation or H Fisher-consumer evidence.
- D2 L/S/G composition lives in
  [LadderPredictiveSystem](../../src/entropic_substrate/ladder/run/construction/ladder_runtime.py#L62).
  D3 [aligned horizon displacements](../../src/entropic_substrate/core/latent/horizons.py#L60)
  and composite heads are retained byte-clock mechanisms, not an implemented H recursion.
- D4 [causal coarse-graining](../../src/entropic_substrate/core/latent/coarse_graining.py#L73)
  includes fixed/learned controls. Its dense candidate evaluation does not establish
  sparse execution or compute savings. H begins with declared sentences instead.
- D5 permutation and analysis components are reusable correctness tools. Separate
  paired training remains necessary to claim learned equivariance.
- Paragraph/document recursion, learned H boundaries, multiple latent slots, Mamba-2,
  encoder unfreezing, richer density families, and 1B expansion are **deferred**, not
  accidental omissions from the current sentence-first tracer.

## F. Prioritized systems checklist

- [x] Preserve the pre-audit thesis unchanged and bind this audit to an exact source tip.
- [x] Locate the actual sentence generative path and numerical controls.
- [x] Verify that training losses, IS estimates, score gains, and scientific replicates
  have distinct meanings; no result here establishes material predictive value.
- [ ] **First: close H0 evidence**, using the currently intended construction, sampling,
  recovery and observation schema. File the result against the actual capture; do not
  substitute this review or a local fixture for the independent scientific lifecycle.
- [ ] **Then: make H1 executable as its declared factorial**, including actual codec
  sharing, observation admission, pilot capture and seed planning, independent review,
  filing and successor eligibility. Keep its missing scientific choices operator-owned.
- [ ] **Before H1 confirmation: implement preservation instruments**, or explicitly
  revise the planned exit before registration. Numerical rank is not detail preservation.
- [ ] **Before H2: build the boundary-matched flat control and separate effects** of
  hierarchy and future-law supervision. Register any theory-dependent interaction.
- [ ] **Before H3: connect geometry to signed observable benefit and disturbance**,
  not merely sensitivity, solve correctness, or component influence.
- [ ] **Before H4/H5: measure transfer and resource envelopes** with newly powered
  regimes, separate shared-configuration and fair-tuning lanes, and explicit quotes.
- [ ] Reconcile remaining stale entry-point prose: `CLAUDE.md`'s October 14 “no H capture,
  analysis, review or verdict” statement is contradicted by the October 22 filed record.
  This docs change corrects the [architecture status](../architecture.md), but leaves
  repository instruction-file changes to their own review. Do not rewrite historical
  verdicts or turn `tracer_partial` into completion.

## Validation ledger

Three bounded read-only review suites completed before the interrupted documentation
commit. Counts are separate, overlapping selections, **not a summed unique-test total**:

| Selection | Observed result | Interpretation |
|---|---|---|
| Sentence model, spine, consumer, head, law, numerics, objective, rollout, scoring, training, recovery-state tests (`pytest -q -n0`) | 181 passed, 19 warnings; exit 0, delegate execution | CPU contract/numerical evidence, not CUDA or scientific confirmation. |
| `test_h_final_probe_checks_hepp7g.py`, `test_h_statistical_design.py`, `test_h_preservation.py`, `test_sentence_worker_request.py` | 59 passed, 1 warning; exit 0, delegate execution | Includes captured old-H0 checks with expected negative assertions. |
| H preservation/design, seed sizing, predictive horizons/coarse-graining, codec equivariance/permutation, span score-gain/probe tests | 159 passed, 19 warnings; exit 0, delegate execution | Reusable numerical/statistical controls; no new H pilot or successor authority. |

Warnings reported pynvml and torch.jit.script deprecations. Host contention was present;
no wall-time performance comparison is inferred. The original commit hook was interrupted
before a result was retained; that attempt is not reported as a passed repository gate.

Repository checks executed during this audit:

- Archival commit `6467801` passed the normal pre-commit hooks, including the repository
  unit-test hook, secrets checks, commitizen and gitlint. Its change is 372 added lines.
- `just fmt-check lint typecheck`: 992 files already formatted; Ruff passed;
  basedpyright reported zero errors, warnings, or notes. No source edits resulted.
- `just lint-imports`: all three contracts kept, zero broken. Its dependency-graph
  domain included 453 files and 3,846 edges; that is not the Git Python-file census above.
- All 97 original checklist local file/line anchors were checked against filesystem
  contents; newly added anchors are checked again before the checklist commit.
- Live-container and new CUDA/Runpod validation were not run. Publication is on hold
  by the operator's latest instruction; no PR or remote branch was created by this task.

Supplementary CodeRabbit review: installed CLI `0.7.6` was checked against the official
release `VERSION` and manifest; the selected diff passed `gitleaks stdin --redact`.
The following command returned `review_completed`, zero findings, and a roster of all
19 `core/sentence` Python files. This is a scoped secondary review, not a clean bill
of health for the entire repository and not an independent scientific review:

```sh
coderabbit review --agent --committed \
  --base-commit bf5abe7fa7a71d94e7de44da2368311f57059eea \
  --dir src/entropic_substrate/core/sentence
```

## Conclusion

The hypothesis has not drifted back into “a better embedding initialization.” The source
now embodies a learned codec, continuous next-sentence prediction, and law-induced
dynamics. What remains is not proof that the code can form a vector: it is the controlled
demonstration that preservation, hierarchy, supervision, and geometry each earn their place.

H0/H1 make that demonstration interpretable. H2/H3 are still the scientific center.
H4/H5 determine transfer and practical usefulness. A useful hierarchy without useful
Fisher consumption remains a legitimate narrower outcome, not full confirmation.
