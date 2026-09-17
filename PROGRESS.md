# Programme Progress

Last updated: 2026-09-15 during preservation-gated implementation.

## Approved H-program implementation

The starting checkout is `feature/codec-separated-sentence-predictor` at `6ecf0cb`, clean.
The approved H0–H5 contract now makes sentence and spine normalization independent axes,
with magnitude-preserving controls and no probability-floor state substitution. Local
implementation is authorized. On September 15 the operator additionally authorized
6,000-step Runpod tiny engineering probes with termination afterward. Publication,
excluded scientific pilots, and confirmation retain separate approval boundaries.
The baseline `just check` passed: 3,635 unit tests, 21 deselected, strict types and lint clean.
This is engineering evidence only. No H-model, remote probe, or H scientific verdict exists yet.

Step 1a now implements immutable UTF-8 source documents, exact byte spans, and a reversible
sentence/document envelope with byte-only permutations and separate BOS/PAD input symbols.
RED-before-GREEN: the new suite first failed because the namespace did not exist. Final
focused result: 36 tests pass; removing the terminal and interior document-end guards each
produced the expected failing malformed-envelope test, then both guards were restored.
The full gate passed with 3,670 tests before the final review-added case. CodeRabbit's two
findings described that same interior-guard coverage gap; it was verified and addressed.
Segmentation, source capture, and the model remain pending; this is not H0 completion.

Step 1b adds pySBD 0.3.4 and exact-coverage segmentation. Leading/trailing whitespace is
accounted for; missing non-whitespace, overlaps, changed text, and engine-version mismatches
fail loudly. An oversize sentence produces a typed whole-document rejection, not truncation.
The suite was RED before the adapter existed; malformed engine objects received their own
RED-before-GREEN regression. Disabling the non-whitespace-gap guard made its test fail.
All 56 boundary/segmentation tests pass with strict types, lint, dependency and security checks.
Source selection/capture and the model remain pending.

Step 1c adds bounded document selection, exact-content deduplication, content-stable
train/validation/test splits, and a new FineWeb document iterator at the existing revision.
Provider text is not coerced or joined with an EOT string. Conflicting source IDs fail;
duplicates, oversize sentences, and byte-budget stops retain their candidate evidence.
RED-first preparation/adapter tests and a source-ID-based split mutation were observed failing;
after restoring content-based assignment, 92 focused and legacy-data tests pass. Strict typing,
lint, and scoped security checks pass. Persisted capture and the model remain pending.

Step 1d adds an atomic, externally hash-bound source capture. Its versioned reader binds the
preparation source and dependency versions, replays all candidates, and compares the derived
split/span/rejection summary. Unknown versions, absent policy fields, duplicate JSON keys,
symlinks, occupied outputs, and forged summaries are refused. The new suite was RED before
implementation; disabling the summary replay comparison made the rehashed-forgery test fail.
All 104 focused/source-regression tests pass after restoration. Full package gates follow.
These are prepared-source artifacts, not scientific capture or successor authority.

Deferred snapshots are protected by local branches (not restored or published):
`codex/deferred-d-host-tail-20260914` and `codex/deferred-d-receipt-20260914`.

Data-package gates passed: `just check` (3,721 unit tests), `just test-all`, and `just audit`.
Live-service tests skipped without a running stack; this is not live-container validation.
CodeRabbit returned nine findings describing six distinct suggestions. Test annotations were
corrected; three extra blank/invalid-provider-row cases verify existing SourceDocument
validation (107 focused/regression tests pass). The shared-loader refactor is deferred to
keep historical loading isolated; both paths already use the same pinned revision. The full
Python patch-version reader pin is intentional until compatibility is validated. No scientific
or remote gate has advanced. The learned codec/model is the next implementation package.

The follow-up capture review's duplicate finding requested direct coverage of remaining
rejection branches. Added tests cover non-SHA hashes, reader byte limits, nonfinite JSON,
forged candidate identities, unexamined suffixes, and boolean/integer summary confusion.
All 20 capture tests pass; disabling the suffix guard admitted a forged suffix and made its
test fail, then the guard was restored. Production behavior is unchanged by this test slice.

Step 2a adds stable directional/log-magnitude views without positive-value floors. The FP32
magnitude carrier is returned unchanged; exact zero directions carry an explicit gap/mask,
not an invented spherical point. FP64 functional views support vmap and first/second derivative
checks. RED-first tests and deliberate FP32-promotion/hidden-normalization mutations failed
as intended. After restoration, all 31 numeric/solver tests and strict type/lint/security
checks pass. The learned law and neural codec remain pending.

Step 2b implements the Power Spherical/log-magnitude law, including surface versus Cartesian
base measures, the -d*rho Jacobian, inactive-radius rejection, RNG replay, and reparameterized
sampling. Independent Beta-marginal and score-moment oracles validate normalization/Fisher
scaling; first/second derivative checks cover densities and fixed-noise sampling. Four new
boundary regressions were RED before correction. Removing the Jacobian changed the expected
log-density difference from -2.5 to -0.5 and failed its test. Restored result: 47 law/numeric
tests pass with strict types and lint. Neural encoder/decoder and the spine remain pending.

Law review follow-up: two distinct suggestions (three findings) identified numeric identity
canonicalization and radius-tensor branch coverage; both are addressed. A separate synthetic
reproducer showed log-magnitude underflow returning zero samples. The new RED-first regression
now requires a numerical sampling defect, including component underflow, with no floor or
resampling fallback. Equal scales 1 and 1.0 share an identity. All 51 law/numeric tests pass.

Step 2c adds the learned bidirectional sentence encoder and local causal byte decoder, with
raw pooled vectors and independently initialized layers. Encoder freezing survives train()
and state reload. Norm views own FP32 islands; the decoder has no raw-history bypass.
RED-first tests caught cloned initializations, lost freeze state, and inherited normalization
precision before correction. Removing latent conditioning failed its influence test. All 75
codec/law/numeric tests pass. A 60-step synthetic reconstruction smoke reduced measured CE
from 5.9338164 to 0.0090350; this is reconstruction, not next-sentence prediction evidence.
The sentence-clock spine, generation loop, and full H0 lineage are still pending.

Codec review requested three additional boundary-test groups (six duplicate findings): invalid
freeze payloads, independent encoder/decoder divisibility, and partial radial underflow.
They are covered; all 82 codec/law/numeric tests pass. The first codec commit attempt passed
code gates but gitlint rejected an 84-character message line; the message was shortened, with
no hook changes or bypass.

Step 3a adds the sentence-clock spine with a zero-byte BOS state and independently selected
spherical/magnitude carriers. Sublayers and initializations are shared across carrier modes;
only the final block accepts a consumer intervention. Context, dtype, shape, and unit-carrier
contracts are validated. The new tests were RED before implementation; an injected final
normalization broke exact raw-carrier preservation as intended. All 98 spine/codec/law/numeric
tests pass with strict type/lint/security checks. Fisher integration and rollout remain pending.

The long codec follow-up review completed with seven findings covering four suggestions.
Test helper annotations and reverse freeze-state restoration are now covered. The independent
initialization comment is retained: it describes code behavior already measured by its paired
layer test, not an unmeasured performance claim. All 98 neural/numeric tests remain green.

Step 3b connects a trainable future-law head with no dormant magnitude parameters in the
spherical interface. The functional feature map includes the directional Fisher factor and
active Gaussian log-radius precision. Twelve RED-first tests now pass; removing the
directional factor made both independent weighted-metric oracles fail, then it was restored.
The head reads rather than replaces raw carriers. Consumer integration and rollout follow.

Step 3c adds the law-correct final-site consumer, reusing the matrix-free differentiated
solver with explicit functional parameters. Its spherical branch transforms tangent motion
and preserves the proposal's radial component. Dense value/derivative oracles cover the 2x2
interface/carrier matrix; spherical derivatives are tested through the actual unit retraction.
Detached-metric mutations lose parameter influence. Zero-residual mixed derivatives remain
live, failed solves clear previous diagnostics, and no isotropic fallback exists.

Runpod preflight measured balance USD 22.3809008535 and zero current hourly spend; no pods
existed. Runpodctl was updated and verified as 2.14.0-dd55bcf. This build and the v2 API lack
the scheduled-termination field assumed by the old adapter. The operator explicitly approved
a separate host watchdog plus controller deadline for tiny probes only; host connectivity
loss remains a disclosed limitation. No scientific-run exception or GPU allocation is implied.

Step 3d composes completed-sentence encoding, sentence-clock prediction, prior sampling,
and local byte readout. Thirteen causal/model tests pass, including observable-loss gradient
reachability with auxiliary supervision absent and the encoder frozen. Prefix changes cannot
alter prior states or the sampled latent. An injected spherical overwrite fails both
magnitude-interface construction tests, then restoration returns them to green. Ended
documents cannot be silently treated as continuation prompts. Rollout/recovery remain pending.

Step 3e adds bytewise rollout with canonical emitted bytes, encoded pending prefixes, sampled
latents, operative context, archived shadow predictions, and exact token/score traces. Fifteen
tests pass, including all four preservation combinations with Fisher under bf16 autocast,
mid-sentence RNG replay, invalid UTF-8/empty/length/context outcomes, and inverse-codec feedback.
Replacing emitted-text feedback with the imagined latent fails its reconstruction guard.
Disk checkpoint/RNG serialization and actual trained free-running generation remain pending;
scripted boundary fixtures establish wiring, not generation quality.

Step 4a separates original-sentence stochastic reconstruction from prior-sampled predictor
training. The optional future loss scores detached absolute target samples per latent
dimension; disabling it skips target-encoder exposure while preserving language gradients.
Boundary-inclusive decoder CE, target bytes, history bytes, and encoder exposures are named
separately. Thirteen RED-first objective tests pass; removing prefix padding protection leaks
the short row's terminal control and fails its guard. Review follow-up adds explicit recoded
operative-feedback, malformed framing, and bf16 geometry checks. Full-model bf16/FP32 equality
is not asserted: low-precision neural sublayers are intentionally distinct from FP32 islands.
The added raw-feedback reference initially differed by 0.0006430149 because it enabled
gradients while rollout disables them; a matched no-grad replay measured zero difference.
The in-flight commit was interrupted, its process group verified gone, and the reference
execution mode corrected without widening tolerances before rerunning the normal gates.

Step 4b adds phase-owned AdamW transactions. Missing/nonfinite gradients, unowned participation,
wrong optimizer ownership, clipping defects, and norm reconciliation fail with structured
context. Measured-zero gradients proceed through AdamW (the test observes weight decay and
moment-step advancement), while frozen/warm-up-inactive roles carry explicit reasons. Nine
RED-first transaction tests pass alongside thirteen objective tests. Disabling ownership
validation reaches the forbidden forward callback and fails the guard, then restoration
returns it to green. This does not yet provide persisted training or scientific custody.

Training review found that ownership alone admitted exact-owner SGD and modified AdamW
groups. Four RED-first regressions reproduce that gap; the transaction now requires AdamW,
matrix/vector partitioning, and the builder's moment/rate/decay policies. Schedule changes
must update the explicit optimizer contract, not silently mutate one group. A suggestion to
reduce per-parameter device synchronization is deferred pending an actual GPU profile.

### Next implementation seam (not completed)

Step 4d adds versioned Torch RNG and primitive/tensor rollout recovery payloads. Fifteen
RED-first tests cover pending-byte replay, safe weights-only serialization, all rollout
outcomes, codec/trace/score/shape validation, and exact operative-context agreement. Removing
the operative-slot guard admitted a shadow substitution and failed its test. Invalid RNG
bytes are checked on isolated generators before changing global state. CUDA replay remains
unverified until the authorized GPU probe; disk model/optimizer custody is the next slice.

Step 4e adds immutable, externally SHA-256-bound disk checkpoints for model, AdamW moments,
phase/cursor, Torch RNG, and rollout state. Nine RED-first tests reproduce the next training
transaction exactly and replay pending generation. Admission validates saved tensor precision
and parameter ordering before Torch can cast/remap moments; rehashed dtype/order mutations
were observed admitted before the fix and rejected afterward. Reader/source/construction
identities are bound, outputs cannot overwrite existing artifacts, and loading validates into
separate model/optimizer objects before restoring RNG. CUDA and remote custody remain pending.

Step 5a implements importance-weighted log-marginal estimates with frozen encoder proposals,
retained decoder/prior/proposal log scores, ESS, relative likelihood MCSE, and fixed-prefix
sample-count sensitivity. Boundary scores and source-byte counts remain distinct. Fourteen
RED-first tests cover an exact uniform-decoder oracle, both effective laws, target/prior
separation, chunked scoring, and typed zero-weight gaps. A mismatched batch broadcast was
reproduced and blocked. Removing the importance correction fails its guard. These are
finite-sample estimates, not certified pointwise bounds or oracle reconstruction scores.

Step 5b adds complete H numerical profiles with explicit optimization, sampling/evaluation
schedule, source digest, precision, byte permutation and all construction switches. Twenty-five
RED-first tests cover round-trip identities, every nested missing-field boundary, CUDA/bf16
remote tiny admission, retained 6,000-step budgets, and H1/H2 mechanism separation. Bypassing
required-field checks silently supplied a missing spine default and failed the guard; it was
restored. Profiles describe requests only and grant no spending or scientific authority.

Step 6a connects the numerical execution engine: source admission, measured parameter
inventories, explicit warm-up/freeze transition, training, RNG-isolated held-out scoring,
and exact checkpoint continuation. Ten RED-first tests include empty held-out rejection,
post-update meter failure, retained zero-direction gaps, and rejected-resume halting.
The execution fixture uses a declared test-only split policy; production never falls back
to training data for evaluation. Artifact journaling, CLI orchestration, and remote custody
remain next; this engine neither allocates pods nor grants scientific authority.

Step 6b introduces H checkpoint v2 with an optional measurement digest for standalone state
caches and a required expected digest when loading a measurement-bound checkpoint. The seal
layer will require this binding for journal custody. Eleven checkpoint tests pass, including
measurement mismatch refusal and explicit rejection of v1 through its original-reader path;
historical bytes are not rewritten or implicitly migrated.

Step 6c adds immutable, bounded JSON journal shards with source replay, exact field schemas,
gradient/absence reconciliation, objective arithmetic, exposure counts, and resource meaning
checks. Thirteen tests pass, including rehashed malformed JSON and substituted in-memory
source. Deliberately trusting that substituted corpus made its test fail; the restored writer
and reader reopen the hash-bound source capture. Journals alone grant no scientific authority;
checkpoint seals and the top-level execution command remain the next wiring boundary.

Step 6d joins journals/checkpoints into bounded, consecutive ancestry seals. Seven tests cover
warm-up → prediction → resumed prediction, skipped/unsealed work, artifact tampering,
optimizer drift, and RNG-neutral inspection. Removing the measurement join admitted a
different valid checkpoint and failed the swap guard. Static construction/group inventories
are instantiated on meta tensors; full tensor/optimizer admission still uses the checkpoint
reader at resume. A seal is identity evidence, not a scientific verdict or spending token.

Step 6e adds the probe training driver over real source, phase execution, journal seals and
held-out evaluation. Seven integration tests cover full budgets, pause/resume, retained
failures, scientific/successor refusal, empty-evaluation rejection and idempotent completed
replay without reinitialization. Removing the evaluation-presence guard failed its test.
Returned references are portable and stable. This status is training-complete only: generation,
diagnostic capture, fresh analysis/review/filing and remote workload validation remain open.

Step 6f exposes `h-train-probe` and `h-inspect-training` through the project CLI and just
recipes. Four public-command tests exercise real training, inspection, pause/resume, source
confinement, unknown-field refusal and off-grid pause rejection. Twenty-two CLI/deployment/
publishing tests pass. The current just formatter normalized interpolation whitespace only;
the existing deployment and publishing guards were unchanged. No command claims full probe
completion from training alone, and no hook configuration was modified.

Step 7a adds generation snapshots tied to the completed training receipt, without duplicating
trained weights. Nine tests cover two-sentence output, unscripted mid-sentence RNG replay,
invalid UTF-8 retention, prompt/RNG/summary forgery rejection and forensic-only defect capture.
Removing the prompt-history guard admitted a forged prompt and failed its test. Generation
uses the declared precision for initial and emitted encoding; inspect leaves caller RNG
unchanged. Generation-complete records an outcome or sentence budget, not semantic quality.

Step 7b exposes `h-generate-probe` and `h-inspect-generation` through the shared manifest CLI
and just recipes. The public-command lineage now covers trained checkpoint → generation →
mid-sentence pause → resume → independent inspection. Five command tests pass. Generation
requires a training-complete receipt and never grants scientific authority.

Step 7c replaces shape-only evaluation admission with independent source replay and numerical
reconstruction of importance weights, ESS, relative MCSE and sample-prefix summaries. Nine
tests reject rehashed source/weight/summary/count/meaning/timing forgeries. Removing summary
reconciliation failed its guard. The training driver invokes this reader before admitting
completion and during resume/inspection; no oracle score can be relabelled predictive.

Step 7d adds portable model-trace capture with measured file/byte budgets, immutable copies,
closed referenced inventories and source relocation. Four capture tests cover copied replay,
budget/occupancy refusal, missing checkpoints, unlisted files and retained generation history.
Generation receipt v2 binds resumed parent prefixes; v1 requires its original reader. The
closed-inventory mutation failed its guard. Capture grants no scientific authority and has
not yet been DVC-published or exercised on Runpod.

Step 7e exposes `h-capture-model` and `h-inspect-model` through strict manifests and just
recipes. Public-command tests cover capture followed by relocated numerical inspection,
unsafe-root refusal and invalid-budget refusal. The returned status is model-trace capture,
not scientific completion; DVC transport and the scientific flywheel remain open.

Step 8a introduces span-aware score-gain evidence with sentence-event and source-byte
coordinates, same-future-target binding, and explicit remaining horizons. Eighteen tests
cover identity/span/target refusal and preserve the historical adjacent-byte reader unchanged.
Disabling the binding guard failed all five mismatch tests. Summaries identify source
positions as nested observations, never independent model replicates or mutual information.

Step 8b adds bounded lag-stratified state collection from the hash-bound source capture.
Sentence windows/reset offsets, canonical targets, event coordinates and both horizons are
recorded; model inputs contain only completed spans. Four tests cover direct-prefix causal
agreement, deterministic selection, split isolation and insufficient-lag refusal. Substituting
an after-state for the before-state failed the direct-prefix guard. States are detached and
collection preserves model mode and Torch RNG; fitting/capture of probes is the next seam.
The approved line-specific B311 annotation documents the non-security statistical RNG;
scoped Bandit and all 22 span collection/score tests pass without changing security hooks.

Step 8c fits frozen CPU-FP32 linear probes on detached training states, conditioned on
log1p remaining source-byte horizon. This fixed-capacity development feature map is bound
in the probe identity, not a claim of complete latent-information recovery. Observed and
horizon-matched shuffled controls record their permutations; reports retain same-byte score
pairs, uniform controls, prediction quality, source identities and separate fit/score time.
Nine tests cover deterministic isolation, zero states, artifact/split/coordinate refusal,
known-signal learning and weight mutation. Disabling the training-split guard failed its
held-out-label test; the guard was restored. Disk capture and execution integration remain next.
Review exposed cross-side assignments when horizons overlap. The reproducing test failed
before the correction; permutations now preserve both horizon and before/after event side.

Step 8d captures both frozen probes, detached selected states, fitting configuration,
source coordinates, permutations and individual score pairs in a bounded report-only artifact.
Its reader verifies lineage bindings and fitting-state identities, then recomputes held-out
scores and summaries (declared FP32 replay tolerance: absolute 1e-5, relative 1e-6).
Twelve artifact tests cover rehashed forgeries, missing config, byte budgets and overwrite
refusal; disabling numerical reconciliation failed the forged-summary test. The nine probe
tests also pass, including the isolated document-overlap guard requested in review.
The adapter joining these tensors to an actual sealed checkpoint remains the next boundary.
Review hardening checks the selected tensor/head byte floor before fitting and uses a
size-limited serialization stream. Fifteen artifact tests now pass; disabling the stream
limit failed the overflow guard. Forgery tests pin their specific rejection messages.

Step 8e restores the actual sealed model before collecting span states, captures both probes,
and joins the diagnostic receipt to its training/checkpoint/source identities. The reader
replays source spans, canonical targets, window exposure and frozen scores without claiming
a second model forward pass. Seven lineage tests pass; removing the checkpoint join failed
the forged-checkpoint test. Population shortages retain typed diagnostic-gap evidence;
zero-direction failures retain affected counts. Collection and probe costs remain separate.
The combined lineage/collection/artifact suite passed 26 tests. CLI/capture integration remains next.

Step 8f exposes `h-diagnose-spans` / `h-inspect-spans` and includes diagnostic roots in
model-capture v2, with explicit v1 read dispatch. Eleven focused command/capture tests pass,
including a single trained → probed → generated → relocated-capture lineage, malformed
configuration refusal and all three diagnostic failure classes. Removing pre-copy diagnostic
admission failed the forged-exposure guard. Probe receipts remain observer evidence, not
scientific authority; no GPU allocation or remote publication occurred.

Step 9a separates portable byte custody from native model/source/RNG admission through one
shared inventory validator. `h-inspect-custody` reports native validation as not performed;
`h-inspect-model` retains all native checks. Seven custody/capture tests pass. Disabling the
hash comparison admitted a same-size tensor mutation and failed its guard. This enables
host-side transfer verification without pretending that a CPU host restored CUDA RNG state.

Step 9b binds native-reader success to a verification request and exact capture. Host receipt
admission rechecks byte custody without rerunning native admission; authentication of the
producing worker remains the transport/controller's responsibility. Four native/custody
tests pass, including idempotent success recovery, request/runtime/timing refusal and separate
records for repeated failures. Substituting custody for native admission failed the pass-receipt
guard. This is verification evidence only, never scientific successor authority.
Review hardening admits the complete receipt payload before its final atomic publication.
The rejected-receipt fixture initially left an unusable success marker; it now leaves only
failure evidence and permits a clean retry. Three native tests cover the corrected order,
capture/run binding forgeries and post-verification byte mutations. Removing payload
admission failed the success-marker guard.

Step 10a records compact pre-update solver convergence/action summaries, state/sample norms,
precision and zero-view counts in training-step/journal v2. Both Fisher and analytical
isotropic consumers retain evidence; codec warm-up is explicitly inactive. Missing active
readout/solver evidence and live-construction drift fail before the optimizer transaction.
Legacy journals remain readable but cannot authorize current restart seals. Thirty-eight
focused mechanism/journal/execution/seal tests pass, including zero norms and FP64 reduction
roundoff. Removing journal mechanism admission failed the forged-measurement guard.

Step 10b refuses ambient precision changes and TF32 overrides without mutating runtime
settings. Initialization is now inside the failure-recording boundary; evaluation records
its current source context and halts on sampling/scoring/timing defects, while zero-direction
gaps retain their scientific classification. Failure v2 binds its profile and run class.
Forty-four focused tests pass, including every guarded construction-drift branch. Removing
the default-dtype check failed its guard test. Native GPU precision remains to be measured
on the authorized probes; the local runtime check is not a substitute for that validation.

Step 11a adds an explicit optional byte cap to the existing immutable S3 object adapter;
H workers will require a cap, while existing callers retain their prior contract. Oversized
reads close the stream before consuming a body, and oversized uploads never reach the
provider. Malformed metadata also closes its response body. Read/write cap mutations failed
their guards; the combined adapter/runtime-precision suite passed 39 tests. The remaining
precision-control branches requested in review are now covered as well.

Step 11b defines canonical H worker requests with explicit source inputs, transfer caps,
probe fitting configuration and pause/resume intent. Remote H0 requires an immutable project
image and bound source proof; requests themselves grant neither spending nor scientific
authority. Later remote rungs remain refused pending their gate adapters. Nine request tests
pass; removing the source/profile join failed its guard. Input transport and worker execution
are the next wiring boundary.

Step 11c transfers byte-verified H inputs through the shared immutable object port, with
whole-input preflight and request-marker-last ordering. Materialization requires an empty
workspace and publishes its local marker only after every input passes. Fifteen request/
transport tests pass, covering tampering, budget refusal, missing readback, namespace drift,
duplicate JSON fields and noncanonical requests. Removing the local input hash check failed
its guard. This layer does not claim native data/model admission or authorize allocation.

Step 11d connects materialized worker requests to real training, generation, span diagnostics,
capture and native verification. The training driver exposes only durable, admitted checkpoint
receipts to its transport callback. Four worker tests pass, including a paused lineage moved
into a second workspace and resumed to its full budget, observer failure and changed-request
refusal. Removing the materialized-request join failed its guard. Workload completion remains
an engineering status, not generation-quality support or a scientific successor gate.

Step 11e publishes explicit output snapshots with byte/object caps and marker-last ordering.
Recovered snapshots check request identity, namespace, per-file hashes and exact measured
inventory before emitting a local marker. Six snapshot tests pass; removing total-byte
reconciliation failed its guard. The worker pause fixture now declares its request before
publication, and failure tests assert defect versus scientific-gap fields. Snapshots remain
byte-custody evidence only; terminal worker orchestration is the next connection.

Step 11f publishes native-input start, admitted checkpoint, paused, complete and failure
events around the same worker path. Completion requires matching training, generation,
diagnostics, capture and native verification before its terminal marker. Evidence-delivery
failure is explicit (`snapshot: null`), never completion. The missing-native-receipt mutation
failed the completion guard. Snapshot contract tests retain immutable paths across snapshot
names and preserve unadmitted partial recovery files without a ready marker; retries use a
new workspace. Remote provider/bootstrap/controller execution remains unstarted.

Step 11g adds host completion admission over the recovered snapshot and bound native receipt.
It checks the request, declared runtime/provider identity, full step budget and captured
training/generation/diagnostic lineage while explicitly not claiming a host CUDA rerun.
The real local worker round-trip admits six successful optimizer steps, and missing verification bytes or a rehashed wrong
request are rejected. Removing the terminal request join failed its guard. The coordinator's
missing-native-receipt rejection now has a specific message and failure-marker assertion.
Review follow-up covers remote runtime identity as a metadata contract, not a real GPU run.
Rejected completion retains byte-custody evidence for inspection without admitting completion.
Removing the pod-identity comparison failed the remote metadata guard test.

### Local validation checkpoint — 2026-09-17

Host control custody now enforces per-role OS locks in a private directory, binds
the exact launch manifest, refuses duplicate event publication and invalidates
released writers. A separate Python process proves exclusion and later acquisition.
Named provider reconciliation includes stopped pods and preserves multiple candidates;
an empty inventory is not proof that a timed-out allocation cannot appear later.
The control/provider focused suite passed 58 tests; removing the lock made its guard
fail. This supplies the durable boundary for the forthcoming watchdog/controller,
not a running supervisor, allocation, or scientific advancement.

Review exposed cross-role manifest initialization as a race. A shared initialization
lock now covers publication and exact-byte adoption. The new cross-process lock test
failed before the fix; concurrent fresh-directory startup and abrupt-owner-exit tests
now pass. The control/provider focused suite passed 64 tests, including the added
non-object/nonfinite inventory guards. No timeout or retry budget was increased.

### Local validation checkpoint — 2026-09-16

The H provider transport, since committed as `288ac0a`, uses the pinned bounded CLI
runner, redacts environment payloads, never retries creation, and retains possible
allocation IDs on ambiguous failures. Ownership is reread before deletion;
acknowledgement is separate from a successful absence check. The provider/launch
focused suite passed 80 tests. Removing the pod-name check exposed an attempted
foreign deletion and failed its guard; the guard was restored. Mutated launches
and incomplete not-found errors also have RED-before-GREEN coverage. No provider
mutation was performed; watchdog/controller composition is not yet implemented.
Command: `uv run pytest tests/test_h_runpod_transport.py tests/test_h_probe_launch.py -q --tb=short`.
The real bounded account path authenticated and measured balance 22.3809008535 USD,
current spend 0 USD/hour. This is account-read evidence, not allocation validation.
Review feedback added separate byte bounds, version rejection and an output-tail guard.
The existing D1 result/runner code disproved its missing-returncode claim.

H0 deployment descriptions now bind the full remote-tiny request, immutable image,
private template, placement, CLI build/digest and separate controller/watchdog budgets.
They generate a command without allocating resources or granting authority. The
template reader checks the documented secret-reference path and rejects changed
images, exposed ports, entrypoints, storage, secret references or serverless/public scope.
Workspace creation uses private permissions below the image-owned `/app` directory;
the shell fixture proves existing evidence is not overwritten. The scope-guard
mutation failed as intended, and the workspace hardening was RED before GREEN.
Focused validation passed 54 tests with
`uv run pytest tests/test_h_probe_launch.py tests/test_sentence_worker_request.py -q --tb=short`.
Removing the remote-only guard failed its local-request test; the two workspace
location/permission tests failed before hardening, then passed after the correction.
Actual provider calls, watchdog arming, template preparation and the prelaunch check-in
remain separate requirements; these declarations do not prove those steps occurred.

The host event reader now admits start/checkpoint/paused/complete/failure observations
from the existing immutable worker protocol. Its observed completion is explicitly
unverified until the separate capture/native-receipt check succeeds. Failure kinds
and failed evidence delivery remain distinct. The focused event/worker/completion
suite passed 43 tests; removing the request-identity join made its guard fail.
Review added a real paused-worker case; all 37 reader tests then passed, including
the restored request guard. Observations still carry no completion authority.
The follow-up review was clean and the repository full-suite hook passed. Manual
branch-message validation required explicit `origin/main..HEAD` refs and then passed.
This host-only addition does not alter the published image's numerical reader or
add provider allocation, teardown, or scientific authority.

The refreshed image is published from tested runtime revision `79a4be2` on the
approved Docker cloud builder. `docs/deploy-h100.md` records its amd64 digest;
the canonical machine-readable publication record is DVC-backed. Exact Git-source verification,
worker import and dependency/topology checks passed inside the pulled image.
This is not CUDA/6K validation; no Runpod was allocated. The existing world-writable
DVC site cache was left untouched; publication custody used a fresh private override.
The unused R2 bootstrap extension is recoverable at `results/deferred-h-r2.LmrgGj/`
(patch plus unchanged test, with hashes and base revision in its README).

Luiz approved publishing one refreshed project image using the existing Docker
cloud builder. Runpod allocation still requires the promised prelaunch check-in.
The old-image R2 bootstrap alternative is excluded from the image; publication
uses an isolated context exported from a tested commit plus the DVC-verified
historical field-weight asset. No working-tree secrets enter that context.

H statistical declarations now have their own strict schema entry point, sharing
validation arithmetic but not D namespace admission. Original bytes and all
scientific choices remain bound; missing effects/margins/endpoints, tiny classes
and cross-namespace declarations are rejected. This is design parsing, not H run
admission, scientific capture, seed sizing from tiny data, or launch authority.

`4e99a9f` commits the public worker command and its nine focused command/completion
tests (466 added lines); normal hooks passed without any scanner suppression.
The H1 preservation-block adapter now validates the complete paired 2x2 numerical
roster, permits only sentence-interface/spine-carrier differences, and exposes
neutral main-effect/interaction coefficients and required codec-sharing groups.
It grants no registration or launch authority and does not attest actual artifact
sharing, parameter matching, power or measurements. H scientific readers and
analysis/review/filing/gate adapters remain unfinished; D readers were not widened.

The worker-command annotation request is obsolete: its credential-forwarding
test now compares against one canonical dummy fixture instead of duplicating a
secret-shaped assertion literal. No annotation, baseline or scanner change was
made. Deliberately forwarding the wrong credential makes that test fail; the
mutation was restored. Image publication remains a separate pending decision.

The pending R2 bootstrap slice passes 40 local bootstrap/transport tests, but
offline inspection found that image `b88de2f7...` has neither `/app/.dvc/config`
nor boto3. Its pre-install client assembly therefore fails before authentication;
it is not a runnable remote bootstrap on that image. No real credentials were
used in this inspection and no R2 objects or Runpods were created. Publication of
one refreshed immutable project image has been proposed to Luiz as the simpler
production route; keeping the old digest would require a separate prerequisite
installer. The R2 slice is preserved uncommitted pending that deployment choice.

`2a8a7e2` adds the stdlib-only bound-source bootstrap (496 added lines), and
`9f54d16` enforces workspace ownership/private internals and explicit pinned uv
selection (94 changed lines). Both passed normal hooks; 36 focused bootstrap and
deployment tests passed. Source extraction also passed offline inside the pinned
project image; host/native readiness records matched byte-for-byte for the same
five-file Git fixture. All owned inspection containers were removed afterward.

The image's PATH selects `/usr/local/bin/uv`, a different binary from the copied
`/bin/uv`. The former's version check was stopped after exceeding 30 seconds under
local x86 emulation. The latter reported 0.11.6 and successfully exported the
current frozen deploy graph offline with pySBD included and Torch pruned. This
does not establish why the first binary stalled or validate dependency installation.
Bootstrap still needs R2 fetch, installation and worker-start composition; no
Runpod was allocated. The worker-command scanner hold is resolved as noted above.

`29bd163` adds manifest-bound source preparation/inspection (327 added lines).
`2cbb5fb` closes the reproduced empty-dataset provenance collision (96 changed
lines): v2 captures bind dataset configuration; v1 requires its original reader.
Both commits passed normal hooks; the provenance follow-up review was clean.
The source/capture/corpus focused suite passed 76 tests before the additional
missing-dataset-field case; that added case passed in the full unit hook.

Offline local inspection of the pinned amd64 project image
`sha256:b88de2f7224b3a40fef1b0fbc2ac78589822c28645fe1aba5503318e5ba599f1`
measured Python 3.12.3 and Torch distribution 2.13.0+cu126. pySBD and the H source
package are absent. Bootstrap must supply bound source/dependencies; prepare
source captures in that runtime rather than weakening reader compatibility.
This was CPU-only inspection, not CUDA/bf16 validation or a Runpod allocation.

The `h-worker` command now composes bounded R2 publication, worker execution and
host-side completion admission. Seven focused tests pass, including a real local
model lineage; disabling the local/remote scope guard makes its test fail. Invalid
transport limits fail before client creation. Review tightened credential-guard
isolation; raw exception logging was declined to avoid leaking provider details.
This slice does not add pod allocation, deployment bootstrap or scientific authority.

At `e524635`, `just test-all` passed 4,196 tests with 10 skips. The sequential live
container gate then passed 4 main integration, 3 allowlist and 3 authentication tests;
no project test containers remained afterward. Every implementation commit in this
continuation passed normal hooks and stayed below 600 added-plus-deleted lines.

This is not H0 scientific completion or remote-probe readiness. The local worker/evidence
path is connected, but deployment/bootstrap, provider/controller/watchdog composition,
formal H-stage analysis/review/filing adapters and remaining preservation/fidelity work
are still open. No Runpod has been allocated. Native R2 secret-reference names have been
requested from Luiz; credential values must not be shared in chat or committed. Check in
before allocation with the exact 6K roster, phase split, quote and teardown policy.

The law head, consumer, absolute future targets, sampled decoding, emitted-text feedback,
and phase-owned transactions are now connected. A synthetic CPU smoke ran 100 warm-up and
100 prediction updates (width-8 fixture, seed 42, AdamW lr 0.001, decay 0.01, clip 1,
future weight 0.3). Observed objectives were 5.1925793 → 0.2889636 during reconstruction
and 6.7724171 → 0.4944233 during prediction; these are different training objectives,
not comparable language-likelihood measurements. Unscripted rollout emitted `C.` and
document-end. This is a memorized engineering fixture, not generation-quality evidence.

Step 4c adds a counter-seeded, split-bound document sampler. Its versioned equal-history
length buckets preserve uniform target-sentence marginals and exact byte spans. Ten
RED-first tests cover replay, source pairing across codec permutations, model-RNG isolation,
empty/forged source rejection, and causal framing. Including the target span in history
fails its causality guard; the mutation was restored. Snapshot cursors bind source/config.

Next: worker/controller wiring and preservation/fidelity measurements;
explicit H registrations and stage adapters; complete shared DVC/R2 custody, remote worker/
controller and analysis/review/filing integration. Then the authorized
6,000-step CUDA/bf16 tiny probes with the approved host-watchdog exception, verified artifact
recovery, termination and billing. Check in with Luiz before allocation, with the exact roster,
phase split and quote. No pod has been allocated. Ordinary CodeRabbit reviews have resumed;
credit mode remains disabled. Normal hooks remain mandatory and unchanged.

## Current direction and deferred old stage

The approved destination is the [codec-separated hierarchical latent predictor](docs/hierarchical-latent-predictor.plan.md): sentence-first next-latent-sequence generation,
with a roughly 500M–1B total deployed-model envelope. The local generative path is implemented;
complete flywheel integration and remote validation remain open.
The following September 12 checkpoint remains historical, not current completion status.

The operator chose to preserve the unfinished old D host stage and move to the new design.
Sol's focused tests passed, but parent review rejected the receipt slice: `run_class` was not
bound to producer evidence and the expected capture was not bound to the producer's exports.
The positive fixture was not producer-faithful. No new feature commit was made; runtime HEAD
at this decision was `256b00b`. Other deferred blockers are the missing host/application
modules and the invalid `dvc_root` argument to the analysis-input contract.

Recoverable snapshots (restore explicit paths, never apply broadly over newer work):

- `35b208b545bd21e3a9027542e97d3e2d597d64f6`: three tracked old-tail changes and eleven
  untracked host/operator/capture files; Git also recorded the then-staged receipt in its
  base/index state.
- `62b9948f8fdae652e166f862c55e51b0f8e287b5`: the latest receipt implementation and expanded
  tests, preserved as two untracked paths.

The first commit hook stopped because an untracked CLI test remained visible while prek hid
its unstaged implementation. After isolating the remainder, the second attempt was interrupted
before committing the incomplete attestation contract. No hook was bypassed and no commit or
test process was left running. The worktree was clean before the north-star documentation edit.

## Historical September 12 handoff

## Incremental commit checkpoint

- Branch: `d1-golden-scientific-flywheel`.
- Delivered checkpoint before this handoff: `6199905`.
- Delivered range: `addea69..6199905`, **43 local commits**. Every commit passed
  the normal hooks, including the full unit suite. The recorded added-plus-deleted
  line counts were audited; the largest commit contains **597 changed lines**.
- The transport scanner blocker is resolved in `750f7d4` using the existing
  `execution_segment_id` helper. Runtime verification and the worker command
  followed in `4befc4e` and `efaa254`; shared attempt/provider contracts and D1
  fresh/resume adoption are committed in `c04146e` and `1ad659b`.
- With explicit operator approval, `prek update` refreshed 11 hook release pins
  in the isolated 22-line commit `cb6bda6`. Rules, exclusions, arguments and stages
  stayed unchanged. Ruff 0.16.7 accepts the unchanged D1 controller that the old
  0.14.13 hook rejected. The separately approved `e9589c5` changes only the
  downloader's executable bit; its contents are identical.
- The separately approved Markdown alignment is `6199905` (44 changed lines).
  dprint now matches the existing asterisk-emphasis lint policy, and code spans
  protect formula and experiment-name literals. Three regression cases pass;
  deliberately restoring the two hazards makes the matching cases fail.
- The remaining tail contains 13 new files and three tracked-file changes. It is
  restored only after committing this handoff; it is not part of the verified
  code checkpoint. Recoverable snapshots are `ed68d15` (remaining delivery with
  the approved segment helper) and `61a7a03` (original September 6 delivery).
  Do not blindly apply either snapshot: newer committed fixes must be preserved.
- Host orchestration is still incomplete. The saved campaign/operator command
  surfaces reference the not-yet-implemented `d_ladder_operator.py` and
  `d_ladder_application.py`. Generic capture/application and later-rung pilot
  preparation require integration before production end-to-end readiness.
- No push, merge, release tag, or paid Runpod run was performed.

September 11 focused validation: 50 shared-attempt/provider/D1 fresh-resume tests
passed. The updated Ruff scan over `src`, `scripts`, and `tests` passed, and
basedpyright 1.40.1 reported zero errors or warnings over `src` and `scripts`.
Code review led to runtime enforcement that failure handlers cannot return, plus
interrupt-cleanup and independent launch-guard tests. These are local software
checks, not scientific pilot, review, filing, or successor-gate evidence.

The handoff's review-adapter formatting cleanup passed 32 focused tests, and the
September 11 dependency/security audit passed with existing annotation-parser
warnings. The complete local host-command lineage and its final integrated/live
validation remain open; the restored tail is not covered by these passing checks.

## Current implementation checkpoint

The August 27 snapshot below is historical. The approved September 4 plan builds
all local rung mechanisms first and gates remote successor execution on filed
predecessor evidence. Implementation is on `d1-golden-scientific-flywheel`; no
paid probes, publication, confirmatory runs, or scientific verdicts were performed
by this implementation pass.

Implemented and locally exercised:

- D1 measured-zero gradients, undefined zero-vector cosine, and construction-v4
  resume separation; strict missing/nonfinite/ownership refusals remain.
- CPU/GPU tiny catalogs; excluded pilot request, operator fresh/resume, capture,
  reload, and source-provenance verification.
- Fully differentiated matrix-free inverse Fisher, including exact-zero inverse
  tangents and compensated true-residual convergence checks at the fixed budget.
- D2 L/S/G, D3 horizons/composite and measured parameter matching, D4 causal
  controls/router, and D5 permutation/independent-training diagnostics.
- Observable future score probes with canonical targets, context-reset exclusion,
  fixed checkpoint identities, training-only fits, and separate resource scopes.
- One tiny/scientific numerical executor, checkpoint/resume and observed source
  prefix custody; D3 horizon arms reserve the same source-offset population.
- Rung-specific authorization, Git/DVC-bound capture, fresh analysis, blind packet,
  filing and predecessor rehydration. D4 fixed-to-learned is a separate gate;
  D3 can carry a short-horizon state directly to D5 when chunking is retired.
- RunPod R2 worker `serve`/`inspect`/`publish` entrypoint with actual startup and
  source-tree verification. Allocation, spending and termination require their
  approved operator phase.
- Pilot-based seed sizing and same-population 95% deciding / 3-sigma reporting.

Earlier measured validation checkpoint: `just check` completed with **3598 passed,
21 deselected** before the later host-integration additions. The dependency and
security audit passed, and the sequential live-container gate passed 4 tracking/R2,
3 host-allowlist, and 3 Caddy-auth tests. These results do not validate the unfinished
restored tail. A same-lineage golden test executes 18 actual tiny scientific cells
across D2–D5; external review and DVC transport in that test are explicit test doubles,
not research evidence. Every delivered commit additionally passed its normal hooks.

Remaining: complete host campaign/capture/analysis/review application wiring and
later-rung pilot preparation, prove the production-shaped host-command lineage,
and run its final integrated gates. The restored tail is unfinished, not another
passing implementation slice. Keep subsequent commits at most 599 added-plus-
deleted lines each. Remote spending/publication remains unapproved.

## Historical August 27 checkpoint

Purpose: durable debrief checkpoint for the current D1 implementation pass, per
`/Users/luizfrias/.claude/commands/debrief-and-pass.md`.

Source note: the seven-step plan below is transcribed from
`/var/folders/m1/_h2kd5ss0yd0s9124mq04lw00000gn/T/codex-clipboard-ceb88b7c-1a62-4b32-89ab-b25da2c492af.png`
and then reconciled against the live branch state.

## Branch Snapshot

- Branch: `d1-runpod-proxy-allocation-hard-cut`
- HEAD: `ab4d7e7377aa3cb8a79ed37a034291794db9d8e4`
- Worktree state at debrief time: clean
- Active scope: D1 only
- D2-D5 status: paused; do not resume higher rungs from this checkpoint

## Screenshot Plan Reconciled To Live State

| # | Step                                                                                                                       | Screenshot state       | Live state | Evidence anchors                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| - | -------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Preserve terminal outcomes and finalized-cell custody across campaign checkpoint/recovery                                  | Completed              | Completed  | Commits `ea1ddcf`, `1bcb707`, `f3e8c1f`, `dc16ad6`; code in `src/entropic_substrate/d1_finalized_cell_custody.py`, `src/entropic_substrate/d1_finalized_cell_custody_repository.py`, `src/entropic_substrate/d1_campaign_bootstrap.py`, `src/entropic_substrate/d1_campaign_authority_loader.py`; tests in `tests/test_d1_finalized_cell_custody_values.py`, `tests/test_d1_finalized_cell_custody_repository.py`, `tests/test_d1_campaign_authority_loader.py`                                                                                                                                                                               |
| 2 | Implement real capture-prepare and capture-finalize application services with strict request/result contracts              | Selected / in progress | Completed  | Commits `d54db3a`, `c3d29cd`, `f1f7d1a`, `ad94f7e`, `1db3666`, `2dacb3a`, `c041e24`, `2cb1fd7`, `c076350`; code in `src/entropic_substrate/d1_capture_prepare_coordinator.py`, `src/entropic_substrate/d1_capture_finalize_coordinator.py`, `src/entropic_substrate/d1_capture_finalize_request.py`, `src/entropic_substrate/d1_capture_cli_services.py`, `src/entropic_substrate/d1_cli_composition.py`; tests in `tests/test_d1_campaign_capture.py`, `tests/test_d1_capture_prepare_coordinator.py`, `tests/test_d1_capture_finalize_coordinator.py`, `tests/test_d1_capture_cli_services.py`, `tests/test_d1_capture_finalize_request.py` |
| 3 | Wire the exact six-command public CLI surface into the production composition root                                         | Pending                | Partial    | The six-command contract is real in `src/entropic_substrate/d1_cli.py` and `tests/test_d1_cli_registration.py`. Default production composition in `src/entropic_substrate/d1_cli_composition.py` wires `d1-run-fresh`, `d1-run-resume`, `d1-capture-prepare`, `d1-capture-finalize`, and `d1-inspect`. `d1-campaign-run` has a real service in `src/entropic_substrate/d1_campaign_cli_service.py` but remains an explicit `operator_gap` in the default composition until the authoritative trainer transport is chosen.                                                                                                                     |
| 4 | Integrate real-arm verdict assembly, filing, and successor gate from authoritative evidence                                | Pending                | Partial    | Real-arm assembly exists in `src/entropic_substrate/analysis/d1_real_arm_verdict.py`; filing and gate types exist in `src/entropic_substrate/evidence/d1_filing_graph.py`, `src/entropic_substrate/evidence/d1_process_filing.py`, and `src/entropic_substrate/evidence/d1_gate.py`; those pieces are pinned by `tests/test_d1_real_arm_verdict_attestation.py`, `tests/test_d1_real_arm_verdict_assembly.py`, `tests/test_d1_filing_graph_gate.py`, and `tests/test_d1_process_manager_gate.py`, but the end-to-end production path is not yet the active D1 default.                                                                        |
| 5 | Prove the full D1 golden flywheel for success, defect, resume, crash adoption, capture, analysis, review, filing, and gate | Pending                | Partial    | The full end-to-end golden flywheel remains to be built and run as the authoritative D1 proof point. Current green tests cover flywheel state and filing guards in `tests/test_d1_process_manager.py` and `tests/test_d1_lifecycle_filing_guards.py`, but not the entire production-shaped path as one harness.                                                                                                                                                                                                                                                                                                                               |
| 6 | Run final local D1 audit, present Graphite/Linear/evidence gaps, and request publication approval                          | Pending                | Open       | Not started. This debrief is a progress checkpoint only, not the final D1 audit. No `just ship-it` run has been attempted from this checkpoint.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 7 | Apply the operator ruling for authoritative RunPod trainer transport and run the bounded excluded infrastructure probes    | Pending                | Partial    | The bounded excluded probes were run and the RunPod create contract was hard-cut in commit `ab4d7e7`. The operator ruling is still open: proxy SSH is now treated as diagnostic-only, not authoritative trainer transport.                                                                                                                                                                                                                                                                                                                                                                                                                    |

## What Landed Most Recently

- `ab4d7e7` hard-cut the RunPod create path to allocation-only semantics.
- The adapter now keeps `--ssh` but drops `--wait`, `--wait-timeout`,
  `--ports`, and `--public-ip` from pod creation.
- The create response no longer treats `runtimeStatus` or a direct `ssh`
  object as required evidence of trainer readiness.
- The matching guard test is
  `tests/test_d1_runpod_adapter.py::test_proxy_only_create_does_not_require_tcp_ssh_or_wait`.

## Latest Verified Gates

- Latest targeted gate before this debrief: 102 RunPod/controller/resume tests
  green after the proxy hard-cut restoration.
- Latest repository gate before this debrief: `just check` green with
  `3095 passed, 21 deselected`.

## Measured Infrastructure Findings Preserved Here

- Measured CLI version during the bounded RunPod probe:
  `runpodctl 2.11.0-0592468`.
- A no-wait CPU pod create probe returned allocation facts without a direct
  `ssh` object and without `runtimeStatus`.
- An earlier proxy-SSH probe showed PTY-framed execution and unreliable local
  exit-code propagation for remote failures.
- Resulting interpretation for D1: RunPod proxy SSH is acceptable for bounded
  diagnostics, but not yet acceptable as the authoritative trainer transport.

## Decision Still Needed

- Authoritative D1 trainer transport remains unresolved.
- Recommended direction: content-addressed request/report/export custody over
  operator-visible artifacts, with RunPod proxy SSH limited to diagnostics and
  emergency inspection.
- This decision blocks default production wiring for `d1-campaign-run`.

## Resume From Here

1. Implement the authoritative trainer transport behind the existing
   `D1RecoveryAttemptControllerPort` path and wire it into the default
   `d1-campaign-run` composition.
2. Integrate real-arm verdict assembly, filing receipts, and successor gating
   from authoritative D1 evidence.
3. Add the production-shaped D1 golden flywheel covering success, defect,
   resume, crash adoption, capture, analysis, blind review, filing, and gate.
4. Run the final local D1 audit and only then request publication approval.
