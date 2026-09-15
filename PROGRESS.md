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

Next: phase-bound disk checkpoint/optimizer/RNG/source-cursor/rollout recovery; preservation,
marginal scoring and span-aware probe diagnostics; explicit H registrations and stage
adapters; complete shared custody/analysis/review/filing integration. Then the authorized
6,000-step CUDA/bf16 tiny probes with the approved host-watchdog exception, verified artifact
recovery, termination and billing. No pod has been allocated. CodeRabbit's included-use
limit was reached on the optimizer-contract follow-up; its quoted $0 credit-mode approval
is pending. Normal hooks remain mandatory and unchanged; no paid review is authorized.

## Current direction and deferred old stage

The approved destination is the [codec-separated hierarchical latent predictor](docs/hierarchical-latent-predictor.plan.md): sentence-first next-latent-sequence generation,
with a roughly 500M–1B total deployed-model envelope. The new generative path is not built.
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
