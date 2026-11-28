# Platform refactor source review at 5f6b6dd9

Written 2026-11-27 on the read-only survey worktree at
`5f6b6dd92b9da007624ce255492f55b779fd4bfd` (origin/main, v0.12.3). It reads `src` and `tests`
against the plan `docs/plans/series-plugin-topology.plan.md` (rung six) and its corpus review
`docs/reviews/2026-11-01-series-plugin-topology-corpus-review.md`. Nothing in this file
authorizes a lane, a push, a pod or a merge.

**How it was produced.** One orchestrated review, journal `wf_a32716f6-5fb`: 25 scouts read every
file in 22 source bundles and 3 test and tooling bundles and wrote one record per file (401
records); 10 cross-module lenses read those records and the code; 10 adversarial verifiers
re-opened every cited line and recounted every number; one synthesis pass read all of it. 45
agents, 2159 tool calls `[measured, workflow usage]`. The verifiers checked 219 findings: 125
confirmed, 93 corrected as imprecise, 1 refuted; three plan-delta rows and three sub-claims were
also refuted `[measured, verify/*.json]`. The scout records, lens reports and verifier files are
not tracked; they sit under `review-artifacts/platform-refactor-2026-11-27/` in the project's
Claude data directory on the operator's machine.

**Two parts.** Part one is the synthesis as the reviewing pass wrote it, unedited apart from this
header. Part two, at the end, is the conductor's reconciliation: where an operator ruling of
2026-11-27 already decides a point the synthesis leaves open or recommends differently, and what
the conductor re-measured before relaying. The plan's rung seven is built from both.

---

## Part one: the synthesis

Its own inputs, as it states them: the plan and the corpus review read in full,
`~/.claude/rules-lib/doctrine/architecture-investment.md` read in full, Python modules 50, 55 and
57 read in full, ten lens reports with ten adversarial verifications, and 401 scout records.

**Evidence labels used here.**

| label | meaning |
| --- | --- |
| `[measured]` | read or grepped by this synthesis in this session, with path:line |
| `[relayed-V Lnn-Fm]` | a lens finding a verifier confirmed or corrected; not re-read here |
| `[relayed-V-only Lnn]` | raised only by a verifier in its "missed" list; one source, never counter-checked |
| `[relayed-L Lnn-Fm]` | a lens claim no verifier confirmed |
| `[scout]` | taken from scout records; unverified |
| `[inferred]` | this synthesis's own reasoning or design recommendation |

Refuted findings are dropped: L07-F11 (test moves do re-pin, through test paths named in
reader-member comments) and the L03 plan-delta that the `init=False` count drifted. Every
verifier correction is applied in the text below.

### TL;DR

The plan's direction holds: engines are platform, series are adapters, specs carry selection and
code carries promises. Its ordering does not hold. The plan assumed one reader re-pin at the end
and a start after H1. Both assumptions are gone. The reader identity hashes 54 module files by
name and bytes and string-imports composition from the run layer, so every milestone that
touches a member re-pins, and tooth 2 is red on day one. The fix is to move the identity door to
the front, replace T1 with a freeze predicate, and move bottom-up so each file lands once in its
final member with no forwarder at any milestone tip. Analogy: the plan treated the reader
identity as a lock to change once when the furniture is in place. It is the building's
foundation survey. Re-survey it first, in a form that tolerates the move, then move the
furniture.

---

## 1. What changed since the plan was written, and what that does to the plan

### 1.1 Code

| change | evidence | effect on the plan |
| --- | --- | --- |
| main moved 157 commits past the plan's SHA | `git rev-list --count b9c570e0..HEAD` = 157 `[measured]` | Context and Phases tables are stale; restate at the sprint-start SHA with one named instrument `[relayed-V L08-F3]` |
| reader tuple grew to 54 members and spans composition 4, kernel 2, core.latent 2, core.probe 1, core.sentence 19, ladder.capture 1, ladder.run 25 | `src/entropic_substrate/ladder/run/checkpoint.py:78-82` shows the composition entries `[measured]`; split counts `[relayed-V-only L09]` | plan:286 "52 members across kernel, core and ladder" is wrong; no single series spec can own the list |
| identity is built by string import and hashes `name:digest` | `checkpoint.py:6`, `:139`, `:144`; `ladder/capture/capture.py:4`, `:44` `[measured]` | the future platform member would need the cli member installed to compute its own identity; tooth 2 and module 50 refuse the form `[relayed-V-only L06, L08, L09]` |
| H checkpoint embeds the reader identity and the loader refuses a mismatch | `checkpoint.py:276`, `:320` `[measured]` | D5's "readers admit v2" clause cannot be implemented; replay at the epoch commit is the only window `[relayed-V L01-F2]` |
| seven H modules added; H pod path doubled to 5527 lines and runs single and pair | `[relayed-V L08-F7]` | P3c absorbs two H shapes, not one; P6 grew |
| fifth ladder package cycle, register/run | `ladder/register/rungs/h1_pilot.py:43` `[relayed-V L04-F2]` | the cycle goes with the file's move to series-h; no TransferLimits move is needed |
| largest file is 1256 lines | `wc -l src/entropic_substrate/ladder/run/object_worker.py` = 1256 `[measured]` | the plan's ceiling baseline of 1007 is stale |
| five model-training loops and five checkpoint custodies, plus two probe-fit loops | `[relayed-V L01-F1]` seven `.step()` sites | "three trainers" was wrong at the plan's own SHA |
| the digest and codec seams never landed; the correctness leg did | `[relayed-V L09-F1, F3]`; 235 clone definitions, 73 canonical writer sites | P0's gate is unmet; the debt plan is not a tracked file `[relayed-V L09-F20]` |

### 1.2 Tests

| change | evidence | effect |
| --- | --- | --- |
| tests are tier folders: unit 400, repo 35, integration 7, architecture 6, e2e 6, live 6, fault 2; 462 test modules, 484 Python files, 19 support modules | `find tests` per tier `[measured]` | plan:299 "one directory of 456 files" is stale |
| the unit tree mirrors src by directory, not by member | 400 of 400 import their mirrored directory `[relayed-V L07-F1]` | a test's destination follows the src split ruling; 56 to 198 files have no single destination `[inferred range, relayed-V L07-F1]` |
| 387 test-to-test import edges onto 94 providers | `tests/architecture/test_test_import_budget.py:31-32` `[measured]` | cross-member edges must reach zero before a member gate can run bare |
| the stage driver knows `tests/<tier>` literals only | `scripts/suite_stages.py:75-82`, `:98` `[measured]` | a test moved under `packages/` is run by no stage with every guard green `[relayed-V-only L07]` |
| the last test relocation forced a regime re-pin | commit c161a032 after PR 437 `[relayed-V L07, refutation of F11]` | test moves are identity-bearing until test paths leave reader-member comments |

### 1.3 Doctrine

| change | source | effect |
| --- | --- | --- |
| a port is extracted at the first named second option, cut thin | architecture-investment.md:71 `[measured]` | the plan's wait-for-three reasoning is superseded; FilingRoles, Codec, Roster extract now, thin |
| rosette topology: about six ports per cell, planes touched through one thin port, contracts per measured wall, members are packaging boundaries | architecture-investment.md:93-109 `[measured]` | one `platform/ports.py` holding 17 named ports fails the count `[relayed-V L08-F22]`; observability as four peer ports is the quadratic shape the doctrine warns against |
| posture is ambitious by default | architecture-investment.md:35 `[measured]` | start from the most capable stable pattern; the counterweights are the admissibility gate, carry tiers and the file ceiling |
| fakes ship with the port in the owner's `testing` module | module 57 MUST "Contract tests ship with the port" `[measured]` | src has no `testing` module today `[relayed-V-only L03]`; the tests move depends on one |

### 1.4 The operator's decision to refactor before H1 completes

| fact | evidence | consequence |
| --- | --- | --- |
| the plan's start trigger is T1, "after H1 completes entirely" | plan:5, plan:95, plan:426 `[measured]` | the trigger is unmet and the operator has reversed the order; the plan's own risk row rates that case high (plan:407) |
| H1 has nothing filed or sealed; the preregistration skeleton is a draft with identity rows blank | `[relayed-V L10-F10, L06 missed]` `docs/h1/preregistration-skeleton.md:316-318` | the window is open now: re-pins invalidate nothing sealed |
| H0 is already epoch-replay only | 21 of 54 member files modified and 3 added since 046cfb05 `[relayed-V L06-F1]` | the refactor costs H0 nothing further |
| L4d's 12 commits are held by one fetch-head ref | `git for-each-ref --contains 35b0b687` returns only `refs/gt-fetch-head/h1-l4d-evaluation-instruments` `[measured]` | a durable ref is the first act of the sprint |

**What this does to the plan.** T1 is replaced by a freeze predicate: no H1 seal, capture, image
re-pin or pod while any reader member is in motion `[relayed-V L10-F11]`. The "one re-pin at P6"
accounting is replaced by one identity door at the front and one routine conductor re-pin per
member-touching milestone. P6 is no longer last for identity reasons; it is last because series
code depends on everything below it.

---

## 2. Plan section verdict table

Status vocabulary: holds, amend, reshape, stale, wrong, missing.

### Rulings

| section | status | change |
| --- | --- | --- |
| R1 ladder protocol is platform | amend | Direction holds. The 2x2 factorial arithmetic and arm-blind custody are platform; the D1 decision contract and verdict schema stay series. Cut the filing minimum thin: the kernel's three identities plus a series-declared custody profile, not D1's thirteen roles `[relayed-V L04-F11, F13]`. |
| T1 own sprint after H1 | reshape | Replace with the freeze predicate plus a pre-motion flush. The calendar coupling goes; the invariant that no evidence is sealed under a moving identity stays `[relayed-V L10-F11, L08-F1]`. |
| R2 explicit registry | amend | One spec per `SeriesId` member, never a glob, because a glob cannot see a missing spec `[relayed-V L03]`. Add a command port generalized from `D1CliApplication`. The registry composes the reader identity at the root. |
| R3 retirement is a flag plus an admission row | holds | `logger_smoke` `SERIES_PATHS` is already this ruling in miniature and folds into `series.toml` `[relayed-L L08 code judo]`. The D2 to D5 lane needs its status ruled. |
| C1 pins from spec files | amend | Add `runpodctl_sha256`. The loader refuses `inf`, `nan` and datetimes, which `tomllib` admits `[relayed-V L06-F13]`. Record spec digests in `run_start` behind one wire bump, not in three manifest families `[relayed-V L06-F11]`. |
| R4 engines are platform | holds | The measured overlap supports one engine `[relayed-V L01]`. Amend the consequence: five loops and five custodies, two cells (custody and step engine), and a series bundle of about nine edges, not "ModelFactory and Objective only". |
| R5 workspace members | amend | Add a migration rule: the residue package is the only one that may import across members, and it only shrinks. Kernel becomes stdlib-only. Core is a portless shared class 1 library. A workspace test tier exists beside member tests. |
| R6 platform as a library | holds | No lens tested it. The foreign-fixture oracle of D7 stands. |
| R7 member CLAUDE.md | holds | No lens tested it. Add: each member file records its identity manifest and any doctrine deviation, such as the hand codec against module 20. |
| C2 observability ports | reshape | On E and H the tracker is the courier for authoritative JSONL and the capture run index, not a mirror `[relayed-V L05-F1]`. Fix the courier first, then the mirror claim becomes true. One thin emit port for the plane; `Tracker` stays as an adapter-side mirror port; `ArtifactStore` and `RunRegistry` move to the custody cell. |
| R8 two consumption modes | holds | No lens tested it. The testing kit must sit where the copy test can resolve it. |
| R9 one Source port | amend | Cut it thin now: `next_batch`, `snapshot`, `restore`, `order_digest`. The two samplers diverge in shape and the D1 engine pins `SequenceSampler` by exact type `[relayed-V L03-F20, F21]`. |
| R10 numerics are a promise | holds | Correct the variable name to `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE`; the gate admits unset or `0` `[relayed-V L06-F14, L08-F10]`. |
| R11 Gymnasium contract, TorchRL first | holds | No lens tested it. |

### Open decisions

| section | status | change |
| --- | --- | --- |
| D1 TOML spec format | amend | Holds with the non-finite and datetime refusal, and one named kernel canonical-JSON scheme for the spec digest `[relayed-V L06-F13, L09-F5]`. |
| D2 reader lists in `series.toml` | wrong | Membership is a promise, not a selection, and the 54 members span five target members `[relayed-V L06-F4]`. Keep each member's list in code as a typed manifest, compose at the composition root into one frozen kernel value handed to the readers, and pin a regime-free manifest digest `[relayed-V L06-F7, relayed-V-only L06]`. |
| D3 wire versions in code | amend | The principle holds. Do not move 348 literals into enumeration modules: 130 sit in 34 reader members and the move narrows the identity. Use a generated catalog plus a define-once lint `[relayed-V L06-F12]`. |
| D5 unified checkpoint custody | reshape | The typed `CheckpointPayload` port holds. Strike "readers admit v2". Restate the proof as decoded-payload equality with `reader`, `measurement` and wall fields excluded `[relayed-V L01-F2, relayed-V-only L01]`. Name one retention rule for `Journal.seal` and the engine to pod contract. |
| D6 Trackio second adapter | amend | Defer until the courier fix lands and the tracker read-side consumers are re-homed, so the adapter is write-only `[relayed-L L05-F22]`. The conformance suite is the existing hand-enumerated MetricSink contract file, converted `[relayed-V L05-F21]`. |

### Sections

| section | status | change |
| --- | --- | --- |
| Context tables | stale | Restate at the sprint-start SHA: 401 src files, 54 members, five ladder cycles, three 64-hex literals, 348 schema literals, 18 to 21 environment names. |
| The seam | amend | Member sets are promises. Classify all environment names, and give `ENTROPIC_ALLOW_UNGATED` and `ENTROPIC_CKPT_TAU_SEMANTICS` a class; the plan's three classes have no slot for a gate override `[relayed-V L08-F10]`. |
| Target topology | reshape | Rosette form, section 3. Per-cell ports with one series-facing `entropic_platform.ports`; engine and custody as two cells; planes; `contracts/` per measured wall; a `testing` module per port owner; scripts gate drivers homed. |
| Placement table | reshape | Seven of fifteen planned ports have no Protocol in src `[relayed-V L03-F1]`. `Schedule` fuses an LR law with a phase plan. `Admission` is three capabilities. `ProcessRunner`, `GitFacts`, `Clock`, `Billing`, `Guard` and `ControlLedger` are missing. `Isolation` is unwired. |
| Typestate paragraph | amend | The claim fails at the FSM: `FiledVerdictEvidence` is a public dataclass `[relayed-V L04-F6]`. Make filed evidence factory-only first, then redesign the gate mint; the token is the only tooth until then `[relayed-V L04-F7]`. |
| What stands in the way | wrong | The tuple is 54 and includes composition; `ModelConfig` is E's block-factory argument, not a neutral kernel type `[relayed-V L09-F14]`; tests are tiered. |
| Tooth 1 identity test | amend | Move it to a workspace identity tier that runs at commit on any member edit; pin regime-free manifests `[relayed-V L07-F10, L06-F9]`. |
| Tooth 2 import contracts | amend | Red on day one until identity composition leaves the run layer. The ast-grep rule must match the call node, since both sites use `from importlib import import_module` `[measured, checkpoint.py:6]`. |
| Tooth 3 spec hook | holds | Still an operator ask on `.pre-commit-config.yaml`. |
| Tooth 4 ports-free series | amend | The name test is the doctrine's own. Add the plugin-internal disposition of module 57 for series strategy Protocols such as `LatentUsageEstimator` `[relayed-V L10-F15]`. |
| Teeth, missing | missing | A collected-test floor across every tests root; an export-origin probe; a residue ratchet; a zero-forwarder count at every milestone tip. |
| P0 | reshape | Becomes flush, freeze and charter. The debt seams fold into the first two build milestones. |
| P1 | amend | Hub list widens to the stage driver, conftest, `repo_paths`, `ship_scope`, `logger_smoke`, the import budget, the CI pytest line, the Containerfile and release-please. Ceiling baseline 1256. Pin reconciliation moves to the pod milestone as a launch-door event `[relayed-V L02-F5]`. |
| P2 | wrong | Not a pure rename. Nine modules, not six, plus a `register.d1_provenance` edge; the three-symbol leaf cut is insufficient; six reader members change bytes `[relayed-V L02-F6, F7, F9]`. |
| P3a | reshape | The scientific ports are new Protocols, each with a fake and a contract test. Add a census of exact-type collaborator pins. H bindings are one batched re-pin `[relayed-V-only L03]`. |
| P3b | reshape | Five loops, two cells, phase tuple with one admission per phase start, restated proofs, and three operator rulings before the gate. |
| P3c | reshape | Neither machinery absorbs the other. Each has a guard the other lacks `[relayed-V L02-F1]`. The proof is a stall exit against a fake Provider, not controller death alone. |
| P4 | holds | No E module is among the 54 members or their 22 direct imports `[relayed-V L06]`. Add the scripts review engine and the E config and checkpoint modules from kernel. |
| P5 | reshape | Register, review, file and gate split per file. The H re-pin involvement is real unless the lifecycle split and the `verified_git_bundle` move land earlier. Whether sealed D1 evidence exists is unknown. |
| P6 | amend | 54 members; it carries the closing pin, not the only pin. |
| P7 | amend | Add the closing identity pin, the Containerfile and image regime, the scripts touch-set, and the plan-versus-actual audit. Forwarders are already zero at every tip. |
| Risks | stale | Row 1's mitigation cites T1. Add the eleven risks of section 6. |

---

## 3. Target topology in rosette vocabulary

### 3.1 Two scopes, two centers

| scope | center | why |
| --- | --- | --- |
| repository | the science: the shared class 1 numerics in `entropic-core` plus the series cells | the repo CLAUDE.md north star is the predictor; the algorithm and its parameters are the operator's |
| platform member | the step engine | R6 makes the platform a library for training machinery; its differentiated value is what sits around the optimizer step |

The comb is polycentric at repository scope by design. Forcing one center would be architecture
by symmetry (architecture-investment.md:97).

### 3.2 Platform member: center, ring cells, planes

Port groupings below are `[inferred]`; the P3a design gate fixes them. Counts follow the rosette
meter: a seventh edge means a conflated port or a cell to split.

| cell | kind | ports it owns | touches planes | source finding |
| --- | --- | --- | --- | --- |
| engine | center | `ModelFactory`, `Objective`, `Source`, `PhasePlan`, `LearningRateLaw`, `StepHook` | observability, meter | `[relayed-V L01-F3, F4, F9, F14]` |
| custody | ring | `Journal` with append and seal, `CheckpointPayload`, `PhaseStart` admission covering fresh, resume, transition and hand-off | storage | `[relayed-V L01-F5, F12; verifier note: split the cell]` |
| pod | ring | `Workload`, `ControlLedger`, `Guard`; a pure lifecycle FSM with `Succeeded`, `Failed`, `ExhaustedPending` | infra, storage | `[relayed-V L02-F10, F15]` |
| register | ring | `RunClassAdmission`, `Roster`, `Preregistration` | infra for git facts | `[relayed-V L04-F8]` |
| capture and analyze | ring | `TelemetrySchema` with a typed record envelope, `AnalysisStrategy` as reader plus design schema plus observation adapter | storage | `[relayed-V L05-F7, L04-F10]` |
| review, file, gate | ring | arm set as a series `StrEnum` parameter, `FilingRoles` cut thin, a factory-only filed-evidence mint | infra for isolation | `[relayed-V L04-F13, F15, F6]` |
| sources | ring | curriculum family and the refusing environment family behind `Source` | storage | plan R9 |

| plane | thin port every cell touches | adapter-side ports | notes |
| --- | --- | --- | --- |
| observability | `MetricSink` emit with a typed envelope | `Tracker` mirror port with MLflow and a fake | decorates rather than contracts; mirrors outcome is a value |
| storage | `ArtifactStore`, capability-shaped, immutable no-overwrite | `ObjectStore`, DVC and cloud verifiers | do not promote D1's DVC-named ports verbatim `[relayed-V-only L05]` |
| infra | `Provider`, `Billing` | `ProcessRunner`, `GitFacts` as small Protocols, `Isolation` | 18 Protocols collapse to about 4 `[relayed-V L03-F11]` |
| kernel values | none; zero-carry types | none | `Sha256`, `GitCommit`, `ContentHash`, `ArtifactIdentity`, `ReaderIdentity`, `Clock` value |

### 3.3 Measured walls and their contracts

| wall | evidence that it is measured | contract directory | owner |
| --- | --- | --- | --- |
| engine to series | E, D and H each implement every scientific capability concretely `[relayed-V L03-F2]` | `contracts/engine-series/`: the series bundle and `PLATFORM_API` | platform |
| engine to pod | D1 has a Protocol called after each durable checkpoint; H has `on_checkpoint`, `stop_after_seal`, `publication_timing` `[relayed-V-only L01]` `ladder/run/trainer/d1_segment_engine.py:58-68`, `ladder/run/runner.py:449-452` | `contracts/engine-pod/`: checkpoint published, stop request | engine |
| pod to host | the hashed launch manifest, request and terminal marker `[relayed-V L02-F5]` | `contracts/pod-host/`: launch door, request bytes, terminal marker, under the pod-to-host wire ladder | pod |
| custody to readers | readers construct the engine at four sites and replay the sampler `[relayed-V L01-F13]` | `contracts/custody-reader/`: seal chain, root kind, `ReaderIdentity` | custody |
| evidence to series | arm set, alias id set and schema id are the three series parameters `[relayed-V L04-F15]` | `contracts/evidence-series/` | platform |
| spec walls | `platform.toml` and one `series.toml` per `SeriesId` member | digest-bound specs | cli registry |

Thin walls, folder plus import direction only: the protocol order register, capture, run,
analyze, review, file, gate, which holds once seven misfiled modules move `[relayed-V L04-F20]`.

### 3.4 Composition root and workspace members

| member | role | depends on | notes |
| --- | --- | --- | --- |
| `entropic-kernel` | zero-carry values, strict codec, two named canonical schemes, digest types, spec loader, paths | stdlib only | stdlib-only requires `config.py`, `checkpoint.py` and `logging.py` to leave `[relayed-V L09-F15]`; a `testing` extra carries the pytest plugin `[inferred]` |
| `entropic-core` | shared class 1 numerics: `substrate.blocks`, `geometry.kappa`, `latent.metric`, `latent.dynamics`, `observation.contracts`, `corpus.batching`, `probe` | kernel, torch | portless; hexagons stop at the engine wall `[relayed-V-only L09]` |
| `entropic-platform` | the cells and planes of 3.2; `ports.py` as the one series-facing surface; `testing` with fakes, a toy series and contract suites | kernel, core | per-cell `ports.py` behind the one public module `[relayed-V L08-F22]` |
| `entropic-series-e`, `-d`, `-h` | plugin cells: implement platform ports, own none | platform | strategy Protocols internal to a series take the plugin-internal disposition |
| `entropic-cli` | the one composition root: registry, `SeriesId`, command service, every process main, reader identity composition | everything | the only place string imports are legal |
| residue `entropic_substrate` | migration vehicle only; the root project until the last milestone | everything landed | no new member may import it; module count ratchets to zero `[inferred]` |

Recommendation on an eighth member: none. The pytest plugin lives in `entropic_kernel.testing`
behind an extra, so every member resolves it and the copy test passes `[inferred, answering
relayed-V L07 note]`.

### 3.5 Where each current subpackage lands

File counts are `[measured]`: 401 scout records equal 401 files in `src_facts.json`. Target
columns are `[scout]`: each scout's free-text `plan_target`, classified here by a prefix regex,
unverified. K kernel, C core, P platform, E, D, H series, X cli, S split, del delete, dis
dissolved package file. The last column is this synthesis's recommendation `[inferred]`.

| current subpackage | files | loc | scout targets | recommended landing |
| --- | --- | --- | --- | --- |
| package root | 2 | 30 | X2 | residue root until the last milestone; version literal stays release-please bound |
| composition top | 5 | 657 | H3 X1 S1 | four are reader members; hashed handler halves to series-h commands, the envelope to the cli command service |
| composition/entrypoints | 10 | 3063 | S4 P2 X2 E1 H1 | every process main to cli; wave monitor reads the capture ledger |
| composition/wiring | 13 | 2845 | D6 S4 X3 | D1 wiring to the series-d plugin; registry and command service to cli |
| core top | 1 | 1 | dis1 | dissolves |
| core/corpus | 4 | 401 | D2 S1 dis1 | `batching` stays shared core |
| core/geometry | 3 | 435 | E2 dis1 | `kappa` stays shared core; scouts differ |
| core/latent | 13 | 2192 | D9 C3 S1 | `metric` and `dynamics` shared core; D latent science to series-d |
| core/observation | 5 | 552 | D4 dis1 | `contracts` shared core; the codec port lifts to platform |
| core/probe | 4 | 598 | C1 H1 S1 dis1 | shared core |
| core/sentence | 19 | 2723 | H13 S4 C1 P1 | series-h; the only core subpackage free to leave whole `[relayed-V L09-F16]` |
| core/substrate | 5 | 1896 | E4 S1 | `blocks` shared core with a frozen `ModelConfig`; Fisher init to series-e; scouts differ |
| historical top | 1 | 1 | del1 | deleted at close |
| historical/d_ladder | 13 | 2468 | P5 S4 D3 del1 | by the D2 to D5 ruling: retire, or bind to the one engine |
| historical/e_series | 3 | 1411 | E1 S1 del1 | loops retire into the engine; adapters to series-e |
| kernel top | 3 | 906 | S2 dis1 | `artifacts` to kernel; `lifecycle` splits, values to kernel and FSM to platform protocol |
| kernel/codec | 2 | 100 | K1 dis1 | kernel |
| kernel/evidence_class | 3 | 815 | P2 dis1 | `EvidenceClass` to kernel; resource provenance to the platform meter plane |
| kernel/identity | 3 | 402 | K1 S1 dis1 | `d1_primitives` splits: generic helpers to kernel, D1 vocabulary to series-d |
| kernel/runtime | 7 | 969 | K2 P2 S2 dis1 | paths to kernel; logging to observability; config and checkpoint out to core and series-e |
| kernel/verdict | 6 | 1074 | D4 S1 dis1 | series-d |
| ladder top | 1 | 1 | del1 | deleted at close |
| ladder/analyze | 30 | 10564 | S13 D7 P5 H4 E1 | 2x2 terms, seed sizing, intervals to platform analyze; per-series readers and decisions to series |
| ladder/capture top | 2 | 236 | P1 H1 | `capture.py` is a source-reader member; custody cell |
| ladder/capture/adapters | 7 | 2145 | P3 D2 S2 | observability plane; the unwired strict D1 chain by ruling |
| ladder/capture/custody | 21 | 4942 | D8 P6 S3 K1 C1 H1 dis1 | content-addressed custody to the storage plane; D1 grammar to series-d |
| ladder/capture/ledger | 5 | 1431 | D2 P1 S1 dis1 | lifecycle repository to platform; D1 process vocabulary to series-d |
| ladder/capture/ports | 3 | 229 | P2 S1 | observability plane; the E kappa extractor to series-e |
| ladder/capture/reader | 8 | 2223 | S3 D2 P1 H1 del1 | reader registry to platform; `sinks.py` dissolves into its owners |
| ladder/capture/records | 13 | 3621 | D11 S1 del1 | series-d; the journal's fsync half to custody |
| ladder/capture/resource | 10 | 3062 | P8 S1 dis1 | meter plane |
| ladder/file | 13 | 4138 | D7 P3 S3 | thin filing skeleton to platform; D1 roles to series-d |
| ladder/gate | 4 | 602 | D1 S1 del1 dis1 | gate mint to platform; `d1_verdict.py` deleted |
| ladder/register | 20 | 4933 | D9 S5 P3 H1 del2 | admission, roster, preregistration to platform; `h1_pilot`, `h_preservation` to series-h |
| ladder/review | 14 | 5661 | S6 P5 D3 | alias custody generic over the arm set to platform; plus the scripts review engine |
| ladder/run top | 38 | 15183 | S20 H9 P7 C1 dis1 | the H engine: custody and step halves to platform, science to series-h, `worker_*` to pod |
| ladder/run/campaign | 16 | 4936 | S8 D5 P3 | pod cell and series-d workload |
| ladder/run/ckptsync | 6 | 1005 | P3 D3 | DVC sync to custody; coordinators to series-d `[relayed-V L02-F22]` |
| ladder/run/closure | 7 | 1688 | S4 P2 dis1 | billing settlement to the pod cell |
| ladder/run/construction | 7 | 1266 | D3 S2 P1 dis1 | series `ModelFactory` adapters |
| ladder/run/provider | 7 | 1464 | P4 S2 dis1 | infra plane; one RunPod adapter |
| ladder/run/resource | 2 | 266 | P1 dis1 | meter plane |
| ladder/run/trainer | 18 | 4072 | S7 P6 D4 del1 | custody and engine; D1 type pins behind the D objective adapter |
| ladder/run/transport | 12 | 3834 | P7 D4 S1 | storage plane and pod cell |
| ladder/run/values | 2 | 150 | S1 dis1 | leaf values to the pod cell |
| ladder/run/worker | 10 | 1890 | P5 S2 H2 dis1 | pod cell; `d1_predecessor_loader` to series-d gate |
| total | 401 | 103140 | S115 D99 P89 H37 dis22 del10 E9 X8 C7 K5 | 115 split records are the real cost centre |

Also outside `src` and outside every planned member: 13 files under `scripts/` reference
`entropic_substrate`, including the H0 K1 to K7 gate driver and a 1318-line arm-blind review
engine with a per-rung registry `[relayed-V-only L04, L05]`. They need homes: platform review
plus a series-e spec, and a series-h gate driver.

---

## 4. Milestones as horizons

### 4.1 Rules that order them

| rule | reason | evidence |
| --- | --- | --- |
| identity door before any reader member moves | the identity string-imports upward and hashes names; a rename out of the `entropic_substrate.` prefix breaks it | `checkpoint.py:139`, `:144` `[measured]` |
| bottom-up moves: a module moves only when everything it imports has moved or dies in the same milestone | new members may never import the residue | `[inferred]`; the plan's layering, plan:221 |
| unify before the series move | split files are written once at their final location by the unification itself | plan:339-343 ordering, kept |
| zero forwarders at every milestone tip | the brief forbids stubs; the compat policy makes internal consumers a same-change cutover; forwarders may exist only between a lane tip and the zip rewrite | policies card "Compat windows"; the w2 ruling in project memory |
| one conductor re-pin per member-touching milestone | re-pins are routine: 41 logged in 61 days `[relayed-V L06 note]`; the door event is counted once | `[relayed-V L06-F2]` |
| import roots rename to `entropic_<member>` as each file lands | each file moves once; the namespace alternative keeps bytes stable but leaves every file to rename later and worsens the export hazard | `[relayed-V L07-F20]`; the opposite view is `[relayed-V-only L08]`; operator question 3 |

### 4.2 The horizons

Each row: done enough when, oracle, depends on, door, reader re-pin, tests piece.

**M0. Flush, freeze and charter.** Serial. No source bytes.

| field | content |
| --- | --- |
| done enough when | every unmerged commit is reachable from a pushed branch or tag; the freeze predicate, the north star and the horizon for this initiative are in the repo CLAUDE.md; the sprint manifest is ratified with corrected H1 statuses; shelf notes exist for L4e and the unwritten DH-35 fix |
| oracle | `git for-each-ref --contains 35b0b687` lists a `refs/heads` or `refs/remotes` ref; `git cherry origin/main <ref>` classification recorded for every ref before any delete; a repo-tier test asserts no `results/h1-*.dvc` pointer and blank preregistration identity rows while the manifest is open |
| depends on | operator rulings 1 and 2 of section 7 |
| door | none; it records the T1 replacement |
| reader re-pin | none |
| tests piece | the freeze test; L4e golden and FOR-1713 pins are shelved or landed by ruling |

**M1. Rails before motion.** Three parallel lanes: test rails, tool rails, deletions. Touches no
reader-member byte.

| field | content |
| --- | --- |
| done enough when | a moved or unregistered test turns a standing gate red; the stage table travels with the tree; the import rewriter handles all three import forms and string patch targets on a dry run; dead paths the operator ruled dead are gone |
| oracle | `just check` and the pre-push stages green at a pinned SHA; `tools/regime_rows.py` output byte-identical before and after; mutation proofs recorded for the collected-test floor and the export-origin probe |
| depends on | M0 |
| door | none |
| reader re-pin | none, by construction; the oracle proves it |
| tests piece | collected-test floor over every tests root; conftest jobs become a pytest plugin still loaded from the root; a test that every collected file is selected by at least one stage; one member-table source for `suite_stages`, `ship_scope`, `logger_smoke`, `dead_code_report` |

**M2. Identity door.** Serial rail. Arch gate. One conductor door commit.

| field | content |
| --- | --- |
| done enough when | no module outside the composition root string-imports; each layer package declares its reader manifest in code; the root composes one frozen `ReaderIdentity` and hands it to every reader; the pinned literal is a regime-free manifest digest per reader; test paths are gone from member comments; the source attestation prefixes derive from one table and fail on any empty prefix |
| oracle | the identity tier test goes red on a one-byte member edit, on a manifest shape edit and on a re-ordered composition, and green on revert; an architecture test finds zero `import_module` call nodes outside composition; a read-old row carries before and after digests; each of the 22 direct non-member imports is a member or a reasoned exclusion |
| depends on | M1 |
| door | reader identity function; label bump to `h-checkpoint-reader-v3` and `h-source-reader-v4` recommended, operator-ruled |
| reader re-pin | the door itself; batches every pending member-comment and guard edit |
| tests piece | `test_identity_epoch.py` moves to a workspace identity tier that the commit stage runs on any member path; the mapping-coverage test leaves the identity file |

**M3. Workspace skeleton and spec door.** Serial. Hub owner for the root manifest and lockfile.

| field | content |
| --- | --- |
| done enough when | seven members plus the residue resolve under one lockfile; each empty member's gate runs from a bare directory; the strict spec loader refuses duplicates, unknown keys, non-finite values and datetimes; `platform.toml` carries both runpodctl pins truthfully named and the image digest; the residue-import ban is in force |
| oracle | a changed pin turns the identity tier red; a malformed spec fails the loader test; `uv sync --all-packages` from a fresh worktree; a `new members never import entropic_substrate` contract red under mutation; the nested CLAUDE.md sentinel measurement, PRESENT or ABSENT, recorded |
| depends on | M2 |
| door | configuration |
| reader re-pin | none if the loader lands at its final kernel path with no member edit |
| tests piece | member `tests/` roots registered with the floor gate; `repo_paths` splits into `WORKSPACE_ROOT` and `MEMBER_ROOT`; the Containerfile COPY list tied to the attestation table by a test |

**M4. Kernel and core members.** Serial: kernel, then core. Lanes inside for seam adoption at
non-member sites.

| field | content |
| --- | --- |
| done enough when | `entropic_kernel` imports only the standard library; one `Sha256` type, one `GitCommit` type, one bound field reader and two named canonical schemes exist and the non-member clone definitions are deleted; `lifecycle` is split into kernel values and a residue FSM; shared class 1 numerics live in `entropic_core` with no Protocol |
| oracle | the kernel gate passes in an environment without torch, yaml or loguru; golden vectors per canonical scheme; the 2012 `pytest.raises` match sites stay green; clone scan count recorded before and after; residue module count ratchets down |
| depends on | M3 |
| door | none; the two schemes are pinned, never merged |
| reader re-pin | yes: `kernel.artifacts`, `kernel.lifecycle`, two `core.latent`, one `core.probe` keys, and 27 member import lines |
| tests piece | `tests/unit/kernel` and the shared `tests/unit/core` directories move with their subjects; `helpers.py` splits; the pytest plugin lands in `entropic_kernel.testing` |

**M5. Platform planes and the shared pod kernel.** Three file-disjoint lanes: infra, storage,
observability. One conductor zip, rewrite and re-pin.

| field | content |
| --- | --- |
| done enough when | the nine D1-named shared modules are series-free leaves in the platform; `verified_git_bundle` and its runner no longer reach the D1 trainer; `ObjectStore`, `ArtifactStore`, `GitFacts`, `ProcessRunner` and `MetricSink` each ship a fake and a contract test; `sinks.py` is dissolved; no host pod file imports torch for a key-set check |
| oracle | static import closure of the RunPod adapter and of the H worker excludes every series-D module and, for host files, torch; contract suite green over each real adapter and its fake; `just ship-it` live gate green |
| depends on | M4 |
| door | none |
| reader re-pin | yes: six pod members and the members importing seven capture modules |
| tests piece | the hand-enumerated MetricSink contract file becomes the parametrized contract suite; the 2136-line `test_sinks.py` hub splits by owner; string monkeypatch targets rewritten with each move |

**M6. Ports, registry and the testing kit.** Serial. Arch gate.

| field | content |
| --- | --- |
| done enough when | all 97 Protocols carry a disposition; each surviving platform port has a fake, a contract test and a named second option; the registry binds E, D and H in place through typed specs; a toy series runs through every port; `run_start` records the resolved registry and spec digests |
| oracle | registry tests for the five fitness functions; a `tests/typing/` fixture per typestate; each series tiny smoke equal through the admissible reader with a named excluded-field list; the census of exact-type collaborator pins reads zero |
| depends on | M5 |
| door | `run_start` wire; `PLATFORM_API` |
| reader re-pin | yes, one batched commit for every H binding |
| tests piece | platform tests stop importing D1 fixture hubs and run on the toy series; cross-member test edges reach zero and the budget test becomes a path-keyed ban |

**M7. One custody, one engine.** Serial. Its own design gate. Staged E, then D1, then H.

| field | content |
| --- | --- |
| done enough when | one run function, linear and total, drives E, D1 and H as phase tuples; one sealed chain with a root kind and typed payload slots is the only checkpoint written; the five loops are deleted or explicitly retired by ruling; readers build the engine with no sink, pod or journal adapter |
| oracle | per series: decoded payload tensors, optimizer moments, RNG and source snapshot equal to the retired trainer's with `reader`, `measurement` and wall fields excluded; resume from a seal reproduces the next-batch order digest; telemetry equal through the admissible reader; the engine file passes the ceiling |
| depends on | M6; operator rulings on durability cadence, evaluate and checkpoint order, and E RNG custody |
| door | checkpoint custody, `platform-training-checkpoint-v3`; read-old adapters for E and D1 only |
| reader re-pin | yes; the commit carries before and after digests |
| tests piece | engine tests run on fakes; per-series fixtures pin the role-partition order; an injectable clock replaces wall reads in H |

**M8. One pod machinery.** Serial after M7. Arch gate.

| field | content |
| --- | --- |
| done enough when | D1 and H launch through one lifecycle FSM, one Provider adapter and two guards per pod; teardown is verified by observed absence; pending billing ends as `ExhaustedPending` with a re-entry; a single run is a pair of one cell if ruled; H observability files leave the pod through object custody after `finish` |
| oracle | against the shipped fake Provider: a pod with no progress inside its sealed budget ends `ExhaustedPending` and is torn down; a dead controller's pod is deleted by the guard; an uncertain create is reconciled; a run with the tracker refused still yields an admissible `metrics.jsonl` |
| depends on | M7 |
| door | launch door v6: provider deadline, one runpodctl pin, opaque request bytes, terminal marker |
| reader re-pin | yes: eight pod-side members |
| tests piece | private provider fakes in four test files collapse onto the kit fake; the transient-code text scan becomes an enum test |

**M9. Evidence protocol split.** One lane beside the M7 to M8 rail, after M6. Arch gate.

| field | content |
| --- | --- |
| done enough when | register, analyze, review, file and gate form a DAG; filed evidence is factory-only and the gate mints from it; alias custody, the attempt ledger and the 2x2 terms are generic over a series arm set; the scripts review engine has a home |
| oracle | a layers contract over the protocol subpackages red under mutation; a compile-fail fixture for hand-built filed evidence; replay equality on every retained D1 fixture; the D golden flywheels green |
| depends on | M6; the M4 lifecycle split; the operator ruling on sealed D1 evidence and on the D2 to D5 lane |
| door | D reader-identity field family and D1 capture manifests |
| reader re-pin | none for H when M4 and M5 did their part; one D-side re-pin |
| tests piece | the 18-entry sentinel map rewritten once; facade-importing tests re-pointed |

**M10. Series members.** Fanout of three lanes, zipped e, then d, then h.

| field | content |
| --- | --- |
| done enough when | each series member holds only adapters, specs, configs and tests; no series imports another; each member's gate is green from a bare directory |
| oracle | per-series tiny tracer or smoke equal through the admissible reader; the member copy test; residue holds composition only |
| depends on | M8 and M9 |
| door | reader identity keys for `core.sentence` and the remaining H run members |
| reader re-pin | yes, in the h lane only, by the conductor after the zip |
| tests piece | remaining unit directories move with their subjects; data-backed H integration tests carry a named `data` marker the copy test deselects |

**M11. Composition root, teeth and close.** Serial.

| field | content |
| --- | --- |
| done enough when | the residue package is deleted; `historical/` and `entrypoints.env` are gone; every contract and tooth is in force; the root CLAUDE.md is the bearing, the member map and the cross-member rules; the plan-versus-actual audit is filed; the freeze is lifted by a filed ruling |
| oracle | each import-linter contract red under a one-line mutation and green on revert, recorded; residue and forwarder counts are zero; the closing identity pin and its read-old row; a fresh image builds and its regime row is captured after the freeze lift, under an operator GO |
| depends on | M10 |
| door | image regime; closing identity pin |
| reader re-pin | the closing pin |
| tests piece | workspace tier holds architecture, repo, e2e, live, identity and cross-member scans; the four strict-JSON adoption scans become one ast-grep rule |

### 4.3 Rails and lanes

| shape | milestones |
| --- | --- |
| serial rails | M0, M2, M3, M4 kernel then core, M6, M7, M8, M11 |
| parallel lanes inside one milestone | M1 three lanes; M5 three lanes; M10 three lanes |
| parallel across milestones | M9 beside M7 then M8, after M6, only when the conductor has measured touch-set disjointness from the M6 inventory; same axis means stop `[inferred]` |
| conductor-only | every re-pin commit, every zip rewrite, every hub file |

---

## 5. The tests refactor as a first-class part of the plan

**Placement rule.** A unit test has no mapping of its own: its destination is its mirrored src
directory's destination. Only the split list is an exception, and a multi-subject test splits in
the same commit as its subject `[relayed-V L07-F1, verifier note]`.

| piece | what | when | evidence |
| --- | --- | --- | --- |
| collected-test floor | a standing gate that enumerates every tests root and fails when a stage's collected count drops or a file is selected by no stage | M1, before any move | `scripts/suite_stages.py:75-82` literals `[measured]`; `[relayed-V-only L07]` |
| export-origin probe | the post-commit export refuses any module whose `__file__` is outside the export | M1 | `[relayed-V L07-F7]` |
| table travels with the tree | the export runs its own stage table; a stack spanning the layout flip fails closed today | M1 | `[relayed-V L07-F8]` |
| pytest plugin | lock, markers, git isolation, RNG and logging leave `tests/conftest.py`; torch determinism registers only where torch is declared | M1 in place, M4 to `entropic_kernel.testing` | `[relayed-V L07-F5, F6]` |
| workspace and member roots | `repo_paths` offers both; each importer picks one | M3 | 78 importers `[relayed-V L07-F3]` |
| canonical rootdir | member-rooted, per R5; the workspace gate runs members sequentially inside one suite lock | M3 | `[relayed-V L07 note, F17]` |
| testing kit | port fakes, a toy series and contract suites in `entropic_platform.testing`; 30 or more per-file doubles collapse | M5 and M6 | `[relayed-V-only L03]` |
| edge budget to a ban | cross-member test-to-test edges reach zero; the ratchet becomes a path-keyed ban over every tests root | by M6 | `test_test_import_budget.py:31` `[measured]` |
| string targets | 43 string monkeypatch targets in 11 files and 5 source-path reads are rewritten with each move; an architecture test asserts each target resolves to its defining module | every move | `[relayed-V-only L02]` |
| identity tier | the door test runs on any member path at commit | M2 | `[relayed-V L07-F10]` |
| workspace tier | architecture, repo, e2e, live, cross-member integration and policy scans stay at the root | M11 shape, populated as members land | `[relayed-V L07-F13]` |
| data marker | four integration files read DVC fixtures; the copy test deselects them by name, never by skip | M10 | `[relayed-V L07-F15]` |
| message pins | 2012 `pytest.raises` match sites in 348 files bound the codec seam; messages stay parameterized by label and error type | M4 | `[relayed-V-only L09]` |
| path-anchored config | the basedpyright hook regex, ruff excludes, the secrets baseline and three just recipes key on `src/` and `tests/` | M3; the hook file is an operator ask | `[relayed-V-only L07]` |
| ship scope | `INFRA_PATTERNS` move with the files; a test asserts each pattern matches a tracked path | every move of a matched file | `[relayed-V L08-F14]` |
| markers | folder is kind; `slow` is the only cost marker; CI selects by folder | M1, by ruling | `[relayed-L L08 code judo]` |

Counts to carry, all `[measured]` this session: unit 400, repo 35, integration 7, architecture
6, e2e 6, live 6, fault 2, support modules 19.

---

## 6. Top risks

| # | risk | mitigation | the test that proves the mitigation |
| --- | --- | --- | --- |
| 1 | an H1 seal, capture or pod opens under an intermediate identity | the freeze predicate, recorded in the manifest and enforced | the repo-tier freeze test of M0, red when a `results/h1-*.dvc` pointer or a filled preregistration identity row appears while the manifest is open |
| 2 | moved tests fall out of every stage and everything stays green | the collected-test floor | mutation: move one test under `packages/` without registering it; the gate goes red |
| 3 | the post-commit record certifies the live checkout under editable installs | the export-origin probe | mutation: point one member at the live path; the probe refuses the record |
| 4 | engine unification changes what a resume restores | restated proofs and a resume test per payload slot | per-series decoded-payload equality and next-batch order digest equality after resume |
| 5 | the merged pod machinery drops a guard and pods idle-burn | two guards per pod, a stall exit, verified absence | fake Provider contract test: stalled pod ends `ExhaustedPending` and is torn down; dead controller's pod is deleted |
| 6 | making the tracker optional loses authoritative files | H files leave through object custody after `finish` | a run with the tracker refused still captures a `metrics.jsonl` that `final_h_run_end` admits |
| 7 | the codec or digest seam silently changes sealed digests | two named schemes, never merged | golden vectors per scheme; replay equality over captured H0 JSONL and retained D1 fixtures |
| 8 | code is placed outside the member set to dodge re-pins and the identity narrows | one reviewed list of direct non-member imports | the M2 test: each of the 22 is a member or a reasoned exclusion |
| 9 | source attestation goes false-green when the layout guards are rewritten | prefixes from one table; fail on any empty prefix | mutation: drop one prefix; verification fails loud. A second test ties the Containerfile COPY list to the table |
| 10 | forwarders or string patch targets go inert | zero forwarders at each tip; resolved-target test | a forwarder count test; the string-target architecture test |
| 11 | in-flight work is lost at branch and worktree cleanup | durable refs before any prune; patch-id classification | `git for-each-ref --contains 35b0b687` lists a branch; `git cherry` output filed per ref |

Unknowns that size risks 1, 7 and 11: whether sealed D1 evidence exists, whether any H1 capture
exists only on the remote, and where `results/h1-dev-quote-v4.499aca.dvc` is
`[relayed-V-only L04, L10]`.

---

## 7. Questions only the operator can answer, ranked

| # | question | recommendation | what it changes |
| --- | --- | --- | --- |
| 1 | Does the freeze predicate replace T1, including no DH-34 quote pods until the closing pin? | Yes. | Every milestone's admissibility; without it risk 1 is unbounded. |
| 2 | L4d at 35b0b687: land before motion, or shelve? Why was PR 435 closed? | Push a durable branch now either way. Land it as the last pre-refactor merge if the closure was not a design objection: it is two commits behind main and its hunks sit in files the refactor splits. If no answer, shelve with a state note and proceed. | Whether M7's design gate sees the evaluation seam as measured code, and whether a later hand port of about 80 lines is needed `[relayed-V L10-F5]`. This recommendation departs from your stated deferral of rebases, so it is yours to rule. |
| 3 | Import roots: rename to `entropic_<member>` as each file lands, or keep `entropic_substrate` as a namespace and rename once at the end? | Rename as each file lands. One move per file, routine conductor re-pins under the freeze. | Under the namespace choice M4 to M10 re-pin on body edits only, at the price of a second move for every file and unmeasured tool support. |
| 4 | D2 reshaped: member sets stay in code, composed at the root, pinned regime-free, with a label bump at M2? | Yes to all four. | The shape of M2 and whether any spec carries module lists. |
| 5 | Does sealed D1 evidence exist that must still replay? Is the D2 to D5 lane alive? What runs E field pre-training after `historical/` retires? | Retire D2 to D5 under R3 unless a capture exists; extract the thin skeleton from D1 custody plus the D2 to D5 shape. | M9's scope, three L04 recommendations, and which of five loops M7 must bind. |
| 6 | Engine doors: durability cadence, evaluate before or after the seal, E RNG custody. | Per-step append with a seal-time digest; seal then evaluate bound to the seal; E adopts full custody for new writes with a read-old adapter. | The checkpoint wire shape of M7. |
| 7 | Pod doors: one launch door v6 covering a provider deadline, one runpodctl pin, opaque request bytes and pair-of-one? D1 onto H's one-way request? Pending billing as `ExhaustedPending`? Stall budget as a sealed field? | Yes to each; measure `--terminate-after` and `billing pods` on 3.0.1 before the gate. | M8's door count and whether the D1 handshake and mailbox worker are deleted. |
| 8 | C2: fix the courier before any second tracker, turn the startup abort into a refused mirror row afterward, one thin emit port, delete the unwired strict D1 chain? | Yes to each; scope the courier fix to H, since E pods are credential-free by ruling. | Whether C2's "mirror" claim is true, and where Trackio falls. |
| 9 | Evidence protocol: a three-identity filing minimum with a series custody profile? Does H1 file through the platform protocol? Which authority trust model? | Thin minimum; yes, H1 files through it; the factory-token model. | M9's port shapes and the H1 filing path after the sprint. |
| 10 | Kernel and core: is core a portless shared library with `core.sentence` in series-h? `ModelConfig` beside `build_block`? `Codec` as a platform port? Hand strict codec over Pydantic? | Yes to each, with the Pydantic deviation recorded. | M4's cut and whether kernel is stdlib-only. |
| 11 | Each unwired path: wire or delete? | Delete unless a constructing production caller is named. | M1's deletion lane and the Protocol inventory's starting size. |
| 12 | Tests: member-rooted rootdir, plugin in `entropic_kernel.testing`, folder-is-kind markers, one workspace CI job, the three `.pre-commit-config.yaml` edits? | Yes to each; the hook edits are asks, not lane actions. | M1 and M3. |
| 13 | Tooth 4: may a series own a strategy Protocol that is not a platform port? | Yes, under the plugin-internal disposition; the test refuses `ports` modules and cross-member Protocol imports. | Whether `LatentUsageEstimator` and four D strategy Protocols are legal. |
| 14 | Where is the debt plan, and should the detail-preservation instruments be filed as an H1 ticket now? | Commit its phases as a tracked ledger in M0; file the ticket. | P0's gate and H1's path after the freeze lifts. |

---

## Recommendation and next steps

1. Rule questions 1 to 4. They gate M0 and M2 and nothing else can start honestly without them.
2. Run M0 now: durable ref for 35b0b687, patch-id classification of every ref, shelf notes, the
   charter sections. This is the pre-work you asked for and it is independent of every other
   ruling.
3. Fold this synthesis into the plan as rung seven on a branch, with the Context tables restated
   at the sprint-start SHA, before any lane launches.
4. Launch M1's three lanes. They touch no reader member, so they are safe under any answer to
   question 3.

---

## Part two: conductor reconciliation (2026-11-27)

### Re-measured by the conductor before relaying

| claim | result | evidence |
| --- | --- | --- |
| the checkpoint reader tuple has 54 members: composition 4, core 22, kernel 2, ladder 26 | holds | `[measured]` `ast` parse of `_READER_MODULES` in `src/entropic_substrate/ladder/run/checkpoint.py` |
| both reader identities string-import their members and hash `name:digest` | holds | `[measured]` `checkpoint.py:6`, `:139`; `ladder/capture/capture.py:4`, `:44`; these are the only four `import_module` lines under `src` |
| the H checkpoint embeds the reader identity and the loader refuses a mismatch | holds | `[measured]` `checkpoint.py` payload key `reader`, and the loader's `row["reader"] != _reader_identity()` refusal |
| the largest source file is 1256 lines | holds | `[measured]` `wc -l src/entropic_substrate/ladder/run/object_worker.py` |
| the stage driver names `tests/<tier>` literals only | holds | `[measured]` `scripts/suite_stages.py` `STAGES` table |
| the L4d head 35b0b687 was held by one local fetch ref | held, now changed | `[measured]` the remote also serves it as `refs/pull/435/head` (`git ls-remote origin`); the conductor added the local branch `shelf/h1/l4d-evaluation-instruments` on 2026-11-27, not pushed |
| refuted findings are absent from part one | holds | `[measured]` L07-F11, the L03 typestate count drift, L09-F10, L09-F18 and the L09 plan-delta row on the file ceiling do not appear as claims; L09-F15 appears only in its corrected form (logging must also leave the kernel) |
| `results/h1-dev-quote-v4.499aca.dvc` cannot be found | holds | `[measured]` no file of that name under the repository root in either checkout; the DVC remote was not queried |

### Where an operator ruling already decides the point

| synthesis text | operator ruling of 2026-11-27 | effect on the plan |
| --- | --- | --- |
| question 2: land L4d before motion | in-flight H1 work is shelved with a state note and re-implemented on the new platform, not rebased | the plan keeps the shelve ruling and records the synthesis's dissent with its cost as an open decision (register row D12); the state note is `docs/h1/shelved-state-2026-11-27.md` |
| question 3 and rule 6 of section 4.1: rename every import root to `entropic_<member>` | hybrid naming: `entropic_kernel` and `entropic_platform` are their own roots; core, the series and the cli stay under `entropic_substrate` | files landing in kernel or platform rename once as they land; files landing in core, a series or the cli keep the `entropic_substrate` root, which becomes a namespace shared by five members. The residue ratchet is therefore counted by path, and the import ban reads "the two liftable roots never import `entropic_substrate`", which is exactly the lift boundary |
| section 3.4: `entropic-platform` depends on kernel and core; `corpus.batching` and `probe` listed as shared core | the core definition: core is the series-neutral science kit, and the platform never imports core | the platform depends on the kernel only. A module the platform needs fails the second question of the core membership test and lands in the platform behind a port: the generic sampler goes to `sources`, checkpoint recovery records to custody |
| D3: keep version literals, add a catalog and a define-once lint | wire schema versions reset under new contract family names at `-v1`, clean break, one epoch tag plus a replay recipe | the catalog and the lint stand as the mechanism; the reset rides the milestone that moves each record's owner, so each literal changes once |
| D5 and M7: read-old adapters for E and D1 | the same clean-break ruling: no read-old adapters anywhere | M7 writes the new family only. Old checkpoints replay at the epoch tag. The compat-window default for persisted artifacts is displaced by the ruling, and the plan names that conflict |
| section 4.2 has no fine-tuning milestone, lift recipe, or tracker order | the reach items are all in the sprint, in priority order; the fine-tuning cell and its rubric were ruled in rounds three to five | the plan adds a fine-tuning horizon after the one-engine horizon, puts the lift recipe in the closing horizon, and places Trackio after the courier fix |

### What the conductor did not verify

The lens findings carried as `[relayed-V ...]` were checked by a verifier and not re-read by the
conductor beyond the rows above. Rows carried as `[relayed-V-only ...]` have one source. Rows
carried as `[scout]` and every target column in section 3.5 are unverified classifications; the
design gate of each horizon re-measures its own touch-set before any file moves.
