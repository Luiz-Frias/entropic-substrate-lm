# p4-h-tracer — H lane migration; both H reader identities bumped

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p4-h-tracer`                                                                                                     |
| parent                                       | `mythwave/p3b-observability-rail`                                                                                          |
| mode (sprint manifest, parallel eligibility) | `tracer`                                                                                                                   |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived. Feeding the manifest `mode` here yields a silent `normal` and no Opus design gate. |

- **engine_mode:** `arch` (manifest field, machine-read by the launch composer; ruled 2026-09-18)
  | wave | `serial` |
  | `gates.liveRequired` | `true` |
  | live gate | `just ship-it` runs INSIDE the lifecycle. House rule 1: never a bare `gt submit`. |
  | gates | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}` |
  | launch | SERIAL phase: the conductor launches the repo's ported cascade DIRECTLY, `Workflow({scriptPath: <repo>/.claude/workflows/td-phase-lifecycle.cascade.js, args})`. Not a wave member. |
  | `repoToplevel` | `/Users/luizfrias/CursorAI/entropic-substrate-lm`. The MAIN checkout, never a worktree, because `advisorLockPath` derives from its basename. |
  | trunk | `refactor/hexagonal-redesign` |

## Contract surface owned

| contract                                                                                                                                            | kind     | frozen_after |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `h-source-reader-v3` :: `sentence_prediction/capture.py` `_READER_MODULES` rewritten to the post-move dotted names                                  | door     | p4           |
| `h-checkpoint-reader-v2` :: `sentence_prediction/checkpoint.py` `_reader_identity()` rewritten from a package glob to an explicit dotted-name tuple | door     | p4           |
| `docs/identity/read-old-table.md` :: the v2→v3 and v1→v2 rows with the p0-epoch digests                                                             | door     | p4           |
| `kernel/artifacts`, `kernel/lifecycle`, `core/latent/{metric,dynamics}` addresses (the reader-closure rail)                                         | contract | p13          |
| `ladder/run/worker` :: the H half, including `h_probe_command` and `h_probe_controller`                                                             | contract | p10          |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- `tools/gen_shims.py`, the landed-set fragment convention and `tests/test_legacy_coupling.py` (p3)
- `kernel/runtime/logging.py`, `ladder/capture/ports/metric_sink.py`, the sink adapters and the run-descriptor door (p3b)
- `src/entropic_substrate/paths.py` (p1)
- the identity fixtures (p0)

## Sacred invariants

The five below are quoted verbatim from `CLAUDE.md` (North star and
*Non-negotiable numerics*). They bind every phase without exception. `SI-*` ids are local to this
brief set; the text is the contract.

| id   | verbatim invariant                                                                                                          | tripwire                                                                                                                                       | guard-teeth recipe (house rule 5, both arms)                                                                                                                                                                                                                                                                                                                                                                                              |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | This phase moves no numerics file: `touch_set ∩ core/geometry = ∅`. The numerics tests must stay byte-identical.                               | CORRECTED 2026-09-19 (p0 round-C measurement, FOR-1517): for `geometry/kappa.py` the dtype assertion is VACUOUS (`_geometry_dtype()` floors to fp32 regardless of autocast state), so the load-bearing arm is an autocast-STATE spy inside the island, as `tests/test_kappa_norm.py` proves for `kappa_norm.py`; that test fix is FOR-1517's. A phase that moves a numerics module records the state-spy arm for it, never the dtype arm. |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | `touch_set ∩ core/geometry = ∅`; the κ clamp and its test move only under a phase that owns `core/`.                                           | Widen the κ clamp bound by one ulp and feed a boundary input; the κ-domain test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                                                                            |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | `touch_set ∩ core/substrate = ∅`; the load-balance test must stay byte-identical.                                                              | Detach κ from the load-balance term; the gradient-reaches-κ unit test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                                                                                      |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | This phase moves `core/sentence`, the carrier's home. No normalization line may change; the carrier magnitude test must arrive byte-identical. | Append a `UnitNorm` to the magnitude-preserving carrier's output; the carrier magnitude test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                                                               |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | This phase moves the H latent path. The latent law must arrive byte-identical.                                                                 | Substitute a `sqrt(p + eps)` for the latent law in the main path; the latent-path law test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                                                                 |

House rules from `CLAUDE.md`, verbatim headline, each binding in this phase:

| id   | verbatim house rule                                                            | tripwire                                                                                   | guard-teeth recipe                                                                                                                                     |
| ---- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| HR-1 | "Publish with `just ship-it`, never a bare `gt submit`."                       | Serial phase: `just ship-it` runs inside the lifecycle and is the only publish path.       | Inspect the lane transcript for any `gt submit` not reached through `just ship-it`; presence is a BLOCKED verdict, not a note.                         |
| HR-2 | "Probe sequentially. Never draw a conclusion from concurrent containers."      | The podman VM is 4 vCPU / 4 GB; two MLflow stacks at once make even a control appear hung. | Any container conclusion in this phase must name the single container it came from. Two concurrent stacks in the transcript invalidate the conclusion. |
| HR-3 | "Never fix flakiness by raising a timeout or a retry budget."                  | A gate that goes green only after a timeout bump.                                          | A diff that raises any timeout or retry constant is a BLOCKED verdict unless it carries the measurement showing the cause was found.                   |
| HR-4 | "Comments record measurements, with the measurement."                          | A comment in this phase's diff stating a fact with no measurement attached.                | A wrong measured claim is retracted explicitly, in the commit that fixes it. Silent overwrite is a defect.                                             |
| HR-5 | "TDD is non-negotiable, and guard-teeth prove both arms."                      | A guard landed without its RED arm demonstrated.                                           | Every guard in this phase carries the mutation listed under *Guard teeth by mutation*, performed and reverted in-branch, evidence in the PR body.      |
| HR-6 | "Secrets never enter git."                                                     | gitleaks runs at pre-commit and pre-push.                                                  | No credential, token or `.env` value in any diff of this phase.                                                                                        |
| HR-7 | "Upgrading an existing venv across the mlflow → mlflow-skinny swap breaks it." | `import mlflow` raising `ModuleNotFoundError` in a lane worktree.                          | Fix with `uv sync --all-groups --reinstall-package mlflow-skinny`, never by pinning around it.                                                         |

## Scope

First strangler tracer: the 55-module H lane leaves __legacy, plus h_probe_command and h_probe_controller and the 4 reader-closure members it digests (d1_artifact_snapshot, evidence.lifecycle, predictive_state.metric/dynamics) so both H identities fire exactly once. checkpoint.py's glob RAISES once package.parent is ladder/run; rewrite it to an explicit tuple. Detail: plan §A-p4.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- Both identities recomputed and pinned to their v3/v2 values; the read-old table carries the p0-epoch v2/v1 digests. just ship-it green with the H lane's first live MLflow run id in the PR body. Mutation: revert one of the four rail modules to __legacy and the checkpoint identity test goes RED with a resolution error. Neither value changes later.
- `just check` green on the lane tip.
- `just pc-ci` (exit gate) green on the lane tip.
- `lint-imports` green: ZERO inversions among the landed layers `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them instead.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the PR body.
- The actual diff's file set is a subset of *Allowed files*. Undeclared overlap is a HALT at the zip, not a note.
- `just ship-it` green; every PR's `gh pr view --json headRefOid` equals `git rev-parse` on its branch.
- Rollback (Q7 ruled (a)): a failed integrated gate at the zip unwinds to tip N-1 and raises DECISION-NEEDED naming the lane; per-tip re-certification localises the failure.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for correctness is a `DECISION`, not an edit.

- Does NOT narrow the h-checkpoint-reader membership. Ruling recommendation E-2: faithful restatement of the same file set at the new dotted names. Narrowing is a scientific change and is not offered.
- Does NOT move `kernel/identity` (123 eager importers, zero H consumers). That is p5.
- Does NOT move `git_environment.py` (zero H importers, two D1 importers). Ruling E-12: `kernel/runtime` at p5.
- No cross-reader resume and no automatic migration of old captures. The forward contract forbids it.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output).

- `.secrets.baseline` (STANDING RULE, conductor 2026-09-19, generalizing p0's ruling: a mechanical companion of allowed content. detect-secrets regenerates it whenever an allowed file's digest literals move or change, so its diff may contain ONLY line renumbering, removal of orphaned entries, the `generated_at` stamp, and new `Hex High Entropy String` entries whose flagged lines are sha256 / git-SHA literals inside this phase's own Allowed files; the lane verifies that in the PR body. New digest literals in TEST files carry an inline `# pragma: allowlist secret` instead of a baseline entry, so the baseline stops churning; JSON records keep baseline entries. Never a credential.)
- `.claude/workflows/landed-set/p4-h-tracer.json` — THIS phase's disjoint fragment (modules landed, targets, `legacy_edge_budget`). No other phase's fragment. `.importlinter` is NOT in this envelope: it is edited only at p0 and p13.
- `src/entropic_substrate/__legacy/sentence_prediction/**` and their shims
- `src/entropic_substrate/__legacy/h_probe_*.py` and their shims (including `h_probe_command.py` and `h_probe_controller.py`, added since the survey)
- `src/entropic_substrate/__legacy/h_runpod_transport.py` and its shim
- `src/entropic_substrate/__legacy/d1_artifact_snapshot.py` and its shim
- `src/entropic_substrate/__legacy/evidence/lifecycle.py` and its shim
- `src/entropic_substrate/__legacy/predictive_state/{metric,dynamics}.py` and their shims
- `src/entropic_substrate/__legacy/d_ladder_worker_runtime.py` (SPLIT: live half only)
- `src/entropic_substrate/core/sentence/**`
- `src/entropic_substrate/core/probe/score_gain.py`
- `src/entropic_substrate/ladder/run/{training,worker,transport,spanprobe,journal}/**`
- `src/entropic_substrate/ladder/capture/custody/h_source.py`
- `src/entropic_substrate/kernel/{artifacts,lifecycle}/**` (NARROWED 2026-09-18, HAZARD 2: dropped `runtime` — p4's own Non-goals already disclaim `git_environment.py` and p5 owns the rest of `kernel/runtime` per ruling E-12; the prior `{artifacts,lifecycle,runtime}` glob weakened the subset-of-Allowed-files envelope check against p5)
- `scripts/entrypoints.env` (NEW 2026-09-18, HAZARD 3: `ENTRYPOINT_H_WORKER` names `entropic_substrate.sentence_prediction.worker_command`, which this phase moves; edited here only to keep that entry's target valid, not to author the parameterized entrypoint source, which p10 owns as its frozen door)
- `src/entropic_substrate/composition/entrypoints/**`
- `tests/test_identity_epoch.py`
- `docs/identity/read-old-table.md`

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status                                                        | ruling                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| H first            | RULED 2026-09-17                                              | The H lane is the first strangler tracer. Measured: its external importers number 5, against `kernel.lifecycle` at 79. It is also the lowest-blast-radius first move available.                                                                                                                                                                                                                                                                                                               |
| E-7                | RULED 2026-09-17 (absorbed)                                   | `d_ladder_worker_runtime` SPLITS HERE, not at the end: `worker_execution.py:12` imports `measure_hardware, verify_source_tree` from it. Live half → `ladder/run/worker`; `verify_worker_source` → `historical/d_ladder` at p12.                                                                                                                                                                                                                                                               |
| E-2                | RATIFIED at sprint-preflight (recommendation)                 | h-checkpoint-reader-v2 membership is a FAITHFUL RESTATEMENT: same file set at the new dotted names.                                                                                                                                                                                                                                                                                                                                                                                           |
| checkpoint glob    | RULED 2026-09-17                                              | `checkpoint.py:45-64` globs `sentence_prediction/*.py`; once `package.parent` is `ladder/run` it RAISES rather than changing value. Rewrite it to an explicit dotted-name tuple, the idiom `capture.py` already uses.                                                                                                                                                                                                                                                                         |
| P-1                | RULED 2026-09-17                                              | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.                                                                                                                                                                                                                                                                             |
| O-1                | RULED 2026-09-17                                              | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                                                                                                                                                                                                                                                                                                        |
| E-4                | RULED 2026-09-17                                              | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                                                                                                                                                                                                                                                                                            |
| rewrite-on-arrival | RULED 2026-09-17                                              | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                                                                                                                                                                                                                                                                                                |
| shim form          | RULED 2026-09-17; AMENDED 2026-09-20 (operator, p3-shim-pair) | Every leaf shim is a typed PAIR: the 3-line `.py` alias (`import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`) plus a `.pyi` stub of DERIVED explicit `from <target> import <name> as <name>` re-exports, never `import *`. `tools/gen_shims.py` is the only writer (`--preflight` before any move, `--verify` at every commit touching src/); a string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green. |
| skeleton           | RULED 2026-09-17                                              | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                                                                                                                                                                                                                                                                                                    |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                         | must go RED                                                                                                                                       |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| revert one of the four reader-closure rail modules to `__legacy` | the checkpoint identity test, with a RESOLUTION error, not a digest mismatch                                                                      |
| reintroduce one old-prefix import in a landed H package          | the no-old-prefix test. Separately: this phase's `legacy_edge_budget` must be STRICTLY LESS than p3b's, and `tests/test_legacy_coupling.py` green |
| leave `checkpoint.py`'s glob in place after the move             | the checkpoint identity test raises rather than returning a new digest, which is the failure mode this phase exists to convert                    |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- FIRST PROOF of the rail. `core/sentence` training and evaluation emit through a `CompositeSink` (JSONL authoritative, MLflow best-effort) under a `TrackingRunDescriptor`.
- The H worker calls `configure_logging` at entry, never at import.
- All 8 bare `print(` sites in the H lane are routed through `logger` in the commit that moves their file. Neither is one of the two pinned raw-`print` sites.
- `sentence_prediction/journal.py` keeps its JSONL journal as the authoritative record, now written through the `MetricSink` port rather than beside it.
- `liveRequired: true`. This is the FIRST time `just ship-it` runs the MLflow container gate on the H lane. The run id goes in the PR body.
- Entry-point / allowlist promotion (CONFORMANCE-PROMOTION RULING, conductor 2026-09-21): `sentence_prediction.worker_command` (p4) is promoted through this phase's fragment key `conforming_entrypoints`. `tests/observability_conformance_config.py` stays p3b's file and is NOT in this phase's envelope; it merges every fragment present. KEYING (conductor 2026-09-21, disambiguation after p3b's L2): `conforming_entrypoints` entries are keyed by the entry point's `tools/mapping.json` dotted name (the PRE-MOVE name, since mapping.json is read-only until p13 and the frozen partition assertion compares against those seven rows) and are NEVER re-keyed before p13; only `print_allowlist` entries are re-keyed to the LANDED dotted name, because the print-ban arm derives names from file paths under the landed roots.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Split d_ladder_worker_runtime and land its live half in ladder/run/worker
2. Move the H domain modules to core/sentence
3. Move the H execution plane to ladder/run
4. Move the H probe plane including h_probe_command and h_probe_controller
5. Rewrite checkpoint _reader_identity to an explicit dotted-name tuple
6. Bump h-source-reader to v3 and h-checkpoint-reader to v2
7. Fill the read-old table rows with the p0-epoch digests
8. Route H training and evaluation through the CompositeSink
