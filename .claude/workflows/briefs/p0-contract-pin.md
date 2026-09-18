# p0-contract-pin — Contract pin: epoch, identities, sentinels, CLI/packaging, import-linter

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p0-contract-pin`                                                                                                 |
| parent                                       | `refactor/hexagonal-redesign (the sprint trunk)`                                                                           |
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

| contract                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | kind     | frozen_after |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `.importlinter` :: layers over the LANDED layers `kernel < core < ladder < historical < composition`, with `__legacy` deliberately OUTSIDE the contract until p13 + forbidden(`core.*` ↛ mlflow \| boto3 \| botocore \| dvc \| runpod \| huggingface_hub \| datasets)                                                                                                                                                                                                                                                                                             | door     | p0           |
| `tests/test_identity_epoch.py` :: h-source-reader-v2, h-checkpoint-reader-v1, the 73-entry D1 closure manifest (RETRACTION 2026-09-18: 52 was the survey's count and is wrong against this tip; `installed_d1_reader_source_files()` returns 73, measured independently by the lane and the conductor at 5464c6c+markers), `CHECKPOINT_FORMAT_VERSION`, `_TAU_ERAS`, both codec construction digests, every `*-v1`/`*-v2` literal                                                                                                                                 | door     | p0           |
| `tests/test_module_identity.py` :: `sys.modules` single-object identity for EVERY module-level `Name = object()` sentinel found by AST PREDICATE (not by name pattern), across both spellings, for all 332 modules. RETRACTED 2026-09-18 (Q5 ruled (a)): previously "`_TOKEN` singleton" — a name-pattern scope misses `_MATRIX_ATTESTATION` and `_DESIGN_ATTESTATION`, same hazard class. Count restated in the plan's `18 module-level`Name = object()`sentinels in 18 modules by AST predicate at 144a9b5 (incl.`_MATRIX_ATTESTATION`,`_DESIGN_ATTESTATION`)`. | contract | p3           |
| `tests/test_cli_surface.py` :: `--help` snapshot for `entropic_substrate.cli` and every subcommand, `-m telemetry`, `-m sentence_prediction.worker_command`; the frozen dotted-name literal set                                                                                                                                                                                                                                                                                                                                                                   | contract | p13          |
| `tests/test_packaging_identity.py` :: `packages.find` result + basedpyright / pre-commit include regexes                                                                                                                                                                                                                                                                                                                                                                                                                                                          | contract | p13          |
| `graph-bbe7976/` :: the re-derived epoch, 332 files / 326 edge-bearing modules / 1783 edges                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | contract | p0           |
| `tools/mapping.json` :: the machine input, covering every node at this epoch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | contract | p5           |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- None. This is the root rail: it owns every contract every later phase consumes.

## Sacred invariants

The five below are quoted verbatim from `CLAUDE.md` (North star and
*Non-negotiable numerics*). They bind every phase without exception. `SI-*` ids are local to this
brief set; the text is the contract.

| id   | verbatim invariant                                                                                                          | tripwire                                                                                                         | guard-teeth recipe (house rule 5, both arms)                                                                                                                       |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | This phase moves no numerics file: `touch_set ∩ core/geometry = ∅`. The numerics tests must stay byte-identical. | Drop one `autocast(enabled=False)` wrapper in `geometry/kappa.py`; the FP32-island dtype assertion must go RED. Revert in-branch, record both arms in the PR body. |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | `touch_set ∩ core/geometry = ∅`; the κ clamp and its test move only under a phase that owns `core/`.             | Widen the κ clamp bound by one ulp and feed a boundary input; the κ-domain test must go RED. Revert in-branch.                                                     |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | `touch_set ∩ core/substrate = ∅`; the load-balance test must stay byte-identical.                                | Detach κ from the load-balance term; the gradient-reaches-κ unit test must go RED. Revert in-branch.                                                               |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | `touch_set ∩ core/sentence = ∅`; carrier code is untouched by this phase.                                        | Append a `UnitNorm` to the magnitude-preserving carrier's output; the carrier magnitude test must go RED. Revert in-branch.                                        |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | `touch_set ∩ core/latent = ∅`; the latent law is untouched by this phase.                                        | Substitute a `sqrt(p + eps)` for the latent law in the main path; the latent-path law test must go RED. Revert in-branch.                                          |

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

Zero src edits. Re-derive the import graph at HEAD; restate 332 files / 326 modules / 1783 edges. Pin the reader identities, CHECKPOINT_FORMAT_VERSION, codec digests and every schema literal as fixtures. Add the sentinel-singleton test over all dotted paths. Snapshot the CLI and packaging. Commit tools/mapping.json. Add .importlinter over the landed layers, __legacy outside it. Detail: plan §C.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- just check and just ship-it green. lint-imports green over the landed layers. Mutation: one core.*→boto3 import turns the forbidden contract RED; one byte in any closure member turns the identity fixture RED; one duplicate module load turns the sentinel test RED. All three reverted in-branch.
- `just check` green on the lane tip.
- `just pc-ci` (exit gate) green on the lane tip.
- `lint-imports` green: ZERO inversions among the landed layers `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them instead.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the PR body.
- The actual diff's file set is a subset of *Allowed files*. Undeclared overlap is a HALT at the zip, not a note.
- `just ship-it` green; every PR's `gh pr view --json headRefOid` equals `git rev-parse` on its branch.
- Rollback (Q7 ruled (a)): a failed integrated gate at the zip unwinds to tip N-1 and raises DECISION-NEEDED naming the lane; per-tip re-certification localises the failure.

### Design-gate rulings (conductor, 2026-09-18, resume of wf_5a1887ce-db2)

- **D1, epoch artifact:** `graph-<sha>/` is NOT committed (217 KB, regenerable by `tools/import_graph.py`). p0 commits `docs/plans/measurements/p0-epoch.json` (HEAD SHA, file/module/edge counts, inversions, cycles, tool versions) and the identity fixtures; that path is in Allowed files. Planned commit 7 (`tools/mapping.json`) is discharged by an assertion that the tracked file covers every module of the graph, with zero edits to it.
- **D2, mapping vs contract:** `tools/mapping.json` now routes the 13 E-series / D2-D5 modules to `historical.*` and its `layers` list is `kernel, core, ladder, historical, composition`; `__legacy` is a quarantine, never a layer, in both oracles (`remap_check.py` and `lint-imports`).
- **Layer packages may exist empty at p0:** import-linter cannot name a layer whose package is absent, so p0 MAY create exactly five empty markers `src/entropic_substrate/{kernel,core,ladder,historical,composition}/__init__.py` and name each package on a bare line of the layers contract (so the package's own `__init__.py` is linted). This is the ONLY `src/` edit permitted in p0 and it moves no code; the Non-goal 'no src edit' is amended to 'no src edit except the five empty layer markers'.
- **Identity fixture regimes:** the composed H digests hash `python_version()` and `torch.__version__`, so the fixture asserts EXACTLY the source-derived components (reader module tuple, per-file sha256, schema literals, format version, codec digests) and records the composed digest per measured regime `(python, torch) -> digest` starting with the local one; an unmeasured regime fails loud naming the row to add, never skips.
- **lint-imports channel (operator ruling 2026-09-18):** `just check` gains `lint-imports` AND `.github/workflows/ci.yaml` gains a `lint-imports` step after the test job, so the contracts have a channel that can stop a merge; mutation-proven in this lane.
- **Sentinel test at p0:** with no `__legacy` namespace yet, the both-spellings arm is vacuous by construction; the double-load mutation and the exact AST-derived count (18 at afc9d09) are the load-bearing assertions and must appear in the PR body.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for correctness is a `DECISION`, not an edit.

- No `src/` edit of any kind. Zero modules move.
- No import rewrite, no shim, no `git mv`.
- Does NOT rewrite the five depth-derived path sites; that is p1.
- Does NOT touch `.pre-commit-config.yaml` (operator ask-first).
- Thoroughness beyond the stated ACCEPTANCE is not welcome.

## Allowed files

- `docs/plans/measurements/p0-epoch.json` (epoch summary record; D1 ruling)
- `src/entropic_substrate/{kernel,core,ladder,historical,composition}/__init__.py` (empty layer markers only; design-gate ruling)

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output).

- `.importlinter`
- `tests/test_identity_epoch.py`
- `tests/test_module_identity.py`
- `tests/test_cli_surface.py`
- `tests/test_packaging_identity.py`
- `pyproject.toml` (dev group: add `import-linter`)
- `justfile` (add `lint-imports` to `just check`)
- `uv.lock` (NEW 2026-09-18, DEFECT 7 carry-forward: the `import-linter` dev-group add in `pyproject.toml` changes the lockfile in the same commit)
- `.github/workflows/ci.yaml` (add the `lint-imports` step; operator ruling 2026-09-18)
- `.secrets.baseline` (CONDUCTOR RULING 2026-09-18, gate-check halt of wf_1e9deec4-d4a: mechanical companion of allowed content, same class as `uv.lock`. Permitted entries are ONLY `Hex High Entropy String` hits whose flagged lines are sha256 / git-SHA literals inside p0's own new files; verified on the lane diff: 6 entries, `docs/plans/measurements/p0-epoch.json:4` and `tests/test_identity_epoch.py:72,78,84,94,98`, all digests, no credential)
- Conductor ruling commits ONLY (message prefix `docs(refactor):`, authored by the conductor between engine runs, never by a lane agent): this brief, `.claude/workflows/briefs/p1-anchors.md`, `.claude/workflows/briefs/p2-d1-closure.md`, `docs/plans/src-hexagonal-refactor.plan.md`, `.claude/workflows/sprint-manifest.json`. They carry rulings and measured retractions and are excluded from the lane's ACCEPTANCE subset check.

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status                                            | ruling                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------ | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E-8                | RATIFIED at sprint-preflight (recommendation)     | `import-linter` enters the dev group and `lint-imports` enters `just check`. `.pre-commit-config.yaml` stays untouched unless the operator adds it.                                                                                                                                                                                                                                    |
| DN-17              | RULED 2026-09-17                                  | Packaging collapses to a pin: `where = ["src"]` already auto-discovers nested packages. Assert, do not edit.                                                                                                                                                                                                                                                                           |
| DN-P1              | RULED 2026-09-17 (conductor, implementation seam) | There is NO import-linter ignore list. `.importlinter` is edited ONLY at p0 (create) and p13 (add `__legacy`, full enforcement). From p3b to p12 it declares the layers contract over the LANDED layers with `__legacy` deliberately OUTSIDE it, so landed-layer inversions are always ZERO and strictly enforced, while quarantine edges are bounded by `legacy_edge_budget` instead. |
| P-1                | RULED 2026-09-17                                  | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.                                                                                                                                                                      |
| O-1                | RULED 2026-09-17                                  | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                                                                                                                                                                                                 |
| E-4                | RULED 2026-09-17                                  | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                                                                                                                                                                                     |
| rewrite-on-arrival | RULED 2026-09-17                                  | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                                                                                                                                                                                         |
| shim form          | RULED 2026-09-17                                  | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`. A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                                                                                                                                                                                   |
| skeleton           | RULED 2026-09-17                                  | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                                                                                                                                                                                             |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                                                           | must go RED                                                                                                   |
| -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| add `import boto3` to a `core.*` module                                                            | `lint-imports` forbidden contract                                                                             |
| flip one byte in a D1 closure member                                                               | the identity fixture                                                                                          |
| load one module under a second name                                                                | the sentinel-singleton test                                                                                   |
| emit a probe shim in the string-only `importlib.import_module` form on a deliberately illegal edge | the layers contract must be shown GREEN-BLIND; this demonstrates why the static-import shim form is mandatory |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- No routing change. p0 PINS the observability surface so later phases can move it safely.
- The two machine-consumed stdout sites stay raw `print` and are pinned by the CLI snapshot: `telemetry final-eval` and `cli eval --json`. Everything else becomes `logger` under rewrite-on-arrival.
- The snapshot records, as a measured baseline, that only the E entry points reach `logging.py` / `sinks.py` today: `cli`, `train`, `field_pretrain`, `mlflow_aggregate`, `capture_results`, `checkpoint`, `telemetry`, and 2 analysis modules.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Re-derive the import graph at HEAD and commit the epoch summary
2. Pin the three reader identities and every schema literal as fixtures
3. Add the sentinel-singleton test over all 332 dotted paths
4. Snapshot the CLI surface and the frozen dotted-name literal set
5. Assert packaging discovery and the basedpyright include regex
6. Add .importlinter over the landed layers, __legacy outside it
7. Commit tools/mapping.json covering every node at this epoch
8. Record the four p0 mutation proofs in the PR body
