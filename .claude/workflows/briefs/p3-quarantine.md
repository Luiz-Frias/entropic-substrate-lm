# p3-quarantine — `__legacy/` quarantine + generated shim layer + the landed-set fragment convention

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p3-quarantine`                                                                                                   |
| parent                                       | `mythwave/p2-d1-closure`                                                                                                   |
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

| contract                                                                                                                                                                                                                        | kind     | frozen_after |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `tools/gen_shims.py` :: one 3-line `sys.modules` alias file per old dotted LEAF path, generated from `tools/mapping.json`; fragments only ADD (DN-P1 gen_shims v2, 2026-09-18); NEVER rebinds or writes a package `__init__.py` | door     | p3           |
| the landed-set FRAGMENT CONVENTION :: `.claude/workflows/landed-set/<id>.json`, one disjoint fragment per phase from p3 on, carrying modules landed, targets and `legacy_edge_budget`                                           | contract | p13          |
| `tests/test_legacy_coupling.py` :: asserts from the AST graph (`tools/import_graph.py`) that edges between landed packages and `__legacy` are ≤ the merged budget, and that EVERY shim carries the static-import form           | contract | p13          |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- every p0 contract (identity fixtures, sentinel test, CLI snapshot, `.importlinter`)
- `src/entropic_substrate/paths.py` (p1)
- the closure retirement (p2)

## Sacred invariants

The five below are quoted verbatim from `CLAUDE.md` (North star and
*Non-negotiable numerics*). They bind every phase without exception. `SI-*` ids are local to this
brief set; the text is the contract.

| id   | verbatim invariant                                                                                                          | tripwire                                                                                                         | guard-teeth recipe (house rule 5, both arms)                                                                                                                                                                                                                                                                                                                                                                                              |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | This phase moves no numerics file: `touch_set ∩ core/geometry = ∅`. The numerics tests must stay byte-identical. | CORRECTED 2026-09-19 (p0 round-C measurement, FOR-1517): for `geometry/kappa.py` the dtype assertion is VACUOUS (`_geometry_dtype()` floors to fp32 regardless of autocast state), so the load-bearing arm is an autocast-STATE spy inside the island, as `tests/test_kappa_norm.py` proves for `kappa_norm.py`; that test fix is FOR-1517's. A phase that moves a numerics module records the state-spy arm for it, never the dtype arm. |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | `touch_set ∩ core/geometry = ∅`; the κ clamp and its test move only under a phase that owns `core/`.             | Widen the κ clamp bound by one ulp and feed a boundary input; the κ-domain test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                                                                            |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | `touch_set ∩ core/substrate = ∅`; the load-balance test must stay byte-identical.                                | Detach κ from the load-balance term; the gradient-reaches-κ unit test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                                                                                      |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | `touch_set ∩ core/sentence = ∅`; carrier code is untouched by this phase.                                        | Append a `UnitNorm` to the magnitude-preserving carrier's output; the carrier magnitude test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                                                               |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | `touch_set ∩ core/latent = ∅`; the latent law is untouched by this phase.                                        | Substitute a `sqrt(p + eps)` for the latent law in the main path; the latent-path law test must go RED. Revert in-branch.                                                                                                                                                                                                                                                                                                                 |

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

git mv the whole package into __legacy/ with zero byte edits, then generate one alias shim per old dotted LEAF path from tools/mapping.json. Package `__init__.py` files are never shimmed (RETRACTED 2026-09-18): the 12 originals are restored unmodified. Each shim static-imports its target so grimp sees the chain. Owns the fragment convention. Detail: plan §mechanism.

## ACCEPTANCE

- Every entrypoint shim (`mapping.json` `entrypoint: true`) runs the target's main under `python -m <old dotted> --help`; p0's `-m` snapshot is unchanged. `--verify` passes on the emitted-files manifest and fails loud when that manifest is stale vs a fresh fragment resolution.

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- p0's sentinel test green for all dotted paths under both spellings — this is the exit gate. RETRACTED 2026-09-18: package `__init__.py` files are never shimmed, no `__path__` double-load path exists; the 12 originals are restored at their old locations. Every -m entry point and the CLI --help snapshot unchanged. Mutation: emit one leaf shim as a star-import and the singleton test goes RED.
- `just check` green on the lane tip.
- `just pc-ci` (exit gate) green on the lane tip.
- `lint-imports` green: ZERO inversions among the landed layers `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them instead.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the PR body.
- The actual diff's file set is a subset of *Allowed files*. Undeclared overlap is a HALT at the zip, not a note.
- `just ship-it` green; every PR's `gh pr view --json headRefOid` equals `git rev-parse` on its branch.
- Rollback (Q7 ruled (a)): a failed integrated gate at the zip unwinds to tip N-1 and raises DECISION-NEEDED naming the lane; per-tip re-certification localises the failure.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for correctness is a `DECISION`, not an edit.

- ZERO byte edits inside any moved file. A `git mv` only.
- No single redirect table and no `MetaPathFinder`. Physical per-module shims are what keep later touch sets disjoint.
- No import rewrite anywhere. Rewrite-on-arrival starts at p3b.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output).

- `.secrets.baseline` (STANDING RULE, conductor 2026-09-19, generalizing p0's ruling: a mechanical companion of allowed content. detect-secrets regenerates it whenever an allowed file's digest literals move or change, so its diff may contain ONLY line renumbering, removal of orphaned entries, the `generated_at` stamp, and new `Hex High Entropy String` entries whose flagged lines are sha256 / git-SHA literals inside this phase's own Allowed files; the lane verifies that in the PR body. New digest literals in TEST files carry an inline `# pragma: allowlist secret` instead of a baseline entry, so the baseline stops churning; JSON records keep baseline entries. Never a credential.)
- `src/entropic_substrate/**` (whole-tree `git mv`, zero byte edits)
- `src/entropic_substrate/__legacy/**`
- `tools/gen_shims.py`
- `tools/mapping.json` (the machine input; read, extended only if p0 left a node unmapped, and PRUNED of rows whose source module no longer exists in the tree: measured at p2, `entropic_substrate.analysis.d1_reader_source_closure` was deleted, and `tools/gen_shims.py` resolves every non-package mapping key into a shim, so a stale row would emit a shim importing a missing `__legacy` module; `retired: true` is p13-only by the generator's own doctrine, so pruning is the p3 act, recorded in the PR body with the row and the measurement)
- `.claude/workflows/landed-set/p3-quarantine.json` (this phase's fragment) and the fragment convention itself
- `tests/test_shim_layer.py`
- `tests/test_legacy_coupling.py`

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status                                                    | ruling                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| gen_shims v2       | RULED 2026-09-18 (DN-P1 gen_shims v2, generator redesign) | Fragments only ADD (no module ever migrates between fragments). The generator NEVER rebinds or writes a package `__init__.py`: leaf shims only; the 12 original package `__init__.py` files are restored at their old locations after the `git mv`. A fragment naming a package key or the root key fails loud. `--verify` checks an emitted-files manifest only, never an `rglob` over `out_dir`. Canonical invocation: `uv run python tools/gen_shims.py .claude/workflows/landed-set tools/mapping.json src --manifest-out .claude/workflows/landed-set/.emitted.json`. |
| shim form          | RULED 2026-09-17                                          | Static import then `sys.modules` rebind. The string-only form is forbidden: grimp cannot see through it and the layers contract goes vacuously green for every edge routed through a shim.                                                                                                                                                                                                                                                                                                                                                                                 |
| E-11               | RULED 2026-09-17 (precondition)                           | `codex/deferred-d-host-tail-20260914` and `codex/deferred-d-receipt-20260914` are landed, re-snapshotted or abandoned BEFORE p3 launches. They stop rebasing after the move.                                                                                                                                                                                                                                                                                                                                                                                               |
| topology           | RULED 2026-09-17                                          | Hexagonal via `__legacy` quarantine, aligned to the gated ladder cadence. `__legacy` is EMPTY after p12.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| P-1                | RULED 2026-09-17                                          | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.                                                                                                                                                                                                                                                                                                                                                          |
| O-1                | RULED 2026-09-17                                          | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                                                                                                                                                                                                                                                                                                                                                                                     |
| E-4                | RULED 2026-09-17                                          | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                                                                                                                                                                                                                                                                                                                                                                         |
| rewrite-on-arrival | RULED 2026-09-17                                          | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                                                                                                                                                                                                                                                                                                                                                                             |
| shim form          | RULED 2026-09-17                                          | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`. A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                                                                                                                                                                                                                                                                                                                                                                       |
| skeleton           | RULED 2026-09-17                                          | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                                 | must go RED                                                                                                                                                                                                                                                                                                                                                          |
| ------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| emit one shim as `from X import *`                                       | the p0 sentinel-singleton test. This is p3's EXIT GATE. RETRACTED 2026-09-18: previously read "12 of the shims are package `__init__` files whose `__path__` can double-load children" — package `__init__.py` files are never shimmed under the redesigned generator, so this hazard class is closed by construction; the mutation still applies to every LEAF shim |
| emit one shim in the string-only `importlib.import_module` form          | the layers contract must be shown green-blind on a deliberately illegal edge; the static-import form must catch it                                                                                                                                                                                                                                                   |
| a fragment names a package key (a dotted prefix of another named module) | `tools/gen_shims.py` fails loud at generation time; no package `__init__.py` is ever rebound                                                                                                                                                                                                                                                                         |
| a fragment names the root key `entropic_substrate`                       | `tools/gen_shims.py` fails loud at generation time; the package root is never rebound to a subpackage                                                                                                                                                                                                                                                                |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- No routing change: zero byte edits. The phase's obligation is that the observability substrate still IMPORTS and the E entry points still configure it through their shims.
- The CLI snapshot from p0 proves the substrate's human stderr prefix is unchanged after the move.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Promote gen_shims.py from the 2026-09-17 survey into tools/
2. Author the landed-set fragment convention and its schema assertion
3. Add test_legacy_coupling.py over the AST graph
4. git mv the whole package into __legacy with zero byte edits
5. Restore the 12 original package `__init__.py` files at their old locations after the git mv
6. Generate the per-module static-import LEAF shim layer
7. Verify the sentinel test over all leaf shims and the restored package `__init__.py` files
8. Record the from-import and string-shim mutation proofs
