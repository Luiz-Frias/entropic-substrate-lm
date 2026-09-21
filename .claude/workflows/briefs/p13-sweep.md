# p13-sweep — Tests onto new paths, shims deleted, contracts tightened

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p13-sweep`                                                                                                       |
| parent                                       | `mythwave/p12-historical-composition`                                                                                      |
| mode (sprint manifest, parallel eligibility) | `ratchet`                                                                                                                  |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived. Feeding the manifest `mode` here yields a silent `normal` and no Opus design gate. |

- **engine_mode:** `arch` (manifest field, machine-read by the launch composer; ruled 2026-09-18)
  | wave | `serial (last, R-train)` |
  | `gates.liveRequired` | `true` |
  | live gate | `just ship-it` runs INSIDE the lifecycle. House rule 1: never a bare `gt submit`. |
  | gates | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}` |
  | launch | SERIAL phase: the conductor launches the repo's ported cascade DIRECTLY, `Workflow({scriptPath: <repo>/.claude/workflows/td-phase-lifecycle.cascade.js, args})`. Not a wave member. |
  | `repoToplevel` | `/Users/luizfrias/CursorAI/entropic-substrate-lm`. The MAIN checkout, never a worktree, because `advisorLockPath` derives from its basename. |
  | trunk | `refactor/hexagonal-redesign` |

## Contract surface owned

| contract                                                                                                                                  | kind     | frozen_after |
| ----------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `.importlinter` :: `__legacy` ADDED to the layers contract, full enforcement with zero exceptions. The second and last edit of this file. | door     | p13          |
| the old dotted namespace :: RETIRED                                                                                                       | door     | p13          |
| `remap_check` at the new HEAD, recorded in-repo as the next refactor's baseline                                                           | contract | p13          |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- every phase's addresses
- `tools/gen_shims.py`, the landed-set fragment convention and `tests/test_legacy_coupling.py` (p3)

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

One mechanical pass rewriting 395 test files onto the new dotted paths, then delete every shim, fragment and the generator. Add __legacy to the import-linter contract and enforce it with zero exceptions. This is the only step that breaks an external caller, so it is announced separately. Detail: plan §A-p13.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- lint-imports green with __legacy inside the contract and zero exceptions. remap_check at the new HEAD recorded in-repo as the next refactor's baseline. Mutation: reintroduce one old dotted import anywhere and both the layers contract and the no-old-prefix test go RED. Observability postconditions green. Deployment image rebuilt and re-pinned once.
- `just check` green on the lane tip.
- `just pc-ci` (exit gate) green on the lane tip.
- `lint-imports` green: ZERO inversions among the landed layers `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them instead.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the PR body.
- The actual diff's file set is a subset of *Allowed files*. Undeclared overlap is a HALT at the zip, not a note.
- `just ship-it` green; every PR's `gh pr view --json headRefOid` equals `git rev-parse` on its branch.
- Rollback (Q7 ruled (a)): a failed integrated gate at the zip unwinds to tip N-1 and raises DECISION-NEEDED naming the lane; per-tip re-certification localises the failure.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for correctness is a `DECISION`, not an edit.

- No `src/` module moves. This is the test-side and contract-tightening pass.
- Does not tighten `layers` to `independence` between the stage hexagons. That is a later step, after the measurement supports it.
- Does not resolve the class-level residue (`d1_trainer_handoff ↔ d1_evaluation`, `sentence_prediction.journal ↔ execution`). Named, measured, scheduled after p13.
- Does NOT edit sealed artifacts (`docs/verdicts/*`, `docs/e1-*preregistration*.md`). They stay stale by design even though this phase's terminal state leaves their dotted paths invalid — ruled 2026-09-18: never edited, stale by design on the operator's one-line disposition of that staleness (Q6).

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output).

- `.secrets.baseline` (STANDING RULE, conductor 2026-09-19, generalizing p0's ruling: a mechanical companion of allowed content. detect-secrets regenerates it whenever an allowed file's digest literals move or change, so its diff may contain ONLY line renumbering, removal of orphaned entries, the `generated_at` stamp, and new `Hex High Entropy String` entries whose flagged lines are sha256 / git-SHA literals inside this phase's own Allowed files; the lane verifies that in the PR body. New digest literals in TEST files carry an inline `# pragma: allowlist secret` instead of a baseline entry, so the baseline stops churning; JSON records keep baseline entries. Never a credential.)
- `tests/**` (all 395 files)
- every remaining shim file
- `tools/gen_shims.py`
- `.claude/workflows/landed-set/` (every fragment; they are deleted with the generator)
- `.importlinter` (second and last edit: add `__legacy` to the contract)
- `tests/test_legacy_coupling.py` (retired with the shims)
- the deployment image pin
- `CLAUDE.md`, `docs/architecture.md`, `docs/deploy-h100.md` (NEW 2026-09-18, Q6 emitted default accepted: p13 owns docs, updating all three to the landed layout)

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status                                        | ruling                                                                                                                                                                                                                                      |
| ------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E-5 (was DN-7)     | RATIFIED at sprint-preflight (recommendation) | The deployment image is re-pinned ONCE, at p13. Spend-adjacent.                                                                                                                                                                             |
| announcement       | RULED 2026-09-17                              | This is the only step that breaks an external caller, so it is announced separately.                                                                                                                                                        |
| E-10 (was DN-20)   | RATIFIED at sprint-preflight (recommendation) | The one-implementation port and the checked-in `*Fake` doubles are DEFERRED PAST p13. Both are behaviour-adjacent and belong in no structural phase.                                                                                        |
| Q10                | RULED 2026-09-18                              | The ~383-file test rewrite is SCRIPTED, not hand-edited: the rewrite script itself is committed and reviewed as script-plus-spot-check against a sample, not as ~383 individual diffs.                                                      |
| Q6 (docs)          | RULED 2026-09-18 (emitted default accepted)   | p13 owns `CLAUDE.md`, `docs/architecture.md`, `docs/deploy-h100.md`. Sealed artifacts (`docs/verdicts/*`, `docs/e1-*preregistration*.md`) are never edited; their staleness disposition is ruled 2026-09-18: never edited, stale by design. |
| P-1                | RULED 2026-09-17                              | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.                           |
| O-1                | RULED 2026-09-17                              | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                                                      |
| E-4                | RULED 2026-09-17                              | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                                          |
| rewrite-on-arrival | RULED 2026-09-17                              | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                                              |
| shim form          | RULED 2026-09-17                              | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`. A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                                        |
| skeleton           | RULED 2026-09-17                              | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                                                  |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                                                                                              | must go RED                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| reintroduce one old dotted import anywhere                                                                                            | BOTH the layers contract and `tests/test_no_old_prefix.py`                                    |
| leave `__legacy` outside the layers contract                                                                                          | the p13 full-enforcement assertion                                                            |
| add a bare `print(` outside the two pinned sites, or drop `configure_logging` from any entry point, or add a second `mlflow` importer | `tests/test_observability_conformance.py`, now running as a postcondition over the whole tree |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- POSTCONDITIONS, asserted not asserted-by-narrative: exactly ONE `mlflow` importer in the tree (`ladder/capture/adapters/mlflow.py`); ZERO unconfigured entry points; ZERO bare `print(` outside the two pinned allowlist sites; the D1 observer protocols GONE.
- The observability conformance test graduates from per-phase to whole-tree here.
- The rewrite-on-arrival exception closes: after p13 there is no structural phase left, so any further log or metric routing change is an ordinary behaviour change under normal rules.
- Entry-point / allowlist promotion (CONFORMANCE-PROMOTION RULING, conductor 2026-09-21): when this phase retires the landed-set fragments with the shim generator, it inlines the merged sets into `tests/observability_conformance_config.py` in the same commit and asserts the postconditions (every entry point conforming, allowlist = the two pinned sites). `tests/observability_conformance_config.py` stays p3b's file and is NOT in this phase's envelope; it merges every fragment present.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Write and commit the scripted test-rewrite tool; review as script-plus-spot-check (Q10), not per-file diff
2. Rewrite the 395 test files onto the new dotted paths with that script
3. Delete every shim and the shim generator
4. Add __legacy to the import-linter contract and enforce zero exceptions
5. Promote the observability conformance test to a whole-tree postcondition
6. Record remap_check at the new HEAD as the next baseline
7. Update CLAUDE.md, docs/architecture.md and docs/deploy-h100.md to the landed layout
8. Rebuild and re-pin the deployment image once
