# p12-historical-composition — `historical/` + `composition/`; `__legacy` empty

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p12-historical-composition`                                                                                      |
| parent                                       | `mythwave/p10-run-upper`                                                                                                   |
| mode (sprint manifest, parallel eligibility) | `ratchet`                                                                                                                  |
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

| contract                                                                                                                         | kind     | frozen_after |
| -------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `historical/e_series` :: governed READ-ONLY (train.py, field_pretrain.py, the E model stack, E charts)                           | contract | p13          |
| `historical/d_ladder` :: the 11 d_ladder modules plus `analysis/d1_probe` and `analysis/ladder_probe`, runnable, same governance | contract | p13          |
| `composition/{cli.py,wiring,entrypoints}` :: the ONE composition root, the only place a concrete adapter is chosen               | contract | p13          |
| `__legacy/` EMPTY postcondition                                                                                                  | door     | p12          |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- every landed address
- the parameterized entrypoint source (p10)

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

E-series to a governed read-only historical/ per proposal B §6; the 11 d_ladder modules plus analysis/d1_probe and analysis/ladder_probe to historical/d_ladder (their only importers are inside the lane, measured). One composition root: cli.py, per-lane wiring, -m entrypoints. Postcondition: __legacy is empty. Detail: plan §E-6.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- A checkable postcondition test asserts zero files under __legacy/. The six E/D CLI subcommands still dispatch into historical/ so E0-E1.5c replay is possible. d_ladder_artifacts+d_ladder_execution move as one unit (the second module-level SCC). Mutation: split them and collection goes RED. just ship-it green.
- `just check` green on the lane tip.
- `just pc-ci` (exit gate) green on the lane tip.
- `lint-imports` green: ZERO inversions among the landed layers `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them instead.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the PR body.
- The actual diff's file set is a subset of *Allowed files*. Undeclared overlap is a HALT at the zip, not a note.
- `just ship-it` green; every PR's `gh pr view --json headRefOid` equals `git rev-parse` on its branch.
- Rollback (Q7 ruled (a)): a failed integrated gate at the zip unwinds to tip N-1 and raises DECISION-NEEDED naming the lane; per-tip re-certification localises the failure.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for correctness is a `DECISION`, not an edit.

- Deletes NOTHING. No module is dropped; `historical/` is governed read-only, not a graveyard.
- Does NOT delete the shim layer or rewrite `tests/`. That is p13.
- Does not re-pin the deployment image. Ruling recommendation E-5: once, at p13.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output).

- `.secrets.baseline` (STANDING RULE, conductor 2026-09-19, generalizing p0's ruling: a mechanical companion of allowed content. detect-secrets regenerates it whenever an allowed file's digest literals move or change, so its diff may contain ONLY line renumbering, removal of orphaned entries, the `generated_at` stamp, and new `Hex High Entropy String` entries whose flagged lines are sha256 / git-SHA literals inside this phase's own Allowed files; the lane verifies that in the PR body. New digest literals in TEST files carry an inline `# pragma: allowlist secret` instead of a baseline entry, so the baseline stops churning; JSON records keep baseline entries. Never a credential.)
- `.claude/workflows/landed-set/p12-historical-composition.json` — THIS phase's disjoint fragment (modules landed, targets, `legacy_edge_budget`). No other phase's fragment. `.importlinter` is NOT in this envelope: it is edited only at p0 and p13.
- the 57 modules mapped to `historical/**` and `composition/**` and their shims
- `src/entropic_substrate/historical/{e_series,d_ladder}/**`
- `src/entropic_substrate/composition/**`
- `justfile`
- `scripts/run_ladder.sh`
- `Containerfile`
- `tests/test_legacy_empty.py`

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status                                        | ruling                                                                                                                                                                                                              |
| ------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E-6 (was DN-10)    | RATIFIED at sprint-preflight (recommendation) | The D-ladder goes to `historical/d_ladder/`, RUNNABLE, same governance as `e_series`. Measured: their only importers are inside the lane. CLAUDE.md's "retained as mechanistic groundwork" argues against deletion. |
| E-4                | RULED 2026-09-17                              | `justfile` and `scripts/run_ladder.sh` consume the parameterized entrypoint source. No forwarders.                                                                                                                  |
| DN-24              | RULED 2026-09-17                              | The six E/D CLI subcommands stay wired from `composition/` into `historical/` so E0-E1.5c replay stays possible.                                                                                                    |
| SCC                | RULED 2026-09-17                              | `d_ladder_artifacts` + `d_ladder_execution` is the second module-level SCC and moves as ONE unit.                                                                                                                   |
| P-1                | RULED 2026-09-17                              | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.   |
| O-1                | RULED 2026-09-17                              | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                              |
| E-4                | RULED 2026-09-17                              | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                  |
| rewrite-on-arrival | RULED 2026-09-17                              | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                      |
| shim form          | RULED 2026-09-17                              | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`. A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                |
| skeleton           | RULED 2026-09-17                              | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                          |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                                     | must go RED                                                                                                                                                             |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| split the `d_ladder_artifacts` / `d_ladder_execution` SCC across two commits | collection-time `ImportError`                                                                                                                                           |
| leave one file under `__legacy/`                                             | `tests/test_legacy_empty.py`, which asserts zero files under `__legacy/`                                                                                                |
| reintroduce one old-prefix import in a landed package                        | `tests/test_no_old_prefix.py`. Separately: this phase's `legacy_edge_budget` must be STRICTLY LESS than the previous phase's, and `tests/test_legacy_coupling.py` green |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- The composition root becomes the ONLY `MetricSink` construction site in the tree. The conformance test's grep assertion moves from advisory to postcondition here.
- Every `python -m` entry point under `composition/entrypoints/` calls `configure_logging` at entry.
- The E entry points stop routing through shims and reach `kernel/runtime/logging.py` and the capture adapters directly. `historical/e_series` keeps emitting through the same port, so E replay keeps its telemetry.
- `verify_worker_source` lands in `historical/d_ladder` carrying the p1 resolver, not the old depth anchor.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Land historical/e_series as governed read-only
2. Land historical/d_ladder with the artifacts+execution SCC as one commit
3. Land composition/cli.py and wiring
4. Land composition/entrypoints
5. Point the justfile, run_ladder.sh and Containerfile at the parameterized source
6. Assert the __legacy empty postcondition
