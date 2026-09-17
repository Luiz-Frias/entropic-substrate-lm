# p5-kernel — `kernel/{identity,evidence_class,verdict,runtime}`

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                                                                               |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p5-kernel`                                                                                                                                                                |
| parent                                       | `mythwave/p4-h-tracer`                                                                                                                                                              |
| mode (sprint manifest, parallel eligibility) | `ratchet`                                                                                                                                                                           |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived. Feeding the manifest `mode` here yields a silent `normal` and no Opus design gate.                                                          |
| wave                                         | `serial`                                                                                                                                                                            |
| `gates.liveRequired`                         | `true`                                                                                                                                                                              |
| live gate                                    | `just ship-it` runs INSIDE the lifecycle. House rule 1: never a bare `gt submit`.                                                                                                   |
| gates                                        | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}`                                                                                     |
| launch                                       | SERIAL phase: the conductor launches the repo's ported cascade DIRECTLY, `Workflow({scriptPath: <repo>/.claude/workflows/td-phase-lifecycle.cascade.js, args})`. Not a wave member. |
| `repoToplevel`                               | `/Users/luizfrias/CursorAI/entropic-substrate-lm`. The MAIN checkout, never a worktree, because `advisorLockPath` derives from its basename.                                        |
| trunk                                        | `refactor/hexagonal-redesign`                                                                                                                                                       |

## Contract surface owned

| contract                                                                                                                                                                                                                                                                                   | kind     | frozen_after |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------ |
| `kernel/identity`, `kernel/evidence_class`, `kernel/verdict`, `kernel/runtime` addresses (the remainder, after p3b's `logging.py` and p4's `artifacts`/`lifecycle`)                                                                                                                        | contract | p13          |
| `tests/test_no_old_prefix.py` :: zero old-prefix imports in any landed package                                                                                                                                                                                                             | contract | p13          |
| the `d1_training_step` / `d1_calibration` SYMBOL SPLIT :: record halves (`d1_training_step_records.py`, `d1_calibration_records.py`) and torch halves, both still under `__legacy/`. p8 later moves only the record halves, p9 only the torch halves, so the two wave lanes share NO file. | contract | p9           |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- `tools/gen_shims.py`, the landed-set fragment convention and `tests/test_legacy_coupling.py` (p3)
- `kernel/runtime/logging.py` (p3b)
- `kernel/artifacts`, `kernel/lifecycle` (p4)

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

Land the remaining kernel: identity (2), evidence_class (2), verdict (5), runtime (4 minus p3b's logging), plus git_environment. kernel/artifacts and kernel/lifecycle landed in p4. Also split d1_training_step and d1_calibration RED-first into record and torch halves, both still under __legacy/, so p8 and p9 share no file. Detail: plan §A-p5.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- legacy_edge_budget in this phase's fragment is strictly less than p4's, and test_legacy_coupling is green. Mutations: one kernel→ladder import turns the layers contract RED; one reintroduced old-prefix import turns the no-old-prefix test RED; landing either symbol split without its RED-first test goes RED at collection.
- `just check` green on the lane tip.
- `just pc-ci` (exit gate) green on the lane tip.
- `lint-imports` green: ZERO inversions among the landed layers `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them instead.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the PR body.
- The actual diff's file set is a subset of *Allowed files*. Undeclared overlap is a HALT at the zip, not a note.
- `just ship-it` green; every PR's `gh pr view --json headRefOid` equals `git rev-parse` on its branch.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for correctness is a `DECISION`, not an edit.

- Does NOT re-own `kernel/runtime/logging.py`. p3b owns it; this phase lands the rest of `kernel/runtime` beside it.
- Does NOT re-own `kernel/artifacts` or `kernel/lifecycle`. p4 owns them.
- No rewrite of importers outside the touch set. They keep the old dotted name until their own phase.
- Does NOT move either split half out of `__legacy/`. p8 moves the record halves, p9 the torch halves; this phase only creates the seam.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output).

- `.claude/workflows/landed-set/p5-kernel.json` — THIS phase's disjoint fragment (modules landed, targets, `legacy_edge_budget`). No other phase's fragment. `.importlinter` is NOT in this envelope: it is edited only at p0 and p13.
- the modules mapped to `kernel/{identity,evidence_class,verdict,runtime}` and their shims
- `src/entropic_substrate/__legacy/git_environment.py` and its shim
- `src/entropic_substrate/kernel/**`
- `src/entropic_substrate/__legacy/d1_training_step.py` (SPLIT) and the new `__legacy/d1_training_step_records.py`
- `src/entropic_substrate/__legacy/d1_calibration.py` (SPLIT) and the new `__legacy/d1_calibration_records.py`
- `tests/test_no_old_prefix.py`
- `tests/test_d1_training_step_split.py` (RED-first split test, both modules)
- `tools/mapping.json` (READ ONLY as the machine input; it already maps `d1_training_step_records` and `d1_calibration_records` → `ladder.capture.records`, and the torch halves keep their existing dests)

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status                                        | ruling                                                                                                                                                                                                                                                                                            |
| ------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E-12               | RATIFIED at sprint-preflight (recommendation) | `git_environment.py` lands at `kernel/runtime` in p5, NOT with the H tracer. Measured: zero H importers, two D1 importers, 12 lines, zero outbound edges. This diverges from the ruling's wording and the divergence is stated, not silent.                                                       |
| ordering           | RULED 2026-09-17                              | `kernel.identity` has 123 eager importers, the largest fan-out in the repo, so p5 lands ALONE after the tracer and before any package that consumes it.                                                                                                                                           |
| R-parallel         | RULED 2026-09-17                              | The `d1_training_step` / `d1_calibration` splits move OUT of wave 2 and into p5. As an intra-wave hub edge between p8 and p9 they were forbidden inside a `parallel_group`. p5 splits both RED-first under `__legacy/`; p8 and p9 then consume the split contract rails-before and share no file. |
| P-1                | RULED 2026-09-17                              | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.                                                                                 |
| O-1                | RULED 2026-09-17                              | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                                                                                                            |
| E-4                | RULED 2026-09-17                              | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                                                                                                |
| rewrite-on-arrival | RULED 2026-09-17                              | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                                                                                                    |
| shim form          | RULED 2026-09-17                              | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`. A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                                                                                              |
| skeleton           | RULED 2026-09-17                              | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                                                                                                        |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                                                 | must go RED                                                                                                                                             |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| add one `kernel` → `ladder` import                                                       | the layers contract                                                                                                                                     |
| reintroduce one old-prefix import in a landed package                                    | `tests/test_no_old_prefix.py`. Separately: this phase's `legacy_edge_budget` must be STRICTLY LESS than p4's, and `tests/test_legacy_coupling.py` green |
| land either split without its RED-first test, or leave a record symbol in the torch half | the split's own test at collection; each half must be importable from its own module before the split commit goes green                                 |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- Rewrite-on-arrival applies: `config.py`, `checkpoint.py`, `git_environment.py` and the verdict modules route their diagnostic `print(` calls through `logger` in the commit that moves them.
- `kernel` must NOT gain a `MetricSink` import. The port lives at `ladder/capture/ports`; kernel is below it in the layer order and the layers contract enforces that.
- No entry point moves in this phase, so no new `configure_logging` call site.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Land kernel/identity and re-point its shims
2. Land kernel/evidence_class
3. Land kernel/verdict
4. Land kernel/runtime beside p3b's logging substrate
5. Land git_environment.py at kernel/runtime per E-12
6. Split d1_training_step into record and torch halves RED-first
7. Split d1_calibration into record and torch halves RED-first
8. Add the no-old-prefix test and the legacy-edge-budget assertion
