# p6-core — `core/{geometry,substrate,observation,corpus,probe,latent}`

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| branch                                       | `mythwave/p6-core`                                                                                                                                                                                                                                                                                                                               |
| parent                                       | `mythwave/p5-kernel`                                                                                                                                                                                                                                                                                                                             |
| mode (sprint manifest, parallel eligibility) | `ratchet`                                                                                                                                                                                                                                                                                                                                        |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived. Feeding the manifest `mode` here yields a silent `normal` and no Opus design gate.                                                                                                                                                                                                                       |
| wave                                         | `w1`                                                                                                                                                                                                                                                                                                                                             |
| `gates.liveRequired`                         | `false`                                                                                                                                                                                                                                                                                                                                          |
| live gate                                    | DEFERRED. `gates.liveDeferred` names the wave zip as re-certification owner: one `just ship-it` on the integrated tip, never K concurrent podman stacks (house rule 2, 4 vCPU VM).                                                                                                                                                               |
| gates                                        | The fanout takes `{"offline":"just check","exit":"just pc-ci"}` and adds no gate vocabulary of its own. `liveRequired:false` is the ported cascade's default under `publishMode:'defer'`; `gates.liveDeferred` lives in the sprint manifest, not in the fanout args.                                                                             |
| launch                                       | WAVE MEMBER: this phase is ONE TICKET `{id,title,scope,acceptance}` inside the single `mass-patch-fanout` invocation for wave `w1`. The overlay plans, provisions and drives the lane; there is no per-lane session and no hand-written wave script. `cascadeScriptPath` is the repo's ported `.claude/workflows/td-phase-lifecycle.cascade.js`. |
| zip order                                    | The fanout planner's `zipOrder` is AUTHORITATIVE and is validated as an exact permutation of the admitted lanes BEFORE any lane launches. The parent above is the ratified manifest's prediction; a non-permutation halts the wave with DECISION-NEEDED.                                                                                         |
| lane worktree                                | NESTED at `<checkout>/.claude/worktrees/p6-core`, never a sibling of the conductor's worktree. Measured law from the 2026-08-27 pilot: a sibling lane is unreachable to workers (`EnterWorktree` reports success, Bash then refuses), a nested path is reachable by plain `cd`.                                                                  |
| lane isolation                               | `cd` to the lane worktree FIRST in every stage, never `EnterWorktree`. Verify `pwd` and branch before acting and FREEZE on mismatch: harness-carved isolation worktrees have landed under the wrong lane when several nested worktrees coexist.                                                                                                  |
| `policy.publishMode`                         | `'defer'` (ruling O-1). The lane ends CERTIFIED and publishes nothing. The wave's zip publishes once.                                                                                                                                                                                                                                            |
| trunk                                        | `refactor/hexagonal-redesign`                                                                                                                                                                                                                                                                                                                    |

## Contract surface owned

| contract                                                                                                  | kind     | frozen_after |
| --------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `core/{geometry,substrate,observation,corpus,probe,latent}` addresses (the 33 remaining `core.*` modules) | contract | p13          |
| the `layers/kappa_norm.py` filing retraction                                                              | contract | p6           |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- kernel addresses (p4, p5)
- `ladder/capture/ports/metric_sink.py` (p3b), consumed BY INJECTION only

## Sacred invariants

The five below are quoted verbatim from `CLAUDE.md` (North star and
*Non-negotiable numerics*). They bind every phase without exception. `SI-*` ids are local to this
brief set; the text is the contract.

| id   | verbatim invariant                                                                                                          | tripwire                                                                                                     | guard-teeth recipe (house rule 5, both arms)                                                                                                                       |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SI-1 | "all geometric math runs in FP32 inside `torch.amp.autocast(enabled=False)` blocks regardless of surrounding bf16 autocast" | This phase moves `core/geometry`; every FP32 island must arrive byte-identical, asserted by diff.            | Drop one `autocast(enabled=False)` wrapper in `geometry/kappa.py`; the FP32-island dtype assertion must go RED. Revert in-branch, record both arms in the PR body. |
| SI-2 | "κ is clamped and inputs are projected into the component domain with a boundary epsilon before any map"                    | This phase moves the κ primitives; the clamp bound and boundary epsilon must arrive byte-identical.          | Widen the κ clamp bound by one ulp and feed a boundary input; the κ-domain test must go RED. Revert in-branch.                                                     |
| SI-3 | "the load-balance loss must stay finite and gradients must reach κ (unit-tested)"                                           | This phase moves `core/substrate`; the load-balance loss and its κ-gradient test must arrive byte-identical. | Detach κ from the load-balance term; the gradient-reaches-κ unit test must go RED. Revert in-branch.                                                               |
| SI-4 | "A magnitude-preserving carrier must not contain a hidden final UnitNorm or RMS overwrite."                                 | This phase moves the carrier's home package; no normalization line may change.                               | Append a `UnitNorm` to the magnitude-preserving carrier's output; the carrier magnitude test must go RED. Revert in-branch.                                        |
| SI-5 | "No probability-floor/square-root substitution is permitted in the main latent path."                                       | This phase moves `core/latent`; the latent law must arrive byte-identical.                                   | Substitute a `sqrt(p + eps)` for the latent law in the main path; the latent-path law test must go RED. Revert in-branch.                                          |

House rules from `CLAUDE.md`, verbatim headline, each binding in this phase:

| id   | verbatim house rule                                                            | tripwire                                                                                                                                         | guard-teeth recipe                                                                                                                                     |
| ---- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| HR-1 | "Publish with `just ship-it`, never a bare `gt submit`."                       | Wave lane under `publishMode:'defer'`: the lane ends CERTIFIED and NEVER publishes. `just ship-it` runs once, at the zip, on the integrated tip. | Inspect the lane transcript for any `gt submit` not reached through `just ship-it`; presence is a BLOCKED verdict, not a note.                         |
| HR-2 | "Probe sequentially. Never draw a conclusion from concurrent containers."      | The podman VM is 4 vCPU / 4 GB; two MLflow stacks at once make even a control appear hung.                                                       | Any container conclusion in this phase must name the single container it came from. Two concurrent stacks in the transcript invalidate the conclusion. |
| HR-3 | "Never fix flakiness by raising a timeout or a retry budget."                  | A gate that goes green only after a timeout bump.                                                                                                | A diff that raises any timeout or retry constant is a BLOCKED verdict unless it carries the measurement showing the cause was found.                   |
| HR-4 | "Comments record measurements, with the measurement."                          | A comment in this phase's diff stating a fact with no measurement attached.                                                                      | A wrong measured claim is retracted explicitly, in the commit that fixes it. Silent overwrite is a defect.                                             |
| HR-5 | "TDD is non-negotiable, and guard-teeth prove both arms."                      | A guard landed without its RED arm demonstrated.                                                                                                 | Every guard in this phase carries the mutation listed under *Guard teeth by mutation*, performed and reverted in-branch, evidence in the PR body.      |
| HR-6 | "Secrets never enter git."                                                     | gitleaks runs at pre-commit and pre-push.                                                                                                        | No credential, token or `.env` value in any diff of this phase.                                                                                        |
| HR-7 | "Upgrading an existing venv across the mlflow → mlflow-skinny swap breaks it." | `import mlflow` raising `ModuleNotFoundError` in a lane worktree.                                                                                | Fix with `uv sync --all-groups --reinstall-package mlflow-skinny`, never by pinning around it.                                                         |

## Scope

The science layer: geometry 5, substrate 6, observation 5, corpus 3, probe 2, latent 11 (metric+dynamics already landed in p4). Zero algorithm or parameter change — operator-ruled zone. Retract the layers/kappa_norm.py dead-end filing in the moving commit: UnitNorm is live in build_norm. Detail: plan §A-p6.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- Measured assertion in CI: zero top-level imports of mlflow, boto3, botocore, dvc, runpod, huggingface_hub or datasets across all core modules. Mutation: add one and the forbidden contract goes RED. Numerics tests byte-identical. legacy_edge_budget < p5's and test_legacy_coupling green. Lane ends CERTIFIED.
- `just check` green on the lane tip.
- `just pc-ci` (exit gate) green on the lane tip.
- `lint-imports` green: ZERO inversions among the landed layers `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them instead.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the PR body.
- The actual diff's file set is a subset of *Allowed files*. Undeclared overlap is a HALT at the zip, not a note.
- Verdict CERTIFIED (tip SHA + exit-gate JSON). The lane does NOT publish.
- The lane branch rebases onto its zip parent with ZERO conflicts. A conflict falsifies the wave's edge-disjointness claim and fails the WAVE, not just this lane.
- The exit gate re-runs at the restacked tip. A restacked tree is a new tree; this lane's pre-zip certificate speaks for a tree that no longer exists.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for correctness is a `DECISION`, not an edit.

- ZERO algorithm or parameter change. `core/` is the operator-ruled zone and this phase is structural only.
- Does NOT re-own `core/latent/{metric,dynamics}`; p4 landed them.
- Does NOT re-own `core/probe/score_gain.py`; p4 landed it.
- Does not split `sentence_prediction/execution.py`. The class-level tangle is out of scope for every structural phase and is scheduled after p13.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output).

- `.claude/workflows/landed-set/p6-core.json` — THIS phase's disjoint fragment (modules landed, targets, `legacy_edge_budget`). No other phase's fragment. `.importlinter` is NOT in this envelope: it is edited only at p0 and p13.
- the 33 remaining `core.*` modules and their shims
- `src/entropic_substrate/core/**`
- `src/entropic_substrate/__legacy/layers/kappa_norm.py` (retraction comment only)
- the numerics tests, for import-path rewrite only

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status           | ruling                                                                                                                                                                                                                                                    |
| ------------------ | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| core purity        | RULED 2026-09-17 | `core.*` may not import mlflow, boto3, botocore, dvc, runpod, huggingface_hub or datasets. The forbidden contract landed at p0.                                                                                                                           |
| DN-18 / graft 5    | RULED 2026-09-17 | `layers/kappa_norm.py`'s "dead end" filing is RETRACTED in the commit that moves the module: `UnitNorm` is live in `build_norm`. House rule 4: a wrong measured claim is retracted explicitly, not quietly overwritten. Placement stays `core/substrate`. |
| O-1                | RULED 2026-09-17 | This is a WAVE LANE. It ends CERTIFIED and never publishes. `liveRequired:false`, `gates.liveDeferred` names the zip as re-certification owner.                                                                                                           |
| P-1                | RULED 2026-09-17 | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.                                         |
| O-1                | RULED 2026-09-17 | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                                                                    |
| E-4                | RULED 2026-09-17 | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                                                        |
| rewrite-on-arrival | RULED 2026-09-17 | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                                                            |
| shim form          | RULED 2026-09-17 | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`. A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                                                      |
| skeleton           | RULED 2026-09-17 | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                                                                |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                                          | must go RED                                                                                                                                                             |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| add a live-service import (`mlflow`, `boto3`, `dvc`, `runpod`) to a `core` module | the forbidden contract                                                                                                                                                  |
| reintroduce one old-prefix import in a landed package                             | `tests/test_no_old_prefix.py`. Separately: this phase's `legacy_edge_budget` must be STRICTLY LESS than the previous phase's, and `tests/test_legacy_coupling.py` green |
| perturb one FP32 island, κ clamp bound or load-balance term                       | the numerics tests; they must be byte-identical across this phase                                                                                                       |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- `core/` emitters take an INJECTED `MetricSink`. `core` must never construct a sink and must never import `mlflow`: the forbidden contract is the tooth, and the mutation is `import mlflow` in `core/sentence` going RED.
- Diagnostic `print(` calls in the 33 moving modules are routed through `logger` in the commit that moves each file.
- No entry point moves in this phase.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Land core/geometry with its FP32 islands byte-identical
2. Land core/substrate and retract the kappa_norm dead-end filing
3. Land core/observation
4. Land core/corpus
5. Land core/probe beside p4's score_gain
6. Land core/latent beside p4's metric and dynamics
7. Assert zero live-service imports across all core modules
