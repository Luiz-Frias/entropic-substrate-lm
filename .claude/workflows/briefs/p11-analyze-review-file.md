# p11-analyze-review-file — `ladder/{analyze,review,file}`

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                                                                               |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p11-analyze-review-file`                                                                                                                                                  |
| parent                                       | `mythwave/p6-core` (CORRECTED 2026-09-18: the w1 zip order is p7 (20) then p6 (33), so the integration tip is `mythwave/p6-core`, not the zip's bottom `mythwave/p7-register-gate`) |
| mode (sprint manifest, parallel eligibility) | `ratchet`                                                                                                                                                                           |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived. Feeding the manifest `mode` here yields a silent `normal` and no Opus design gate.                                                          |

- **engine_mode:** `arch` (manifest field, machine-read by the launch composer; ruled 2026-09-18)
  | wave | `w2` |
  | `gates.liveRequired` | `false` |
  | live gate | DEFERRED. `gates.liveDeferred` names the wave zip as re-certification owner: one `just ship-it` on the integrated tip, never K concurrent podman stacks (house rule 2, 4 vCPU VM). |
  | gates | The fanout takes `{"offline":"just check","exit":"just pc-ci"}` and adds no gate vocabulary of its own. `liveRequired:false` is the ported cascade's default under `publishMode:'defer'`; `gates.liveDeferred` lives in the sprint manifest, not in the fanout args. |
  | launch | WAVE MEMBER: this phase is ONE TICKET `{id,title,scope,acceptance}` inside the single `mass-patch-fanout` invocation for wave `w2`. The overlay plans, provisions and drives the lane; there is no per-lane session and no hand-written wave script. `cascadeScriptPath` is the repo's ported `.claude/workflows/td-phase-lifecycle.cascade.js`. |
  | zip order | The fanout planner's `zipOrder` is AUTHORITATIVE and is validated as an exact permutation of the admitted lanes BEFORE any lane launches. The parent above is the ratified manifest's prediction; a non-permutation halts the wave with DECISION-NEEDED. |
  | lane worktree | NESTED at `<checkout>/.claude/worktrees/p11-analyze-review-file`, never a sibling of the conductor's worktree. Measured law from the 2026-08-27 pilot: a sibling lane is unreachable to workers (`EnterWorktree` reports success, Bash then refuses), a nested path is reachable by plain `cd`. |
  | lane isolation | `cd` to the lane worktree FIRST in every stage, never `EnterWorktree`. Verify `pwd` and branch before acting and FREEZE on mismatch: harness-carved isolation worktrees have landed under the wrong lane when several nested worktrees coexist. |
  | `policy.publishMode` | `'defer'` (ruling O-1). The lane ends CERTIFIED and publishes nothing. The wave's zip publishes once. |
  | trunk | `refactor/hexagonal-redesign` |

## Contract surface owned

| contract                                                                   | kind     | frozen_after |
| -------------------------------------------------------------------------- | -------- | ------------ |
| `ladder/analyze` (24), `ladder/review` (13), `ladder/file` (12) addresses  | contract | p13          |
| `ladder/review/isolation` :: the container / ACL / hermetic isolation ring | contract | p13          |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- kernel (p4, p5), core (p6)
- the retired D1 reader closure (p2)
- `kernel/runtime/logging.py` (p3b)

RETRACTED 2026-09-18 (Q2 ruled (b), DEFECT 4): this list previously also carried
`` `ladder/capture/reader` (p8) `` — a same-wave contract edge R-parallel forbids inside a single
`parallel_group`. Struck: `gen_shims.py` keeps p8's `ladder.capture.reader` module shimmed under
its old dotted name for as long as it is named in p8's own landed-set fragment, i.e. for the whole
of wave w2, so this phase's analyze modules resolve it through the shim rather than through a
same-wave contract edge. This rests on `tests/test_no_old_prefix.py` being scoped to each phase's
OWN merged fragment rather than the union of every landed fragment — recorded as an auditable
premise, not re-litigated here. This phase keeps `parallel_group: w2` and now parents on
`mythwave/p6-core` (see Branch, above).

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

| id   | verbatim house rule                                                            | tripwire                                                                                                                                                                                    | guard-teeth recipe                                                                                                                                     |
| ---- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| HR-1 | "Publish with `just ship-it`, never a bare `gt submit`."                       | Wave lane under `publishMode:'defer'`: the lane ends CERTIFIED and NEVER publishes. `just ship-it` runs once, at the zip, on the integrated tip. A `gt submit` inside the lane is a defect. | Inspect the lane transcript for any `gt submit` not reached through `just ship-it`; presence is a BLOCKED verdict, not a note.                         |
| HR-2 | "Probe sequentially. Never draw a conclusion from concurrent containers."      | The podman VM is 4 vCPU / 4 GB; two MLflow stacks at once make even a control appear hung.                                                                                                  | Any container conclusion in this phase must name the single container it came from. Two concurrent stacks in the transcript invalidate the conclusion. |
| HR-3 | "Never fix flakiness by raising a timeout or a retry budget."                  | A gate that goes green only after a timeout bump.                                                                                                                                           | A diff that raises any timeout or retry constant is a BLOCKED verdict unless it carries the measurement showing the cause was found.                   |
| HR-4 | "Comments record measurements, with the measurement."                          | A comment in this phase's diff stating a fact with no measurement attached.                                                                                                                 | A wrong measured claim is retracted explicitly, in the commit that fixes it. Silent overwrite is a defect.                                             |
| HR-5 | "TDD is non-negotiable, and guard-teeth prove both arms."                      | A guard landed without its RED arm demonstrated.                                                                                                                                            | Every guard in this phase carries the mutation listed under *Guard teeth by mutation*, performed and reverted in-branch, evidence in the PR body.      |
| HR-6 | "Secrets never enter git."                                                     | gitleaks runs at pre-commit and pre-push.                                                                                                                                                   | No credential, token or `.env` value in any diff of this phase.                                                                                        |
| HR-7 | "Upgrading an existing venv across the mlflow → mlflow-skinny swap breaks it." | `import mlflow` raising `ModuleNotFoundError` in a lane worktree.                                                                                                                           | Fix with `uv sync --all-groups --reinstall-package mlflow-skinny`, never by pinning around it.                                                         |

## Scope

ANALYZED, BLIND REVIEWED and FILED stages. The container/ACL/hermetic isolation ring lands in review/isolation, not analyze, per C's named deviation. The two d1_external_* modules arrive already free of the retired reader closure (p2). Detail: plan §A-p11.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- Okabe-Ito chart lane and the alias commitment/reveal round-trip tests pass unchanged. Mutation: give ladder/analyze a live-service import and the forbidden contract goes RED. Zero bare print( survives in the 49 moved modules. legacy_edge_budget < p7's, test_legacy_coupling green. Lane ends CERTIFIED.
- `just check` green on the lane tip.
- `just pc-ci` (exit gate) green on the lane tip.
- `lint-imports` green: ZERO inversions among the landed layers `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them instead.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the PR body.
- The actual diff's file set is a subset of *Allowed files*. Undeclared overlap is a HALT at the zip, not a note.
- Verdict CERTIFIED (tip SHA + exit-gate JSON). The lane does NOT publish.
- The lane branch rebases onto its zip parent with ZERO conflicts. A conflict falsifies the wave's edge-disjointness claim and fails the WAVE, not just this lane.
- The exit gate re-runs at the restacked tip. A restacked tree is a new tree; this lane's pre-zip certificate speaks for a tree that no longer exists.
- Rollback (Q7 ruled (a)): a failed integrated gate at the zip unwinds to tip N-1 and raises DECISION-NEEDED naming the lane; per-tip re-certification localises the failure.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for correctness is a `DECISION`, not an edit.

- Does NOT re-introduce a reader source closure. p2 retired it under ruling E-1.
- Does not put the isolation ring in `analyze/`. It lands in `review/isolation`, proposal C's named deviation.
- Does not change alias commitment / reveal semantics or any verdict arithmetic.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output). RETRACTED 2026-09-18 (Q1 ruled (b)): this section previously granted "the analyze/review/file tests, for import-path rewrite only" as a prose test grant. Struck — no phase before p13 edits EXISTING tests; shims keep them green under rewrite-on-arrival, and the prose form cannot discharge the ACCEPTANCE subset-of-Allowed-files check anyway.

- `.claude/workflows/landed-set/p11-analyze-review-file.json` — THIS phase's disjoint fragment (modules landed, targets, `legacy_edge_budget`). No other phase's fragment. `.importlinter` is NOT in this envelope: it is edited only at p0 and p13.
- the 49 modules mapped to `ladder/{analyze,review,file}` and their shims
- `src/entropic_substrate/ladder/analyze/**`
- `src/entropic_substrate/ladder/review/**`
- `src/entropic_substrate/ladder/file/**`

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status           | ruling                                                                                                                                                                                                            |
| ------------------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| isolation ring     | RULED 2026-09-17 | The container / ACL / hermetic isolation ring lands in `review/isolation`, not `analyze`. This is the ring H1 needs.                                                                                              |
| no live services   | RULED 2026-09-17 | `ladder/analyze` is construction-blind arithmetic with NO live-service imports. The forbidden contract is the tooth.                                                                                              |
| O-1                | RULED 2026-09-17 | Wave lane. Ends CERTIFIED, never publishes. `liveRequired:false`, zip re-certifies.                                                                                                                               |
| P-1                | RULED 2026-09-17 | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase. |
| O-1                | RULED 2026-09-17 | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                            |
| E-4                | RULED 2026-09-17 | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                |
| rewrite-on-arrival | RULED 2026-09-17 | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                    |
| shim form          | RULED 2026-09-17 | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`. A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.              |
| skeleton           | RULED 2026-09-17 | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                        |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                              | must go RED                                                                                                                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| give `ladder/analyze` a live-service import           | the forbidden contract                                                                                                                                                  |
| reintroduce one old-prefix import in a landed package | `tests/test_no_old_prefix.py`. Separately: this phase's `legacy_edge_budget` must be STRICTLY LESS than the previous phase's, and `tests/test_legacy_coupling.py` green |
| perturb the alias commitment / reveal round trip      | its existing test, which must move unchanged and still pass                                                                                                             |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- Analysis diagnostics route through `logger`. The Okabe-Ito chart lane keeps writing files, not stdout.
- The two analysis modules that already reach `sinks.py` are re-pointed at p3b's adapters rather than keeping a private path.
- Neither of the two pinned raw-`print` sites lives in this phase, so ZERO bare `print(` may survive in the 49 moved modules.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Land ladder/analyze with the design and figures sub-packages
2. Land ladder/review with the aliases and packet modules
3. Land ladder/review/isolation as the named deviation
4. Land ladder/file with its ports and graph modules
5. Assert zero live-service imports across ladder/analyze
