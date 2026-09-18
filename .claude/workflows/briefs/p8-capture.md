# p8-capture — `ladder/capture`, five sub-packages

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                                                                               |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p8-capture`                                                                                                                                                               |
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
  | lane worktree | NESTED at `<checkout>/.claude/worktrees/p8-capture`, never a sibling of the conductor's worktree. Measured law from the 2026-08-27 pilot: a sibling lane is unreachable to workers (`EnterWorktree` reports success, Bash then refuses), a nested path is reachable by plain `cd`. |
  | lane isolation | `cd` to the lane worktree FIRST in every stage, never `EnterWorktree`. Verify `pwd` and branch before acting and FREEZE on mismatch: harness-carved isolation worktrees have landed under the wrong lane when several nested worktrees coexist. |
  | `policy.publishMode` | `'defer'` (ruling O-1). The lane ends CERTIFIED and publishes nothing. The wave's zip publishes once. |
  | trunk | `refactor/hexagonal-redesign` |

## Contract surface owned

| contract                                                                                                                                            | kind     | frozen_after |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `ladder/capture/{records,reader,custody,resource,ledger}` addresses                                                                                 | contract | p13          |
| `ladder/capture/reader/telemetry.py` :: the old dotted name disposition; `telemetry.py` remains the ONLY admissible read side for `results/*.jsonl` | door     | p8           |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- kernel (p4, p5), core (p6)
- the `d1_training_step` / `d1_calibration` SPLIT CONTRACT (p5), already landed under `__legacy/`
- `ladder/capture/ports/metric_sink.py` and `ladder/capture/adapters/**` (p3b)

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

DVC-CAPTURED stage as five measured sub-packages: custody 20, records 10, resource 9, reader 9, ledger 4 (ext fan-in 74/35/46/27/20, measured), landing beside p3b's ports/ and adapters/. Moves the record halves of p5's two symbol splits; p9 takes the torch halves, so no file is shared. telemetry.py stays the only admissible read side. Detail: plan §D.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- telemetry.py still raises rather than returning empty on inadmissible telemetry, asserted by mutation. scripts/run_ladder.sh's two float()-consuming -m calls still parse (p0 CLI snapshot). remap_check at capture sub-package granularity reports no NEW cycle versus target. legacy_edge_budget < p7's, test_legacy_coupling green. Lane ends CERTIFIED.
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

- Does NOT re-own `ladder/capture/ports` or `ladder/capture/adapters`. p3b owns them; this phase lands the five sub-packages BESIDE them.
- Does NOT re-own `ladder/capture/custody/h_source.py`; p4 landed it.
- Does not add a second sink implementation. The three adapters are complete.
- Does NOT perform either symbol split. p5 did that; this phase only MOVES the record halves.
- Does NOT touch the torch halves. p9 moves them. p8 and p9 share no file, which is what makes wave 2 legal.
- Does not relax `telemetry.py`'s raise-on-inadmissible behaviour. It never returns an empty value.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output). RETRACTED 2026-09-18 (Q1 ruled (b)): this section previously granted "the capture-side tests, for import-path rewrite only" as a prose test grant. Struck — no phase before p13 edits EXISTING tests; shims keep them green under rewrite-on-arrival, and the prose form cannot discharge the ACCEPTANCE subset-of-Allowed-files check anyway.

- `.claude/workflows/landed-set/p8-capture.json` — THIS phase's disjoint fragment (modules landed, targets, `legacy_edge_budget`). No other phase's fragment. `.importlinter` is NOT in this envelope: it is edited only at p0 and p13.
- the 53 remaining `ladder.capture` modules and their shims
- `src/entropic_substrate/ladder/capture/{records,reader,custody,resource,ledger}/**`
- `__legacy/d1_training_step_records.py` and `__legacy/d1_calibration_records.py` (record halves ONLY; p5 created them) and their shims

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status           | ruling                                                                                                                                                                                                                              |
| ------------------ | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| sub-packages       | RULED 2026-09-17 | `ladder/capture` decomposes into five measured sub-packages: custody 20, records 10, resource 9, reader 9, ledger 4 (external fan-in 74/35/46/27/20, measured). Do not leave it as one bucket.                                      |
| splits             | RULED 2026-09-17 | `d1_training_step` and `d1_calibration` are SPLIT AT p5, serially, under `__legacy/`. p8 moves only the record halves and p9 only the torch halves, so the two wave lanes share no file. R-parallel forbids an intra-wave hub edge. |
| O-1                | RULED 2026-09-17 | Wave lane. Ends CERTIFIED, never publishes. `liveRequired:false`, zip re-certifies.                                                                                                                                                 |
| P-1                | RULED 2026-09-17 | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.                   |
| O-1                | RULED 2026-09-17 | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                                              |
| E-4                | RULED 2026-09-17 | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                                  |
| rewrite-on-arrival | RULED 2026-09-17 | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                                      |
| shim form          | RULED 2026-09-17 | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`. A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                                |
| skeleton           | RULED 2026-09-17 | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                                          |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                                               | must go RED                                                                                                                                                             |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| make `telemetry.py` return an empty value instead of raising on inadmissible telemetry | the admissibility test. `telemetry.py` is the only admissibility predicate for `results/*.jsonl` and is shared by the CLI preflight and `scripts/run_ladder.sh`         |
| reintroduce one old-prefix import in a landed package                                  | `tests/test_no_old_prefix.py`. Separately: this phase's `legacy_edge_budget` must be STRICTLY LESS than the previous phase's, and `tests/test_legacy_coupling.py` green |
| move a torch half in this lane, or edit a file p9 owns                                 | the predicted-vs-actual touch-set diff at the zip, which HALTS the wave on undeclared overlap                                                                           |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- Capture is where the metric path LIVES. Every moving emitter resolves its sink through p3b's port; none constructs one.
- `telemetry.py` and the reader sub-package log through the substrate, EXCEPT `telemetry final-eval`, which is one of the two machine-consumed stdout sites pinned raw at p0 and must stay a raw `print`.
- Resource and ledger metering emits through `MetricSink` under the run descriptor, replacing any direct write.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Land ladder/capture/custody beside p4's h_source
2. Land ladder/capture/records with p5's d1_training_step record half
3. Land p5's d1_calibration record half
4. Land ladder/capture/resource
5. Land ladder/capture/reader with telemetry.py
6. Land ladder/capture/ledger
7. Commit the remap_check output for the capture sub-package boundaries
