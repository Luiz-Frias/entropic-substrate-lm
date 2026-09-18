# p2-d1-closure — Retire the D1 reader source closure and its manifest arm

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-17 against the approved plan `docs/plans/src-hexagonal-refactor.plan.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p2-d1-closure`                                                                                                   |
| parent                                       | `mythwave/p1-anchors`                                                                                                      |
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

| contract                                                                         | kind | frozen_after |
| -------------------------------------------------------------------------------- | ---- | ------------ |
| D1 external-analysis control envelope :: `reader_source_manifest_base64` REMOVED | door | p2           |
| `analysis/d1_reader_source_closure.py` :: retired                                | door | p2           |

## Contracts consumed

R-rails: consumed-before-owned is a topological violation and a generation-time HALT. Every item below is LANDED before this lane launches.

- `src/entropic_substrate/paths.py` (p1)
- `tests/test_identity_epoch.py` (p0)

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

Retire the D1 reader source closure per the operator ruling. Forced earlier than the D lane because d1_external_analysis_control.py:100 calls the closure at MODULE IMPORT, so the quarantine move makes the review ring unimportable. Drop the reader_source manifest arm from the control envelope and its worker verification. Measured: 2 importers, 0 stored envelopes. Detail: plan §0.5, §E-1.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- just check green with the review ring importable. Mutation: reintroduce a parents[2] anchor anywhere in the closure's former call path and the new test that imports the review ring from a relocated tree goes RED. Identity fixture drops the 52-entry manifest row and records the retraction (house rule 4).
- `just check` green on the lane tip.
- `just pc-ci` (exit gate) green on the lane tip.
- `lint-imports` green: ZERO inversions among the landed layers `kernel < core < ladder < historical < composition`. `__legacy` sits OUTSIDE the contract until p13, so edges to and from the quarantine are not linted; `tests/test_legacy_coupling.py` bounds them instead.
- Every mutation under *Guard teeth by mutation* performed and reverted in-branch, both arms in the PR body.
- The actual diff's file set is a subset of *Allowed files*. Undeclared overlap is a HALT at the zip, not a note.
- `just ship-it` green; every PR's `gh pr view --json headRefOid` equals `git rev-parse` on its branch.
- Rollback (Q7 ruled (a)): a failed integrated gate at the zip unwinds to tip N-1 and raises DECISION-NEEDED naming the lane; per-tip re-certification localises the failure.

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything below that looks necessary for correctness is a `DECISION`, not an edit.

- Does NOT build a replacement closure over the new tree. Ruling E-1: that is a later, separately ruled step if H1 wants reader-source pinning.
- Does NOT move the two `analysis/d1_external_*` modules; they land in `ladder/review` at p11.
- No compatibility shim for the removed envelope field. Measured: zero stored envelopes carry it.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return status `DECISION` with the verification you RAN attached (command + observed output).

- `src/entropic_substrate/analysis/d1_reader_source_closure.py`
- `src/entropic_substrate/analysis/d1_external_analysis_control.py`
- `src/entropic_substrate/analysis/d1_external_worker.py`
- `tests/test_d1_reader_source_closure.py`
- `tests/test_d1_external_worker.py`
- `tests/test_review_packet.py`

## Already ruled

Settled. The design gate must not re-open any of these as a `DECISION`.

| id                 | status           | ruling                                                                                                                                                                                                                                       |
| ------------------ | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| E-1                | RULED 2026-09-17 | Retire the D1 reader source closure AND the `reader_source_manifest_base64` envelope field OUTRIGHT in p2. Zero stored envelopes carry it. A closure over the new tree, if H1 wants reader-source pinning, is a later separately ruled step. |
| sequencing         | RULED 2026-09-17 | p2 lands before the quarantine because `d1_external_analysis_control.py:100` calls the closure at MODULE IMPORT and both importers land in `ladder/review`, the ring H1 needs.                                                               |
| P-1                | RULED 2026-09-17 | Observability is wired through DURING the refactor on the E-series pattern. Rail phase p3b plus the extended rewrite-on-arrival rule. This is the one permitted behaviour-change class inside a structural phase.                            |
| O-1                | RULED 2026-09-17 | `publishMode:'defer'`. Wave lanes end CERTIFIED, the zip publishes once through `just ship-it`, and `/pr-stack-comment-triage` files into the Linear project after every wave publish.                                                       |
| E-4                | RULED 2026-09-17 | Edit the scripts, no forwarders. The dotted entrypoint names live in ONE parameterized source consumed by `run_ladder.sh`, the justfile and the worker allowlists.                                                                           |
| rewrite-on-arrival | RULED 2026-09-17 | A file's import lines are rewritten exactly once, by the phase that moves it. This phase never rewrites an importer outside its own touch set.                                                                                               |
| shim form          | RULED 2026-09-17 | `import entropic_substrate.__legacy.<x> as _target` then `sys.modules[__name__] = _target`. A string-only `importlib.import_module` shim is forbidden: it makes the layers contract vacuously green.                                         |
| skeleton           | RULED 2026-09-17 | Merged skeleton (proposal C tree + proposal B grafts), layer contract `kernel < core < ladder < historical                                                                                                                                   |

## Guard teeth by mutation

House rule 5: a guard is proven load-bearing by breaking the guarded thing and showing the test goes RED. The happy path passing does not distinguish "the guard worked" from "the hazard never reached it". Each mutation is performed and reverted in-branch; both arms go in the PR body.

| mutation                                                                     | must go RED                                                                      |
| ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| reintroduce a `parents[2]` anchor anywhere in the closure's former call path | the new relocated-import test that imports the review ring from a relocated tree |
| re-add `reader_source_manifest_base64` to the control envelope               | the envelope schema test                                                         |

## Observability conformance

Per the PRIORITY AMENDMENT (ruling P-1). Observability conformance is the ONE behaviour-change class permitted inside a structural phase, scoped to log routing, metric routing and entry-point configuration. Measured scientific behaviour (losses, metric values, identities) stays frozen; the identity fixtures and the numerics tests are the tripwire.

- The retraction itself is recorded as a measured claim under house rule 4: the identity fixture drops the 52-entry manifest row IN THIS COMMIT, with the measurement that zero stored envelopes carried the field.
- No routing change. The D lane still emits nothing through the substrate; p3b lands the rail.

## Planned commit branches

One Linear issue per line, inside this phase's milestone in project "entropic-substrate-lm — hexagonal refactor" (team Form_fxn, key FOR). Each is one commit branch in this phase's stack.

1. Write the RED-first relocated-import test for the review ring
2. Drop the reader_source manifest arm from the control envelope
3. Drop the closure verification from the external worker
4. Delete d1_reader_source_closure.py and its test
5. Retract the 52-entry manifest fixture row with its measurement
