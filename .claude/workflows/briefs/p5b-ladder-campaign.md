# p5b-ladder-campaign — the `ladder_campaign` artifact half lands in `kernel/artifacts`

Authoritative brief for one `td-phase-lifecycle.cascade` lane. Written 2026-09-22 under operator ruling D-WAVES r2 (keep the fan-out waves; land the one target whose importers span both waves' slices as a SERIAL rail between p5 and w1). Plan: `docs/plans/src-hexagonal-refactor.plan.md` (conductor-rulings table, D-WAVES r2). Decision file: `refactor-survey-2026-09-17/decision-wave-legality-2026-09-22.md`. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives.

## Branch

| field                                        | value                                                                                                                      |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| branch                                       | `mythwave/p5b-ladder-campaign`                                                                                             |
| parent                                       | `mythwave/p5-kernel`                                                                                                       |
| mode (sprint manifest, parallel eligibility) | `tracer` (serial rail)                                                                                                     |
| cascade `phase.mode` (model tier)            | `arch`. Always explicit, never derived. Feeding the manifest `mode` here yields a silent `normal` and no Opus design gate. |

- **engine_mode:** `arch`
  | wave | `serial` (rail: it must land BEFORE w1 launches; w1's file-disjointness depends on it) |
  | `gates.liveRequired` | `true` |
  | live gate | `just ship-it` runs INSIDE the lifecycle. House rule 1: never a bare `gt submit`. FOR-1541 workaround as ruled for p5. |
  | gates | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}` |
  | launch | SERIAL phase: the conductor launches the ported cascade DIRECTLY. Not a wave member. |
  | `repoToplevel` | `/Users/luizfrias/CursorAI/entropic-substrate-lm`. The MAIN checkout, never a worktree. |
  | trunk | `refactor/hexagonal-redesign` |

## Contract surface owned

| contract                                                                                                                                                                                                                                                                                                                                                                                                                  | kind     | frozen_after |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `kernel.artifacts` gains the artifact half of `evidence.ladder_campaign`: `artifact_mapping`, `artifact_from_mapping`, `read_bound_artifact`, `object_fields`, `read_json_artifact` (and the private helpers only they use). `tools/mapping.json` already declares this split (`action: split`, dest `kernel.artifacts`).                                                                                                 | contract | p13          |
| `__legacy/evidence/ladder_campaign.py` REMAINDER = the gate half (`RunAuthorization`, `SuccessorAuthorization`, `FiledLadderGate`, `validate_ladder_phase_transition`, `gate_from_d1_filing`, `read_campaign_design`, `verify_ladder_registration`, `authorize_ladder_run`, `actor_from_mapping`) which RE-EXPORTS the moved names from `kernel.artifacts` (p4's D-E precedent). p7 moves the remainder to `ladder.gate`. | contract | p7           |

## Contracts consumed

R-rails: every item below is LANDED before this lane launches.

- `tools/gen_shims.py`, the landed-set fragment convention, `tests/test_legacy_coupling.py` with p5's monotone-budget arm (p3, p5)
- `kernel/artifacts.py` (p4: `d1_artifact_snapshot` landed there; this phase APPENDS to it, never re-owns what p4 put there) and `kernel/lifecycle.py` (`ArtifactIdentity`, `ContentHash`)
- `tests/test_no_old_prefix.py` (p3b), `tools/regime_rows.py` + `tests/test_regime_rows.py` (identity mirrors, ruling Q1), ruling R-PIN (p5)
- `tools/lane_slices.py` (conductor tooling, landed at the p5 gap): the wave-legality instrument this phase's PR body cites

## Sacred invariants

Verbatim from the sprint invariants (the lane args carry the full list). Binding here: the layer contract (`kernel` imports nothing above it: the moved functions depend only on `kernel.lifecycle` and the standard library, MEASURED: `read_bound_artifact`, `object_fields`, `read_json_artifact`, `artifact_from_mapping`, `artifact_mapping` use `ArtifactIdentity`, `ContentHash`, `Path`, `json`; a `ladder`/`__legacy` import inside `kernel/artifacts.py` turns `lint-imports` RED); zero old-prefix imports in landed packages; identities: label `h-checkpoint-reader-v2` and its module set are frozen, the regime-row VALUES of BOTH composed readers re-measure here under R-PIN (DESIGN-GATE RULING 2026-09-22, wf_d8e28935-075, D-A): MEASURED, `kernel.artifacts` is a member of h-source-reader-v3's seven-module set AND of h-checkpoint-reader-v2's 51, so appending to it changes 15 of the 51 checkpoint members (the 14 importers plus kernel.artifacts itself) and 1 of the 7 source members; labels `h-checkpoint-reader-v2` and `h-source-reader-v3` and both module sets unchanged; no metric, loss or scientific behaviour changes.

## Scope

One split, one importer sweep, one re-pin. `evidence.ladder_campaign`'s five artifact functions move into `src/entropic_substrate/kernel/artifacts.py`; the legacy module keeps everything else and re-exports the five names; the 14 landed importers (MEASURED at 1c10969, every one a member of `_READER_MODULES`: `composition/{capture_cli,cli,source_cli,span_cli}.py`, `ladder/run/{evaluation_records,generation,journal,object_worker,runner,seals,span_diagnostics,trace_capture,worker_completion,worker_request}.py`) rewrite their `from entropic_substrate.__legacy.evidence.ladder_campaign import artifact_from_mapping` line ONCE to `entropic_substrate.kernel.artifacts`; both checkpoint regime rows re-pin with the tool in the same commit. No other module moves. Detail: Linear FOR-1543, FOR-1544, FOR-1545.

## ACCEPTANCE

Machine-checkable, one line each. The lane's `acceptance` arg points here.

- `tests/test_ladder_campaign_split.py` (RED-first): each moved name's `__module__` is `entropic_substrate.kernel.artifacts` and `entropic_substrate.evidence.ladder_campaign.<name>` IS that object (identity). Mutation: define one name in the remainder → RED.
- `legacy_edge_budget` in `.claude/workflows/landed-set/p5b-ladder-campaign.json` is exactly p5's minus 14 (the rewritten edges), gated by the monotone arm in `tests/test_legacy_coupling.py`; measured with `tools/import_graph.py` at the tip, never typed.
- `tools/lane_slices.py src tools/mapping.json <w1 lanes> --regenerable <zip-owned set>` and the same for w2 exit 0 at the tip; both outputs in the PR body (wave-legality evidence).
- Both `_CHECKPOINT_REGIME_TABLE` rows AND both `_SOURCE_REGIME_TABLE` rows re-pinned with `tools/regime_rows.py` in the same commit as the split + importer rewrite (D-A: kernel.artifacts is a member of both readers; the read-old table's dated p5b row records the measured member counts, 15 of 51 and 1 of 7, and replaces p5's "h-source-reader-v3 untouched" note); `tests/test_regime_rows.py`, `tests/test_identity_epoch.py` green; `docs/identity/read-old-table.md` gains a dated p5b row; tool output in the PR body.
- `just check`, `just pc-ci` (exit gate via the runner), `lint-imports` zero inversions, `just ship-it` green on the tip; `gh pr view --json headRefOid` equals `git rev-parse`. (PR-body / lifecycle evidence.)
- The diff's file set is a subset of *Allowed files*. (PR-body evidence; the zip's predicted-vs-actual check is not involved, this is serial.)

## Non-goals

Thoroughness beyond the stated ACCEPTANCE is NOT welcome. Anything that looks necessary for correctness is a `DECISION`, not an edit.

- Does NOT move the gate half; p7 owns it. Does NOT touch `RunAuthorization` consumers.
- Does NOT rewrite any importer that is not in the 14-file list (tests and `__legacy` modules keep the old dotted name through the shim; the shim keeps pointing at `__legacy.evidence.ladder_campaign`, `landed: false`, because a shim has one target).
- Does NOT bump an identity label. Does NOT edit `tools/mapping.json`, `.importlinter`, briefs, plan docs, the manifest, engine scripts, CI.
- Does NOT touch the p5 kernel modules or the two split halves p5 created.

## Allowed files

The escalation door: if correctness would require touching a contract surface, a file outside this envelope, or contradicting a SACRED INVARIANT, DO NOT EDIT. Return `DECISION` with the verification you RAN.

- `src/entropic_substrate/kernel/artifacts.py` (append the five functions + helpers; extend `__all__`)
- `src/entropic_substrate/__legacy/evidence/ladder_campaign.py` (remove the five, re-export them from `kernel.artifacts`)
- the 14 landed importers listed under Scope (one import line each; their docstrings if they name the old address, in the same commit, R-PIN)
- `tests/test_ladder_campaign_split.py` (new, RED-first)
- `tests/test_identity_epoch.py`, `tools/regime_rows.py`, `tests/test_regime_rows.py`, `docs/identity/read-old-table.md` (R-PIN re-pin)
- `.claude/workflows/landed-set/p5b-ladder-campaign.json` (THIS phase's fragment: the `evidence.ladder_campaign` row stays `landed: false` with its `__legacy` target; the fragment carries the measured `legacy_edge_budget` and a `note` recording the split)
- `.secrets.baseline` (STANDING RULE: mechanical companion, digest literals only, stated in the PR body)
- generator outputs (D-B, DESIGN-GATE RULING 2026-09-22): the re-derived `src/entropic_substrate/evidence/ladder_campaign.pyi` stub if the remainder's public names change, AND the re-derived `src/entropic_substrate/d1_artifact_snapshot.pyi` (its target is `entropic_substrate.kernel.artifacts`; appending public names there changes the stub; MEASURED by the gate: exactly these two stubs have a target this phase edits, out of 322) (`--verify` byte-compares a fresh re-derivation); listed in the PR body from `git status --porcelain` after generate + `--verify`
- `.claude/workflows/landed-set/.emitted.json` is untracked (FOR-1539); never `git add` it

## Already ruled

| id                  | status                                  | ruling                                                                                                                                                                                                          |
| ------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D-WAVES r2          | OPERATOR 2026-09-22                     | Keep the waves; this rail exists so that w1 (p6 ∥ p7) and w2 (p8 ∥ p9 ∥ p11) are file-disjoint under the slice rule (lane slice = targets ∪ landed importers ∪ shim pairs), verified by `tools/lane_slices.py`. |
| R-PIN               | CONDUCTOR 2026-09-22 (p5 precedent)     | Rewriting a `_READER_MODULES` member re-pins both checkpoint regime rows in the same commit; label and module set unchanged.                                                                                    |
| remainder re-export | CONDUCTOR 2026-09-22 (p4 D-E precedent) | A split's quarantined remainder re-exports the moved names so old-dotted-name consumers keep working through the one-target shim; the guard mutation targets DEFINITION in the remainder, not presence.         |
| split membership    | tools/mapping.json (ratified)           | The five artifact functions are the split's artifact half ("the only symbols 14 sentence_prediction modules import"); `actor_from_mapping` and everything gate-shaped stays with the remainder for p7.          |
| rewrite-on-arrival  | RULED 2026-09-17                        | A file's import lines are rewritten exactly once, by the phase that moves its target. The 14 importers are this phase's because their target moves here.                                                        |
| FOR-1541            | CONDUCTOR 2026-09-22                    | Publish workaround as in the p5 ruling (branch-only `gt submit` after a green `just ship-it` whose sole failure is the commitizen-branch empty range on an up-to-date lower branch).                            |

## Guard teeth by mutation

| mutation                                                                                                             | must go RED                                                                                                                                 |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| define `artifact_from_mapping` in the legacy remainder (shadow the re-export)                                        | `tests/test_ladder_campaign_split.py` (identity arm)                                                                                        |
| add `from entropic_substrate.__legacy.evidence.ladder_campaign import RunAuthorization` inside `kernel/artifacts.py` | `lint-imports` (kernel → __legacy is outside the contract, but `tests/test_legacy_coupling.py` budget goes RED: measured edges ≠ committed) |
| leave one of the 14 importers on the old `__legacy` address                                                          | `tests/test_legacy_coupling.py` (budget mismatch by one) and the monotone arm                                                               |
| type the regime-row digest instead of measuring it                                                                   | `tests/test_regime_rows.py` (control test vs the tool)                                                                                      |

## Observability conformance

Nothing to route: the five functions have no `print(` sites (MEASURED at 1c10969) and no entry point moves. No `MetricSink` import may enter `kernel`.

## Planned commit branches

1. FOR-1543 Split the artifact half into kernel/artifacts.py RED-first; remainder re-exports
2. FOR-1544 Rewrite the 14 landed importers and re-pin the checkpoint regime rows (R-PIN)
3. FOR-1545 Fragment, measured budget, wave-disjointness evidence, gates
