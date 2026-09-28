# p14-j-register: Register and gate codecs adopt strict_json (FOR-1575 + FOR-1599 merged)

Authoritative brief for one `td-phase-lifecycle.cascade` lane, written 2026-09-28 by the p14 planning lane from the ratified-pending manifest. Plan: `docs/plans/p14-consolidation.plan.md#p14-j-register`. Linear: FOR-1575, FOR-1599, FOR-1577, FOR-1613. The lane's `acceptance` arg reads "Per the brief's ACCEPTANCE section"; this file is where that section lives. Premises were measured at `eb93d5d`, whose `src` equals the epoch `651af30`; the prelaunch ceremony re-probes them at the re-stamped `verified_at`.

## Branch

| field                                 | value                                                                                            |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| branch                                | `mythwave/p14-j-register`                                                                        |
| parent                                | `mythwave/p14-t-local`                                                                           |
| mode (manifest, parallel eligibility) | `ratchet`                                                                                        |
| cascade `phase.mode` (model tier)     | `normal`, always explicit                                                                        |
| wave                                  | w1 (`p14-w1`), launched as one `mass-patch-fanout` wave, K <= 3                                  |
| `gates.liveRequired`                  | `false`                                                                                          |
| gates                                 | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":false}` |
| `repoToplevel`                        | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree           |
| trunk                                 | `refactor/p14-consolidation`                                                                     |
| publish                               | `publishMode: defer`: the lane ends CERTIFIED; the wave zip runs ONE `just ship-it`              |

## Contract surface owned

None. This phase changes no door and freezes no new contract.

## Contracts consumed

R-rails: every item is LANDED before this lane launches.

- kernel.codec.strict_json API (p14-r1-strict-json)
- kernel.identity.d1_primitives.require_sha256_hex_any_case (trunk, chore/for-1615)

## Sacred invariants

Verbatim source, then the tripwire that detects a breach at fire time, then the guard-teeth recipe.

| invariant                                               | verbatim                                                                                                                                                                             | tripwire                                                                                                                                                  | guard-teeth recipe                                                                                                             |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| House rule 5                                            | "**TDD is non-negotiable, and guard-teeth prove both arms.** Failing test first. Then break the guarded thing and show the test goes RED — a test that cannot fail is not evidence." | every behaviour-pinning test in this brief is committed RED before the fix commit                                                                         | perform each mutation in *Guard teeth by mutation*, show RED, revert, show GREEN; both arms in the PR body                     |
| House rule 3                                            | "**Never fix flakiness by raising a timeout or a retry budget.**"                                                                                                                    | `git diff` of this phase adds no larger timeout, deadline, retry count or backoff constant                                                                | a reviewer greps the diff for timeout, retry, deadline, backoff and max_attempts; any hit needs a measurement or is a DECISION |
| House rule 4                                            | "**Comments record measurements, with the measurement.**"                                                                                                                            | a changed comment that states a number cites how it was measured                                                                                          | a wrong measured claim is retracted explicitly in the commit that fixes it                                                     |
| Reader identity (R-PIN, `tests/test_identity_epoch.py`) | "the door frozen after p4 is the LABEL and the module-set SHAPE, while the composed digest below re-measures on every phase that edits a member's bytes (ruling R-PIN)"              | `git diff --name-only <parent>...HEAD` intersected with the 51 reader files listed under *Identity members* is empty for every phase but `p14-z-identity` | proven at p0: flipping one byte in a member turns the regime-row tests RED; only the rail re-shows it                          |
| Failure semantics                                       | "measurement/eval path → fail LOUD, never a fail-open wrapper."                                                                                                                      | no new `except` that swallows, counts-and-continues or returns a default on an evidence path                                                              | a reviewer greps the diff for `except` and `return None`; each hit is justified in the PR body                                 |
| Layer contract                                          | `kernel < core < ladder < historical < composition`, enforced by `lint-imports` with zero exceptions since p13                                                                       | `lint-imports` green at the tip                                                                                                                           | proven at p0 and p13: an upward import turns the layers contract RED                                                           |
| Exception surface                                       | "no change to any caller-visible exception type/message beyond what the remedy text states" (every p14 ticket's re-validation checklist)                                             | existing tests that assert exception types and messages pass unmodified                                                                                   | the adoption phases wire one site's `fail` to the wrong type and show that site's test RED                                     |

## Identity members

None: this phase's touch set has an empty intersection with the 51 reader-tuple files, measured by `comm` against `ladder/run/checkpoint.py:65-116` and `ladder/capture/capture.py:29-37`. Editing any of them is a HALT.

<details><summary>The 51 reader-tuple files (never edit outside the identity rail)</summary>

`composition/capture_cli.py`, `composition/cli.py`, `composition/source_cli.py`, `composition/span_cli.py`, `core/latent/dynamics.py`, `core/latent/metric.py`, `core/probe/score_gain.py`, `core/sentence/__init__.py`, `core/sentence/boundary.py`, `core/sentence/codec.py`, `core/sentence/consumer.py`, `core/sentence/corpus.py`, `core/sentence/head.py`, `core/sentence/law.py`, `core/sentence/model.py`, `core/sentence/numerics.py`, `core/sentence/objective.py`, `core/sentence/profile.py`, `core/sentence/records.py`, `core/sentence/recovery_state.py`, `core/sentence/rollout.py`, `core/sentence/runtime_precision.py`, `core/sentence/sampling.py`, `core/sentence/scoring.py`, `core/sentence/segmentation.py`, `core/sentence/spine.py`, `kernel/artifacts.py`, `kernel/lifecycle.py`, `ladder/capture/capture.py`, `ladder/run/checkpoint.py`, `ladder/run/evaluation_records.py`, `ladder/run/execution.py`, `ladder/run/generation.py`, `ladder/run/journal.py`, `ladder/run/mechanism_evidence.py`, `ladder/run/native_verification.py`, `ladder/run/object_worker.py`, `ladder/run/runner.py`, `ladder/run/seals.py`, `ladder/run/span_diagnostics.py`, `ladder/run/span_probe.py`, `ladder/run/span_probe_artifacts.py`, `ladder/run/span_states.py`, `ladder/run/trace_capture.py`, `ladder/run/training.py`, `ladder/run/worker_command.py`, `ladder/run/worker_completion.py`, `ladder/run/worker_execution.py`, `ladder/run/worker_request.py`, `ladder/run/worker_snapshot.py`, `ladder/run/worker_transport.py`

</details>

## Scope

Adopt `kernel.codec.strict_json` in the register and gate codecs, FOR-1575 and FOR-1599 merged, plus the register facet of FOR-1577.

- `ladder/register/d1_registration_codec.py` and `d1_launch_attempt_codec.py`: their `_pairs`/`_nonfinite` pair becomes `strict_loads(raw, fail=_fail)`; `_mapping` becomes `require_mapping`; `_exact` delegates to `exact_keys` and keeps its current message text.
- `ladder/register/d1_plan.py`: `_mapping` and `_exact_keys` only, keeping its split missing/unknown messages; it validates already-decoded objects, so no `strict_loads`.
- `ladder/gate/ladder_campaign.py`: `_mapping`/`_exact` style helpers only where byte-identical.
- Each codec keeps its own `_text`, its own `_fail`, its own exception type (`D1RegistrationCodecError`, `D1LaunchAttemptCodecError`, `D1PlanManifestError`) and every message.
- SHA: the eight importers of `require_sha256_hex_any_case` in `ladder/{register,gate}` are `d1_registration_codec.py`, `d1_plan.py`, `ladder.py`, `d1_confirmatory_matrix.py`, `d1_launch_attempts.py`, `d1_provenance.py`, `d1_registered_design.py` and `gate/ladder_campaign.py` [M, AST import scan at eb93d5d, before the rename]. Each call that `docs/plans/measurements/p14-sha-callers.txt` classifies BOUNDARY moves to `require_lowercase_sha256(value, fail=...)`, one RED-first mixed-case test per repointed site. IN-PROCESS callers stay. `ladder.py`, `d1_confirmatory_matrix.py`, `d1_launch_attempts.py`, `d1_provenance.py` and `d1_registered_design.py` are in the set only for this.

## ACCEPTANCE

Machine-checkable, one line each.

- `tests/test_strict_json_adoption_register.py` (new): an AST walk over `ladder/register` and `ladder/gate` finds zero definitions of `_pairs`, `_strict_pairs`, `_nonfinite`, `_reject_constant`.
- The epoch count of primitive-defining files, 36, falls by exactly 2 at the tip, measured with the grep in the plan's context table.
- Every existing test in `tests/test_d1_registration_codec.py`, `test_d1_launch_attempt_codec.py`, `test_d1_plan.py`, `test_ladder_campaign.py` passes unmodified; any needed edit is a DECISION.
- Per repointed BOUNDARY SHA site: a mixed-case digest now raises the site's own exception; the audit row is cited in the PR body.
- remap_check at the tip over a mapping re-derived by the p13 procedure, never the committed `p13-sweep-mapping.json`: inversions <= 1, package cycles <= 4. Output in the PR body.
- Mechanical counts at the tip at or under the epoch values below, output in the PR body.
- Admissibility evidence in the PR body for every abstraction introduced: consolidated sites as `path:line`, and the AST branch count at each call site holding flat or falling (ruling 14).
- `git diff --name-only mythwave/p14-t-local...HEAD` is a subset of *Allowed files*; the identity tripwire above is empty.
- `just check` green; exit gate `just pc-ci` via `uv run python scripts/run_exit_gate.py`; `lint-imports` clean.

Ratchet commands:

```text
uv run python tools/import_graph.py src/entropic_substrate <out>
uv run python - <out>/mapping.json   # stdin: the reproducing procedure in docs/plans/measurements/p13-sweep-remap.txt
uv run python tools/remap_check.py <out>/edges.json <out>/mapping.json
# must print: layer inversions <= 1, package-level cycles <= 4 (epoch 651af30: 1 and 4, 378 modules, 1805 edges)
```

```text
grep -rlE 'def (_pairs|_strict_pairs|_nonfinite|_reject_constant)\b' src --include='*.py' | wc -l   # epoch 36
grep -rn 'def _mapping(' src --include='*.py' | wc -l                                             # epoch 23
grep -rn 'separators=(",", ":")' src --include='*.py' | wc -l                                      # epoch 68
grep -rliE 'lowercase SHA-?256|lowercase sha256' src --include='*.py' | wc -l                      # epoch 29
grep -rn 'type: ignore\[arg-type\]' src --include='*.py' | wc -l                                  # epoch 19
find src scripts -name '*.py' -exec wc -l {} + | awk '$1>1000 && $2!="total"' | wc -l              # epoch 2
```

## Non-goals

Thoroughness beyond ACCEPTANCE is NOT welcome. Anything that looks necessary for correctness is a `DECISION`, not an edit.

- Does NOT change `_text` semantics at any site; FOR-1614 was settled by the pre-phase rung.
- Does NOT extract `_text`, `_canonical` or the round-trip idiom into shared code; the strict_json API is frozen.
- Does NOT touch `kernel/artifacts.py` `object_fields`, ruling 5.
- Does NOT launch pods, spend GPU or start any h-ladder flywheel run (ruling 8).

## Allowed files

The escalation door: if correctness would require touching a file outside this envelope, a contract surface, or a sacred invariant, DO NOT EDIT. Return `DECISION` with the verification you ran. Every path was verified to exist at `eb93d5d` unless marked new.

- `src/entropic_substrate/ladder/register/d1_registration_codec.py`
- `src/entropic_substrate/ladder/register/d1_launch_attempt_codec.py`
- `src/entropic_substrate/ladder/register/d1_plan.py`
- `src/entropic_substrate/ladder/register/ladder.py`
- `src/entropic_substrate/ladder/gate/ladder_campaign.py`
- `src/entropic_substrate/ladder/register/d1_confirmatory_matrix.py`
- `src/entropic_substrate/ladder/register/d1_launch_attempts.py`
- `src/entropic_substrate/ladder/register/d1_provenance.py`
- `src/entropic_substrate/ladder/register/d1_registered_design.py`
- `tests/test_d1_registration_codec.py`
- `tests/test_d1_launch_attempt_codec.py`
- `tests/test_d1_plan.py`
- `tests/test_ladder_campaign.py`
- `tests/test_strict_json_adoption_register.py` (new)
- `.secrets.baseline` (standing rule: mechanical companion, digest literals only, stated in the PR body)

Never edit, in this phase:

- `tests/test_identity_epoch.py`, `tools/regime_rows.py`, `docs/identity/*` (conductor only, and only in `p14-z-identity`)
- `tools/mapping.json`, `docs/plans/measurements/p13-sweep-mapping.json`, `docs/plans/measurements/p13-sweep-remap.txt`
- `.importlinter`, `.pre-commit-config.yaml`, `.github/`, `justfile`, `scripts/run_ladder.sh`, `pyproject.toml`, `uv.lock`
- `CLAUDE.md`, `docs/architecture.md`, `docs/plans/*`, `docs/reviews/*`, `docs/verdicts/*` (sealed, never edited), briefs, the manifest, engine scripts
- the 51 reader-tuple files listed under *Identity members*, except in `p14-z-identity`
- `ladder/capture/capture.py`, `ladder/run/journal.py`, and `object_fields` in `kernel/artifacts.py` as JSON sites (ruling 5)

## Already ruled

| id          | status                                         | ruling                                                                                                                                                        |
| ----------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1           | OPERATOR 2026-09-28                            | Trunk `refactor/p14-consolidation`; pushed only after tag `v0.1.0` exists.                                                                                    |
| 2           | OPERATOR 2026-09-28                            | Green baseline: full `just check` at the epoch, no other lanes active, SHA pinned by the conductor.                                                           |
| 3           | OPERATOR 2026-09-28                            | FOR-1612 held out of every phase.                                                                                                                             |
| 4           | OPERATOR 2026-09-28                            | strict_json home `kernel/codec/strict_json.py` with a `fail` callback.                                                                                        |
| 5           | OPERATOR 2026-09-28                            | Identity-bearing JSON sites exempt: `capture.py`, `journal.py`, `kernel/artifacts.py` `object_fields`.                                                        |
| 6           | OPERATOR 2026-09-28                            | One identity rail, landed last, one conductor re-pin.                                                                                                         |
| 8           | OPERATOR 2026-09-28                            | No h-ladder flywheel run without a fresh operator green light; no pods, no spend in any lane.                                                                 |
| 12          | OPERATOR 2026-09-28                            | FOR-1614 and FOR-1615 land as pre-phase rungs; SHA is lowercase-strict at every boundary; the loose kernel function is renamed `require_sha256_hex_any_case`. |
| 14          | OPERATOR 2026-09-28                            | Abstraction admissible only when it consolidates 2+ concrete sites or stabilizes a public surface.                                                            |
| P14-SHA-SEQ | PLANNING LANE 2026-09-28, pending ratification | `require_lowercase_sha256` lives in strict_json; each adoption phase repoints the audited BOUNDARY callers in its packages.                                   |

## Guard teeth by mutation

| mutation                                                                   | must go RED                                                           |
| -------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| re-add a local `def _pairs` in `d1_registration_codec.py`                  | `tests/test_strict_json_adoption_register.py`                         |
| pass `fail=` a lambda raising `ValueError` in `d1_launch_attempt_codec.py` | the existing duplicate-key test asserting `D1LaunchAttemptCodecError` |
| repoint one BOUNDARY SHA site back to `require_sha256_hex_any_case`        | that site's mixed-case test                                           |

## Planned commit branches

1. FOR-1575/FOR-1599 register trio and ladder_campaign adopt strict_json
2. FOR-1577 register facet: strict_loads replaces local pairs hooks
3. FOR-1613 boundary SHA callers in register/gate go lowercase-strict
