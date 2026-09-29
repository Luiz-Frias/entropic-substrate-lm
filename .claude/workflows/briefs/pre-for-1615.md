# pre-for-1615: Rename the loose kernel SHA check and audit its callers

Authoritative brief for one pre-phase rung, written 2026-09-28 by the rung-brief lane. Plan: `docs/plans/p14-consolidation.plan.md`, section *Prerequisites and pre-phase rungs*, row 4, and *Why FOR-1615 splits this way* (P14-SHA-SEQ, ratified 2026-09-28). Linear: FOR-1615. Every premise below was measured on the live tree whose trunk ref `refactor/p14-consolidation` read `4e0ae6a` at authoring. The phase briefs cite `eb93d5d`; that drift is recorded for the conductor.

## Branch

| field                | value                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| branch               | `chore/for-1615-lowercase-sha-boundary`                                                           |
| parent               | `refactor/p14-consolidation` after rungs 1 to 3 landed; the last rung before `p14-r1-strict-json` |
| kind                 | pre-phase rung, not a manifest phase (ruling 12)                                                  |
| cascade `phase.mode` | `normal`, explicit                                                                                |
| `gates.liveRequired` | `false`                                                                                           |
| gates                | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":false}`  |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree            |
| publish              | by the conductor through `just ship-it`; never a bare `gt submit`                                 |

## Contract surface owned

| contract                                                                                                                                                           | kind     | frozen_after |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | ------------ |
| `kernel.identity.d1_primitives.require_sha256_hex_any_case(value: str, *, label: str) -> None`, the renamed loose predicate, behaviour and messages byte-identical | door     | true         |
| the old name `require_sha256` no longer resolves from `kernel.identity.d1_primitives`; no alias, no shim (internal consumers cut over in the same change)          | door     | true         |
| `docs/plans/measurements/p14-sha-callers.txt`: one row per call site, classified boundary or in-process, with its owning adoption phase                            | evidence | true         |

## Contracts consumed

- nothing beyond the trunk at the parent. The lowercase-strict `require_lowercase_sha256` is born later in `kernel/codec/strict_json.py` by the rail; this rung defines no new predicate.

## Sacred invariants

| invariant               | verbatim                                                                                                                                                                             | tripwire                                                                                                                                       | guard-teeth recipe                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| House rule 5            | "**TDD is non-negotiable, and guard-teeth prove both arms.** Failing test first. Then break the guarded thing and show the test goes RED — a test that cannot fail is not evidence." | the absence and audit tests are committed RED before the rename; the any-case pin passes on today's code, so its RED arm is the mutation below | both arms in the PR body                                                   |
| House rule 3            | "**Never fix flakiness by raising a timeout or a retry budget.**"                                                                                                                    | the diff adds no timeout, deadline, retry count or backoff constant                                                                            | a reviewer greps the diff                                                  |
| House rule 4            | "**Comments record measurements, with the measurement.**"                                                                                                                            | the audit's counts name the AST command that produced them                                                                                     | retraction below: the ticket's closure-manifest checklist item             |
| Reader identity (R-PIN) | "the door frozen after p4 is the LABEL and the module-set SHAPE, while the composed digest below re-measures on every phase that edits a member's bytes (ruling R-PIN)"              | the diff's intersection with the 51 reader files is empty                                                                                      | proven at p0: flipping one byte in a member turns the regime-row tests RED |
| Failure semantics       | "measurement/eval path → fail LOUD, never a fail-open wrapper."                                                                                                                      | zero behaviour change: every call site keeps its argument, label and exception                                                                 | existing tests asserting the digest messages pass unmodified               |
| Layer contract          | `kernel < core < ladder < historical < composition`, enforced by `lint-imports` with zero exceptions since p13                                                                       | `lint-imports` green at the tip                                                                                                                | proven at p0 and p13                                                       |
| Exception surface       | "no change to any caller-visible exception type/message beyond what the remedy text states" (every p14 ticket's re-validation checklist)                                             | the message `must be a 64-character hexadecimal SHA-256 digest` is unchanged at `d1_primitives.py:39-44`                                       | an edited message turns the existing message assertions RED                |

## Identity members

None `[M, comm of the 35 source files against both tuples]`. `kernel/identity/d1_primitives.py` is in no identity epoch: `tests/test_identity_epoch.py` does not reference it. No conductor re-pin.

Retraction, house rule 4: the ticket's checklist asks to re-run "the 52-entry D1 closure manifest fixture". That row no longer exists: it was measured at 73 entries at p0 and dropped at p2 (`tests/test_identity_epoch.py` module docstring, the p2 retraction paragraph). Nothing to re-run.

## Scope

Premises `[M, AST import scan of src, tests, scripts, tools at 4e0ae6a]`:

- The loose predicate lives at `src/entropic_substrate/kernel/identity/d1_primitives.py:37-44` and accepts any case via `string.hexdigits`; one internal call at `:227`.
- 34 `src` files import it by name, with 60 name-load references among them; `tests`, `scripts` and `tools` import it 0 times. None of the 34 is a reader-tuple member.
- The same name also exists, lowercase-strict, at `ladder/capture/records/d1_telemetry_payloads.py:68`. After this rung that is the only `require_sha256` in `src`, which removes the collision the ticket names.

Work:

1. RED first: the absence test and the audit test below.
2. Rename the definition and its internal call; repoint the 34 importers mechanically, import lists re-sorted by the formatter, no other edit.
3. Commit the audit. Classify each call site: **boundary** when the digest reached the function as parsed JSON, a file, a CLI argument, an environment value or an R2 payload; **in-process** when it was produced by `hashlib` in the same process or is an already-admitted typed value. Columns: `path:line`, enclosing symbol, class, one-line evidence, owning adoption phase. Owners `[M, overlap against sprint-manifest.json]`: `p14-j-register` 8 files, `p14-t-historical` 2, `p14-j-kernel-capture` 4, `p14-j-analyze-review-file` 6, `p14-j-run` 14, and `d1_primitives.py` itself in `p14-t-dead`.

<details><summary>The 34 importing files, relative to <code>src/entropic_substrate/</code></summary>

`composition/entrypoints/d1_r2_trainer_mailbox_worker.py`, `historical/d_ladder/d_ladder_remote_request.py`, `historical/d_ladder/d_ladder_worker_command.py`, `ladder/analyze/d1_inspect_request.py`, `ladder/analyze/d1_inspect_values.py`, `ladder/analyze/d1_machine_analysis.py`, `ladder/analyze/ladder_analysis.py`, `ladder/analyze/ladder_registration.py`, `ladder/capture/custody/d1_capture_aggregate_codec.py`, `ladder/capture/custody/d1_capture_preparation_custody.py`, `ladder/capture/reader/d1_journal_snapshot.py`, `ladder/capture/records/d1_calibration_records.py`, `ladder/gate/ladder_campaign.py`, `ladder/register/d1_confirmatory_matrix.py`, `ladder/register/d1_launch_attempts.py`, `ladder/register/d1_plan.py`, `ladder/register/d1_provenance.py`, `ladder/register/d1_registered_design.py`, `ladder/register/d1_registration_codec.py`, `ladder/register/ladder.py`, `ladder/review/d1_execution_receipt.py`, `ladder/run/campaign/d1_campaign_checkpoint.py`, `ladder/run/campaign/d1_campaign_request.py`, `ladder/run/campaign/d1_operator_evidence.py`, `ladder/run/campaign/d1_operator_values.py`, `ladder/run/ckptsync/d1_checkpoint_dvc_recovery.py`, `ladder/run/ckptsync/d1_checkpoint_dvc_sync.py`, `ladder/run/closure/d1_finalized_cell_custody.py`, `ladder/run/construction/d1_r2_runtime_composition.py`, `ladder/run/transport/d1_trainer_evidence_adapter.py`, `ladder/run/transport/d1_trainer_mailbox_contracts.py`, `ladder/run/transport/d1_trainer_mailbox_worker.py`, `ladder/run/transport/d1_trainer_object_transport.py`, `ladder/run/transport/d1_trainer_process_contracts.py`

</details>

## ACCEPTANCE

- `tests/test_sha256_caller_audit.py` (new, RED-first):
  - importing `require_sha256` from `kernel.identity.d1_primitives` raises `ImportError`, and the module has no such attribute;
  - `require_sha256_hex_any_case` accepts an uppercase and a mixed-case 64-character digest and rejects 63, 65 and a non-hex character with the unchanged message;
  - an AST scan of `src` run inside the test finds every importer of `require_sha256_hex_any_case` listed in the audit (live set a subset of audited set), and the audit's row counts equal the counts recorded in its own header.
- The subset form is deliberate: the adoption phases repoint boundary callers away later and must stay GREEN without editing `docs/plans/`. Equality at this rung's tip, live set equal to audited set at 34 files, is shown as command output in the PR body.
- `git grep -n "require_sha256\b" -- src` lists only `d1_telemetry_payloads.py` and its own importers.
- Zero behaviour change: every existing test passes unmodified, except a docstring mention in `tests/test_d1_launch_attempts_refusal.py:16`, which may be updated to the new name.
- remap_check at the tip: inversions <= 1, package cycles <= 4; mechanical counts at or under the epoch values; outputs in the PR body.
- `git diff --name-only <parent>...HEAD` is a subset of *Allowed files*; its reader-file intersection is empty.
- `just check` green; exit gate `just pc-ci` through `uv run python scripts/run_exit_gate.py`; `lint-imports` clean. One full suite at a time, with `PYTEST_XDIST_AUTO_NUM_WORKERS=2`.

Ratchet commands:

```text
uv run python tools/import_graph.py src/entropic_substrate <out>
uv run python - <out>/mapping.json   # stdin: the reproducing procedure in docs/plans/measurements/p13-sweep-remap.txt
uv run python tools/remap_check.py <out>/edges.json <out>/mapping.json
# must print: layer inversions <= 1, package-level cycles <= 4 (epoch 4e0ae6a (src tree identical to 651af30): 1 and 4, 378 modules, 1805 edges)
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

- Does NOT repoint any caller to a lowercase-strict check; the adoption phases do, per the audit.
- Does NOT add `require_lowercase_sha256` or any predicate anywhere; the rail owns it (P14-SHA-SEQ).
- Does NOT touch `ladder/capture/records/d1_telemetry_payloads.py`; the rail promotes its contract and `p14-j-kernel-capture` adopts it.
- Does NOT leave a compatibility alias: internal coordinated consumers take a same-change cutover.
- Does NOT touch `require_commit` or any other `d1_primitives` function.

## Allowed files

The escalation door: a file outside this envelope, a contract surface, or a sacred invariant means DO NOT EDIT; return `DECISION` with the verification you ran. Every path verified at `4e0ae6a` unless marked new.

- `src/entropic_substrate/kernel/identity/d1_primitives.py`
- the 34 importing files listed under *Scope*, the import line and the call names only
- `tests/test_sha256_caller_audit.py` (new)
- `tests/test_d1_launch_attempts_refusal.py` (docstring name only)
- `docs/plans/measurements/p14-sha-callers.txt` (new; the one carve-out under `docs/plans/`)
- `.secrets.baseline` (standing rule)

Touch set: 35 source files, 1 docstring-only test, 2 new, 38 files. Each of the 35 sits in exactly one later phase's touch set, serial-after `[M]`.

Never edit, in this rung:

- `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `tools/mapping.json`, `tools/regime_rows.py`, `.importlinter`, `pyproject.toml`, `uv.lock`
- `docs/identity/*`, `tests/test_identity_epoch.py`
- everything under `docs/plans/` except the one new audit file; `docs/reviews/*`, `docs/verdicts/*`, `CLAUDE.md`, briefs, the manifest, engine scripts
- the 51 reader-tuple files

## Already ruled

| id          | status                             | ruling                                                                                                                                                                |
| ----------- | ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 12          | OPERATOR 2026-09-28                | FOR-1614 and FOR-1615 land as pre-phase rungs; SHA is lowercase-strict at every boundary; the loose kernel function is renamed `require_sha256_hex_any_case`.         |
| P14-SHA-SEQ | PLANNING LANE, ratified 2026-09-28 | The chore lands the rename and the audit; `require_lowercase_sha256` lives in strict_json; each adoption phase repoints the audited BOUNDARY callers in its packages. |
| 4           | OPERATOR 2026-09-28                | strict_json home `kernel/codec/strict_json.py` with a `fail` callback.                                                                                                |

## Guard teeth by mutation

| mutation                                                             | must go RED                        |
| -------------------------------------------------------------------- | ---------------------------------- |
| re-add `require_sha256` as an alias of the renamed function          | the absence test                   |
| make `require_sha256_hex_any_case` lowercase-only                    | the any-case acceptance arm        |
| delete one row from the audit                                        | the live-subset-of-audit assertion |
| add a new importer outside the audit in a scratch module under `src` | the live-subset-of-audit assertion |

## Planned commits

Conventional Commits, title <= 72 characters, `git commit -F`, hooks on, never `--no-verify`, never amend.

1. `test(identity): pin the any-case SHA rename and its caller audit` (RED)
2. `refactor(identity): rename require_sha256 to require_sha256_hex_any_case`
3. `docs(plans): commit the p14 SHA caller audit`
