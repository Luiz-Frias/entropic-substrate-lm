# pre-for-1614: Audit persisted records and pin each codec's `_text` trim boundary

Authoritative brief for one pre-phase rung, written 2026-09-28 by the rung-brief lane. Plan: `docs/plans/p14-consolidation.plan.md`, section *Prerequisites and pre-phase rungs*, row 3. Linear: FOR-1614. Every premise below was measured on the live tree whose trunk ref `refactor/p14-consolidation` read `4e0ae6a` at authoring. The phase briefs cite `eb93d5d`; that drift is recorded for the conductor.

## Branch

| field                | value                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| branch               | `chore/for-1614-text-trim-semantics`                                                             |
| parent               | `refactor/p14-consolidation` after rungs 1 and 2 landed                                          |
| kind                 | pre-phase rung, not a manifest phase (ruling 12)                                                 |
| cascade `phase.mode` | `normal`, explicit                                                                               |
| `gates.liveRequired` | `false`                                                                                          |
| gates                | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":false}` |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree           |
| publish              | by the conductor through `just ship-it`; never a bare `gt submit`                                |

## Contract surface owned

| contract                                                                                                                                                                                                                   | kind     | frozen_after |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| the per-codec `_text` boundary, pinned: `d1_launch_attempt_codec` accepts untrimmed-but-nonblank and round-trips it byte for byte; `d1_registration_codec` and `ladder_campaign` reject it with their own type and message | behavior | true         |
| `docs/plans/measurements/p14-text-trim-audit.txt`: the searched corpus, method, counts, and what was not searched                                                                                                          | evidence | true         |

## Contracts consumed

- nothing beyond the trunk at the parent

## Sacred invariants

| invariant               | verbatim                                                                                                                                                                             | tripwire                                                                                                                                                                     | guard-teeth recipe                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| House rule 5            | "**TDD is non-negotiable, and guard-teeth prove both arms.** Failing test first. Then break the guarded thing and show the test goes RED — a test that cannot fail is not evidence." | a pin of today's behaviour passes on first run, so its RED arm is the mutation: flip that codec's `_text`, show the pin RED, revert, show GREEN, before the pin is committed | both arms per codec in the PR body                                         |
| House rule 3            | "**Never fix flakiness by raising a timeout or a retry budget.**"                                                                                                                    | the diff adds no timeout, deadline, retry count or backoff constant                                                                                                          | a reviewer greps the diff                                                  |
| House rule 4            | "**Comments record measurements, with the measurement.**"                                                                                                                            | every count in the audit file names its command and scope                                                                                                                    | a wrong measured claim is retracted explicitly in the commit that fixes it |
| Reader identity (R-PIN) | "the door frozen after p4 is the LABEL and the module-set SHAPE, while the composed digest below re-measures on every phase that edits a member's bytes (ruling R-PIN)"              | the diff's intersection with the 51 reader files is empty                                                                                                                    | proven at p0: flipping one byte in a member turns the regime-row tests RED |
| Failure semantics       | "measurement/eval path → fail LOUD, never a fail-open wrapper."                                                                                                                      | the audit script, if committed, raises on an unreadable record instead of skipping it                                                                                        | an unreadable record in the scanned corpus is reported, never dropped      |
| Layer contract          | `kernel < core < ladder < historical < composition`, enforced by `lint-imports` with zero exceptions since p13                                                                       | `lint-imports` green at the tip                                                                                                                                              | proven at p0 and p13                                                       |
| Exception surface       | "no change to any caller-visible exception type/message beyond what the remedy text states" (every p14 ticket's re-validation checklist)                                             | the pins assert each site's own exception type and message text                                                                                                              | wiring one site to another type turns its pin RED                          |

## Identity members

None `[M, comm of the three _text source files plus the envelope against both tuples: empty]`: no `ladder/register` or `ladder/gate` file is a reader-tuple member. No conductor re-pin.

## Scope

Premises `[M, code reading at 4e0ae6a]`:

- `ladder/register/d1_launch_attempt_codec.py:58-61`: rejects only non-`str` or blank after strip; an untrimmed value passes, and `parse_launch_attempt_ledger` checks only byte round-trip (`:172-186`).
- `ladder/register/d1_registration_codec.py:121-124` and `ladder/gate/ladder_campaign.py:48-51`: reject any value whose strip differs.
- 28 `def _text` exist in `src` in total `[M, grep]`. Only these three are in FOR-1614; the other 25 are listed in the audit file as data, unpinned.

Commit 1, always:

1. Locate where each record type is persisted by tracing its writers, then enumerate the corpus. Starting points: launch-attempt ledgers are embedded in finalized-cell custody (`ladder/run/closure/d1_finalized_cell_custody.py:177`, read back at `:510`), excluded capture (`ladder/run/closure/d1_excluded_capture.py:204`) and operator custody (`ladder/run/campaign/d1_operator_custody_adapter.py`); decision contracts travel base64-encoded (`ladder/review/d1_external_analysis_control.py:403`), so a schema-string grep misses them and the audit must decode; campaign designs are read through `read_campaign_design` (`ladder/gate/ladder_campaign.py:189`).
2. Count, per schema and per `_text` field, the records whose value is nonblank but untrimmed. Record the corpus paths, the method, the counts, and every location NOT searched (R2 objects not in the local DVC cache, the MLflow artifact store) as UNMEASURED.
3. Pin each of the three codecs through its public parse function, on a field whose only admission gate is `_text` (verify it: an enum or digest check behind `_text` would hide the boundary).

Authoring-time measurement, to be re-run by the lane: a search of `results/`, `data/`, `configs/`, `docs/` and `models/` in the main checkout for the five schema strings found only source copies inside a build context, zero persisted records `[M]`; the same search over the 88 files of the local `.dvc/cache` exited 1, zero matches `[M]`. Both legs cover plain-text schema strings only: a base64-embedded record would not match, so they bound the audit from below and do not replace it.

Commit 2, only when BOTH hold: the audit counts zero untrimmed records over a corpus the conductor accepts as complete, and the conductor records an explicit go in the lane report. Then `d1_launch_attempt_codec._text` becomes strict-trimmed with its own message, the launch-attempt pin flips, both arms shown. A non-zero count halts as DN-1614-READ-OLD. Without the go, the rung ends after commit 1.

Conflict named aloud: the conductor's instruction for this brief says "NO semantic change now"; the ratified plan's row 3 lands commit 2 whenever the count is zero. The gate above satisfies both until the conductor rules; the lane never takes commit 2 on the count alone.

## ACCEPTANCE

- `tests/test_text_trim_pins.py` (new): per codec, one untrimmed-but-nonblank value (leading space, trailing space, trailing newline) and one blank value, through the public parse function; the launch-attempt arm also asserts byte-identical re-serialization.
- Each pin's mutation arm shown RED then GREEN in the PR body.
- `docs/plans/measurements/p14-text-trim-audit.txt` (new) states scope, commands, per-schema counts, the untrimmed count, the unsearched locations, and the 28-row `_text` inventory with each row's semantics.
- Commit 2, if taken: the conductor's go is quoted in the PR body, and every existing test in `tests/test_d1_launch_attempt_codec.py` passes unmodified.
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

- Does NOT change any `_text` semantics in commit 1, nor any site but `d1_launch_attempt_codec._text` in commit 2.
- Does NOT pin or edit the other 25 `_text` definitions.
- Does NOT add a `_text` helper to `kernel/codec/strict_json.py`; the rail freezes four functions.
- Does NOT pull from DVC, list R2, or read the MLflow store; those legs are reported UNMEASURED for the conductor.
- Does NOT edit the three existing codec test files, which `p14-j-register` owns later.

## Allowed files

The escalation door: a file outside this envelope, a contract surface, or a sacred invariant means DO NOT EDIT; return `DECISION` with the verification you ran. Every path verified at `4e0ae6a` unless marked new.

- `tests/test_text_trim_pins.py` (new)
- `docs/plans/measurements/p14-text-trim-audit.txt` (new; the one carve-out under `docs/plans/`)
- `src/entropic_substrate/ladder/register/d1_launch_attempt_codec.py` (commit 2 only, gated as above)
- `.secrets.baseline` (standing rule)

Touch set: 2 new plus 1 conditional, 3 files. `d1_launch_attempt_codec.py` is in `p14-j-register`'s touch set, serial-after `[M, overlap against sprint-manifest.json]`.

Never edit, in this rung:

- `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `tools/mapping.json`, `tools/regime_rows.py`, `.importlinter`, `pyproject.toml`, `uv.lock`
- `docs/identity/*`, `tests/test_identity_epoch.py`
- everything under `docs/plans/` except the one new audit file; `docs/reviews/*`, `docs/verdicts/*`, `CLAUDE.md`, briefs, the manifest, engine scripts
- the 51 reader-tuple files

## Already ruled

| id               | status                                     | ruling                                                                                                                                   |
| ---------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 12               | OPERATOR 2026-09-28                        | FOR-1614 keeps per-site semantics now and becomes strict-trimmed after a persisted-record audit, as a `chore/` branch before the phases. |
| DN-1614-READ-OLD | OPERATOR 2026-09-28, deferred to the audit | Opens only if the audit finds untrimmed persisted records: the read-old window for them.                                                 |
| 14               | OPERATOR 2026-09-28                        | Abstraction admissible only when it consolidates 2+ concrete sites or stabilizes a public surface.                                       |

## Guard teeth by mutation

| mutation                                                            | must go RED                            |
| ------------------------------------------------------------------- | -------------------------------------- |
| make `d1_launch_attempt_codec._text` strict-trimmed (commit 1 only) | the launch-attempt permissive pin      |
| make `d1_registration_codec._text` permissive                       | the registration strict pin            |
| make `ladder_campaign._text` permissive                             | the campaign strict pin                |
| raise `TypeError` instead of the site's own error in one `_text`    | that site's message-and-type assertion |

## Planned commits

Conventional Commits, title <= 72 characters, `git commit -F`, hooks on, never `--no-verify`, never amend.

1. `test(register): pin _text trim boundaries and commit the record audit`
2. only on zero count plus the conductor's go: `fix(register): launch-attempt _text rejects untrimmed text`
