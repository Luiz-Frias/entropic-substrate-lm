# pre-for-1567c: `publication` block and bounded verify-read retry on the FINAL snapshot leg

Authoritative brief for one pre-phase rung, written 2026-09-29 by the conductor on an operator ruling (DN-1567-C → option C: "add a small branch for option C because you never know if and when you might need it until you need it"). Plan: `docs/plans/p14-consolidation.plan.md`, pre-phase rungs table (row added by this commit). Linear: FOR-1621 (sub-issue of FOR-1567). Every premise below was measured on `fix/for-1617-graceful-operator-stop` at its conductor commit `7055158`.

## Branch

| field                | value                                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| branch               | `fix/for-1567-final-leg-publication`                                                                   |
| parent               | `fix/for-1617-graceful-operator-stop` (top of the rung stack: trunk → 1567 → 1617 → this)              |
| kind                 | pre-phase rung, not a manifest phase (ruling 7)                                                        |
| cascade `phase.mode` | `arch`, always explicit                                                                                |
| `gates.liveRequired` | `false`: no file under `ladder/capture/adapters/` is in the envelope                                   |
| gates                | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":false}`       |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree                 |
| publish              | by the conductor after its re-pin commit and the live gate, draft PR onto the parent branch (fallback) |
| hand-off             | DG-A applies (both production members are reader-tuple members): REPIN-HANDOFF, one conductor commit   |

## Contract surface owned

| contract                                                                                                                                                                                                                                                                                                                                             | kind     | frozen_after |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `h-worker-event-v1` `complete` event carries the `publication` block (same inner key set as the checkpoint events: `l1_seconds,l2_seconds,l3_seconds,objects_put,objects_skipped,verification_reads,retries`); written always (write-new); a `complete` event WITHOUT the block is still admitted by both readers for one rollback window (read-old) | door     | true         |
| final snapshot leg publishes O(new) objects through the run's ledger and retries its verification READS through `VerificationReadRetry` (3 attempts, existing class); never a PUT                                                                                                                                                                    | behavior | true         |

Door amendment is OPERATOR-RULED 2026-09-29 (DN-1567-C). Started and failure snapshots keep their current contract (no `publication`, no retry) — out of scope, stated in *Non-goals*.

## Contracts consumed

- pre-for-1567's `PublicationLedger`, `VerificationReadRetry`, `publish_worker_snapshot(..., ledger=, retry=)`, the checkpoint-leg `publication` block shape (`object_worker.py`), `h_probe_events._admit_publication`.
- pre-for-1617's `stopped` terminal (untouched; a stopped run's terminal snapshot is NOT the final snapshot of this rung unless the lane measures that it routes through the same `final_snapshot` stage — if it does, the same ledger + retry apply and the `stopped` event gets the block under the same read-old/write-new rule; report which).

## Sacred invariants

Same table as `pre-for-1567.md` (house rules 3 with the ruled carve-out, 4, 5; reader identity R-PIN; fail-LOUD; layer contract; exception surface; flywheel prerequisite). The carve-out extends to the final leg's verification READS only; `worker_command.py:68` `max_attempts` stays 0; no new numeric literal for attempts or backoff (reuse `VerificationReadRetry()` defaults).

## Identity members

Expected, 2, both in the checkpoint tuple: `ladder/run/object_worker.py`, `ladder/run/worker_completion.py`. None in the source tuple. `h_probe_events.py` and `h_probe_collection.py` are not tuple members.

Re-pin: the conductor, never the lane (DG-A). Consequence: the H image must be rebuilt from the stacked tip before any green-lit run (already true after 1567 and 1617).

## Scope

Measured at `7055158`:

- `object_worker.py` `stage = "final_snapshot"` publishes with `publish_worker_snapshot(...)` WITHOUT `ledger=`/`retry=`; the checkpoint leg passes both. The `complete` event is built by `_event(...)` without a `publication` block.
- `worker_completion.py:79` admits the `complete` event with `recovery_record(event, <exact keys>)` — exact key-set equality; a `publication` key today raises.
- `h_probe_events.py:153` admits the `publication` extra only when `kind == "checkpoint"`; `:189` validates it via `_admit_publication`.

Work, in order:

1. Final leg: pass the run's `ledger` and a fresh `VerificationReadRetry()` to the final-snapshot `publish_worker_snapshot` call; measure L3 seconds around it; build the `publication` block exactly as the checkpoint leg does (l1/l2 from the timing carrier at the final seal, or `0.0` when the runner exposes none for the final boundary — state which, with the measurement).
2. Write-new: `_event("complete", ...)` carries `publication` always.
3. Read-old in both readers: `worker_completion.py` admits the `complete` event with OR without `publication` (mirror `h_probe_events.py:153`'s extra-key pattern; validate the block when present with the same `_admit_publication` rules — reuse, do not copy: if the validator must be shared, it moves to the reader-neutral site the lane measures as lowest-coupling and both import it; report the choice); `h_probe_events.py` admits the extra for `kind in {"checkpoint", "complete"}`.
4. Host consumers: grep every reader of the `complete` event (`h_probe_collection.py`, `h_probe_controller.py`, `h_probe_watchdog.py`, host materialize) and pin that an event with the block passes unchanged; edit only if a consumer re-validates the key set.

## ACCEPTANCE

- `tests/test_checkpoint_publication_history.py` (extend): the final snapshot after k checkpoints PUTs only files new since checkpoint k, plus its marker and event; mutation: drop the ledger from the final leg → RED.
- `tests/test_sentence_worker_transport.py` or the history file: on the final leg, two transient read failures then success publishes once and records `retries: 2`; three failures raise the original class; `put_object` exactly once in both arms.
- `tests/test_sentence_worker_completion.py`: a `complete` event WITH and WITHOUT `publication` both admit (read-old); with an invalid block → raises. Mutation: drop the extra-key admission → the write-new event test RED.
- `tests/test_h_probe_events.py`: same two arms for `kind == "complete"`; checkpoint arms unchanged.
- `tests/test_sentence_object_worker.py`: the worker always writes the block on `complete` (write-new). Mutation: drop it from `_event("complete")` → RED.
- Started and failure snapshots: existing tests pass unmodified.
- remap_check at the tip: inversions <= 1, package cycles <= 4 (ratchet commands as in `pre-for-1567.md`); mechanical counts at or under epoch (the block builder is shared with the checkpoint leg, net +0 `separators` literals).
- `git diff --name-only <parent>...HEAD` ⊆ *Allowed files*; reader-tuple intersection ⊆ *Identity members*.
- `just check` green at the conductor's tip; exit gate `just pc-ci` via `scripts/run_exit_gate.py`; lint-imports clean; one suite at a time under the host lock.

## Non-goals

- Does NOT retry the started or failure snapshot legs, any PUT, list or request read; does NOT touch `worker_command.py:68`.
- Does NOT change any profile field, schedule, deadline or budget.
- Does NOT change the `stopped` terminal's contract unless the measurement in *Contracts consumed* says it shares the final-snapshot stage (then: same rule, reported).
- Does NOT build an image, launch pods, spend, or start a flywheel run (ruling 8).

## Allowed files

- `src/entropic_substrate/ladder/run/object_worker.py`
- `src/entropic_substrate/ladder/run/worker_completion.py`
- `src/entropic_substrate/ladder/run/h_probe_events.py`
- `src/entropic_substrate/ladder/run/h_probe_collection.py` (only if it re-validates the `complete` key set)
- `tests/test_checkpoint_publication_history.py`, `tests/test_sentence_worker_transport.py`, `tests/test_sentence_worker_completion.py`, `tests/test_h_probe_events.py`, `tests/test_sentence_object_worker.py`, `tests/test_h_probe_collection.py`
- `tests/test_identity_epoch.py`, `docs/identity/read-old-table.md` (conductor re-pin commit only)
- `.secrets.baseline` (mechanical companion)

Never edit: as in `pre-for-1567.md` (`.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `tools/mapping.json`, `tools/regime_rows.py`, `.importlinter`, `pyproject.toml`, `uv.lock`, plans, briefs, the manifest, engine scripts, every reader-tuple file not listed under *Identity members*).

## Already ruled

| id              | status               | ruling                                                                                                                               |
| --------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 7, 8            | OPERATOR 2026-09-28  | own branch on the stack; no flywheel run, no pods, no spend                                                                          |
| 9-sub           | OPERATOR 2026-09-28  | bounded retry on R2 verification READS only, count recorded in the receipt                                                           |
| DN-1567-C       | OPERATOR 2026-09-29  | option C: amend the `h-worker-event-v1` `complete` event (read-old/write-new `publication`), retry the final leg                     |
| DG-A/B/C        | CONDUCTOR 2026-09-28 | staged hand-off; evaluation_records in 1567's envelope; checkpoint-leg-only in 1567 (superseded here for the final leg by DN-1567-C) |
| 7, 8 (standing) | CONDUCTOR 2026-09-28 | host suite lock; `just pc-ci` base `origin/main`                                                                                     |

## Guard teeth by mutation

| mutation                                                            | must go RED                                      |
| ------------------------------------------------------------------- | ------------------------------------------------ |
| republish every accumulated path on the final leg (drop the ledger) | the O(new) final-leg test                        |
| retry the PUT instead of the read on the final leg                  | the `put_object` exactly-once assertion          |
| swallow the third read failure                                      | the exhaustion test expecting the original class |
| drop `publication` from `complete`'s accepted keys (either reader)  | the write-new event test                         |
| drop the block from `_event("complete")`                            | the worker always-writes test                    |

## Planned commits

ONE conductor commit per DG-A: `fix(run): publication block and verify-read retry on the final snapshot leg` (lane change set + re-pin). RED-first evidence lives in the lane's report.
