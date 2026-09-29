# pre-for-1617: Graceful operator stop for the H probe lane

Authoritative brief for one pre-phase rung, written 2026-09-28 by the rung-brief lane. Plan: `docs/plans/p14-consolidation.plan.md`, section *Prerequisites and pre-phase rungs*, row 2. Linear: FOR-1617. Every premise below was measured on the live tree whose trunk ref `refactor/p14-consolidation` read `4e0ae6a` at authoring. The phase briefs cite `eb93d5d`; that drift is recorded for the conductor, and the prelaunch ceremony re-probes every premise at the re-stamped SHA, which by then includes rung 1.

## Branch

| field                | value                                                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1617-graceful-operator-stop`                                                                                                     |
| parent               | `refactor/p14-consolidation` AFTER `fix/for-1567-checkpoint-publication-history` landed and was re-pinned                                 |
| kind                 | pre-phase rung, not a manifest phase (ruling 7)                                                                                           |
| cascade `phase.mode` | `arch`, always explicit                                                                                                                   |
| `gates.liveRequired` | `false`: `ladder/capture/adapters/mlflow.py` is outside the envelope; if the lane needs it, that is a DECISION and the gate flips to live |
| gates                | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":false}`                                          |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree                                                    |
| publish              | by the conductor, after its re-pin commit, through `just ship-it`; never a bare `gt submit`                                               |
| launch precondition  | no H run in flight or pending resume; see *Identity members*                                                                              |

## Contract surface owned

| contract                                                                                                                                                                                                                                  | kind     | frozen_after |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| R2 stop marker at `<request prefix>/control/stop.json`, its own schema name (NOT `h-probe-stop-v1`, which is the host control message at `worker/h_probe_controller.py:222-224` and `h_probe_watchdog.py:219`), carrying `request_sha256` | door     | true         |
| terminal event kind `stopped`, status `stopped_by_operator`, carrying the last committed global step and the final snapshot                                                                                                               | door     | true         |
| collection of a `stopped` terminal: `workload_verified` false; the controller then records `termination_verified` from the watchdog as for any terminal                                                                                   | behavior | true         |
| host verb `just h-probe-stop` over `ladder/run/worker/h_probe_stop.py`: writes the marker, waits with a required bound, prints the published keys                                                                                         | door     | true         |

## Contracts consumed

- rung 1, LANDED: per-snapshot publication is history-independent and its event carries the `publication` block; the stop path reuses both unchanged.

## Sacred invariants

| invariant               | verbatim                                                                                                                                                                             | tripwire                                                                                                                                                   | guard-teeth recipe                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| House rule 5            | "**TDD is non-negotiable, and guard-teeth prove both arms.** Failing test first. Then break the guarded thing and show the test goes RED — a test that cannot fail is not evidence." | every behaviour-changing test is committed RED before its fix commit                                                                                       | perform each mutation in *Guard teeth by mutation*, show RED, revert, show GREEN; both arms in the PR body |
| House rule 3            | "**Never fix flakiness by raising a timeout or a retry budget.**"                                                                                                                    | the diff adds no timeout, deadline, retry count or backoff constant; the verb's wait bound and poll interval are REQUIRED arguments with no default        | a reviewer greps the diff for timeout, retry, deadline, backoff, max_attempts                              |
| House rule 4            | "**Comments record measurements, with the measurement.**"                                                                                                                            | a changed comment that states a number cites how it was measured                                                                                           | a wrong measured claim is retracted explicitly in the commit that fixes it                                 |
| Reader identity (R-PIN) | "the door frozen after p4 is the LABEL and the module-set SHAPE, while the composed digest below re-measures on every phase that edits a member's bytes (ruling R-PIN)"              | ruling 7 exempts this rung from the rail-only rule: the diff's intersection with the 51 reader files equals exactly the 3 members under *Identity members* | proven at p0: flipping one byte in a member turns the regime-row tests RED until the conductor re-pins     |
| Failure semantics       | "measurement/eval path → fail LOUD, never a fail-open wrapper."                                                                                                                      | an unreadable or foreign-schema marker fails loud through the existing failure snapshot; no `except` swallows a poll error                                 | the invalid-marker test expects `execution_defect` with a failure snapshot                                 |
| Layer contract          | `kernel < core < ladder < historical < composition`, enforced by `lint-imports` with zero exceptions since p13                                                                       | `lint-imports` green; `runner.py` never imports the object port, the poll is a callback                                                                    | proven at p0 and p13: an upward import turns the layers contract RED                                       |
| Exception surface       | "no change to any caller-visible exception type/message beyond what the remedy text states" (every p14 ticket's re-validation checklist)                                             | existing pause, completion and failure tests pass unmodified                                                                                               | the existing paused-collection test stays GREEN byte for byte                                              |
| Flywheel prerequisite   | "no p14 phase or pre-phase rung launches an h-ladder flywheel experimental run, spends GPU, or starts a pod without a fresh operator green light obtained at that time" (registry)   | the verb is exercised only against a fake object port in tests; no RunPod or R2 call in the lane                                                           | any such action is a DECISION-NEEDED halt                                                                  |

## Identity members

3, all in the checkpoint tuple (`ladder/run/checkpoint.py:65-117`) and none in the source tuple (`ladder/capture/capture.py:29-37`) `[M, comm]`: `ladder/run/runner.py`, `ladder/run/worker_execution.py`, `ladder/run/object_worker.py`. Only the `h-checkpoint-reader-v2` composed rows re-measure. The plan's row names `worker_command.py`; it is NOT in this envelope, because `run_worker_command` already returns exit 0 for any event `run_object_worker` returns `[M, worker_command.py:176-228]`.

Re-pin: the conductor runs `tools/regime_rows.py` in a separate commit after the lane, touching `tests/test_identity_epoch.py` and one dated row in `docs/identity/read-old-table.md`. `checkpoint.py:304-305` refuses a foreign reader digest, so this re-pin again invalidates any checkpoint written under rung 1's digest; the image is rebuilt from this tip before any green-lit run.

## Scope

Premises `[M, code reading at 4e0ae6a]`: `run_probe_training` already takes `stop_after_seal`, observed only after a seal (`runner.py:162`, `:271-279`); the request `pause` must select a declared seal boundary (`worker_request.py:130-152`); collection admits only `complete`, `paused`, `failure` (`h_probe_collection.py:46-55`) and its `paused` branch requires `request.pause` (`:125-153`); `_admit_result` admits `paused` only (`object_worker.py:106-111`); `read_seal`'s continuation check needs consecutive steps, not a cadence (`seals.py:72-85`).

1. Runner: a second keyword callback, default `None`, polled after every committed step. When it reports a stop, that step falls through the `continue` at `runner.py:232-235` and is sealed. The training receipt keeps its three-status vocabulary and records `paused` (a stop is a pause at an operator-chosen step), so the status check at `runner.py:86-87` and `read_training_progress` keep their contract.
2. Worker: `object_worker` builds the poll from the object port and the request prefix; the result, the final snapshot and the terminal event carry `stopped_by_operator` and the last committed global step (`successful_optimizer_steps`). Exit 0 follows unchanged.
3. Host: `h_probe_events` gains the `stopped` kind; `h_probe_collection` gains a `stopped` branch that verifies the reported step against the training receipt and keeps `workload_verified` false. The controller then tears the pod down as it does for any terminal; touch `worker/h_probe_controller.py` only if its stop-reason mapping needs the new outcome.
4. Verb: `ladder/run/worker/h_probe_stop.py` (new) parses the launch record with `parse_probe_launch`, builds the object port with `build_worker_objects` (imported, not edited), publishes the marker idempotently, waits for the `stopped` event within the required bound, and prints the published snapshot keys. `justfile` gains only the `h-probe-stop` recipe.
5. Evaluation at seal time: already true at the epoch `[M]`: an evaluation step is a seal point, the report is written before the progress receipt, and `on_checkpoint` then snapshots `run/` (`runner.py:229-270` and `:304-314`; `object_worker.py:201-212`). Pin it with a test; if GREEN before any change, item 3 of the ticket closes as "already true, now pinned".

## Open before launch (conductor rules; the lane halts if unruled)

- DECISION-NEEDED, poll cost and failure exposure. A per-step poll is one R2 GET per step, whose latency is UNMEASURED, and under `max_attempts` 0 each poll is a new way for a transient R2 error to end a run. Recommendation: poll at every committed step as ruled, route the poll read through rung 1's bounded read retry, fail loud after exhaustion, and record per-poll seconds with rung 1's timers. This extends a retry that was ruled for verification reads only, hence the ruling. A yes-ruling holds this envelope only if rung 1's retry helper is importable from `worker_transport.py` without editing that file; otherwise `worker_transport.py` joins the envelope and the member count goes from 3 to 4, which the ruling must state.

## ACCEPTANCE

- `tests/test_h_probe_operator_stop.py` (new, RED-first): a marker appearing mid-phase in warm-up and in prediction yields a `stopped` event naming the last committed global step and every sealed artifact key (journal, checkpoint, evaluation report when one exists), and the worker returns normally.
- Boundary cases pinned at authoring: marker present before step 1 stops after sealing step 1; marker at a checkpoint step seals once, not twice; marker for another `request_sha256` is ignored; an unparseable marker fails loud with a failure snapshot.
- A collection test: a `stopped` terminal is collected with `workload_verified` false; `tests/test_h_probe_controller.py` shows that collection ending in `termination_verified: true` under the fake watchdog.
- The verb, against a fake port: writes the marker once (a second invocation is not an error), returns the published keys after the event appears, and exits non-zero at its bound without it.
- An evaluation report is present in the snapshot published by its own seal.
- Existing paused, complete and failure tests pass unmodified.
- remap_check at the tip: inversions <= 1, package cycles <= 4; mechanical counts at or under the epoch values below; outputs in the PR body.
- `git diff --name-only <parent>...HEAD` is a subset of *Allowed files*; its reader-file intersection equals the 3 members.
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

- Does NOT resume a stopped run; does NOT add a new training-receipt status.
- Does NOT reuse `paused` for the terminal event or the collection branch, and does NOT relax `request.pause`'s boundary rule.
- Does NOT touch `composition/cli.py` (a reader-tuple member calling `run_probe_training` by keyword) or `worker_command.py`.
- Does NOT add a stop control to the watchdog or change the host `h-probe-stop-v1` message.
- Does NOT edit the runbook memory or `docs/deploy-h100.md`; the conductor documents the verb as the only sanctioned stop.
- Does NOT launch pods, touch RunPod or R2, spend GPU, or start any h-ladder flywheel run (ruling 8).

## Allowed files

The escalation door: a file outside this envelope, a contract surface, or a sacred invariant means DO NOT EDIT; return `DECISION` with the verification you ran. Every path verified at `4e0ae6a` unless marked new.

- `src/entropic_substrate/ladder/run/runner.py`
- `src/entropic_substrate/ladder/run/worker_execution.py`
- `src/entropic_substrate/ladder/run/object_worker.py`
- `src/entropic_substrate/ladder/run/h_probe_events.py`
- `src/entropic_substrate/ladder/run/h_probe_collection.py`
- `src/entropic_substrate/ladder/run/worker/h_probe_controller.py` (only if the stop-reason mapping needs it)
- `src/entropic_substrate/ladder/run/worker/h_probe_stop.py` (new)
- `justfile` (add the `h-probe-stop` recipe only)
- `tests/test_h_probe_operator_stop.py` (new)
- `tests/test_sentence_runner.py`, `tests/test_sentence_worker_execution.py`, `tests/test_sentence_object_worker.py`, `tests/test_h_probe_events.py`, `tests/test_h_probe_collection.py`, `tests/test_h_probe_controller.py`
- `tests/test_identity_epoch.py`, `docs/identity/read-old-table.md` (conductor re-pin commit only, never the lane)
- `.secrets.baseline` (standing rule: mechanical companion, digest literals only)

Touch set: 13 existing plus 2 new, 15 files; the manifest phases share none `[M]`.

Never edit, in this rung:

- `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile` beyond the one recipe, `scripts/run_ladder.sh`, `tools/mapping.json`, `tools/regime_rows.py`, `.importlinter`, `pyproject.toml`, `uv.lock`
- `docs/identity/*` and `tests/test_identity_epoch.py`, except the conductor's re-pin commit
- `docs/plans/*`, `docs/reviews/*`, `docs/verdicts/*`, `CLAUDE.md`, `docs/architecture.md`, briefs, the manifest, engine scripts
- every reader-tuple file not listed under *Identity members*

## Already ruled

| id       | status               | ruling                                                                                                        |
| -------- | -------------------- | ------------------------------------------------------------------------------------------------------------- |
| 7        | OPERATOR 2026-09-28  | FOR-1617 is a `fix/` branch stacked on the trunk before the phases, with its own conductor re-pin.            |
| 8        | OPERATOR 2026-09-28  | No h-ladder flywheel run without a fresh operator green light; no pods, no spend.                             |
| H-PREREQ | OPERATOR 2026-09-28  | FOR-1617 lands before any flywheel run; only the sanctioned stop path ends a run (runbook memory, section 6). |
| R-PIN    | CONDUCTOR 2026-09-22 | Rewriting a reader-tuple member re-pins its regime rows in the same change; label and module set unchanged.   |

Conflict named aloud: the registry says `tests/test_identity_epoch.py` is touched "exactly once for the whole sprint"; ruling 7 grants this rung its own conductor re-pin. This brief follows ruling 7.

- **P14-1617-POLL (conductor, autonomous mode, 2026-09-28; DECISION-NEEDED for the operator's morning review, proceed under it):** the stop marker is checked at every CHECKPOINT and EVALUATION boundary, never at every committed step, so the training loop gains no per-step R2 read and its latency stays out of the hot path; the check uses the same bounded-retry verification-READ path that pre-for-1567 lands (a control read is a read, never a publish), so no new retry budget is introduced and `worker_transport.py` needs no separate edit for it. Consequence recorded: a stop lands within one checkpoint interval (250 steps at the current profile), which is the bound the operator accepted when ruling 'cut after the capture'. Also recorded: each reader re-pin (1567, then 1617) makes checkpoints written under the previous digest unreadable, so the H image is rebuilt after 1617 and before any flywheel run (already a flywheel prerequisite). Ruling 7 overrides the registry's 'touched exactly once' note: each rung carries its own conductor re-pin.

## Guard teeth by mutation

| mutation                                         | must go RED                                         |
| ------------------------------------------------ | --------------------------------------------------- |
| ignore the marker                                | the mid-phase stop test (the run completes instead) |
| stop without sealing the current step            | the last-committed-step assertion                   |
| accept a marker for another request              | the foreign-marker test                             |
| collect `stopped` with `workload_verified` true  | the collection test                                 |
| drop `stopped` from the collected terminal names | the controller test                                 |
| drop `run` from the snapshot's sections          | the evaluation-in-snapshot test                     |
| return from the verb before the event appears    | the verb bound test                                 |

## Planned commits

Conventional Commits, title <= 72 characters, `git commit -F`, hooks on, never `--no-verify`, never amend.

1. `test(run): pin operator stop, collection and evaluation publication` (RED)
2. `feat(run): honour an R2 stop marker at the next committed step`
3. `feat(run): collect a stopped_by_operator terminal`
4. `feat(just): h-probe-stop host verb`
5. conductor: `tools/regime_rows.py` re-pin of `h-checkpoint-reader-v2`, then `just ship-it`
