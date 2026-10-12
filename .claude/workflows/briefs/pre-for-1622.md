# pre-for-1622: MLflow metric writes off the training hot path (async sink)

Authoritative brief for pre-phase rung 6, written 2026-09-29 by the conductor on an operator ask ("background the persistence logic so it doesn't interrupt the hot path"). Linear: FOR-1622. Worked by an Opus agent in the parallel worktree `.claude/worktrees/rung-for-1622` on branch `feat/for-1622-mlflow-async-sink`, based on the trunk `refactor/p14-consolidation` @ `3c8db2a`; the conductor rebases it onto the trunk tip at the next phase gap and runs the lifecycle's L2 / review / validate before publication.

## Measured (code reading @ 3c8db2a)

- `ladder/run/runner.py:223`: `sink.log(step_row)` on EVERY optimizer step; `runner.py:267`: `sink.log(report)` per evaluation (line numbers re-measured by the lane at 3c8db2a; the conductor's first reading used the 1617 branch).
- `ladder/capture/adapters/composite.py:85`: `CompositeSink.log` calls the authoritative `JsonlSink.log` (one line write) then every best-effort sink's `log`.
- `ladder/capture/adapters/mlflow.py:373`: `MlflowSink.log` extracts metrics and calls `self._client.log_metric(run_id, key, value, step=...)` ONCE PER METRIC, synchronously (one HTTP round trip each); failures are counted in `_failed_writes` and warned, never raised (best-effort contract). `finish()` (:419) stops the system-metrics monitor, uploads the results JSONL artifact, ends the run.
- `mlflow==3.15.0` (installed): `MlflowClient.log_batch(run_id, metrics=[Metric(key, value, timestamp, step), ...])` exists; `log_metric(..., synchronous=False)` exists (experimental async queue).
- `mlflow.py` is NOT a reader-tuple member (no `capture.adapters` entry in `checkpoint._READER_MODULES` or `capture._READER_MODULES` at 3c8db2a): no identity re-pin.
- The `MetricSink` port (`ladder/capture/ports/metric_sink.py`) is `log / log_artifact / finish`. It is a public boundary of the p3b door: this rung does NOT widen it.

## Contract surface owned

| contract                                                                                                                                                                                                                                                                                       | kind     | frozen_after |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ------------ |
| `MlflowSink.log` never performs network I/O on the caller's thread: it enqueues; one background worker drains a BOUNDED queue into `log_batch` calls; `finish()` drains with a bounded wait, then runs the existing finish sequence                                                            | behavior | true         |
| Best-effort semantics unchanged: a failed batch counts every record it carried in `failed_writes` and warns once per batch; a dead worker is detected and reported once at `finish()` (and at the next `log()` call), never swallowed; `log()` never raises for network reasons                | behavior | true         |
| Backpressure: a full queue BLOCKS the producer (never drops): MLflow must never disagree with the JSONL by omission; the bound is a declared budget whose value is recorded next to its definition with the reasoning (records per step × steps per checkpoint interval, or a measured number) | behavior | true         |

## Contracts consumed

- `MetricSink` port shape unchanged; `CompositeSink` unchanged; `build_sink_for_run` unchanged.
- `tests/test_sinks.py`, `tests/test_tracking_sinks.py`, `tests/test_mlflow_live.py`: existing arms stay green unmodified except where a test asserted per-metric `log_metric` calls on the fake client (restate those to the batch shape and say so in the report).
- MEASURED by the lane at 3c8db2a (widening ruled by the conductor 2026-09-29): `tests/test_field_calibration.py:541,551,845,859`, `tests/test_gram_penalty.py:470,481` and `tests/test_d1_tracking.py:285,301` read the fake client's metrics right after `sink.log(...)` with no `finish()`, which races an async worker; each gets ONE added `flush()` line before the read (plus a thread join where needed) and nothing else. `test_d1_tracking.py:222` overrides `log_metric` to record steps for the :301 assertion: `FakeMlflowClient.log_batch` therefore delegates per metric to `self.log_metric(...)` so overrides keep their teeth, with its own mutation (a bypassing `log_batch` must turn :301 RED).

## Sacred invariants

- House rule 3 (no timeouts/retries to mask flakiness): the bounded `finish()` drain wait is a declared budget with the measurement, not a retry; no retry is added anywhere.
- House rule 4: comments record measurements with the measurement.
- House rule 5: TDD RED-first; every guard below proven by mutation, both arms quoted.
- Failure semantics: measurement path fails LOUD — but this sink is BEST-EFFORT by existing contract (the JSONL is authoritative); the loud part is the accounting: nothing is lost silently, `failed_writes` and a dead worker are always reported.
- Layer contract: `lint-imports` green; `mlflow` imported only in this adapter.
- The unit suite spawns no thread against a real client: tests inject a fake client (the `client=` seam) and, where a thread is needed, a fake that records calls; the worker thread in tests must terminate (join) inside the test.

## Scope

1. RED-first tests in `tests/test_sinks.py` (or a new `tests/test_mlflow_async_sink.py`): (a) `log()` returns before the fake client's write completes (fake client with an injectable barrier/latch, not a sleep); (b) all queued metrics reach the fake client after `finish()`, batched, same keys/values/steps as the per-metric shape; (c) a batch failure counts every record in `failed_writes` and does not raise; (d) a worker that dies (fake client raising `BaseException`-class or the worker loop crashing) is reported at `finish()` and the next `log()`; (e) a full queue blocks the producer until drained (bounded queue with maxsize 1 in the test), never drops; (f) `finish()` twice is idempotent (existing test).
2. `MlflowSink`: `queue.Queue(maxsize=BUDGET)`, one `threading.Thread(daemon=True)` started lazily on the first `log()` (never at construction, so untracked runs pay nothing), worker loop drains all currently queued items into one `log_batch` call (chunk at MLflow's 1000-metric batch limit), `flush(timeout)` public method on the CLASS (not the port) used by `finish()`; `finish()` drains with a bounded wait recorded with its reasoning, then the existing sequence.
3. Report the measured per-step overhead before/after with the fake client (a counting fake, not a network) in the PR body, labelled as a unit-test measurement, not a pod measurement.

## Non-goals

- Does NOT change the `MetricSink` port, `CompositeSink`, `JsonlSink`, the runner, `worker_command.py`, or any reader-tuple member.
- Does NOT touch the R2 publication path (FOR-1623 is a separate, measurement-gated rung).
- Does NOT run the live gate, build a container, or start an MLflow server: the conductor runs `just container-test` at the gap (the adapter is live-required).
- Does NOT add a retry anywhere.

## Allowed files

- `src/entropic_substrate/ladder/capture/adapters/mlflow.py`
- `tests/test_sinks.py`, `tests/test_tracking_sinks.py`, `tests/test_mlflow_async_sink.py` (new, optional)
- `tests/test_field_calibration.py`, `tests/test_gram_penalty.py`, `tests/test_d1_tracking.py` (ruled 2026-09-29: one `flush()` line before each read-after-log, thread joins, nothing else)
- `.secrets.baseline` (mechanical companion only if a literal moves)

Never edit anything else. If correctness would require another file (the port, the composite, the runner), STOP and report a DECISION with the verification you ran.

## ACCEPTANCE

- Tests (a)–(f) above GREEN; each of the FIVE mutations RED then GREEN after revert (the fifth: a fake `log_batch` that bypasses `log_metric` → `test_d1_tracking.py:301` RED): make `log()` write synchronously → (a) RED; drop the `finish()` drain → (b) RED; swallow a batch failure without counting → (c) RED; drop-on-full instead of block → (e) RED.
- `just check` exit 0 at the branch tip (under the host suite lock, no worker cap); `lint-imports` clean.
- No change under `src/` outside `mlflow.py` (`git diff --name-only 3c8db2a...HEAD`).
- Commit ladder: one or two Conventional Commits via `git commit -F <file>` in the worktree with hooks on (the pre-commit hook runs the unit suite: take the host lock first); never `--no-verify`, never amend, never push.

## Already ruled

| id             | status               | ruling                                                                                                                                                                                 |
| -------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 8              | OPERATOR 2026-09-28  | no pods, no spend, no flywheel run                                                                                                                                                     |
| P14-LIVE       | ratified 2026-09-28  | the phase touching `mlflow.py` is live-required; the conductor runs the live gate before publication                                                                                   |
| (7) suite lock | CONDUCTOR 2026-09-29 | `until mkdir /tmp/esl-suite.lock 2>/dev/null; do sleep 15; done` before any suite; `rmdir` after, on failure too (trap); no worker cap under the lock                                  |
| port unchanged | CONDUCTOR 2026-09-29 | `flush` is a method on `MlflowSink`, not on the `MetricSink` port (a port widening is an operator-ruled door change; not needed: the worker drains continuously and `finish()` drains) |
