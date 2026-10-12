# for-1637-h-worker-log-lines: the H worker narrates every mechanism through the log substrate

Authoritative brief for one `td-phase-lifecycle.cascade` run, written 2026-10-12 by the conductor
from the operator's ruling (2026-10-12: kill the wedged probe pods, apply the log fix, then verify
twice: once locally with TDD-first tests under the podman live gate, once by re-running the paired
H0 probes on RunPod; logs must be printed, displayed and captured end to end for every mechanism,
including R2 uploads, the log sink and MLflow).

## Branch

| field                | value                                                                                                                  |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1637-h-worker-log-lines`                                                                                      |
| parent               | `main` (`edabe74`, release v0.2.1); the operator merges the PR                                                         |
| cascade `phase.mode` | `arch`, always explicit (a new observable surface: the worker's log-event vocabulary; reader-tuple members are edited) |
| wave                 | serial, main checkout, no lanes                                                                                        |
| `gates.liveRequired` | `true` (the MLflow adapter and a live integration test are in the touch set)                                           |
| gates                | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}`                        |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree                                 |
| publish              | the cascade's publish stage via `just ship-it` (check + live gate + `gt submit`), ONE draft PR on `main`               |

The branch EXISTS at launch: the conductor created it with `git checkout -b fix/for-1637-h-worker-log-lines edabe74`
and committed this brief as its first commit (adopted in `args.conductorCommits`). Verify
`git merge-base --is-ancestor edabe74 HEAD` and implement on top. Published history is never
amended, rebased or force-pushed.

## Why (measured 2026-10-12, wave `h0-plumbing-1k.c9fb15`)

Two A100 pods ran the paired H0 probe on image `762ed810...` (source `65542d4`). Both trained
1000 global steps (MLflow per-step scalars n=1000), logged evaluation rows at 500/750/1000, and
published checkpoints through step 1000 to R2. The publication blocks measured about 3, 3 and 16
minutes. Both workers then sat in the final leg for 17+ minutes with no R2 object and the MLflow
runs still RUNNING; the FOR-1617 stop marker went unacknowledged (`stop_not_confirmed`) on both.
The RunPod container log of each pod held exactly nine lines, all from startup, the last being
`mlflow: tracking run ...` at 19:32Z. Nothing was emitted at any phase change, checkpoint,
evaluation, upload, retry or at the point where the worker wedged. `grep -rl loguru` over
`core/sentence`, `ladder/run/worker`, `ladder/run/worker_command.py`, `ladder/run/trainer` and
`ladder/run/runner.py` returns nothing: the substrate is configured at entry and never spoken to
again. The operator killed the pods at 20:24Z; the cause of the final-leg stall is unknown
BECAUSE of this gap. Linear: FOR-1637.

## Contract surface owned

| contract                                                                                                                                                                                                                                                                                                                                                                                         | kind                      | frozen_after |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- | ------------ |
| The H worker log-event vocabulary: a closed set of `event` names emitted through `kernel.runtime.logging.logger` with `logger.bind(event=..., **fields)`, INFO for milestones and WARNING for every retry, every bounded wait that exceeds its budget and every abandoned drain. The set is declared ONCE as a frozenset constant in `ladder/run/worker_log_events.py` (new) and pinned by test. | observable surface (door) | this phase   |
| The worker's structured log file `<workspace>.observability/worker-log.jsonl` (the substrate's JSON sink, same directory as `metrics.jsonl`), uploaded at `MlflowSink.finish()` as the run artifact `results/worker-log.jsonl` beside the existing `results/metrics.jsonl` upload.                                                                                                               | capture artifact          | this phase   |

Minimum event set (names are the contract; fields in parentheses are required):

- `worker.start` (request_sha256, image, code_commit) and `worker.end` (status, seconds)
- `phase.start` / `phase.end` (phase, step_from, step_to, seconds)
- `checkpoint.seal` (phase, step, global_step, seconds)
- `evaluation.start` / `evaluation.end` (global_step, seconds)
- `publish.start` / `publish.end` (key, bytes, seconds) for EVERY R2 object put, including journals, `.pt` snapshots, checkpoint markers, `started.json`, the completion marker and the stop-record
- `publish.verify_read` (key, attempt, max_attempts, seconds) at INFO; `publish.retry` (key, attempt, max_attempts, reason) at WARNING
- `stop_marker.observed` (key) at INFO; `stop_marker.polled` is NOT an event (no per-poll noise)
- `completion.written` (key, bytes)
- `sink.finish.start` / `sink.finish.end` (sink, drained, stranded, seconds); `sink.drain_exceeded` (sink, budget_seconds, stranded) at WARNING; `sink.write_failed` (sink, count) at WARNING
- `mlflow.run` (run_id, uri) already exists in prose form; keep it and add the `event` field

The human stderr line keeps the substrate's existing format (`<ts> | LEVEL | message`); the
message is one sentence that names the event and its key fields, so the RunPod container log reads
without the JSON file.

## Contracts consumed

- `kernel.runtime.logging.configure_logging(json_path=...)` and `logger`: the only logging API; `configure_logging` is called once at `worker_command.main` (already true) and the JSON sink path is passed from `run_worker_command` once the observability root is known (one more `configure_logging` call is idempotent by design; measure that no line is double-printed).
- `ladder.capture.adapters.mlflow.MlflowSink.finish()`: already uploads `results_path`; the log file upload follows the same guard (missing file is a WARNING, never a raise).
- `ladder.run.runner.run_probe_training(... on_checkpoint, publication_timing)`: the loop that owns phase, checkpoint and evaluation milestones.
- `ladder.run.worker_snapshot.publish_worker_snapshot`, `ladder.run.transport.d1_trainer_object_transport` (`_attempt`, the put/verify-read path with its `sleep` injection) and `d1_trainer_output_publication`: the R2 upload path.
- `ladder.run.object_worker.run_object_worker` and `worker_completion`: the final leg and the completion marker.
- `tests/test_sentence_execution.py::engine` (2 `codec_warmup` + 4 `prediction` steps): the fast CPU path every unit test drives.
- `tests/test_h_lane_live.py`: the live-gate pattern (skip without `MLFLOW_TRACKING_URI`, real server, run id written under `results/`).
- `tests/test_d1_r2_trainer_mailbox_worker.py`: the pattern for touching the real R2 bucket under a test prefix with cleanup.

## Sacred invariants

| invariant                     | verbatim                                                                                            | tripwire                                                                                                                                      |
| ----------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| House rule 5                  | "TDD is non-negotiable, and guard-teeth prove both arms."                                           | every new assertion shown RED on `edabe74` before the implementation commit; mutations M1 to M6 both arms quoted                              |
| House rule 3                  | "Never fix flakiness by raising a timeout or a retry budget."                                       | no retry count, deadline or sleep value changes in the touch set (`git diff` grep for `sleep`, `timeout`, `attempts`, `deadline`)             |
| Publication timing            | checkpoint publication behaviour (what is published, when, with which retries) is unchanged         | `tests/test_sentence_worker_snapshot.py`, `tests/test_sentence_worker_completion.py`, `tests/test_sentence_object_worker.py` green unmodified |
| Metric path                   | metric values and keys reaching JSONL and MLflow are unchanged                                      | `tests/test_sinks.py`, `tests/test_h_worker_command_observability.py` existing assertions green; `tests/test_h_lane_live.py` green            |
| Print ban                     | no new `print(` outside `PRINT_ALLOWLIST`                                                           | `tests/test_observability_conformance.py` green                                                                                               |
| Reader identity (R-PIN, DG-A) | every reader-tuple-member edit lands in ONE conductor commit together with the regime re-pin        | `tests/test_identity_epoch.py` green at the tip; `docs/identity/read-old-table.md` gains the dated row                                        |
| Layer contract                | `kernel < core < ladder < historical < composition`; `mlflow` imported only by `adapters/mlflow.py` | `lint-imports` green                                                                                                                          |
| Numerics                      | untouched                                                                                           | no file under `core/` changes except none; `tests/test_sentence_*` numerics tests green                                                       |

## Identity members

`ladder.run.{object_worker, worker_execution, worker_snapshot, worker_completion, worker_command, runner, execution, training}` are checkpoint-reader-tuple members (`ladder/run/checkpoint.py::_READER_MODULES`). Editing any of them re-pins the regime rows, which the pre-commit unit suite checks, so a lane can NEVER commit such a change itself. Protocol DG-A: the implementer stages the complete member change set together with the non-member files of the same commit, runs the unit tests that do not depend on the identity rows, and HALTS with `REPIN-HANDOFF` naming the staged files and the measured new digests (`uv run python tools/regime_rows.py` output quoted). The conductor makes ONE commit (change set + `tests/test_identity_epoch.py` re-pin + `docs/identity/read-old-table.md` dated row), adds its SHA to `args.conductorCommits`, and resumes. Never edit `tests/test_identity_epoch.py` or `docs/identity/*` in a lane.

## Scope

Commit ladder (the implementer may split further, never merge steps):

1. **C1, tests RED.** `tests/test_h_worker_log_lines.py` (new): a loguru capture fixture (`logger.add(list.append, format="{message}", level="INFO")` plus a JSON-record capture) and tests that (a) drive `run_probe_training` through the fast engine and assert `phase.start/end`, `checkpoint.seal` per interval and `evaluation.start/end` per interval with their fields; (b) drive `publish_worker_snapshot` and the transport put/verify-read path with an in-memory object adapter and assert `publish.start/end` with key and bytes, `publish.verify_read` per attempt, and `publish.retry` at WARNING when the adapter fails once; (c) drive `run_object_worker` to completion on the fast path and assert `completion.written` and `worker.end`; (d) drive `CompositeSink.finish()` with a `MlflowSink` whose client is a fake and assert `sink.finish.start/end` with `drained`, plus `sink.drain_exceeded` at WARNING when the drain budget is exceeded; (e) assert the event vocabulary constant equals the set of events the tests observed (closed set, pinned both ways). Extend `tests/test_h_worker_command_observability.py` so the existing successful-run test also asserts that the human stderr stream carries the `phase.start` and `completion.written` sentences and that `<workspace>.observability/worker-log.jsonl` exists with JSON records carrying `event`. Add a subprocess-level test in the style of `tests/test_h_probe_process.py` that runs `python -m entropic_substrate.ladder.run.worker_command` on the fast local request and asserts the sentences appear on the child's stderr (proves the process-level stream, not an in-process sink). Add `tests/test_h_worker_live_logs.py` (integration, skip without `MLFLOW_TRACKING_URI`): the fast path through a real MLflow run; after `finish()`, the server lists the artifact `results/worker-log.jsonl` and its records contain `sink.finish.end`; and one snapshot published through the real R2 adapter under a test prefix (pattern of `tests/test_d1_r2_trainer_mailbox_worker.py`, objects deleted in teardown) with `publish.start/end` and `publish.verify_read` observed. All of these RED on `edabe74`, counts quoted.
2. **C2, implementation (DG-A).** `ladder/run/worker_log_events.py` (new, the vocabulary); log calls at the milestones in `runner.py`, `object_worker.py`, `worker_execution.py`, `worker_snapshot.py`, `worker_completion.py`, `worker_command.py` (JSON sink path + `worker.start/end`), `transport/d1_trainer_object_transport.py` and `transport/d1_trainer_output_publication.py` (publish and verify-read events), `capture/adapters/mlflow.py` and `composite.py` (sink finish events and the log-file upload). Reader-tuple members land through the REPIN-HANDOFF. No behaviour change other than log lines and the one extra artifact upload.
3. **C3, docs.** `docs/architecture.md` observability paragraph gains two sentences naming the event vocabulary and the log file; `docs/deploy-h100.md` gains one sentence under the current pin saying the next image rebuild carries FOR-1637 (the re-pin itself is a later conductor ceremony, not this phase).

## ACCEPTANCE

- Every test named in C1 is RED on `edabe74` (quote the failing assertion and the count) and GREEN at the tip; the five existing files named under *Sacred invariants* stay green unmodified.
- `grep -rn "logger\." src/entropic_substrate/ladder/run/{runner,object_worker,worker_snapshot,worker_completion,worker_execution,worker_command}.py src/entropic_substrate/ladder/run/transport/d1_trainer_object_transport.py` is non-empty at the tip for every file listed (quote the per-file counts); at `edabe74` the same grep is empty.
- The vocabulary constant and the observed-event set are equal in `tests/test_h_worker_log_lines.py` (closed set pinned both ways).
- Zero new `print(` sites (`tests/test_observability_conformance.py` green); zero changes to any retry count, sleep, deadline or timeout value (quote the grep over the diff).
- Guard teeth by mutation, both arms quoted (below).
- `just check` green at the tip under the host lock; `just container-test` green inside `just ship-it` (the live test's run id written under `results/` and quoted in the PR body); exit gate certified at the final tip by the runner's JSON; `tests/test_identity_epoch.py` green at the tip after the conductor's DG-A commit.
- Draft PR on `main` with a filled body: Linear ref FOR-1637 and the operator ruling; the event vocabulary table; RED/GREEN counts; mutation arms with counts; the live run id and the R2 test prefix used and cleaned; the DG-A commit SHA and the new regime digests; the three honest exits.

## Non-goals

- Does NOT change what is published, when, or with how many retries (FOR-1567 and the measured publication growth are a separate finding; do not touch timing).
- Does NOT add the log file to the R2 completion envelope (`worker_completion.admit_publication` keys are a door; unchanged).
- Does NOT touch the host controller or watchdog (`ladder/run/worker/h_probe_*`); the controller's blindness to an externally deleted pod is FOR-1638.
- Does NOT change `core/sentence/*`, the numerics, the profile schema or the solver.
- Does NOT edit `Containerfile` (`PYTHONUNBUFFERED=1` is already set at line 78).
- Does NOT launch pods, spend, build images or re-pin the deploy doc (conductor ceremony after the merge).

## Allowed files

- `src/entropic_substrate/ladder/run/worker_log_events.py` (new)
- `src/entropic_substrate/ladder/run/{runner,object_worker,worker_execution,worker_snapshot,worker_completion,worker_command}.py` (reader-tuple members: DG-A hand-off)
- `src/entropic_substrate/ladder/run/transport/{d1_trainer_object_transport,d1_trainer_output_publication}.py`
- `src/entropic_substrate/ladder/capture/adapters/{mlflow,composite,jsonl}.py`
- `src/entropic_substrate/kernel/runtime/logging.py` (only if a bind helper is needed; say so)
- `tests/test_h_worker_log_lines.py` (new), `tests/test_h_worker_live_logs.py` (new), `tests/test_h_worker_command_observability.py`, `tests/test_h_probe_process.py` (one added test) or a new `tests/test_h_worker_process_logs.py`
- `docs/architecture.md`, `docs/deploy-h100.md` (C3 sentences only)
- `.secrets.baseline` (mechanical companion, regenerated by the prek hook only)

Never edit: `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/regime_rows.py` (conductor-only, DG-A), `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `scripts/run_container_test.sh`, `tools/mapping.json`, `Containerfile`, `compose.yaml`, the manifest, the plans, the briefs, `pyproject.toml`, `uv.lock`, `tests/observability_conformance_config.py`, anything under `core/`.

## Already ruled

- Operator 2026-10-12: kill the wedged pods (done 20:24Z), apply the log fix, verify twice (podman live gate with TDD-first tests, then the RunPod probe re-run); logs printed, displayed and captured end to end for every mechanism.
- FOR-1637 filed 2026-10-12 with the measurement; FOR-1638 filed for the controller.
- `PYTHONUNBUFFERED=1` is already in the image; the stream is not the problem, the absence of lines is.
- DG-A is the protocol for reader-tuple members; the conductor commits the re-pin.
- Message lint before every commit (`uv run gitlint --msg-filename`); write a fresh message file per commit.

## Guard teeth by mutation

| id | mutation                                                                | expected RED                                                                               |
| -- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| M1 | remove the `checkpoint.seal` log call in `runner.py`                    | `test_h_worker_log_lines.py` checkpoint assertion RED; vocabulary-equality pin RED         |
| M2 | remove the `publish.retry` WARNING in the transport                     | the fails-once adapter test RED                                                            |
| M3 | drop the `configure_logging(json_path=...)` call so no JSON sink exists | the `worker-log.jsonl` existence assertion in `test_h_worker_command_observability.py` RED |
| M4 | drop the log-file upload from `MlflowSink.finish()`                     | `test_h_worker_live_logs.py` artifact listing RED (live gate)                              |
| M5 | remove the `completion.written` log call                                | `run_object_worker` completion assertion RED; subprocess stderr test RED                   |
| M6 | add an event name to the vocabulary constant that no code emits         | vocabulary-equality pin RED (closed set holds both ways)                                   |
