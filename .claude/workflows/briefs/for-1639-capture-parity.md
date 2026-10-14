# for-1639-capture-parity: H runs are capturable by the E-series capture lane and carry run lifecycle records

Authoritative brief for one `td-phase-lifecycle.cascade` run (slice 3 of the H observability parity
stack, umbrella FOR-1640; issues FOR-1639 and FOR-1641). Written 2026-10-13 by the conductor while
slice 2 (FOR-1649) runs. Every anchor below was measured at `a43ef9f` (the published head of slice 1,
draft PR #341) with `git show`; the evidence line sits beside each claim. Slice 2 touches only
`tests/conftest.py`, test files and at most an additive sink in `kernel/runtime/logging.py`, so these
anchors hold at its PR head; the implementer re-verifies the line numbers in recon.

## Branch

| field                | value                                                                                                             |
| -------------------- | ----------------------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1639-capture-parity`                                                                                     |
| parent               | `fix/for-1649-conftest-sink-binding` at its PUBLISHED PR head (slice 2); the operator merges the stack in order   |
| cascade `phase.mode` | `arch` (a new telemetry record schema and a new admissibility predicate are contract surfaces)                    |
| wave                 | serial, main checkout, no lanes                                                                                   |
| `gates.liveRequired` | `true` (the capture lane and the MLflow adapter path change; the live test proves the lane against a real server) |
| gates                | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}`                   |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree                            |
| publish              | the cascade's publish stage via `just ship-it` once, ONE draft PR stacked on slice 2                              |

The branch EXISTS at launch: the conductor creates it with `git checkout -b fix/for-1639-capture-parity <slice-2 PR head>`
and commits this brief (plus the FOR-1637 brief's `drained` to `settled` wording amendment) as its first commit,
adopted in `args.conductorCommits`. Verify `git merge-base --is-ancestor <slice-2 head> HEAD` and implement on top.

## Why (measured at `a43ef9f`)

- `ladder/capture/reader/capture_results.py::capture_finished_runs` (`:88-126`) iterates `client.search_runs`, skips non-FINISHED runs (`:108`) and `-tiny` names (`:41`, `:110`), downloads every `results/*.jsonl` artifact (`:114`) and calls `final_eval_record(target)` on each (`:123`). `final_eval_record` (`telemetry.py:183-192`) returns the last `event == "eval"` row (`:188`) and raises `TelemetryGap` (`telemetry.py:104`) when there is none (`:192`).
- The H worker builds its sink at `ladder/run/worker_command.py:191-203`: `TrackingRunDescriptor(run_name=f"h-worker-{identity}", tiny=True, params={"run_class"}, tags={"schema": "h-probe-worker-result-v1"})`, `results_path=<workspace>.observability/metrics.jsonl`, `log_path=worker-log.jsonl` (FOR-1637), `metric_extractor=HTrainingStepExtractor`. The run name has no `-tiny` suffix, so the tiny filter never excludes it; `build_sink_for_run` reads the experiment from `MLFLOW_EXPERIMENT_NAME` with the default `entropic-substrate-lm` (`composite.py:160`, `sinks.py:103`), the same experiment `cmd_capture_results` reads (`composition/entrypoints/cli.py:575-576`).
- H `metrics.jsonl` rows are `h-training-step-v2` step rows (`runner.py:284-285`) and `h-predictive-evaluation-v1` reports (`runner.py:357-362`); none carries `event == "eval"`, and `tests/test_h_metric_routing.py:46` pins step rows reaching the sink unmodified. So one FINISHED H run in the shared experiment makes `capture-results` raise `TelemetryGap` after downloading its file; `cmd_capture_results` catches only `LookupError` (`cli.py:588`) and `main` only `CheckpointFormatError` (`cli.py:1138`), so it escapes as a traceback. Since FOR-1637 the same run also carries `results/worker-log.jsonl`, a second `.jsonl` the E reader would refuse.
- `MlflowSink.finish()` always calls `set_terminated(status="FINISHED")` (`adapters/mlflow.py:866`), and `worker_command.py:205-215` finishes the sink in a `finally`, so a worker that raised still leaves a FINISHED run. MLflow status therefore cannot distinguish a complete H run from a failed one; the record stream must.
- E runs carry `run_start` (config, parameter count, device, tokenizer; `historical/e_series/train.py:746`) and `run_end` with `wall_s` (`train.py:825`) in the authoritative JSONL. The H stream has no lifecycle records; the start lives only in R2 as `worker/start.json` (`object_worker.py:268-293`, schema `h-worker-start-v1`, payload `runtime` from `worker_execution._runtime` `:48` which includes `measure_hardware()` `:77`), and the end only in the admitted result row (`object_worker._admit_result` `:158-161`: `status`, `runtime`, `successful_optimizer_steps`, `wall_seconds`), written at `:383` before `complete.json`.
- `HTrainingStepExtractor._project` (`h_metric_extractor.py:87-97`) returns `None` for any schema other than the two above, and `MlflowSink.log` returns on `None` (`mlflow.py:546-548`), while `JsonlSink.log` writes every record (`jsonl.py:37-39`) and `CompositeSink.log` fans out authoritative-then-best-effort (`composite.py:96-99`). New records therefore reach `metrics.jsonl` without touching the MLflow metric surface.
- The capture fake, `tests/test_capture_results.py::FakeCaptureClient` (`:83-117`), models runs as `_Run(info)` with `run_id`, `run_name`, `status` only (`:61-69`); the real `mlflow.entities.Run` exposes `run.data.tags`. The fixture `_client()` (`:119-136`) holds `e1-fisher` FINISHED, `e1-fisher-tiny` FINISHED and an `e1-fisher` RUNNING run. 24 tests there, 44 in `tests/test_telemetry.py`, 7 in `tests/test_h_metric_routing.py`; no test drives `capture_finished_runs` over an H-shaped run (matrix A8).
- Live precedent: `tests/test_mlflow_live.py:161-162` monkeypatches `MLFLOW_EXPERIMENT_NAME` to a per-test experiment on the gate's server; `tests/test_h_lane_live.py:64-73` runs the fast engine into a real MLflow run named `h-lane-live-<identity>`.

## Contract surface owned

| contract                                                                                                                                                                                                                                                                                                                                                                                    | kind                                                                                  | frozen_after |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | ------------ |
| `h-run-lifecycle-v1` records in the H metric stream: exactly one `run_start` and one `run_end` per run, written through the run's `MetricSink` by `ladder/run/object_worker.py` (fields below), never projected to MLflow metrics by the extractor.                                                                                                                                         | telemetry record schema (door after this phase; additive fields only)                 | this phase   |
| H admissibility predicate `final_h_run_end(path)` in a new `ladder/capture/reader/h_telemetry.py`: a `metrics.jsonl` is admissible iff it holds one `run_start`, one `run_end` and at least one committed `h-training-step-v2` row; anything else raises `TelemetryGap` naming the file. Other `.jsonl` artifacts of an H run (`worker-log.jsonl`) must be well-formed JSON lines or raise. | admissibility gate (same posture as `final_eval_record`: raises, never returns empty) | this phase   |
| `capture_finished_runs` partitions runs by the `schema` tag: no tag or a non-`h-` tag = E lane, unchanged; a tag starting with `h-` = H lane through `final_h_run_end`. Captured H files use the same run-scoped naming (`<run_name>-<run_id[:8]>-<file>`).                                                                                                                                 | capture lane behaviour                                                                | this phase   |
| `cmd_capture_results` catches `TelemetryGap`, logs it at ERROR with the file named, and returns 2 (the idiom already used at `cli.py:374` and `:733`; LookupError at `:588`).                                                                                                                                                                                                               | CLI exit contract                                                                     | this phase   |

Record fields (conductor ruling, below the seam; the design gate may add fields, never remove):
`run_start` = `{"schema": "h-run-lifecycle-v1", "event": "run_start", "request_sha256", "image", "code_commit", "profile_identity", "run_class", "runtime": <the same dict written to worker/start.json>}`, emitted inside `object_worker`'s `start(runtime)` callback beside `worker/start.json`.
`run_end` = `{"schema": "h-run-lifecycle-v1", "event": "run_end", "status": "ok" | "failed", "wall_seconds", "successful_optimizer_steps" (int, or null on failure)}`, emitted on the ok path from the admitted result row (`object_worker.py:383`) and on the failure path before the exception propagates, with the same perf-counter origin as the run. The sink is the one threaded at `object_worker.py:378`; both records go through `sink.log`, never through a new port method.

## Contracts consumed

- `MetricSink` port (`ladder/capture/ports/metric_sink.py`), `CompositeSink` / `JsonlSink` / `MlflowSink` adapters as above; `build_sink_for_run` and `TrackingRunDescriptor` are a frozen door (`run_descriptor.py:9`), untouched here.
- `TelemetryGap` and `final_eval_record` (`telemetry.py:104`, `:183`): consumed, never weakened.
- `HTrainingStepExtractor` (`h_metric_extractor.py:52`): its `None` for unknown schemas is relied on and pinned; no edit expected.
- The FOR-1637 vocabulary (`ladder/run/worker_log_events.py`): a human log line per captured run is emitted through `logger` (existing `logger.info(f"captured {target}")` at `capture_results.py:125`); no new event name is required in the worker (the closed set pinned by `tests/test_h_worker_log_lines.py` stays equal).
- `tools/regime_rows.py` (conductor-only) for the DG-A re-pin.

## Sacred invariants

| invariant                       | verbatim                                                                                    | tripwire                                                                                                                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| House rule 5                    | "TDD is non-negotiable, and guard-teeth prove both arms."                                   | every new test RED at the parent head before the change, GREEN after; mutations M1-M5 both arms quoted                                                                   |
| CLAUDE.md data flow             | "inadmissible telemetry raises, it never returns an empty value"                            | `final_h_run_end` raises on a missing `run_end`; the capture test for a file without `run_end` expects `TelemetryGap`; no `include_h`/skip flag that silently drops runs |
| E capture unchanged             | `final_eval_record` and the E branch of `capture_finished_runs` byte-identical in behaviour | all 24 existing `tests/test_capture_results.py` tests green unmodified except the fake gaining `data.tags`; `tests/test_telemetry.py` 44 green unmodified                |
| Step rows unmodified            | `tests/test_h_metric_routing.py:46`                                                         | green unmodified; lifecycle records are additional rows, never edits of step rows                                                                                        |
| MLflow metric surface unchanged | the extractor projects only the two existing schemas                                        | new pin: lifecycle records yield `None` from `HTrainingStepExtractor.extract`; `tests/test_tracking_sinks.py` green                                                      |
| Reader identity (DG-A)          | every reader-tuple-member edit lands in ONE conductor commit with the regime re-pin         | `tests/test_identity_epoch.py` green at the tip; `docs/identity/read-old-table.md` gains the dated row                                                                   |
| House rule 3                    | "Never fix flakiness by raising a timeout or a retry budget."                               | no retry, sleep, deadline or timeout literal changes                                                                                                                     |
| Print ban and layer contract    | unchanged                                                                                   | `tests/test_observability_conformance.py`, `lint-imports` green                                                                                                          |

## Identity members

`ladder.run.object_worker` is a checkpoint-reader-tuple member (`ladder/run/checkpoint.py::_READER_MODULES`, `:102`); so are `worker_command` (`:111`), `runner` (`:103`) and `worker_execution` (`:113`) if the design gate moves an emission site there. `composition.cli` is a member (`:67`) but `composition.entrypoints.cli` is NOT (matrix, verified at `edabe74`); `ladder.capture.reader.capture_results`, `ladder.capture.reader.telemetry`, `ladder.run.h_metric_extractor` and the new `h_telemetry.py` are not members. Protocol DG-A: the implementer stages the complete member change set together with the non-member files of the same commit, runs the unit tests that do not depend on the identity rows, and HALTS with `REPIN-HANDOFF` naming the staged files and the measured new digests (`uv run python tools/regime_rows.py` output quoted for the three regimes). The conductor makes ONE commit (change set + `tests/test_identity_epoch.py` re-pin, table at `:567`, + `docs/identity/read-old-table.md` dated row), adds its SHA to `args.conductorCommits`, and resumes. Never edit `tests/test_identity_epoch.py` or `docs/identity/*` in a lane. Expect exactly ONE hand-off in this slice.

## Scope

Commit ladder (the implementer may split further, never merge steps):

1. **C1, tests RED.** (a) `tests/test_capture_results.py`: the fake grows `data.tags` (a `_Data` dataclass with `tags: dict[str, str]`, default empty, so the 24 existing tests stay green); new tests: a FINISHED run tagged `schema: h-probe-worker-result-v1` whose `results/metrics.jsonl` holds `run_start`, committed step rows and `run_end` plus a `results/worker-log.jsonl` is captured under run-scoped names and `final_eval_record` is never applied to it (observe through a monkeypatched `final_eval_record` that raises if called with an H path); the same run without `run_end` raises `TelemetryGap` naming the file; an H run whose `worker-log.jsonl` has a non-JSON line raises; an E run beside an H run is still captured. (b) `tests/test_h_telemetry.py` (new): `final_h_run_end` over admissible, missing-start, missing-end, duplicate-end, zero-committed-step and malformed-line fixtures. (c) `tests/test_h_worker_command_observability.py`: the successful-run test (`:30`) also asserts exactly one `run_start` and one `run_end` in the JSONL sink with the fields above in order (start first, end last), and the admission-failure test (`:100`) asserts neither record is written; a new failure-path test over the in-memory store (a training step that raises) asserts `run_end` with `status: failed` and `successful_optimizer_steps: null`. (d) `tests/test_h_metric_routing.py`: new pin that `HTrainingStepExtractor.extract` returns `None` for both lifecycle records and that step rows are still unmodified. (e) The CLI file holding the existing `capture-results` `LookupError` test (`tests/test_cli_surface.py` or `tests/test_cli_charts.py` at `a43ef9f`; measure in recon): `TelemetryGap` from the capture lane exits 2 with the file named on stderr, no traceback. (f) `tests/test_h_lane_live.py` (integration, skips without `MLFLOW_TRACKING_URI`): monkeypatch `MLFLOW_EXPERIMENT_NAME` to `h-capture-live-<uuid>` (precedent `test_mlflow_live.py:161-162`), drive `run_object_worker` over the in-memory object store with a real `build_sink_for_run` sink (precedent `tests/test_h_worker_live_logs.py`), then `capture_finished_runs(real_client, [that experiment], tmp_path)` returns the two files and never raises; the captured `metrics.jsonl` passes `final_h_run_end`. All RED at the parent head, counts quoted.
2. **C2, implementation (DG-A for `object_worker.py`).** `ladder/run/object_worker.py` emits `run_start` in `start(runtime)` and `run_end` on both exits through the threaded sink; `ladder/capture/reader/h_telemetry.py` (new) with `final_h_run_end` and the JSON-lines check; `capture_results.py` partitions by `run.data.tags.get("schema")` (missing `data` or `tags` on a run object is treated as the E lane; a non-string tag raises); `composition/entrypoints/cli.py` `cmd_capture_results` catches `TelemetryGap`. No change to `final_eval_record`, to step rows, to the extractor, to the descriptor door, to retry or timing values.
3. **C3, docs.** `docs/architecture.md` capture paragraph gains two sentences (the H lane, the lifecycle records); the capture-lane docstring at the top of `capture_results.py` names both lanes. No identity-bearing file.

## ACCEPTANCE

- Every test named in C1 is RED at the parent head (quote the failing assertion and the count) and GREEN at the tip; the existing 24 + 44 + 7 tests named above stay green with no assertion changed.
- `grep -n "h-run-lifecycle-v1" src/entropic_substrate/ladder/run/object_worker.py` is non-empty at the tip and empty at the parent; `grep -rn "h-run-lifecycle-v1" src/entropic_substrate/ladder/run/h_metric_extractor.py` is empty at the tip (the extractor never learns the schema).
- `grep -n "data.tags\|\.tags" src/entropic_substrate/ladder/capture/reader/capture_results.py` is non-empty at the tip and empty at the parent.
- Guard teeth by mutation, both arms quoted (below).
- No retry, sleep, deadline or timeout literal changed (grep over the diff quoted); zero new `print(` sites.
- `just check` green at the tip under the host lock; `just container-test` green inside `just ship-it` with the live capture test's experiment name and run id quoted in the PR body; exit gate certified at the final tip by the runner's JSON; `tests/test_identity_epoch.py` green after the conductor's DG-A commit.
- Draft PR stacked on the slice-2 PR head with a filled body: Linear refs FOR-1639 and FOR-1641; the record schema table; RED/GREEN counts; mutation arms; the live run id and experiment; the DG-A commit SHA and the new regime digests; the three honest exits.

## Non-goals

- Does NOT give H runs their own MLflow experiment or extra pod env keys (slice 5, FOR-1647, ruling DN-OBS-TEMPLATE); the shared experiment becomes safe here, not separate.
- Does NOT add descriptor params or tags (`config_hash`, `git_sha`, image digest) to the H run (slice 4, FOR-1642); `mlflow_aggregate._leaf_runs` (`mlflow_aggregate.py:66-74`) still excludes H runs until then.
- Does NOT change `MlflowSink.finish()`'s unconditional `FINISHED` status (`mlflow.py:866`); measured here and recorded for the conductor to file, not fixed in this slice.
- Does NOT capture H checkpoints through `capture_checkpoints` (`capture_results.py:129`); H checkpoints are sealed in R2 under worker custody.
- Does NOT touch the host controller, the launch template, `core/sentence/*`, numerics, schedules or publication timing.
- Does NOT launch pods, spend, build images or re-pin the deploy doc.

## Allowed files

- `src/entropic_substrate/ladder/run/object_worker.py` (reader-tuple member: DG-A hand-off)
- `src/entropic_substrate/ladder/capture/reader/capture_results.py`, `src/entropic_substrate/ladder/capture/reader/h_telemetry.py` (new)
- `src/entropic_substrate/composition/entrypoints/cli.py` (the `except TelemetryGap` arm in `cmd_capture_results` only)
- `tests/test_capture_results.py`, `tests/test_h_telemetry.py` (new), `tests/test_h_worker_command_observability.py`, `tests/test_h_metric_routing.py`, `tests/test_h_lane_live.py`, the CLI test file measured in C1(e)
- `docs/architecture.md` (C3 sentences only)
- `.secrets.baseline` (mechanical companion, regenerated by the prek hook only)

Never edit: `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/regime_rows.py` (conductor-only, DG-A), `src/entropic_substrate/ladder/run/{worker_command,runner,worker_execution,worker_transport,worker_snapshot}.py` unless the design gate moves an emission site there (then DG-A, same single hand-off), `src/entropic_substrate/ladder/run/h_metric_extractor.py`, `src/entropic_substrate/ladder/capture/reader/telemetry.py`, `src/entropic_substrate/ladder/capture/adapters/*`, `src/entropic_substrate/ladder/capture/ports/*`, `tests/conftest.py`, `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `scripts/run_container_test.sh`, `tools/mapping.json`, `pyproject.toml`, `uv.lock`, `tests/observability_conformance_config.py`, anything under `core/`, the manifest, the plans, the briefs.

## Already ruled

- Operator 2026-10-12: observability as feature parity across E::H; slices stacked linearly on the previous PR head; verify twice.
- Operator 2026-10-13: DN-1637-DOOR, DN-OBS-CONF and DN-OBS-TEMPLATE ratified; the conftest fix is slice 2 (FOR-1649) directly below this slice.
- Conductor 2026-10-13 (below the seam): H runs are CAPTURED by the shared lane through an H predicate, not skipped (the matrix's recommendation; the ticket's option (a) "skip with named reason" is rejected because a skipped run reads as "captured everything"); lifecycle records are emitted by `object_worker`, the one module that holds both the start payload and the admitted result.
- DG-A is the protocol for reader-tuple members; the conductor commits the re-pin.
- Message lint before every commit (`uv run gitlint --msg-filename`); a fresh message file per commit.
- One full test suite at a time on this host under `/tmp/esl-suite.lock`, no worker cap.
- FOR-1637 shipped the `settled` field; FOR-1649 fixes the suite's observation contract; neither is revisited here.

## Guard teeth by mutation

| id | mutation                                                                                             | expected RED                                                                                          |
| -- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| M1 | remove the `schema`-tag partition in `capture_finished_runs` (every run through `final_eval_record`) | the H capture test RED with `TelemetryGap` from `final_eval_record`; the E-beside-H test RED          |
| M2 | drop the `run_end` emission in `object_worker`                                                       | the JSONL lifecycle test RED; `final_h_run_end` over the worker's file RED; the live capture test RED |
| M3 | make `final_h_run_end` return on a missing `run_end` instead of raising                              | the missing-end test RED                                                                              |
| M4 | remove the `except TelemetryGap` arm in `cmd_capture_results`                                        | the CLI exit-2 test RED (traceback)                                                                   |
| M5 | teach `HTrainingStepExtractor._project` the lifecycle schema                                         | the extractor `None` pin RED                                                                          |
