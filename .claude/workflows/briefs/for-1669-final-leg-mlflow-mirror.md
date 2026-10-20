# for-1669-final-leg-mlflow-mirror: final-leg stage and admission measurements mirrored to MLflow

DRAFT 2026-10-19 by the conductor (umbrella FOR-1640, milestone "H observability parity"; issue
FOR-1669, filed from the third paired 1K plumbing probe, wave `h0-plumbing-1k-v3.HEpP7g`). Serial
cascade lifecycle in a SEPARATE worktree (`../entropic-substrate-lm-for-1669`) while that probe
finishes in the main checkout; the base of a NEW stack on `main` `04ad7790` (v0.3.2). Anchors
measured by the conductor at `04ad7790` with `sed -n`/`grep`; the implementer re-verifies them in recon.
The Linear ticket body is the authoritative statement; this brief restates it with the anchors.

## Branch

| field                | value                                                                                 |
| -------------------- | ------------------------------------------------------------------------------------- |
| branch               | `fix/for-1669-final-leg-mlflow-mirror`                                                |
| parent               | `main` at `04ad7790` (v0.3.2; #350-#355 merged)                                       |
| cascade `phase.mode` | `normal` (free modules only; no door, no reader-tuple member, no identity re-pin)     |
| `gates.liveRequired` | `true` (`just ship-it` inside the lifecycle: check, live container gate, `gt submit`) |
| publish              | ONE draft PR on `main`                                                                |

## Why (measured 2026-10-19 on wave HEpP7g, MLflow queried live at 14:34Z, both runs RUNNING)

In MLflow: training metrics through step 1000 (`HTrainingStepExtractor._training`, schema
`h-training-step-v2`), eval rows at 500/750/1000 (`_evaluation`, `h-predictive-evaluation-v1`),
checkpoint evidence tags (`object_worker.py` `sink.annotate`, FOR-1642), 13 `system/*` metrics.
NOT in MLflow (no metric, no tag): the final-leg stages `training_readback`, `generation`,
`span_diagnostics`, `trace_capture`, `native_verification` and their admission walks. Those exist
only as `stage.start/end` (`src/entropic_substrate/ladder/run/worker_execution.py:31`
`narrated_stage`) and `admission.start/end` (`src/entropic_substrate/ladder/run/seals.py:139`
`narrated_admission`) log lines, which reach container stdout and `worker-log.jsonl`; that artifact
is uploaded once at finish (`tests/test_h_worker_live_logs.py::test_live_worker_log_is_uploaded_as_a_run_artifact`),
so the artifact list was `[]` at 14:34Z, 12 minutes into the leg.

| stage                | 2026-10-17 wave (memo=None walks) | 2026-10-19 wave (FOR-1662) | walk detail (admission.end)                                 |
| -------------------- | --------------------------------- | -------------------------- | ----------------------------------------------------------- |
| completion admission | 880 s                             | 277 s                      | memo=fresh, seals_walked=4, evaluations=3, journals_read=4  |
| training_readback    | 881 s                             | 144 s                      | memo=shared, seals_walked=0, evaluations=3, journals_read=0 |
| generation           | 1071 s                            | 155 s (stage 200 s)        | memo=shared, seals_walked=0, evaluations=3, journals_read=0 |
| span_diagnostics     | 2380 s                            | running at filing          |                                                             |

Both arms within 1 s of each other. The two previous final-leg wedges (2026-10-12, 2026-10-17)
were invisible from MLflow and diagnosable only through pod stdout; a host without pod-log access
sees RUNNING plus system metrics and cannot tell a stage from a hang.

## Anchors (measured at `04ad7790`)

- `src/entropic_substrate/ladder/run/worker_execution.py:31` `narrated_stage(stage)`: binds `event=STAGE_START, stage=` before, `event=STAGE_END, stage=, seconds=` after (READER-TUPLE MEMBER, never edit).
- `src/entropic_substrate/ladder/run/seals.py:139` `narrated_admission(caller, root, memo)`: binds `event=ADMISSION_START, caller, root, memo` ("fresh"/"shared"), then `event=ADMISSION_END, caller, root, memo, seals_walked, evaluations, journals_read, seconds`; with `memo is None` it emits nothing (READER-TUPLE MEMBER, never edit).
- `src/entropic_substrate/ladder/run/worker_log_events.py:75-99`: `STAGE_START`, `STAGE_END`, `ADMISSION_START`, `ADMISSION_END`, `ADMISSION_CALLERS` (closed set: completion, training_readback, generation, span_diagnostics, trace_capture, native_verification, result_admission), `ADMISSION_ROOTS` (FREE module).
- `src/entropic_substrate/ladder/capture/adapters/composite.py:138` `build_sink_for_run(descriptor, *, results_path, client, metric_extractor, log_path)` returns `JsonlSink` alone when tracking is off and no client is given, else `CompositeSink(jsonl, MlflowSink(...))`; `CompositeSink.finish(status=)` (`:122`) finishes the authoritative leg then the best-effort legs; no `logger` reference in the module today (FREE module).
- `src/entropic_substrate/ladder/run/h_metric_extractor.py:52` `HTrainingStepExtractor`; `_project` (`:87`) branches on `schema` in {`h-training-step-v2`, `h-predictive-evaluation-v1`} else `None`; `_training` (`:99`) computes the global step `step + (0 | warmup_steps)` (FREE module).
- `src/entropic_substrate/ladder/capture/ports/metric_sink.py:40` `MetricSink` Protocol: `log(record)`, `log_artifact`, `annotate(tags)`, `finish(*, status)`.
- `src/entropic_substrate/ladder/capture/ports/run_descriptor.py:89` `ExtractedMetrics(step: int, metrics: Mapping[str, float])` and NOTHING ELSE: the extractor path carries no tags, so memo provenance must ride `sink.annotate` (see the seam).
- `src/entropic_substrate/ladder/capture/adapters/mlflow.py:542` `MlflowSink.log`: extractor runs on the caller's thread, an extractor exception is counted in `sink.write_failed`; `log()` after `finish()` is counted as failed and warned, never delivered.
- `src/entropic_substrate/kernel/runtime/logging.py:229-244`: the three `logger.add` handlers (stderr, optional JSON file); loguru's `logger.add(sink, filter=...)` returns a handler id removable with `logger.remove(id)`.
- Worker sink built at `src/entropic_substrate/ladder/run/worker_command.py:210` `build_sink_for_run(...)` (READER-TUPLE MEMBER, never edit; the bridge installs itself inside `build_sink_for_run`, so this call site needs no change).
- Reader tuple `src/entropic_substrate/ladder/run/checkpoint.py:66-116` lists `ladder.run.object_worker`, `ladder.run.runner`, `ladder.run.seals`, `ladder.run.worker_command`, `ladder.run.worker_execution`; it does NOT list `ladder.run.worker_log_events`, `ladder.run.h_metric_extractor`, `ladder.capture.adapters.composite`, `kernel.runtime.logging` (grep at `04ad7790`: zero matches for the four).
- Tests: `tests/test_h_metric_routing.py` (extractor tests `:101-289`, sink routing `:52,:259`), `tests/test_tracking_sinks.py` (`FakeMlflowClient` from `test_sinks`, builder tests `:77-140`), `tests/test_h_worker_log_lines.py` (narration idiom), `tests/test_h_worker_live_logs.py` (live-log artifact idiom), `tests/observability_conformance_config.py` (grep `eval/|h-training-step` = 0 hits at `04ad7790`: it does NOT enumerate metric keys; confirm in recon, then no edit).

## Contract surface owned

Additive only: a record schema constant `h-worker-stage-v1` and its pure projection in
`worker_log_events.py`; one loguru bridge handler installed by `build_sink_for_run` when a
`metric_extractor` is given and removed at `finish()`; a `h-worker-stage-v1` branch in
`HTrainingStepExtractor._project`. Metric keys: `stage/<stage>/seconds`,
`admission/<caller>/seconds`, `admission/<caller>/seals_walked`, `admission/<caller>/evaluations`,
`admission/<caller>/journals_read`; tag `admission/<caller>/memo` = `fresh` | `shared`. The
authoritative D1 journal, `worker-log.jsonl`, the sealed JSONL schemas and the log vocabulary of
FOR-1637/FOR-1662 are untouched. No door, no profile/launch field, no reader-tuple member.

## Ruling (ticket seam, ruled; alternative recorded)

Bridge the log substrate to the sink at the composition root, in FREE modules only:

1. `worker_log_events.py`: `WORKER_STAGE_SCHEMA = "h-worker-stage-v1"` and a pure function that turns a bound `stage.end` / `admission.end` loguru record (`record["extra"]`: `event`, `stage` | `caller`, `root`, `memo`, `seconds`, counts) into a sink record dict carrying that schema; closed vocabulary (`ADMISSION_CALLERS`, a closed stage set); any other event, an unknown name, a missing or non-finite `seconds` returns `None`.
2. `composite.py` `build_sink_for_run`: when `metric_extractor` is given, install ONE `logger.add(<bridge>, filter=<event in {STAGE_END, ADMISSION_END}>)` whose sink calls `sink.log(record_dict)` and, for `admission.end`, `sink.annotate({f"admission/{caller}/memo": memo})`; the handler id is kept on the composite and removed in `finish()` BEFORE the legs finish (so a `stage.end` logged after finish lands nothing and is not counted as a failed write). One writer (the logger), one bridge, same backpressure and `sink.write_failed` counting the MLflow sink already has. A bridge exception never propagates into the logging call (count it through the sink's own failure path; the worker must not die because the mirror did).
3. `h_metric_extractor.py` `_project`: branch for `h-worker-stage-v1` projecting the keys above onto the LAST global step this extractor projected from a training row (stateful: it already sees every committed row through `log()`); `None` if no training row was ever projected; never a guessed step; unknown stage/caller refused.

Alternative NOT taken, recorded: emit `sink.log` inside `narrated_stage` / `narrated_admission`.
Both are reader-tuple members, so it would cost a DG-A conductor re-pin of
`tests/test_identity_epoch.py` and `docs/identity/read-old-table.md` for a pure mirror. A design
question about the ruled seam is a DECISION, not an edit.

## Scope

1. **C1, extractor tests RED at the parent `04ad7790`** (`tests/test_h_metric_routing.py`): a `h-worker-stage-v1` record after a committed training row projects `stage/generation/seconds` (and the four admission keys for an admission record) on the training row's GLOBAL step; before any training row -> `None`; unknown stage/caller name -> `None`. RED at the parent because the schema is unknown (`_project` returns `None`).
2. **C2, bridge tests RED at the parent** (`tests/test_tracking_sinks.py` or a new `tests/test_h_stage_metrics_bridge.py`): with a composite sink from `build_sink_for_run(metric_extractor=HTrainingStepExtractor(warmup_steps=...), client=FakeMlflowClient())`, after one committed training row is logged, running `narrated_stage("generation")` and `narrated_admission("generation", "run", memo)` (a real `SealMemo`, populated and empty arms) lands the stage/admission metric records in the JSONL results sink and the fake client on the training row's global step, and the memo tag via `annotate`; after `finish()` a later `stage.end` lands nothing and `write_failed` stays 0; with `metric_extractor=None` (bare JSONL branch) no handler is installed. RED at the parent (no handler).
3. **C3, conformance**: confirm in recon that `tests/observability_conformance_config.py` enumerates no metric keys; if it does, add the new keys; otherwise no edit (say so in the report).
4. **C4, docs**: one paragraph in `docs/architecture.md` beside the FOR-1662 paragraph ("once per artifact root per process"): the final-leg measurements are mirrored to MLflow through the log-substrate bridge; the worker-log artifact stays the authoritative narrative.
5. **Live arm** (inside `just ship-it`'s container gate, the `tests/test_mlflow_live.py` family): one tiny run shows `stage/runtime_admission/seconds` (or whichever stage the tiny path brackets first; measure, do not assume) on the live server.

## Guard teeth by mutation

| id | mutation                                             | expected RED                          |
| -- | ---------------------------------------------------- | ------------------------------------- |
| M1 | drop the handler install in `build_sink_for_run`     | C2 RED (no record reaches either leg) |
| M2 | project `admission.start` instead of `admission.end` | C2 RED (counts and seconds missing)   |
| M3 | keep the handler after `finish()`                    | the post-finish arm of C2 RED         |
| M4 | project onto step 0 instead of the last global step  | the axis assertion in C1 and C2 RED   |

Preserved green: every existing test in `tests/test_h_metric_routing.py`, `tests/test_tracking_sinks.py`,
`tests/test_sinks.py`, `tests/test_h_worker_log_lines.py`, `tests/test_h_worker_live_logs.py`,
`tests/test_observability_conformance.py`, `tests/test_identity_epoch.py` (no reader-tuple member changes).

## Already ruled

- The seam above (operator ticket 2026-10-19). House rules 3 (no timeout/retry/sleep literal), 4 (measured comments, explicit retractions), 5 (RED first, both arms by mutation).
- REVIEW CONVERGENCE (standing): a finding blocks only for brief-named arms missing or green under mutation, demonstrated incorrect behaviour at the tip, or an envelope / never-edit breach; prose-only findings go to the docs sweep (FOR-1660).
- One full test suite at a time under `/tmp/esl-suite.lock` (mkdir lock, shared with the main checkout where the probe's controllers run; they do not hold it); the podman live gate runs one container stack at a time (house rule 2).
- No pods, spend, image builds, template or Linear writes inside the lifecycle.

## Allowed files

- `src/entropic_substrate/ladder/run/worker_log_events.py`, `src/entropic_substrate/ladder/capture/adapters/composite.py`, `src/entropic_substrate/ladder/run/h_metric_extractor.py`
- `tests/test_h_metric_routing.py`, `tests/test_tracking_sinks.py`, `tests/test_h_stage_metrics_bridge.py` (new, optional), `tests/observability_conformance_config.py` (only if it enumerates keys)
- `docs/architecture.md` (one paragraph)

Never edit: `ladder/run/worker_execution.py`, `ladder/run/seals.py`, `ladder/run/object_worker.py`,
`ladder/run/runner.py`, `ladder/run/worker_command.py`, every other reader-tuple member
(`checkpoint.py:66-116`), `kernel/runtime/logging.py` (the bridge is installed by the composite,
not by the process logger setup), `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/*`,
`.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/*`, `pyproject.toml`,
`uv.lock`, the briefs, the manifest.

## Acceptance

RED at `04ad7790` for C1 and C2; GREEN at the tip; M1 to M4 RED one at a time and restored;
`just check` exit 0 and the exit-gate runner green at the tip under the suite lock; the live gate
inside `just ship-it` green with the live arm measured; `git diff 04ad7790...HEAD --name-only`
inside *Allowed files* plus this brief (the branch's first commit); branch
`fix/for-1669-final-leg-mlflow-mirror` on `main`; the PR body copies the RED/GREEN counts and the
M1-M4 table verbatim and quotes the 2026-10-19 wave table above as the motivation; append-only history.
