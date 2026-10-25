# for-1552-strict-mlflow-observer: re-base D1MlflowObserver on the MetricSink port with a STRICT sink

DRAFT 2026-10-21 by the conductor (wave B `wave-b-2026-10-21`, lane of a K=2 mass-patch-fanout; issue FOR-1552,
deferred out of p8 by ruling P8-OBSERVERS-RENAME-ONLY, 2026-09-23). Anchors measured at the wave base
`434b63d2` (= analysis/h0-hepp7g, the top of the H-series session's stack on #362) with `grep -n`; the implementer
re-verifies them in recon. The Linear ticket is authoritative.

## Branch

| field                | value                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1552-strict-mlflow-observer`                                                                       |
| parent               | the wave base branch `fix/wave-b-2026-10-21` (`434b63d2` + the briefs commit)                               |
| cascade `phase.mode` | `arch` (a port/adapter contract: the strictness seam of the MLflow sink is a contract surface; design gate) |
| publish              | none in the lane (`publishMode: defer`, CERTIFIED); the zip publishes                                       |

## Why (measured at `434b63d2`)

- `src/entropic_substrate/ladder/capture/adapters/d1_observability.py`: `D1JournalObserver` Protocol (`:40`), `D1MlflowMetricClient` Protocol (`:47`, `log_metric(run_id, key, value, step)`), `D1EventMetricExtractor` Protocol (`:60`), `D1MlflowObserver` (`:84`, `__init__` `:94` takes `client`, `run_id`, extractor; `observe` `:111` projects and calls `client.log_metric`, raising through), `ObservedD1Journal` (`:124`). The observer is NOT a `MetricSink` and the client is not the sink port.
- `src/entropic_substrate/ladder/capture/ports/metric_sink.py:40` `MetricSink` Protocol: `log(record)`, `log_artifact`, `annotate(tags)`, `finish(*, status)`.
- `src/entropic_substrate/ladder/capture/adapters/mlflow.py:279` `MlflowSink`: best-effort by contract; every tracking-write failure is counted in `_failed_writes` (`:351`, `:514`, `:529`, `:539`, `:577`, `:589`, `:669`, `:683`) and never raised to the caller; `log()` (`:542`) extracts on the caller's thread and enqueues for a worker.
- Tests: `tests/test_d1_observability.py` `_RecordingMlflowClient` (`:150`), observer tests `:206`, `:231` (failure PROPAGATES after the event is durable: the fail-loud guard), `:288`, `:301`, journal tests `:181`, `:247`. ZERO production constructors of `D1MlflowObserver` in `src` (grep); `ObservedD1Journal` / `D1JournalObserver` are consumed by `composition/entrypoints/d1_trainer_process_worker.py`, `composition/entrypoints/d1_trainer_worker_command.py`, `composition/wiring/d1_trainer_cli_services.py`, `ladder/run/trainer/d1_resume_runtime.py`, `ladder/run/trainer/d1_resume_trainer.py`, and `tests/test_d1_durable_failure_cursors.py`.
- Reader tuple (`ladder/run/checkpoint.py:66-116`) lists neither `ladder.capture.adapters.d1_observability` nor `ladder.capture.adapters.mlflow`; the four `composition.*_cli` members are untouched by this lane.
- Plan rows `P8-OBSERVERS-RENAME-ONLY` (`docs/plans/src-hexagonal-refactor.plan.md:872`) and `P13-OBS` (`:877`, postcondition AMENDED): the p13 "observer protocols gone" postcondition was amended, so this lane is pure consolidation of the deferred re-basing, not a p13 gate.

## Contract surface owned (arch: the design gate rules the form)

Option S1: a STRICT raising MLflow sink, a second adapter (`StrictMlflowSink`) implementing `MetricSink` that raises on any tracking-write failure synchronously (no worker thread, no counting). Option S2: a `strict: bool` keyword on `MlflowSink` that switches the failure path from count-and-continue to raise. Either way: the observer's fail-loud guard (`tests/test_d1_observability.py:231`) is preserved; `D1MlflowObserver` becomes an adapter of the `MetricSink` port (it takes a sink, not a `D1MlflowMetricClient`), keeping its journal-step projection; `D1MlflowMetricClient` stays public for one read-old window (the ticket says so) with a dated retraction note; `ObservedD1Journal` and its five consumers are untouched in behaviour. Recommendation for the design gate: S1 (a separate strict adapter keeps the best-effort sink's measured backpressure contract byte-identical and makes strictness a type, not a flag; the Strategy pattern at the second implementation of the same port). The design gate records the alternative.

## Scope

1. **C1, tests RED at the parent** (`tests/test_d1_observability.py`, explicit carve-out ruled by the ticket): the observer accepts a `MetricSink` (`isinstance(sink, MetricSink)` true for the fake) and refuses a non-sink; a strict sink raises on a failing client and the observer propagates AFTER the authoritative journal fsync (the existing `:231` guard, re-expressed on the port); the best-effort `MlflowSink` keeps counting (`write_failed`) and never raises (unchanged tests in `tests/test_sinks.py` / `tests/test_tracking_sinks.py`). RED at the parent: the observer rejects a sink (TypeError at `:101`) and no strict adapter exists.
2. **C2, implementation**: the strict adapter or flag (per the design gate), the observer re-based on the port, the client protocol kept with a retraction note; conformance: `tests/test_observability_conformance.py` universe unchanged (verify in recon; a new adapter module may need registering if the conformance config enumerates adapters).
3. **C3**: `docs/plans/src-hexagonal-refactor.plan.md` row `P8-OBSERVERS-RENAME-ONLY` gains a one-line "DISCHARGED by FOR-1552" note (plan rows are the exception to the no-docs rule: they are the ruling ledger).

## Guard teeth by mutation

| id | mutation                                                  | expected RED                                       |
| -- | --------------------------------------------------------- | -------------------------------------------------- |
| M1 | strict sink counts instead of raising                     | the propagation pin RED                            |
| M2 | observer accepts any object as sink (drop the port check) | the refusal pin RED                                |
| M3 | best-effort `MlflowSink` raises on a failed write         | the existing count-and-continue pins RED           |
| M4 | observer logs BEFORE the journal fsync                    | the durable-first ordering pin (`:181` family) RED |

## Allowed files

`src/entropic_substrate/ladder/capture/adapters/d1_observability.py`, `src/entropic_substrate/ladder/capture/adapters/mlflow.py` (S2) or a new `src/entropic_substrate/ladder/capture/adapters/mlflow_strict.py` (S1), `src/entropic_substrate/ladder/capture/adapters/__init__.py` (export only), `tests/test_d1_observability.py`, `tests/test_sinks.py` / `tests/test_tracking_sinks.py` (additive pins only), `tests/observability_conformance_config.py` (only if it enumerates adapters), `docs/plans/src-hexagonal-refactor.plan.md` (the one row note). Never: the five `ObservedD1Journal` consumers, any reader-tuple member, `docs/architecture.md` (post-zip sweep), `kernel/*`.

## Acceptance

C1 RED at the parent, GREEN at the tip; M1-M4 RED one at a time and restored; `just check` exit 0 and the exit-gate runner green under the suite lock; the design-gate ruling (S1 or S2) recorded in the commit message with its alternative; diff confined to the Allowed files.
