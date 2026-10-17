# for-1665-host-terminal-probe-retry: the host controller survives a transient R2 error on its per-tick terminal probe

DRAFT 2026-10-17 by the conductor (umbrella FOR-1640; issue FOR-1665). Wave lane 1 of 2 in the
mass-patch fan-out launched from `main`; the other lane is FOR-1662 (`for-1662-final-leg-walks.md`).
Anchors measured at `ec378408` (main, release 0.3.0) with `sed`/`grep`; the implementer re-verifies
them in recon.

## Branch

| field                | value                                                                                |
| -------------------- | ------------------------------------------------------------------------------------ |
| branch               | `fix/for-1665-host-terminal-probe-retry`                                             |
| parent               | the wave parent branch (the FOR-1666 overlay commit stacked on `main` @ `ec378408`)  |
| cascade `phase.mode` | `normal` (no contract surface, no reader-tuple member, no door)                      |
| wave                 | lane in a nested worktree; `publishMode: defer`; the zip publishes ONCE              |
| `gates.liveRequired` | `false` in the lane; the zip's single `just ship-it` runs the live gate on the stack |
| zip order            | FIRST (least risk): host-side files only, no identity re-pin                         |

## Why (measured 2026-10-17, wave `h0-plumbing-1k-v2.l4sbyp`, pod 5kkh7mq35wne05)

- `worker/h_probe_controller.py:353-363`: every poll tick sets `stage = "terminal_capture"` and calls `collect_probe_terminal(objects, guard.launch.request, root, expected_pod_id=pod_id, max_marker_bytes=...)`. `h_probe_collection.py:46-53` reads the four marker names (`complete`, `paused`, `stopped`, `failure`) through `read_probe_event` (`h_probe_events.py:107`, `objects.read(key)` at `:133`; a missing object returns `None`, never an error). The shared object client is built at `worker_command.py:71` with `Config(connect_timeout=15, read_timeout=15, retries={"max_attempts": 0})`.
- Any exception raised by those reads leaves the poll loop through `except BaseException` at `:386`, which appends the stage and falls into stop handoff; the controller then removes the pod (`host.stop.handoff reason=cancelled`, `watchdog_join ... status=terminated`).
- MEASURED: at 18:37:08Z the magnitude controller recorded `error_types=["D1TrainerS3ObjectError"]`, `failure_stages=["terminal_capture"]`, `termination_verified=true`, `workload_verified=false`, `collection=null`, and removed a healthy pod (its system metrics ran to 18:37:13Z; its container log showed `stage.start stage=trace_capture` at 18:31:05Z). The error MESSAGE is nowhere: the result record carries `type(error).__name__` only (`:507`), the host log has no WARNING line for it.
- The provider-side probe already has the ruled pattern: `_owned_pod_absent` (`:204-228`) catches `ProbeProviderError`, and for a code in `_TRANSIENT_PROVIDER_CODES` emits `HOST_PROVIDER_TRANSIENT` (`"host.provider.transient"`, `h_probe_log_events.py:75`) as a WARNING and returns `False` so the poll continues "at the existing cadence (no counter, sleep or timeout is added)". The terminal probe has no such envelope.
- `D1TrainerS3ObjectError` (`transport/d1_trainer_s3_object_adapter.py:33-50`) carries `operation` and `code` and a `safe_diagnostic()` that returns allowlisted facts only (never messages, URLs, keys, headers or request ids); the worker's own poll (`object_worker.py:376-405`, `stop_requested`) already treats a raised read as retryable with `VerificationReadRetry` (`worker_transport.py:45`, ruled count, backoff) and logs `STOP_MARKER_RETRY` per attempt.
- Tests: `tests/test_h_probe_controller.py` builds a fake `Provider`/`Guard`/`Control` (`:34-78`) and monkeypatches `collect_probe_terminal` at `:142`, `:269`, `:290`; `:449` pins `failure_stages == ["terminal_capture"]` for a case the implementer must read before touching (it may be the deadline path, not a raised read). `tests/test_h_probe_log_events.py:121` pins `len(events.H_HOST_LOG_EVENTS) == 17` (the vocabulary count moves with every new host event; retract the pinned number in the same commit, with the measurement).

## Contract surface owned

None new. Host-side behaviour only: `worker/h_probe_controller.py`, `h_probe_collection.py` (if the retry lives at the read site), `worker/h_probe_log_events.py` (new host events), the `h-probe-controller-result-v1` record gains ADDITIVE fields (`terminal_probe_transient_count`, `last_terminal_probe_error` = the `safe_diagnostic()` dict) and keeps every existing key; no reader-tuple member (`ladder/run/checkpoint.py:65-117` lists none of these), no door, no profile or launch field.

## Scope

1. **C1, tests RED at the parent tip.** (a) `tests/test_h_probe_controller.py`: a fake object port whose terminal-marker `read` raises `D1TrainerS3ObjectError(operation="get", code="Timeout")` on the first tick and returns `None` afterwards; the controller keeps polling, never hands off, and later collects the terminal normally (`status == "terminal_collected"`), with `terminal_probe_transient_count == 1` and `last_terminal_probe_error` equal to the error's `safe_diagnostic()`; the host log carries ONE `host.terminal_probe.transient` WARNING with `operation`, `code`, `attempt` fields (use the existing exact-stderr idiom of `tests/test_h_probe_operator_stop.py` or the log sink the host tests already capture). (b) The same port raising on EVERY tick: the controller runs to its DEADLINE (the existing `TimeoutError("controller deadline expired without terminal evidence")` path), the pod is removed by the deadline authority, `failure_stages` carries `deadline`-side evidence and `terminal_probe_transient_count` equals the tick count; it must NOT carry `terminal_capture` as the failing stage. (c) `tests/test_h_probe_log_events.py`: the vocabulary count pin moves to the measured new count with a dated retraction comment naming 17 as the previous measured value. (d) An existing-behaviour pin: a `ValueError` from `collect_probe_terminal` ("conflicting terminal claims") STILL fails the controller immediately (not every exception is transient; only `D1TrainerS3ObjectError` is).
2. **C2, implementation**: wrap the per-tick `collect_probe_terminal` call so that a `D1TrainerS3ObjectError` is emitted as `HOST_TERMINAL_PROBE_TRANSIENT` (`"host.terminal_probe.transient"`) with `pod_name`, `operation`, `code`, `provider_code`, `status`, `attempt` (the running count), the tick continues to `_progress_tick` and the wait, and the count plus the last `safe_diagnostic()` land in the result record; mirror `_owned_pod_absent` EXACTLY: no sleep, no counter-based abort, no timeout literal (house rule 3); the controller deadline and the watchdog remain the only removal authorities. Do NOT change `worker_command.py:71` (the `Config` line; FOR-1567 non-goal stands). Do NOT retry inside `read_probe_event` or the adapter; the envelope is the controller's poll loop.
3. **C3, docs**: one paragraph in `docs/architecture.md` next to the host controller description (a transient R2 read error is a logged tick, not a terminal); the conductor amends the runbook memory.

## Guard teeth by mutation

| id | mutation                                                    | expected RED                                                      |
| -- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| M1 | remove the transient envelope (re-raise on the first error) | test (a) RED: `controller_failed` with `terminal_capture`         |
| M2 | swallow EVERY exception in the envelope                     | test (d) RED: the conflicting-claims `ValueError` no longer fails |
| M3 | drop the `host.terminal_probe.transient` emission           | test (a) RED on the log assertion                                 |
| M4 | stop recording the count / diagnostic in the result         | tests (a) and (b) RED on the record keys                          |

Preserved green: every existing test in `tests/test_h_probe_controller.py`, `tests/test_h_probe_collection.py`, `tests/test_h_probe_operator_stop.py`, `tests/test_observability_conformance.py` (the universe stays eleven; no new entrypoint).

## Already ruled

- Operator 2026-10-17: FOR-1665 and FOR-1662 run as a two-lane mass-patch fan-out off `main`, zipped into ONE linear stack and published by ONE `just ship-it`; triage follows; merge via `gt merge` then the release PR squash.
- Conductor 2026-10-17: the envelope mirrors `_owned_pod_absent` (ruled pattern, FOR-1645 slice 5); only `D1TrainerS3ObjectError` is transient here; the deadline is the bound; additive result fields, no schema rename; `safe_diagnostic()` is the only error payload (secrets never enter logs or records).
- REVIEW CONVERGENCE (standing): a finding blocks only for brief-named arms missing or green under mutation, demonstrated incorrect behaviour at the tip, or an envelope breach.
- No pods, spend, image builds or template changes inside the lifecycle. The wave commit lock applies (see `implRuling`).

## Allowed files

- `src/entropic_substrate/ladder/run/worker/h_probe_controller.py`, `src/entropic_substrate/ladder/run/worker/h_probe_log_events.py` (new event constants + rendering), `src/entropic_substrate/ladder/run/h_probe_collection.py` (only if the envelope is placed at the read site; prefer the controller)
- `tests/test_h_probe_controller.py`, `tests/test_h_probe_log_events.py`, `tests/test_h_probe_collection.py` (only for (d) if it fits better there)
- `docs/architecture.md` (one paragraph)

Never edit: `ladder/run/worker_command.py`, `ladder/run/object_worker.py`, `ladder/run/h_probe_events.py`, `ladder/run/h_probe_launch.py`, `ladder/run/transport/*`, `ladder/run/worker_transport.py`, every reader-tuple member (`ladder/run/checkpoint.py:65-117`), `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/mapping.json`, `tests/observability_conformance_config.py`, `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/*`, `pyproject.toml`, `uv.lock`, anything the FOR-1662 lane owns (`runner.py`, `seals.py`, `worker_execution.py`, `generation.py`, `span_diagnostics.py`, `trace_capture.py`, `worker_log_events.py`, their tests).

## Acceptance

RED at the parent tip for C1 (a), (b), (c); GREEN at the lane tip; M1 to M4 RED when applied one at a time and restored; `just check` exit 0 and the exit-gate runner green at the lane tip under the wave commit lock; no file outside *Allowed files* in `git diff <parent>...HEAD --name-only`; the PR body (written by the zip's publish) copies the implementer's RED/GREEN counts and the four mutation arms verbatim and names the measured 2026-10-17 kill as the motivating event.
