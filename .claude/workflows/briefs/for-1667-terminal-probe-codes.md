# for-1667-terminal-probe-codes: permanent D1 object-read codes fail fast; only recoverable reads stay transient

DRAFT 2026-10-18 by the conductor (umbrella FOR-1640; issue FOR-1667, review remediation of PR #351 /
FOR-1665, filed by the stack triage run `entropic-substrate-lm-350-352-20261018T184412Z`). Serial
cascade lifecycle on a follow-up branch at the stack top; published branches (#350 `8982165a`, #351
`c3cff2ce`, #352 `3f2f8dd8`) are never amended, rebased or force-pushed. Anchors measured by the
triage verification lane at `3f2f8dd8` with `git show`; the implementer re-verifies them in recon.

## Branch

| field                | value                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1667-terminal-probe-codes`                                                           |
| parent               | `fix/for-1662-final-leg-walks` at its PUBLISHED PR head (#352) = `3f2f8dd8`                   |
| cascade `phase.mode` | `normal` (no door, no reader-tuple member; an adapter predicate and a controller filter)      |
| `gates.liveRequired` | `true` (`just ship-it` inside the lifecycle: check, live container gate, `gt submit --stack`) |
| publish              | ONE draft PR stacked on #352                                                                  |

## Why (measured at `3f2f8dd8` by the triage lane, scratch `git archive`, repro file 10 passed, control 33 passed)

- `worker/h_probe_controller.py:252-254` docstring says "Only that type is transient"; `:269-287` catches `D1TrainerS3ObjectError` on the per-tick `collect_probe_terminal` call, re-raises only when the destination is non-empty (`:270`), otherwise counts, emits `host.terminal_probe.transient` and returns `None`; `:431` defers a provider-absent observation while the tick's read raised (`if absent and probe_tally.count == transient_before`); `:441-444` names the deadline stage and raises `TimeoutError`.
- The adapter `transport/d1_trainer_s3_object_adapter.py` raises the SAME type for permanent failures: `_read` `:195-225` raises `provider_failure` (any boto error that is not not-found, incl. AccessDenied 403), `response_invalid`, `body_read`, `object_bytes`, `byte_limit`, `metadata_invalid`; `safe_diagnostic()` (`:42-`) already derives `cause_type`, `provider_code`, `http_status` from allowlisted facts.
- MEASURED: with the marker read raising a permanent code every tick (`byte_limit`, `object_bytes`, `metadata_invalid`, `response_invalid`, `provider_failure` with an AccessDenied/403 cause) the controller polls to the fixture deadline and records `error_types=['TimeoutError']`, `failure_stages=['deadline']`, `terminal_probe_transient_count=3`, `last_terminal_probe_error={...code...}`; with the pod ABSENT too, `provider_absent_events=0` in all five cases: the FOR-1638 early exit is masked and a vanished pod with a corrupt marker also runs to the deadline (`controller_timeout_seconds`; 10800 s in the 2026-10-17 launch records).
- No test at `3f2f8dd8` pins a permanent code: every envelope test builds its error via `_timeout_error()` = `D1TrainerS3ObjectError(operation="read", code="provider_failure")` with no cause (`tests/test_h_probe_controller.py:503-504`); `:545` pins the absorb-everything deadline shape; `:571` covers only `ValueError` and bare `RuntimeError`; `:613` pins transient-then-success with an absent pod. `tests/test_d1_trainer_s3_object_adapter.py:136-215` pins that the adapter raises `byte_limit`, `metadata_invalid`, `object_bytes` and never exercises the controller.
- The ruled sibling: `_owned_pod_absent` (`:204-228`) filters `ProbeProviderError.code` against `_TRANSIENT_PROVIDER_CODES` (`:66`): classification lives with the transport, the controller filters.

## Contract surface owned

None new. Additive: an adapter-owned recoverability predicate on `D1TrainerS3ObjectError` (name it with the adapter; a method or module function reading the same facts `safe_diagnostic()` exposes) and the controller's filter on it. `h-probe-controller-result-v1` keeps every FOR-1665 key. No door, no profile/launch field, no reader-tuple member (`ladder/run/checkpoint.py:65-117` lists neither file).

## Ruling (conductor, 2026-10-18, inside the seam; alternative recorded)

PERMANENT (fail fast on the first tick, `failure_stages == ["terminal_capture"]`, `error_types == ["D1TrainerS3ObjectError"]`, no transient event, count unchanged): the integrity codes `response_invalid`, `metadata_invalid`, `byte_limit`, `object_bytes`; and `provider_failure` whose provider code is AccessDenied, InvalidAccessKeyId, SignatureDoesNotMatch, ExpiredToken, InvalidToken, NoSuchBucket, InvalidArgument, AuthorizationHeaderMalformed, RequestTimeTooSkewed, or whose HTTP status is 400 or 403, or whose cause is a ParamValidationError. TRANSIENT (the FOR-1665 envelope, deadline-bounded): everything else, INCLUDING `provider_failure` with an `unrecognized` cause type, connection and timeout causes (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError, ConnectionClosedError, ProxyConnectionError), SlowDown / InternalError / ServiceUnavailable / 500 / 503, and `body_read` with a timeout or closed-connection cause. Rationale: the 2026-10-17 kill that motivated FOR-1665 left no recorded code; a narrow transient allow-list could reintroduce it, and a deny-list of known-permanent codes is the asymmetric-safe branch (a wrongly-killed healthy run loses its evidence tail; a corrupt marker polled to the deadline is bounded and visible). Alternative NOT taken, recorded for the operator: transient allow-list with `unrecognized` failing fast. The absence deferral at `:431` keys on "a TRANSIENT read happened this tick"; a permanent read never defers it (it propagates first).

## Scope

1. **C1, tests RED at the parent `3f2f8dd8`.** (a) `tests/test_d1_trainer_s3_object_adapter.py`: the predicate returns False for every permanent code/provider-code pair above and True for each transient case above and for `unrecognized`; RED at the parent because the predicate is absent. (b) `tests/test_h_probe_controller.py`: (i) permanent code on the first tick propagates (`failure_stages == ["terminal_capture"]`, `error_types == ["D1TrainerS3ObjectError"]`, no `host.terminal_probe.transient` line, `terminal_probe_transient_count == 0`); (ii) a transient cause every tick keeps the FOR-1665 behaviour (deadline, count == ticks); (iii) absent pod + permanent code fails at the read on tick 1, not at the deadline; (iv) absent pod + transient code every tick documents the deferral (deadline), as today; (v) `_timeout_error()` gains a transient cause (ReadTimeoutError) so the existing FOR-1665 arms keep meaning "transient", and the every-tick pin at `:545` is re-read as the transient arm (retract its comment if it claims every code).
2. **C2, implementation**: the predicate in `transport/d1_trainer_s3_object_adapter.py` (allowlisted facts only; never messages, keys or request ids), its consumption in `_probe_terminal` (`worker/h_probe_controller.py`), the deferral keyed on a transient tick; no sleep, retry, counter abort or timeout literal (house rule 3); `worker/h_probe_log_events.py` only if a new event is needed (prefer none; the existing WARNING carries `code`, `provider_code`, `status`).
3. **C3**: fold the duplicated `_warnings` test helper (`tests/test_h_probe_controller.py:466-481`) into the `captured` idiom of `tests/test_h_probe_log_events.py:88-104` if convenient (trivial, #351 comment 4308232655); one sentence in `docs/architecture.md` beside the FOR-1665 host-controller paragraph (permanent object-read codes fail fast; recoverable ones are logged ticks).

## Guard teeth by mutation

| id | mutation                                                       | expected RED                    |
| -- | -------------------------------------------------------------- | ------------------------------- |
| M1 | absorb every code (drop the predicate check in the controller) | C1 (b)(i) and (b)(iii) RED      |
| M2 | predicate returns True for `byte_limit`                        | C1 (a) RED and (b)(i) RED       |
| M3 | predicate returns False for `unrecognized`                     | the unrecognized arm of (a) RED |
| M4 | permanent read no longer names stage `terminal_capture`        | (b)(i) RED on `failure_stages`  |

Preserved green: every FOR-1665 test re-read as the transient arm, `tests/test_h_probe_collection.py`, `tests/test_h_probe_operator_stop.py`, `tests/test_observability_conformance.py` (universe stays eleven).

## Already ruled

- Operator 2026-10-17: major/material triage findings route to a cascade-lifecycle re-run; stacked follow-up, never a rewrite of published history; after Linear capture, `gt merge` the stack, squash-merge release-please, hand back to live driving before the third 1K attempt.
- Conductor 2026-10-18: the classification ruling above; the adapter file (forbidden to the FOR-1665 lane) is ALLOWED here because it owns the classification; REVIEW CONVERGENCE (standing); house rules 3, 4, 5.
- No pods, spend, image builds, template or Linear writes inside the lifecycle. One full test suite at a time under `/tmp/esl-suite.lock` (mkdir lock, no worker cap).

## Allowed files

- `src/entropic_substrate/ladder/run/transport/d1_trainer_s3_object_adapter.py`, `src/entropic_substrate/ladder/run/worker/h_probe_controller.py`, `src/entropic_substrate/ladder/run/worker/h_probe_log_events.py` (only if needed)
- `tests/test_d1_trainer_s3_object_adapter.py`, `tests/test_h_probe_controller.py`, `tests/test_h_probe_log_events.py` (helper fold only)
- `docs/architecture.md` (one sentence)

Never edit: `ladder/run/worker_command.py` (incl. its `Config` line), `ladder/run/h_probe_events.py`, `ladder/run/h_probe_collection.py`, `ladder/run/object_worker.py`, every reader-tuple member, `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/*`, `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/*`, `pyproject.toml`, `uv.lock`, the briefs, the manifest.

## Acceptance

RED at `3f2f8dd8` for C1 (a) and (b)(i)(iii); GREEN at the lane tip; M1 to M4 RED one at a time and restored; `just check` exit 0 and the exit-gate runner green at the tip under the suite lock; the live gate inside `just ship-it` green; `git diff 3f2f8dd8...HEAD --name-only` inside *Allowed files* plus this brief and the two conductor doc corrections in the branch's first commit; the PR body copies the RED/GREEN counts and the four mutation arms verbatim and names the triage reproduction (five permanent codes, present and absent, all to the deadline at `3f2f8dd8`).
