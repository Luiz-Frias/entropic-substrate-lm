# for-1662-final-leg-walks: one memo-less admission walk per process, instrumented, and the five later stages reuse it

DRAFT 2026-10-17 by the conductor (umbrella FOR-1640; issue FOR-1662; measured on wave
`h0-plumbing-1k-v2.l4sbyp`). Wave lane 2 of 2 in the mass-patch fan-out launched from `main`; the
other lane is FOR-1665. Anchors measured at `ec378408` (main, release 0.3.0); the implementer
re-verifies them in recon. This lane edits reader-tuple members: DG-A applies (see *Identity members*).

## Branch

| field                | value                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1662-final-leg-walks`                                                                      |
| parent               | the wave parent branch (the FOR-1666 overlay commit stacked on `main` @ `ec378408`)                 |
| cascade `phase.mode` | `arch` (the Opus design gate rules on the reading of DG-B below; seven reader-tuple members change) |
| wave                 | lane in a nested worktree; `publishMode: defer`; the zip publishes ONCE                             |
| `gates.liveRequired` | `false` in the lane; the zip's single `just ship-it` runs the live gate on the stack                |
| zip order            | SECOND (identity re-pin; restacked onto the FOR-1665 tip)                                           |

## Why (measured 2026-10-17 from the pods' container logs and the published `checkpoint-*.json` events)

- `runner.py:396-412`: after `EVALUATION_END` the runner writes the progress receipt and calls `read_training_progress(... memo=None if (complete or stopped) else memo)` before `on_checkpoint` publishes the snapshot; the comment names the ruling: "DG-B/C12: only the mid-training, per-checkpoint call trusts the memo; a genuine completion or pause always re-walks the full chain as its admission predicate". Nothing is logged inside that window (the FOR-1637 vocabulary has no event between `evaluation.end` and `snapshot.start`).
- MEASURED per-checkpoint `publication.l2_seconds` (the memoized walk): 44.6 / 55.2 / 99.8 s at checkpoints 250 / 500 / 750 (magnitude; sphere 45.1 / 56.0 / 100.2). At completion (`memo=None`): `evaluation.end` 17:04:10Z to `snapshot.start name=checkpoint-00001000` 17:18:50Z = **880 s** (magnitude), 877 s (sphere). Pod during the walk: GPU 0%, network ~30 KB/min, CPU one core.
- The same memo-less walk is then repeated by every later stage of the same process: `worker_execution.py:184-192` (`training_readback`, MEASURED 880.6 s), `generation.py:55` (stage `generation` 1070.5 s total), `span_diagnostics.py:46` plus `read_seal` at `:56` (stage 2379.5 s total), `trace_capture.py:121` and `:352` (stage killed after 362 s by the FOR-1665 defect), `object_worker.py:186` (native verification / result, not reached). Seven call sites per completed run; `composition/cli.py:124` is an eighth, out of scope.
- `read_training_progress` (`runner.py:72-150`) walks `read_seal` on `last_seal`, then for EVERY evaluation `read_evaluation` AND `read_seal(point)`; `read_seal` (`seals.py:121-250`) verifies the source corpus against `profile.source_digest` on each call (`:145`) and walks `cursor` through journals (`read_journal`, 690 KB each at 1K) until a memo hit (`SealMemo.get`, `:72`) or the root. The per-node cost that makes four nodes plus three evaluations take 880 s is UNMEASURED (candidates: the 2.7 MB source re-read per `read_seal`, journal JSON parse and hash, `.pt` hashing).
- The 2026-10-12 wave wedged at the identical point, so the behaviour predates the parity stack; the FOR-1647 system-metrics risk is ruled out by that comparison.

## Contract surface owned (draft; the design gate confirms)

| contract                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | kind                               | frozen_after |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- | ------------ |
| Admission walk cardinality: a completed (or paused) run performs the memo-less full-chain admission walk EXACTLY ONCE per worker process, at the runner's completion point (DG-B's predicate, unchanged); that walk populates a fresh `SealMemo` which `run_probe_training` returns beside the receipt, and `worker_execution` passes it to `training_readback`, `run_probe_generation`, `run_span_diagnostics`, the two `trace_capture` reads and the `object_worker` result read, each of which keeps calling `read_training_progress` (same predicate, same raises) with that memo. A fresh PROCESS (resume, host replay, `composition/cli.py`) still walks the full chain with `memo=None`: the admission guarantee is per process, exactly as `SealMemo`'s docstring (`seals.py:47-70`) already states. | custody protocol (reading of DG-B) | this phase   |
| Instrumentation: new worker log events `admission.start` and `admission.end` in `worker_log_events.py` emitted around every `read_training_progress` call with fields `caller` (one of `completion`, `readback`, `generation`, `span_diagnostics`, `trace_capture`, `native_verification`), `memo` (`fresh` / `shared`), `seals_walked`, `evaluations`, `journals_read`, `seconds`; plus `seal_walk.node` only at DEBUG. Rendering stays in `worker_log_events.py`; no `print`.                                                                                                                                                                                                                                                                                                                              | log vocabulary (additive)          | this phase   |
| Measurement, not a change: the per-node cost of the memo-less walk on a 1K-shaped local fixture, profiled (`cProfile`, top 15 by cumulative time), reported in the PR body with its regime. Any speed-up of the walk itself beyond memo reuse is a DECISION, not an edit.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | evidence                           | n/a          |

## Identity members

h-checkpoint-reader-v2 members in scope: `ladder/run/runner.py`, `ladder/run/seals.py` (only if the memo needs a public constructor or a counter), `ladder/run/worker_execution.py`, `ladder/run/generation.py`, `ladder/run/span_diagnostics.py`, `ladder/run/trace_capture.py`, `ladder/run/object_worker.py`. `worker_log_events.py` is NOT a member. ONE DG-A hand-off for all of them: the lane implements the whole rung, runs its RED/GREEN and mutation arms, STAGES the complete change set, saves `git diff --cached --binary` to `/tmp/for-1662-staged.patch`, runs `just check` under the wave lock and quotes the expected RED (the regime-row digest tests in `tests/test_identity_epoch.py` and nothing else), and returns `DECISION` whose text begins with `REPIN-HANDOFF`. The conductor lands ONE commit (lane change set + the three checkpoint regime rows via `tools/regime_rows.py` + a `docs/identity/read-old-table.md` row), puts its SHA in `args.conductorCommits`, appends a dated `implRuling` clause, and resumes the wave. A second member hand-off is a defect.

## Scope

1. **C1, tests RED at the parent tip.** (a) `tests/test_final_leg_admission.py` (new): on the local tiny fixture idiom of `tests/test_sentence_worker_execution.py`, run the complete final leg through `execute_probe_request` with `read_seal` wrapped to count calls made with `memo is None`; at the parent the count is 7 (quote it), after the change exactly 1; every stage still produces its artifacts and the result status is unchanged. (b) The same fixture: the worker log carries exactly one `admission.start`/`admission.end` pair per caller with the field set above and `memo=fresh` only for `completion`. (c) `tests/test_sentence_runner.py`: a resume in a fresh process (new `SealMemo` / `memo=None`) still walks every node (`seals_walked == chain length`) and still refuses a tampered journal (existing refusal tests stay green and are named in the PR body). (d) `tests/test_sentence_seals.py`: `SealMemo` populated by a memo-less walk answers `get` for every node of that walk.
2. **C2, implementation**: `runner.py` (the completion/pause call builds and returns the populated memo; `admission.*` events; the receipt and counters unchanged), `worker_execution.py` (threads the memo through the five later stages), `generation.py`, `span_diagnostics.py`, `trace_capture.py`, `object_worker.py` (each accepts an optional `memo: SealMemo | None = None` and passes it to `read_training_progress` / `read_seal`; defaults keep every other caller byte-for-byte on the memo-less path), `worker_log_events.py` (constants + render), `seals.py` only if needed.
3. **C3, measurement + docs**: the profiled per-node cost table in the PR body; `docs/architecture.md`: one paragraph (admission walk once per process; the memo is the per-process carrier; fresh processes re-walk); the conductor's `docs/identity/read-old-table.md` row.

## Guard teeth by mutation

| id | mutation                                                       | expected RED                                              |
| -- | -------------------------------------------------------------- | --------------------------------------------------------- |
| M1 | pass `memo=None` in `training_readback` again                  | (a) RED: count 2                                          |
| M2 | pass the shared memo to the COMPLETION call in the runner      | (a) RED: count 0 with `memo=fresh` absent; (b) RED        |
| M3 | drop the `admission.end` emission                              | (b) RED                                                   |
| M4 | make `SealMemo.remember` a no-op                               | (d) RED and (a) RED (every later stage walks again)       |
| M5 | skip `read_seal`'s source-digest check when a memo is supplied | an existing refusal test or (c) RED (admission unchanged) |

## Already ruled

- Operator 2026-10-17: two-lane fan-out, linear stack, one `just ship-it`, triage, `gt merge`, release squash; the third 1K attempt waits for both fixes.
- DG-B (FOR-1567 design gate, 2026-09-28): the fresh-process full-chain walk is the admission predicate at resume, pause and completion. This brief READS it as per process, which `SealMemo`'s own docstring already says; the design gate confirms or returns DECISION (then the conductor surfaces it to the operator; no auto-ruling).
- House rule 3: no timeout or retry is added anywhere in this lane. House rule 4: the `publication` timings and the 880 s measurement go into the PR body as measurements; any comment that claimed the walk was cheap is retracted explicitly.
- DG-A for the seven members (above). The wave commit lock (`implRuling`) applies to every `just check` / exit-gate run.
- No pods, spend, image builds or template changes inside the lifecycle.

## Allowed files

- `src/entropic_substrate/ladder/run/runner.py`, `seals.py`, `worker_execution.py`, `generation.py`, `span_diagnostics.py`, `trace_capture.py`, `object_worker.py` (members; ONE DG-A hand-off), `src/entropic_substrate/ladder/run/worker_log_events.py`
- `tests/test_final_leg_admission.py` (new), `tests/test_sentence_runner.py`, `tests/test_sentence_seals.py`, `tests/test_sentence_worker_execution.py`, `tests/test_worker_log_events.py` (if a vocabulary count is pinned there)
- `docs/architecture.md` (one paragraph); `docs/identity/read-old-table.md` and `tests/test_identity_epoch.py` (conductor re-pin commit ONLY); `.secrets.baseline` (hook-regenerated only)

Never edit: `ladder/run/worker_command.py`, `ladder/run/worker_completion.py`, `ladder/run/evaluation_records.py`, `ladder/run/journal.py`, `ladder/run/execution.py`, `ladder/run/checkpoint.py`, `ladder/run/worker_snapshot.py`, `ladder/run/worker_transport.py`, `composition/*`, anything under `core/`, every host file the FOR-1665 lane owns (`worker/h_probe_*`, `h_probe_collection.py`, `h_probe_events.py`), profile or schedule fields, `tools/regime_rows.py`, `tests/conftest.py`, `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/*`, `pyproject.toml`, `uv.lock`.

## Acceptance

RED at the parent tip for C1 (a), (b), (d); GREEN at the lane tip (after the conductor's re-pin commit, `just check` exit 0); M1 to M5 RED one at a time and restored; `git diff <parent>...HEAD --name-only` inside *Allowed files*; the per-node cost table and the 7-to-1 count in the PR body; the exit-gate runner green at the conductor's tip under the wave commit lock.
