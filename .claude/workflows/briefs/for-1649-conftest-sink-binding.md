# for-1649-conftest-sink-binding: capsys tests assert on real log lines, not closed-stream dumps

Authoritative brief for one `td-phase-lifecycle.cascade` run, written 2026-10-13 by the conductor
from the operator's ruling (2026-10-13: slot the conftest loguru sink fix into the parity stack near
the bottom, directly on the FOR-1637 PR head). Every anchor below was measured at `edabe74` with
`git show` or reproduced outside the repo; the evidence line sits beside each claim.

## Branch

| field                | value                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| branch               | `fix/for-1649-conftest-sink-binding`                                                                         |
| parent               | `fix/for-1637-h-worker-log-lines` at its PUBLISHED PR head (slice 1); the operator merges the stack in order |
| cascade `phase.mode` | `arch` (the test suite's observation contract changes: every capsys assertion starts reading real lines)     |
| wave                 | serial, main checkout, no lanes                                                                              |
| `gates.liveRequired` | `false` (no adapter or live path changes; `tests/conftest.py` and test files only)                           |
| gates                | `{"offline":"just check","exit":"just pc-ci","liveRequired":false}`                                          |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout, never a worktree                       |
| publish              | the cascade's publish stage via `just ship-it` once (its live gate runs regardless), ONE draft PR on slice 1 |

The branch EXISTS at launch: the conductor creates it with `git checkout -b fix/for-1649-conftest-sink-binding a43ef9f1af0683e80970bebc36dcee6eadd5c30c`
and commits this brief as its first commit (adopted in `args.conductorCommits`). Verify
`git merge-base --is-ancestor a43ef9f HEAD` and implement on top.

## Why (measured)

- `tests/conftest.py:68-69` at `edabe74`: `@pytest.fixture(autouse=True) def _configure_logging(capsys)` calls `configure_logging(json_path=None)` at fixture SETUP; its docstring (`:70-83`) promises it rebinds loguru's sinks "to the CURRENT per-test stdio streams".
- Reproduced by the conductor 2026-10-13 outside the repo (`scratchpad/for1649/test_capsys_identity.py`, pytest 9.1.1, loguru 0.7.3, `.venv/bin/python -m pytest -q -s -p no:cacheprovider -c /dev/null`): an autouse fixture that records `id(sys.stderr)` and adds a loguru sink at setup, then a test that logs at call time: `SETUP_ID 4449484384 CALL_ID 4449478112 SAME False`; `capsys.readouterr().err` begins `--- Logging error in Loguru Handler #1 ---\nRecord was: {...}`. The sink is bound to a stream object that is not the call-time stream.
- The FOR-1637 implementer measured the same inside the repo at 245e33a (ids 4541609600 vs 4541601536) and found that `tests/test_mlflow_async_sink.py` assertions count stderr lines by substring: `:245` "metric batch", `:318/:323/:327` "worker died", `:413` "drain", `:482` "after finish". They pass only because loguru's single-line `Record was: {...}` dump contains the message text. That is the false-green the module docstring of `src/entropic_substrate/kernel/runtime/logging.py` (`:21-34`) names as the failure mode this substrate exists to prevent.
- `tests/test_logging.py:39` (`test_configure_logging_rebinds_to_current_stderr`) and its siblings call `configure_logging(json_path=None)` INSIDE the test body, which is why they observe real lines; the conftest path is what the rest of the suite relies on.
- Consequence already paid: FOR-1637 named its `sink.finish.end` field `settled` instead of `drained` because a `drained` key in the dump added a second "drain" substring per line (`tests/test_mlflow_async_sink.py::test_finish_past_its_drain_budget_counts_stranded_records_and_still_terminates` went RED under the correct name).

## Contract surface owned

| contract                                                                                                                                                                                                                                                                                | kind                | frozen_after |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ------------ |
| The suite's observation contract: a `logger.*` call made during a test's CALL phase lands as a formatted `<ts> \| LEVEL \| message` line in that test's `capsys.readouterr().err` (or `.out` for `report=True`), never as a Loguru "Logging error" block. Owned by `tests/conftest.py`. | test infrastructure | this phase   |
| A regression pin that fails whenever a logger call inside a capsys test reaches a closed stream (RED by mutation: restore the setup-time binding).                                                                                                                                      | guard               | this phase   |

## Contracts consumed

- `kernel.runtime.logging.configure_logging(level, json_path)` (`logging.py:124-162` at `edabe74`): removes all sinks, adds the human stderr sink bound to `sys.stderr` and the report stdout sink bound to `sys.stdout` at CALL time of `configure_logging`. Its docstring's guarantee ("LAZY BY DESIGN", per-test rebind) must stay true; the module is NOT edited unless the root fix requires a stream-resolving sink, in which case the change is additive and the docstring is updated in the same commit (say so).
- `_pytest.capture` in the installed pytest 9.1.1 (`.venv/lib/python3.12/site-packages/_pytest/capture.py`): `CaptureFixture`, `suspend_global_capture` / `resume_global_capture`, `suspend_capturing` / `resume_capturing`. The root cause explanation must cite the lines that swap `sys.stderr` between fixture setup and test call.
- `tests/test_mlflow_async_sink.py` assertions listed above; `tests/test_logging.py`; `tests/test_sinks.py` (84 tests) and any other test that reads `capsys.readouterr()` after a logger call (enumerate with `grep -ln "readouterr" tests/` and run them).

## Sacred invariants

| invariant                     | verbatim                                                                                                                 | tripwire                                                                                                                                        |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| House rule 5                  | "TDD is non-negotiable, and guard-teeth prove both arms."                                                                | the regression pin is RED on the parent head before the conftest change and GREEN after; the mutation (rebind at setup only) turns it RED again |
| House rule 3                  | "Never fix flakiness by raising a timeout or a retry budget."                                                            | no retry, sleep or timeout literal changes anywhere                                                                                             |
| Production messages unchanged | tests are fixed, never the production log text                                                                           | `git diff --name-only` contains no file under `src/` except, at most, `kernel/runtime/logging.py` for an additive stream-resolving sink         |
| FOR-1637 branch untouched     | no edit under `src/entropic_substrate/ladder/run/*`, none to FOR-1637's test files beyond assertions that change meaning | `git diff --name-only` grep                                                                                                                     |
| Reader identity               | no checkpoint-reader-tuple member or source-reader member is edited                                                      | `tests/test_identity_epoch.py` green unmodified; no DG-A                                                                                        |
| Print ban and layer contract  | unchanged                                                                                                                | `tests/test_observability_conformance.py`, `lint-imports` green                                                                                 |

## Identity members

None. `tests/conftest.py`, the test files and `kernel/runtime/logging.py` are outside both reader tuples (`ladder/run/checkpoint.py::_READER_MODULES` at `edabe74` lists no `kernel.runtime.logging` and no test module). Editing a member is a HALT.

## Scope

1. **C1, root cause and RED pin.** A minimal reproduction committed as a test (`tests/test_conftest_sink_binding.py`, new): records `id(sys.stderr)` at fixture setup and at call, asserts that a `logger.warning` made in the test body appears in `capsys.readouterr().err` as a formatted line and that no `Logging error` block appears. Quote the `_pytest/capture.py` lines that explain the swap in the test's docstring. RED on the parent head (the dump appears), counts quoted.
2. **C2, the fix in `tests/conftest.py`.** Either rebind the sinks at call time (a fixture that yields after capsys is active for the call phase, or an autouse fixture ordered after capsys's own setup) or install sinks that resolve `sys.stderr` / `sys.stdout` at write time (a callable sink), whichever the measured capture internals support without weakening the module docstring's guarantee. If the callable-sink route is chosen and it belongs in `configure_logging`, the change is additive with a docstring update and a `tests/test_logging.py` pin.
3. **C3, the suite sweep.** Run the full unit suite under the host lock; list every test whose assertions change meaning once real lines are captured (expected: the `tests/test_mlflow_async_sink.py` substring counts, possibly `tests/test_sinks.py` and others found by `grep -ln readouterr tests/`); fix those tests to assert on the formatted lines (message text and level), never by changing production messages; record the before/after list in the PR body.
4. **C4, the `drained` rename is NOT reverted here.** FOR-1637 shipped `settled`; renaming back is a FOR-1637 follow-up decision for the conductor after this lands, not part of this slice.

## ACCEPTANCE

- `tests/test_conftest_sink_binding.py` RED on the parent head (dump observed, quoted) and GREEN at the tip; mutation (restore setup-time binding) RED, quoted.
- The sweep list: every test whose meaning changed, with its old and new assertion, in the PR body; the full unit suite green at the tip under the host lock; `tests/test_logging.py` and `tests/test_sinks.py` green.
- `grep -rn "Logging error" <captured stderr of the full suite>` is zero at the tip (capture the suite's `-rA` output under the lock and quote the count); on the parent head it is non-zero (quote).
- No file under `src/entropic_substrate/ladder/run/`, no reader-tuple member, no retry/sleep/timeout literal changed (grep over the diff quoted).
- `just check` green at the tip under the host lock; exit gate certified at the final tip by the runner's JSON; draft PR stacked on the slice-1 PR head with the filled body (root cause with `_pytest/capture.py` lines, the sweep table, RED/GREEN counts, mutation arm, three honest exits).

## Non-goals

- Does NOT change any production log message or event vocabulary (FOR-1637 owns those).
- Does NOT touch `src/entropic_substrate/ladder/run/*`, the adapters, or the FOR-1637 branch's commits.
- Does NOT rename `settled` back to `drained` (conductor follow-up after this lands).
- Does NOT raise pytest's capture timeouts or add retries.

## Allowed files

- `tests/conftest.py`
- `tests/test_conftest_sink_binding.py` (new)
- `tests/test_mlflow_async_sink.py`, `tests/test_sinks.py`, `tests/test_logging.py`, and any test file the sweep names (assertion changes only; list each in the PR body)
- `src/entropic_substrate/kernel/runtime/logging.py` ONLY for an additive stream-resolving sink option with its docstring update (say so; otherwise untouched)
- `.secrets.baseline` (mechanical companion, regenerated by the prek hook only)

Never edit: `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/regime_rows.py`, `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/run_ladder.sh`, `scripts/run_container_test.sh`, `tools/mapping.json`, `pyproject.toml`, `uv.lock`, `tests/observability_conformance_config.py`, anything under `src/entropic_substrate/ladder/run/`, anything under `core/`, the manifest, the plans, the briefs.

## Already ruled

- Operator 2026-10-13: the conftest sink fix is slice 2 of the parity stack, near the bottom, on the FOR-1637 PR head; publish through the sanctioned flow.
- FOR-1637 shipped `settled` (conductor ruling 2026-10-13); its reversal is not this slice's call.
- Message lint before every commit (`uv run gitlint --msg-filename`); write a fresh message file per commit.
- One full test suite at a time on this host under `/tmp/esl-suite.lock`, no worker cap.

## Guard teeth by mutation

| id | mutation                                                                                                       | expected RED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| -- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M1 | restore the setup-time `configure_logging` call and remove the call-time rebind (or the stream-resolving sink) | `tests/test_conftest_sink_binding.py` RED (dump observed, no formatted line)                                                                                                                                                                                                                                                                                                                                                                                                                 |
| M2 | change a swept assertion back to a substring count over the dump text                                          | MEASURED GREEN at the tip (2026-10-14, PR #342: `24 passed`, 0 dumps): the formatted line still carries the message substring, so a reverted count matches it. The brief's original expectation (RED, count 0) was wrong and is retracted here, not overwritten (house rule 4). The load-bearing arm is M1 over the swept files: the new assertions go RED on dumps (17 failed / 142 passed / 19 errors) while the old substring assertions passed against dumps on the parent (154 passed). |
| M3 | make the human sink write to a captured setup-time stream explicitly                                           | the regression pin RED and `tests/test_logging.py::test_configure_logging_rebinds_to_current_stderr` RED                                                                                                                                                                                                                                                                                                                                                                                     |
