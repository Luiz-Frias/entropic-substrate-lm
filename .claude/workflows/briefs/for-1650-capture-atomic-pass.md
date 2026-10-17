# for-1650-capture-atomic-pass: a capture pass lands in `dest` only when every file in it is admissible

Authoritative brief for one `td-phase-lifecycle.cascade` run (slice 3b of the H observability parity
stack, umbrella FOR-1640; issue FOR-1650). Written 2026-10-15 by the conductor; every anchor measured
at `18f94ae` (the published head of slice 3, draft PR #343) with `git show`.

## Branch

| field                | value                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1650-capture-atomic-pass`                                                              |
| parent               | `fix/for-1639-capture-parity` at its PUBLISHED PR head `18f94ae` (slice 3)                      |
| cascade `phase.mode` | `arch` (the capture lane's on-disk contract changes: what "in `dest`" means)                    |
| wave                 | serial, main checkout, no lanes                                                                 |
| `gates.liveRequired` | `true` (the capture lane is exercised by the live H capture test inside `just ship-it`)         |
| gates                | `{"offline":"just check","exit":"just pc-ci","live":"just container-test","liveRequired":true}` |
| `repoToplevel`       | `/Users/luizfrias/CursorAI/entropic-substrate-lm`, the MAIN checkout                            |
| publish              | `just ship-it` once, ONE draft PR stacked on slice 3                                            |

The branch EXISTS at launch: the conductor creates it at `18f94ae` and commits this brief (plus the ruled
CLAUDE.md status edits, DN-CLAUDE-STATUS) as its first commit, adopted in `args.conductorCommits`.

## Why (measured at `18f94ae`, verified by the slice-3 CodeRabbit lane with a probe and a control)

- `capture_finished_runs` (`ladder/capture/reader/capture_results.py:175-223`) walks every FINISHED run, applies the tiny gate (`:202`) and the schema-tag partition (`:205-207`), and for the E lane downloads each `results/*.jsonl` straight to its final name in `dest` (`:213-215`, via `_download_artifact` `:81-94`, which stages in a `tempfile.TemporaryDirectory(dir=dest)` and `shutil.move`s the one file to `target`), then applies `final_eval_record(target)` (`:219`) and appends it to `captured` (`:220`). A file whose `target.exists()` is skipped (`:213`).
- The H lane (`_capture_h_run`, `:134-173`) downloads only when `fresh = not target.exists()` (`:160-162`), re-validates an existing file (`:166-168`) but appends to `captured` only when `fresh` (`:169-171`).
- `cmd_capture_results` (`composition/entrypoints/cli.py:548-618`) catches `LookupError` (`:594`) and `TelemetryGap` (`:595-600`, exit 2), then: `if not captured: "nothing new to capture", return 0` (`:601-603`); `--no-dvc` returns before DVC (`:604-606`); otherwise `dvc add <dest>` and `dvc push` (`dvc_commands`, `capture_results.py:341-343`) and `verify_cloud_sync` (`:346`).
- Consequence: a refusal anywhere in a pass (E or H lane) propagates before `dvc add`; the admissible files already moved into `dest` stay there untracked. The next pass skips them (E lane `:213`) or re-validates without counting them (H lane `:169`), returns `[]`, and the CLI exits 0 with "nothing new to capture". Probe (lane): E run listed before a refused H-tagged run; pass 1 raises with the E file on disk; pass 2 (refused run removed) returns `[]`; a fresh `dest` returns the E file.
- `capture_checkpoints` (`:225-338`) has the same shape: `target.exists()` skip (`:277`), download (`:279`), per-file admissibility (`CheckpointFormatError`, `:296-304`), and a per-file atomic strip-swap (`:315-333`, `.partial` sibling + `os.replace`), but a mid-pass refusal strands earlier admissible checkpoints identically.
- Existing pins: `tests/test_capture_results.py::test_capture_is_idempotent_across_invocations` (`:170`), `::test_a_refused_h_file_kept_on_disk_is_refused_again_on_rerun` (`:338-349`, asserts the refused file is re-validated from disk and `download_calls == 1`), `::test_h_capture_is_idempotent_once_admitted` (`:376`), `::test_capture_checkpoints_is_idempotent_across_invocations` (`:622`), `::test_cli_capture_results_telemetry_gap_is_a_controlled_refusal` (`:695`), `::test_cli_success_path_runs_the_cloud_gate_after_push` (`:826`). The `_capture_h_run` docstring (`:137-143`) states the re-validate-not-skip rule.

## Contract surface owned

| contract                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | kind                          | frozen_after |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ------------ |
| A capture pass is atomic with respect to `dest`: every artifact of the pass is downloaded and validated in a per-pass staging directory that is a SIBLING of `dest` (`<dest>.pending/<pass-id>/`, never inside `dest`, so `dvc add <dest>` never tracks it); only when every file of the pass is admissible are the files moved into `dest` (`os.replace`, same filesystem) and returned as `captured`. A refused pass moves nothing into `dest`; the refused file stays in the staging directory as evidence and is named in the exception. Both lanes (`capture_finished_runs` for E and H, `capture_checkpoints`). | capture lane on-disk contract | this phase   |
| "A file in `dest`" therefore means "captured by a fully admissible pass"; the existing `target.exists()` idempotency key keeps its meaning and the E-lane skip becomes correct. The H lane's re-validation of an existing `dest` file stays (a file in `dest` that fails validation is still refused, named).                                                                                                                                                                                                                                                                                                         | idempotency key               | this phase   |
| The CLI's "nothing new to capture" (exit 0) is reached only when a pass validated nothing new; after a refused pass the next pass re-downloads and captures the admissible files.                                                                                                                                                                                                                                                                                                                                                                                                                                     | CLI exit contract             | this phase   |

## Contracts consumed

- `final_eval_record`, `final_h_run_end`, `require_json_object_lines`, `TelemetryGap`, `CheckpointFormatError`: unchanged, never weakened.
- `_download_artifact` (`:81-94`): its temp-dir staging is replaced by the per-pass staging directory (one scratch per pass, not per file); the flat capture set in `dest` is preserved.
- `dvc_commands` / `verify_cloud_sync`: unchanged.
- `tests/test_capture_results.py::FakeCaptureClient`: unchanged in shape.

## Sacred invariants

| invariant                                   | verbatim                                                                                                | tripwire                                                                                                               |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| House rule 5                                | "TDD is non-negotiable, and guard-teeth prove both arms."                                               | the strand test is RED at `18f94ae` and GREEN at the tip; mutations M1-M4 both arms                                    |
| CLAUDE.md data flow                         | "inadmissible telemetry raises, it never returns an empty value"                                        | every refusal still raises with the file named; no fail-open path; the refused file is kept as evidence                |
| `dvc add <dest>` tracks only captured files | the staging directory is outside `dest`                                                                 | new pin: after a refused pass `dest` is empty and `<dest>.pending/` holds the evidence; `dvc_commands(dest)` unchanged |
| House rule 3                                | "Never fix flakiness by raising a timeout or a retry budget."                                           | no retry, sleep, deadline or timeout literal changes                                                                   |
| Reader identity                             | no reader-tuple member touched (`capture_results.py`, `composition/entrypoints/cli.py` are not members) | `tests/test_identity_epoch.py` green unmodified; no DG-A                                                               |

## Identity members

None. `ladder.capture.reader.capture_results` and `composition.entrypoints.cli` are outside both reader tuples (verified at `a43ef9f`, unchanged since). A member edit is a HALT.

## Scope

1. **C1, tests RED.** In `tests/test_capture_results.py`: (a) the strand test: an admissible E run listed before a refused H run; pass 1 raises `TelemetryGap` and `dest` holds NO file (the E file is in `<dest>.pending/<pass>/`), the refused H file is in the staging dir and named in the exception; pass 2 with the refused run removed returns the E file and places it in `dest`. (b) The same with the H lane first and the E lane refused. (c) The checkpoint lane: an admissible checkpoint before a `CheckpointFormatError` one; pass 1 raises and `ckpt_dest` is empty; pass 2 returns the first. (d) The CLI: with a fake client that refuses once then succeeds, the first invocation exits 2 and the second exits 0 with "1 file(s) captured", never "nothing new to capture" (patch `dvc_commands`/`subprocess.run` as the existing CLI tests do, `:695`, `:826`). (e) `test_a_refused_h_file_kept_on_disk_is_refused_again_on_rerun` (`:338`) keeps its intent (a refused file never reads as captured) with its assertion restated: the refused file lives in the staging dir, the rerun re-downloads (`download_calls == 2`) and refuses again; list this assertion change in the PR body. (f) Idempotency pins (`:170`, `:376`, `:622`) stay green unmodified. All RED at `18f94ae` where they must be (a-d), counts quoted.
2. **C2, implementation.** `capture_results.py`: a per-pass staging directory `<dest>.pending/<pass-id>/` (uuid4 hex, created at pass start), both lanes download into it, validate there, and a final commit step `os.replace`s every validated file into `dest` and returns them; on any exception nothing is moved and the staging directory is left as evidence; `_download_artifact` takes the staging directory instead of a per-file temp dir; `capture_checkpoints` keeps its strip-swap but performs it in the staging directory. Docstrings of both lane functions and `_capture_h_run` state the atomic-pass rule. `cli.py`: no behaviour change required; its docstring names the rule in one sentence if the implementer finds the `if not captured` branch needs explaining.
3. **C3, docs.** `docs/architecture.md` capture paragraph: one sentence on the atomic pass and the sibling staging directory.

## ACCEPTANCE

- C1 (a)-(d) RED at `18f94ae` (quoted) and GREEN at the tip; (e) restated and listed; (f) green unmodified; `tests/test_capture_results.py` whole file green.
- `grep -n "pending" src/entropic_substrate/ladder/capture/reader/capture_results.py` non-empty at the tip, empty at the parent; `grep -n "TemporaryDirectory(dir=dest)"` empty at the tip (quote both arms).
- Mutations M1-M4 both arms; no retry/sleep/timeout literal changed; zero new `print(`; no file under `ladder/run/`, no reader-tuple member.
- `just check` green under the lock; `just container-test` green inside `just ship-it` (the live H capture test exercises the new staging on a real server; quote its experiment and run id); exit gate certified at the final tip; draft PR stacked on `fix/for-1639-capture-parity` with the filled body (the strand probe before/after, RED/GREEN counts, mutation arms, the (e) assertion change, three honest exits).

## Non-goals

- Does NOT change any admissibility predicate, the schema-tag partition, the tiny gate or the run-name validator.
- Does NOT handle a `dvc add`/`dvc push` failure after a fully admissible pass (the CLI's `check=True` already raises; the operator re-runs DVC by hand); recorded, separate.
- Does NOT touch FOR-1651 (`telemetry.py` decode) or FOR-1652 (further teeth); `telemetry.py` and `h_telemetry.py` are never-edit here.
- No pods, spend, image builds.

## Allowed files

- `src/entropic_substrate/ladder/capture/reader/capture_results.py`
- `src/entropic_substrate/composition/entrypoints/cli.py` (docstring sentence only, if any)
- `tests/test_capture_results.py`
- `docs/architecture.md` (C3 sentence only); `.secrets.baseline` (hook-regenerated only)

Never edit: anything under `src/entropic_substrate/ladder/run/`, `ladder/capture/reader/{telemetry,h_telemetry}.py`, `ladder/capture/adapters/*`, `ladder/capture/ports/*`, `tests/test_identity_epoch.py`, `docs/identity/*`, `tools/regime_rows.py`, `tests/conftest.py`, `.pre-commit-config.yaml`, `.github/workflows/*`, `justfile`, `scripts/*`, `tools/mapping.json`, `pyproject.toml`, `uv.lock`, `tests/observability_conformance_config.py`, anything under `core/`, the manifest, the plans, the briefs.

## Already ruled

- Operator 2026-10-12/13: parity as one linear stack; sanctioned publish flow only.
- Conductor 2026-10-15: FOR-1650 lands as slice 3b BEFORE slice 4 (the capture evidence path for the probe re-run must not read green with nothing captured); the fix covers both lanes in one cut (the defect class, not one site).
- Conductor 2026-10-15 (REVIEW CONVERGENCE): a review finding blocks only if it is a brief-named or ruled guard-teeth arm that is missing or green under mutation, a demonstrated incorrect behaviour at the tip, or an envelope / never-edit breach; a further surviving mutation on correct behaviour is a deferred finding (severity minor, verdict PASS) with its recipe, copied into the PR body for the conductor to file.
- Message lint before every commit; one suite at a time under the host lock.

## Guard teeth by mutation

| id | mutation                                                                     | expected RED                                                              |
| -- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| M1 | move files into `dest` as each one validates instead of after the whole pass | strand tests (a), (b) RED (E file found in `dest` after the refused pass) |
| M2 | put the staging directory INSIDE `dest`                                      | the "`dest` empty after a refused pass" pin RED                           |
| M3 | skip validation for files already present in the staging directory           | the H refused-rerun test (e) RED                                          |
| M4 | checkpoint lane moves before validating                                      | strand test (c) RED                                                       |
