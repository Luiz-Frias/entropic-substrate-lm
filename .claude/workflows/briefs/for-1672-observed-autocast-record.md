# for-1672-observed-autocast-record: record the OBSERVED bf16 autocast state beside the declared precision

DRAFT 2026-10-22 by the conductor (wave C, lane of a K=3 mass-patch-fanout; issue FOR-1672, from the H0 blind
review MAJOR finding that nothing captured distinguishes "bf16 requested and applied" from "bf16 requested,
ignored"). Anchors measured at origin/main `70d2b5f1` with `grep -n`; the implementer re-verifies them in recon.
The Linear ticket is authoritative. **DG-A lane: two touched modules are reader-tuple members.**

## Branch

| field                | value                                                                                               |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| branch               | `fix/for-1672-observed-autocast-record`                                                             |
| parent               | the wave base branch (wave B's zipped tip + the briefs commit)                                      |
| cascade `phase.mode` | `arch` (the run_start runtime precision record is a captured contract read by the analysis side)    |
| publish              | none in the lane; the lane returns `REPIN-HANDOFF` and the conductor lands the single commit (DG-A) |

## Why (measured at `70d2b5f1`)

- `src/entropic_substrate/core/sentence/runtime_precision.py`: `require_execution_precision()` (`:23-49`) builds `observed` with `default_dtype`, `matmul_precision`, `cuda_matmul_fp32_precision`, `effective_tf32`, the two tf32 override env vars, and refuses non-FP32 defaults (`:40-45`); it never mentions autocast. Callers: `ladder/run/execution.py:22` and `ladder/run/worker_execution.py:9` (import `require_execution_precision`); test `tests/test_sentence_runtime_precision.py:13`.
- Autocast sites: `ladder/run/execution.py:224` and `:345`, `ladder/run/generation.py:261` and `:285`, `ladder/run/span_states.py:179`, all `torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.profile.precision == "bf16")`.
- Declared-only copies of `profile.precision`: `composition/wiring/h_tiny_tracer.py:175`, `ladder/run/worker_command.py:204`.
- `core/sentence/profile.py:146` `precision: Literal["fp32", "bf16"]`, `:171` non-local classes require `("cuda", "bf16")`.
- Reader tuple (`ladder/run/checkpoint.py:65-118`) MEMBERS among the touch set: `core.sentence.runtime_precision`, `ladder.run.execution`, `ladder.run.worker_execution`, `core.sentence.profile`. Any byte change to them fails the unit-marked regime-row digest test in `tests/test_identity_epoch.py` at pre-commit, so the lane CANNOT commit: DG-A protocol (below).
- `mechanism_evidence.py:20-21` refuses non-FP32 state by contract; mechanism-row `dtype torch.float32` is correct and stays.

## Contract surface owned (arch: the design gate rules the form)

The observed entry: `runtime.precision.autocast = {"enabled": bool, "dtype": str, "device_type": str}`, measured INSIDE an autocast context opened exactly as the execution site opens it (same `device_type`, `dtype`, `enabled` expression), via `torch.is_autocast_enabled(device_type)` / `torch.get_autocast_dtype(device_type)`, and emitted on `run_start` beside the existing fields. Design question for the gate: D1 measure once at run start by opening a probe autocast context with the profile's settings in `require_execution_precision(profile)` (one site, record shape owned by the precision module) vs D2 measure at the first real execution step and splice into the record (observes the real site but moves the record off `run_start`). Recommendation: D1, with the probe context built from the same expression the sites use, so the record is "what torch reports for this profile on this device", declared and observed side by side; the gate records the alternative.

## Scope

1. **C1, tests RED at the parent** (`tests/test_sentence_runtime_precision.py`, additive): a bf16 profile on a CPU run records what torch reports for CPU bf16 autocast (quote the measured value in the test and the commit message, do not assume `True`); an fp32 profile records `enabled: False`; the record carries both the declared `profile.precision` and the observed entry, labelled.
2. **C2, implementation** per the design gate; `run_start` emission point verified in recon (the worker's run_start assembly in `ladder/run/worker_command.py` / `worker_execution.py`); the FP32-defaults refusal (`:40-45`) unchanged.
3. **C3, DG-A hand-off**: the lane implements, proves RED/GREEN and M1-M4 locally, STAGES the whole change set, saves `git diff --cached --binary` under its worktree, runs `just check` and quotes the expected RED (regime-row digest tests only), quotes the regime rows, and returns DECISION beginning with the literal `REPIN-HANDOFF`. The conductor lands ONE commit (change set + `tools/regime_rows.py` re-pin + dated read-old-table row), adds its SHA to `args.conductorCommits`, appends the implRuling verify-only clause, and resumes.

## Guard teeth by mutation

| id | mutation                                                   | expected RED                                |
| -- | ---------------------------------------------------------- | ------------------------------------------- |
| M1 | hard-code `enabled: True`                                  | the fp32-profile pin RED                    |
| M2 | report the declared precision as the observed dtype        | the bf16 CPU pin (measured torch value) RED |
| M3 | drop the `autocast` entry from the record                  | the record-shape pin RED                    |
| M4 | open the probe context with `enabled=True` unconditionally | the fp32-profile pin RED                    |

## Allowed files

`src/entropic_substrate/core/sentence/runtime_precision.py`, `src/entropic_substrate/ladder/run/execution.py` and/or `worker_execution.py` (call-site only, per the gate), the run_start assembly site named in recon, `tests/test_sentence_runtime_precision.py`, `tests/test_sentence_execution.py` (additive pins). Never: `tests/test_identity_epoch.py` (the conductor re-pins), `docs/identity/*`, `tools/*`, numerics (`kernel/*`, the FP32 islands), `profile.py` unless the gate rules it necessary.

## Acceptance

C1 RED at the parent, GREEN at the staged tip; M1-M4 RED one at a time and restored with the RED output quoted; `just check` at the staged tip RED ONLY on the regime-row digest tests (quoted) and GREEN after the conductor's re-pin commit; the exit-gate runner green at the conductor's commit; the design-gate ruling (D1 or D2) with its alternative in the hand-off report and the commit message; diff confined to the Allowed files. Consequence recorded, not acted on: a reader-identity re-pin means the NEXT probe needs an image rebuild and deploy-doc re-pin (ticket FOR-1672 acceptance 5 is satisfied only by that later probe).
