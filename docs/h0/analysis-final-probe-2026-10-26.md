# H0 final-probe analysis: waves h0-final-probe-1k-v4.92c9db and h0-final-probe-resume-v4.1755b0

Fresh-session analysis of the H0 final probe, in the shape of
`docs/h0/analysis-hepp7g-2026-10-20.md`. This is a PLUMBING analysis with a test-only gate:
no scientific claim, no H1 eligibility beyond "the machinery exists", no spend. Every table
row carries its evidence class and the file plus row (`L<n>` is the 1-based line in the named
file) or record that supports it. Nothing below was read from MLflow, R2, a pod, or any session
ledger; every number is from the four DVC captures named in "Inputs read" or from a reader
executed over them in this session.

Evidence classes: MEASURED (read or executed over the captured files by this session),
DERIVED_FROM_MEASURED (arithmetic over MEASURED values, from raw floats), RECORD (a value
written by the controller, watchdog or in-image check run and read verbatim from its captured
record; not re-executed here), CONDUCTOR-REPORTED (stated in the brief, not in any capture).

## Inputs read

| item | value | evidence |
| --- | --- | --- |
| MLflow-side capture | `results/h0-final-probe-1k-v4-92c9db.dvc`: md5 `ed953d4af265275a9609e7e4a09e2903.dir`, 12 files, 14,096,149 bytes. Six h-series runs: the four pods of this probe plus the two HEpP7g runs (`988d8388`, `a163161f`, code commit `9d353fcc`), present because `capture-results` captures every admitted h-series run and files are bound by the `run_start` row's `request_sha256`, never by name | MEASURED, pointer text and `run_start` rows |
| Wave 1 host side | `results/h0-final-probe-1k-v4.92c9db.dvc`: md5 `8d9c98013938f623bbb35e870e518aac.dir`, 161 files, 57,337,118 bytes | MEASURED, pointer text |
| Wave 2 host side | `results/h0-final-probe-resume-v4.1755b0.dvc`: md5 `45395630615c9efca60cf87ad9b7bec6.dir`, 123 files, 46,441,594 bytes | MEASURED, pointer text |
| In-image check run | `results/h0-final-probe-checks-1755b0.dvc`: md5 `bfaedab30df9e4ec5a1038e1ff1e9b0c.dir`, 7 files, 18,505 bytes | MEASURED, pointer text |
| Pull result | `dvc pull` of the four pointers: 174 files fetched, 303 files added (12 + 161 + 123 + 7) | MEASURED (dvc output), DERIVED_FROM_MEASURED (sum) |
| Readers executed here | `h_telemetry.final_h_run_end` over all six metrics files; `scripts/h0_final_probe_checks.py` over four locally written plans (section K); `json.loads` over every line of every JSONL file | MEASURED |
| Code read for semantics only | `core/sentence/rollout.py` (outcome vocabulary and the `invalid_utf8` branch), `ladder/run/generation.py` (`_summary`, `continuation_seed`), `ladder/analyze/h_final_probe_checks.py` (K1 to K7 criteria) at main `0f0f36c4` | read, not evidence |

### Arm-to-file binding

Each captured metrics file is bound to an arm by its `run_start.request_sha256` (metrics L3 of
each file) equal to the arm's `request_sha256` in `wave.json`. File names below are shortened
to the MLflow run id suffix; the full name is `h-worker-<profile_identity>-<run id>-{metrics,worker-log}.jsonl`.

| arm | request_sha256 | profile_identity | run id | pod_id | metrics lines | log lines | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| W1 sphere | `8eb81425…f97173` | `3390f577…238b66` | `38c25d26` | `g8087m4ah3ops2` | 1031 | 278 | MEASURED, `wave.json` arms.sphere; metrics L3; `runtime.pod_id` |
| W1 magnitude | `111640fe…d9c8f` | `5ece5d1b…004880` | `14637e75` | `xm1c6coypgd2by` | 1031 | 278 | MEASURED, same rows |
| W2 A, magnitude, declared pause `["prediction", 250]` | `c658b4f0…465f5` | `5ece5d1b…004880` | `f25744b7` | `i11npiybn4nwnt` | 515 | 99 | MEASURED, `wave.json` arms.magnitude, `inputs/request.json` `pause` |
| W2 B, magnitude-resume | `f679b1a6…99810` | `5ece5d1b…004880` | `5e76f21c` | `m8ootdwxmkuveg` | 528 | 256 | MEASURED, `wave.json` arms.magnitude-resume, `inputs/request.json` `resume` |

Profile names are `h0-sphere-b64-1k-v4-fisher-tiny` and `h0-magnitude-b64-1k-v4-fisher-tiny`,
schema `h-run-profile-v3`, solver `max_iterations` 64, `seed` 42, `schedule.generation_continuations`
4, `schedule.importance_samples` `[2, 4]` with `diagnostic_samples` 8 (so the evaluated k-set is
2, 4, 8, which `wave.json` carries as `evaluated_samples`) (MEASURED, `<arm>/profile.json`,
`wave.json`). The carrier name is NOT in any metrics or worker-log row; it is carried by
`profile.json` (`law.mode`, `spine.carrier`, `name`), by the controller records and by the
check-run plan.

## K. The check suite: in-image run and local re-execution

`scripts/h0_final_probe_checks.py` emits one PASS / FAIL / UNPROVEN verdict per H0 exit
(criteria ruled 2026-10-22, module docstring). The in-image run is the capture
`h0-final-probe-checks-1755b0`; this session re-executed every regime-independent check on this
Mac from plans it wrote itself, over the same captured files.

| check | what a PASS proves (criterion, from the module docstring) | in-image verdict | in-image facts | local re-execution | evidence |
| --- | --- | --- | --- | --- | --- |
| K1 chain | from B's completion receipt every `parent_receipt` hop admits through `read_training_progress`; `successful_optimizer_steps` strictly decreases; the declared stopped receipt (path AND sha256) is in the chain and is `paused`; the final receipt is `training_complete` at `warmup_steps + steps` | PASS | `chain_optimizer_steps [1000, 750, 500, 250]`, statuses `training_complete, checkpoint, paused, checkpoint`, 4 hops, `stopped_hop 2` | FAIL on this Mac (python 3.12.15, torch 2.13.0): `hop 0 … source capture reader identity is incompatible`. Expected: the walk is regime-bound (section O2) | RECORD `report.jsonl` L1; MEASURED local plan m3 |
| K1 request | the resumed request binds the collected `resume_artifact` through `resume`, same profile, fresh nonce | PASS | previous `c658b4f0…`, resumed `f679b1a6…`, resume `run/1e13435f…-prediction-00000250-progress.json` | PASS | RECORD L2; MEASURED plan m2 |
| K1 seam | the stopped pod and its continuation admit as ONE run on a continuous global step axis | PASS | segments `[[1, 500], [501, 1000]]`, `warmup_steps 250`, one `profile_identity` | PASS | RECORD L3; MEASURED plan m2 |
| K2 | one `terminal_collected` controller record per declared arm, workload and termination verified, empty `error_types`, native completion receipt verified, distinct carriers | PASS over {W1 sphere, W2 B} | `arms 2, carriers [sphere, magnitude]` | PASS over {W1 sphere, W1 magnitude} (plan m1) and over {W1 sphere, W2 B} (plan m2) | RECORD L4; MEASURED |
| K3 | every captured metrics file passes `final_h_run_end` and its `request_sha256` is a declared request, exactly once | PASS over three files (W1 sphere, W2 A, W2 B) | A `result_status paused` `status ok`; B and sphere `probe_workload_complete` | PASS over all four files, W1 magnitude included (plan m1) | RECORD L5; MEASURED |
| K4 (D5) | every declared admission site has exactly one `admission.end` in the closed stage vocabulary; every `memo=shared, root=run` row reads 0 evaluations from disk; fresh walks reported | PASS over three logs | `shared_disk_reads []` on every log; fresh walks `completion` and `native_verification` (A: `completion` only) | PASS over all four logs (plan m1) | RECORD L6; MEASURED |
| K5 (D1) | boundary arm: each rollout's terminal outcome is consistent with its observations (256 ends a sentence, 257 the document); stochasticity PROVEN only when a recorded sampling law carries `continuations > 1` | PASS over two receipts (W1 sphere, W2 B) | each: `boundary PASS`, `continuations 4`, `outcome invalid_utf8`, `sampling_law_recorded true` | PASS over three receipts, W1 magnitude included (plan m1) | RECORD L7; MEASURED |
| K6 (D2) | every evaluation example's `by_sample_count` key set EQUALS the profile's evaluated k-set; every finite ESS in `[1, k]` | PASS, `evaluated_samples [2, 4, 8]`, 6 rows | | PASS, 9 rows over four files (plan m1) | RECORD L8; MEASURED |
| K7 | every arm admits through the analysis surface (single pod or resumed lineage), reports and aggregates with the other arms | PASS over {magnitude lineage A+B, W1 sphere}: `paired_training_steps 1000`, `paired_evaluation_steps [500, 750, 1000]` | | PASS over {sphere, magnitude uninterrupted} (m1) and {sphere, magnitude resumed lineage} (m2), same paired steps | RECORD L9; MEASURED |

Notes on scope. The in-image plan (`plan.json`) names W1 sphere, W2 A and W2 B; the
uninterrupted W1 magnitude pod (`14637e75`) is NOT in that plan (the plan filler took
`--single-arms sphere`, `Containerfile.checks` line 14), so its K2 to K7 coverage above is the
local re-execution only. K2 and K7 require distinct carriers and distinct profile identities
across completed arms, which is why the local plans pair each magnitude arm with sphere
separately rather than running all three completed arms together. The in-image run's own
record: `exit=0` (`exit.txt`), regime `3.12.3 2.13.0+cu126 0.3.4` (`regime.txt`), image index
`sha256:02e2582f…`, source commit `046cfb05…`, `report_sha256 2e104321…`, venue
`docker buildx cloud-luizffrias-small-lang-model linux/amd64` (`metadata.json`, RECORD).
`metadata.json` also records two host runs: Mac 3.12.15/2.13.0 K1 chain FAIL (source identity)
and Mac 3.12.3/2.13.0 K1 chain FAIL (checkpoint identity, cu126), all other checks PASS in both
(RECORD); this session's plan m3 reproduces the first. Added 2026-10-26 (blind review round 1):
the captured `report.jsonl` hashes to sha256 `2e10432125a1ff3a7cd52941467e2dac5e50eaefb59fbf989e79e7712db0289a`,
equal to `metadata.json` `report_sha256` byte for byte (MEASURED, `sha256sum`). The reviewer's
packet copy could not match because it was carrier-redacted for blinding (K2 `carriers` and the
K7 arm label carry the carrier names); see the review response.

The test-only gate against the HEpP7g capture (an open exit of the previous verdict) was also
run locally (plan m4, the two `9d353fcc` files in the same MLflow capture): K3 PASS, K6 PASS
(`evaluated_samples [2, 4]`), K7 PASS, K4 FAIL on both logs (`shared-memo stage
training_readback read 3 evaluations from disk (root=run; the floor is 0)`, likewise
`generation` 3, `span_diagnostics` 6, `trace_capture` 9, `result_admission` 3). That is the
pre-D5 worker reading evaluations on every shared-memo stage, which the final probe's worker
no longer does (`shared_disk_reads []` on all four new logs). MEASURED.

## Q1. Admissibility and row census

The H lane predicate (`final_h_run_end`): well-formed JSON-object lines, exactly one
`run_start`, exactly one `run_end` of schema `h-run-lifecycle-v1`, at least one committed step
row; `run_end.status` is read, not required. Executed in this session over all six files.

| check | W1 sphere (`38c25d26`) | W1 magnitude (`14637e75`) | W2 A (`f25744b7`) | W2 B (`5e76f21c`) | evidence |
| --- | --- | --- | --- | --- | --- |
| `final_h_run_end` | admissible; `status ok`, `result_status probe_workload_complete`, `successful_optimizer_steps 1000` | same | admissible; `status ok`, `result_status paused`, `successful_optimizer_steps 500` | admissible; `status ok`, `probe_workload_complete`, 1000 | MEASURED, executed. The predicate accepts `NaN` and `Infinity` tokens because Python's JSON parser does; finiteness is Q6's separate scan (noted by the blind review) |
| Every line parses as a JSON object | 1031 of 1031; 278 of 278 | 1031; 278 | 515; 99 | 528; 256 | MEASURED |
| `stage.end` rows (`h-worker-stage-v1`), now projected into the metrics file | 8 (L1, L2, then after training) | 8 | 4 (materialization, runtime_admission, training_readback, result_admission) | 8 | MEASURED |
| `run_start` / `run_end` lines | L3 / L1031 | L3 / L1031 | L3 / L515 | L3 / L528 | MEASURED |
| `h-training-step-v2` rows, all `status committed` | 1000 | 1000 | 500 | 500 | MEASURED |
| Phase-local step axis as written | `codec_warmup` 1..250 (L4 to L253), `prediction` 1..750 (L255 to L1008) | same | `codec_warmup` 1..250, `prediction` 1..250 (L255 to L504) | `prediction` 251..750 only (L4 to L505), no warmup rows, no discontinuity | MEASURED |
| Global axis | carried by the worker log `checkpoint.seal` / `evaluation.start` `global_step` (250, 500, 750, 1000) and by K1 seam's composition `[1, 500] + [501, 1000]` | same | 250, 500 | 750, 1000 | MEASURED (log events), RECORD (K1 seam facts) |
| `h-predictive-evaluation-v1` rows, metrics `step` (global) | 3: 250 (500), 500 (750), 750 (1000) at L505, L757, L1009 | same lines | 1: 250 (500) at L505 | 2: 500 (750) at L254, 750 (1000) at L506 | MEASURED |
| `admission.end` rows in the metrics file | 7 | 7 | 3 | 7 | MEASURED |
| `annotate` rows (no schema key) | 11 | 11 | 5 | 9 | MEASURED |
| Row class total | 8 + 1 + 1000 + 3 + 7 + 11 + 1 = 1031 | same | 4 + 1 + 500 + 1 + 3 + 5 + 1 = 515 | 8 + 1 + 500 + 2 + 7 + 9 + 1 = 528 | DERIVED_FROM_MEASURED |
| Worker-log levels | 278 INFO, 0 WARNING, 0 ERROR | 278 INFO | 99 INFO | 256 INFO | MEASURED |
| Worker-log timestamps non-decreasing | yes | yes | yes | yes | MEASURED |

Difference from HEpP7g: the metrics file now also carries `stage.end` and `admission.end`
rows (schema `h-worker-stage-v1`) and `admission/<stage>/memo` annotations, so the final-leg
admission accounting is readable from the metrics file as well as the log.

## Q2. Lineage

| field | W1 sphere | W1 magnitude | W2 A | W2 B | evidence |
| --- | --- | --- | --- | --- | --- |
| `code_commit` (run_start, run_start.runtime, worker.start) | `046cfb05ab4b4b835e4819b9d9be71da9e698a00` | same | same | same | MEASURED, metrics L3, log L1 |
| `image` | `ghcr.io/luiz-frias/entropic-substrate-lm@sha256:88e21c59b17b3b3656671c4bb5aba92a3e921b1d5d93249efbd6b03f8acfc537` | same | same | same | MEASURED, same rows; equals the pin in `docs/deploy-h100.md` and `wave.json` `image` |
| `runtime.source_attestation` / `kind` / `accelerator` | `exact_git_tree_match` / `remote_tiny` / `{count 1, model NVIDIA A100-SXM4-80GB}` | same | same | same | MEASURED, metrics L3 |
| `runtime.precision` | `declared_precision bf16`, `autocast {device_type cuda, dtype torch.bfloat16, enabled true}`, `default_dtype torch.float32`, `matmul_precision highest`, `effective_tf32 false`, overrides null | identical | identical | identical | MEASURED, metrics L3; also `host/capture/worker/start.json` and `collection.json` per arm |
| `profile_identity` = `run_identity` on every step and evaluation row | `3390f577…` (1003 of 1003) | `5ece5d1b…` (1003 of 1003) | `5ece5d1b…` (501 of 501) | `5ece5d1b…` (502 of 502) | MEASURED |
| `profile.json` (host copy, `<arm>/profile.json`) | differs from magnitude only in `name`, `spine.carrier`, `law.mode` (48 flattened keys compared) | byte-identical across W1 magnitude, W2 A, W2 B (sha256 `d42ff706…`) | same bytes | same bytes | MEASURED, flattened key diff; sha256 |
| Captured `host/capture/run/profile.json` vs host copy | equal as JSON content; captured bytes sha256 `70956009…` | equal; captured sha256 `f4eca963…`, which is the `profile.sha256` every magnitude progress receipt names | same | same | MEASURED |
| `source_batch` of training row 1 | `b60570f4…` | same | same | B row 1 is prediction 251: `f0a7c4be…` | MEASURED |
| Ordered per-row `source_batch` over all 1000 training rows, sphere vs magnitude | 0 mismatches, 1000 distinct digests per arm | | | | MEASURED |
| Sums of `source_bytes` / `encoder_bytes` / `history_bytes` over training rows | 402,212 / 1,534,453 / 1,132,241 | identical | 202,417 / 583,983 / 381,566 (500 rows) | 199,795 / 950,470 / 750,675 (500 rows); A + B equals W1 on each field | DERIVED_FROM_MEASURED |
| `mlflow.run` id (log L2) | `38c25d265cf847f185896a3c49eb0c16` | `14637e757a79410cb42fb1a6d2f70fbf` | `f25744b722d94585a70693353489817d` | `5e76f21c84b941d4be44d1b19f9e38bc` | MEASURED |
| R2 prefix (`wave.json` `prefix`, equals publish keys) | `h-probes/v1/ba48fd75…` | `…/0b451dda…` | `…/10d1a612…` | `…/13b92be2…` | MEASURED |
| `request.json` `nonce` | `ba48fd75…` | `0b451dda…` | `10d1a612…` | `13b92be2…` (fresh; differs from A) | MEASURED |
| `request.json` `inputs` | corpus `f1c8db69…` + `code.bundle` `a46ed3ed…` | same two | same two | the same two plus 10 `run/` files: A's four `codec_warmup-00000250` artifacts, four `prediction-00000250` artifacts, the 250 evaluation and `run/profile.json` (`wave.json` `inputs 12`) | MEASURED |
| `wave.json` globals | `code_commit 046cfb05…`, `solver_budget 64`, `budget_ladder [64, 72, 128]`, `continuations_requested 4`, `gpu NVIDIA A100-SXM4-80GB`, `data_center US-MD-1`, `template_id w1rl2rxaqw`, `scientific_authority none`, `spending_authority none`, `launch_schema h0-probe-launch-v3`, `schema h0-final-probe-wave-v1` | identical in both wave files except `resume_plan.stop_mode` (`operator` in wave 1, `declared` in wave 2) | | | MEASURED. Added 2026-10-26 (round 2 asked why wave 1's operator stop did not occur): wave 1 planned an operator-mode stop of the magnitude arm at global 500, and that arm ran to 1000 with no stop marker observed (`stop_marker.observed 0`, Q6); the worker polls the stop marker only at seal boundaries, so the stop had to be published between the 250 and 500 seals (16:24:21Z to 16:28:36Z on that pod, MEASURED from its `checkpoint.seal` rows), and the conductor missed that window, then re-ran the pair as wave 2 with `--stop-mode declared`. The missed window is CONDUCTOR-REPORTED (the conductor's own record of 2026-10-26); the captures show only its consequence |

Lineage conclusion: all four pods ran the same code commit, image digest, attestation,
accelerator class and precision record, over an identical source stream (ordered per-row match
sphere vs magnitude; A + B row-for-row equal to W1 magnitude, Q4). The three magnitude pods
share one profile byte-for-byte; sphere differs only in the carrier setting and its name. The
pods differ in request digest, nonce, run id, pod and R2 prefix, as one request per pod predicts.

## Q3. Final leg: stages, admissions and clocks

All eight `WORKER_STAGES` appear as a `stage.start`/`stage.end` pair on the three completed
pods; A (paused) ran four: `materialization`, `runtime_admission`, `training_readback`,
`result_admission`. The `completion` caller has an admission pair but no stage bracket and sits
inside the prediction phase window on every pod (`admission.end caller=completion` precedes
`phase.end prediction`; log L104 then L128 on both W1 pods, L82 then L106 on B, L52 then L76 on A).

Stage seconds (`stage.end.seconds`) and admission seconds (`admission.end.seconds`), rounded to
3 decimals from the raw floats:

| stage (caller) | memo | root | seals / evals / journals | W1 sphere adm / stage | W1 magnitude adm / stage | W2 A adm / stage | W2 B adm / stage | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| materialization | n/a | n/a | n/a | n/a / 0.349 | n/a / 1.090 | n/a / 0.552 | n/a / 2.142 | MEASURED, log L3-L4 |
| runtime_admission | n/a | n/a | n/a | n/a / 2.422 | n/a / 2.463 | n/a / 2.461 | n/a / 2.523 | MEASURED, log L5-L6 |
| completion (no bracket) | fresh | run | 4 / 3 / 4 (A: 2 / 1 / 2) | 275.446 / n/a | 276.380 / n/a | 122.082 / n/a | 276.738 / n/a | MEASURED, `admission.end caller=completion` |
| training_readback | shared | run | 0 / 0 / 0 | 44.053 / 44.114 | 44.219 / 44.272 | 22.268 / 22.316 | 44.509 / 44.557 | MEASURED |
| generation | shared | run | 0 / 0 / 0 | 55.035 / 102.083 | 55.490 / 101.567 | not run | 55.703 / 101.786 | MEASURED |
| span_diagnostics | shared | run | 0 / 0 / 0 | 121.040 / 338.527 | 121.549 / 310.566 | not run | 121.678 / 309.643 | MEASURED |
| trace_capture | shared | run | 0 / 0 / 0 | 153.787 / 210.522 | 154.636 / 211.676 | not run | 154.941 / 212.228 | MEASURED |
| native_verification | fresh | capture | 4 / 3 / 4 | 384.898 / 442.202 | 387.162 / 444.725 | not run | 387.820 / 445.157 | MEASURED |
| result_admission | shared | run | 0 / 0 / 0 | 43.816 / 43.971 | 44.082 / 44.471 | 22.240 / 22.327 | 44.327 / 44.483 | MEASURED |

The shared-memo rows read 0 evaluations and 0 journals on all four pods (the D5 read floor;
HEpP7g read 3 / 3 / 6 / 9 / 3 on the same stages). The `native_verification` fresh walk reads
3 evaluations on this probe (9 on HEpP7g).

Added 2026-10-26 (blind review round 1, finding re-measured here): on the callers
`generation`, `span_diagnostics`, `trace_capture` (shared memo) and `native_verification`
(fresh memo, capture root), the `admission.end` row's `seconds` field is smaller than the
`admission.start` to `admission.end` timestamp bracket (W1 sphere: 55.035 vs 102.043, 121.040 vs
338.435, 153.787 vs 210.438, 384.898 vs 442.079; the same shape on W1 magnitude and W2 B). The
bracket is within 0.040 to 0.123 s of the stage seconds, so the `admission.end` event is emitted
at stage end while `seconds` is the admission's own clock. The `seconds` values in the table are
the correct field reads; a reader that brackets admission timing by timestamps would overstate
it. A finding for the gate author, MEASURED. RETRACTION 2026-10-26 (round 2): the first version
of this paragraph called `native_verification` a shared-memo caller, and said the bracket
"equals" the stage seconds when it differs by 0.040 to 0.123 s.

Clocks per pod:

| quantity | W1 sphere | W1 magnitude | W2 A | W2 B | evidence |
| --- | --- | --- | --- | --- | --- |
| `worker.start` (log L1) | 16:22:22.534Z | 16:22:58.378Z | 16:52:35.927Z | 17:05:26.309Z | MEASURED (2026-10-26) |
| `phase.end codec_warmup` seconds | 75.810 | 77.645 | 75.912 | no warmup phase | MEASURED |
| `phase.end prediction` seconds (includes the `completion` admission) | 1221.787 | 1089.939 | 389.940 | 912.107 | MEASURED |
| prediction phase minus `completion` admission | 946.341 | 813.558 | 267.859 | 635.368 | DERIVED_FROM_MEASURED |
| sum of per-row `wall_seconds`, warmup / prediction | 5.281 / 573.150 | 5.472 / 436.163 | 5.495 / 159.494 | 0 / 282.104 | DERIVED_FROM_MEASURED |
| per-row `wall_seconds` per committed prediction step (sum over prediction rows / rows) | 0.764 (750 rows) | 0.582 (750) | 0.638 (250) | 0.564 (500) | DERIVED_FROM_MEASURED |
| `checkpoint.seal` seconds at global 250 / 500 / 750 / 1000 | 22.153 / 33.255 / 33.276 / 33.265 | 22.111 / 33.487 / 33.419 / 33.412 | 22.140 / 33.618 / - / - | - / - / 99.887 / 33.552 | MEASURED. B's first seal (global 750) took 99.887 s against 33.3 to 33.6 s elsewhere; no captured row itemises it |
| `publish.end … started.json` (log L17) to `phase.start` (L18), the window in which a resumed worker restores its inputs | 34.288 | 34.441 | 34.565 | 378.134 | DERIVED_FROM_MEASURED (timestamp difference). Added 2026-10-26 (blind review round 1): the resumed pod spends 343.7 s more in this window than the other three, and no log event names the restore; together with the 99.887 s first seal, about 444 s of resume-related time has no itemised row. Resume works; its cost is not observable from the log |
| `evaluation.end` seconds at global 500 / 750 / 1000 | 66.618 / 66.408 / 66.428 | 66.710 / 66.738 / 66.717 | 67.012 / - / - | - / 66.837 / 66.881 | MEASURED |
| six bracketed final-leg stages, sum | 1181.421 | 1157.277 | 44.643 (two stages) | 1157.855 | DERIVED_FROM_MEASURED |
| `run_end.wall_seconds` | 2494.925 | 2340.940 | 548.590 | 2418.778 | MEASURED, last metrics line; equals `worker/result.json` `wall_seconds` |
| `worker.start` to last log row (`sink.finish.start sink=mlflow`) | 2565.572 | 2418.925 | 577.542 | 2492.898 | DERIVED_FROM_MEASURED |
| difference of the two clocks | 70.646 | 77.986 | 28.952 | 74.120 | DERIVED_FROM_MEASURED; `result_admission` (43.971 / 44.471 / 22.327 / 44.483) accounts for most of it, the remainder is not itemised |
| watchdog `observed_at_utc` ready / terminated | 16:20:27Z / 17:05:41Z | 16:21:35Z / 17:03:42Z | 16:50:36Z / 17:02:38Z | 17:05:12Z / 17:47:32Z | RECORD, `watchdog-ready.json`, `watchdog-result.json` |
| watchdog `elapsed_seconds`, `status`, `reason` | 2714.516, `terminated`, `capture_verified` | 2526.343, `terminated`, `capture_verified` | 721.907, `terminated`, `partial_capture` | 2540.260, `terminated`, `capture_verified` | RECORD |
| `termination_verified` / `attributed_pod_absent` / `unattributed_pod_ids` / `provider_error_count` | true / true / [] / 0 | same | same | same | RECORD |
| `cost_per_hour` (controller pod observation) | 1.79 | 1.79 | 1.79 | 1.79 | RECORD, `controller-pod.json` |
| elapsed × cost_per_hour | 1.350 | 1.256 | 0.359 | 1.263 | arithmetic over RECORD values; sum 4.228 over 8503.025 s. RETRACTION 2026-10-26 (blind review round 1): the first version labelled this row DERIVED_FROM_MEASURED; it is an ESTIMATE of the billed cost (watchdog elapsed time times the observed hourly rate), not a measured bill, and no bill is in any capture. The brief's "$4.22" is CONDUCTOR-REPORTED |
| `terminal.json` publication: objects put / skipped / retries / verify reads | 39 / 21 / 0 / 39 | 39 / 21 / 0 / 39 | 2 / 11 / 0 / 2 | 39 / 21 / 0 / 39 | MEASURED |
| `publish.start` / `.end` / `.verify_read` / `.retry` in the log | 72 / 72 / 72 / 0 | 72 / 72 / 72 / 0 | 21 / 21 / 21 / 0 | 68 / 68 / 68 / 0 | MEASURED |

The wave ran the two W1 pods concurrently (worker starts 36 s apart), then A, then B after A's
terminal collection (A terminated 17:02:38Z, B ready 17:05:12Z). B's runtime was in the same
band as the uninterrupted pods because B re-ran the full final leg over all four seals.

## Q4. The resume seam

The pause and resume mechanism, as the captures show it:

| item | value | evidence |
| --- | --- | --- |
| A's declared pause | `request.json` `pause ["prediction", 250]`; `wave.json` `resume_plan {arm magnitude, seal_boundary [prediction, 250], stop_global_step 500, stop_mode declared}` | MEASURED |
| A's terminal state | `run_end` `result_status paused`, `successful_optimizer_steps 500` (L515); `worker/result.json` `status paused`, `training` = `run/1e13435f…-prediction-00000250-progress.json` sha256 `5f4a3272…`, `generation null`, `capture null`, `verification null`; `terminal.json` `status paused`, snapshot `snapshots/paused.json` 3,922 bytes, 12 files, 3,380,599 artifact bytes | MEASURED |
| A's sealed progress receipt | `h-probe-training-progress-v1`, `status paused`, `phase prediction`, `step 250`, `successful_optimizer_steps 500`, `last_seal` `…-prediction-00000250-seal.json` sha256 `2d91e9fb…`, `parent_receipt` `…-codec_warmup-00000250-progress.json` sha256 `53d13588…`, `profile.sha256 f4eca963…`, inventory `total_deployed 112,233` (decoder 46,720, encoder 38,496, future_law 1,089, spine 25,928), `optimizer_active 73,737` | MEASURED, `magnitude/host/capture/run/…-prediction-00000250-progress.json`, 1,034 bytes |
| Host collection of A | `collection.json` `capture_status pause_bytes_verified`, `native_validation not_replayed_by_host`, `reported_outcome paused`, `reported_optimizer_steps 500`, `resume_artifact` = the receipt above (same path and sha256), `workload_verified false`; controller `status terminal_collected`, `error_types []`, stop reason `partial_capture` | MEASURED |
| B's request | `resume {path run/1e13435f…-prediction-00000250-progress.json, sha256 5f4a3272…}`, `pause null`, fresh nonce `13b92be2…`, `previous_request_sha256 c658b4f0…` (wave.json), `inputs 12`; the copy under `magnitude-resume/inputs/run/` is byte-identical to A's captured receipt (sha256 equal) | MEASURED |
| B's chain | `…-prediction-00000500-progress.json` (`status checkpoint`, `successful_optimizer_steps 750`, `parent_receipt` sha256 `5f4a3272…` = A's paused receipt) then `…-prediction-00000750-progress.json` (`status training_complete`, 1000, parent = the 500 receipt sha256 `e875ac3e…`); `evaluations` lists A's 250 evaluation (sha256 `8b991646…`) then B's 500 and 750 | MEASURED |
| B's terminal state | `run_end` `probe_workload_complete`, `successful_optimizer_steps 1000`; `worker/result.json` `status probe_workload_complete`, `training` = the 750 receipt sha256 `7dc13cd0…`; controller `terminal_collected`, `workload_verified true`, stop reason `capture_verified` | MEASURED |
| Resume event in B's log | none named `resume` or `restore`; B's log opens with `worker.start`, `mlflow.run`, materialization, runtime admission, then `phase.start prediction` at L18 with no `codec_warmup` phase, first `checkpoint.seal` at `global_step 750` (L19) | MEASURED (event names) |

Continuity across the seam, both sides measured from the captured metrics files. The stitched
A then B training rows (500 + 500) were compared row i against row i with W1 magnitude's 1000
rows, which ran uninterrupted on a different pod from the same profile and seed:

| comparison | result | evidence |
| --- | --- | --- |
| `phase`, `step`, `source_batch`, `source_bytes`, `encoder_bytes`, `history_bytes` | 0 mismatches on 1000 rows | MEASURED |
| `decoder_ce` (1000 rows), `future_nll_per_dimension` (750 prediction rows), `aggregate_norm` (1000 rows) | maximum absolute difference 0.0 on every field (bit-identical floats) | MEASURED |
| every flattened field of every training row (lists as single leaves: 31 dotted keys on warm-up rows, 54 or 56 on prediction rows, 57 in the union), row i against row i | every field identical on all 1000 rows except 3, including `objective`, all `gradients.*`, the whole `mechanism.solver.*` block and `mechanism.*.range`; the 3 that differ: `wall_seconds` and `cuda_seconds` (every row) and `cuda_peak_bytes` (136 rows, stitched rows 501 to 636, the resumed pod's first 136 steps, where a fresh process reports a lower allocator high-water mark) | MEASURED 2026-10-26 (added after blind review round 1; the first version compared only the nine fields above). RETRACTION 2026-10-26 (round 2): "31 dotted keys per row" held only for the 250 warm-up rows |
| checkpoint files at the same global step, resumed lineage vs W1 magnitude (`codec_warmup-250`, `prediction-250` written by the stopped pod; `prediction-500`, `prediction-750` written by the resumed pod) | zip archives with 157 / 172 entries; every tensor data entry byte-identical (155 of 157, 170 of 172); the two entries that differ on every pair are torch's per-save `.data/serialization_id` and `data.pkl`; all 166 tensor storage entries under `data/` are byte-identical on every pair, 1,044,048 storage bytes each side; on the two checkpoints the resumed process wrote, `data.pkl` is 120 bytes larger (21,330 vs 21,210) and the file 128 bytes larger (1,097,203 vs 1,097,075); the stopped pod's two match W1's sizes exactly. What the 120 extra pickle bytes encode was not inspected | MEASURED 2026-10-26 (added after blind review round 2, which found the size difference in the snapshot manifests). The tensor payload is identical and every downstream logged quantity through step 1000 and every continuation is identical, so the difference is in the pickle's metadata, not in the parameters or optimizer state; what the extra bytes encode stays a finding for the gate author |
| A's last three rows / B's first three rows (`decoder_ce`, `future_nll_per_dimension`, `aggregate_norm`) | prediction 248: 4.588965 / 2.466202 / 3.417502; 249: 2.705522 / -0.409267 / 0.940906; 250: 2.798786 / 0.049914 / 1.038090; then 251: 2.662143 / -0.218496 / 1.012599; 252: 2.850077 / 0.187250 / 1.359469; 253: 2.713864 / -0.328660 / 0.842984. W1 magnitude's rows 498 to 503 carry exactly these values | MEASURED |
| Evaluation at global 500 (A, metrics `step` 250) vs W1 magnitude's | `log_estimate`, `effective_sample_size` and `decoder_log_prob` identical on all 12 entries and 8 × 4 decoder values | MEASURED |
| Evaluation at global 750 and 1000 (B) vs W1 magnitude's | identical on all 24 entries | MEASURED |
| Generation receipt of B vs W1 magnitude's | all four continuations have identical `log_scores`, `sentences`, `outcome`, `seed` and `sampling_law`, and `summary` is identical; the receipts differ in the per-continuation and top-level `snapshot.{path, sha256}`, in `training.{path, sha256}` (each names its own pod's final progress receipt) and in `wall_seconds` / `cuda_seconds` of the generation stage. RETRACTION 2026-10-26 (blind review round 1): the first version said "differ only in `snapshot` paths and sha256" | MEASURED |
| Checkpoint seal sha256 at the same global step, B vs W1 magnitude | differ (`43fa0d70…` vs `0cf8f0fd…` at 750) | MEASURED. The seal names run-local artifact paths, so this does not contradict the row-level identity; the checkpoint bytes themselves were not compared here |

Nothing discontinuous appears at the seam. The stronger statement the captures support: the
paused-then-resumed lineage reproduced the uninterrupted run's logged model quantities,
evaluations and generated continuations bit-for-bit, across three different pods of the same
accelerator class and image; only the timing fields and the allocator high-water mark differ.
RETRACTION 2026-10-26 (blind review round 1): the first version said "every logged metric",
which `wall_seconds`, `cuda_seconds` and 136 `cuda_peak_bytes` rows contradict. That is a
determinism observation about this image on this hardware class; it is not a claim about any
other regime. The visible differences are operational: B's first checkpoint seal took 99.887 s
and its restore window is 343.7 s longer than a fresh start (Q3).

## Q5. Generation receipts: N = 4, the sampling law, and `invalid_utf8`

All three completed pods published a `h-generation-receipt-v3` receipt
(`host/capture/run/<id>-generation.json`, sha256 named in `worker/result.json` `generation`).

| field | W1 sphere | W1 magnitude | W2 B | evidence |
| --- | --- | --- | --- | --- |
| `status` | `generation_complete` | same | same | MEASURED |
| `sampling` | `{continuations 4, sampler multinomial_temperature_1, temperature 1.0, seed 42, rng_digest dcc6348d…}` | identical | identical | MEASURED. The `rng_digest` is the digest over the per-continuation `rng_digest` strings, which depend on `seed` and `schedule.steps` only (`generation.py` `continuation_seed`, `sampling_digest`), so identical across arms is expected |
| per-continuation `seed` (index 0 to 3) | 4112311562, 4112311563, 4112311564, 4112311565 | same | same | MEASURED; equals `(42 × 0x9E3779B1 + ((750 << 8) \| index)) mod 2^32` by the documented formula (code-read) |
| per-continuation `sampling_law` | `{latent rsample, sampler multinomial_temperature_1, seed, rng_digest}` with a distinct `rng_digest` per index (`6d6a7baf…`, `2a71d63d…`, `b92d3f6f…`, `20e3c304…`) | same four digests | same four | MEASURED |
| `boundary` per continuation | 256 | 256 | 256 | MEASURED |
| `outcome` per continuation (index 0..3) | `invalid_utf8` × 4 | `invalid_utf8`, `sentence_budget`, `invalid_utf8`, `invalid_utf8` | identical to W1 magnitude | MEASURED |
| `sentences` per continuation (count) | 1, 0, 0, 0 | 0, 2, 0, 0 | 0, 2, 0, 0 | MEASURED |
| `log_scores` per continuation (count = bytes and controls sampled) | 184, 112, 280, 296 | 96, 106, 225, 61 | same | MEASURED |
| distinct continuations within the receipt (by `log_scores`) | 4 of 4 | 4 of 4 | 4 of 4 | MEASURED |
| `summary.outcome` / `summary.pending` length / `summary.observations` length | `invalid_utf8` / 295 bytes / 872 | `invalid_utf8` / 60 bytes / 488 | same as W1 magnitude | MEASURED. `summary` is the aggregate in index order: `outcome` is the LAST continuation's outcome, `pending` the last continuation's undecodable bytes (`generation.py` `_summary` docstring) |
| `summary.prompt` | `["- Yes, this is a good time to plant native grass seed in the ground. "]` | same | same | MEASURED |
| `wall_seconds` / `cuda_seconds` of the generation stage proper | 3.123 / 3.112 | 2.044 / 2.036 | 2.058 / 2.049 | MEASURED |

What `invalid_utf8` means, from `core/sentence/rollout.py` (code-read, lines 95 to 116 at main
`0f0f36c4`): the decoder samples one byte or control per step from `log_softmax` with
`torch.multinomial`, no probability floor, no top-k or top-p. Bytes below 256 accumulate in
`pending`. When the sentence-end control 256 is sampled, the pending bytes are decoded as
strict UTF-8; a `UnicodeDecodeError` sets the rollout status to `invalid_utf8` and the
continuation stops there. The receipt's `pending` on the magnitude arm (60 bytes) reads
`[45, 44, 32, 105, …, 226, 128, 57, …]`: the pair 226, 128 opens a three-byte UTF-8 sequence
and 57 is not a continuation byte, which the strict decoder refuses. The sphere `pending`
(295 bytes) carries the same 226, 128, 57 triple and, later, 244 then 67, where 244 opens a
four-byte sequence that 67 does not continue.

Interpretation, bounded to what the files show:

- The outcome is in the rollout's declared vocabulary (`active`, `document_end`,
  `empty_sentence`, `invalid_utf8`, `length_limit`, `context_limit`, plus the receipt-level
  `sentence_budget` and `paused`), the receipt records it per continuation, and K5's boundary
  arm PASSes on every receipt: the terminal outcome is consistent with the observations. The
  tracer handled it as a terminal state, not as a failure; `run_end.status` is `ok` and no
  WARNING or ERROR row exists.
- For a 112,233-parameter model after 250 warm-up and 750 prediction steps, byte-level sampling
  at temperature 1.0 with no UTF-8 grammar constraint will emit multi-byte prefixes that the
  strict decoder refuses. The 8 continuations across the two carriers produced 3 finished
  sentences in total (sphere continuation 0, magnitude continuation 1 twice over), none of
  them meaningful text (sphere: `- Uneo f Mutove hmw t otpe e in thered be ale phabon.7`;
  magnitude: `Lagean ala ouctrhermits …`, `C-) das, ry, wa yxpFe Dis.`). That is scientific
  content about a tiny model, reported as captured; no quality claim attaches and none is
  implied.
- Two findings for the gate and H1 authors, not refusals: (1) a continuation terminates at its
  FIRST invalid sentence, so continuation length and sentence count are censored by UTF-8
  validity, which any H1 estimand over generated text must state; (2) `summary.outcome` is the
  last continuation's outcome, not an aggregate over all N, so a reader that keys on `summary`
  alone sees one outcome of four.

The stochastic-rollout exit: the sampling law is recorded (sampler, temperature, seed and an
RNG-state digest per continuation), N = 4 continuations per prompt were produced, and the four
differ. Closed by this evidence; K5 reports `sampling_law_recorded true`, `continuations 4`.

## Q6. Numerics

| check | W1 sphere | W1 magnitude | W2 A | W2 B | evidence |
| --- | --- | --- | --- | --- | --- |
| numeric leaves scanned, metrics / log (every int or float in every row, bool excluded) | 102,314 / 527 | 104,628 / 527 | 40,075 / 179 | 64,572 / 483 | MEASURED |
| nonfinite values (NaN, +inf, -inf) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | MEASURED; 313,305 leaves over the four pods, 0 nonfinite (DERIVED_FROM_MEASURED sum) |
| training `step` strictly +1 within each phase | yes, one transition at row 250 | yes | yes | yes, no transition (prediction only) | MEASURED |
| `mechanism.solver` on prediction rows | `conjugate_gradient` 750 of 750; `max_residual_ratio` min 0.175505, max 0.999855; rows at or above 1.0: 0 | 750 of 750; 0.172860 to 0.999992; 0 | 250 of 250; 0.172860 to 0.999834; 0 | 500 of 500; 0.438662 to 0.999992; 0 | MEASURED |
| `decoder_ce` first / last training row | 5.951573 / 2.673644 | 5.952548 / 2.682614 | 5.952548 / 2.798786 | 2.662143 / 2.682614 | MEASURED (plumbing readback only) |
| `future_nll_per_dimension` first prediction row / last / minimum | 4.643691 / -0.540911 / -0.740573 | 5.355175 / 0.552849 / -0.708967 | 5.355175 / 0.049914 / -0.708967 | -0.218496 / 0.552849 / -0.623228 | MEASURED; identical to the HEpP7g values on both carriers, which is consistent with the same profile, seed and source stream at a different code commit |
| `aggregate_norm` first / last | 1.010331 / 1.007102 | 1.022729 / 1.146669 | 1.022729 / 1.038090 | 1.012599 / 1.146669 | MEASURED |
| `cuda_peak_bytes` maximum, element 0 (allocated) / element 1 (reserved) | 60,756,992 / 81,788,928 | 56,941,568 / 77,594,624 | 52,188,672 / 77,594,624 | 56,941,568 / 77,594,624 | MEASURED |
| prediction rows with `history_bytes` 0 (zero sentence states) | 18, at prediction steps 3, 39, 178, 208, 233, 278, 299, 308, 415, 440, 487, 496, 534, 565, 593, 636, 725, 748 | same 18 | 5 (3 to 233) | 13 (278 to 748) | MEASURED; A + B = the same 18 |
| `mechanism.sentences` on those rows | `{count 0, range null, status structurally_absent, reason no_sentence_states, zero_count 0, dtype torch.float32}` | same | same | same | MEASURED. The absence is now labelled in the row (HEpP7g carried only `range null`) |
| `mechanism.sentences.range` on rows with history | sphere norms within [0.99999994, 1.00000004] on the first such row | magnitude [0.980726, 2.007671] on the first such row | | | MEASURED (one row each, shown for the carrier tell only) |
| evaluation entries: `gap` null / finite ESS / ESS range | 36 / 36 / [1.0000, 1.9678] | 36 / 36 / [1.0000, 3.7683] | 12 / 12 / [1.0237, 3.7683] | 24 / 24 / [1.0000, 3.2848] | MEASURED; K6 bounds ESS by k = 8 |
| `run_end.status` / `FAILED` anywhere / WARNING or ERROR rows | ok / none / 0 | ok / none / 0 | ok / none / 0 | ok / none / 0 | MEASURED |
| `stop_marker.observed` / `sink.write_failed` / `sink.drain_exceeded` | 0 / 0 / 0 | same | same | same | MEASURED |
| `sink.finish.end sink=jsonl` | `settled true`, `stranded 0` | same | same | same | MEASURED |
| `sink.finish.end sink=mlflow` and `worker.end` | absent (structural: the log is uploaded at MLflow sink finish) | same | same | same | MEASURED |

Not `numerical_defect`: no nonfinite, nonmonotonic, failed or stranded evidence on any pod.

## O. Observability findings

O1. Controller progress view for resumed arms. The brief states that the controller's R2
progress view is blind for a resumed arm and is advisory only. No captured record carries a
progress-view field: `watchdog-process.log` is 0 bytes on all four arms, and no controller or
watchdog record names a progress key. What the captures do show is that the blindness did not
reach any decision: B's controller reached `terminal_collected` with `workload_verified true`,
`termination_verified true`, `reason capture_verified`, `provider_error_count 0`, and the
watchdog terminated B 33.074 s after the worker's last log row (17:47:32Z against 17:46:59Z).
The finding itself is CONDUCTOR-REPORTED and stays advisory.

O2. Regime-bound reader identities. K1 chain walks the completion chain through
`read_training_progress`, whose checkpoint and source reader identities are bound to the image
regime (python 3.12.3, torch 2.13.0+cu126, pysbd 0.3.4). `metadata.json` records two Mac runs
failing K1 chain at hop 0 (source identity on 3.12.15, checkpoint identity on 3.12.3), and this
session's plan m3 reproduces the first failure verbatim (`source capture reader identity is
incompatible`). The consequence for filing: K1 chain is PASS only as a RECORD from the in-image
run (`report.jsonl` L1, `exit=0`, `report_sha256 2e104321…`); it cannot be re-executed on an
operator machine, by design. The other eight checks are regime-independent and were
re-executed here.

## The HEpP7g open exits, closed or not

| exit (verdict-hepp7g-2026-10-22) | status now | basis |
| --- | --- | --- |
| Stochastic rollout | CLOSED | receipt v3 with sampler, temperature, seed, per-continuation seeds and RNG digests; N = 4 continuations, all four distinct (Q5); K5 PASS on three receipts |
| Recovery and resume | CLOSED | A paused at global 500 by a declared pause, B resumed from A's sealed receipt with a fresh nonce to a completion receipt (Q4); K1 chain (RECORD), K1 request and K1 seam PASS; logged metrics identical to the uninterrupted run |
| Test-only gate against this capture | CLOSED for the final-probe captures (K3, K4, K6, K7 PASS in-image and locally over all four pods); run against the HEpP7g capture it gives K3, K6, K7 PASS and K4 FAIL on the pre-D5 worker's disk reads (section K) | MEASURED |
| Autocast observability | CLOSED | `run_start.runtime.precision.autocast {enabled true, dtype torch.bfloat16, device_type cuda}` and `declared_precision bf16` on all four pods (Q2) |
| Declared measure of the future density per carrier | OPEN | a docstring question, not a capture question: `objective.py` still says "with its declared measure" without defining it; nothing in this wave bears on it |

## Verdict proposal

`tracer_complete` was this analysis's proposal before the review. Every check in the suite
PASSes in the image regime (RECORD) and every regime-independent check PASSes when re-executed
here over all four pods, including the uninterrupted magnitude pod the in-image plan did not
cover. The four HEpP7g exits that evidence could close are closed; the one that stays open is a
documentation question outside the H0 exit list. (Filed 2026-10-26 as `tracer_partial` in
`docs/h0/verdict-final-probe-2026-10-26.md`: the blind review raised two blocking concerns
about what its packet could verify, both resolved by evidence in the captures but not by the
reviewer; see the review response. Superseded the same day: blind review round 2, on a packet
carrying the host control records, the pre-redaction hashes, the warm-up receipt and the local
plans, returned `tracer_complete` with no blocking finding, and the verdict file's round-2
section files `tracer_complete`. Retractions dated "round 2" in this file are its corrections.)

Proven by the captured evidence:

- Both carriers ran 1000 committed steps to `probe_workload_complete` on one code commit, one
  image, one attestation, one accelerator class, one precision record, one source stream (Q1, Q2).
- The resume seam carries no discontinuity; the stitched lineage reproduces the uninterrupted
  run bit-for-bit on every model quantity, evaluation and continuation, with only timing and
  allocator fields differing (Q4; retraction dated 2026-10-26 there).
- N = 4 continuations with a recorded sampling law, differing from each other; terminal
  outcomes consistent with observations (Q5).
- Final leg: all eight stages on the three completed pods, seven admission pairs, shared-memo
  stages reading nothing from disk, zero retries, zero stop markers, JSONL sink settled (Q3).
- 0 nonfinite among 313,305 numeric leaves, monotone steps and timestamps, solver residual
  ratio below 1 on all 1,500 prediction rows of the two complete magnitude lineages and 750
  sphere rows (Q6).
- Capture, analysis (this file), check-suite gate: exercised. Review and filing follow.

Findings that must reach the gate and H1 authors (not refusals): the continuation-length
censoring by `invalid_utf8` and the `summary.outcome` semantics (Q5); the 99.887 s first seal
and the unlogged 378 s restore window on the resumed pod (Q3); the `admission.end` timestamp
semantics on shared-memo stages (Q3); K1 chain being image-only (O2); the controller progress
view for resumed arms (O1, advisory); the HEpP7g capture failing K4 under the current gate
(section K).

Not `numerical_defect`, not `missing_evidence`: the captures are complete for every exit they
were built to carry.

## What an independent reviewer must re-derive

1. Pull the four pointers and confirm the file counts and the arm-to-file binding by
   `run_start.request_sha256` (Inputs read).
2. Execute `final_h_run_end` on all six metrics files and re-count the row classes (Q1).
3. Re-check the lineage equalities field by field, including the ordered per-row `source_batch`
   match sphere vs magnitude and the byte identity of the three magnitude profiles (Q2).
4. Re-derive the stage and admission table and the clocks from `stage.end`, `admission.end`,
   `phase.end`, `checkpoint.seal` and `evaluation.end` rows (Q3).
5. Re-run the seam comparison: stitched A + B against W1 magnitude row by row, the three
   evaluations, and the two magnitude receipts (Q4). State whether any field differs.
6. Re-read the three receipts: `sampling`, per-continuation `seed`, `sampling_law`, `outcome`,
   `boundary`, `sentences`, `log_scores` lengths, and whether the four continuations differ (Q5).
7. Re-scan every numeric leaf for finiteness and `max_residual_ratio < 1` on every prediction
   row (Q6).
8. Re-execute the regime-independent checks (K1 request, K1 seam, K2 to K7) from plans you
   write yourself, and state that K1 chain can only be read from the in-image record.
9. Decide whether `invalid_utf8` on 7 of 8 carrier continuations is a tracer defect or expected
   content for this model size, from the receipt fields and the rollout semantics stated in Q5.
