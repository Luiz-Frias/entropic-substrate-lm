# H0 plumbing analysis: wave h0-plumbing-1k-v3.HEpP7g

Fresh-session analysis of the third paired 1K plumbing probe, under the brief
`.claude/workflows/briefs/h0-analysis-hepp7g.md`. This is a PLUMBING analysis with a
test-only gate: no scientific claim, no H1 eligibility beyond "the machinery exists", no
spend. Every table row carries its evidence class and the file plus row (`L<n>` is the
1-based line in the named file) or event that supports it. Nothing below was read from
MLflow, R2, a pod, or any session ledger.

## Inputs read

| item | value | evidence |
| --- | --- | --- |
| DVC pointer | `results/h0-plumbing-1k-v3-HEpP7g.dvc` at commit `1d3d67d1e481f871f4f58917b6dda0cadf93f05d`: md5 `2dc65e2bdf918d43858c6b2d92a63c58.dir`, 4 files, 5,597,931 bytes | MEASURED, `git show 1d3d67d1:results/h0-plumbing-1k-v3-HEpP7g.dvc` |
| Pull result | `dvc pull` fetched 4 files; `dvc status` reports "Data and pipelines are up to date"; byte sizes 2,669,045 + 125,658 + 2,677,562 + 125,666 = 5,597,931 | MEASURED (sizes), DERIVED_FROM_MEASURED (sum) |
| Sphere arm files (`S-metrics`, `S-log`) | `h-worker-ce3740e51defbdc89e6f50bad68f459b60cc31b98302a871f382f490ad6c46a4-988d8388-metrics.jsonl` (1009 lines), `...-988d8388-worker-log.jsonl` (260 lines) | MEASURED, `wc -l` |
| Magnitude arm files (`M-metrics`, `M-log`) | `h-worker-ed98fc2e7333d94403fffd4762baacedced6296dd97587537e1d3dff5c6b18da-a163161f-metrics.jsonl` (1009 lines), `...-a163161f-worker-log.jsonl` (260 lines) | MEASURED, `wc -l` |
| Readers consulted | `src/entropic_substrate/ladder/capture/reader/telemetry.py`; `docs/architecture.md` "Evidence plane" and "H run surface"; `src/entropic_substrate/ladder/run/worker_log_events.py`; the H0 row of `docs/hierarchical-latent-predictor.plan.md` | MEASURED (read in full) |

Carrier assignment used below follows the brief: `988d8388` = sphere (pod `euikqmf95pnrga`),
`a163161f` = magnitude (pod `211tvsc7277d22`). The pod ids are confirmed by `runtime.pod_id`
in each `run_start` row (S-metrics L1, M-metrics L1). The carrier name itself is NOT carried in
any captured row; it is taken from the brief and must be re-derived by the reviewer from the
profile (see "What a reviewer must re-derive").

## Q1. Admissibility under the H lane predicate

The capture gate has two parts. The H metrics predicate (`h_telemetry.final_h_run_end` over
`metrics.jsonl`; architecture "Evidence plane"): well-formed JSON-object lines, exactly one
`run_start`, exactly one `run_end` of schema `h-run-lifecycle-v1`, at least one committed step
row; `run_end.status` is read, not required (docstring: "deliberately no `status == "ok"`
condition"). Separately, the companion `worker-log.jsonl` is checked by `require_json_object_lines`
in the capture pass; the predicate itself does not read it. The brief's wave completion checks, reported separately
from admission: committed steps on one global axis through 1000, and evaluation rows at global 500/750/1000.

| check | sphere (988d8388) | magnitude (a163161f) | evidence |
| --- | --- | --- | --- |
| Every line parses as a JSON object | 1009 of 1009; 260 of 260 | 1009 of 1009; 260 of 260 | MEASURED, `json.loads` over every line of all four files |
| `run_start` rows (`h-run-lifecycle-v1`) | 1 (L1) | 1 (L1) | MEASURED, S-metrics L1, M-metrics L1 |
| `run_end` rows (`h-run-lifecycle-v1`) | 1 (L1009), `status: ok`, `result_status: probe_workload_complete`, `successful_optimizer_steps: 1000` | 1 (L1009), same three fields with the same values | MEASURED, S-metrics L1009, M-metrics L1009 |
| Committed training rows (`h-training-step-v2`) | 1000, all `status: committed` | 1000, all `status: committed` | MEASURED, S-metrics L2 to L251 and L253 to L1006; M-metrics same lines |
| Step axis as written in the rows | phase `codec_warmup` steps 1..250 (L2 to L251), then phase `prediction` steps 1..750 (L253 to L1006); each run strictly +1 | identical line ranges and step ranges | MEASURED, `step` and `phase` fields of every training row |
| One global axis through 1000 | NOT carried in the metrics rows: `step` restarts at 1 for `prediction`. The global axis is carried only by the worker log: `checkpoint.seal` events report `global_step` 250, 500, 750, 1000 with phase-local `step` 250, 250, 500, 750 | same | MEASURED (log events S-log L19, L48, L74, L100; M-log same event sequence); the mapping "row order + phase = global step" is DERIVED_FROM_MEASURED |
| Evaluation rows (`h-predictive-evaluation-v1`) | 3, at metrics `step` 250, 500, 750 (L503, L755, L1007) | 3, at metrics `step` 250, 500, 750 (L503, L755, L1007) | MEASURED |
| Evaluation rows on the global axis | `evaluation.start` events carry `global_step` 500, 750, 1000 (S-log L49, L75, L101); line positions L503, L755, L1007 sit after 500, 750 and 1000 training rows respectively | same (M-log same events) | MEASURED (log `global_step`), DERIVED_FROM_MEASURED (line-position argument). The brief's "500/750/1000" holds on the global axis; the metrics rows themselves say 250/500/750 |
| `annotate` rows (no schema key, excluded from schema-keyed readers by design) | 4, checkpoint tags for 00000250, 00000500, 00000750, 00001000 (L252, L504, L756, L1008) | 4, same positions | MEASURED |
| Row class totals | 1 + 1000 + 3 + 4 + 1 = 1009 | same | DERIVED_FROM_MEASURED |
| Worker-log events outside the closed vocabulary | 0 | 0 | MEASURED against `H_WORKER_LOG_EVENTS` |
| Worker-log levels | 260 INFO, 0 WARNING, 0 ERROR | 260 INFO, 0 WARNING, 0 ERROR | MEASURED |

Admissibility conclusion: both arms satisfy the H lane predicate as written in the
architecture (one `run_start`, one `run_end`, at least one committed row) and the companion
file passes its separate JSON-object-lines check; `status: ok` with 1000 committed rows is the observed result for these arms,
not part of the predicate. The wave completion check "one global axis through 1000" is satisfied only through the
worker log's `global_step` plus row order; the metrics rows carry a phase-local axis. That is
a finding for the reviewer and the gate author, not a refusal: the predicate as restated in
`docs/architecture.md` "Evidence plane" names no step-axis condition. RETRACTION 2026-10-22
(blind review round 1): the first version said "no reader in scope refuses it", which read as an
executed result; this analysis never ran a reader on these files (see the retraction in the next
section).

Note on the reader: `telemetry.py` as read in this checkout implements the E-series
`event == "eval"/"train"` readers and the D1 readers; it carries no `h-run-lifecycle-v1`
predicate. The H lane predicate is implemented by `h_telemetry.py` `final_h_run_end` in the same
package (lifecycle rows and committed-step check; no status condition), and the rule applied above
is that one, as also stated in `docs/architecture.md` "Evidence plane". The capture pass that
produced this directory is itself the admission instrument; this session re-applied the rule by
hand and reports the result as such. RETRACTION 2026-10-22 (blind review round 1): the
admission rows above were labelled MEASURED although `final_h_run_end` was not executed by this
analysis; the JSON-object counts were measured, the predicate verdict was re-applied by hand. The
reviewer executed `final_h_run_end` and `require_json_object_lines` through a standalone copy of
the reader on the captured files: both arms admissible, both worker logs pass (reviewer-reported,
review of record). RETRACTION 2026-10-22 (house rule 4, CodeRabbit thread
4343303939): the first version of this paragraph said the rule "lives in prose" only; `h_telemetry.py`
existed at every commit in scope, including the brief's bb732354.

## Q2. Lineage: same source, profile, image, code commit

| field | sphere (988d8388) | magnitude (a163161f) | same? | evidence |
| --- | --- | --- | --- | --- |
| `code_commit` (run_start, run_start.runtime, worker.start) | `9d353fcc2d6f82e19927ad8fec77eb78a75b4df0` in all three places | `9d353fcc2d6f82e19927ad8fec77eb78a75b4df0` in all three | yes | MEASURED, S-metrics L1, S-log L1; M-metrics L1, M-log L1 |
| `image` (run_start, runtime.requested_image, worker.start) | `ghcr.io/luiz-frias/entropic-substrate-lm@sha256:8181443b6c410c45a0e39eb472fe77ace8793c1dbe84cc38ce86de4b1af801ea` | same digest | yes | MEASURED, same rows |
| `runtime.source_attestation` | `exact_git_tree_match` | `exact_git_tree_match` | yes | MEASURED, run_start L1 |
| `run_class` / `runtime.kind` | `remote_tiny` / `remote_tiny` | same | yes | MEASURED, run_start L1 |
| `runtime.accelerator` | `{count: 1, model: NVIDIA A100-SXM4-80GB}` | same | yes | MEASURED, run_start L1 |
| `runtime.precision` | `default_dtype torch.float32`, `matmul_precision highest`, `effective_tf32 false`, both overrides null | identical | yes | MEASURED, run_start L1 |
| `profile_identity` = `run_identity` on every step and eval row | `ce3740e51defbdc89e6f50bad68f459b60cc31b98302a871f382f490ad6c46a4` (1003 of 1003 rows) | `ed98fc2e7333d94403fffd4762baacedced6296dd97587537e1d3dff5c6b18da` (1003 of 1003) | differs, as expected for different carriers | MEASURED |
| `request_sha256` (run_start equals worker.start) | `dec0d35e311bf8d49bfad068b7f37e7e83644e14d5501c0a8e89558f5cdfa96a`, equal in both files | `eb60d2147fb76fb5876cc9032aa16d7d668f588bcc5a17022eb09858f98dcbd2`, equal in both files | differs per arm (one request per arm) | MEASURED |
| MLflow run id (`mlflow.run` event) | `988d8388ee054a60b8756fe62b4e6232` | `a163161f21f44f57a547fe47b3f098e3` | differs (run identity) | MEASURED, S-log L2, M-log L2 |
| `runtime.pod_id` | `euikqmf95pnrga` | `211tvsc7277d22` | differs (run identity) | MEASURED, run_start L1 |
| R2 probe prefix (from publish keys) | `h-probes/v1/126e334ec651f924e69518aaaa6f2e528b111094aedb7ab6904584aff0cb3a8f/` | `h-probes/v1/0553aa53d3a3d30a53746e5558e0da074f9f9f8e4c2690ae5eb6764e0a071fe6/` | differs (one prefix per arm) | MEASURED, every `publish.*` event |
| Source batch digest of training row 1 | `b60570f47cdad94ead64bd5c46f6be1b10c4bbf72e298c1422109b3056658eab` | identical | yes | MEASURED, S-metrics L2, M-metrics L2 |
| Evaluation `source_batch` and example `document` digest | `710ac05827741c7dd8f01a93dc1165f104fe84f0baf13befe73d314ac717298f`; document `d8ec7203ccaf6faac53742e664264959a60ae00c7417df5ffeea19ec923d72f1`; `source_bytes [155, 143, 185, 116]`; identical across the 3 eval rows | identical to sphere, all 3 eval rows | yes | MEASURED, L503, L755, L1007 of both metrics files |
| Sum of `source_bytes` / `encoder_bytes` / `history_bytes` over 1000 training rows | 402,212 / 1,534,453 / 1,132,241 | 402,212 / 1,534,453 / 1,132,241 | yes | DERIVED_FROM_MEASURED (sums of per-row fields) |
| Ordered per-row comparison of `source_batch`, `source_examples`, `source_bytes`, `encoder_bytes`, `history_bytes`, `boundary_symbols`, `phase`, `step` over all 1000 training rows, row i against row i | 1000 distinct `source_batch` digests | 1000 distinct `source_batch` digests | yes: 0 mismatches on every field | MEASURED 2026-10-22, `python3 -I` over both metrics files in row order (added after CodeRabbit thread 4343303926 noted that row 1 plus sums did not prove it) |
| `boundary_symbols` on every training row | 4 | 4 | yes | MEASURED |
| Published `source/prepared-corpus.json` size | 2,725,370 bytes | 2,725,370 bytes | yes (size only; content hash is in R2, not captured) | MEASURED, `publish.end` events |
| Profile `config_hash` | not present in any captured row | not present | cannot be compared from the captured files | MEASURED absence. `config_hash` is an MLflow param and tag per the H run surface; it is not projected into `metrics.jsonl` or `worker-log.jsonl`. RETRACTION 2026-10-22 (blind review round 1): "same profile `config_hash`" was ill-posed as worded, because two profiles that differ in carrier by design cannot share a hash over the whole profile. The meaningful leg is "identical except carrier and name"; see the paragraph after the lineage conclusion |

Lineage conclusion: both arms ran the same code commit, the same image digest, the same
attestation, the same accelerator class and precision, over an identical source stream (ordered
per-row match on all 1000 training rows, identical evaluation batch). They differ in profile
identity, request digest, run id, pod and R2 prefix, which is what "differing only in carrier
and run identity" predicts. The shared `config_hash` claim is not provable from the captured
files alone.

RETRACTION 2026-10-22 (blind review round 1): the preceding sentence inherited the brief's
"same profile config_hash" wording, which cannot hold between a sphere profile and a magnitude
profile. Restated leg: the two profiles are identical except for `law.mode`, `spine.carrier` and
`name` (the carrier setting and its name). Evidence, PACKET-ONLY: the
reviewer fetched `outputs/run/profile.json` for both arms from R2 and diffed them; I re-diffed
the same two files from the review packet (`artifacts/sphere-profile.json`,
`artifacts/magnitude-profile.json`, flattened to dotted keys) and the only differing fields are
`law.mode` (sphere / magnitude), `spine.carrier` (sphere / magnitude) and `name`
(`h0-sphere-b64-1k-v3-fisher-tiny` / `h0-magnitude-b64-1k-v3-fisher-tiny`). MEASURED 2026-10-22
from the packet copies, PACKET-ONLY evidence: the copies are bound to their arms by the receipt
`run_identity` and the worker-logged published size, not by a hash of the R2 object, so this row
establishes the packet profiles' contents, not a proven property of the captured runs.

## Q3. Final leg: admission deltas and stage wall seconds

All eight `WORKER_STAGES` appear once as a `stage.start`/`stage.end` pair in each log. Seven
`admission.start`/`admission.end` pairs appear: one per final-leg stage after training, plus the
`completion` caller, which has no stage bracket of its own. RETRACTION 2026-10-22 (blind review
round 1): the first version placed `completion` between `phase.end prediction` and the first
final-leg stage. It sits INSIDE the prediction phase window: `admission.start` at S-log L103
(14:20:23.098Z), `admission.end` at L104 (14:25:00.481Z), `phase.end prediction` at L128
(14:25:04.007Z), against `phase.start prediction` at L47 (14:02:42.771Z). Magnitude has the
same line numbers. `materialization` and `runtime_admission` have stage
brackets but no admission pair.

| stage (caller) | memo | root | seals walked | evaluations | journals read | admission s, sphere | stage s, sphere | admission s, magnitude | stage s, magnitude | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| materialization | n/a | n/a | n/a | n/a | n/a | n/a | 0.677 | n/a | 0.805 | MEASURED, S-log L3-L4, M-log L3-L4 |
| runtime_admission | n/a | n/a | n/a | n/a | n/a | n/a | 2.673 | n/a | 3.091 | MEASURED, S-log L5-L6, M-log L5-L6 |
| completion (no stage bracket) | fresh | run | 4 | 3 | 4 | 277.383 | n/a | 276.919 | n/a | MEASURED, S-log L103-L104, M-log same events |
| training_readback | shared | run | 0 | 3 | 0 | 143.292 | 143.292 | 143.823 | 143.824 | MEASURED, S-log L129-L132 |
| generation | shared | run | 0 | 3 | 0 | 154.805 | 200.350 | 154.387 | 199.508 | MEASURED, S-log L133-L136 |
| span_diagnostics | shared | run | 0 | 6 | 0 | 320.943 | 542.210 | 319.886 | 506.737 | MEASURED, S-log L137-L140 |
| trace_capture | shared | run | 0 | 9 | 0 | 452.803 | 509.957 | 452.113 | 509.181 | MEASURED, S-log L141-L144 |
| native_verification | fresh | capture | 4 | 9 | 4 | 586.632 | 644.006 | 585.400 | 642.853 | MEASURED, S-log L145-L148 |
| result_admission | shared | run | 0 | 3 | 0 | 143.045 | 143.157 | 143.265 | 143.378 | MEASURED, S-log L149-L152 |

Counts (`seals walked`, `evaluations`, `journals read`, memo, root) are identical on both arms
(MEASURED, each `admission.end` row). Seconds are rounded to 3 decimals from the raw floats in
the rows; the rounding is DERIVED_FROM_MEASURED.

Supporting clocks:

| quantity | sphere | magnitude | evidence |
| --- | --- | --- | --- |
| Final-leg stage seconds, sum of the six bracketed stages after training | 2,182.973 | 2,145.481 | DERIVED_FROM_MEASURED (sum of `stage.end.seconds` raw floats: sphere 2182.9731697, magnitude 2145.4812543). RETRACTION 2026-10-22 (blind review round 1): sphere was written 2,182.972, a 1 ms slip from summing rounded values; corrected |
| `phase.end` seconds, `codec_warmup` / `prediction` | 76.443 / 1,341.236 | 77.519 / 1,187.615 | MEASURED, S-log L46, L128; M-log same events. RETRACTION 2026-10-22 (blind review round 1): the prediction figure includes the `completion` admission (277.383 / 276.919 s, L103-L104), so it is not training time alone |
| `phase.end` prediction seconds minus `completion` admission seconds | 1,063.853 | 910.696 | DERIVED_FROM_MEASURED (1341.2359 - 277.3826; 1187.6145 - 276.9187), valid because L103-L104 lies wholly inside the phase window |
| Sum of per-row training `wall_seconds`, warmup / prediction | 5.410 / 590.423 | 5.409 / 434.894 | DERIVED_FROM_MEASURED |
| Sum of per-row training `cuda_seconds` | 595.429 | 439.906 | DERIVED_FROM_MEASURED |
| `checkpoint.seal` seconds at global 250 / 500 / 750 / 1000 | 22.170 / 33.561 / 33.594 / 33.570 | 22.103 / 33.513 / 33.500 / 33.534 | MEASURED |
| `evaluation.end` seconds at global 500 / 750 / 1000 | 66.962 / 67.050 / 67.187 | 66.886 / 66.734 / 66.602 | MEASURED |
| `snapshot.end` seconds: started / ckpt 250 / 500 / 750 / 1000 / complete | 1.155 / 3.620 / 2.816 / 2.811 / 2.982 / 19.003 | 1.063 / 4.676 / 7.876 / 2.693 / 2.779 / 17.270 | MEASURED |
| `annotate` `checkpoint/<step>/publication_seconds` (string-typed) vs `snapshot.end` seconds | 3.6227 vs 3.6199; 2.8200 vs 2.8159; 2.8164 vs 2.8106; 2.9899 vs 2.9825 | 4.6786 vs 4.6762; 7.8805 vs 7.8762; 2.6985 vs 2.6928; 2.7865 vs 2.7792 | MEASURED both sides; they are two clocks around the same publication, not one value |
| `run_end.wall_seconds` (status ok: worker's own execution time, copied before result admission and final snapshot) | 3,518.295 | 3,328.438 | MEASURED, L1009 |
| `worker.start` timestamp to first `sink.finish.start` timestamp | 3,687.364 | 3,496.569 | DERIVED_FROM_MEASURED (timestamp difference) |
| Difference of the two clocks above | 169.070 | 168.130 | DERIVED_FROM_MEASURED from raw floats (169.0698 / 168.1304). RETRACTION 2026-10-22 (blind review round 1): was written 169.069 / 168.131, 1 ms slips from differencing rounded values; corrected; the architecture defines `wall_seconds` for `status: ok` as ending before result admission (143.157 / 143.378) and final-snapshot custody (19.003 / 17.270), which accounts for most of the gap; the remainder is not itemised in the captured files. Residual after subtracting both named parts, DERIVED_FROM_MEASURED 2026-10-22: 169.070 - 143.157 - 19.003 = 6.910 s (sphere) and 168.130 - 143.378 - 17.270 = 7.482 s (magnitude); the residual assumes the architecture's ordering and is not itself an observed interval |
| Wave start (`worker.start`) | 2026-10-19T14:00:19.075Z | 2026-10-19T14:00:18.503Z | MEASURED, L1 of each log |
| Last log row (`sink.finish.start sink=mlflow`) | 2026-10-19T15:01:46.440Z | 2026-10-19T14:58:35.072Z | MEASURED, L260 of each log |

## Q4. Stochastic rollout: what the captured files show

| item | sphere | magnitude | evidence |
| --- | --- | --- | --- |
| `generation` stage ran once, bracketed | start 14:27:27.305Z, end 14:30:47.655Z, 200.350 s | start 14:24:55.770Z, end 14:28:15.278Z, 199.508 s | MEASURED, S-log L133/L136, M-log same events |
| Admission inside the stage | memo shared, root run, 0 seals, 3 evaluations, 0 journals, 154.805 s | same counts, 154.387 s | MEASURED, `admission.end caller=generation` |
| Generation artifacts published (bytes) | `outputs/capture/126390cec5e749329c88f1884125add1-generation.json` 7,654; `...-generation.pt` 13,009; the same two under `outputs/run/` | `outputs/capture/de04cc32cb6e447b96ae76598f816a52-generation.json` 4,057; `...-generation.pt` 11,281; same two under `outputs/run/` | MEASURED, `publish.end` events |
| Verify-read of each generation artifact | attempt 1 of max 1 or 3, no retry | same | MEASURED, `publish.verify_read` rows; `publish.retry` count is 0 in both logs |
| Continuation count per prompt | not in any captured row | not in any captured row | MEASURED absence |
| Boundary emitted or explicit truncation outcome per continuation | not in any captured row | not in any captured row | MEASURED absence |
| Sampling law, temperature, seed of the rollout | not in any captured row | not in any captured row | MEASURED absence |
| `h-predictive-evaluation-v1` rows (teacher-forced importance-weighted estimate, `meaning: importance_weighted_log_marginal_estimate`) | 3 rows; `by_sample_count` keys `2` and `4`, 4 entries each; `log_weights` 4 x 4; `effective_sample_size` range 1.0000 to 1.8753 across the 24 entries; every `gap` is null | 3 rows; same shape; ESS range 1.0000 to 1.4707; every `gap` null | MEASURED. These rows are evaluation under the sealed checkpoint, not a free rollout, and are listed here only to say what the metrics file does and does not carry |

Addendum 2026-10-22 (blind review round 1): the reviewer's packet carried the two
`generation.json` receipts (R2 `outputs/run/`). Re-read by this lane from the packet copies,
MEASURED: schema `h-generation-receipt-v2`, `status: generation_complete`, one prompt, two
generated sentences per arm, `outcome: sentence_budget`, `pending: []`, 284 (sphere) and 136
(magnitude) entries in `log_scores` and `observations`. No sampler, temperature or seed field is
in the receipt, and there is one continuation per prompt, so stochasticity is still not shown
(see the review response).

What needs R2 custody to answer Q4: the two `generation.json` files (one per arm) under
`outputs/run/` or `outputs/capture/` at the prefixes in Q2, and the `complete.json` /
`worker/result.json` markers (787 / 788 and 1,712 / 1,713 bytes). The captured files prove
the generation stage ran, admitted, published and verify-read its artifacts. RETRACTION
2026-10-22 (PR #373 review): the packet receipts, read above, DO show one finished continuation
per prompt and its termination outcome (`sentence_budget`); what they do not show is
stochasticity or an explicit token length, and R2 custody is still needed only to bind the
packet receipts to the R2 objects by hash. The stochastic-rollout
exit is therefore NOT proven by the captured evidence.

## Q5. Recovery (checkpoint and resume)

| item | value | evidence |
| --- | --- | --- |
| Checkpoints sealed and published | 4 per arm (global 250, 500, 750, 1000), each with key, sha256, size and publication seconds | MEASURED, `checkpoint.seal` events and `annotate` rows L252, L504, L756, L1008 |
| Checkpoint snapshot sizes (json marker) | 2,044 / 3,699 / 5,353 / 7,007 bytes, identical on both arms | MEASURED, `annotate` rows |
| Resume from a checkpoint exercised in this wave | no. Each log has exactly one `worker.start`, one `mlflow.run`, one `phase.start` per phase, and no event names a resume, a restored checkpoint, or a second start | MEASURED absence (event counts) |
| `stop_marker.observed` / `stop_marker.retry` | 0 / 0 on both arms | MEASURED |

No resume was exercised remotely. The recovery exit stays OPEN. Nothing here is inferred from
the local checkpoint/resume unit tests; those are out of this brief's reading scope and do not
count toward the H0 exit.

## Q6. Numerics: nonfinite, nonmonotonic, missing, failed, stranded

| check | sphere | magnitude | evidence |
| --- | --- | --- | --- |
| Numeric leaves scanned, metrics file only (every int or float in every row, bool excluded) | 102,050 | 104,364 | MEASURED. RETRACTION 2026-10-22 (blind review round 1): the first version said "every row" but counted the metrics files only |
| Numeric leaves scanned, worker log only (same rule) | 484 | 484 | MEASURED 2026-10-22 (`python3 -I` over the review-packet copies of the captured files) |
| Numeric leaves scanned, metrics plus worker log | 102,534 | 104,848 | DERIVED_FROM_MEASURED (sum of the two rows above); nonfinite values across all of them: 0 / 0, MEASURED |
| Nonfinite values (NaN, +inf, -inf) | 0 | 0 | MEASURED |
| Training `step` strictly +1 within each phase; phases in file order warmup then prediction | yes; one transition at training row 250 | yes; same | MEASURED |
| Worker-log timestamps non-decreasing | yes, 260 rows | yes, 260 rows | MEASURED |
| Negative `seconds` in any log row | 0 | 0 | MEASURED |
| `future_nll_per_dimension` null | 250 rows, all `codec_warmup`; the same rows carry `gradients.future_law.status` and `gradients.spine.status` = `structurally_absent_codec_warmup` and `mechanism.status` = `structurally_absent` with `reason: codec_warmup` | same 250 rows, same labels | MEASURED. Absence is labelled, not silent |
| `gradients.encoder.status` | `measured_nonzero` 250 (warmup), `structurally_absent_frozen_encoder` 750 (prediction) | same | MEASURED |
| `gradients.decoder.status` | `measured_nonzero` 1000 | same | MEASURED |
| `mechanism.solver` (prediction rows) | method `conjugate_gradient` 750 of 750; `max_residual_ratio` min 0.175505, max 0.999855, rows at or above 1.0: 0; largest iteration bucket 49 | 750 of 750; min 0.172860, max 0.999992, rows at or above 1.0: 0; largest iteration bucket 50 | MEASURED. A ratio below 1 on every row means no row reports a residual above its tolerance; no `solver_nonconvergence` status appears |
| `mechanism.*.zero_count` and `zero_views` | all 0 | all 0 | MEASURED |
| `mechanism.sentences.range` null on prediction rows | 18 of 750 prediction rows, at prediction steps 3, 39, 178, 208, 233, 278, 299, 308, 415, 440, 487, 496, 534, 565, 593, 636, 725, 748 (metrics L255, L291, L430 first three, L1004 last); each carries `count: 0`, `zero_count: 0`, `dtype: torch.float32`; the same rows carry `states.count: 4` with a non-null `states.range` | the same 18 steps, same shape | MEASURED 2026-10-22 over both metrics files (omitted by the first version, found by the blind review). Explanation: structurally absent, not missing. `_norms` returns `"range": _range(norms) if norms.numel() else None` (`mechanism_evidence.py:30`), and `sentences` is the tensor of history sentence states; on these rows every source example has an empty history (the first sentence of its document), so there are zero sentence states. Corroboration, MEASURED: the 18 rows are exactly the prediction rows with `history_bytes` equal to 0 on both arms; the smallest nonzero prediction `history_bytes` is 100. The "first sentence of its document" cause is the code-reading explanation; no captured row names it |
| Declared precision versus observed precision | profile `precision: bf16`, `device: cuda` (R2 `profile.json`, packet copy); runtime record `default_dtype torch.float32`, `matmul_precision highest`, `effective_tf32 false` (run_start L1); mechanism rows `dtype torch.float32` on every row | same | MEASURED for each side separately. The two are not in contradiction by construction: autocast is applied inside the objective as `torch.autocast(..., dtype=torch.bfloat16, enabled=self.profile.precision == "bf16")` (`execution.py:224-226`, and again for evaluation at L345-349), so `bf16` is the autocast dtype, while the runtime record is the enforced FP32-islands contract (`runtime_precision.py:23-49`, `require_execution_precision`, called at `execution.py:214` and `:326`) and the mechanism rows' `float32` is the FP32 carrier contract (`mechanism_evidence.py:21-23`). What is NOT recorded in any captured row is whether autocast was actually enabled on the live step, so "bf16 was applied" is code-read, not measured. Open observability item (ticketed); see the review response |
| `decoder_ce` first / last training row | 5.951573 / 2.673644 | 5.952548 / 2.682614 | MEASURED, L2 and L1006 (reported as plumbing readback only; no claim attaches) |
| `future_nll_per_dimension` first prediction row / last row; minimum over run | 4.643691 / -0.540911; min -0.740573 | 5.355175 / 0.552849; min -0.708967 | MEASURED. Negative per-dimension values occur; whether the field's documented domain admits them is for the reviewer to check against the schema, which this brief did not put in scope. Addendum 2026-10-22: answered from code in `docs/h0/review-response-hepp7g-2026-10-22.md` row 7 |
| `aggregate_norm` first / last | 1.010331 / 1.007102 | 1.022729 / 1.146669 | MEASURED |
| `cuda_peak_bytes` maximum over run, element 1 (max memory reserved) | 81,788,928 | 77,594,624 | MEASURED. RETRACTION 2026-10-22 (blind review round 1): the field is a two-element list `[allocated, reserved]` on every one of the 1000 training rows (`execution.py:293` writes `memory.peak_bytes()`; `resource_meter_runtime.py:136-143` returns `(max_memory_allocated, max_memory_reserved)`; `journal.py` requires length 2 and `peak[1] >= peak[0]`). The first version reported only the second element, the reserved high-water, without saying so |
| `cuda_peak_bytes` maximum over run, element 0 (max memory allocated) | 60,756,992 | 56,941,568 | MEASURED 2026-10-22; `peak_scope` is `process_allocator_high_water` (`execution.py:294`), so both values are process-wide high-water marks, not per-step peaks |
| Evaluation `gap` field | null in all 24 `by_sample_count` entries (8 per eval row) | same | MEASURED. A null field; its meaning is not defined by any reader in scope |
| Evaluation `effective_sample_size` | all 24 entries finite, min 1.0000 | same | MEASURED |
| `run_end.status` | `ok` | `ok` | MEASURED |
| `FAILED` anywhere (run status, event, level) | none | none | MEASURED |
| `sink.write_failed` / `sink.drain_exceeded` | 0 / 0 | 0 / 0 | MEASURED |
| `sink.finish.end sink=jsonl` | `settled: true`, `stranded: 0`, 0.000142 s | `settled: true`, `stranded: 0`, 0.000142 s | MEASURED, L259 |
| `sink.finish.end sink=mlflow` | absent; the log ends at `sink.finish.start sink=mlflow` (L260) | same | MEASURED. Structural: the log file is uploaded at MLflow sink finish, so its own end cannot appear in it. The MLflow drain outcome is therefore not in the captured files |
| `worker.end` event | absent in both logs | absent | MEASURED. Same structural reason as the row above; the completion marker `completion.written` (L257) is the last worker-authored milestone |
| Publish accounting | 66 `publish.start`, 66 `publish.end`, 66 `publish.verify_read`, 0 `publish.retry`; 17,533,056 bytes; 34.545 s | 66 / 66 / 66 / 0; 17,552,900 bytes; 38.680 s | MEASURED counts, DERIVED_FROM_MEASURED sums |
| String-typed numerics | the four `annotate` rows carry `publication_seconds` and `size_bytes` as strings | same | MEASURED. Consistent with the tag contract (`str -> str`); no schema-keyed reader consumes them |

## Verdict proposal

`tracer_partial`. (Filed in `docs/h0/verdict-hepp7g-2026-10-22.md` after the independent review;
the corrections in this file are those the review found.)

Proven by the captured evidence, both arms, one lineage:

- Both carriers ran to `run_end status ok` with 1000 committed steps on the same code commit,
  image digest, attestation, accelerator and source stream (Q1, Q2).
- Capture through the H lane predicate succeeded and the pull is byte-identical to the
  pointer (Inputs, Q1).
- The final leg ran all eight closed-vocabulary stages on both arms, with admission pairs
  recorded for six of them plus the `completion` caller (seven pairs; `materialization` and
  `runtime_admission` have stage brackets only, Q3), zero retries, zero stop markers, zero write failures, JSONL sink settled with zero
  stranded (Q3, Q6).
- Zero nonfinite values across 206,414 numeric leaves in the metrics files, and across 207,382 including the worker logs (RETRACTION 2026-10-22, blind review round 1: the first version labelled the metrics-only total as covering every row); monotone steps and timestamps (Q6).
- This analysis exists (the analysis exit).

Unproven exits, each named:

1. **Stochastic rollout.** The generation stage ran and published two artifacts per arm, but
   no continuation, continuation count, boundary emission or truncation outcome is in the
   captured files. The packet receipts (Q4 addendum) show one finished continuation per prompt
   and its `sentence_budget` termination; they do not show stochasticity (no sampling law, one
   continuation). R2 custody is still needed only to hash-bind the packet receipts to the R2
   objects (keys in Q4).
2. **Recovery.** No resume was exercised in this wave (Q5). Open.
3. **Test-only gate.** Not yet exercised against this capture.
4. **Shared profile `config_hash`.** RETRACTION 2026-10-22 (blind review round 1): this exit
   was ill-posed as worded. It is replaced by "profiles identical except `law.mode`,
   `spine.carrier` and `name`", which the two packet profile copies satisfy (Q2, restated leg;
   PACKET-ONLY evidence, bound to the arms by receipt `run_identity` and logged size, not
   hash-bound to the R2 objects). Not an open exit any more; the hash
   itself is still not projected into the captured rows.
   (Review and filing left this list on 2026-10-22: the blind review round 1 and the filed
   `tracer_partial` verdict are recorded in `verdict-hepp7g-2026-10-22.md`.)

Findings that are not refusals but must reach the gate author:

- The metrics rows carry a phase-local `step` axis (1..250 then 1..750). The global axis
  through 1000 exists only in the worker log's `checkpoint.seal` / `evaluation.start`
  `global_step` and in row order. A reader that keys on `step` alone will see the evaluation
  rows at 250 / 500 / 750, not 500 / 750 / 1000.
- `telemetry.py` in this checkout carries no H lane predicate; the H predicate is
  `h_telemetry.py` `final_h_run_end` (lifecycle rows plus committed-step check, no status
  condition), restated in `docs/architecture.md`. The brief as used named `telemetry.py` as the
  only read side; for the H lane that name pointed at the wrong module (corrected in the brief,
  retraction dated 2026-10-22).
- `future_nll_per_dimension` goes negative on both arms (minimum -0.740573 and -0.708967), and
  `gap` is null in every evaluation entry. Neither is refused by any reader in scope; whether
  they are in-domain is a schema question for the reviewer.

Not `numerical_defect`: no nonfinite, nonmonotonic, failed or stranded evidence was found.
Not `missing_evidence` for the plumbing claim: the capture is complete and admissible for
what it was meant to carry; the missing pieces are exits the wave did not exercise or
artifacts the capture was never designed to project.

## What an independent reviewer must re-derive

1. Pull the same pointer (`1d3d67d1`, md5 `2dc65e2bdf918d43858c6b2d92a63c58.dir`) and confirm
   the four file sizes and line counts in "Inputs read".
2. Re-count the row classes per metrics file (1 / 1000 / 3 / 4 / 1) and the event counts per
   log (the Q1 and Q6 tables), from the raw files, with their own script.
3. Re-check the lineage equalities in Q2 field by field, including the per-row
   `run_identity` constancy and the identical source-batch and evaluation digests.
4. Re-derive the Q3 stage and admission table from the `stage.end` and `admission.end` rows,
   and re-compute the two clocks (`run_end.wall_seconds` versus `worker.start` to
   `sink.finish.start`) and their difference.
5. Re-scan every numeric leaf for finiteness and re-check `max_residual_ratio < 1` on all
   1500 prediction rows.
6. Confirm the carrier-to-run-id assignment (`988d8388` sphere, `a163161f` magnitude) from the
   profile, since no captured row names the carrier. This needs the profile artifact under
   R2 `outputs/run/profile.json` or the request, which this session did not read. (Done by the
   blind review on 2026-10-22: three independent links, reviewer-reported; see the verdict file.)
7. Decide, against the schemas, whether negative `future_nll_per_dimension` and a null
   evaluation `gap` are in-domain.
8. If the stochastic-rollout exit is to be closed in this round, read the two
   `generation.json` artifacts from R2 under the keys in Q4 and report continuation count,
   boundary emission and truncation per prompt; this analysis did not.
