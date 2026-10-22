# H0 blind review round 1: author response, wave HEpP7g

Response to the independent blind review of `docs/h0/analysis-hepp7g-2026-10-20.md`. The
reviewer (model `claude-opus-5-5`, network-isolated sandbox, 2026-10-22T16:12Z) saw the
analysis, the brief, the four captured JSONL files, the per-arm `profile.json` and
`generation.json` fetched from R2, and a standalone H reader. The reviewer's full six-section
file was lost by the tooling (ticketed: FOR-1671); the review of record is the reviewer's chat summary, archived at
`~/.claude/projects/-Users-luizfrias-CursorAI-entropic-substrate-lm/refactor-survey-2026-09-17/h-sprint/h0-review/review-cloud-round1-summary.md`
(operator machine, not in git). Every reviewer statement below is therefore REVIEWER-REPORTED.
Every number marked MEASURED was recomputed by this lane from the packet copies of the captured
files with `python3 -I`, independently of the reviewer.

Dispositions: `corrected` means the analysis file was fixed in place with a dated retraction;
`explained` means no change to the analysis beyond a pointer; `ticketed` means a follow-up
exists outside this repo state; `open` means the exit stays unproven.

## Major concerns

| # | concern | author response | evidence | disposition |
| --- | --- | --- | --- | --- |
| M1 | Profiles declare `precision: bf16`, yet the runtime record and every tensor report float32 | Not a contradiction by construction, but the analysis missed the relation. `bf16` is the autocast dtype; the runtime record is the enforced FP32-islands contract; mechanism rows' float32 is the FP32 carrier contract. What no captured row records is whether autocast was on during the live step, so "bf16 was applied" is code-read, not measured | profile `precision: bf16` (both packet profiles, MEASURED); autocast site `src/entropic_substrate/ladder/run/execution.py:224-226` and `:345-349`, `enabled=self.profile.precision == "bf16"`; contract check `src/entropic_substrate/core/sentence/runtime_precision.py:23-49` called at `execution.py:214`; carrier check `src/entropic_substrate/ladder/run/mechanism_evidence.py:21-23`; run_start L1 `runtime.precision` | corrected (row added to analysis Q6); autocast capture ticketed as an observability gap |
| M2 | Rollout not shown to be stochastic | Agreed. One prompt, one continuation per arm, and no sampler, temperature or seed field in either receipt, so a deterministic decoder would produce the same receipts. The receipts do show finished continuations: `generation_complete`, `outcome: sentence_budget`, `pending: []`. That proves "generation produced finished continuations", not stochastic rollout | `artifacts/sphere-generation.json`, `artifacts/magnitude-generation.json` in the packet: 1 prompt, 2 sentences each, 284 and 136 `log_scores` entries (MEASURED); no sampler key in either receipt (MEASURED absence) | open |
| M3 | Admissibility predicate never executed by the analysis, yet labelled MEASURED | Agreed. The analysis re-applied the predicate by hand and cited the wrong module before the first correction. The JSON-object counts were measured; the predicate verdict was not an execution. The reviewer ran `final_h_run_end` and `require_json_object_lines` through the standalone reader: both arms admissible (REVIEWER-REPORTED). The reviewer also notes the predicate accepts NaN and Infinity because Python's JSON parser does; the finiteness scan in Q6 is the separate check that covers it | analysis Q1 note on the reader, retraction added; Q6 nonfinite row, 0 / 0 over metrics plus logs (MEASURED) | corrected |
| M4 | "Shared profile `config_hash`" exit is ill-posed | Agreed. Two profiles that differ in carrier by design cannot share a hash over the whole profile. The meaningful leg is "identical except carrier and name". Re-diffed from the packet copies: differing fields are `law.mode`, `spine.carrier`, `name`, nothing else. The packet copies are not hash-bound to the R2 objects | `artifacts/sphere-profile.json` vs `artifacts/magnitude-profile.json`, flattened to dotted keys (MEASURED 2026-10-22) | corrected (exit restated and closed; hash projection into captured rows stays absent) |

## Minor concerns

| # | concern | author response | evidence | disposition |
| --- | --- | --- | --- | --- |
| m1 | Three 1 ms rounding slips | Reviewer's values confirmed, no disagreement. Recomputed from raw floats: final-leg stage sum sphere 2182.9731697, rounds to 2182.973 (was 2182.972); clock difference sphere 169.0697937, rounds to 169.070 (was 169.069); magnitude 168.1303992, rounds to 168.130 (was 168.131). The first version differenced or summed already-rounded values | `stage.end.seconds` rows S-log L4, L6, L132, L136, L140, L144, L148, L152; `run_end.wall_seconds` 3518.2946262 (sphere) and 3328.4384169 (magnitude); `worker.start` to first `sink.finish.start` 3687.3644194 and 3496.5688162 s (MEASURED) | corrected |
| m2 | `completion` admission placed after the prediction phase, but it sits inside it | Agreed. `admission.start` S-log L103, `admission.end` L104, `phase.end prediction` L128, `phase.start prediction` L47, same lines on magnitude. Reported prediction `phase.end` seconds therefore include the admission: 1341.236 s less 277.383 s is 1063.853 s (sphere); 1187.615 s less 276.919 s is 910.696 s (magnitude) | S-log and M-log L47, L103, L104, L128 (MEASURED); subtraction DERIVED_FROM_MEASURED | corrected |
| m3 | Numeric-leaf totals cover metrics only | Agreed. Metrics files: 102,050 / 104,364. Worker logs: 484 / 484, reviewer's count confirmed. Both files: 102,534 / 104,848. Nonfinite across both: 0 / 0. Sum over both arms: 206,414 metrics-only, 207,382 including logs | recount over the four packet files (MEASURED) | corrected |
| m4 | `cuda_peak_bytes` is a two-element list, only the second reported | Agreed. Element 0 is max memory allocated, element 1 is max memory reserved, a high-water mark over the process (`peak_scope` is `process_allocator_high_water`). Maxima over 1000 rows, sphere / magnitude: element 0 60,756,992 / 56,941,568; element 1 81,788,928 / 77,594,624 | `execution.py:293-294`; `src/entropic_substrate/ladder/capture/resource/resource_meter_runtime.py:136-143`; `src/entropic_substrate/ladder/run/journal.py` length-2 check (MEASURED values, code-read order) | corrected |
| m5 | 18 prediction rows per arm with `mechanism.sentences.range: null` | Agreed that the first version omitted them. They are structurally absent, not missing: zero sentence states on rows whose every source example has an empty history (first sentence of its document). The 18 rows are exactly the prediction rows with `history_bytes` 0 on both arms; the smallest nonzero prediction `history_bytes` is 100. Steps: 3, 39, 178, 208, 233, 278, 299, 308, 415, 440, 487, 496, 534, 565, 593, 636, 725, 748, identical on both arms | `src/entropic_substrate/ladder/run/mechanism_evidence.py:30` (`range` is `None` when `norms.numel()` is 0); metrics rows L255, L291, L430, ..., L1004 (MEASURED) | corrected (row added to Q6) |
| m6 | Two files in the archive outside the manifest | Packaging artifacts of the review tooling, not evidence. `REVIEWER-PROMPT.txt` is byte-identical to `reviewer-prompt-round1.txt` (`diff` exit 0, MEASURED). The stale Python 3.12 bytecode copy of `reader.py` is not present in the local packet directory, so its origin is not established by this lane; the reviewer reports not using it and loading `reader.py` from source | local packet listing; `packet-manifest-round1.txt` lists 12 files | explained |
| m7 | Item 7: negative `future_nll_per_dimension` and null `gap` undecidable without schemas | See the two-part answer below the table | below | partly explained, partly open |
| m8 | Unexplained timing gaps | The 169.070 / 168.130 s gap between `run_end.wall_seconds` and the last log clock is mostly named by the architecture: result admission (143.157 / 143.378 s) and final-snapshot custody (19.003 / 17.270 s). Residual after both: 6.910 s (sphere) and 7.482 s (magnitude). The residual rests on the architecture's ordering and is not an observed interval; no captured row itemises it | analysis Q3 supporting clocks; subtraction DERIVED_FROM_MEASURED | explained, residual stays unitemised |

## Item 7, answered from code

| sub-question | answer | anchor | disposition |
| --- | --- | --- | --- |
| Is a negative `future_nll_per_dimension` in domain? | The quantity is `-log_prob(target).mean() / width` of a continuous latent law (`prediction_loss`), a differential quantity. A continuous density may exceed 1, so its negative log may be below 0. The H aggregate documents that the field is "negative on most prediction rows" and reports it "as captured, never clamped, abs-ed or filtered". So the sign is expected and not a defect signal. This is an in-domain statement about sign only; it makes no claim that the value is good or that the declared measure is the intended one | `src/entropic_substrate/core/sentence/objective.py` (`prediction_loss`, line `future = -readout.prediction.law.log_prob(target).mean() / model.codec.spec.width`; class docstring "Future density is averaged per latent dimension, with its declared measure"); `src/entropic_substrate/ladder/analyze/h_aggregate.py:29-31` | explained; the declared measure per carrier is not stated in either file and stays open |
| Is a null evaluation `gap` in domain? | Yes, it is the normal case. `_check_evaluation_entry` admits two shapes: `gap` null with three finite numbers (`log_estimate`, `effective_sample_size`, `likelihood_relative_mcse`), or `gap == "zero_importance_weight_mass"` with three nulls. A non-null gap marks the exceptional zero-weight case, so null across all 24 entries per arm means no zero-weight entries | `h_aggregate.py:261-282` (`_EVALUATION_ENTRY_FIELDS`, `_ZERO_MASS_GAP`, `_check_evaluation_entry`) | explained |

## Open after this response

| item | why it stays open |
| --- | --- |
| Stochastic rollout | one continuation per prompt, no sampler, temperature or seed recorded (M2) |
| Recovery and resume | no resume event in either log (analysis Q5) |
| Test-only gate against this capture | not exercised |
| Autocast observability | whether autocast was enabled on the live step is not recorded (M1) |
| Declared measure of the future density per carrier | named in the objective docstring, not defined there (item 7) |
