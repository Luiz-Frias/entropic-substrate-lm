# H0 verdict: final probe, waves h0-final-probe-1k-v4.92c9db and h0-final-probe-resume-v4.1755b0

## Verdict

`tracer_partial`. The author analysis proposed `tracer_complete`; the independent blind review
returned `tracer_partial` with two blocking concerns about what its packet could verify. Both
concerns are resolved by evidence in the DVC captures, measured by the author after the review,
but the rule for this filing is "`tracer_complete` iff every check PASSes and the review raises
no blocking finding", and a verdict comes from a non-producer reading sealed evidence. The
author resolving the reviewer's concerns is not the reviewer agreeing. One exit stays open:
the test-only gate as independently verified. This is a plumbing gate with no scientific claim
and no H1 eligibility beyond "the machinery exists"; it authorizes no spend.

Vocabulary used: `tracer_complete` / `tracer_partial` / `numerical_defect` / `missing_evidence`
(H0 row of `docs/hierarchical-latent-predictor.plan.md`). Not `numerical_defect`: no nonfinite,
nonmonotonic, failed or stranded evidence on any of the four pods. Not `missing_evidence`: the
captures carry every exit the probe was built to prove; what the reviewer could not verify was
left out of the review packet, not out of the captures.

## Exits proven

| exit | basis | evidence class |
| --- | --- | --- |
| Capture admissibility | all four pods pass `final_h_run_end` (one `run_start`, one `run_end`, committed step rows; the paused pod admits with `result_status paused`); every line of every file is a JSON object; K3 PASS in-image over three pods and locally over all four | MEASURED (analysis Q1, section K); reviewer executed the standalone reader on all four, REVIEWER-REPORTED |
| Lineage | one code commit `046cfb05`, one image digest `88e21c59…`, one attestation, accelerator class and precision record on all four pods; ordered per-row source stream identical sphere vs magnitude; the three magnitude profiles byte-identical, sphere differing only in `law.mode`, `spine.carrier`, `name` | MEASURED (analysis Q2); reviewer recomputed field by field, REVIEWER-REPORTED |
| Recovery and resume | declared pause at global 500 (`request.pause ["prediction", 250]`), sealed progress receipt `5f4a3272…` collected as `resume_artifact`, resumed request binding it through `resume` with a fresh nonce, sha-linked chain 1000 → 750 → 500 → 250, completion receipt on the resumed pod; stitched stopped + resumed rows identical to the uninterrupted magnitude pod on 28 of 31 flattened fields over 1000 rows (timing and allocator fields excepted), evaluations and continuations identical; K1 request and K1 seam PASS in-image and locally, K1 chain PASS in-image | MEASURED (analysis Q4, section K); K1 chain RECORD only, regime-bound; reviewer reproduced the seam on 54 non-timing leaves and the three sha hops, REVIEWER-REPORTED |
| Stochastic rollout | receipt v3 on three pods: `sampling {continuations 4, multinomial_temperature_1, temperature 1.0, seed 42}`, per-continuation seeds and RNG digests, four distinct continuations per receipt, terminal outcomes consistent with observations; K5 PASS | MEASURED (analysis Q5); reviewer replayed every continuation's outcome from its observations, REVIEWER-REPORTED |
| Autocast observability | `run_start.runtime.precision.autocast {enabled true, dtype torch.bfloat16, device_type cuda}` and `declared_precision bf16` on all four pods. This is the recorded configuration the HEpP7g exit asked for; it does not show that bf16 arithmetic took effect on the live step, and that is not claimed | MEASURED (analysis Q2); reviewer confirmed the field on all four, REVIEWER-REPORTED |
| Final leg | all eight stages on the three completed pods, seven admission pairs, shared-memo stages reading 0 evaluations and 0 journals from disk (D5), zero retries, zero stop markers, JSONL sink settled with zero stranded; K4 PASS; controller `terminal_collected`, `workload_verified true`, `termination_verified true`, `error_types []` on the completed arms and `partial_capture` on the paused arm; K2 PASS | MEASURED worker side (analysis Q3, section K); controller records RECORD; reviewer verified the worker side and could not see the controller records, REVIEWER-REPORTED |
| Numerics | 0 nonfinite among 313,305 numeric leaves; steps strictly +1 within phases; timestamps non-decreasing; `max_residual_ratio` below 1 on all 2,250 prediction rows; ESS within [1, k] and `by_sample_count` keys {2, 4, 8} on every evaluation example; zero-sentence rows labelled `structurally_absent`; K6 PASS | MEASURED (analysis Q6); reviewer recomputed every figure, REVIEWER-REPORTED |
| Capture, analysis, review, filing | this filing | files in git |

## Exits open

| exit | why it is open | next evidence that would close it |
| --- | --- | --- |
| Test-only gate, independently verified | The in-image check run PASSes all nine checks (`exit=0`, `report_sha256 2e104321…` equal to the captured file, MEASURED) and the eight regime-independent checks PASS locally over all four pods. The reviewer could not verify this: the packet's report copy was carrier-redacted so its hash could not match the recorded one, and the controller records K2 reads were not in the packet. Both are packaging defects of review round 1, not evidence defects | a second blind round on a packet carrying the host control records for all four arms, the pre-redaction report hash beside the redacted copy, the codec_warmup-00000250 progress receipt, and the local plans with their outputs; round 1 measured about $8.41 of API spend, so this is an operator spend decision |
| Declared measure of the future density per carrier | a docstring question carried from HEpP7g; not an H0 exit | the density's measure written into a schema or docstring |

Findings carried to the gate and H1 authors, none a refusal: continuation length is censored by
`invalid_utf8` at the first invalid sentence and `summary.outcome` is the last continuation's
outcome (analysis Q5); the resumed pod's restore window is 343.7 s longer than a fresh start
and its first seal took 99.887 s, with no row itemising either (Q3); `admission.end` on
shared-memo stages is emitted at stage end so timestamp brackets overstate admission time (Q3);
K1 chain can only be re-executed inside the image (O2); the controller's progress view for
resumed arms is advisory (O1); the HEpP7g capture fails K4 under the current gate's read floor
(section K).

## Evidence chain

| link | value | evidence class |
| --- | --- | --- |
| MLflow-side capture | `results/h0-final-probe-1k-v4-92c9db.dvc`, md5 `ed953d4af265275a9609e7e4a09e2903.dir`, 12 files, 14,096,149 bytes | MEASURED (pointer) |
| Wave 1 host side | `results/h0-final-probe-1k-v4.92c9db.dvc`, md5 `8d9c98013938f623bbb35e870e518aac.dir`, 161 files, 57,337,118 bytes | MEASURED (pointer) |
| Wave 2 host side | `results/h0-final-probe-resume-v4.1755b0.dvc`, md5 `45395630615c9efca60cf87ad9b7bec6.dir`, 123 files, 46,441,594 bytes | MEASURED (pointer) |
| In-image check run | `results/h0-final-probe-checks-1755b0.dvc`, md5 `bfaedab30df9e4ec5a1038e1ff1e9b0c.dir`, 7 files, 18,505 bytes; `report.jsonl` sha256 `2e10432125a1ff3a7cd52941467e2dac5e50eaefb59fbf989e79e7712db0289a`; image index `sha256:02e2582f…`, source `046cfb05…`, regime `3.12.3 2.13.0+cu126 0.3.4` | MEASURED (pointer, hash), RECORD (metadata) |
| Pods | sphere `g8087m4ah3ops2`, magnitude `xm1c6coypgd2by`, stopped `i11npiybn4nwnt`, resumed `m8ootdwxmkuveg`; requests `8eb81425…`, `111640fe…`, `c658b4f0…`, `f679b1a6…` | MEASURED (`run_start` rows, `wave.json`) |
| Image at the time of the run | `ghcr.io/luiz-frias/entropic-substrate-lm@sha256:88e21c59b17b3b3656671c4bb5aba92a3e921b1d5d93249efbd6b03f8acfc537`, code commit `046cfb05ab4b4b835e4819b9d9be71da9e698a00` (v0.6.0) | MEASURED (`run_start` of all four pods) |
| Analysis | `docs/h0/analysis-final-probe-2026-10-26.md`, corrected in the commit that adds this file with dated retractions from the review | file in git |
| Review | blind round 1, reviewer model `claude-opus-5-5`, network-isolated sandbox, 2026-10-26T19:31Z, one round, 426 s, 1,911,284 input and 38,245 output tokens; verdict `tracer_partial`; full file md5 `897b5d48861c7be593ccf47458fa975a` | REVIEWER-REPORTED; usage MEASURED |
| Review archive | `~/.claude/projects/-Users-luizfrias-CursorAI-entropic-substrate-lm/refactor-survey-2026-09-17/h-sprint/h0-review/final-probe-round1/` (review, prompt, packet, manifest, run log, local plans and reports, sealed mapping; operator machine, outside git) | path, not in git |
| Author response and mapping disclosure | `docs/h0/review-response-final-probe-2026-10-26.md` | file in git |

## Scope statement

This is a plumbing gate with no scientific claim. The generated continuations in the receipts
(7 of 8 carrier continuations ending in `invalid_utf8`, 3 finished sentences in total, none of
them meaningful text) are reported as captured content of a 112,233-parameter model after 1000
steps and carry no quality claim. No decoder-quality, predictive or geometric statement follows.
H1 eligibility beyond "the machinery exists" is not implied. The previous verdict's open exits
for stochastic rollout, recovery and resume, and autocast observability are closed by this
evidence; the test-only gate exit is closed by the author's measurement and stays open as an
independently verified exit until a reviewer reads the records that close it.
