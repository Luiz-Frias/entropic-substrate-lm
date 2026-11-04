# FOR-1704 [h1-l4d] Evaluation instruments: prior and mixture proposals with a ratio-of-sums headline, prior/q radius and cosine, latent usage per evaluation, through the extractor to MLflow and telemetry (DG-A re-pin row 4)

Fourth serial W2 slice, after L4c (h1-l4c-remote-pair); arch mode (owns the evaluation-record door); Opus xhigh at the design gate and the implementation; the lane cannot commit member edits, so it stages the patch, the proposed message and the evidence as a DG-A handoff and the conductor lands patch plus `tools/regime_rows.py` re-pin plus the read-old-table row as one commit. Rulings: DH-32 (operator 2026-10-28, aligned with the recs plus the 0.15 and 0.35 ESS/k readings), DH-33 and DH-48 (operator 2026-11-02, FOR-1716 as this lane's sub-issue; section below) (add the instrument and metrics capture to the MLflow sink and telemetry; verify what exists first). Consumer: DH-34's two confirming quote pods after W2 merges and an interim re-pin read every new key from the captured records alone. Every anchor below was re-measured at main e3d1524b (v0.10.0) on 2026-10-31; none of the anchored files changed between v0.9.1 and v0.10.0.

## Touch-set

- members (DG-A, move the five checkpoint digest rows at tests/test_identity_epoch.py:904-931): src/entropic_substrate/core/sentence/scoring.py (`importance_statistics` :22, `log_estimate` and ESS at :49-50 with the `[:k]` slice applied by the caller at :142; `ImportanceEvaluation` :58-75 with `meaning` at :67 and the eight listed fields at :68-75; `importance_evaluate` :79, proposal drawn from `model.codec.posterior(complete)` at :110, `log_weights = decoder_log + prior_log - proposal_log` at :132); src/entropic_substrate/ladder/run/execution.py (`evaluate` :305, fork_rng :330, step-keyed seed :340, `evaluation_batches` :341, the `examples[]` block :364-378 with `source_bytes` :368, `boundary_symbols` :369, `by_sample_count` :370-373 and four log-prob arrays :374-377); src/entropic_substrate/ladder/run/trace_capture.py (`_SCHEMAS`, 17 entries at main 8c1f0c13, RETRACTED 2026-11-04: "14 entries" and the :23-40/:30/:89-90 anchors were stale; locate by symbol; `h-predictive-evaluation-v1` is one entry); src/entropic_substrate/ladder/run/object_worker.py (member; DH-51: both `run_start` write sites gain `profile` = the `HProfile` manifest); src/entropic_substrate/ladder/run/evaluation_records.py (member; `recovery_record` admits EXACTLY the v1 key sets: 8-key envelope, 9 keys per example, 4 keys per row, exact schema and meaning strings, measured 2026-11-03 at 3e6f6ecb: a superset row is refused) and src/entropic_substrate/ladder/run/runner.py (member; exact v1 string at :398). RETRACTED 2026-11-03: the earlier reading that the schema bump is conditional. Any added key or row is refused by the reader today, so "keep v1" is not an arm; the L4d design gate rules the door: `h-predictive-evaluation-v2` with v1 read-old for the H0 fixtures, or a separate record schema beside v1. Either arm edits `evaluation_records` (the five checkpoint rows move; the 54-member count does not). Non-member exact-string v1 gates to update with it: ladder/analyze/h_aggregate.py:83, ladder/analyze/h_final_probe_checks.py:102, ladder/capture/reader/h_telemetry.py, ladder/run/h_metric_extractor.py:120.
- non-members: src/entropic_substrate/ladder/run/h_metric_extractor.py (not a reader member; keys at :195-202: `eval/log_marginal_k{k}`, `eval/ess_k{k}`, `eval/gap_rows_k{k}`, `eval/wall_seconds`, `eval/cuda_seconds`; wired by `worker_command.py:22` and used at :232); src/entropic_substrate/ladder/analyze/h_elm_projection.py (`project_elm`: ESS guard :174-175 admits ESS = 1 within `_ESS_SLACK` 1e-9 at :59; mean of per-row `log_estimate / source_bytes` at :176; `eval/ess_ratio_median_k{k}` at :200 for every k, headline k = 64 at :52); src/entropic_substrate/ladder/capture/adapters/mlflow.py (`MlflowSink` :279, `log` :542) only if the record shape needs it; src/entropic_substrate/ladder/analyze/h_aggregate.py (`_check_step_scalars`, the null `future_nll_per_dimension` admission; DH-48, FOR-1716, non-member); the strict adapter mlflow_strict.py (`_RECORD_FIELDS` :64, `_has_exact_keys` :116) must still hold.
- tests: tests/test_sentence_scoring.py; tests/test_h_pair_request.py (the P5-15 measurement pin flips to load-through, DH-48); tests/test_h_worker_command_observability.py (the exact-key `run_start` pin moves with DH-51); tests/test_h_metric_routing.py (the extractor's eval keys are tested here, :249-253; there is no tests/test_h_metric_extractor.py, create it test-first or extend routing); tests/test_h_elm_projection.py; tests/test_mlflow_async_sink.py; tests/test_identity_epoch.py and docs/identity/read-old-table.md (conductor re-pin).
- NOT touched: the ruled 0.25 gate; the pilot's future_weight (core/sentence/profile.py:251-254 requires 0 at H1); tests/test_observability_conformance.py frozen counts (PRINT_ALLOWLIST 10 at :95, 11 mapping rows at :176) unless a new entrypoint or print is added.

## Why (FOR-1703 recon, verdict B)

With future_weight = 0 the prior head drifts from the encoder posterior q, the q-proposal importance weights go one-hot (log_w sd 107 to 157 nats across draws) and ESS/k reads about 1/k by construction, so the ruled 0.25 gate would exclude every H1 block; sampling from the prior on the same checkpoint gives ESS 64/64. The per-byte headline is a mean of per-row ratios, so a 2-byte sentence supplied 92.6 percent of one reading. The dev-width codec is near latent-blind after the 250 warm-up steps (2.69 vs 2.71 nats per byte with a null latent). Nothing in the evaluation record or in MLflow exposes any of this today.

## Scope

1. Second and third proposals in the evaluation record: for every example and k, beside the q-proposal rows, rows drawn from the PRIOR (`prediction.law`) and from a q plus prior mixture (equal weights; the mixture log density is `log q_mix = logsumexp(log q, log p) - log 2`, computed inside an FP32 or FP64 island, never as a log of a sum of exponentials, since the measured log-weight spreads of 107 to 157 nats underflow a naive sum), each with its own importance rows; `meaning` extended or a `proposal` field added so the record names the proposal; the q rows stay byte-identical in shape so every existing reader keeps its names.
2. Diagnostics per example: prior log-radius mean, q log-radius mean, cos(`prediction.law.direction`, the q proposal's `direction`) with both operands the law and proposal mean directions, unit norm by construction (`core/sentence/law.py` requires it), so a zero norm is unreachable and no fallback exists; log-weight spread (sd across draws) per proposal. In sphere-law cells (`law.mode` sphere, half of the H1 grid) `log_radius_mean` is None, so both log-radius keys are ABSENT for those cells, never invented.
3. Latent usage per example (DH-33; operand ruled DH-52, operator 2026-11-04): `latent_usage_nats_per_byte = null_nats_per_byte - q_nats_per_byte`, the decoder nats per byte with a near-zero latent (radius 1e-4) minus the same with the q POSTERIOR MEAN vector on the same bytes (deterministic per evaluation; never a draw average). Structure under D-PY-1 (a named second option earns the port): the estimator is a `Protocol` port with a closed `StrEnum` {`posterior_mean`, `draw_average`}, implementations injected at the composition root, the chosen operand recorded in the run record and carried as an operand segment in the emitted key; `posterior_mean` is the only implementation this lane builds and the registered instrument; `draw_average` is the anticipated second implementation (FOR-1721, W3, measurement-gated on the DH-34 pods) and is NOT built here; the FOR-1703 recon's 2.69 vs 2.71 figure is a draw-average reading, not comparable to this key without re-measurement under the same regime; signed, never clamped (a clamp would be fail-open on a measurement path); a positive value means the q latent helps.
4. Extractor keys, k-suffixed where applicable (FOR-1664): `eval/ess_prior_k{k}`, `eval/ess_mixture_k{k}`, `eval/log_marginal_prior_k{k}`, `eval/log_marginal_mixture_k{k}`, `eval/log_marginal_per_byte_ratio_of_sums_k{k}` (q), `eval/log_marginal_per_byte_ratio_of_sums_prior_k{k}`, `eval/log_marginal_per_byte_ratio_of_sums_mixture_k{k}`, `eval/prior_log_radius_mean`, `eval/q_log_radius_mean`, `eval/prior_q_cosine_mean`, `eval/log_weight_sd_k{k}` (q), `eval/log_weight_sd_prior_k{k}`, `eval/log_weight_sd_mixture_k{k}`, `eval/latent_usage_nats_per_byte` (the q keys keep the bare names and the prior and mixture keys carry the proposal segment, because the extractor's flat metrics dict keeps only the last writer of a name); the ratio of sums for a proposal sums `log_estimate` and `source_bytes` over the same finite rows of THAT proposal, reports `gap_rows` beside it, and leaves the key absent (never 0) when no row of the proposal is finite, following `h_elm_projection`'s gap-row convention; all through the existing extractor into the MetricSink port (`ladder/capture/ports/metric_sink.py`, class :40, `log` :43) and the MLflow sink unchanged, and logged as loguru telemetry events on the worker in the existing eval event family.
5. Projection: `project_elm` reports the ratio-of-sums headline beside the mean of ratios and selects the proposal by a parameter (default q); L3's adapter already reads the gate metric key as a parameter (merged in #416).

## RED-first tests

tests/test_sentence_scoring.py: prior and mixture rows present with the exact mixture density; on a fixture where the prior equals q all three proposals agree; q rows unchanged byte for byte; latent usage non-negative on a trained fixture and zero when the decoder ignores the latent by construction. Extractor tests (routing file or a new file): every new key emitted, k-suffixed, absent when the record lacks the rows, never invented. tests/test_h_elm_projection.py: ratio of sums and mean of ratios differ on a two-row fixture with unequal byte counts; the proposal selector; old keys unchanged. Numerics and sign cases: the mixture density on two components more than 1000 nats apart equals the larger component minus log 2 (a naive sum underflows); latent usage positive when q helps, negative and preserved when q hurts, exactly 0 for a decoder that ignores the latent; the log-radius keys absent on a sphere-law fixture. tests/test_mlflow_async_sink.py extended with the new keys. The H0 fixtures (results/h0-plumbing-1k-v3-HEpP7g, results/h0-final-probe-1k-v4-92c9db) carry `h-predictive-evaluation-v1` events and suit the reader-side tests here.

## Guard teeth (mutation, each RED)

drop the mixture normalization; replace the logsumexp with a log of a sum (RED on the extreme-separation case); compute the cosine on raw states instead of the law and proposal directions; count boundary symbols in the ratio of sums; include a gap row's bytes in a proposal's denominator; emit a bare (non-k-suffixed) `eval/ess_prior`; emit the prior ratio of sums under the q key name (RED: the q value is overwritten); clamp a negative latent usage to zero; read the null latent with the q radius.

## DH-48: loader admission of the null future term (FOR-1716, sub-issue; ruled 2026-11-02)

Pre-existing defect exposed by L4b's pair tracer, measured at 5449e2b5: an `H1` profile requires
`optimization.future_weight == 0` (core/sentence/profile.py:251-254), the objective then writes a
null `future_nll_per_dimension` on every prediction row (core/sentence/objective.py:96-101), and
`ladder/analyze/h_aggregate.load_h_run` admits that null only under `phase == codec_warmup`
(`_check_step_scalars`), so it refuses BOTH completed H1 cells at their first prediction row.
`tests/test_h_pair_request.py::test_a_completed_h1_cells_files_are_measured_against_the_analysis_reader`
(P5-15) pins the refusal as a measurement today.

Ruling DH-48, option (a): `load_h_run` admits a null `future_nll_per_dimension` on a prediction row
only when the run's RECORDED `optimization.future_weight == 0`, read from the `run_start` profile
manifest in the log beside the metrics; never keyed on the null itself and never on the stage string
alone. Every other run loads byte-identically; a null under a non-zero weight keeps the existing
refusal message byte-identical; a warm-up row stays admitted as today. Options (b) worker writes
`0.0` (changes H1 metric bytes, member edit) and (c) a `future_term=` loader argument (pushes the
decision to every caller) are REJECTED. `h_aggregate.py` is a non-member: no re-pin row.

- RED first: a completed H1 cell's (metrics, log) pair from the L4b pair tracer loads through
  `load_h_run` with `future_nll_per_dimension` summarized as absent (ImportError or `TelemetryGap`
  at the parent tip); the P5-15 pin FLIPS in the same commit from "refused at the first prediction
  row" to "loads through", with the old expectation retracted in its docstring (house rule 4).
- Teeth, each RED one at a time and restored byte-identical: key the admission on the null instead
  of the recorded weight (RED on the non-zero-weight fixture); drop the recorded-weight read (RED);
  admit the null on a run whose log carries no `run_start` profile manifest (RED).
- Linear: FOR-1716 moves with this lane; its comment thread carries the measurement and the ruling.

RETRACTED 2026-11-04 (L4d design gate, re-verified by the conductor at main 8c1f0c13): "read from the
`run_start` profile manifest in the log beside the metrics" named a record that does not exist.
`run_start` is written at both `object_worker` sites with exactly eight keys (`schema`, `event`,
`request_sha256`, `image`, `code_commit`, `profile_identity`, `run_class`, `runtime`), pinned by
equality in `tests/test_h_worker_command_observability.py`; `worker.start` carries three keys; no
capture carries `future_weight`. "Non-member, no re-pin row" was wrong too: the writer is a member.

## DH-51: where the recorded future weight lives (ruled 2026-11-04, operator: R1)

`run_start` gains `profile` = the `HProfile` manifest (the canonical form whose `content_identity`
is `profile_identity`) at BOTH `object_worker` write sites; the schema name `h-run-lifecycle-v1` is
unchanged; this is a dated additive widening with absence as the declared default (doctrine:
additive evolution first; persisted artifacts read-old; sealed evidence never migrates, it replays
at its epoch). `load_h_run` admits a null `future_nll_per_dimension` on a prediction row only when
`run_start.profile` is present, `content_identity(canonical json of it) == run_start.profile_identity`
(identity derived, never bytes; the same construction `profile.py` uses), and
`profile.optimization.future_weight == 0`. A missing or mismatched manifest keeps today's refusal
byte-identical, so the H0 fixtures and every existing capture load exactly as now. The admission is
builder-class: one validation of the whole invariant set, then a frozen admission; no fallback.
`object_worker` joins the DG-A member patch; the exact-key `run_start` pin moves; the read-old-table
gains the widening row. R2 (the manifest on `worker.start`) and R3 (split FOR-1716 out) are
rejected.

- RED first: the completed H1 cell pair from the pair tracer loads through `load_h_run` once its
  `run_start` carries the manifest; the same files with the manifest stripped are refused with
  today's message byte-identical; a manifest whose identity differs from `profile_identity` is
  refused by name; a null under a recorded non-zero weight is refused with the existing message.
- Teeth, each RED and restored byte-identical: key the admission on the null; drop the identity
  check; drop the recorded-weight read; admit on a missing manifest; forge a manifest with a
  mismatched identity.

## Design-gate constraints (pass one, 2026-11-04)

The evaluation-record door is ruled `h-predictive-evaluation-v2` with v1 read-old for the H0
fixtures (constraint 1); the remaining fifteen constraints of the gate's first pass bind as written
in the run journal (wf_c06c35f9-698). Corrections to this brief measured by the gate: `_SCHEMAS`
has 17 entries; `ladder/capture/reader/h_telemetry.py` carries a docstring note, not a code gate,
so the "four non-member exact-string v1 gates" are three code gates plus one docstring to amend.
Cost risk relayed, unmeasured: evaluation compute and record bytes roughly triple against v1; both
are measured on the DH-34 pods against the 3 h deadline and `non_checkpoint_bytes`, never
estimated into the registration.

## Handoff

DG-A as in L4a: members scoring.py, execution.py, trace_capture.py (the schema bumps to v2), evaluation_records.py, runner.py and object_worker.py (DH-51); a 55th reader member is NOT expected here (h_metric_extractor stays a non-member), so the re-pin is the five checkpoint rows only (`_CHECKPOINT_REGIME_TABLE` and `_FROZEN_CHECKPOINT_READER_MODULES` in tests/test_identity_epoch.py, located by name; the set is 54 since L4a added `codec_handoff`; RETRACTED 2026-11-03: "stays at 53" and the :827-881 / :904-931 anchors, measured stale at 3e6f6ecb). The attempt record `h-evaluation-attempt-v1` (execution.py:316) is not in `_SCHEMAS` today; leave it unless the slice captures it.

## Out of scope

changing the ruled 0.25 gate; the pilot's future_weight; the interim image re-pin and the DH-34 quote re-run pods (separate GO); the non_checkpoint_bytes dev-width measurement (taken on the DH-34 pods, sealed by L8).
