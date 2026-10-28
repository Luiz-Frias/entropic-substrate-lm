# FOR-1691 [h1-l3] Add the H1 observation adapter to ladder/analyze (H captures to PilotContrast, last-5 endpoint, ESS gate, whole-block exclusion, plan_seed_count)

Source of truth: Linear FOR-1691 (project P-FOR-44). In-repo copy for the W1 lane (preflight corrections of 2026-10-28 applied on the Linear issue; anchors measured at main 45330219). The Linear issue text is authoritative where the two differ.

Wave W1 (K=3, non-member). Seat Opus xhigh. Class: feature.

## Touch-set (non-member)

- NEW src/entropic_substrate/ladder/analyze/h1_observations.py: Mapping[cell, HRun] per block to StatisticalRuns (arm = cell label, seed = profile.seed, run_class pilot, family and construction digests from the DH-14 fixture rule; one StatisticalEvaluation(global_step, cumulative canonical bytes, (("eval/log_marginal_per_byte_k64", v),)) per evaluation), the H PilotContrasts for the three deciding contrasts (sentence main, spine main, interaction), the per-cell ESS gate (per-row median ESS/k at k=64 >= 0.25, parameter with the ruled default) dropping the whole block with a reported reason, independent_warmup mapped to exclusion, then plan_seed_count and dual_confidence_intervals unchanged
- NEW tests/test_h1_observations.py (fixtures: results/h0-final-probe-1k-v4-92c9db captures as k<=8 diagnostics refused by plan_seed_count for the right reasons; a synthetic 8-block fixture that plans n)
- optional NEW scripts/h1_pilot_sizing.py
- NOT touched: ladder/analyze/ladder_registration.py (adapter sits at the RungRunReader seam, ladder_analysis.py:77), h_report.py (the H0 check suite imports it, h_final_probe_checks.py:66), h_aggregate.py (W2 owns its edits), seed_sizing.py

## Why

paired_contrasts refuses H designs by design (ladder_registration.py:275) and _d1_runs refuses H at :341; the statistical machinery is complete (seed_sizing.py PairedSeed/PilotContrast :17-35, plan_seed_count :184, dual_confidence_intervals :251, PilotVarianceUnknown; read_h_statistical_design :150; h_aggregate.load_h_run; h_elm_projection.project_elm). last_k_mean (d1_factorial.py:407-427) and the whole-block exclusion (:177-294, :536-594) are D1Arm-typed and get ported.

## RED-first tests

endpoint = mean over the last 5 evaluations of eval/log_marginal_per_byte_k64; whole-block exclusion ported to string arms with one pairing_id per seed; ESS gate at 0.25 on the median; independent_warmup -> exclusion; exposure = cumulative canonical bytes (step rows carry source_bytes, execution.py:262-271; the adapter sums it); exposure mismatch refused; non-finite endpoint refused; plan_seed_count from the synthetic fixture; PilotVarianceUnknown on zero variance.

## Guard teeth (mutation, each RED)

last evaluation instead of last-5; drop one cell instead of the block; mean instead of median at the gate; sign flip of a contrast; threshold 0.0.

## Measured input (quote v4, 2026-10-28)

at dev widths after 1000 steps eval/ess_k64 was 1.00, 1.11, 1.02 (ESS/k 0.016): the 0.25 gate would exclude such blocks; the adapter must report the reason, never silently drop.

## Out of scope

the sensitivity-map reporter (W3 L7), h_report wiring, any analysis of real pilot data.
