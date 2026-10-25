# H1 preregistration skeleton: preservation 2×2

**DRAFT / NOT SEALED.** Drafted 2026-10-23 against base `17aaf1826eb07c5bcb79f6c3670afc9e88dd4db9`
(main, v0.5.0 plus the python 3.12.15 pin). This file seals nothing. It lays out every field that
`docs/hierarchical-latent-predictor.plan.md` requires the H1 registration to seal, so the operator
and the conductor can fill them together. No run, pod or spend is authorized by it.

Evidence labels used on every row: **MEASURED** (file and computation named), **ARITHMETIC** (pure
computation, no data), **PROPOSAL** (for the operator to rule), **PLAN** (quoted or paraphrased
binding text), **BLANK** (filled later, by whom stated).

## 1. Status

| field | value | label |
| --- | --- | --- |
| Status | DRAFT skeleton, not a registration | PLAN |
| Base SHA | `17aaf1826eb07c5bcb79f6c3670afc9e88dd4db9` | MEASURED, `git log -1` |
| Sealed so far | nothing | PLAN |
| Binding sources | plan sections "Independent preservation controls", "Effective directional/magnitude law", "Sequential experimental route", "Evidence and statistical contract"; FOR-1664 brief `.claude/workflows/briefs/for-1664-ess-k-curve.md`; H0 verdict `docs/h0/verdict-hepp7g-2026-10-22.md` | PLAN |
| Code already present for H1 | `ladder/register/h_preservation.py` (`H1PreservationBlock`: four profiles, one paired block, axes keyed `(law.mode, spine.carrier)`, `factorial_terms`, `codec_sharing_groups`); `ladder/analyze/seed_sizing.py` (`plan_seed_count`, `dual_confidence_intervals`); `HProfile` refuses `stage == "H1"` unless `consumer == "isotropic"` and `future_weight == 0` | MEASURED, read at base |
| Format precedent | `docs/e1-preregistration.md`, `docs/e1-5c-preregistration.md` (hypothesis, design, endpoints, estimator, decision rule, pilots, exclusion, integrity, amendments) | PLAN |

Sealing sequence (each step is its own commit or run; no step is skipped or merged):

| # | step | output | who |
| --- | --- | --- | --- |
| 1 | Fill this skeleton's PROPOSAL and BLANK fields with the operator | edits to this file | operator + conductor |
| 2 | Commit the filled skeleton | commit on main | conductor |
| 3 | H0 final probe closes the open H0 exits (stochastic rollout, recovery/resume, test-only gate) | `tracer_complete` or not | operator GO |
| 4 | Excluded pilot: 8 complete paired 2×2 blocks per contrast family, at the registered regime | captured pilot data, never confirmatory | operator GO and quote |
| 5 | Dated amendment sealing n, margins, bands from the pilot variance via `plan_seed_count` | amendment commit | operator + conductor |
| 6 | Confirmation runs, capture, fresh-session analysis, arm-blind review, filed verdict | verdict file | per plan |

An unaffordable powered n at step 5, or a missing margin, blocks execution (PLAN, "Replication and
registration").

## 2. Question and constructions

H1 question (PLAN, H1 row): what do normalization and its interaction lose? Required exit: paired
2×2 controls, independent detail checks, numerical and observable predictive evidence. H1 uses
isotropic dynamics and no auxiliary future-law objective. It combines fixed-state interventions and
matched diagnostic decoders with paired retraining. Recoverability, adaptation and usefulness are
different tests. Every construction meeting registered eligibility is retained; if both
normalization choices fail, H1 pauses rather than scaling.

| cell id | sentence representation (`law.mode`) | spine carrier (`spine.carrier`) | plan definition | run in H0? |
| --- | --- | --- | --- | --- |
| MM | `magnitude`: continuous vector retaining magnitude | `magnitude`: residual magnitude retained, normalized sublayer views allowed | magnitude-preserving control on both axes | yes, arm `a163161f` (profile identity `ed98fc2e...`) |
| SM | `sphere`: unit direction | `magnitude` | sphere sentence, magnitude carrier | **never run** |
| MS | `magnitude` | `sphere`: unit retraction | magnitude sentence, sphere carrier | **never run** |
| SS | `sphere` | `sphere` | spherical candidate on both axes | yes, arm `988d8388` (profile identity `ce3740e5...`) |

How the profile fields express the axes (MEASURED, read at base):

| field | where it acts | what it does |
| --- | --- | --- |
| `law.mode` | `core/sentence/model.py:88` `apply_interface(encoded, self.codec.law_spec.mode)`; `numerics.py:76-90` | `sphere` returns the unit direction; `magnitude` keeps the FP32/FP64 tensor, no renormalization |
| `spine.carrier` | `core/sentence/spine.py:78` (residual retraction) and `:137` (initial states) | `sphere` retracts to the unit direction; `magnitude` returns `current + step` with no final normalization |
| Cell ordering | `ladder/register/h_preservation.py` `_AXES = (MM, SM, MS, SS)` keyed `(law.mode, spine.carrier)` | `factorial_terms` weights: sentence `(-0.5, 0.5, -0.5, 0.5)`, spine `(-0.5, -0.5, 0.5, 0.5)`, interaction `(1, -1, -1, 1)`; sign is sphere minus magnitude, "positive means increase, not improvement" |
| Codec sharing | `codec_sharing_groups` = (MM, MS) and (SM, SS) | same sentence interface shares a codec artifact within a block |

Consequence (MEASURED from the H0 verdict lineage row): the two HEpP7g profiles differ only in
`law.mode`, `spine.carrier` and `name`. H0 therefore ran the **diagonal** cells MM and SS. Their
difference confounds both axes and their interaction; HEpP7g gives no single-axis scale.

Open construction check (not a claim either way): the plan says the existing D encoder does not
satisfy the no-hidden-final-normalization condition unchanged. At base, `core/sentence/codec.py:137`
carries the comment "no final normalization of the carrier", and the magnitude arm's measured
sentence norms (section 3) are not 1. Neither proves the absence of a hidden RMS overwrite on every
path. See operator question Q7.

## 3. Estimands

### 3a. Candidate estimands

Direction is two-sided unless the operator rules otherwise; a discovery posture is valid. Every
"deciding or exploratory" entry is a PROPOSAL. Contrast weights are `factorial_terms` above. No norm
ratio, correlation or rank is ever converted into a percentage of information retained (PLAN).

| id | definition | units | direction | deciding or exploratory (PROPOSAL) | measured scale from HEpP7g | candidate band for the operator (basis) | 3-sigma band from pilot (8 paired blocks) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E-LM | importance-weighted log-marginal at the registered headline `eval/log_marginal_k64`, divided by the row's canonical byte count; the captured evaluation row carries `source_bytes` per row (MEASURED: `[155, 143, 185, 116]` in the sphere arm's first evaluation), so the per-byte value is computable from captures, but the current extractor projection `eval/log_marginal_k{k}` is per sentence and a per-byte analysis-side projection is still to be written | nats per canonical byte, boundary-augmented task, named as an estimate (Jensen lower bound in expectation) | two-sided | **deciding** (primary observable usefulness) | k=64 not recorded; k=2, 4 per evaluation in 3c, in nats per held-out sentence, not per byte | none proposable: no k=64 data, no seed variance; the 0.01 nats/byte E reference does not transfer (PLAN) | filled by pilot |
| E-DCE | final-window prior-sample decoder CE, `prediction/decoder_ce`, last-W committed steps mean | nats per decoder symbol, boundary-inclusive | two-sided | **deciding** (training-side usefulness), or exploratory if E-LM alone decides (Q2) | 3b row `prediction/decoder_ce` | the within-run step sd (0.18 to 0.19) is the only scale; it is not a seed band | filled by pilot |
| E-REC | oracle reconstruction CE at end of codec warm-up, `codec_warmup/decoder_ce` | nats per decoder symbol | two-sided | exploratory; recoverability, sentence axis only (codec is shared within same-interface pairs) | 3b row `codec_warmup/decoder_ce` | none (n=1 seed) | filled by pilot |
| E-ESS | ESS against k over `(2, 4, 8, 16, 32, 64)` plus the 256 diagnostic, per cell | effective samples; also ESS/k | not a contrast | **instrument admissibility gate** (section 8), never an effect estimand | k=2, 4 only, 3c; ESS 1.02 to 1.22, bounded above by k | admissibility threshold is a PROPOSAL (Q4) | filled by pilot |
| E-SPAN | span-aware predictive score gain at lags {2, 4, 16, 64}, signed, against uniform and horizon-matched shuffled controls | nats per target byte (signed gain) | two-sided | exploratory; the plan keeps it diagnostic unless a registration promotes it (Q3) | not run in HEpP7g; instrument exists (section 8) | none | filled by pilot |
| E-NORM | sentence-representation and spine-state norm distributions (min, median, max per step) | dimensionless norm | descriptive | exploratory, descriptive only; sphere cells are 1 by construction | 3b rows `sentences_norm_*`, `states_norm_*`, `sample_norm_median` | none; descriptive | filled by pilot |
| E-GRAD | per-component gradient norms and statuses (decoder, spine, encoder in warm-up) | gradient norm | descriptive | exploratory; structural participation check (missing or wrong-owner gradients are defects, PLAN) | 3b rows `grad_norm/*`, `aggregate_norm` | none | filled by pilot |
| E-ANG | reconstruction and angular error of the sentence representation around pooling, normalization, sampling, precision conversion | radians or nats, to be fixed | two-sided | BLANK (Q3) | not recorded by the current worker; instrument needed. Closest code: `ladder/analyze/embedding_geometry.py:225 angular_drift` (E-series embedding tables, not the H path) | none | filled by pilot |
| E-RANK | covariance, effective rank, conditioning of sentence representations and spine states | dimensionless | two-sided | BLANK | not recorded by the current worker; instrument needed. Closest code: `effective_rank` in D1 telemetry (`ladder/capture/records/d1_telemetry_measurements.py:41`), not wired to H | none | filled by pilot |
| E-SAT | saturation and zero counts | counts | descriptive | exploratory | zero counts recorded: `zero_count` per part and `zero_views` = 0 on all 750 prediction rows per arm (MEASURED); saturation proper: not recorded, instrument needed, no site found | none | filled by pilot |
| E-DTYPE | actual computation dtype and autocast state | categorical | descriptive | admissibility check | mechanism dtype `torch.float32` on every prediction row both arms (MEASURED); autocast-enabled not recorded (FOR-1672) | none | n/a |
| E-DETAIL | detail preservation: entities, numbers, negation, relations, order | to be fixed | two-sided | BLANK (plan requires "independent detail checks" for the H1 exit) | not recorded; no instrument found in the H path (grep for entity/negation found only corpus and boundary code) | none | filled by pilot |
| E-SHADOW | shadow versus emitted state | to be fixed | two-sided | BLANK | not recorded: `core/sentence/rollout.py` computes `shadow` (line 21) but no metrics row emits it; instrument needed | none | filled by pilot |
| E-PERT | matched perturbation effects on output laws and signed predictive scores (fixed-state interventions, matched diagnostic decoders) | nats | two-sided | BLANK (plan names this as H1's intervention leg) | not recorded; no intervention instrument found in the H path | none | filled by pilot |

### 3b. Measured scale from HEpP7g: within-run step dispersion

**Label on every row: WITHIN-RUN STEP DISPERSION, n=1 seed, NOT a seed band.** Regime caveat on
every row: HEpP7g ran consumer `fisher` with the future-law objective active (`future_nll_per_dimension`
populated, `grad_norm/future_law` measured). H1 forbids both. These numbers come from the wrong n
AND the wrong regime, and only the two diagonal cells; the plan forbids transferring tiny-run
variance. They show orders of magnitude and distribution shape, nothing else.

Computation (MEASURED): `hepp7g_scale.py` kept outside the repo (operator scratchpad), run with the
main checkout's venv as `python -I`. Admission is `ladder.analyze.h_aggregate.load_h_run` on
`results/h0-plumbing-1k-v3-HEpP7g/h-worker-<identity>-<suffix>-{metrics,worker-log}.jsonl` (DVC
pointer md5 `2dc65e2bdf918d43858c6b2d92a63c58.dir`); scalar series come from the reader's
projection, mechanism and gradient fields from the same admitted rows. Statistics: sample mean,
sample sd (n-1), mean ± 3 sd, empirical 0.135th and 99.865th percentiles (linear interpolation,
the 3-sigma-equivalent coverage, shows non-normal tails), and the last-50-step mean. The bracket is
an iid percentile bootstrap (20,000 resamples, seed 20261023) of the last-50 mean at 99.73%
coverage. It is an **exploratory bracket only**: it resamples autocorrelated training steps as if
independent, and under positive serial correlation iid resampling can understate the variance, so
it bounds nothing in either direction and is not a confidence interval of anything H1 decides. A
block bootstrap or a HAC estimator would be required before any bound is claimed. (Corrected
2026-10-23, PR #380 review: the first draft called this bracket a lower bound on within-run noise,
which autocorrelation alone does not establish.) With n = 250 or
750, the 0.135th and 99.865th percentiles are interpolated between the two most extreme steps.
Arms: sphere = SS (`988d8388`), magnitude = MM (`a163161f`). Warm-up has 250 steps, prediction 750.

Rows that survive into H1 (still Fisher-shaped updates in HEpP7g):

| key | meaning | arm | n | mean | sd | mean - 3 sd | mean + 3 sd | p0.135 | p99.865 | last-50 mean | last-50 iid bootstrap, 99.73% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prediction/decoder_ce` | prior-sample decoder CE, nats per decoder symbol (boundary-inclusive) | sphere | 750 | 2.742 | 0.1809 | 2.199 | 3.285 | 2.45 | 4.222 | 2.689 | [2.635, 2.754] |
| `prediction/decoder_ce` | prior-sample decoder CE, nats per decoder symbol (boundary-inclusive) | magnitude | 750 | 2.754 | 0.1921 | 2.177 | 3.33 | 2.449 | 4.413 | 2.694 | [2.639, 2.759] |
| `prediction/objective` | training objective (decoder CE plus future-law term in H0) | sphere | 750 | 2.836 | 0.3486 | 1.791 | 3.882 | 2.334 | 4.647 | 2.715 | [2.618, 2.831] |
| `prediction/objective` | training objective (decoder CE plus future-law term in H0) | magnitude | 750 | 2.806 | 0.3922 | 1.63 | 3.983 | 2.34 | 5.063 | 2.706 | [2.598, 2.833] |
| `prediction/aggregate_norm` | pre-clip aggregate gradient norm | sphere | 750 | 1.165 | 0.4857 | -0.292 | 2.622 | 0.664 | 3.741 | 0.9529 | [0.874, 1.051] |
| `prediction/aggregate_norm` | pre-clip aggregate gradient norm | magnitude | 750 | 1.092 | 0.5382 | -0.5229 | 2.707 | 0.5581 | 4.787 | 0.9009 | [0.812, 1.017] |
| `prediction/grad_norm/decoder` | decoder gradient norm | sphere | 750 | 0.759 | 0.1749 | 0.2345 | 1.284 | 0.4662 | 1.999 | 0.7167 | [0.6631, 0.7819] |
| `prediction/grad_norm/decoder` | decoder gradient norm | magnitude | 750 | 0.7725 | 0.1869 | 0.2118 | 1.333 | 0.468 | 1.926 | 0.7217 | [0.6675, 0.7848] |
| `prediction/grad_norm/spine` | spine gradient norm | sphere | 750 | 0.3154 | 0.2367 | -0.3946 | 1.025 | 0.09261 | 1.654 | 0.2483 | [0.1981, 0.3161] |
| `prediction/grad_norm/spine` | spine gradient norm | magnitude | 750 | 0.2886 | 0.3105 | -0.6427 | 1.22 | 0.07114 | 2.284 | 0.2137 | [0.1543, 0.3144] |
| `prediction/sentences_norm_min` | per-step min of history sentence-representation norms | sphere | 732 | 1 | 1.722e-08 | 1 | 1 | 1 | 1 | 1 | [1, 1] |
| `prediction/sentences_norm_min` | per-step min of history sentence-representation norms | magnitude | 732 | 1.164 | 0.1783 | 0.6295 | 1.699 | 0.7547 | 1.705 | 1.128 | [1.065, 1.19] |
| `prediction/sentences_norm_median` | per-step median of history sentence-representation norms | sphere | 732 | 1 | 9.474e-09 | 1 | 1 | 1 | 1 | 1 | [1, 1] |
| `prediction/sentences_norm_median` | per-step median of history sentence-representation norms | magnitude | 732 | 1.679 | 0.0927 | 1.401 | 1.957 | 1.359 | 1.892 | 1.647 | [1.608, 1.687] |
| `prediction/sentences_norm_max` | per-step max of history sentence-representation norms | sphere | 732 | 1 | 1.732e-08 | 1 | 1 | 1 | 1 | 1 | [1, 1] |
| `prediction/sentences_norm_max` | per-step max of history sentence-representation norms | magnitude | 732 | 2.097 | 0.1827 | 1.549 | 2.645 | 1.609 | 3.232 | 2.077 | [2.003, 2.16] |
| `prediction/states_norm_min` | per-step min of spine state norms | sphere | 750 | 1 | 2.049e-08 | 1 | 1 | 1 | 1 | 1 | [1, 1] |
| `prediction/states_norm_min` | per-step min of spine state norms | magnitude | 750 | 1.359 | 0.06515 | 1.163 | 1.554 | 0.9677 | 1.425 | 1.419 | [1.412, 1.423] |
| `prediction/states_norm_median` | per-step median of spine state norms | sphere | 750 | 1 | 1.289e-08 | 1 | 1 | 1 | 1 | 1 | [1, 1] |
| `prediction/states_norm_median` | per-step median of spine state norms | magnitude | 750 | 2.29 | 0.2269 | 1.609 | 2.971 | 1.325 | 2.695 | 2.343 | [2.235, 2.422] |
| `prediction/states_norm_max` | per-step max of spine state norms | sphere | 750 | 1 | 2.021e-08 | 1 | 1 | 1 | 1 | 1 | [1, 1] |
| `prediction/states_norm_max` | per-step max of spine state norms | magnitude | 750 | 3.088 | 0.3596 | 2.009 | 4.167 | 1.325 | 3.917 | 3.203 | [3, 3.34] |
| `prediction/sample_norm_median` | per-step median of prior-sample norms | sphere | 750 | 1 | 1.811e-08 | 1 | 1 | 1 | 1 | 1 | [1, 1] |
| `prediction/sample_norm_median` | per-step median of prior-sample norms | magnitude | 750 | 1.612 | 0.1655 | 1.116 | 2.109 | 0.919 | 2.02 | 1.621 | [1.572, 1.668] |
| `codec_warmup/decoder_ce` | oracle reconstruction CE in codec warm-up, nats per decoder symbol | sphere | 250 | 3.372 | 0.7318 | 1.176 | 5.567 | 2.673 | 5.902 | 2.917 | [2.853, 2.996] |
| `codec_warmup/decoder_ce` | oracle reconstruction CE in codec warm-up, nats per decoder symbol | magnitude | 250 | 3.351 | 0.6587 | 1.375 | 5.327 | 2.697 | 5.9 | 2.94 | [2.875, 3.021] |
| `codec_warmup/grad_norm/encoder` | encoder gradient norm in codec warm-up | sphere | 250 | 0.06579 | 0.05746 | -0.1066 | 0.2382 | 0.0125 | 0.4257 | 0.09577 | [0.06402, 0.1394] |
| `codec_warmup/grad_norm/encoder` | encoder gradient norm in codec warm-up | magnitude | 250 | 0.1582 | 0.1391 | -0.259 | 0.5754 | 0.03152 | 0.7044 | 0.1172 | [0.08377, 0.1591] |

Notes on these rows (MEASURED):

- Negative `mean - 3 sd` on gradient norms (a nonnegative quantity) is the visible sign that the
  normal 3-sigma reading does not fit those right-skewed series; the percentile columns are the
  honest tails.
- Sentence-norm rows have n = 732: 18 prediction steps per arm (prediction-local steps 3, 39, 178,
  208, 233, 278, 299, 308, 415, 440, 487, 496, 534, 565, 593, 636, 725, 748, identical in both arms)
  carry `sentences.count == 0` and `history_bytes == 0`, so the row records `range: null`. These are
  empty-history batches, not missing data.
- Sphere-cell norms equal 1 by construction; sd of order 1e-8 is float32 rounding.

Rows that are **NOT H1 quantities** (Fisher consumer or future law active in H0; structurally absent
or inert under the H1 profile rule). Reported for completeness, never as a candidate band:

| key | meaning | arm | n | mean | sd | mean - 3 sd | mean + 3 sd | p0.135 | p99.865 | last-50 mean | last-50 iid bootstrap, 99.73% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prediction/future_nll_per_dimension` | future-law NLL per latent dimension (H0 only) | sphere | 750 | 0.3146 | 0.8374 | -2.198 | 2.827 | -0.6881 | 4.895 | 0.08813 | [-0.09494, 0.2916] |
| `prediction/future_nll_per_dimension` | future-law NLL per latent dimension (H0 only) | magnitude | 750 | 0.1753 | 0.9569 | -2.695 | 3.046 | -0.6231 | 6.251 | 0.04014 | [-0.1955, 0.3357] |
| `prediction/grad_norm/future_law` | future-law gradient norm (H0 only) | sphere | 750 | 0.7815 | 0.4692 | -0.626 | 2.189 | 0.3179 | 3.386 | 0.5603 | [0.5033, 0.6303] |
| `prediction/grad_norm/future_law` | future-law gradient norm (H0 only) | magnitude | 750 | 0.6466 | 0.5025 | -0.8609 | 2.154 | 0.225 | 4.053 | 0.4629 | [0.3889, 0.5514] |
| `prediction/solver_max_residual_ratio` | Fisher CG max residual ratio (H0 only) | sphere | 750 | 0.9293 | 0.08598 | 0.6714 | 1.187 | 0.205 | 0.9996 | 0.9505 | [0.9081, 0.9753] |
| `prediction/solver_max_residual_ratio` | Fisher CG max residual ratio (H0 only) | magnitude | 750 | 0.9513 | 0.0659 | 0.7536 | 1.149 | 0.4388 | 0.9998 | 0.9295 | [0.8751, 0.9612] |

### 3c. Measured evaluation rows from HEpP7g

MEASURED: `HRun.evaluations` from `load_h_run`, values are the extractor's projection (mean over the
finite rows of one held-out example batch of 4 rows). Each evaluation replayed the same held-out
batch from cursor 0 with the same seed (the pre-FOR-1664 behavior; the step-keyed reseed landed
later). `log_marginal` is in nats per held-out sentence (summed over its decoder symbols), not per
byte. ESS at k = 2 and 4 is bounded above by k (FOR-1664): these rows are plumbing evidence and must
not be read as convergence. Per evaluation there is no dispersion to report beyond these values.

| arm | global step | `eval/log_marginal_k2` | `eval/log_marginal_k4` | `eval/ess_k2` | `eval/ess_k4` | `eval/gap_rows_k2` | `eval/gap_rows_k4` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sphere | 500 | -443.38 | -438.16 | 1.2190 | 1.0248 | 0 | 0 |
| sphere | 750 | -430.42 | -425.34 | 1.0768 | 1.0212 | 0 | 0 |
| sphere | 1000 | -427.89 | -422.66 | 1.0165 | 1.0704 | 0 | 0 |
| magnitude | 500 | -430.54 | -428.76 | 1.1193 | 1.0277 | 0 | 0 |
| magnitude | 750 | -417.67 | -416.03 | 1.1053 | 1.0652 | 0 | 0 |
| magnitude | 1000 | -412.28 | -411.35 | 1.0602 | 1.0806 | 0 | 0 |

The H1 development curve differs: profile schema `h-run-profile-v2` requires
`importance_samples = (2, 4, 8, 16, 32, 64)` and `diagnostic_samples = 256` for every non-smoke run
class (`core/sentence/profile.py`, `DEVELOPMENT_IMPORTANCE_SAMPLES`, `DEVELOPMENT_DIAGNOSTIC_SAMPLES`).
Pilot keys will be `eval/*_k2` through `eval/*_k64` plus `eval/*_k256`.

## 4. Primary contrasts and the deciding family

PROPOSAL: the deciding family is the three `factorial_terms` contrasts (sentence main effect, spine
main effect, interaction) on each deciding estimand. With E-LM alone deciding, m = 3; with E-LM and
E-DCE both deciding, m = 6. Bonferroni over the complete deciding family, 95% familywise (PLAN).

Note on scale (MEASURED from the code): the main-effect weights are ±0.5 (a mean of two
differences), the interaction weights are ±1 (a difference of differences). On the same per-cell
noise, the interaction contrast has twice the sd of a main effect, so its margin needs its own
justification, not a copy of the main-effect margin.

ARITHMETIC (scipy `norm.ppf`, `t.ppf`; `power_arith.py` outside the repo): per-contrast two-sided
alpha and critical quantiles. df 7 is the 8-block pilot; df 15 is one example of a confirmatory n.

| family size m | per-contrast alpha = 0.05/m | z | t, df 7 | t, df 15 |
| --- | --- | --- | --- | --- |
| 3 | 0.016667 | 2.3940 | 3.1276 | 2.6937 |
| 4 | 0.012500 | 2.4977 | 3.3353 | 2.8366 |
| 6 | 0.008333 | 2.6383 | 3.6358 | 3.0363 |
| 3-sigma secondary (report-only, not deciding) | 0.002700 | 3.0000 | 4.5300 | 3.5864 |

## 5. Decision rule template

Deciding rule (PLAN, unchanged): 95% familywise, Bonferroni over the complete deciding family,
two-sided simultaneous intervals from `dual_confidence_intervals(pairs, family_size=m, ...)`.
Secondary (PLAN): the predeclared 3-sigma interval (`THREE_SIGMA_CONFIDENCE = 0.99730020393674`) on
the same sealed data, report-only, no multiplicity adjustment. The operator reads bands at 3 sigma;
that reading is this secondary report and never replaces the deciding rule.

Per contrast c and deciding estimand e, with margin Δ(c, e) sealed in the amendment (BLANK). The
hypothesis kind of each contrast is sealed before the pilot (Q1), and the outcome is read ONLY from
the table of that kind. Within a kind the conditions partition the possible intervals, so every
interval gets exactly one label. (Corrected 2026-10-23, PR #380 review: the first draft listed the
labels in one table whose conditions overlapped; an interval (0.10, 0.20) with Δ = 0.50 matched
both "equivalent" and "not supported".)

Kind = difference (two-sided superiority):

| outcome | condition on the deciding interval (L, U) | label |
| --- | --- | --- |
| supported difference | L > +Δ or U < -Δ | PROPOSAL |
| no practically relevant difference | -Δ < L and U < +Δ; reported with the annotation "detectable" when the interval also excludes 0 | PROPOSAL |
| inconclusive | every other interval | PROPOSAL |

Kind = equivalence:

| outcome | condition on the deciding interval (L, U) | label |
| --- | --- | --- |
| equivalent within margin | -Δ < L and U < +Δ | PROPOSAL |
| not equivalent | L > +Δ or U < -Δ | PROPOSAL |
| inconclusive | every other interval | PROPOSAL |

Kind = noninferiority (the harmful direction sealed with the contrast; written here for "lower is
worse"):

| outcome | condition on the deciding interval (L, U) | label |
| --- | --- | --- |
| noninferior | L > -Δ | PROPOSAL |
| inferior | U < -Δ | PROPOSAL |
| inconclusive | every other interval | PROPOSAL |

Per cell (eligibility, PLAN "retain every construction meeting registered eligibility"): BLANK
criteria (Q5). If both normalization choices fail eligibility, H1 pauses.

Each verdict quotes the deciding label, the secondary 3-sigma interval, and the estimand's units.
A contrast whose estimand lacks a margin cannot be labelled; it is reported as an interval only.

## 6. Power and n

Formula (mirrors `seed_sizing._power` and `plan_seed_count`): for a paired contrast with pilot
variance bound σ² and planned effect Δ (distance from expected effect to the decision bound), the
smallest n with `nct.sf(t_{1-α/(2m), n-1}, n-1, Δ·sqrt(n)/σ) >= 0.90`. `plan_seed_count` replaces
the pilot sample variance s² by its one-sided 95% upper bound `(k-1)s²/chi2.ppf(0.05, k-1)` with
k = 8 pilot blocks: variance × 3.2298, sd × 1.7972 (ARITHMETIC).

ARITHMETIC, superiority shape only (no variance assumed; Δ expressed as a multiple c of the pilot
sd). "point sd" treats s as the true sd; "chi2-bounded sd" is what `plan_seed_count` would use.

| per-contrast alpha | Δ = 0.5 s: point / bounded | Δ = 0.75 s: point / bounded | Δ = 1.0 s: point / bounded |
| --- | --- | --- | --- |
| m = 3 (0.016667) | 57 / 178 | 27 / 81 | 17 / 47 |
| m = 4 (0.0125) | 61 / 188 | 29 / 86 | 18 / 50 |
| m = 6 (0.008333) | 65 / 202 | 31 / 92 | 19 / 54 |
| 3-sigma (0.0027), shown for reference only | 78 / 242 | 38 / 110 | 23 / 64 |

Each n is a count of complete paired 2×2 seed blocks (four runs per block). Equivalence or
noninferiority kinds need different, generally larger, n (`_power` uses the union lower bound
`max(0, lower_pass + upper_pass - 1)` for equivalence); this table is superiority only (Q1).
Power is a projection conditional on the declared effect and variance bound, never a guaranteed
achieved power (PLAN). An unaffordable powered n blocks execution (PLAN).

## 7. Pilot design (excluded, never confirmatory)

| field | content | label |
| --- | --- | --- |
| Size | 8 complete paired blocks per contrast family | PLAN |
| Block | one `H1PreservationBlock`: four profiles MM, SM, MS, SS, identical outside `law.mode` and `spine.carrier` (validated by the block), one seed, one paired source stream; codec artifacts bound within (MM, MS) and (SM, SS) in every block, unconditionally (`codec_sharing_groups` requires these pairs; section 2), each pair's codec warmed once on the block's paired source stream and budget; a block that warms a pair's members independently breaks the declared design and is excluded as a defect (corrected 2026-10-23, PR #380 review: the first draft made the sharing conditional) | PLAN + MEASURED (code) |
| Run class | `pilot`, never `confirmatory`; profile names stage-prefixed `h1-` and not `-tiny` | MEASURED (`HProfile` rules) |
| Regime | identical to confirmation: width, depth, steps, batch, curve `(2..64)+256`, consumer `isotropic`, `future_weight = 0` | PROPOSAL; values BLANK (Q6) |
| Pilot seeds | disjoint from confirmatory seeds | PROPOSAL; values BLANK |
| Selection rules | frozen before pilot inspection; no pilot outcome selects an arm, a margin direction or an estimand | PLAN |
| Exclusion | a block is excluded only for a nonfinite or missing primary endpoint, or an admission refusal by the H reader; an exclusion drops the whole block and is reported; finite collapse stays in | PROPOSAL (E1 precedent; PLAN for collapse) |
| What is read | per-contrast paired sd of each deciding estimand (the variance input to `plan_seed_count`), ESS-against-k curve admissibility, cost per block (GPU-seconds, wall), defect counts | PROPOSAL |
| What is NOT read | the sign or size of any contrast mean as an input to margins, directions, estimand choice or family composition | PLAN |
| Zero pilot variance | `PilotVarianceUnknown`: no powered design, no epsilon substitution | MEASURED (code) |
| Quote | separately quoted variance proposal, operator GO | PLAN |

## 8. Instruments

| instrument | what it measures | status at base | label |
| --- | --- | --- | --- |
| ESS against k (FOR-1664) | `eval/ess_k{k}` and `eval/log_marginal_k{k}` for k in `(2, 4, 8, 16, 32, 64)` plus `256`; keys always k-suffixed | code landed (`profile.py` constants; `h_metric_extractor.py`); step-keyed evaluation seed `(seed * 0x9E3779B1 + step) % 2**32` | MEASURED |
| ESS estimator class | per-row ESS from normalized importance weights over a single nested proposal population drawn once; the projection is the mean over finite rows | BLANK: the class (per-row mean, median, or pooled) and the admissibility threshold are sealed here (Q4) | PROPOSAL |
| Importance-weighted log-marginal | `core/sentence/scoring.py:79 importance_evaluate`: frozen-encoder proposal, decoder log-probs in FP64, `log_weights = decoder + prior - proposal` | landed | MEASURED |
| Span-aware predictive score gain | `ladder/run/span_probe.py` (horizon-conditioned linear probes, observed and shuffled controls, uniform control), `span_diagnostics.py`; commands `just h-diagnose-spans`, `just h-inspect-spans` | exists, not run in HEpP7g | MEASURED |
| Reader identity, checkpoint | `h-checkpoint-reader-v2` (`ladder/run/checkpoint.py:53`, module tuple `_READER_MODULES`) | pin the composed digest row at the sealing SHA; BLANK | PLAN |
| Reader identity, source | `h-source-reader-v3` (`ladder/capture/capture.py:51`, with python version, pysbd version and file digests) | pin at the sealing SHA; BLANK | PLAN |
| Regime rows | `tools/regime_rows.py`, pinned in `tests/test_identity_epoch.py` | pin at the sealing SHA | PLAN |
| Run admission and aggregation | `entropic-substrate mlflow-aggregate h --run METRICS_JSONL WORKER_LOG_JSONL` (repeatable, one per arm; `nargs=2`, `composition/entrypoints/cli.py:133-147`) `--arm IDENTITY=LABEL` (`ladder/analyze/h_aggregate.py`, offline, no MLflow) | landed | MEASURED |
| Single-run report | `entropic-substrate eval h --run METRICS_JSONL WORKER_LOG_JSONL` (`ladder/analyze/h_report.py`) | landed | MEASURED |
| Charts | `entropic-substrate charts h --run METRICS_JSONL WORKER_LOG_JSONL --arm IDENTITY=LABEL` | landed | MEASURED |
| Final-probe check suite | `scripts/h0_final_probe_checks.py` over `ladder/analyze/h_final_probe_checks.py` (K1 to K7) | landed; H0 exit tool, reused as a lineage check | MEASURED |
| Paired planning and intervals | `ladder/analyze/seed_sizing.py` `plan_seed_count`, `dual_confidence_intervals` | landed | MEASURED |
| Not yet instrumented | angular and reconstruction error, effective rank and conditioning, saturation, detail preservation, shadow versus emitted, matched perturbations and diagnostic decoders, autocast-enabled field (FOR-1672) | instrument needed before the pilot if any is deciding | MEASURED (grep) |

## 9. Seed matrix, budgets, exclusions, successor clauses

| field | value | label |
| --- | --- | --- |
| Pilot seed list | BLANK | operator |
| Confirmatory seed list | BLANK, size from the amendment | operator |
| Seed semantics | seed blocks are model replicates; documents, probe positions and generated samples are nested measurements | PLAN |
| Profile regime (widths, layers, steps, batch, lr) | BLANK; must be admitted by inventory and a representative resource measurement | PLAN |
| Matching tolerance | 1% parameter match, otherwise not labelled matched; dormant magnitude parameters cannot pad capacity | PLAN |
| Budget, pilot | BLANK; quoted separately | operator |
| Budget, confirmation | BLANK; quoted after the amendment | operator |
| Endpoint exposure | identical across a block (`seed_sizing` refuses mismatched exposure) | MEASURED (code) |
| Exclusions | only the section 7 rule; no outcome-selected replacements, no extra looks | PLAN + PROPOSAL |
| Defects | missing, nonfinite or wrong-owner gradients are defects; measured-zero and structurally absent gradients are distinct | PLAN |
| Successor clauses | constructions meeting eligibility carry to H2 as constructions, never trained weights; successor spines start from fresh paired initializations; if both fail, pause | PLAN |
| Analysis | fresh session, captured files only, arm-blind review where possible, filed verdict | PLAN |

## 10. Open questions for the operator

1. **Hypothesis kind per contrast.** Options: superiority (two-sided difference), equivalence
   within ±Δ, noninferiority. Recommendation: two-sided difference for the sentence and spine main
   effects and the interaction on E-LM, with an equivalence reading reported beside it; H1 asks
   what normalization loses, so "no loss within Δ" is a result worth stating. The kind changes n
   (section 6 is superiority only).
2. **Deciding estimands.** Options: E-LM alone (m = 3); E-LM and E-DCE (m = 6). Recommendation:
   E-LM alone deciding, E-DCE exploratory. E-LM is the honest generative score the plan names;
   adding E-DCE doubles the family and raises every n.
3. **Promotion of instruments not yet built.** The plan's H1 exit requires independent detail
   checks and numerical evidence (angular error, rank, conditioning, shadow versus emitted,
   matched perturbations). Options: build them before the pilot and keep them exploratory; promote
   one to deciding; or narrow the H1 exit. Recommendation: build E-DETAIL and E-PERT before the
   pilot as exploratory, keep E-SPAN diagnostic as the plan says.
4. **ESS estimator class and admissibility threshold.** Options: per-row mean ESS (the current
   projection), median, or pooled; threshold as ESS/k at k = 64. Recommendation: seal per-row
   median ESS/k at k = 64 plus the full curve as an admissibility gate, threshold fixed before the
   pilot from first principles, not from pilot data.
5. **Cell eligibility criteria.** What makes a construction eligible to carry to H2 (for example
   no defects, admissible ESS, E-LM not worse than the best cell by more than Δ). Recommendation:
   defect-free plus admissible instruments plus the E-LM rule, sealed with Δ in the amendment.
6. **Regime for pilot and confirmation.** Tiny profiles are smoke-only: `HProfile` lets a
   `-tiny` smoke class carry any valid schedule, but a `pilot` or `confirmatory` H1 profile must
   carry the development curve and cannot be named `-tiny`. Options: the development profile (sentence width 384, 12-layer 768-wide
   spine) or a reduced non-tiny profile. Recommendation: the smallest non-tiny profile whose
   inventory and resource measurement fit the quote, declared before the pilot.
7. **No-hidden-normalization check on the magnitude path.** The plan says the old D encoder fails
   this unchanged. Recommendation: a guard-teeth test that the MM and MS cells' sentence and state
   norms are not overwritten anywhere, proven by mutation, before the pilot.
8. **Margins and their units.** Every margin needs units and a prospective justification; the
   0.01 nats/byte E reference does not transfer. Recommendation: express E-LM margins in nats per
   canonical byte, justified from a declared practical scale, sealed before pilot inspection, with
   the pilot used only for variance.
