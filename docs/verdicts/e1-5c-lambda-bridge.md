# E1.5c full-budget lambda-bridge decision

> **JOINT LANE RUNS at lambda = 0.3.** All three bridge lambdas pass the
> Amendment 3 clause-(1) calibration gate. Clause (2) selects lambda 0.3 by
> minimum terminal last-5 backbone nats/byte. Clause (3)'s smaller-lambda
> tie-break is not reached, and clause (4) does not fire.

## 0. Scope and decision status

This document applies the pre-committed gate-first rule in
`docs/e1-5c-preregistration.md` Amendment 3 to the full-budget bridge. It does
not revise the rule, relax the registered `1.308881` threshold, or blend field
calibration and backbone quality into a composite judgment.

This is not a rung verdict and does not create a `*-verdict.json`. The joint
lane remains the registered exploratory secondary described by Amendment 3.
No frozen-wave F, H, or matched-total capture was read.

## 1. Estimator and evidence labels

The terminal estimator is the exact decimal mean of the last five admitted
eval records, at steps `{18000, 18500, 19000, 19500, 20000}`. Every outcome
metric was read from the named DVC-materialized JSONL through
`entropic_substrate.telemetry.read_eval_series`. That reader admitted all 40
eval records in each bridge capture and validated finiteness, documented
metric domains, and completeness of the five-member field-calibration group.

Evidence labels used below:

- **MEASURED THIS SESSION**: read from an admitted capture or computed over
  the materialized capture bytes, such as an md5.
- **DERIVED FROM MEASURED**: deterministic arithmetic over admitted telemetry
  or the quoted measured gate constant.
- **QUOTED MEASURED INPUT**: supplied measured reference, not re-measured in
  this analysis.
- **INFERRED / EXTRAPOLATED**: depends on an assumption beyond the admitted
  bridge measurements.
- **INFERRED LIMITATION**: an uncertainty qualification that is plausible but
  cannot be quantified from one seed.

The bridge `run_start` records verify the declared regime: F arm
(`e1-5c-field`), `field_mode=joint`, seed 42, CUDA, `max_steps=20000`, and the
same model, data, optimizer, and schedule configuration apart from the
declared `field_loss_lambda`. This is **MEASURED THIS SESSION** metadata read
through telemetry's validated record iterator.

The registered reference constants are **QUOTED MEASURED INPUT**:

| quantity | evidence class | value |
|---|---|---:|
| standalone field CE, final | **QUOTED MEASURED INPUT** | 1.256011 |
| standalone field CE, last-5 mean | **QUOTED MEASURED INPUT** | 1.258881 |
| clause-(1) gate (`1.258881 + 0.05`) | **DERIVED FROM QUOTED MEASURED INPUT** | 1.308881 |

## 2. Gate-first decision table

Signed gap is defined as `terminal field CE - 1.308881`; a non-positive gap
passes. All numeric metric cells retain the full precision available from the
mean of five serialized six-decimal readings.

| lambda | evidence class | terminal last-5 field CE | signed gap to gate | clause-(1) gate | terminal last-5 backbone nats/byte |
|---:|---|---:|---:|:---:|---:|
| 0.3 | **DERIVED FROM MEASURED** | 1.3061124 | -0.0027686 | **PASS** | 0.773199 |
| 0.5 | **DERIVED FROM MEASURED** | 1.281115 | -0.027766 | **PASS** | 0.773947 |
| 1.0 | **DERIVED FROM MEASURED** | 1.2700688 | -0.0388122 | **PASS** | 0.7798882 |

Mechanical application of Amendment 3:

1. **Clause (1), feasibility, fires first:** the feasible set is
   `{0.3, 0.5, 1.0}` because every terminal last-5 field CE is at or below
   `1.308881`.
2. **Clause (2), objective among feasible lambdas, selects lambda 0.3:** its
   terminal last-5 backbone nats/byte, `0.773199`, is the unique minimum.
3. **Clause (3), smaller-lambda tie-break, is not reached:** there is no tie in
   the clause-(2) objective.
4. **Clause (4), empty feasible set, does not fire:** the joint lane therefore
   **RUNS at lambda 0.3**.

The backbone difference between lambda 0.3 and lambda 0.5 is
`0.773947 - 0.773199 = 0.000748` nats/byte, **DERIVED FROM MEASURED**. With
`n = 1` per lambda, seed noise could plausibly reorder adjacent arms, and this
bridge supplies no seed distribution or confidence interval with which to
quantify that risk. That limitation does not alter the registered mechanical
selection.

Lambda 0.3 clears the gate by only `0.0027686` nats/byte. It is therefore a
**weak PASS**: unmeasured seed noise could plausibly flip this narrow margin.
This is an **INFERRED LIMITATION**, not a quantitative claim that seed noise
was measured to exceed the margin. Lambdas 0.5 and 1.0 have wider measured
margins, but one seed still does not establish seed-stable gate passage for
either arm.

## 3. Terminal calibration diagnostics

These diagnostics use the same terminal five eval records as the decision
table. They are context only and are not additional selection criteria.

| lambda | evidence class | field Brier | field ECE | field entropy | field top-1 |
|---:|---|---:|---:|---:|---:|
| 0.3 | **DERIVED FROM MEASURED** | 0.4786776 | 0.020220 | 1.3616096 | 0.6245076 |
| 0.5 | **DERIVED FROM MEASURED** | 0.4734216 | 0.0162986 | 1.3326396 | 0.6241244 |
| 1.0 | **DERIVED FROM MEASURED** | 0.4702548 | 0.0170738 | 1.3123866 | 0.6292682 |

The diagnostics are intentionally not collapsed into a calibration score.
For example, lambda 0.5 has the lowest terminal ECE, while lambda 1.0 has the
lowest Brier and entropy and the highest top-1. Amendment 3 permits none of
those context metrics to replace field CE as the feasibility predicate.

## 4. Check of the short-pilot extrapolations

The earlier 6,000-step diagnostic made two explicitly
**INFERRED / EXTRAPOLATED** predictions:

- the log-linear assumption predicted that all arms would cross the field-CE
  gate by step 20,000;
- the inverse-step assumption predicted that only lambda 0.5 would cross by
  step 20,000, at approximately step 19,857.

The full-budget bridge provides the **DERIVED FROM MEASURED** terminal outcome
for its declared set `{0.3, 0.5, 1.0}`: all three pass. Therefore the measured
pass/fail pattern **matches the log-linear extrapolation over every arm the
bridge tested** and does **not** match the inverse-step extrapolation. The
inverse-step model correctly predicted lambda 0.5's passage but incorrectly
predicted failures for lambdas 0.3 and 1.0.

This bridge does not test the log-linear prediction for lambda 0.1, which
Amendment 3 excluded before the runs. The conclusion is consequently about
the bridge's three-arm domain, not a retrospective measurement of the omitted
arm. Extrapolation accuracy is diagnostic context for future rungs and has no
effect on the lambda selection in Section 2.

## 5. Differently scheduled budget endpoints

Each cell is a terminal last-5 field-CE mean from a separate admitted capture.

| lambda | evidence class | 3,000-step grid | 6,000-step grid | 20,000-step bridge |
|---:|---|---:|---:|---:|
| 0.3 | **DERIVED FROM MEASURED** | 1.8776158 | 1.5597004 | 1.3061124 |
| 0.5 | **DERIVED FROM MEASURED** | 1.7855836 | 1.5048136 | 1.281115 |
| 1.0 | **DERIVED FROM MEASURED** | 1.718257 | 1.4815124 | 1.2700688 |

**Schedule caveat:** the 3,000-, 6,000-, and 20,000-step columns are terminal
points of **differently scaled cosine schedules**. The trainer parameterizes
cosine decay by each run's own `max_steps`; therefore these columns are not
samples from one run's curve and must not be plotted or interpreted as a
single trajectory without this caveat. They show measured terminal outcomes
at three separately scaled budgets, not within-run convergence at steps 3,000,
6,000, and 20,000 of the bridge.

## 6. Input integrity and md5s

The main decision inputs are the three bridge captures. The six short-grid
captures below were read only for Section 5's requested context. The md5s are
**MEASURED THIS SESSION** over the materialized DVC JSONL bytes.

| role | lambda | evidence class | input | md5 |
|---|---:|---|---|---|
| 20,000-step decision | 0.3 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-13bd4827-e1-5c-bridge20k-lam03.jsonl` | `b611272de6c8b402f0343794e2164ac5` |
| 20,000-step decision | 0.5 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-f46f1068-e1-5c-bridge20k-lam05.jsonl` | `4e6b6c1f62cf27959745da290cba8dab` |
| 20,000-step decision | 1.0 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-44d03e1f-e1-5c-bridge20k-lam10.jsonl` | `ed7d328ff0baacb7f024ac9302f8f877` |
| 6,000-step context | 0.3 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-6a0884cb-e1-5c-pilot6k-lam03.jsonl` | `e206b592e7ed1556bfd397d890a31b51` |
| 6,000-step context | 0.5 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-45230629-e1-5c-pilot6k-lam05.jsonl` | `0ff522300fe1e0affa1c0e97eba2151b` |
| 6,000-step context | 1.0 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-297b42cf-e1-5c-pilot6k-lam10.jsonl` | `a3bf6861ac7638014b6afdb8c58a3951` |
| 3,000-step context | 0.3 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-fd30c01c-e1-5c-pilot-lam03.jsonl` | `60f96e4174fc6e47b0fd9333780d0fe2` |
| 3,000-step context | 0.5 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-cccab401-e1-5c-pilot-lam05.jsonl` | `3aacf337605ee82aaf33467453d3cf05` |
| 3,000-step context | 1.0 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-c66e4903-e1-5c-pilot-lam10.jsonl` | `56a2b5f59da6040ce8d697cea91eff86` |

`dvc status results/captured.dvc` reports the tracked capture directory up to
date. Its current DVC directory md5 is
`1579fd960b30e2374b856436a084bc8f.dir`. No capture was repaired, imputed,
dropped, or silently substituted.
