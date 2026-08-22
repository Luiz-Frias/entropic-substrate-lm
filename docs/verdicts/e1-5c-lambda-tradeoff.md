# E1.5c lambda-field trade-off diagnostic

## 0. Scope and answer

This is the exploratory diagnostic authorized by
`docs/e1-5c-preregistration.md` **Amendment 2**. It is not a rung verdict,
does not create a `*-verdict.json`, and does not re-select lambda. The
registered pilot rule has already selected **lambda* = 0.1**; that selection
stands unless the operator explicitly amends the rule before the joint lane.

**Decision-relevant answer:** no arm reaches the registered joint-arm
calibration gate at 6,000 steps. The best 6,000-step field CE is `1.4815124`
at lambda 1.0, still `0.1726314` nats/byte above the `1.308881` gate.

The captures do establish that field CE is moving toward the gate for every
lambda observed at both budgets. They do **not** establish that the gate will
be reached by 20,000 steps: transparent extrapolations with two plausible
functional forms disagree. Gate reachability is therefore **plausible but not
demonstrated** for lambda 0.3, 0.5, and 1.0; among these data, lambda 0.5 is
the only arm that reaches by 20,000 under both illustrative assumptions.
Reachability at the registered lambda* = 0.1 is especially assumption-sensitive
and is not supported robustly by this diagnostic.

No frozen-wave F, H, or matched-total capture was read. All outcome metrics
come only from the nine named DVC-materialized pilot captures below, through
`entropic_substrate.telemetry.read_eval_series`.

## 1. Estimator and evidence labels

Every metric reading in this document uses the registration's **last-k
(`k = 5`) eval-mean convention**. At 3,000 steps, the five eval records end at
steps `{1000, 1500, 2000, 2500, 3000}`. At 6,000 steps, they end at
`{4000, 4500, 5000, 5500, 6000}`. Calibration diagnostics use the same five
records as field CE and backbone nats/byte.

Evidence labels are deliberately explicit:

- **MEASURED THIS SESSION**: read directly from a named admitted capture, or
  computed directly over its bytes (md5).
- **DERIVED FROM MEASURED**: deterministic arithmetic over admitted telemetry
  values or over the quoted-measured reference constant.
- **QUOTED MEASURED INPUT**: supplied measured reference or registered cost,
  not re-measured in this session.
- **INFERRED / EXTRAPOLATED**: depends on a stated functional assumption beyond
  the observed step range.
- **OPERATOR RECOMMENDATION**: proposed action, not an experimental result or
  registration amendment.

Reference constants are **QUOTED MEASURED INPUT**: standalone final held-out
field CE `1.256011`, standalone last-5 mean `1.258881`, and joint-arm gate
`1.258881 + 0.05 = 1.308881`. The last-5 reference is the registered estimator;
the standalone single-final value is provenance, not the gate anchor.

All arms are seed 42 and there is only one run per lambda at each budget.
There is no confidence interval to report. In particular, the small backbone
differences between adjacent arms could plausibly reorder under seed or run
noise; this diagnostic cannot quantify that possibility from `n = 1`.

The admitted `run_start` metadata matches across all nine captures on the
regime-defining configuration, model/data parameters, CUDA device, and seed.
The only substantive configured differences are the declared lambda and
`max_steps` axes (`3000` versus `6000`); run names and artifact paths also
differ. This is **MEASURED THIS SESSION** metadata plus a deterministic config
comparison, not an assumption from the filenames.

## 2. The 6,000-step curve

Every numeric metric cell below is **DERIVED FROM MEASURED** as the exact
decimal mean of five serialized six-decimal telemetry readings. Full available
precision is retained.

| lambda | evidence class | field CE | backbone nats/byte | field Brier | field ECE | field entropy | field top-1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.1 | **DERIVED FROM MEASURED** | 1.7947244 | 0.8606104 | 0.6325506 | 0.0421996 | 1.970649 | 0.4933626 |
| 0.2 | **DERIVED FROM MEASURED** | 1.6350418 | 0.8662608 | 0.5854566 | 0.035894 | 1.7695386 | 0.5331986 |
| 0.3 | **DERIVED FROM MEASURED** | 1.5597004 | 0.8649732 | 0.5618232 | 0.0300366 | 1.6759294 | 0.552659 |
| 0.5 | **DERIVED FROM MEASURED** | 1.5048136 | 0.8694038 | 0.5420638 | 0.0275672 | 1.5957722 | 0.569206 |
| 1.0 | **DERIVED FROM MEASURED** | 1.4815124 | 0.8702916 | 0.536599 | 0.022259 | 1.5529724 | 0.570382 |

Field CE, Brier, ECE, and entropy decrease across the lambda grid, while
top-1 increases. Backbone nats/byte is not strictly monotone: lambda 0.3 is
`0.0012876` lower than lambda 0.2 while also having field CE `0.0753414`
lower. Thus lambda 0.3 dominates lambda 0.2 in this observed two-objective,
single-seed grid. That dominance is a measured-grid relation, not a claim
that the ordering is seed-stable.

## 3. Three candidate middles

These answer different questions and are not alternative selections of
lambda*. None changes the registered value `0.1`.

### 3.1 Normalised crossing: relative balance

For each metric, lower is better. I min-max normalised over the five observed
6,000-step arms:

```text
field_norm(lambda) = (field_CE(lambda) - min(field_CE))
                     / (max(field_CE) - min(field_CE))
backbone_norm(lambda) = (backbone(lambda) - min(backbone))
                        / (max(backbone) - min(backbone))
```

The resulting curves bracket a crossing between lambda 0.1 and 0.2.

| lambda | evidence class | normalised field CE | normalised backbone | field minus backbone |
|---:|---|---:|---:|---:|
| 0.1 | **DERIVED FROM MEASURED** | 1 | 0 | 1 |
| 0.2 | **DERIVED FROM MEASURED** | 0.4901772601305186263616975084 | 0.5836466553733008304755608809 | -0.0934693952427822041138633725 |
| 0.3 | **DERIVED FROM MEASURED** | 0.2496328365452153812753023511 | 0.4506466140561087468495641036 | -0.2010137775108933655742617525 |
| 0.5 | **DERIVED FROM MEASURED** | 0.07439433993589006806891179137 | 0.9082964921703921001528736107 | -0.8339021522345020320839618193 |
| 1.0 | **DERIVED FROM MEASURED** | 0 | 1 | -1 |

I linearly interpolated the difference
`field_norm(lambda) - backbone_norm(lambda)` in lambda between 0.1 and 0.2:

```text
lambda_cross = 0.1914520337149418559483240695
```

This is a relative balance point, not a gate result. Min-max normalisation
fixes both zero and one to the observed endpoints, so changing the lambda
grid or extending either observed range can move this crossing even if the
local measurements do not change.

### 3.2 Pareto knee: raw two-objective frontier geometry

I used coordinates `(backbone nats/byte, field CE)`, both in nats/byte, and
the chord from the two extreme arms lambda 0.1 and 1.0. For each arm, the
knee score is its perpendicular Euclidean distance from that chord in the
raw coordinate plane. No min-max normalisation is used for this estimator.

| lambda | evidence class | Pareto status in observed grid | perpendicular chord distance |
|---:|---|---|---:|
| 0.1 | **DERIVED FROM MEASURED** | frontier endpoint | 0 |
| 0.2 | **DERIVED FROM MEASURED** | dominated by lambda 0.3 | 0.00071436292393305904 |
| 0.3 | **DERIVED FROM MEASURED** | frontier; **maximum distance** | **0.0029002694664736158** |
| 0.5 | **DERIVED FROM MEASURED** | frontier | 0.00016749352432778286 |
| 1.0 | **DERIVED FROM MEASURED** | frontier endpoint | 0 |

Under this declared geometric definition, the **Pareto knee is lambda 0.3**.
It marks the strongest bend away from the extreme-arm chord: after this arm,
the remaining absolute field-CE gains are smaller. This is a deterministic
description of this `n = 1` grid, not a new lambda selection.

### 3.3 Gate-reaching lambda: operational feasibility

| gate question | evidence class | result |
|---|---|---|
| Smallest lambda with 6,000-step field CE <= 1.308881 | **DERIVED FROM MEASURED** | **None** |

No swept lambda reaches the gate at 6,000 steps. This is a finding, not a
missing analysis result.

## 4. Budget trajectory and reachability

### 4.1 Measured-budget endpoints

Each endpoint is the registered last-5 mean from its separately captured
budget run. Change is `CE_6000 - CE_3000`; a negative value means movement
toward the gate.

| lambda | evidence class | field CE at 3,000 | field CE at 6,000 | change | remaining gap at 6,000 |
|---:|---|---:|---:|---:|---:|
| 0.1 | **DERIVED FROM MEASURED** | 2.2137002 | 1.7947244 | -0.4189758 | 0.4858434 |
| 0.3 | **DERIVED FROM MEASURED** | 1.8776158 | 1.5597004 | -0.3179154 | 0.2508194 |
| 0.5 | **DERIVED FROM MEASURED** | 1.7855836 | 1.5048136 | -0.2807700 | 0.1959326 |
| 1.0 | **DERIVED FROM MEASURED** | 1.718257 | 1.4815124 | -0.2367446 | 0.1726314 |

**Measured-direction conclusion:** yes, field CE is moving toward 1.308881
for every common lambda. The 3,000-to-6,000 change closes `46.3049192590078%`
of lambda 0.1's prior gate gap and `55.8987070951171%`,
`58.8983571727949%`, and `57.8306007191433%` for lambdas 0.3, 0.5, and 1.0
respectively. These percentages describe observed gap closure only; they are
not estimates of a stable convergence rate.

The within-run 6,000-step telemetry also warns against assuming a constant
rate. Using the same last-5 convention, field CE decreases monotonically in
every arm, but the absolute improvement from ending-step 4,500 to 6,000 is
only about 39--41% of the improvement from ending-step 3,000 to 4,500.
Those rolling means overlap, so this is descriptive evidence of flattening,
not an independent sample or uncertainty estimate.

There is a second structural limitation. Capture metadata verifies
`max_steps = 3000` for the registered pilot and `max_steps = 6000` for the
extension, and the trainer's cosine schedule is parameterised by that field.
The registered 20,000-step lane uses `max_steps = 20000`. These budgets
therefore represent differently scaled schedules rather than observations
from one uninterrupted 20,000-step trajectory. This statement is **DERIVED
FROM MEASURED capture metadata and the inspected scheduler implementation**.
The two short-run endpoints are useful budget measurements, but they do not
identify the shape of the registered full-budget curve.

### 4.2 Explicit extrapolation sensitivity

Every value in this subsection is **INFERRED / EXTRAPOLATED**, not measured.
Two points cannot validate a functional form, so the calculations below are
shown to test whether the reachability conclusion is assumption-sensitive.
They are not forecasts to average together.

**Assumption A: log-linear in steps.** For each lambda,

```text
CE(s) = CE(3000)
        + (CE(6000) - CE(3000)) / ln(2) * ln(s / 3000)
```

This assumes the same absolute field-CE improvement for each doubling of
steps. It is optimistic relative to the flattening visible within the 6,000-
step captures.

**Assumption B: inverse-step diminishing return.** For each lambda,
`CE(s) = a + b/s`, with `a` and `b` fixed exactly by the two budget endpoints.
This forces progressively smaller gains and a finite asymptote. It is a
sensitivity model, not a uniquely justified learning-curve law.

| lambda | evidence class | log-linear CE at 20,000 | log-linear gate-crossing step | inverse-step CE at 20,000 | inverse-step gate-crossing step |
|---:|---|---:|---:|---:|---:|
| 0.1 | **INFERRED / EXTRAPOLATED** | 1.06697785061174 | 13403.7080459213 | 1.50144134 | never; fitted asymptote 1.3757486 is above gate |
| 0.3 | **INFERRED / EXTRAPOLATED** | 1.00749228834441 | 10366.899974208 | 1.33715962 | 28429.3012996304 |
| 0.5 | **INFERRED / EXTRAPOLATED** | 1.01712577012595 | 9732.44796271651 | 1.3082746 | 19857.0441809862 |
| 1.0 | **INFERRED / EXTRAPOLATED** | 1.07029517519536 | 9946.23716922921 | 1.31579118 | 22155.6184997785 |

**Reachability judgment:** the direction of convergence is measured, but
full-budget gate reachability is insufficiently identified. The log-linear
assumption says every arm reaches by 20,000; the diminishing-return assumption
says only lambda 0.5 reaches, and only just before the budget. Consequently:

- lambda 0.3, 0.5, and 1.0 are **plausible** gate-reaching candidates because
  they are materially closer at 6,000, but only lambda 0.5 is robust to both
  displayed assumptions;
- lambda 0.1 reaches only under the more optimistic model and remains
  `0.4858434` above the gate at 6,000, so this diagnostic does not provide a
  robust basis for expecting the registered arm to pass at 20,000; and
- none of these statements is a confidence claim. One seed per lambda cannot
  distinguish curve shape from seed/run noise or show that adjacent-arm
  ordering will persist.

The defensible answer is therefore not “the gate will be reached at lambda
X.” It is: **the gate is moving into range, but the available budgets do not
support a model-independent 20,000-step reachability verdict.**

## 5. Operator recommendation -- advice, not a decision

The registered lambda* remains `0.1`. The options below do not choose for the
operator.

| option | evidence / authority class | action | cost | principal risk |
|---|---|---|---|---|
| (a) Run registered joint lane | **REGISTERED OPTION informed by measured diagnostic** | Run the joint lane uniformly at lambda* = 0.1 without amendment. | Quoted registered plan: 12 full pods, approximately `$21`. | Lambda 0.1 has the largest 6,000-step gate gap (`0.4858434`); the lane may spend the full budget and then fail calibration clause 3. |
| (b) Amend rule now with field-CE gate filter | **OPERATOR DECISION informed by derived result** | Add `field CE <= 1.308881` as a pilot feasibility filter before the existing backbone-constrained argmin. | No new compute for the amendment itself. | On the measured 6,000-step data this selects **no lambda**, leaving the rule with an empty feasible set unless the operator also defines new evidence or failure handling. |
| (c) Declare a full-budget bridge pilot first | **OPERATOR RECOMMENDATION** | Before amending or launching the 12-pod lane, predeclare seed-42 20,000-step field-arm runs at the original lambdas `{0.1, 0.3, 0.5, 1.0}` under the exact joint-lane schedule, then apply a predeclared gate-first rule to those terminal last-5 readings. | Derived from quoted `$1.59/hr` and about 65 min/run: approximately `$6.89` for four pods. | Still `n = 1` and outcome-aware; it resolves terminal reachability but not seed stability, and must be declared before the runs to avoid another post-outcome rule. |

**My recommendation to the operator is option (c), not a lambda selection.**
It measures the actual decision variable at the registered terminal budget and
schedule for less than the quoted 12-pod joint-lane cost. Option (a) commits
the full lane while the registered lambda has the weakest measured calibration
trajectory; option (b) is currently non-operational because its measured
feasible set is empty. If the operator declines additional pilot spend, the
epistemically honest fallback is option (a) with calibration-failure risk
accepted explicitly, not an inferred switch to another lambda.

Lambda 0.2 is omitted from the proposed bridge because lambda 0.3 dominates it
on both 6,000-step objectives in this grid. That omission is itself an
operator-facing economy recommendation, not evidence that the dominance is
seed-stable; adding lambda 0.2 would add one quoted approximately 65-minute
pod if the operator wants the normalised-crossing region measured at full
budget.

## 6. Completeness and input integrity

`read_eval_series` accepted every eval record in every named input, validating
finiteness, documented metric domains, and completeness of the five-member
field-calibration group. Each 3,000-step capture has six eval records through
step 3000; each 6,000-step capture has twelve through step 6000. No arm was
dropped, repaired, imputed, or silently excluded.

The `run_start` records were read through telemetry's validated record
iterator for the regime check above: all report seed 42, CUDA, identical
model/data configuration apart from `field_loss_lambda`, and the declared
`max_steps` for their budget.

The md5s below were computed over the materialized DVC JSONL bytes in this
analysis session. The DVC directory pointer records
`47f69088c163251d64bcd469e5222571.dir` for `results/captured`.

| budget | lambda | evidence class | input | md5 |
|---:|---:|---|---|---|
| 6,000 | 0.1 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-0648f648-e1-5c-pilot6k-lam01.jsonl` | `26592928de226e9bb14a4b1a203add5f` |
| 6,000 | 0.2 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-a7eb6e99-e1-5c-pilot6k-lam02.jsonl` | `bb591ecd98d72dc1ce8f00ec4eb989db` |
| 6,000 | 0.3 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-6a0884cb-e1-5c-pilot6k-lam03.jsonl` | `e206b592e7ed1556bfd397d890a31b51` |
| 6,000 | 0.5 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-45230629-e1-5c-pilot6k-lam05.jsonl` | `0ff522300fe1e0affa1c0e97eba2151b` |
| 6,000 | 1.0 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-297b42cf-e1-5c-pilot6k-lam10.jsonl` | `a3bf6861ac7638014b6afdb8c58a3951` |
| 3,000 | 0.1 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-2d85b537-e1-5c-pilot-lam01.jsonl` | `87c9109130f6ac7dad6a5b1a523f8528` |
| 3,000 | 0.3 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-fd30c01c-e1-5c-pilot-lam03.jsonl` | `60f96e4174fc6e47b0fd9333780d0fe2` |
| 3,000 | 0.5 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-cccab401-e1-5c-pilot-lam05.jsonl` | `3aacf337605ee82aaf33467453d3cf05` |
| 3,000 | 1.0 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-c66e4903-e1-5c-pilot-lam10.jsonl` | `56a2b5f59da6040ce8d697cea91eff86` |

The standalone reference values are quoted measured inputs rather than a
capture re-read in this diagnostic. Their previously filed provenance hashes
are checkpoint `edc3e3e55ccc340bf565e14ac547618c` and standalone pretrain
results `893e37225011c97322f6cf5c0d68c4e0`; they are recorded here as
**QUOTED MEASURED INPUT**, not as hashes recomputed this session.
