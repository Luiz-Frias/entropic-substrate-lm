# E1.5a confirmatory analysis — frozen Fisher table

## 0. Scope and result

This is the fresh-session confirmatory analysis for rung **E1.5a**. The
binding design is `docs/e1-5a-preregistration.md`; every primary number below
was computed from the twelve DVC-captured JSONLs named in §2. Metrics were
read through `entropic_substrate.telemetry`, never from the live tracking
server. No E1.5c capture was read.

The registered result is:

- **freeze-both: `INCONCLUSIVE`** on the primary sufficiency clause. Its
  registered 97.5% CI is
  **[0.0013623925061083918, 0.019845607493891724]**, which crosses
  **ε = 0.01 nats/byte**. Mandatory direction rider: the CI is entirely
  above 0, so there is a **detectable freezing cost**, but the registered
  interval does not determine whether that cost is smaller or larger than ε.
- **input-only: `INCONCLUSIVE`** on the primary sufficiency clause. Its
  registered 97.5% CI is
  **[-0.007278657794102857, 0.012627357794103008]**, which crosses
  **ε = 0.01 nats/byte**. Mandatory direction rider: the CI contains 0, so
  there is **no detectable cost** under the registered interval.
- **Output-only contingency trigger: does not fire.** The registered ordering
  is `GEOMETRY-SUFFICIENT > INCONCLUSIVE > GEOMETRY-INSUFFICIENT`; input-only
  and freeze-both are both `INCONCLUSIVE`, so
  `label(input-only) < label(freeze-both)` is false.

These paired labels and riders are the verdict. Neither rider is optional,
and neither changes the primary ε-clause label.

## 1. Registered constants and evidence boundary

| item | value | evidence class |
|---|---:|---|
| Registered variants | freeze-both; input-only | **QUOTED** from preregistration |
| Registered seeds | {1, 2, 3, 4} | **QUOTED** |
| Primary metric | mean of final `k = 5` eval `nats_per_byte` records | **QUOTED** |
| Primary eval steps | {18000, 18500, 19000, 19500, 20000} | **QUOTED** |
| Paired gap | `g_s = y_variant,s - y_fisher,s`; positive means freezing costs nats/byte | **QUOTED** |
| n; df | 4; 3 | **QUOTED** |
| Total α; registered variants | 0.05; 2 | **QUOTED** |
| Per-variant two-sided CI | 97.5%; `alpha_per_variant = 0.025` | **QUOTED** |
| Critical value | `t_{0.9875,3} = 4.177` | **QUOTED** and used literally |
| Practical threshold | `ε = 0.01 nats/byte` | **QUOTED** |
| Looks | one; no stage 2 | **QUOTED** |
| Convergence target | `L* = 0.80 nats/byte` | **QUOTED** |
| Prior E1 cumulative `sd(d)` | `0.00300307217695479` | **QUOTED measured value** from the prior verdict; power context only, not used in this estimator |
| Preregistration commit | `be85f0c3f5e727f857a4cdb8ea3266d98cb230bd` | **MEASURED** from git history |

Clause precedence is applied verbatim: **sufficiency versus ε is the primary
verdict label**, deliberately inverting E1, and **direction versus 0 is the
mandatory co-equal rider**. Both are reported on every quotation above.

### Telemetry access

Primary and eval-series measurements used the public
`read_eval_series`/`final_eval_metric` readers. The train-vs-eval probe used
the public `read_train_series` reader because train records are required for
that registered probe. `data_order_digest` and run configuration are not
exposed by a public telemetry reader on this branch; those integrity-only
fields were read through `telemetry._iter_records`, the module's own validated
JSONL iterator. No independent JSONL parser was used. This is a disclosed
instrument-surface deviation; it does not alter the registered estimator or
decision rule.

## 2. Inputs and integrity

### 2.1 Content hashes

All hashes are MD5 values computed in this analysis session over the
materialized DVC captures.

| arm | seed | input | md5 |
|---|---:|---|---|
| fisher control | 1 | `results/captured/e1-fisher-bd0fb0d3-e1-fisher-s1.jsonl` | `f9e3498ab923b27ae233aa9e1b85c2bc` |
| fisher control | 2 | `results/captured/e1-fisher-56d129a1-e1-fisher-s2.jsonl` | `e57e834fc356ef35b9618c15ae151af8` |
| fisher control | 3 | `results/captured/e1-fisher-7d84caf7-e1-fisher-s3.jsonl` | `ba7d61140cc2b7cc281aa881d2f67e35` |
| fisher control | 4 | `results/captured/e1-fisher-6015d8d1-e1-fisher-s4.jsonl` | `8269945108deacfaa8f0adcce83c61bf` |
| freeze-both | 1 | `results/captured/e1-5a-both-4eea3777-e1-5a-both-s1.jsonl` | `0899540b921cfec3f0d6d3e11ec47140` |
| freeze-both | 2 | `results/captured/e1-5a-both-894279d3-e1-5a-both-s2.jsonl` | `14ad5269d67beb10da74729ba55dd99c` |
| freeze-both | 3 | `results/captured/e1-5a-both-4cb52e05-e1-5a-both-s3.jsonl` | `ad52f5224b1e560525674b616765dfc3` |
| freeze-both | 4 | `results/captured/e1-5a-both-4ecaec2b-e1-5a-both-s4.jsonl` | `39dfa3e358d9dd25c0c083c98469ae95` |
| input-only | 1 | `results/captured/e1-5a-input-a8557edd-e1-5a-input-s1.jsonl` | `0060c39eb12547d979183161d894ef74` |
| input-only | 2 | `results/captured/e1-5a-input-5a1426da-e1-5a-input-s2.jsonl` | `87f90e1e2f3252a345d1fb61da9dc5a5` |
| input-only | 3 | `results/captured/e1-5a-input-2e1f6b52-e1-5a-input-s3.jsonl` | `49d81a7b42767a4a88d4983e2871b7be` |
| input-only | 4 | `results/captured/e1-5a-input-7d223c8f-e1-5a-input-s4.jsonl` | `e6274065745c6e6c1fdcb2eed12e003c` |

Each input contains exactly 40 admissible eval records at 500-step intervals
from 500 through 20000, exactly 1000 admissible train records, and the final
five eval steps are exactly 18000, 18500, 19000, 19500, and 20000.

### 2.2 Pairing verification

The implementation-session claim was independently rechecked. Each digest
occurs exactly once in each run; all **8/8 variant-control pairs match**.

| seed | `data_order_digest` | freeze-both match | input-only match |
|---:|---|---|---|
| 1 | `3f05412324b70543fba0e6537d130ecf240b1339355ea4a366d0db4f18626e27` | yes | yes |
| 2 | `8a39105f5aa9f1ed382a5103765f3dcd106be70404044231f9805d10a6d0f91e` | yes | yes |
| 3 | `4f26fe9084deda26957f274ac7147c84df6f91d8f9e7daf0d40d10c208c67a26` | yes | yes |
| 4 | `6e4e11046c3f652c28e813d31aad31b10dd76d1d84623439ae9b38bb11c8cb5b` | yes | yes |

Zero pairs are invalidated by pairing.

### 2.3 Exclusion rule

The only registered exclusion is: **a non-finite final loss drops the whole
pair and is always reported**. `final_eval_metric(path, "loss")` returned a
finite value for all twelve inputs. Therefore:

- freeze-both: 4/4 pairs retained; 0 excluded;
- input-only: 4/4 pairs retained; 0 excluded;
- no other exclusion, outlier handling, or winsorization was applied.

### 2.4 Regime check

Run-start telemetry verifies the shared registered regime for every seed:
enwik8; byte tokenizer; `d_model = 768`; 12 layers; 12 heads; batch size 16;
sequence length 1024; learning rate 0.0006; cosine schedule to 20000 steps;
eval interval 500; bf16; CUDA; and Fisher √p initialization. Freeze-both is
tied with `embedding_freeze = "both"` and reports 85150464 parameters.
Input-only is untied with `embedding_freeze = "input"` and reports 85347072
parameters. Those are the registered arm definitions, not post-hoc
differences.

The E1 controls predate later config-schema fields present in the E1.5a run
records (`state_source`, zero-valued Gram/field defaults, and related field
settings). The E1.5a records report `state_source = "table"`,
`gram_lambda = 0.0`, and `field_loss_lambda = 0.0`. This is an observed config
shape difference; no mechanism claim is inferred from it.

The registered integrity chain is only partially checkable from these
captures: `config_hash` and `git_sha` are null in all inspected run-start
records, and the pod launch/container-image digest is not carried in these
JSONLs. Pairing, configs, seeds, record completeness, finiteness, and content
hashes are verified; the config-hash → git-SHA → image-digest portion remains
unverified from the admissible inputs and is **not silently inferred**.

## 3. Primary endpoint

### 3.1 Freeze-both versus trainable Fisher

| seed | freeze-both k=5 | fisher k=5 | `g_s` k=5 | freeze-both k=1 | fisher k=1 | `g_s` k=1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.775032 | 0.7702034 | 0.004828600000000072 | 0.775184 | 0.770528 | 0.0046559999999999935 |
| 2 | 0.7827998 | 0.7672032 | 0.015596600000000072 | 0.782955 | 0.767641 | 0.015313999999999939 |
| 3 | 0.778674 | 0.767401 | 0.011272999999999977 | 0.778799 | 0.765805 | 0.012994000000000061 |
| 4 | 0.7755178 | 0.7647999999999999 | 0.01071780000000011 | 0.774974 | 0.766741 | 0.008233000000000046 |

Registered k=5 statistics:

| quantity | measured value |
|---|---:|
| n; df | 4; 3 |
| mean gap | 0.010604000000000058 |
| sample sd | 0.004424997603012529 |
| se | 0.0022124988015062644 |
| registered critical value | 4.177 |
| CI half-width | 0.009241607493891666 |
| registered 97.5% CI | [0.0013623925061083918, 0.019845607493891724] |

Application of the registered clauses:

1. **Primary sufficiency clause:** the upper bound
   0.019845607493891724 is not below ε = 0.01, and the lower bound
   0.0013623925061083918 is not above ε. Therefore **`INCONCLUSIVE`**.
2. **Mandatory direction rider:** the CI is entirely above 0. Therefore
   freezing both tables has a **detectable cost**. The CI is not entirely
   within `(0, ε)`, so practical relevance is unresolved rather than declared
   sub-ε.

### 3.2 Input-only versus trainable Fisher

| seed | input-only k=5 | fisher k=5 | `g_s` k=5 | input-only k=1 | fisher k=1 | `g_s` k=1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7669370000000001 | 0.7702034 | -0.0032663999999998916 | 0.769101 | 0.770528 | -0.001426999999999956 |
| 2 | 0.7686792 | 0.7672032 | 0.0014760000000000328 | 0.767307 | 0.767641 | -0.0003340000000000565 |
| 3 | 0.7719242000000001 | 0.767401 | 0.0045232000000000605 | 0.773284 | 0.765805 | 0.007479000000000013 |
| 4 | 0.7727646 | 0.7647999999999999 | 0.0079646000000001 | 0.774358 | 0.766741 | 0.007616999999999985 |

Registered k=5 statistics:

| quantity | measured value |
|---|---:|
| n; df | 4; 3 |
| mean gap | 0.0026743500000000753 |
| sample sd | 0.004765624991191254 |
| se | 0.002382812495595627 |
| registered critical value | 4.177 |
| CI half-width | 0.009953007794102933 |
| registered 97.5% CI | [-0.007278657794102857, 0.012627357794103008] |

Application of the registered clauses:

1. **Primary sufficiency clause:** the upper bound
   0.012627357794103008 is not below ε = 0.01, and the interval is not
   entirely above ε. Therefore **`INCONCLUSIVE`**.
2. **Mandatory direction rider:** the CI contains 0. Therefore there is
   **no detectable cost** under the registered interval.

## 4. Registered output-only trigger

The registration orders labels:

`GEOMETRY-SUFFICIENT > INCONCLUSIVE > GEOMETRY-INSUFFICIENT`.

The trigger is the strict inversion
`label(input-only) < label(freeze-both)`. Measured labels are
`INCONCLUSIVE = INCONCLUSIVE`; therefore the strict inequality is false.
**The `e1-5a-output` contingency arm does not run under this registration.**

## 5. Predeclared probes — report, never decide

Nothing in this section changes §3 or §4.

### 5.1 Steps to L* = 0.80 nats/byte

Every run reaches the target. The value shown is the first eval at or below
0.80.

| seed | fisher step (value) | freeze-both step (value) | input-only step (value) |
|---:|---|---|---|
| 1 | 11500 (0.79988) | 14000 (0.796037) | 12000 (0.798407) |
| 2 | 11500 (0.798781) | 14500 (0.797715) | 11500 (0.799858) |
| 3 | 11500 (0.798781) | 14000 (0.796908) | 13000 (0.798387) |
| 4 | 11500 (0.799785) | 14000 (0.795117) | 12500 (0.798515) |

Measured median steps are 11500 for Fisher, 14000 for freeze-both, and
12250 for input-only. Measured paired delays versus Fisher are
{2500, 3000, 2500, 2500} steps for freeze-both and
{500, 0, 1500, 1000} for input-only. This is directional only; no registered
decision threshold applies.

### 5.2 Per-eval-step paired gap trajectory

Each row reports `mean_s(y_variant,s - y_fisher,s)`, sample sd across the
four paired gaps, and the number of positive gaps. All quantities are
**MEASURED** from matched arms under the same seed/data order.

| step | freeze-both mean gap | freeze-both sd | positive | input-only mean gap | input-only sd | positive |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.07061875000000001 | 0.01827571686464486 | 4/4 | 0.021168500000000035 | 0.01752480134932585 | 4/4 |
| 1000 | 0.055654750000000086 | 0.003639012354563595 | 4/4 | 0.014199000000000017 | 0.003931624515812003 | 4/4 |
| 1500 | 0.042611250000000045 | 0.0040753967802737824 | 4/4 | 0.008312250000000021 | 0.004106007093272036 | 4/4 |
| 2000 | 0.03678575000000006 | 0.0024552525158660814 | 4/4 | 0.004310249999999988 | 0.005927966788874557 | 3/4 |
| 2500 | 0.031092750000000002 | 0.003861626460012273 | 4/4 | 0.007283999999999985 | 0.0018567862917776854 | 4/4 |
| 3000 | 0.027961499999999972 | 0.0071311860864795885 | 4/4 | 0.00830450000000002 | 0.003748223445847372 | 4/4 |
| 3500 | 0.026424499999999962 | 0.003818028592524325 | 4/4 | 0.008290249999999971 | 0.004163958563274434 | 4/4 |
| 4000 | 0.025010249999999984 | 0.003009109876691074 | 4/4 | 0.004545250000000001 | 0.0006316319471548966 | 4/4 |
| 4500 | 0.02429624999999999 | 0.0033359685625017677 | 4/4 | 0.008035249999999994 | 0.0012868189137041266 | 4/4 |
| 5000 | 0.022696750000000016 | 0.005415313925957281 | 4/4 | 0.006077999999999972 | 0.005742746671526891 | 3/4 |
| 5500 | 0.02210300000000004 | 0.004520874251734928 | 4/4 | 0.010264750000000017 | 0.0011289530179182394 | 4/4 |
| 6000 | 0.024253250000000004 | 0.0017315490896882605 | 4/4 | 0.008900500000000006 | 0.0035443047743292516 | 4/4 |
| 6500 | 0.018057250000000052 | 0.004699015810074863 | 4/4 | 0.006798500000000013 | 0.0055763689799008115 | 4/4 |
| 7000 | 0.01954724999999996 | 0.0030851889596371215 | 4/4 | 0.0036059999999999703 | 0.004489551870732773 | 3/4 |
| 7500 | 0.021586750000000016 | 0.0018790824667019373 | 4/4 | 0.007185249999999976 | 0.006576213316440804 | 3/4 |
| 8000 | 0.022752750000000016 | 0.005367308100901239 | 4/4 | 0.0048729999999999885 | 0.003912417155672446 | 4/4 |
| 8500 | 0.02254400000000001 | 0.002152106100234498 | 4/4 | 0.0076157500000000045 | 0.006523454727621941 | 3/4 |
| 9000 | 0.021686750000000005 | 0.003688301449628762 | 4/4 | 0.005135000000000028 | 0.008158642820142765 | 2/4 |
| 9500 | 0.019083250000000024 | 0.002703500493187755 | 4/4 | 0.004228250000000017 | 0.004433809977509905 | 4/4 |
| 10000 | 0.01783925 | 0.0025498272327094634 | 4/4 | 0.0041932500000000095 | 0.005332346098732434 | 3/4 |
| 10500 | 0.017989000000000033 | 0.006353595779818981 | 4/4 | 0.005888000000000032 | 0.005763219644145656 | 4/4 |
| 11000 | 0.018602500000000022 | 0.00256387434689506 | 4/4 | 0.003555750000000024 | 0.0025758218591355004 | 4/4 |
| 11500 | 0.019561250000000002 | 0.002505650730515584 | 4/4 | 0.005681750000000013 | 0.004526974808486017 | 4/4 |
| 12000 | 0.016525749999999978 | 0.0018920708892639492 | 4/4 | 0.0035424999999999762 | 0.0070204976319346675 | 3/4 |
| 12500 | 0.016836500000000032 | 0.0028856307802627252 | 4/4 | 0.006125249999999999 | 0.001982938791289337 | 4/4 |
| 13000 | 0.015664999999999984 | 0.0019167576789985838 | 4/4 | 0.005180999999999991 | 0.0038396061429960752 | 4/4 |
| 13500 | 0.014677500000000038 | 0.0031867898686086336 | 4/4 | 0.004248500000000016 | 0.004288970078390989 | 3/4 |
| 14000 | 0.01587074999999996 | 0.004683215802202561 | 4/4 | 0.0028829999999999967 | 0.004119696995330266 | 3/4 |
| 14500 | 0.013636250000000016 | 0.00523224396570595 | 4/4 | 0.004273000000000027 | 0.005062724760442757 | 3/4 |
| 15000 | 0.01549600000000001 | 0.006428346184413704 | 4/4 | 0.0058952500000000185 | 0.002899874638094993 | 4/4 |
| 15500 | 0.01312350000000001 | 0.004369505959869236 | 4/4 | 0.002581749999999966 | 0.0045655841083041705 | 3/4 |
| 16000 | 0.012154500000000013 | 0.003924426200096035 | 4/4 | 0.0025390000000000135 | 0.0020723392579401253 | 3/4 |
| 16500 | 0.011151250000000001 | 0.004773578558063077 | 4/4 | 0.003464999999999968 | 0.002430940970077221 | 4/4 |
| 17000 | 0.01148374999999996 | 0.006453062651434073 | 4/4 | 0.0034842499999999665 | 0.004225762682640867 | 3/4 |
| 17500 | 0.011022500000000018 | 0.006189649990104422 | 4/4 | 0.0018282500000000035 | 0.005234404415340213 | 3/4 |
| 18000 | 0.009833250000000016 | 0.004447505321338399 | 4/4 | 0.002257250000000044 | 0.004297626428235329 | 3/4 |
| 18500 | 0.00957849999999999 | 0.005191752465850693 | 4/4 | 0.0011740000000000361 | 0.00408124196783283 | 3/4 |
| 19000 | 0.011674250000000025 | 0.005456400790203477 | 4/4 | 0.003088750000000001 | 0.0057798253361729356 | 3/4 |
| 19500 | 0.011634749999999972 | 0.004053724326673743 | 4/4 | 0.003518000000000021 | 0.005695250477371432 | 3/4 |
| 20000 | 0.01029925000000001 | 0.004779297533808346 | 4/4 | 0.0033337499999999964 | 0.004886936966717154 | 2/4 |

Measured trajectory reading, with no causal inference: freeze-both has a
positive gap in 4/4 pairs at all 40 evaluations and declines from a mean gap
of 0.07061875000000001 at step 500 to 0.01029925000000001 at step 20000.
Input-only declines from 0.021168500000000035 to
0.0033337499999999964, with mixed per-seed signs at the terminal evaluation
(2/4 positive). These are descriptive probe results, not extra looks.

### 5.3 Train-vs-eval gap

The preregistration names the probe but does not specify a smoothing window.
The operationalization here is therefore explicitly **EXPLORATORY**:
terminal eval `nats_per_byte` minus the final train record's
`nats_per_byte`, both at step 20000.

| arm | seed 1 | seed 2 | seed 3 | seed 4 | mean | sample sd |
|---|---:|---:|---:|---:|---:|---:|
| fisher | 0.089032 | 0.11662499999999998 | 0.10394799999999993 | 0.021333999999999964 | 0.08273474999999997 | 0.04245883684523164 |
| freeze-both | 0.07117399999999996 | 0.10887799999999992 | 0.09684300000000001 | 0.0015370000000000106 | 0.06960799999999998 | 0.048027740324386115 |
| input-only | 0.08572900000000006 | 0.11130499999999999 | 0.10882099999999995 | 0.02597399999999994 | 0.08295724999999998 | 0.03969593544579429 |

No registered decision rule applies, and no causal overfitting claim is made.

### 5.4 k-sensitivity

The same registered estimator and `t = 4.177` were applied at
`k in {1, 3, 5}`. Only k=5 is the registered primary endpoint.

| variant | k | mean gap | sample sd | 97.5% CI | primary ε-label | direction rider |
|---|---:|---:|---:|---|---|---|
| freeze-both | 1 | 0.01029925000000001 | 0.004779297533808346 | [0.00031768710064128056, 0.02028081289935874] | INCONCLUSIVE | entirely above 0 |
| freeze-both | 3 | 0.011202750000000011 | 0.004465318821307838 | [0.001876931641698592, 0.02052856835830143] | INCONCLUSIVE | entirely above 0 |
| freeze-both | **5** | **0.010604000000000058** | **0.004424997603012529** | **[0.0013623925061083918, 0.019845607493891724]** | **INCONCLUSIVE** | **entirely above 0** |
| input-only | 1 | 0.0033337499999999964 | 0.004886936966717154 | [-0.006872617854988779, 0.013540117854988772] | INCONCLUSIVE | contains 0 |
| input-only | 3 | 0.0033135000000000248 | 0.0052282600922861425 | [-0.007605721202739583, 0.014232721202739633] | INCONCLUSIVE | contains 0 |
| input-only | **5** | **0.0026743500000000753** | **0.004765624991191254** | **[-0.007278657794102857, 0.012627357794103008]** | **INCONCLUSIVE** | **contains 0** |

No primary label or direction rider changes across k.

### 5.5 Frozen-table drift and trainable-half Gram retention

The admissible JSONLs contain no angular-drift, frozen-table, Gram-retention,
or checkpoint diagnostic field. Under the brief's JSONL-only input boundary,
this probe is **not answerable** in this analysis:

- frozen-table angular drift is **not measured here**, so it is not asserted
  to be zero;
- trainable-half Gram retention for input-only is **not measured here**;
- the missing measurement is an E1.5-D endpoint diagnostic run over each new
  E1.5a final checkpoint and its initialization/reference table, reporting
  angular drift of each frozen half and Gram retention of the trainable half.

If that later measurement reports any nonzero angular drift on a frozen
table, the preregistration classifies it as a **DEFECT, not a finding**.

## 6. Limitations and lifecycle status

1. **Narrow estimand.** The verdict is scoped exactly as registered: this
   shared configuration, approximately 85M parameters, enwik8, and 20000
   steps. It is not a tuned-frontier or scale-general claim.
2. **Integrity chain gap.** Pairing and captured content are verified, but
   null `config_hash`/`git_sha` fields and the absent launch image digest
   prevent verification of the complete registered chain from JSONL alone.
   This is reported, not repaired by inference.
3. **Checkpoint probes unavailable.** The JSONL-only evidence boundary cannot
   establish frozen-table invariance or trainable-half Gram retention.
4. **Train-gap operationalization.** The registration did not specify its
   smoothing/window rule; the terminal-minus-terminal reading in §5.3 is
   labelled exploratory and cannot decide the verdict.
5. **Artifact schema extension.** The lifecycle protocol illustrates scalar
   `mean_d`, `sd_d`, `ci`, and `verdict` fields, but E1.5a registers two
   independent variants and no aggregate rung rule. The machine artifact
   preserves `mean_d`, `sd_d`, and `ci` as per-variant objects and carries
   full per-variant records. Because both registered primary labels happen to
   be `INCONCLUSIVE`, the required top-level scalar `verdict` is also
   `INCONCLUSIVE`; `verdict_scope` states that this is the common label, not
   an invented aggregate decision rule. `alpha_per_look = 0.05` records the
   single registered look, while `alpha_per_variant = 0.025` records the
   Bonferroni split that produces each 97.5% CI.
6. **Review pending.** `docs/reviews/e1-5a-review.md` does not yet exist and
   was neither read nor created in this analysis session. The JSON records
   the expected review path with `review_status = "pending"`; filing remains
   the next lifecycle phase.

## 7. Filed analysis verdict

> **freeze-both: `INCONCLUSIVE`** on geometry sufficiency at
> **ε = 0.01 nats/byte**; mandatory rider: **detectable freezing cost**
> because the registered CI is entirely above 0.

> **input-only: `INCONCLUSIVE`** on geometry sufficiency at
> **ε = 0.01 nats/byte**; mandatory rider: **no detectable cost** because the
> registered CI contains 0.

> **`e1-5a-output` trigger: NOT FIRED** because
> `INCONCLUSIVE` is not less than `INCONCLUSIVE` under the registered ordering.
