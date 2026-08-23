# E1.5c confirmatory analysis — frozen four-arm contract

## 0. Scope and filed result

This is the fresh-session confirmatory analysis for rung **E1.5c, frozen
regime only**. The binding design is `docs/e1-5c-preregistration.md`, including
its three in-file amendments. All run measurements came from the twenty
DVC-captured JSONLs named in §2 and were read through
`entropic_substrate.telemetry`; the live MLflow server was never queried. No
joint, pilot, bridge, or LR-diagnostic capture was opened or pooled.

The registered frozen-lane result is:

- **`Thesis supported`: false.** The registration requires all three clauses.
  Clause (1) fails: the registered `F−T` learning-curve estimator is
  **MEASURED** `-0.11300426250000001` nats/byte with **MEASURED** 95% CI
  **[-0.11800596590322686, -0.10800255909677316]**. The estimator is
  `L_T - L_F`, so the entirely negative interval means F is worse than T.
  Clause (2) also fails: the terminal `L_F - L_T` point gap is **MEASURED**
  `0.05196629999999999` nats/byte with **MEASURED** 95% CI
  **[0.042401195669669725, 0.06153140433033026]**, not below the **QUOTED**
  noninferiority margin `δ = 0.005`. Clause (3) passes: terminal frozen field
  CE is **MEASURED** `1.256355`, a **MEASURED** `-0.002525999999999806`
  difference from the **QUOTED** standalone reference `1.258881`, within the
  **QUOTED** `±0.02` tolerance.
- **`Practically valuable`: false.** F reaches none of the **QUOTED** four
  registered budgets with a positive `F−T` point improvement, never reaches
  the **QUOTED** `L* = 0.80` target by step **QUOTED** `20000`, and the amended
  F matched-total arm does not match or beat T at terminal under **QUOTED**
  `δ = 0.005`.
- **`Compute-efficient`: not established.** F never reaches `L*`, so
  GPU-seconds-to-target is censored for every `F−T` pair. Estimated FLOPs,
  FLOPs-to-target, peak VRAM, and field FLOP share are absent from the
  admissible telemetry; those measurements are not filled by inference.

The causal ladder is nevertheless informative. `H−T` is negative,
`P−H` is negative, and `F−P` is positive. Thus the **MEASURED** evidence says
that contextual preprocessing at all is worse than the static table at this
fixed configuration; raw probabilities are worse than hidden state; and the
square-root map repairs part, but not all, of that loss. The registered direct
`F−H` contrast confirms the residual deficit. This decomposition is not a
replacement for the failed headline `F−T` clauses.

## 1. Registered constants and evidence boundary

Every numeric value in this table is **QUOTED** from the binding
preregistration; none is re-estimated here.

| item | registered value | evidence class |
|---|---:|---|
| Frozen arms | T; H; P; F; F matched-total | **QUOTED** |
| Confirmatory seeds | `{1, 2, 3, 4}` | **QUOTED** |
| Frozen field checkpoint sharing | same frozen `p_phi` checkpoint for every field arm | **QUOTED** |
| Field estimator shape | 2 layers; `d = 128`; 4 heads; tied 256-way head | **QUOTED** |
| Primary metric | last-`k = 5` eval mean of `nats_per_byte` | **QUOTED** |
| Eval interval | 500 steps | **QUOTED** |
| Registered source-byte budgets | `{49.152, 98.304, 196.608, 294.912}M` bytes | **QUOTED** |
| Registered budget steps | `{3000, 6000, 12000, 18000}` | **QUOTED** |
| Bytes per step | 16,384 = batch 16 × sequence 1024 | **QUOTED** |
| Per-seed primary statistic | `(1/4) * sum_B d_B,s` | **QUOTED** |
| Primary pair count; df | `n = 4`; `df = 3` | **QUOTED** |
| Primary CI | paired, one look, two-sided 95% | **QUOTED** |
| Primary critical value | `t_{0.975,3} = 3.182` | **QUOTED**, used literally |
| Learning practical threshold | `epsilon_L = 0.01` nats/byte | **QUOTED** |
| Convergence target | `L* = 0.80` nats/byte | **QUOTED** |
| Crossing rule | first eval whose last-5-ending-there mean is `<= L*` | **QUOTED** |
| Censoring boundary | no crossing by step 20000 censors the pair | **QUOTED** |
| Co-primary minimum | at least 3 uncensored pairs, else `INCONCLUSIVE` | **QUOTED** |
| Co-primary direction | one-sided 95% LCB of mean(S) `> 0` = `convergence improvement` | **QUOTED** |
| Co-primary practical threshold | point estimate `>= 10%` = `epsilon_S` | **QUOTED** |
| Co-primary strong threshold | one-sided LCB `> 10%` = `strong support` | **QUOTED** |
| Terminal step and byte budget | 20000; 327.68M bytes | **QUOTED** |
| Terminal noninferiority margin | `delta = 0.005` nats/byte | **QUOTED** |
| Frozen calibration reference | standalone field CE `1.258881` | **QUOTED** |
| Frozen calibration tolerance | `±0.02` nats/byte | **QUOTED** |
| Calibration ECE bins | 15 | **QUOTED** |
| Shared learning rate | `6e-4` | **QUOTED** |
| Shared schedule | cosine to 20000 steps | **QUOTED** |
| Shared regime | approximately 85M parameters; enwik8; byte level | **QUOTED** |
| Original matched-total width | `d_model = 756`; 12 heads; head dimension 63 | **QUOTED** pre-amendment |
| Amended matched-total width | `d_model = 744`; 12 heads; head dimension 62 | **QUOTED** Amendment 1 |
| Original/amended matched totals | 82,940,244 → 80,344,872 parameters | **QUOTED** Amendment 1 |
| Static baseline total | 85,150,464 parameters | **QUOTED** Amendment 1 |
| Amended shortfall | 5.6% under the static baseline | **QUOTED** Amendment 1 |
| Looks | one | **QUOTED** |
| Joint-lane constants | not applied; joint is a separate registered-secondary report | **QUOTED scope boundary** |
| Initial preregistration commit | `975d9a376acfc0c0ca7a45e967dc4ed156f140e9` | **MEASURED** from `git log --follow -- docs/e1-5c-preregistration.md` |
| Binding amendment commit | `8f31e62cba024c0e97af1c37c48541803b87004b` | **MEASURED** from the same file history |

Clause precedence is applied verbatim: direction of the `F−T` primary CI
versus zero is the primary label, and practical relevance versus
`epsilon_L` is the mandatory co-equal rider. The hierarchical rule is quoted
as three separate clauses in §7; it is never collapsed into a substitute
label.

### Telemetry access

Eval measurements used the public `read_eval_series` and
`final_eval_metric` readers. Seeds used the public `read_seed` reader.
`run_start` configuration, `data_order_digest`, `wall_s`, and run-end records
are not exposed by dedicated public readers on this branch, so those
integrity and systems fields were read through `telemetry._iter_records`, the
module's validated JSONL iterator. No independent JSON parser was used.

## 2. Inputs and integrity

### 2.1 Content hashes

All hashes below are **MEASURED** MD5 values computed over the materialized
DVC captures in this analysis session.

| arm | seed | input | MD5 (**MEASURED**) |
|---|---:|---|---|
| T | 1 | `results/captured/e1-fisher-bd0fb0d3-e1-fisher-s1.jsonl` | `f9e3498ab923b27ae233aa9e1b85c2bc` |
| T | 2 | `results/captured/e1-fisher-56d129a1-e1-fisher-s2.jsonl` | `e57e834fc356ef35b9618c15ae151af8` |
| T | 3 | `results/captured/e1-fisher-7d84caf7-e1-fisher-s3.jsonl` | `ba7d61140cc2b7cc281aa881d2f67e35` |
| T | 4 | `results/captured/e1-fisher-6015d8d1-e1-fisher-s4.jsonl` | `8269945108deacfaa8f0adcce83c61bf` |
| H | 1 | `results/captured/e1-5c-hidden-07b96872-e1-5c-hidden-frozen-s1.jsonl` | `0800209979d30712a383bb7df26498a0` |
| H | 2 | `results/captured/e1-5c-hidden-56cb0255-e1-5c-hidden-frozen-s2.jsonl` | `8eb7a218515969b7f666072bb1aaa616` |
| H | 3 | `results/captured/e1-5c-hidden-fdbd546a-e1-5c-hidden-frozen-s3.jsonl` | `267d492eea241b62b8747fb6f50b5f5f` |
| H | 4 | `results/captured/e1-5c-hidden-473f6f95-e1-5c-hidden-frozen-s4.jsonl` | `54f866157d8cacdb48a6f2d77106658c` |
| P | 1 | `results/captured/e1-5c-prob-2790013d-e1-5c-prob-frozen-s1.jsonl` | `44be506cae8c8e81eaeb8d4fe03d440e` |
| P | 2 | `results/captured/e1-5c-prob-dd1af0fc-e1-5c-prob-frozen-s2.jsonl` | `9f3630a7d47827aedd5c7f1af87477c3` |
| P | 3 | `results/captured/e1-5c-prob-ecc1e957-e1-5c-prob-frozen-s3.jsonl` | `6d461b6c4f3ca6687bc42c994c3dc9a5` |
| P | 4 | `results/captured/e1-5c-prob-ea3e2da0-e1-5c-prob-frozen-s4.jsonl` | `4c04187773de5cba98a9aa1df8e289f4` |
| F | 1 | `results/captured/e1-5c-field-08356fe7-e1-5c-field-frozen-s1.jsonl` | `fc1f9d951005536c036221d9a3ce08b6` |
| F | 2 | `results/captured/e1-5c-field-9697c9f4-e1-5c-field-frozen-s2.jsonl` | `5511d7b22fe2d6ce9f26a556b87b07d9` |
| F | 3 | `results/captured/e1-5c-field-60eec6a2-e1-5c-field-frozen-s3.jsonl` | `c5c972eb73e03cdc1685f6966da0a40a` |
| F | 4 | `results/captured/e1-5c-field-768068a9-e1-5c-field-frozen-s4.jsonl` | `6164e004b3f311cc7fd850b02bbd2e34` |
| F matched-total | 1 | `results/captured/e1-5c-field-matched-total-c1be4bc8-e1-5c-field-matched-total-frozen-s1.jsonl` | `33827cd400291cced7513e655e98b064` |
| F matched-total | 2 | `results/captured/e1-5c-field-matched-total-12d8c962-e1-5c-field-matched-total-frozen-s2.jsonl` | `5dd49f8dc6b1aeea2c7d84538fd92f68` |
| F matched-total | 3 | `results/captured/e1-5c-field-matched-total-a900cb5d-e1-5c-field-matched-total-frozen-s3.jsonl` | `f41daf8fe76b4e9a292e858c47d0dc20` |
| F matched-total | 4 | `results/captured/e1-5c-field-matched-total-729217e7-e1-5c-field-matched-total-frozen-s4.jsonl` | `1ef5457eadd7e5c3cfc249c0c84287be` |

Every input contains **MEASURED** 40 admissible eval records at the **QUOTED**
500-step interval through step **QUOTED** 20000 and one `run_start` plus one
`run_end`. `dvc status results/captured.dvc` reports the data and pipelines
up to date. The materialized capture directory has **MEASURED** DVC directory
MD5 `b779345128e5484f99b3336383c5dbe7.dir`.

### 2.2 Extended integrity chain

| link | hash | evidence class |
|---|---|---|
| `data/enwik8.dvc` | `7722de7ace6575299cd8ab788c2fd201.dir` | **MEASURED** from current DVC pointer |
| `models/field-pphi-v1.pt` | `edc3e3e55ccc340bf565e14ac547618c` | **MEASURED** materialized MD5; matches DVC pointer |
| `results/field-pretrain-s42.jsonl` | `893e37225011c97322f6cf5c0d68c4e0` | **MEASURED** materialized MD5; matches DVC pointer; contents not analyzed |
| Wave image digest | unavailable in admissible JSONLs | declared gap |
| Per-run `config_hash` | unavailable in inspected `run_start` records | declared gap |
| Per-run `git_sha` | unavailable in inspected `run_start` records | declared gap |

The pretrain JSONL was hashed only for the registered integrity chain. Its
metrics were not parsed or pooled; the standalone CE used by the calibration
gate is the **QUOTED** registered reference.

### 2.3 Pairing verification

Each digest occurs once in each run. All **MEASURED** 16/16 non-control
arm-to-T seed pairs match; therefore zero pairs are invalidated.

| seed | `data_order_digest` | H | P | F | F matched-total |
|---:|---|---|---|---|---|
| 1 | `3f05412324b70543fba0e6537d130ecf240b1339355ea4a366d0db4f18626e27` | match | match | match | match |
| 2 | `8a39105f5aa9f1ed382a5103765f3dcd106be70404044231f9805d10a6d0f91e` | match | match | match | match |
| 3 | `4f26fe9084deda26957f274ac7147c84df6f91d8f9e7daf0d40d10c208c67a26` | match | match | match | match |
| 4 | `6e4e11046c3f652c28e813d31aad31b10dd76d1d84623439ae9b38bb11c8cb5b` | match | match | match | match |

### 2.4 Exclusion and regime checks

The only registered exclusion is a non-finite final loss, which drops the
whole pair. `final_eval_metric(path, "loss")` returned a finite value for
all **MEASURED** 20/20 inputs. Zero pairs are excluded and no other outlier
handling is applied.

Every `run_start` verifies enwik8, byte tokenizer, batch **MEASURED** 16,
sequence length **MEASURED** 1024, 12 layers, 12 heads, learning rate
**MEASURED** 0.0006, cosine schedule to **MEASURED** 20000 steps, eval
interval **MEASURED** 500, bf16, CUDA, and `resume = null`. The field arms
all report `state_source = "field"`, `field_mode = "frozen"`,
`field_loss_lambda = 0.0`, and the same checkpoint path. State modes are
`hidden` for H, `prob` for P, and `sqrt_prob` for F and F matched-total.

Run-start parameter counts are **MEASURED** 85,150,464 (T), 85,675,392 (H),
85,577,088 (P), 85,577,088 (F), and 80,344,872 (F matched-total). The
matched-total shortfall computed from the measured run-start counters is
**MEASURED** `0.05643647461510016` of T, consistent with Amendment 1's
**QUOTED** 5.6% description.

## 3. Scientific primary — full contrast ladder

For each named substitution `candidate−base`, arithmetic follows the
registered loss-improvement orientation `L_base - L_candidate`; positive
means the candidate is better. Each per-seed value is the **QUOTED** average
over the four registered budgets. All statistics in this section are
**MEASURED**; every CI uses the **QUOTED** literal `t = 3.182`.

The user-requested decomposed ladder is H−T, P−H, F−P, and F−T. The binding
text also registers direct F−H, so it is reported rather than silently
dropped. F matched-total−T uses the registration's same estimators.

| contrast | per-seed curve averages | mean | sample sd | 95% CI | direction vs 0 | practical rider vs `epsilon_L` |
|---|---|---:|---:|---|---|---|
| H−T | `[-0.07776660000000005, -0.07165659999999999, -0.07498410000000003, -0.07986584999999999]` | -0.07606828750000001 | 0.0035564118350698702 | [-0.08172653872959618, -0.07041003627040385] | entirely below 0; H worse | point average is below 0; no registered budget favours H |
| P−H | `[-0.05117925000000004, -0.05506295, -0.0502214, -0.04665345000000001]` | -0.05077926250000001 | 0.0034565641877387877 | [-0.05627865612269242, -0.0452798688773076] | entirely below 0; P worse | point average is below 0; no registered budget favours P |
| F−P | `[0.016185750000000027, 0.01796605000000004, 0.010796, 0.010425349999999972]` | 0.01384328750000001 | 0.0038058095528142074 | [0.007788244501472605, 0.019898330498527413] | entirely above 0; F better | point average is `>= 0.01`; budgets 6000, 12000, and 18000 are `>= 0.01` |
| F−H (direct registered) | `[-0.03499350000000001, -0.03709689999999996, -0.0394254, -0.03622810000000004]` | -0.036935975 | 0.001870599710208813 | [-0.039912099138942225, -0.03395985086105778] | entirely below 0; F worse | point average is below 0; no registered budget favours F |
| **F−T (headline)** | `[-0.11276010000000006, -0.10875349999999995, -0.11440950000000003, -0.11609395000000003]` | **-0.11300426250000001** | **0.003143748210701978** | **[-0.11800596590322686, -0.10800255909677316]** | **entirely below 0; F worse** | **point average is below 0; no registered budget favours F** |
| F matched-total−T | `[-0.11125164999999998, -0.10953084999999999, -0.11231615, -0.11522545000000001]` | -0.112081025 | 0.002389835961420794 | [-0.11588325401462049, -0.10827879598537951] | entirely below 0; matched-total F worse | point average is below 0; no registered budget favours matched-total F |

### 3.1 Budget-indexed paired readouts

Every number in this table is **MEASURED**. Cells are paired mean
improvement `[95% CI]` in nats/byte.

| contrast | 3000 / 49.152M | 6000 / 98.304M | 12000 / 196.608M | 18000 / 294.912M |
|---|---|---|---|---|
| H−T | -0.17093390000000003 `[-0.17780351887144383, -0.16406428112855623]` | -0.0739619 `[-0.08609077917334453, -0.06183302082665547]` | -0.03387545000000003 `[-0.039468281365343645, -0.028282618634656413]` | -0.025501899999999994 `[-0.02986730484232308, -0.021136495157676908]` |
| P−H | -0.03903555000000003 `[-0.05017044261771331, -0.027900657382286757]` | -0.060858050000000025 `[-0.0771115018107803, -0.044604598189219746]` | -0.05545039999999998 `[-0.06388214096210525, -0.04701865903789472]` | -0.04777305000000001 `[-0.05607914370878632, -0.0394669562912137]` |
| F−P | -0.0004828499999999236 `[-0.008731519221714884, 0.007765819221715037]` | 0.018646549999999984 `[0.007853971654083658, 0.02943912834591631]` | 0.019635050000000015 `[0.0081398194860377, 0.03113028051396233]` | 0.017574399999999962 `[0.008063875972608487, 0.02708492402739144]` |
| F−H | -0.039518399999999954 `[-0.04416141502831835, -0.03487538497168156]` | -0.04221150000000004 `[-0.05304107206343664, -0.03138192793656344]` | -0.03581534999999997 `[-0.04033872918080527, -0.031291970819194666]` | -0.03019865000000005 `[-0.034683658560767605, -0.025713641439232493]` |
| **F−T** | **-0.21045229999999998** `[-0.21571542481540346, -0.2051891751845965]` | **-0.11617340000000004** `[-0.1243782653268875, -0.10796853467311257]` | **-0.0696908** `[-0.07825325989583513, -0.06112834010416486]` | **-0.05570055000000004** `[-0.06366932410860458, -0.0477317758913955]` |
| F matched-total−T | -0.20952065000000003 `[-0.21777857546968926, -0.2012627245303108]` | -0.11383704999999994 `[-0.11701016989872379, -0.11066393010127609]` | -0.06891185 `[-0.07328780651184202, -0.06453589348815797]` | -0.05605455000000001 `[-0.06096281154839772, -0.051146288451602295]` |

For context only, arm-level **MEASURED** mean last-5 losses at the four
budgets are:

| arm | 3000 | 6000 | 12000 | 18000 |
|---|---:|---:|---:|---:|
| T | 1.0140855 | 0.88825555 | 0.80542505 | 0.76997855 |
| H | 1.1850194 | 0.96221745 | 0.8393005 | 0.79548045 |
| P | 1.22405495 | 1.0230755 | 0.8947509 | 0.8432535 |
| F | 1.2245378 | 1.00442895 | 0.87511585 | 0.8256791 |
| F matched-total | 1.22360615 | 1.0020926 | 0.8743369 | 0.8260331 |

## 4. Engineering co-primary — steps and bytes to target

The **QUOTED** crossing is the first eval whose last-five mean is at or below
`L* = 0.80`. All values below are **MEASURED**. Bytes equal the measured step
times the **QUOTED** 16,384 bytes/step.

| arm | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|
| T | 13000 / 212.992M (0.7962222) | 13000 / 212.992M (0.7984608) | 12500 / 204.800M (0.7997213999999999) | 13000 / 212.992M (0.7965635999999999) |
| H | 17500 / 286.720M (0.7991195999999999) | 17500 / 286.720M (0.7977568) | 16000 / 262.144M (0.7998852) | 17500 / 286.720M (0.7998308) |
| P | censored | censored | censored | censored |
| F | censored | censored | censored | censored |
| F matched-total | censored | censored | censored | censored |

`H−T` has four uncensored pairs with **MEASURED**
`S = [-0.34615384615384615, -0.34615384615384615, -0.28,
-0.34615384615384615]`, mean **MEASURED** `-0.32961538461538464`, sample sd
**MEASURED** `0.03307692307692306`, and one-sided 95% LCB **MEASURED**
`-0.3685363952678763` (computed critical value **MEASURED**
`2.3533634348018233`). It does not meet the registered phrases
`convergence improvement`, `practically relevant`, or `strong support`.

`P−H`, direct `F−H`, `F−T`, and F matched-total−T have zero uncensored
pairs because each candidate fails to cross. `F−P` also has zero uncensored
pairs because both arms fail to cross. Under the registered minimum of three
uncensored pairs, every one of those co-primary readouts is exactly
**`INCONCLUSIVE`**. A censored pair is not assigned an estimated crossing.

## 5. Terminal noninferiority and matched-total readout

All statistics are **MEASURED** last-five values at step **QUOTED** 20000.
The arithmetic here is `L_candidate - L_base`; negative favours the
candidate. CIs use the **QUOTED** `t = 3.182`.

| contrast | terminal point gap | sample sd | 95% CI | point gap `< delta = 0.005` |
|---|---:|---:|---|---|
| H−T | 0.024638049999999995 | 0.004090399668736528 | [0.018130224127040177, 0.031145875872959812] | no |
| P−H | 0.045220200000000016 | 0.004355473647415744 | [0.03829064142696157, 0.05214975857303846] | no |
| F−P | -0.017891950000000018 | 0.006277481466055217 | [-0.027879423012493867, -0.007904476987506168] | yes |
| F−H | 0.02732825 | 0.002902110723249602 | [0.022710991839309882, 0.03194550816069011] | no |
| **F−T** | **0.05196629999999999** | **0.006012007750050452** | **[0.042401195669669725, 0.06153140433033026]** | **no — headline guardrail fails** |
| F matched-total−T | 0.052195500000000034 | 0.003712235389447499 | [0.046289333495389065, 0.058101666504611] | no — matched-total practical condition fails |

Arm-level terminal mean losses are **MEASURED** `0.7674019` (T),
`0.7920399499999999` (H), `0.83726015` (P), `0.8193682` (F), and
`0.8195974` (F matched-total).

Amendment 1 makes the matched-total result approximate, not exact: the arm
actually uses **MEASURED** `d_model = 744` and **MEASURED** 80,344,872 total
parameters. Its unfavourable result is therefore partly attributable to the
registered and explicitly conservative parameter shortfall; it still does
not satisfy the registered fixed-budget condition. H/P matched-total twins
remain declared follow-ups and no admissible captures for them were supplied
or analyzed.

## 6. Calibration gate and diagnostics

Every frozen field arm reports the complete five-member calibration group in
all eval records. Because the same checkpoint is frozen, the last-five values
are identical across H, P, F, F matched-total, and all four seeds. These
values are **MEASURED**:

| diagnostic | terminal last-five value | decision role |
|---|---:|---|
| field nats/byte | 1.256355 | frozen clause-(3) gate |
| field Brier | 0.464835 | report, never an added criterion |
| field ECE | 0.012982 | report, never an added criterion |
| field entropy | 1.287068 | report, never an added criterion |
| field top-1 accuracy | 0.631744 | report, never an added criterion |

The field-CE difference from the **QUOTED** standalone reference `1.258881`
is **MEASURED** `-0.002525999999999806` for 16/16 field-arm seeds. Its
absolute magnitude is within the **QUOTED** `±0.02` tolerance, so the frozen
calibration gate passes. Brier, ECE, entropy, and top-1 remain context only;
they are not promoted into unregistered criteria.

## 7. Hierarchical decision rule — clauses kept separate

The binding rule says **`Thesis supported`** only if all three clauses hold:

1. **QUOTED clause (1):** learning-curve CI entirely above 0.
   **MEASURED application:** false. The headline `F−T` CI is
   `[-0.11800596590322686, -0.10800255909677316]`, entirely below 0.
2. **QUOTED clause (2):** terminal noninferiority within `delta = 0.005`.
   **MEASURED application:** false. `L_F - L_T = 0.05196629999999999`.
3. **QUOTED clause (3):** the field stays calibrated.
   **MEASURED application:** true. Frozen field CE differs from the
   standalone reference by `-0.002525999999999806`, inside `±0.02`.

Because only one of the three clauses passes, **`Thesis supported`: false**.
This boolean applies the registered vocabulary without inventing a negative
verdict word that the registration did not define.

The **QUOTED** `Practically valuable` additions are also applied separately:

- `>= 10%` fewer bytes/steps to `L*`: false; F never crosses;
- `>= 0.01` lower nats/byte at any registered F−T budget: false; every
  **MEASURED** point gap is negative;
- matched-total F terminal point gap `< delta`: false; the **MEASURED** gap
  is `0.052195500000000034`.

Therefore **`Practically valuable`: false**. `Compute-efficient` is not
established because the to-target denominator is censored and the required
FLOP telemetry is absent.

## 8. Compute and systems readouts — separate from primaries

All numeric entries are **MEASURED** from `run_start`, `run_end`, and eval
`wall_s` telemetry. Processed bytes are the **QUOTED** 327.68M terminal
budget. Since the tokenizer reports **MEASURED** one byte/token, the computed
byte/s values are also token/s for these runs.

| arm | total params | field params | field parameter share | mean full-run GPU-s | mean bytes/s (= tokens/s) |
|---|---:|---:|---:|---:|---:|
| T | 85150464 | 0 | 0.0 | 2308.875 | 141948.11875735782 |
| H | 85675392 | 524928 | 0.006126940160367168 | 2317.08375 | 141430.12189108934 |
| P | 85577088 | 426624 | 0.004985259605935645 | 2350.381 | 139416.0024332637 |
| F | 85577088 | 426624 | 0.004985259605935645 | 2322.84475 | 141072.76016528404 |
| F matched-total | 80344872 | 426624 | 0.005309909511088648 | 2551.551 | 128435.83246988268 |

Per-seed full-run GPU-seconds are **MEASURED**:

- T: `[2263.148, 2299.342, 2346.719, 2326.291]`;
- H: `[2308.06, 2334.8, 2337.526, 2287.949]`;
- P: `[2345.375, 2352.901, 2348.955, 2354.293]`;
- F: `[2315.598, 2339.788, 2306.164, 2329.829]`;
- F matched-total: `[2537.772, 2545.418, 2529.67, 2593.344]`.

T GPU-seconds-to-target are **MEASURED** `[1480.142, 1506.215, 1476.504,
1521.443]`; H values are **MEASURED** `[2025.517, 2045.288, 1890.195,
2009.892]`. P, F, and F matched-total are censored for all seeds.

Estimated training FLOPs, FLOPs-to-target, peak VRAM, and field FLOP share
are not present in the admissible telemetry. No effective-parameter formula
or hardware estimate is substituted, so the registered compute claim remains
unanswered rather than inferred.

## 9. Predeclared probes — report, never decide

Nothing in this section changes §§3–7.

1. **LR-sensitivity diagnostic and rider.** The brief explicitly places every
   `*lrdiag*` capture out of scope, so none was read. The registered rider
   condition therefore cannot be evaluated in this artifact. The fixed-
   configuration verdict remains answerable; any quotation claiming that the
   rider did or did not fire would exceed this evidence.
2. **Context-sensitivity angle spread.** The admissible JSONLs contain no
   within-byte state-angle-spread field, and checkpoints are outside this
   JSONL-only evidence set. F/H/P spread and T's exact-zero control are
   therefore **not measured here**. The required measurement is the registered
   confirmatory checkpoint probe; it is a declared gap, not inferred from
   `state_source`.
3. **WSD branch-decay schedule diagnostic.** No admissible WSD capture is in
   the scoped input set. The registration makes this machinery-gated and
   explicitly non-gating for the frozen wave, so it is reported unavailable
   and does not alter the filed clauses.
4. **Calibration diagnostics.** Reported in §6 and used only as registered:
   field CE decides clause (3); Brier, ECE, entropy, and top-1 never decide.
5. **Schedule interpretation.** Budget-indexed values are the registered
   mid-schedule shared-trajectory estimand under the common cosine schedule;
   they are not described as budget-tuned frontier losses.

## 10. Limitations and disclosed deviations

1. **Narrow estimand.** The result is scoped exactly as registered: shared
   learning rate 6e-4, approximately 85M parameters, enwik8, byte level, and
   cosine-to-20k. It is not a tuned-frontier or larger-scale claim.
2. **Matched-total approximation.** Amendment 1's 744-wide arm is 5.6% under
   the static baseline. The fixed-budget result is valid under the amendment,
   but it is not an exact-total comparison; an unfavourable result is partly
   attributable to the shortfall.
3. **Integrity-chain gaps.** Pairing, configurations, materialized hashes,
   checkpoint/pretrain hashes, record completeness, and finiteness are
   verified. Per-run `config_hash`, per-run `git_sha`, and wave image digest
   are unavailable from the admissible captures and are not inferred.
4. **Compute gaps.** FLOPs, FLOPs-to-target, peak VRAM, and field FLOP share
   are absent. `Compute-efficient` cannot be established.
5. **Probe gaps.** LR rider status, checkpoint angle spread, and WSD
   schedule-sensitivity are unavailable under the supplied input boundary.
6. **Review-file access deviation.** During an early repository-wide filename
   search, command output surfaced two isolated path-and-line snippets from
   `docs/reviews/`, contrary to the brief's no-review-read boundary. No review
   file was opened, the snippets contained no E1.5c frozen outcome, and they
   support no claim in this artifact. All quantitative results were computed
   exclusively from the twenty scoped captures. This deviation is disclosed
   rather than silently erased.
7. **No invented aggregate vocabulary.** The registration defines the
   positive phrase `Thesis supported` but no negative rung word. The machine
   artifact therefore records the three clause booleans and
   `thesis_supported: false`; it does not manufacture `FAILED`,
   `DISPROVED`, or another unregistered label.
8. **Review status.** No `docs/reviews/*` artifact was intentionally read or
   created. Independent review remains a later lifecycle phase.

## 11. Filed verdict block

> **`Thesis supported`: FALSE.** Clause (1) fails: the **MEASURED** headline
> `F−T` learning-curve CI is entirely below zero. Clause (2) fails: the
> **MEASURED** terminal `L_F - L_T` gap is `0.05196629999999999`, not below
> the **QUOTED** `delta = 0.005`. Clause (3) passes: the **MEASURED** frozen
> field CE `1.256355` is within the **QUOTED** `±0.02` tolerance around the
> **QUOTED** standalone reference `1.258881`.

> **`Practically valuable`: FALSE.** F never reaches the **QUOTED** `L* =
> 0.80` target, is not lower than T at any registered budget, and the amended
> matched-total F terminal gap does not satisfy **QUOTED** `delta = 0.005`.

> **`Compute-efficient`: NOT ESTABLISHED.** F is censored at `L*` for all four
> seeds, and the admissible captures contain no FLOP measurement.

> **Causal ladder, reported without replacing the headline:** H is worse than
> T; P is worse than H; F is better than P; F remains worse than H and T.
