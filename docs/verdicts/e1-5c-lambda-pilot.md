# E1.5c lambda-field pilot readout

## 0. Scope and selection

This is the exploratory E1.5c joint-regime lambda-field pilot readout, not a
confirmatory rung verdict. The binding rule is
`docs/e1-5c-preregistration.md` section **Pilots**, item 2:

> **Selection rule (mechanical):** lambda* = the lambda whose held-out field
> CE lands closest to the standalone reference, subject to backbone
> nats/byte at step 3000 within +0.005 of the best pilot arm; ties break
> toward the smaller lambda.

Applied as the registered constrained argmin, the selection is:

> **lambda* = 0.1**

Lambda 0.1 is the best pilot arm on backbone nats/byte and the only arm that
passes the backbone feasibility filter. The field-CE objective is therefore
evaluated over the singleton feasible set `{0.1}`. The tie-break toward the
smaller lambda was **not needed**.

All metrics below come only from the four named DVC-materialized pilot
captures in `results/captured/`, read through
`entropic_substrate.telemetry.read_eval_series`. No live tracking-server data
and no frozen-wave capture was read.

## 1. Endpoint and reference reading

The registration resolves the step-3000 endpoint as the mean of the five
consecutive eval records ending at step 3000. For every pilot arm, those eval
steps are `{1000, 1500, 2000, 2500, 3000}`. The field CE and backbone
nats/byte values below are those registered last-5 means.

The primary standalone reference is **1.258881 nats/byte**, the supplied
last-5 mean from the 20,000-step standalone field pretrain. This reading is
used because:

1. item 1 of the registration calls the lambda-selection anchor the
   standalone held-out **last-5 mean**; and
2. the registration's endpoint section says all loss readings are last-k
   (`k = 5`) eval means.

The supplied standalone final value, **1.256011 nats/byte**, is therefore not
the primary reference. Substituting it changes none of the feasibility
results or the selection: all pilot field CEs lie above both reference
figures, and lambda 0.1 remains the only feasible arm. The alternate absolute
distances, using the registered pilot last-5 endpoints against the standalone
final value, are `{0.1: 0.9576892, 0.3: 0.6216048, 0.5: 0.5295726, 1.0:
0.462246}` nats/byte. Thus item 2's phrase “the standalone reference” is
locally underspecified when read alone, but the surrounding registration
resolves the intended estimator and the supplied alternative does not change
lambda*.

## 2. Mechanical constrained argmin

Every numeric cell in this section is **DERIVED FROM MEASURED TELEMETRY**:
each mean is exact decimal arithmetic over the five serialized six-decimal
eval readings ending at step 3000, and each distance is computed from that
mean and the quoted-measured reference `1.258881`.

| lambda | evidence class | held-out field CE, last-5 mean | backbone nats/byte at step 3000, last-5 mean | absolute field-CE distance from 1.258881 | backbone feasibility |
|---:|---|---:|---:|---:|:---:|
| 0.1 | **DERIVED FROM MEASURED** | 2.2137002 | 0.9980398 | 0.9548192 | **PASS** |
| 0.3 | **DERIVED FROM MEASURED** | 1.8776158 | 1.004792 | 0.6187348 | **FAIL** |
| 0.5 | **DERIVED FROM MEASURED** | 1.7855836 | 1.0114252 | 0.5267026 | **FAIL** |
| 1.0 | **DERIVED FROM MEASURED** | 1.718257 | 1.013958 | 0.459376 | **FAIL** |

### 2.1 Feasibility filter

The best pilot arm on backbone nats/byte is lambda 0.1 at `0.9980398`.
Therefore the inclusive registered ceiling is:

```text
0.9980398 + 0.005 = 1.0030398 nats/byte
```

| lambda | evidence class | backbone gap from best | within +0.005 of best? |
|---:|---|---:|:---:|
| 0.1 | **DERIVED FROM MEASURED** | 0.0000000 | **PASS** |
| 0.3 | **DERIVED FROM MEASURED** | 0.0067522 | **FAIL** |
| 0.5 | **DERIVED FROM MEASURED** | 0.0133854 | **FAIL** |
| 1.0 | **DERIVED FROM MEASURED** | 0.0159182 | **FAIL** |

The closeness objective is evaluated only after this filter. Although lambda
1.0 is closest to the standalone field reference over the unconstrained
four-arm grid, it is inadmissible under the backbone clause. Lambda 0.1 is
selected by the exact clause “closest to the standalone reference, subject
to backbone nats/byte at step 3000 within +0.005 of the best pilot arm”
because it is the sole feasible arm. No objective tie exists, so the
smaller-lambda tie-break is unused.

## 3. Calibration diagnostics — context only, not selection criteria

These are last-5 means over the same eval steps as the selection endpoint.
They are reported as context only. The mechanical rule uses none of
`field_brier`, `field_ece`, `field_entropy`, or `field_top1`.

| lambda | evidence class | field_brier | field_ece | field_entropy | field_top1 |
|---:|---|---:|---:|---:|---:|
| 0.1 | **DERIVED FROM MEASURED** | 0.7427592 | 0.0335846 | 2.419373 | 0.3791382 |
| 0.3 | **DERIVED FROM MEASURED** | 0.6555074 | 0.0320748 | 2.021654 | 0.467896 |
| 0.5 | **DERIVED FROM MEASURED** | 0.6294074 | 0.0334878 | 1.9204596 | 0.4932616 |
| 1.0 | **DERIVED FROM MEASURED** | 0.6109364 | 0.0289082 | 1.8269102 | 0.5101518 |

> **CALIBRATION CAUTION:** selected lambda 0.1 has the numerically largest
> ECE in this pilot grid (`0.0335846`) as well as the largest field CE and
> Brier score. This is a measured relative warning: the selected arm looks
> least favorable on these pilot calibration readouts. The registration
> defines no absolute pilot-ECE threshold, so this readout does **not**
> manufacture an absolute “poor ECE” classification. The warning does not
> override the mechanical lambda rule. It is operator-facing context for the
> private-code failure mode the calibrated-field repair is intended to catch.

These are 3,000-step pilot readings, roughly one seventh of the full training
budget. No claim about lambda behavior at 20,000 steps, or about whether the
relative ECE ordering persists at full length, is made or inferred here.

## 4. Completeness and admissibility

`read_eval_series` accepted every record in every input, which validates
finiteness, documented metric domains, and completeness of the five-member
field-calibration group on every eval record. Each run contains six eval
records at steps `{500, 1000, 1500, 2000, 2500, 3000}` and terminates at the
registered pilot step.

| lambda | evidence class | eval records | terminal step | finite and telemetry-admissible | incomplete or missing? |
|---:|---|---:|---:|:---:|:---:|
| 0.1 | **MEASURED** | 6 | 3000 | yes | no |
| 0.3 | **MEASURED** | 6 | 3000 | yes | no |
| 0.5 | **MEASURED** | 6 | 3000 | yes | no |
| 1.0 | **MEASURED** | 6 | 3000 | yes | no |

No arm was dropped, repaired, imputed, or silently excluded.

## 5. Inputs and integrity

The capture hashes below were computed in this analysis session over the
materialized DVC JSONLs. The standalone artifact hashes and reference values
are **QUOTED MEASURED PROVENANCE** supplied with the fresh-session brief;
they were not rederived from a tracking server in this session.

| lambda / reference | evidence class | input | md5 |
|---:|---|---|---|
| 0.1 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-2d85b537-e1-5c-pilot-lam01.jsonl` | `87c9109130f6ac7dad6a5b1a523f8528` |
| 0.3 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-fd30c01c-e1-5c-pilot-lam03.jsonl` | `60f96e4174fc6e47b0fd9333780d0fe2` |
| 0.5 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-cccab401-e1-5c-pilot-lam05.jsonl` | `3aacf337605ee82aaf33467453d3cf05` |
| 1.0 | **MEASURED THIS SESSION** | `results/captured/e1-5c-field-c66e4903-e1-5c-pilot-lam10.jsonl` | `56a2b5f59da6040ce8d697cea91eff86` |
| standalone field checkpoint | **QUOTED MEASURED** | `models/field-pphi-v1.pt` | `edc3e3e55ccc340bf565e14ac547618c` |
| standalone pretrain results | **QUOTED MEASURED** | standalone pretrain JSONL | `893e37225011c97322f6cf5c0d68c4e0` |

Standalone reference provenance: 20,000 steps, seed 42, MPS; final held-out
field CE `1.256011` nats/byte and last-5 mean `1.258881` nats/byte. The
last-5 mean is the selected reference estimator for this readout.

## 6. Joint-lane parameter consumed by the registration

The E1.5c joint-regime lane consumes:

```text
field_loss_lambda = 0.1
```

This document is the pilot-selection input to that lane. It intentionally
does not create a `*-verdict.json` and makes no confirmatory rung claim.
