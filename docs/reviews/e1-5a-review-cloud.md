I'll start by extracting the packet and reading the preregistration.
# INDEPENDENT STATISTICAL REVIEW — E1.5a paired sufficiency packet (A, B vs C)

Everything below was recomputed from the extracted tree. No number in the packet's prose was taken on trust. All metric parsing went through `reader.py` (`read_eval_series`, `read_train_series`, `read_seed`, `final_eval_metric`) only.

---

## 1. INTEGRITY — PASS

All 15 md5s in `manifest.txt` verified against the extracted tree; **15/15 match, 0 mismatches**, no unlisted files, no listed-but-missing files.

```
OK abstract.md            OK data/B-seed1..4.jsonl     OK preregistration.md
OK data/A-seed1..4.jsonl  OK data/C-seed1..4.jsonl     OK reader.py
```

Proceeding.

---

## 2. ADHERENCE — no deviation found

| Registered item | Registration says | Observed in packet | Verdict |
|---|---|---|---|
| Arms | A, B (variants), C (control) | exactly A, B, C | ✔ |
| Seeds | {1,2,3,4} each arm | {1,2,3,4} each arm; `read_seed()` on each file returns the seed in its filename | ✔ |
| Run count | 12 | 12, no duplicates, no extras | ✔ |
| Primary metric | mean `nats_per_byte` over FINAL FIVE eval records, steps 18000–20000 @ interval 500 | every file has 40 eval records, steps 500…20000, cadence exactly 500; last five = **18000, 18500, 19000, 19500, 20000** | ✔ |
| Estimator | `g_s = y_variant,s − y_C,s`, n=4, df=3, per variant vs C at same seed | recomputed as specified; A vs B never formed | ✔ |
| Alpha split | 5% Bonferroni over **2** registered variants → per-variant two-sided 97.5% CI | applied as written | ✔ |
| Critical value | `t_{0.9875,3} = 4.177` | exact = **4.176534846**; rounds to 4.177 | ✔ (see below) |
| Looks | ONE (no stage 2) | one dataset, no stage-2 artifact, no second analysis file | ✔ |
| Exclusion rule | non-finite final loss only; drops whole seed | all 12 final eval losses finite (0.7658–0.7830); **no exclusion triggered, none claimed** | ✔ (vacuously) |
| Resume | fresh runs only | `train.resume = null` in all 12; exactly one `run_start` and one `run_end` (step 20000) per file | ✔ |

**Critical value ruling (asked explicitly):** the registered 4.177 **is the correct value** for df = 3 at a two-sided 97.5% CI (per-variant α = 0.025 → upper tail 0.0125 → quantile 0.9875). Exact `t_{0.9875,3} = 4.1765348`. The registered figure is the correct 3-dp rounding, rounded *up*, so it is very slightly conservative (widens each half-width by ~1.1×10⁻⁶ nats/byte). I carried the **registered 4.177** through the primary recomputation; the exact-value sensitivity is reported in §3 and changes nothing.

**No MAJOR adherence finding.** The registered analysis was executed as written.

---

## 3. RECOMPUTATION (primary, last-k=5)

Per-run primary metric (mean nats/byte over steps 18000–20000):

| seed | A | B | C |
|---|---|---|---|
| 1 | 0.7669370 | 0.7750320 | 0.7702034 |
| 2 | 0.7686792 | 0.7827998 | 0.7672032 |
| 3 | 0.7719242 | 0.7786740 | 0.7674010 |
| 4 | 0.7727646 | 0.7755178 | 0.7648000 |

Paired gaps `g_s = y_variant,s − y_C,s` (positive = freezing costs nats/byte):

| seed | A − C | B − C |
|---|---|---|
| 1 | **−0.0032664** | +0.0048286 |
| 2 | +0.0014760 | +0.0155966 |
| 3 | +0.0045232 | +0.0112730 |
| 4 | +0.0079646 | +0.0107178 |

| | mean | sd (n−1) | se | registered 97.5% CI (t=4.177) |
|---|---|---|---|---|
| **A** | 0.00267435 | 0.00476562 | 0.00238281 | **[−0.00727866, +0.01262736]** |
| **B** | 0.01060400 | 0.00442500 | 0.00221250 | **[+0.00136239, +0.01984561]** |

Sensitivity to the exact critical value 4.1765348: A → [−0.00727755, +0.01262625]; B → [+0.00136342, +0.01984458]. **No label or rider changes.**

### Registered verdict assignment (rule quoted verbatim)

> **GEOMETRY-SUFFICIENT** — CI upper bound < epsilon … **GEOMETRY-INSUFFICIENT** — CI entirely above epsilon … **INCONCLUSIVE** — otherwise.
> Rider (direction): the CI's position versus 0, reported verbatim (entirely below 0 = frozen strictly better; contains 0 = no detectable cost; entirely within (0, epsilon) = detectable cost below practical relevance).

ε = 0.01 nats/byte.

**VARIANT A — primary label: `INCONCLUSIVE`.** Upper bound +0.012627 is not < 0.01, and the interval is not entirely above 0.01. *(Mandatory co-equal direction rider: the CI **contains 0** — "contains 0 = no detectable cost".)*

**VARIANT B — primary label: `INCONCLUSIVE`.** Upper bound +0.019846 is not < 0.01; lower bound +0.001362 is not > 0.01, so the CI is not entirely above ε. *(Mandatory co-equal direction rider: the CI lies **entirely above 0**, [+0.00136, +0.01985] — a detectable cost. Note the registration's rider vocabulary enumerates only "entirely within (0, epsilon)" for a positive interval; B's interval is entirely above 0 but **crosses ε**, a case the registered rider vocabulary does not name. I report the position verbatim rather than force it into an unfitting bin — see MINOR-1.)*

Neither variant is GEOMETRY-SUFFICIENT; neither is GEOMETRY-INSUFFICIENT.

---

## 4. PAIRING AUDIT — PASS

`data_order_digest` + `data_order_batches` present on exactly **one** train record per file (step 40, i.e. the first logged train record at/after the window), in **all 12 files**. No arm is unattested; no resumed run.

| seed | digest (first 16 hex) | A | B | C | within-seed match |
|---|---|---|---|---|---|
| 1 | `3f05412324b70543…` | ✔ | ✔ | ✔ | **identical** |
| 2 | `8a39105f5aa9f1ed…` | ✔ | ✔ | ✔ | **identical** |
| 3 | `4f26fe9084deda26…` | ✔ | ✔ | ✔ | **identical** |
| 4 | `6e4e11046c3f6522…` | ✔ | ✔ | ✔ | **identical** |

`data_order_batches = 32` in all 12, matching "first 32 training batches".

**Seeds genuinely varied:** the four digests are **pairwise distinct**. Had they been equal across seeds, the "seeds" would have been the same data order relabelled; they are not. Pairing is attested, not assumed. No seed is invalidated under the pairing rule.

---

## 5. OPTIONAL-STOPPING AUDIT — PASS

Preregistration commit asserted at `be85f0c3f5e7 2026-08-20T22:07:44Z`.

- **All 8 registered-variant runs (A, B) start after the commit** — earliest is A-seed3 at 2026-08-21T01:48:20Z, i.e. **+3h41m**. No confirmatory variant run predates the registration.
- **All 4 control runs (C) start 2026-08-18T16:26–16:27Z, ~2.2 days before the commit.** The registration discloses this explicitly ("C … **PREVIOUSLY CAPTURED, no new runs**"), so this is expected, not a finding. The disclosure **does cover** what I observe: no C run is dated after the commit (which would have contradicted "no new runs"), and all four C timestamps sit in one tight 19-second launch window consistent with a single prior capture batch.
- **No unregistered second look.** One file per arm×seed, one `run_start`/`run_end` per file, `resume=null` throughout, no stage-2 or interim-analysis artifact in the manifest or tree.
- **Manifest timestamps are internally consistent with the data:** manifest (end−start) exceeds the in-file `run_end.wall_s` by 3.8–9.1 s in every run — a plausible uniform process-startup offset, with no run where the asserted wall-clock is shorter than the logged wall time.

One unexplained timing pattern is recorded in MINOR-3 below.

---

## 6. CONTINGENCY-TRIGGER EVALUATION — **the trigger does NOT fire**

Registered ordering: `GEOMETRY-SUFFICIENT > INCONCLUSIVE > GEOMETRY-INSUFFICIENT`. Trigger condition: `label(A) < label(B)`.

Computed: `label(A) = INCONCLUSIVE`, `label(B) = INCONCLUSIVE` ⇒ `label(A) = label(B)` ⇒ `label(A) < label(B)` is **FALSE**. The deferred third variant **does not run under this registration**. The expected pattern `label(A) ≥ label(B)` holds (with equality), and the point estimates are ordered as the registration expected (mean gap A = 0.00267 < B = 0.01060, i.e. A ranks better).

**Is the packet's content consistent with the trigger not having fired?** Yes. The packet contains exactly three arms and no third-variant data, matching "It is not part of this packet" and "Under any other outcome it does not run."

**Robustness of the trigger determination to the sealed A/B role assignment:** the packet's scrub makes it impossible for me to verify that the file labelled A is in fact the subset leaving strictly more capacity free (see MAJOR-1). Here that is **moot**: because both labels are identical, the trigger is FALSE under *either* assignment of the role label. Had the labels differed, the trigger's evaluation would have been unauditable from this packet, and I would have raised it as MAJOR.

---

## 7. ROBUSTNESS PROBES (reported, not decisive)

**Sign test on paired gaps (exact, two-sided).** A: 3 positive / 1 negative, **p = 0.625**. B: 4 positive / 0 negative, **p = 0.125** (the minimum attainable at n = 4). Neither reaches conventional significance; B's is the strongest possible sign-test evidence at this n, which is itself a statement about the design's resolution, not about B.

**Per-seed spread.** A: sd 0.0047656, range 0.011231 (min −0.0032664, max +0.0079646). B: sd 0.0044250, range 0.010768 (min +0.0048286, max +0.0155966). Both ranges exceed ε.

**Leave-one-out (df = 2, t₀.₉₈₇₅,₂ = 6.2053).**

| dropped | A: mean / CI / label | B: mean / CI / label |
|---|---|---|
| seed 1 | 0.004655, [−0.00698, +0.01628] — INCONCLUSIVE | 0.012529, [+0.00296, +0.02210] — INCONCLUSIVE |
| seed 2 | 0.003074, [−0.01754, +0.02369] — INCONCLUSIVE | 0.008940, [−0.00385, +0.02173] — INCONCLUSIVE |
| seed 3 | 0.002058, [−0.01814, +0.02226] — INCONCLUSIVE | 0.010381, [−0.00894, +0.02970] — INCONCLUSIVE |
| seed 4 | 0.000911, [−0.01315, +0.01497] — INCONCLUSIVE | 0.010566, [−0.00885, +0.02998] — INCONCLUSIVE |

**No seed's removal changes either primary LABEL** — both stay INCONCLUSIVE in all eight leave-one-out fits. The labels are LOO-stable. The *rider* is not: dropping seed 2, 3, or 4 moves B's interval from "entirely above 0" to "contains 0", so B's direction rider rests on all four seeds jointly (and, mechanically, on the df=2 penalty). A's mean is strongly leveraged by seed 4 (drop it and the mean falls to 0.00091, a 66% reduction) and by seed 1 (drop it and the mean rises 74%).

**Smoothing-window (k) sensitivity, including the registered k=1 endpoint reported alongside.**

| k | A mean | A CI | A label | B mean | B CI | B label |
|---|---|---|---|---|---|---|
| **1** (endpoint) | 0.003334 | [−0.00687, +0.01354] | INCONCLUSIVE | 0.010299 | [+0.00032, +0.02028] | INCONCLUSIVE |
| 2 | 0.003426 | [−0.00746, +0.01431] | INCONCLUSIVE | 0.010967 | [+0.00187, +0.02006] | INCONCLUSIVE |
| 3 | 0.003314 | [−0.00761, +0.01423] | INCONCLUSIVE | 0.011203 | [+0.00188, +0.02053] | INCONCLUSIVE |
| 4 | 0.002779 | [−0.00747, +0.01302] | INCONCLUSIVE | 0.010797 | [+0.00152, +0.02008] | INCONCLUSIVE |
| **5 (registered)** | **0.002674** | **[−0.00728, +0.01263]** | **INCONCLUSIVE** | **0.010604** | **[+0.00136, +0.01985]** | **INCONCLUSIVE** |
| 6 | 0.002533 | [−0.00750, +0.01257] | INCONCLUSIVE | 0.010674 | [+0.00093, +0.02042] | INCONCLUSIVE |
| 8 | 0.002769 | [−0.00633, +0.01186] | INCONCLUSIVE | 0.010835 | [+0.00066, +0.02101] | INCONCLUSIVE |
| 10 | 0.002727 | [−0.00577, +0.01123] | INCONCLUSIVE | 0.011196 | [+0.00151, +0.02088] | INCONCLUSIVE |

**Both labels and both riders are invariant across k ∈ {1,…,10}.** The registered choice of k=5 is not load-bearing for any conclusion, and the k=1 endpoint the registration promises alongside agrees.

*But the stated rationale for k=5 does not hold in this data.* The registration adopted last-k=5 on the prior rung's finding that "single-eval jitter [is] roughly 40% of pair variance" and that "the margin WIDENED under smoothing." Recomputed here: gap variance falls only **4.9% (A)** and **14.3% (B)** going from k=1 to k=5 — an order of magnitude less variance reduction than the 40% premise implies. Within-window per-run sd is 0.0002–0.0024 nats/byte, far below the between-seed gap sd of ~0.0044–0.0048. **At this rung the paired-gap variance is dominated by run/seed-level variation, not eval jitter**, so smoothing buys almost nothing. This does not violate the registration (k=5 was registered and was used), but it invalidates the *reason* given for it and matters for the power discussion in §8.

**Anomaly scan — clean.** No non-finite values in any eval or train record across all 12 files. No truncated series: all arms have exactly 40 eval records (500→20000, uniform cadence 500) and exactly 1000 train records (20→20000, uniform cadence 20); **no arm's series differs in length from any other**. Every file terminates with `run_end` at step 20000. No `kappa` and no field-calibration group is present in any eval record — uniformly absent across all arms, so no partial-group corruption and no cross-arm asymmetry; the registered metric does not use them.

**Two exploratory patterns worth naming (n = 4, so weak evidence).** (i) A's gaps increase monotonically with seed index (−0.0033, +0.0015, +0.0045, +0.0080); under exchangeability, one specific monotone ordering has probability 1/24. (ii) A's gap is near-perfectly negatively correlated with the control's own level (r = −0.955, p = 0.045), i.e. A looks best on the seed where C did worst. Under the registered paired estimand this correlation is not required to vanish, and at n = 4 it is fragile, but it is the kind of structure that would make the paired gap non-exchangeable across seeds and inflate the effective uncertainty beyond the t-interval. Reported, not decided.

---

## 8. POWER AND ESTIMAND

**The named assumption.** The registration projects power "[DECLARED ASSUMPTION: the frozen-vs-trainable within-pair sd is of the same order as that measured between-arm sd — no direct measurement of it existed at registration]", using sd(d) = 0.00300307.

**Observed:** sd = 0.0047656 (A) and 0.0044250 (B) — **1.59× and 1.47× the assumed value.** "Same order" survives in the loose sense; the *projection built on it does not*. Projected half-width was 0.0063; observed half-widths are **0.00995 (A)** and **0.00924 (B)**. The registration's own contingency language covers this outcome almost exactly: "If the sd doubles, half-width ≈ 0.0125, giving INCONCLUSIVE — an informative heterogeneity result, not a nullity." We landed between the two branches, and the INCONCLUSIVE branch is what materialised. This is a *predicted* failure mode of the design, honestly pre-declared — which is to the registration's credit — but it means the rung's central question was not answerable at the n it committed to. Note also (§7) that the registered smoothing, adopted partly to control this variance, delivered ≤14% variance reduction.

**Not a multiplicity artifact.** Re-running both variants at an uncorrected 95% CI (t₀.₉₇₅,₃ = 3.1824) gives A [−0.00491, +0.01026] and B [+0.00356, +0.01764] — **both still INCONCLUSIVE**. The Bonferroni split is not what produced the indeterminacy; the observed spread is.

**What the interval can and cannot settle at n = 4.**
- *Can settle:* for **B**, that the paired cost of freezing is **detectably greater than zero** at the registered per-variant level (CI excludes 0). For **A**, that any cost is **not detectably different from zero**. Also that neither variant's cost is *demonstrably* ≥ ε.
- *Cannot settle:* **the registered question itself — sufficiency — for either variant.** A's interval [−0.0073, +0.0126] is compatible with freezing being a modest *improvement* and with a cost 26% above the practical threshold. B's [+0.0014, +0.0198] is compatible with a cost one seventh of ε and with a cost twice ε. Neither can distinguish "the frozen geometry carries the structure" from "freezing practically hurts."
- The estimand is further bounded by the registration's own **declared limitation**: the contrast is the *arm effect at one shared configuration* (lr 6e-4, batch 16×1024, cosine to 20k, ~85M/enwik8), not at each arm's tuned frontier. Since freezing changes effective trainable capacity, part of any measured B cost could be tuning-response rather than a substrate claim. The registration is right to flag this and right not to assume equivalence; nothing in the packet resolves it, and a reader must not generalise these labels past this configuration and regime.

**What would resolve each INCONCLUSIVE, and at what n (holding the observed mean and sd).**
- **A** → `GEOMETRY-SUFFICIENT` requires upper bound < 0.01, i.e. mean + t₀.₉₈₇₅,ₙ₋₁·sd/√n < 0.01. At mean 0.002674, sd 0.004766: **n = 6 seeds** (half-width 0.00616). Only two more seeds — this is the cheap, decisive extension, though it is fragile: two adverse seeds would push the mean up and undo it. Additionally pinning A's *direction* (excluding 0) would take **n ≈ 19**.
- **B** → `GEOMETRY-SUFFICIENT` is **unreachable at any n**, because the observed point estimate 0.01060 already exceeds ε = 0.01. Driving B to `GEOMETRY-INSUFFICIENT` (lower bound > ε) at the observed mean/sd needs **n ≈ 273 seeds** — not practical. B is therefore parked almost exactly on the practical threshold: the honest reading is "a real but threshold-straddling cost", and no feasible seed count under this design will label it. Resolving B needs a different lever than n: a larger ε (which would require its own pre-registration and cannot be chosen now), a lower-variance endpoint, or a design that reduces the between-seed component that §7 shows dominates.

---

## 9. VERDICT

### Registered primary labels, each with its mandatory co-equal direction rider

Registered criterion, quoted: *"**GEOMETRY-SUFFICIENT** — CI upper bound < epsilon … **GEOMETRY-INSUFFICIENT** — CI entirely above epsilon … **INCONCLUSIVE** — otherwise. Rider (direction): the CI's position versus 0, reported verbatim (entirely below 0 = frozen strictly better; contains 0 = no detectable cost; entirely within (0, epsilon) = detectable cost below practical relevance)."*

> **VARIANT A — `INCONCLUSIVE`** (mean gap +0.002674, 97.5% CI **[−0.007279, +0.012627]**; upper bound 0.012627 ≥ ε = 0.01 and interval not entirely above ε).
> **Mandatory direction rider — "contains 0 = no detectable cost."**

> **VARIANT B — `INCONCLUSIVE`** (mean gap +0.010604, 97.5% CI **[+0.001362, +0.019846]**; upper bound 0.019846 ≥ ε = 0.01 and lower bound 0.001362 ≤ ε).
> **Mandatory direction rider — the CI lies entirely above 0** (detectable cost), **but it crosses ε**, so it does not satisfy the registered phrase "entirely within (0, epsilon)"; the registered rider vocabulary has no bin for this position and I decline to force one. Position reported verbatim.

Both labels must be read with their riders; reporting either clause alone misquotes the registration.

**Contingency trigger: DOES NOT FIRE** (label(A) = label(B) = INCONCLUSIVE, so label(A) < label(B) is false). No deferred third variant runs. Packet contents are consistent with this.

**Bottom line:** the analysis was executed faithfully — hashes, arms, seeds, metric, estimator, critical value, alpha split, look count, exclusion rule, pairing attestation, and run chronology all check out, and I found no arithmetic or procedural deviation. The rung nonetheless does **not** answer its own question: at n = 4 with the observed spread, neither variant's interval separates sufficiency from practical harm.

---

### MAJOR concerns

**MAJOR-1 — The treatment contrast is unverifiable from the packet.** After the scrub, all twelve `run_start` configurations are **byte-identical except `config.name`/`run` (`arm-A`/`arm-B`/`arm-C`) and `train.seed`**. There is no freeze-axis field anywhere, in any arm. I can therefore confirm the registration's build-time gate ("A's and B's `run_start` configurations are IDENTICAL after scrubbing") — but that gate is precisely the property that makes the intervention invisible. **Nothing in the packet provides positive evidence that A froze any parameters, that B froze a *different* subset, or that C trained all parameters.** Everything I have recomputed is conditional on the arm labels being truthful. The registration discloses the sealing, so this is a disclosed design consequence rather than concealment — but it is a hard ceiling on what an independent reviewer can certify, and it should be stated whenever these labels are quoted. *Resolution:* an unblinded attestation (e.g. a hash over the frozen-parameter mask per arm, or trainable-parameter counts per arm) that distinguishes the arms without naming the project.

**MAJOR-2 — The registered question is not answered, and the shortfall traces to a named assumption that failed quantitatively.** Observed within-pair sd is 1.47–1.59× the assumed 0.00300307, inflating the half-width from a projected 0.0063 to 0.0092–0.0100 and producing INCONCLUSIVE for both variants. This is not a multiplicity artifact (uncorrected 95% intervals are also INCONCLUSIVE) and not a smoothing artifact (labels invariant for k = 1…10). Compounding it, the empirical justification the registration gave for its k=5 endpoint — 40% of pair variance being single-eval jitter — **does not reproduce here** (4.9%/14.3% variance reduction; between-seed variation dominates). Any downstream write-up that presents this rung as evidence *for* geometric sufficiency, rather than as an underpowered non-result with a pre-declared heterogeneity reading, would overstate the packet. The registration deserves credit for anticipating this branch in advance; the finding is that the branch fired.

### MINOR concerns

**MINOR-1 — The registered rider vocabulary is incomplete.** It enumerates "entirely below 0", "contains 0", and "entirely within (0, ε)", but not "entirely above 0 and crossing ε" — which is exactly where B landed, and which is a perfectly reachable region whenever the label is INCONCLUSIVE with a positive mean. This is a registration-design weakness, not an execution error; I have kept the label as the registration defines it and reported B's interval position literally rather than reclassifying it. A future revision should close the vocabulary over all interval positions.

**MINOR-2 — The A/B role definition is partly circular and unauditable.** "Arm A is defined by ROLE: it is the variant the registration expects to rank at least as high as B", with the physical subset sealed. Combined with MAJOR-1, a reviewer cannot check that the file labelled A is the one leaving strictly more capacity free, which is the substantive precondition for the "strange divergence" trigger to mean anything. Here it is moot — equal labels make the trigger false under either assignment — but the trigger's auditability depends on an unblinding this packet does not supply.

**MINOR-3 — Unexplained launch-time stagger on seed 1, coinciding with the most variant-favourable seed.** Six of the eight variant runs launched in a 13-second window (01:48:20–01:48:33Z). The two exceptions are **B-seed1 (02:10:02Z, +22 min)** and **A-seed1 (02:21:58Z, +34 min)**. Seed 1 is simultaneously A's only negative gap (−0.0033, its most favourable seed) and B's smallest gap (+0.0048, its most favourable seed). I found **no evidence of a discarded or repeated run**: every file has `resume = null`, one `run_start`, one `run_end` at step 20000, a complete 40-point eval series, and a wall-clock consistent with the manifest. Scheduler queueing is a fully adequate innocent explanation. But the coincidence of "the two off-schedule runs" with "the two most variant-favourable observations" is the exact signature a selective re-run would leave, and the manifest's timestamps are *asserted operator metadata*, not raw record content. *Resolution:* the launcher/scheduler log for 2026-08-21T01:40–02:30Z, plus confirmation that no seed-1 run was started and discarded.

**MINOR-4 — Small internal inconsistency in the registration's schema note.** It states arm C "was captured under an earlier revision of the training code, before the freeze axis existed. Its records may therefore lack fields that A and B carry." In fact C's schema is **identical** to A's and B's on every key (only `name`/`run`/`seed` differ), and C carries the `data_order_digest` pairing attestation exactly as the variants do. Benign — the scrub evidently equalised the schemas — but the note as written predicts an asymmetry that does not exist, and a reviewer checking for it could be misled about which absences are expected.

**MINOR-5 — Exploratory structure in the gaps at n = 4.** A's gaps are monotone increasing in seed index (p ≈ 1/24 for that specific ordering under exchangeability), and A's gap correlates −0.955 (p = 0.045) with the control's own level. Not registered, not decisive at n = 4, and not a criticism of the analysis — but if the paired gap is genuinely a function of the control's level, the seed-exchangeability underlying the t-interval is strained and the true uncertainty is wider than reported.

**MINOR-6 — Rider fragility for B.** B's "entirely above 0" rider is not leave-one-out stable: removing seed 2, 3, or 4 returns an interval containing 0. The primary labels are LOO-stable; the rider is not, and since the registration makes the rider co-equal, that fragility should travel with any quotation of B's result.

### What would change my verdict

1. **Two to three additional seeds on A and C (n = 6–7).** At the observed mean and sd this flips A to **GEOMETRY-SUFFICIENT** at n = 6 — the single highest-value, lowest-cost extension in the packet. It would also test MINOR-5's monotone-in-seed pattern, which n = 4 cannot.
2. **An arm-distinguishing attestation** (per-arm frozen-mask hash or trainable-parameter count) that survives blinding. Absent it, MAJOR-1 stands and all labels remain conditional on unverified arm identity; with it, MAJOR-1 is discharged outright.
3. **The scheduler/launcher log for the 2026-08-21 window**, confirming no seed-1 run was started and discarded. This would retire MINOR-3; a log showing a discarded seed-1 run would instead promote it to MAJOR and require the whole primary analysis to be re-read as outcome-contaminated.
4. **A direct measurement of the frozen-vs-trainable within-pair sd** (the quantity the power projection had to assume). If it stabilises near 0.0044–0.0048 rather than 0.0030, this design is structurally underpowered for ε = 0.01 and the rung should be re-registered at a larger n or a lower-variance endpoint before any sufficiency claim is filed.
5. **For B specifically:** since GEOMETRY-SUFFICIENT is arithmetically unreachable at its observed mean and GEOMETRY-INSUFFICIENT needs n ≈ 273, only a *pre-registered* change of lever — a different endpoint, a variance-reduction strategy, or a re-argued ε — could move B off INCONCLUSIVE. I would not accept a post-hoc ε revision on this data.
6. **Any evidence of a second look or a stage 2** (an interim analysis artifact, a second manifest, a re-registration) would override the §5 PASS and invalidate the single-look error control.
