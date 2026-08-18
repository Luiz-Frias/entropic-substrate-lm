# E1 cumulative (n=9) — independent blind review (cloud sandbox lane)

Produced 2026-08-18 by `scripts/blind_review_cloud.py` (zip delivery): model
`claude-opus-5` in Anthropic's code-execution sandbox — server-enforced
no-network isolation, single-archive Files API delivery, zero repository or
operator-machine access. Packet: `docs/reviews/e1-n9-packet` (18 runs, both
registrations, clause precedence, complete run ledger incl. the declared
abandoned duplicate). Filed unedited below.

Filing notes: the reviewer independently recomputed the extension criterion —
t = 4.591 vs threshold 4.277, one-sided 99.865% lower bound +0.00031 > 0,
9/9 pairs positive — holding STRONGLY_SUPPORTED "arithmetically" while
correctly declining to treat the outcome-aware extension as part of the
original registration's alpha budget (the extension registration says the
same of itself). Its endpoint-noise finding (single-eval jitter ~40% of pair
variance; last-5-eval mean gives cumulative CI [+0.0027, +0.0085]) is
adopted as a design input for the E1.5 preregistrations (FOR-900/904/907).
The official extension verdict remains the fresh analysis session's to file.

---

I'll start by extracting the packet and surveying its contents.
Now step 1: integrity verification of every md5 in the manifest.
All 21 manifest hashes verify; no extra or missing files. Proceeding to adherence.
Structure is uniform. Now the full adherence + recomputation audit, parsing exclusively through `reader.py`.
Data structure is clean and fully paired. Now the statistical recomputation (stage 1, cumulative, and the extension threshold).
Critical: the stage-1 CI does **not** straddle 0. Now the optional-stopping / timestamp audit, plus a check of whether the data contain any independently verifiable clock.
All checks are complete. Below is my review.

---

# Independent Statistical Review — E1 review packet ("paired two-arm comparison, arms A/B")

**Reviewer posture:** anonymous, adversarial replication. Every number below was recomputed by me from the packet; nothing is taken from the abstract, the prose, or any asserted summary. All data parsing went through `reader.py` (`final_eval_metric`, `read_eval_series`, `read_train_series`, `read_seed`, `read_final_kappa`, and `reader._iter_records` for non-metric fields such as `config` and `data_order_digest`, so that even structural inspection used the packet's own validated JSON parser).

---

## 1. INTEGRITY — PASS

All 21 files listed in `manifest.txt` were hashed and compared:

* **21/21 md5 digests match exactly.** 0 mismatches.
* **No file listed in the manifest is missing**, and **no file in the extracted tree is unlisted** (i.e. no undeclared extra data files smuggled into `data/`).

Integrity is clean; I proceed.

Secondary integrity probes (not required, but run):

* Internal consistency of every eval record: `loss == nats_per_byte` (since `bytes_per_token = 1.0`), `bits_per_byte · ln2 == nats_per_byte` (max abs. dev. < 1e-6), `perplexity == exp(nats_per_byte)` (max rel. dev. 2.3e-5, i.e. pure 4-decimal rounding of a derived field). **No inconsistency of substance.**
* Declared run durations vs. in-file terminal `wall_s`: overhead 3.76–5.14 s for all 18 runs (ratio 1.00165–1.00228). Mutually consistent; note this only shows the two sources agree, not that either is independently true (see §4).

---

## 2. ADHERENCE — one MAJOR deviation

### Seeds present

Read via `reader.read_seed()` from each `run_start` record, and cross-checked against the filename label (**all 18 agree**):

| Arm | Seeds present |
|---|---|
| A | 1, 2, 3, 4, 5, 6, 7, 8, 9 |
| B | 1, 2, 3, 4, 5, 6, 7, 8, 9 |

* Registered stage-1 set {1,2,3,4,5}: **present, complete, both arms.**
* Registered stage-2 set {6,7,8,9}: **present, complete, both arms.**
* Unregistered seeds: **none**. No seed appears twice; no arm/seed cell is missing or duplicated.

### Exclusions

The registration permits exactly one exclusion ground: *"a run is excluded solely for a non-finite final loss; an exclusion always drops the whole PAIR and is reported."*

* Final-eval `loss` and `nats_per_byte` are **finite in all 18 runs**. Every eval and train record in all 18 files is finite (verified by `read_eval_series` / `read_train_series`, which raise on non-finite values — none raised).
* **Therefore no exclusion is triggered, and none has been applied.** No pair was dropped. Correct.

### Pairing (`data_order_digest`, Appendix A)

* All 18 runs attest exactly **one** digest, emitted on the train record at step 40 — the first logged train step ≥ 32 (log_interval = 20) — with `data_order_batches = 32`. **Appendix-A compliant in all 18 runs.**
* `resume = null` in all 18 runs, so all are fresh runs and all are required to attest. None is silently un-attested.
* **All 9 within-pair digests match exactly. 0 pairs invalidated.**
* Digests are **distinct across all 9 seeds** (9 distinct values, each appearing exactly twice = once per arm). No duplicate-digest-across-seeds pathology; the seeds were varied as far as the digest can evidence it.
* Within-pair config diff: for **every** pair the only difference across `config.model`, `config.data`, `config.train` is `model.embedding_init: 'init-A'` vs `'init-B'`. Across seeds within an arm, the config is identical except `train.seed`. This is exactly the registered pairing claim.

### Metric and estimator

* Metric: **as registered** — `final_eval_metric(path, "nats_per_byte")` on the final eval record, which is step 20000 = `max_steps` in all 18 runs.
* Estimator: **as registered** — `mean(d) ± t_{0.9875,n-1}·sd(d)/√n`, `d_s = y_B,s − y_A,s`. My recomputed critical values are `t_{0.9875,4} = 3.495406` and `t_{0.9875,8} = 2.751524`, matching the registration's stated ≈3.50 and ≈2.75.

### ⚠ MAJOR DEVIATION — the stage-2 trigger never fired

The registration states: *"Stage 2 seeds: {6, 7, 8, 9} (cumulative n = 9, df = 8) — **run ONLY if the stage-1 CI straddles 0**."*

The stage-1 CI I recompute is **[+0.00081214, +0.00896826]** — it **excludes 0**. The registered gate therefore **did not open**, yet a full stage-2 (8 runs, seeds 6–9, both arms) exists in the packet. Under the preregistration as committed at 14:46:29Z, these runs should not exist.

The packet is transparent about this: the *extension registration* (19:22:29Z) explicitly re-authorises seeds 6–9 and states it was filed **after** the stage-1 outcome was known. That disclosure is creditable, and I verified it is timing-consistent (§4). But it does not convert the extension into a preregistered confirmatory look — it is, by its own text, **outcome-aware**. Consequences are itemised in §6.

---

## 3. RECOMPUTATION — all figures below are mine

### Paired differences (positive favours arm A; lower nats/byte is better)

| seed | y_A (arm A) | y_B (arm B) | d = y_B − y_A |
|---:|---:|---:|---:|
| 1 | 0.770528 | 0.772992 | +0.002464 |
| 2 | 0.767641 | 0.769449 | +0.001808 |
| 3 | 0.765805 | 0.772037 | +0.006232 |
| 4 | 0.766741 | 0.772891 | +0.006150 |
| 5 | 0.765865 | 0.773662 | +0.007797 |
| 6 | 0.763245 | 0.766413 | +0.003168 |
| 7 | 0.773541 | 0.776814 | +0.003273 |
| 8 | 0.768590 | 0.778289 | +0.009699 |
| 9 | 0.772754 | 0.773521 | +0.000767 |

### Stage 1 — the registered primary analysis (seeds 1–5, per-look α = 2.5%)

| quantity | value |
|---|---|
| n / df | 5 / 4 |
| mean(d) | **+0.00489020** nats/byte |
| sd(d) | 0.00260880 |
| se = sd/√n | 0.00116669 |
| t_{0.9875,4} | 3.495406 |
| half-width | 0.00407806 |
| **CI** | **[+0.00081214, +0.00896826]** |
| observed t = mean/se | 4.1915 |

* CI **entirely above 0**: **yes**.
* CI **straddles 0**: **no**.
* CI **entirely below ε = 0.01**: **yes** (upper limit 0.008968 < 0.010).

In interpretable units: mean(d) = 0.0049 nats/byte = 0.0071 bits/byte, i.e. **0.64 % relative** to the stage-1 arm-A mean of 0.7673 nats/byte.

Note on α: because the stage-2 gate never opened, only **one** look legitimately occurred, yet 2.5% (not 5%) was spent. Stage 1 is therefore *conservative* relative to a single-look 5% design. That is in the authors' disfavour and I record it as such.

### Cumulative n = 9 at the registered stage-2 estimator (recomputed for completeness only)

mean(d) = **+0.00459533**, sd = 0.00300307, se = 0.00100102, t_{0.9875,8} = 2.751524, **CI = [+0.00184099, +0.00734967]**; entirely above 0, entirely below ε.

### Extension criterion (outcome-aware; recomputed, not endorsed)

t = mean/se = **4.590632** vs. the registered threshold t_{0.99865,8} = 4.277 (I recompute the quantile as **4.276577** — the registration's rounding is accurate). One-sided 99.865 % lower bound = **+0.00031395 > 0**. So the label **STRONGLY_SUPPORTED does arithmetically hold**, on data whose existence violates the original gate and under a threshold that is not part of the registered α budget.

**No number in the packet's prose was taken on trust; all of the above are my own computations.**

---

## 4. OPTIONAL-STOPPING AUDIT

Registration commits (UTC, from `manifest.txt`): preregistration **14:46:29Z**; amendment 1 (ε) **17:19:15Z**; extension registration **19:22:29Z**; clause precedence **20:20:15Z**.

| Check | Result |
|---|---|
| Any confirmatory run **starting** before the preregistration commit | **No.** Earliest start 16:26:31.610Z (B-seed5), 1 h 40 m after 14:46:29Z. |
| Any confirmatory run **ending** before the preregistration commit | **No.** |
| Runs with no declared run time | **None** — all 18 have start+end. |
| Stage-1 window (seeds 1–5, 10 runs) | 16:26:31.610Z → 17:10:14.730Z |
| Stage-2 window (seeds 6–9, 8 runs) | 20:04:31.757Z → 21:01:23.785Z |
| ε amendment posted **after** all stage-1 runs finished | **Yes** (last stage-1 end 17:10:14.730Z < 17:19:15Z) — exactly as the registration itself declares ("post-run / pre-analysis"). |
| All stage-2 runs launched **after** the extension registration | **Yes** — earliest stage-2 launch 42.0 min after the 19:22:29Z commit. |
| Clause-precedence commit after first stage-2 launch, before any stage-2 completion | **Yes** (20:20:15Z; first launch 20:04:31.757Z, first completion 20:42:13.637Z) — consistent with its self-description. |

**Stage-2 existence vs. the registered trigger: FAILS** (see §2). Stage-2 data does *not* exist "only under the registered trigger"; it exists under a later, outcome-aware re-registration.

### Two things the audit cannot close

**(a) There is no independently verifiable clock anywhere in the data.** The union of all record keys across all 18 files is: `aux_loss, bits_per_byte, bytes_per_token, config, data_order_batches, data_order_digest, device, event, grad_norm, loss, lr, nats_per_byte, params, perplexity, run, step, tokenizer, train_tokens, val_tokens, wall_s`. There is **no absolute timestamp field** — only relative `wall_s`. The manifest itself concedes the run times are "operator-supplied … asserted metadata, not raw record content." **The entire optional-stopping audit therefore rests on unfalsifiable assertions.** Nothing in the packet could distinguish the declared timeline from a reconstructed one.

**(b) The A-seed9 provenance anomaly.** Stage-2 launch times cluster tightly — 7 of 8 runs launched within a **26-second** window (20:04:31.757 → 20:04:57.799). **A-seed9 alone launched at 20:21:00.405, 16.5 minutes late**, and is the *only* stage-2 run launched **after** the clause-precedence commit (20:20:15Z). A-seed9 is also precisely the run for which the ledger declares a KILLED duplicate (id prefix `7561f0b1`), asserted to have started *after* the registered run completed (21:01:23.785Z), with "no results artifact."

The packet contains no evidence that discriminates the declared story ("registered run first, duplicate afterwards") from the equally consistent alternative ("an A-seed9 attempt launched at ~20:04 with its siblings, died or was killed, and the artifact shipped here is a 20:21 relaunch"). Under the alternative, the shipped A-seed9 is a re-roll of a seed whose first attempt was discarded — a textbook selection channel. I note, in the authors' favour, that seed 9 carries the **smallest** difference (d = +0.000767); dropping it *raises* mean(d) to +0.005074. So if any selection occurred here it worked *against* H1. But the ambiguity is unresolved by the packet and must be closed by the raw server export.

---

## 5. ROBUSTNESS PROBES (reported, not decisive)

### Sign test (exact, one-sided, H1: d > 0)

* Stage 1: **5/5 positive, p = 0.03125.** Note this **does not clear the registered per-look α of 0.025** — and *cannot*, since 1/2⁵ = 0.03125 is the smallest attainable p at n = 5. Stage-1 significance is therefore wholly dependent on the parametric t-assumption; the distribution-free evidence at n = 5 is inherently incapable of reaching the registered threshold.
* Cumulative n = 9: **9/9 positive, p = 0.001953.** The consistent sign across all nine independent pairs is the single most persuasive feature of this dataset.

### Leave-one-out CI sensitivity

Stage 1 (n → 4, CI recomputed at t_{0.9875,3}):

| dropped | mean | CI | excludes 0? |
|---|---:|---|---|
| seed 1 | 0.005497 | [+0.000123, +0.010870] | yes |
| seed 2 | 0.005661 | [+0.000937, +0.010384] | yes |
| seed 3 | 0.004555 | [−0.001470, +0.010580] | **no** |
| seed 4 | 0.004575 | [−0.001482, +0.010632] | **no** |
| seed 5 | 0.004164 | [−0.000758, +0.009085] | **no** |

**The registered stage-1 conclusion is overturned by deleting any one of three of the five pairs.** Much of this is the brutal df penalty at n = 4 (t rises 3.495 → 4.177) rather than genuine instability of the mean — mean(d) only moves within [0.00416, 0.00566] — but the headline verdict is nonetheless one seed away from "inconclusive." This is a fragility of the registered n = 5 design, not a computational error.

Cumulative n = 9 (n → 8): **all nine LOO CIs exclude 0** (lower limits +0.00123 to +0.00224) **and all nine lie entirely below ε**. The nine-pair picture is genuinely robust to single-pair deletion; the five-pair picture is not.

### Per-pair spread

d ranges from **+0.000767 (seed 9)** to **+0.009699 (seed 8)** — a **12.6×** span; median +0.003273; sd(d) = 0.003003, i.e. **65 % of the mean effect**. Pairing is doing real work: corr(y_A, y_B) across seeds = **+0.627**, and pairing removes ~63 % of the unpaired variance (sd(y_A) = 0.00341, sd(y_B) = 0.00354 vs. sd(d) = 0.00300). The effect is small, consistently signed, and heterogeneous in magnitude.

### Evaluation-noise scale (relevant to a single-checkpoint metric)

The registered metric is one eval at step 20000 (20 eval batches). Across the last five evals (steps 18000–20000) of each run, the median within-run sd is **0.001342** nats/byte, implying a jitter sd of ~**0.001898** on a paired difference. That is **~40 % of the observed pair-to-pair variance**, and **mean(d) is only 2.42× this single-readout jitter sd**. The chosen readout is therefore noisy relative to the effect.

Unregistered sensitivity check — replacing the single final eval with the mean of the last five evals: stage 1 mean +0.004901, CI [+0.000123, +0.009679] (still above 0, still below ε); cumulative +0.005608, CI [+0.002739, +0.008477], 9/9 positive. **The conclusion is not an artifact of the particular checkpoint.**

### Metric-timing sensitivity

Mean d recomputed at earlier eval steps — step 19500: +0.005584 (5/5) / +0.005674 (9/9); 19000: +0.004429 / +0.005708; 18000: +0.004785 / +0.006226; 15000: +0.007023 / +0.007592. Sign is positive for every pair at every step checked. No evidence of a cherry-picked stopping step (if anything, the gap narrows toward the end).

### Anomaly flags

* **Non-finite values:** none, anywhere (all 18 files).
* **Truncated series:** none. All 18 files are structurally identical — 1042 lines, 1 `run_start`, 1000 `train`, 40 `eval`, 1 `run_end`; train steps 20→20000, eval steps 500→20000.
* **Irregular eval cadence:** none. Every run has exactly 40 evals at a uniform 500-step gap, matching `eval_interval = 500` and terminating at `max_steps = 20000`.
* **Duplicate digests across seeds:** none (9 distinct digests, each shared only within its pair — the correct pattern).
* **Duplicate/copy-pasted series:** none. md5 over each run's full (step, loss, nats_per_byte) eval series is unique across all 18 runs.
* **`kappa` telemetry:** absent from every run — `reader.read_final_kappa()` raises `TelemetryGap` for all 18 files. The reader ships a substantial, carefully-validated curvature-readout API (`kappa_spreads`, `KappaReadout`, shape-drift detection) that no file in this packet exercises. Irrelevant to the registered metric, but it means a non-trivial share of the "self-contained validated reader" is unexercised by the shipped data, and any packet-external chart built on it is unverifiable here.
* **Minor timing oddities:** `B-seed2` end timestamp is the only one lacking fractional seconds; `B-seed4` ran 43.4 min vs. a 37.6–40.4 min norm (its in-file `wall_s` = 2600.4 s agrees, so this is a genuinely slower run, not a metadata error).

---

## 6. VERDICT

### **SUPPORTED** — for the registered stage-1 primary criterion only, and only when quoted with the mandatory practical-relevance rider.

Quoting the registered criterion verbatim:

> *"Verdict rule: CI entirely above 0 -> H1 supported. CI entirely below +epsilon -> no practically relevant improvement."*

My recomputed stage-1 CI is **[+0.00081214, +0.00896826]** nats/byte (n = 5, df = 4, mean 0.00489020, sd 0.00260880, t_{0.9875,4} = 3.495406).

* It is **entirely above 0** → **H1 is supported**: arm A's embedding initialization attains lower final nats/byte than arm B's.
* Its upper limit (0.008968) is **entirely below ε = 0.01** → **the improvement is registered as not practically relevant.**

**Both clauses fire simultaneously**, and per the clause-precedence commit the direction label is primary while the practical-relevance rider is *mandatory on every quotation of the verdict*. Accordingly, the only admissible one-line statement of this result is:

> **"H1 supported (direction), with no practically relevant improvement (CI entirely below ε = 0.01)."**

Quoting "H1 supported" without the rider misrepresents the registration. I would reject any abstract or headline that does so.

The extension label **STRONGLY_SUPPORTED** recomputes correctly (t = 4.591 > 4.277) but is **not a preregistered confirmatory finding** and must be reported as an outcome-aware, exploratory extension, never as a replacement for or upgrade of the stage-1 verdict — a constraint the extension registration itself imposes and which I endorse.

I did **not** return INVALID: the stage-1 confirmatory analysis is itself fully compliant (correct seeds, correct metric through the registered reader, correct estimator and per-look α, zero exclusions needed or taken, zero pairing invalidations, all runs launched after the preregistration commit, α under-spent rather than over-spent). Later-executed runs cannot retroactively inflate stage-1's error rate. The protocol failure is real but is confined to the *stage-2/extension* material, which I quarantine as exploratory.

---

### MAJOR concerns

1. **The stage-2 gate never opened, yet stage-2 was run.** Registered: *"run ONLY if the stage-1 CI straddles 0."* The stage-1 CI [+0.000812, +0.008968] excludes 0. Eight runs (seeds 6–9, both arms) exist that the committed protocol did not authorise. Disclosed, but a deviation from the registered design.
2. **The extension changes the estimator and steps outside the registered α budget.** The registration allocated 5 % across exactly two looks (2.5 % each, Bonferroni). The extension instead tests a t-statistic against t_{0.99865,8} = 4.277 — a third look, on data that **re-uses all five stage-1 pairs**, at a threshold nowhere in the registered spending plan, and chosen *after* the stage-1 outcome was known. No familywise error statement for the combined reporting survives. "STRONGLY_SUPPORTED" has no preregistered error guarantee.
3. **A verdict-grammar ambiguity was resolved after the outcome was known, in the favourable direction.** The clause-precedence commit (20:20:15Z) — which declares *"the direction label is primary"* — post-dates the stage-1 analysis by hours. At that moment the authors already knew that both the favourable clause (above 0) and the unfavourable clause (below ε) had fired. Ranking the favourable one as primary is an outcome-aware interpretive choice. Mitigating: the same commit makes the unfavourable rider mandatory on every quotation, which materially limits the benefit obtained.
4. **The entire timing/optional-stopping audit is unfalsifiable from this packet.** The JSONL carries no absolute clock — only relative `wall_s`. Every ordering claim (runs after preregistration; ε after runs but before analysis; extension before extension runs) rests on operator-supplied metadata the manifest itself labels "asserted." I can confirm the assertions are mutually consistent; I cannot confirm they are true.
5. **A-seed9's provenance is unresolved.** It launched 16.5 min after the other seven stage-2 runs (which launched within 26 s of each other) and is the only stage-2 run launched after the clause-precedence commit — and it is exactly the run with a declared KILLED duplicate that has "no results artifact" and no timestamps. The packet cannot distinguish "registered run then duplicate" from "first attempt discarded, relaunch shipped." (Direction of any resulting bias would favour the null, since seed 9 has the smallest d.)

### MINOR concerns

1. **The abstract is stale and inconsistent with the packet.** It states *"Five seed pairs (seeds 1-5)"* while the packet ships **nine** pairs. Any quotation of a nine-pair number alongside this abstract misdescribes the study.
2. **ε was fixed after the runs completed** (17:19:15Z vs. last stage-1 end 17:10:14Z) and is decision-relevant: it alone determines whether the practical-relevance clause fires. I verified its stated arithmetic (0.01 bits/byte × ln2 = 0.006931 ✓; 0.01/0.7738 = 1.29 % ✓), and note the rounding 0.0069 → 0.010 is a 44 % inflation. **Crucially, it is self-penalising:** at the un-rounded 0.0069 the CI would *not* lie entirely below ε and the unfavourable rider would not fire. The post-hoc choice thus moved the verdict *against* the authors, which argues strongly against outcome-gaming here. Flagged for hygiene, not for suspicion.
3. **The n = 5 stage-1 result is one seed from inconclusive** — 3 of 5 leave-one-out CIs include 0 (driven mainly by the df penalty, not by movement of the mean). A design that can be overturned by a single pair is under-powered for the effect it is chasing.
4. **The distribution-free evidence at stage 1 cannot meet the registered α.** The exact sign-test p is 0.03125 > 0.025, and 0.03125 is the floor at n = 5. Stage-1 significance is entirely contingent on the t-assumption.
5. **The metric is a single noisy checkpoint.** mean(d) is only 2.42× the paired single-readout eval jitter, and jitter explains ~40 % of pair-to-pair variance. A registered multi-eval average would have been a materially better-conditioned estimator (and, as I show, gives the same answer here).
6. **Environment claims are unverifiable.** The extension registration asserts "same container image digest, same GPU class," but `run_start` logs only `device: "cuda"` — no image digest, no GPU model, no library versions. Nothing in the packet can check this.
7. **The digest attests pairing only if it is what it claims to be.** Within-pair equality and across-seed distinctness are exactly the right pattern, but I cannot verify from the packet that the digest is genuinely SHA-256 over the first 32 batches' source offsets rather than any other seed-derived value. The pairing evidence is only as strong as the unshipped emitting code.
8. **The declared pilot run (2026-08-17) is not in the packet**, so I cannot confirm it was a pilot rather than a first confirmatory attempt.
9. **`kappa` telemetry is absent from all 18 runs**, leaving a large, validated portion of the shipped reader unexercised by the shipped data.
10. Cosmetic: `B-seed2`'s end timestamp lacks fractional seconds; `perplexity` carries ~2e-5 rounding error (derived field, not the registered metric); `B-seed4` is a 43.4-min duration outlier corroborated by its own `wall_s`.

---

### Specific additional evidence that would change my verdict

**Would move SUPPORTED → INVALID:**

1. **A raw, unredacted tracking-server export** (machine-generated, with server-side record IDs and creation times, ideally externally anchored — e.g. signed commits or a timestamping authority) covering **every** run touching arms A/B on 2026-08-17/18, including all abandoned, killed and failed runs. Specifically: the **complete launch history for seed 9, arm A**. If an A-seed9 run exists with a start near 20:04:4x, the shipped A-seed9 is a relaunch after a discarded attempt and the extension set is contaminated by run selection.
2. **Evidence of any analysis of arms A/B predating 14:46:29Z**, or of more than one stage-1 analysis (the registration binds stage 1 to being "analyzed once"). Either would void the confirmatory status of stage 1 itself.
3. **Evidence that any additional seed-pair was run and not shipped** (the manifest lists exactly nine pairs; an undeclared tenth would make the reported set a selected subset).
4. **Evidence that ε was chosen with sight of A/B confirmatory outcomes**, contradicting the amendment's explicit blinding claim.

**Would move SUPPORTED → NOT SUPPORTED (or force retraction of the direction label):**

5. Any demonstration that `final_eval_metric` is not reading what the registration means by "final nats/byte" — e.g. an eval harness whose held-out split differs between arms, or a `data_order_digest` that is not in fact computed over batch offsets (making the pairing claim vacuous and the paired estimator inappropriate).

**Would materially strengthen the result (and let me drop MAJOR items 1–3 to MINOR):**

6. **A fresh, properly preregistered confirmatory replication** on new seeds (say 10–20) with a single look, a full 5 % budget, ε fixed *before* any run, and the primary metric defined as the mean of the last k evals rather than one checkpoint. Given mean(d) ≈ 0.0046 and sd(d) ≈ 0.0030, ~10–12 pairs would give comfortable power for the direction claim — but note that **no achievable n rescues practical relevance**: the effect is ~0.6 % relative and the CI already sits entirely below the ratified ε.
7. **The filed stage-1 verdict document with its own commit timestamp**, demonstrating it was written between 17:19:15Z and 19:22:29Z and that the extension was registered only afterwards.
8. **Container image digest, GPU model, and library-version attestations inside `run_start`**, substantiating the extension registration's "same image, same GPU class" claim.

**Standing recommendation regardless of the above:** correct the abstract (five pairs → nine, with the extension explicitly labelled outcome-aware/exploratory), and never quote the direction label without the mandatory rider — *the improvement is real in sign and statistically distinguishable from zero, but the registered practical-relevance threshold classes it as not practically relevant.*
