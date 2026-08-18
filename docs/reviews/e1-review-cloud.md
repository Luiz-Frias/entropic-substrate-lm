# E1 stage-1 — second independent blind review (cloud sandbox lane)

Produced 2026-08-18 by `scripts/blind_review_cloud.py`: model `claude-opus-5`
reviewing inside Anthropic's code-execution sandbox — server-side enforced
NO-network isolation, packet files delivered via the Files API only, zero
repository or operator-machine access. Same packet, same verbatim protocol
prompt as the first review (`e1-review.md`). Filed unedited below.

Cross-review reconciliation: every recomputed quantity (mean, sd, t, CI,
per-pair d) matches the first review and the filed verdict artifact to every
printed digit; same SUPPORTED-per-registered-criterion verdict. New in this
review: the clause-precedence defect (both registered clauses fire in the
observed region with no declared precedence), the wall_s-vs-asserted-duration
compatibility check, and the 32.3s launch-window evidence against seed
peeking. The clause-precedence finding is answered for the extension by the
precedence declaration added to docs/e1-extension-preregistration.md before
any extension analysis.

---

I'll start by inventorying the packet and verifying integrity.
Now let me verify integrity — every md5 in the manifest.
All 13 hashes verify. Proceeding to adherence and recomputation via `reader.py` only.
Primary estimate recomputed. Now the pairing invariant, timestamps, and anomaly probes.
# Independent Statistical Review — E1 Review Packet

**Reviewer:** anonymous, independent. No prior contact with the authors or the project; no external sources consulted. Every number below was recomputed by me from the packet files; nothing is taken from the abstract or preregistration on trust.

**Scope note:** the packet contains a preregistration, an amendment, a manifest, a validated reader, and ten JSONL run series. It contains **no results section and no numeric claim** — the abstract states the design and the direction of the registered hypothesis but reports no effect size, no interval, and no conclusion. I therefore adjudicate the *registered decision rule* against the data, rather than auditing an author-stated number.

---

## 1. INTEGRITY — PASS

All 13 md5 digests in `manifest.txt` verify byte-exactly against the supplied (path-flattened) files.

| file | manifest md5 | status |
|---|---|---|
| abstract.md | 6e0deb09a69bdba774c730abe5b4fe4c | OK |
| data/A-seed1.jsonl | 05bfd13e6504abfd55484b6daab61700 | OK |
| data/A-seed2.jsonl | 4127a48c7f184cf245f6b16b285fceb1 | OK |
| data/A-seed3.jsonl | a846d3eceace2bac3ce0d09aef75db72 | OK |
| data/A-seed4.jsonl | 308038a9a31c6a17c47692f0d599710e | OK |
| data/A-seed5.jsonl | 3df29981e57e947d02cefa3fb70b01e3 | OK |
| data/B-seed1.jsonl | c8e6570c9990272c3959f4ca2fb91781 | OK |
| data/B-seed2.jsonl | 1fddf06adc61d80cefe0086db0aab514 | OK |
| data/B-seed3.jsonl | d19202904c1256887d361c1bdc21888b | OK |
| data/B-seed4.jsonl | c0c17ea0943944989ee0f9de461fab20 | OK |
| data/B-seed5.jsonl | a2a5a529059110a8e23c967261d5f771 | OK |
| preregistration.md | 0f25d2ae72ef9246e74a8d203ae5cc13 | OK |
| reader.py | 733ca1e63f3961f88bdf190ced6504a1 | OK |

No mismatches, no missing files, no unlisted files other than `manifest.txt` itself (self-reference, expected). Review proceeds.

Caveat of principle: the manifest is unsigned. It attests self-consistency of the packet, not that the packet is the complete set of runs performed. A hash list cannot detect a file that was never included.

---

## 2. ADHERENCE — no deviation found

Seeds enumerated from `data/` and cross-checked two ways: from the filename, and independently through `reader.read_seed()` (which reads `run_start.config.train.seed`). All ten agree.

- **Arm A seeds present:** {1, 2, 3, 4, 5}. **Arm B seeds present:** {1, 2, 3, 4, 5}. Registered stage-1 set: {1, 2, 3, 4, 5}. **Exact match, complete pairs, no extras.**
- **Stage-2 seeds {6, 7, 8, 9}:** absent. Correct — see §4.
- **Exclusions:** the registration permits exactly one ground ("a run is excluded solely for a non-finite final loss"). I applied `final_eval_metric` to all ten runs: zero raise, zero non-finite values, so **zero exclusions were warranted and zero were taken**. Analysed n = 5 pairs, df = 4, matching the registered "Stage 1 seeds: {1,2,3,4,5} (df = 4)". No partial-pair analysis.
- **Metric:** read exclusively via `reader.final_eval_metric(path, "nats_per_byte")`, as registered. The final eval record in every run is at `step = 20000`, i.e. the end of the registered 20,000-step budget — not an early or hand-picked checkpoint.
- **Estimator:** `mean(d) ± t_{0.9875,n-1}·sd(d)/√n` with `d = y_B − y_A`, sample sd (ddof = 1). I verified the registration's own critical values: t(0.9875, 4) = 3.495406 ("≈3.50" ✓) and t(0.9875, 8) = 2.751524 ("≈2.75" ✓). The sign convention in the registration (H1 predicts mean(d) > 0 with lower-is-better) is internally coherent.
- **Pairing invariant (Appendix A):** each run emits exactly one `data_order_digest`, at step 40, attesting `data_order_batches = 32`. This is the correct emission point: `log_interval = 20`, so step 40 is the first logged train record at/after batch 32. All runs have `config.train.resume = null` (fresh runs, so attestation is legitimate). **Within-pair digests match for all five pairs; the five pair-digests are mutually distinct.** No pair is invalidated, and there is no evidence of unvaried seeds.
- **Arm isolation:** flattening the full `run_start` config and diffing A against B within each seed, the *only* differing keys are `config.model.embedding_init` (`init-A` vs `init-B`) and the two cosmetic labels `config.name` / `run`. All 40+ other config fields — architecture (12L/12H/d768, 85,150,464 params), optimiser, lr schedule, data path, `seed`, `eval_interval`, `log_interval` — are identical. The registration's claim that "only the embedding initialization differs" holds in the logged configuration.

**Adherence verdict: no MAJOR adherence finding.** This is the strongest part of the packet.

---

## 3. RECOMPUTATION (all values my own, via `reader.py` only)

Final nats/byte at step 20000, and paired differences `d = y_B − y_A` (positive favours A):

| seed | y_A | y_B | d |
|---:|---:|---:|---:|
| 1 | 0.770528 | 0.772992 | +0.002464 |
| 2 | 0.767641 | 0.769449 | +0.001808 |
| 3 | 0.765805 | 0.772037 | +0.006232 |
| 4 | 0.766741 | 0.772891 | +0.006150 |
| 5 | 0.765865 | 0.773662 | +0.007797 |

```
n        = 5            df = 4
mean(d)  = 0.0048902
sd(d)    = 0.0026088    (ddof = 1)
se       = 0.0011667
t(0.9875, 4) = 3.495406
REGISTERED CI (per-look alpha 2.5%, two-sided 97.5%):
         [ +0.0008121 , +0.0089683 ]
```

- **CI entirely above 0:** yes (lower bound +0.00081).
- **CI entirely below +epsilon (= 0.01):** **also yes** (upper bound +0.00897).
- CI straddles 0: no.

Estimator-implementation sensitivity (the sign of the lower bound is not an artefact of my choices): rounded t = 3.50 → [+0.000807, +0.008974]; ddof = 0 (incorrect) → [+0.001243, +0.008538]; normal quantile instead of t (incorrect) → [+0.002275, +0.007505]. All lower bounds > 0.

Supporting, **not registered**: paired t = 4.1915, two-sided p = 0.0138. Mean improvement = 0.633 % of mean y_B.

Amendment arithmetic checks out: 0.01 bits/byte × ln 2 = 0.006931 nats/byte ("≈0.0069" ✓); 1.3 % × 0.7738 = 0.010059, rounded to 0.01 ✓. Internal record consistency also checks out: `bits_per_byte = nats_per_byte / ln 2` to 1e-6, `loss == nats_per_byte` exactly (`bytes_per_token = 1.0`), and `perplexity ≈ exp(loss)` to ≤ 2.1e-5 relative — the last being pure 5-significant-digit rounding of the logged field, benign.

---

## 4. OPTIONAL-STOPPING AUDIT — passes on the evidence available, but the evidence is weak by construction

Registration commit 2026-08-18T14:46:29Z; epsilon amendment 2026-08-18T17:19:15Z.

| run | start (UTC) | end (UTC) | end−start (s) | logged `wall_s` | Δ |
|---|---|---|---:|---:|---:|
| A-seed1 | 16:26:49.998 | 17:04:36.909 | 2266.9 | 2263.1 | 3.8 |
| A-seed2 | 16:26:41.109 | 17:05:04.376 | 2303.3 | 2299.3 | 3.9 |
| A-seed3 | 16:27:00.088 | 17:06:11.280 | 2351.2 | 2346.7 | 4.5 |
| A-seed4 | 16:27:00.008 | 17:05:50.393 | 2330.4 | 2326.3 | 4.1 |
| A-seed5 | 16:26:40.786 | 17:04:36.425 | 2275.6 | 2271.8 | 3.9 |
| B-seed1 | 16:26:31.935 | 17:04:41.407 | 2289.5 | 2285.3 | 4.2 |
| B-seed2 | 16:26:33.152 | 17:05:03 | 2309.8 | 2305.5 | 4.4 |
| B-seed3 | 16:27:03.925 | 17:05:40.490 | 2316.6 | 2311.8 | 4.8 |
| B-seed4 | 16:26:49.491 | 17:10:14.730 | 2605.2 | 2600.4 | 4.8 |
| B-seed5 | 16:26:31.610 | 17:04:06.215 | 2254.6 | 2250.4 | 4.2 |

Findings:

1. **No confirmatory run predates the preregistration commit.** Earliest start 16:26:31.610Z = prereg + 100.0 min. All ten pass.
2. **All ten runs end before the epsilon amendment.** Latest end 17:10:14.730Z = amendment − 9.0 min. This is consistent with the registration's own description ("committed after the runs completed but before any analysis"), and it means epsilon was *not* fixed in advance of seeing the data — see MAJOR-2.
3. **Consistency cross-check:** every asserted duration exceeds the run's own logged `wall_s` by 3.8–4.8 s, a tight and plausible process-startup/teardown margin. The asserted metadata is at least *arithmetically compatible* with the record content. This is the only independent purchase available on the timestamps.
4. **Launch pattern:** all ten starts fall inside a 32.3-second window — a single simultaneous batch launch, not a sequential or adaptive schedule. This is affirmative evidence against seed-by-seed peeking within stage 1.
5. **Stage 2 correctly absent.** The registered trigger is "run ONLY if the stage-1 CI straddles 0." My recomputed stage-1 CI does not straddle 0, so stage 2 must not exist — and no seed 6–9 files are present. Adherent. Note this also means the Bonferroni split is *unused* in the direction that would have cost the authors: they spent 2.5 % on one look and never took the second, so the realised procedure is conservative relative to a single 5 % look.

**Limitation, and it is structural:** the JSONL records contain **no absolute wall-clock timestamps of any kind** — the only time-like field anywhere in 10,420 records is the relative `wall_s`. Every absolute time above comes from the manifest, which itself flags them as "operator-supplied … asserted metadata, not raw record content." The optional-stopping audit is therefore **not falsifiable from the packet**: a discarded earlier confirmatory batch would leave no trace here, and the declared 2026-08-17 arm-A pilot is not included, so I cannot confirm it is the only pre-run.

---

## 5. ROBUSTNESS PROBES (reported, not decisive)

**Sign test.** 5/5 differences favour A. Exact binomial two-sided p = 0.0625; one-sided p = 0.03125. These are the *minimum attainable* values at n = 5 — the two-sided sign test cannot reach 0.05 with five pairs, so its failure to clear 0.05 is a property of the design, not evidence against the effect. Wilcoxon signed-rank (exact): W = 0, p = 0.0625, likewise floored.

**Leave-one-out CI sensitivity — the most important robustness result.** Dropping one whole pair and recomputing at the registered per-look 2.5 %:

| dropped | n | mean | sd | CI | CI > 0? |
|---|---:|---:|---:|---|---|
| seed 1 | 4 | 0.005497 | 0.002573 | [+0.000123, +0.010870] | **yes** |
| seed 2 | 4 | 0.005661 | 0.002262 | [+0.000937, +0.010384] | **yes** |
| seed 3 | 4 | 0.004555 | 0.002885 | [−0.001470, +0.010580] | **no** |
| seed 4 | 4 | 0.004575 | 0.002901 | [−0.001482, +0.010632] | **no** |
| seed 5 | 4 | 0.004164 | 0.002357 | [−0.000758, +0.009085] | **no** |

**Three of five single-pair deletions destroy the registered conclusion.** With the full sample the lower bound is +0.00081 — 0.7 σ_mean above zero. The conclusion is knife-edge with respect to sample composition.

Non-registered percentile bootstrap (10,000 resamples, [1.25 %, 98.75 %]): [+0.00269, +0.00714]; 0.00 % of bootstrap means ≤ 0. The bootstrap looks reassuring but is not trustworthy at n = 5 (it cannot represent tail uncertainty with five distinct values) — I report it only for completeness and would not weight it.

**Per-pair spread.** min d = 0.001808, max d = 0.007797 — a **4.31× range**; CV(d) = 0.533. The differences look grouped rather than homogeneous: seeds 1–2 give +0.0018/+0.0025, seeds 3–5 give +0.0062/+0.0062/+0.0078. The effect is carried by seeds 3–5, which is exactly why the LOO table breaks when one of those is removed. With n = 5 this could easily be noise, but it is unmodelled heterogeneity and it is the mechanism of the fragility.

**Pairing did not work.** sd(d) = 0.002609 is *larger* than the within-arm sd of A (0.001946) or B (0.001646), and Pearson r(y_A, y_B) = −0.049. The registered premise — same seed ⇒ correlated outcomes ⇒ variance reduction — is not borne out. Note the direction of this: it makes the registered paired estimator the *conservative* choice (unpaired Welch on the same numbers gives t = 4.29, p = 0.0028, versus paired p = 0.0138). The authors did not gain from the pairing; they lost power. The pairing remains the correct pre-registered analysis and I would not accept a switch to the unpaired test.

**Anomalies — systematically checked:**

- Non-finite values: **none**, in any `loss` or `nats_per_byte`, in any train or eval record, in any run.
- Truncated series: **none**. Every run has exactly 1 `run_start`, 1000 `train`, 40 `eval`, 1 `run_end`. Train steps 20…20000 with a uniform gap of 20 (`log_interval = 20`); eval steps 500…20000 with a uniform gap of 500 (`eval_interval = 500`). All ten runs terminate at step 20000 with a `run_end`. No JSON parse failures.
- Irregular eval cadence: **none** — the gap set is exactly {500} for all ten runs.
- Duplicate digests across seeds: **none** — 5 distinct digests over 10 runs, one per pair, exactly as the design requires. Seeds were genuinely varied.
- Kappa telemetry: `read_eval_series` raises on no run, but **zero** eval records carry `kappa`. The reader ships substantial curvature machinery (`KappaReadout`, shape-drift detection) that is entirely dormant on this packet; `read_final_kappa` would raise on all ten runs. Not a defect in the primary analysis, but a sign that the reader was written for a superset of this experiment.

**Read-out fragility (this is not a registered analysis; I ran it only to characterise noise).** The registered metric is a single point — the step-20000 eval. In 9 of 10 runs the step-20000 eval is *not* the run's minimum; the best eval sits at step 17500–19500. Median within-run sd across the final six evals is 0.001931, so **the claimed effect is only ~2.5× the eval-to-eval noise of a single run**. Substituting other (non-registered) read-outs:

| read-out | mean(d) | CI (2.5% per-look) | CI > 0? |
|---|---:|---|---|
| step 20000 (**registered**) | +0.004890 | [+0.000812, +0.008968] | yes |
| mean of last 3 evals | +0.004968 | [+0.000214, +0.009721] | yes |
| mean of last 5 evals | +0.004901 | [+0.000123, +0.009679] | yes |
| step 19500 | +0.005584 | [+0.000416, +0.010752] | yes |
| step 19000 | +0.004429 | [−0.001595, +0.010452] | **no** |
| min over all evals | +0.005538 | [−0.000301, +0.011376] | **no** |

Two of six plausible read-outs would have reversed the conclusion. The registered choice was fixed in advance, so this is **not** a specification-search finding against the authors — but it quantifies how little margin the result has.

The strongest thing in the packet cuts the other way and deserves equal prominence: **the direction is unanimous across the whole tail of training.** Over the last six eval points, B − A > 0 in 6/6 looks for every one of the five seeds — 30/30 look-seed cells favour A. That is far more internally consistent than a single lucky final checkpoint, and it is real evidence that the sign of the effect is not a read-out artefact.

**Measurement-precision observation.** From `config.data`: `num_eval_batches = 20`, `batch_size = 16`, `seq_len = 1024` ⇒ each eval scores 327,680 bytes, i.e. **6.55 % of the 5,000,000-byte validation split**. The primary endpoint is a small-subsample estimate, which plausibly explains the ~0.0019 within-run eval noise. Whether the same 20 batches are used at every eval and in both arms is not recoverable from the packet; if the eval subsample is resampled, a material part of `sd(d)` is measurement noise rather than seed variance.

---

## 6. VERDICT

### **SUPPORTED** — narrowly, and with a registration defect that partially self-negates the finding.

Quoting the registered criterion: *"Verdict rule: CI entirely above 0 -> H1 supported."*

My recomputed registered interval is **[+0.0008121, +0.0089683]** (n = 5, df = 4, mean(d) = 0.0048902, sd(d) = 0.0026088, t(0.9875,4) = 3.4954). It lies entirely above 0. The registered criterion for H1 is met, on the registered metric, with the registered estimator, at the registered per-look alpha, on the registered seed set, with zero exclusions and no stage-2 look. I find no adherence deviation and no integrity failure, so INVALID is not available; and the interval does not straddle or fall below 0, so NOT SUPPORTED is not available.

**This verdict must be read with its immediate rider.** The very next sentence of the same rule states: *"CI entirely below +epsilon -> no practically relevant improvement."* With epsilon = 0.01, my CI upper bound of +0.00897 satisfies **that clause too**. Both branches fire simultaneously. On the registration's own terms, arm A's advantage is **directionally established and practically negligible**: mean(d) = 0.0049 nats/byte, 48.9 % of epsilon, 0.63 % of the baseline level. Any statement of this result that reports the first clause without the second misrepresents the preregistration.

### MAJOR concerns

1. **The registered decision rule is self-contradictory in the region actually observed.** The interval (0, epsilon) satisfies both "H1 supported" and "no practically relevant improvement." The registration provides no precedence between the clauses, so the packet cannot yield a single unambiguous readout. Any writeup claiming a clean "H1 supported" is selectively quoting its own protocol.
2. **Epsilon was fixed after the confirmatory data existed on disk.** Last run ended 17:10:14.730Z; the epsilon amendment was committed 17:19:15Z — **9 minutes later**. The registration asserts the amender had "NO access to any confirmatory outcome," and its stated basis (0.01 b/B ≈ 0.0069 nats/B; 1.3 % of 0.7738) is arithmetically sound and references quantities outside the comparison. But this is an unverifiable self-attestation about a 9-minute window in which the numbers were readable. I record that the resulting bias runs *against* the authors' hypothesis (a larger epsilon makes their effect look less relevant), which is mitigating but does not make the claim checkable.
3. **The conclusion is not robust to sample composition.** Dropping any one of pairs 3, 4, or 5 moves the lower bound below zero (3/5 LOO deletions overturn the result). The full-sample lower bound is +0.00081, i.e. 0.7 σ_mean from the boundary. A result this close to its threshold at n = 5 should not be presented as an established effect.
4. **The optional-stopping audit is unfalsifiable from the packet.** No record anywhere carries an absolute timestamp; all run times are operator-asserted manifest comments (so labelled). The strong start-clustering (32.3 s) and the tight `wall_s` agreement (3.8–4.8 s overhead) are genuinely reassuring, but a discarded prior batch, or a stage-1 set chosen after a peek, would be invisible. The declared 2026-08-17 arm-A pilot is not in the packet, so "pilots are never part of the confirmatory set" cannot be checked either.
5. **The effect is small relative to the measurement noise of its own endpoint.** mean(d) = 0.0049 versus median within-run tail eval sd = 0.0019 (2.5×); the step-20000 eval is not the minimum in 9/10 runs; each eval samples only 6.55 % of the validation split. Two of six alternative (non-registered) read-outs would have flipped the verdict. The point estimate is a noisy single draw, and the CI does not account for endpoint measurement error.
6. **Single-condition evidence base.** One dataset (enwik8), one architecture (85.15 M, 12L/12H), one step budget (20,000 ≈ 3.45 epochs), one lr schedule, five seeds. Nothing in the packet speaks to whether a 0.005 nats/byte initialization effect survives longer training, larger models, or another corpus — and initialization effects are precisely the class of effect expected to shrink with training.

### MINOR concerns

1. The pairing bought nothing: sd(d) = 0.002609 > sd(y_A) = 0.001946, r(y_A, y_B) = −0.049. The design's variance-reduction rationale is unsupported by its own data (though it makes the registered test conservative). Report this rather than silently keeping the "paired" framing as a strength.
2. Marked heterogeneity in d (4.31× min-to-max; seeds 1–2 ≈ 0.002 vs seeds 3–5 ≈ 0.006–0.008), unmodelled and unremarked.
3. The abstract reports no result at all — no n after exclusions, no mean, no interval, no verdict. A reader cannot check any claim because none is made; conversely the packet gives no assurance about how the numbers will eventually be characterised.
4. `reader.py` exposes **no accessor for `data_order_digest`/`data_order_batches`**, despite Appendix A making that field the sole evidence for pair validity, and no pair-level validity helper. Verifying the registration's own invalidation rule required the private `_iter_records`. The validated reader should implement the check the protocol depends on.
5. Extensive kappa machinery in `reader.py` is dead on this packet (0 of 400 eval records carry `kappa`); `read_final_kappa` would raise on all ten runs. Suggests the reader was not written for, or trimmed to, this experiment.
6. `perplexity` disagrees with `exp(loss)` by up to 2.1e-5 relative — 5-significant-digit rounding of the logged field, harmless, but it means logged derived fields should not be used for arithmetic.
7. The registration's t-value gloss ("≈3.50") is a rounding of 3.4954; harmless (the CI is unchanged in sign either way) but the packet should state which was used.
8. Requiring a two-sided 97.5 % lower bound > 0 for a one-sided H1 is a one-sided test at 1.25 % per look (2.5 % total), i.e. stricter than the nominal 5 % budget. Conservative, in the authors' disfavour, and worth stating explicitly rather than leaving as an inference.
9. The sign and Wilcoxon tests are floored at p = 0.0625 two-sided at n = 5 and cannot corroborate at conventional levels regardless of the data — a design limitation that should be acknowledged rather than reported as a negative result.

### Evidence that would change my verdict

**Toward an unqualified positive claim:**
- The registered **stage 2 (seeds 6–9, cumulative n = 9, t(0.9875,8) = 2.7515)**, run as a fresh preregistered extension with the epsilon clause given explicit precedence *before* the runs. If the cumulative lower bound stays above 0 with LOO stability across all nine pairs, MAJOR-3 dissolves. This is the single highest-value addition.
- A prespecified, endpoint-noise-aware primary metric — full 5 M-byte validation split, and/or an averaged terminal window (e.g. mean of the last k evals) — which would remove MAJOR-5. I note the last-3 and last-5 averages already give CI > 0, so this is likely to help the authors.
- Cryptographic or third-party timestamping of the amendment (signed commit, notarised hash, tracking-server export with per-record absolute times), plus the pilot's JSONL, retiring MAJOR-2 and MAJOR-4.
- Absolute UTC timestamps emitted **inside** each record, and a complete run ledger including abandoned runs, making the optional-stopping audit falsifiable.
- Replication on a second corpus, a second model scale, or a longer budget showing the effect does not decay — addressing MAJOR-6.

**Toward NOT SUPPORTED / INVALID:**
- Any additional stage-1-eligible pair, or any run within the registered seed set, that exists but is not in the packet — this would make the analysed set a selected subset and would render the packet **INVALID**, whatever the arithmetic.
- Evidence of a confirmatory run (this arm pair, this configuration) before 2026-08-18T14:46:29Z, or of stage-1 results having been inspected before the seed set or epsilon were fixed: **INVALID**.
- Evidence that the eval subsample differed between arms within a pair, or that `data_order_digest` does not in fact bind the training order it claims to (e.g. the digest is computed over something arm-invariant by construction): the pairing claim collapses and with it the estimator's justification.
- An authoritative reading of the protocol in which the epsilon clause takes precedence over the CI-above-0 clause — under which the correct readout becomes "no practically relevant improvement," i.e. effectively **NOT SUPPORTED** as a useful finding.
- A stage-2 extension whose cumulative CI includes 0: **NOT SUPPORTED**.

**Bottom line.** The packet's mechanics are unusually clean — hashes verify, seeds match the registration exactly, pairs are digest-attested and mutually distinct, only the embedding initialization differs, the series are complete and regular, no exclusions were needed or taken, the stage-2 trigger was correctly not fired, and the direction is unanimous across 30/30 terminal look-seed cells. That earns SUPPORTED on the letter of the registered criterion. But the margin is thin (lower bound +0.0008, overturned by 3 of 5 single-pair deletions), the endpoint is noisy relative to the effect, the key practical-relevance threshold was fixed nine minutes after the data landed, and the registration's own epsilon clause simultaneously classifies this improvement as not practically relevant. The defensible summary is: **a real but small and fragile directional effect, statistically supported at n = 5 and practically negligible on the authors' own registered yardstick.** It should not be reported as a clean confirmation without the epsilon clause attached.
