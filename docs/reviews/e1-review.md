# E1 stage-1 — independent blind review (filed 2026-08-18)

Produced by an isolated reviewer session: `claude --bare --print` (model
claude-opus-5), stripped environment, neutral cwd outside any repository, no
web tools, zero conversation history — input was solely the verbatim reviewer
prompt from `docs/peer-review-protocol.md` plus the packet directory
(`docs/reviews/e1-stage1-packet/`, committed on this branch's parent). Filed
unedited below. The sealed A/B mapping was disclosed only after this review
completed (`e1-stage1-sealed-mapping.md`, same commit).

---

# Independent Statistical Review — Packet `./packet` (E1: paired A/B embedding-init comparison)

I recomputed every quantity below myself, from the raw JSONL, via `reader.py` exclusively where the task required it. I did not use or trust any number asserted in `abstract.md` or `preregistration.md`.

---

## 1. INTEGRITY

Recomputed MD5 of all 13 listed files (Python `hashlib.md5`, not a system tool that wasn't installed in the sandbox) and diffed against `manifest.txt`:

| Result |
|---|
| All 13 files match their manifest MD5 exactly (`abstract.md`, 10 `data/*.jsonl`, `preregistration.md`, `reader.py`). |

**No mismatches. Proceeding.**

---

## 2. ADHERENCE

- **Seeds present:** `data/` contains exactly `A-seed{1..5}.jsonl` and `B-seed{1..5}.jsonl` — 10 files. This is exactly the registered **Stage 1** seed set `{1,2,3,4,5}`. **No Stage 2 seeds (6-9) are present.**
- **Stage-2 trigger check:** Stage 2 is registered to run "only if the stage-1 CI straddles 0." My recomputed Stage-1 CI (§3) is `[0.00081, 0.00897]`, entirely above 0 — it does **not** straddle 0, so the absence of Stage-2 data is correct, not a deviation.
- **Pairing / digest attestation:** Each run's `run_start.config.train.seed` matches its filename seed exactly, and every A/B pair at a given seed shares an identical `data_order_digest` (SHA-256 over the first 32 batches' source offsets, per Appendix A), while every seed's digest is distinct from every other seed's. No cross-seed digest collisions (which would indicate seeds weren't actually varied) and no in-pair digest mismatches — so **no pair is excludable on the pairing-invalidation rule.**
  - Caveat (MINOR): the digest attests only the first 32 batches of a 20,000-step run. Identical data order beyond batch 32 is inferred from the pairing design, not attested.
- **Exclusion rule:** the only registered exclusion is a non-finite final loss, dropping the whole pair. A raw scan of every `loss` and `nats_per_byte` field in all 10 files (10,420 lines total) found **zero non-finite values**. No exclusions were needed, and none were applied — consistent.
- **Config equivalence:** For every seed, I diffed the full `run_start.config` between A and B. The **only** differing keys are `model.embedding_init` (`init-A`/`init-B`) and the cosmetic `name`/`run` label. Everything else (architecture, data, batch size, seq_len, optimizer, seeds, max_steps=20000, etc.) is identical — matches the prereg's claim.
- **Metric/estimator adherence:**
  - Metric read exclusively via `reader.py`'s `final_eval_metric(path, "nats_per_byte")`, as registered. Confirmed the "final" record picked by the reader is at `step=20000` for all 10 runs (i.e., true training completion, not a mid-run eval).
  - Estimator: paired difference `d_s = y_B,s − y_A,s`, mean, sample sd (n−1), CI = mean ± t·sd/√n. I independently implemented the Student-t quantile (regularized incomplete beta, no scipy) and got t₀.₉₈₇₅,₄ = 3.49541 and t₀.₉₈₇₅,₈ = 2.75152 — matching the registration's stated "≈3.50" / "≈2.75". Estimator matches registration.
  - Epsilon arithmetic check: 0.01 bits/byte × ln2 = 0.006931 nats/byte ✓ ("≈0.0069" as stated). 0.01 / 0.7738 = 1.29% ✓ ("≈1.3%" as stated). The amendment's stated arithmetic is internally consistent.

No adherence deviations found. All are consistent with the registration **as documented in this packet.**

---

## 3. RECOMPUTATION

Parsed only through `reader.py::final_eval_metric`. Per-seed values (nats/byte) and paired differences `d = B − A`:

| seed | A | B | d = B−A |
|---|---|---|---|
| 1 | 0.770528 | 0.772992 | 0.002464 |
| 2 | 0.767641 | 0.769449 | 0.001808 |
| 3 | 0.765805 | 0.772037 | 0.006232 |
| 4 | 0.766741 | 0.772891 | 0.006150 |
| 5 | 0.765865 | 0.773662 | 0.007797 |

- n = 5, mean(d) = **0.0048902**, sd(d) = **0.0026088** (n−1), SE = 0.0011667
- Registered Stage-1 CI (t₀.₉₈₇₅,₄ = 3.4954, precise): **[0.000812, 0.008968]**
- Using the registration's rounded t=3.50: [0.000807, 0.008974] — negligible difference from precision.

**CI is entirely above 0.** It is *also* entirely below epsilon = 0.01 (upper bound 0.00897 < 0.01). Both registered conditions fire simultaneously (see §6).

Effect size in context: mean(d) ≈ 0.0049 nats/byte is ≈ 0.6% of the ~0.77 nats/byte baseline, and about half of the registered practical-relevance threshold (epsilon = 0.01).

---

## 4. OPTIONAL-STOPPING AUDIT

Using the manifest's operator-supplied run timestamps (UTC):

- Preregistration commit: **2026-08-18T14:46:29Z**
- Earliest confirmatory run start: 2026-08-18T16:26:31.610Z (B-seed2) — **after** prereg commit ✓ (all 10 runs start after prereg, verified individually)
- Latest confirmatory run end: 2026-08-18T17:10:14.730Z (B-seed4)
- Amendment 1 (epsilon) commit: **2026-08-18T17:19:15Z** — **after** every run ended (by ≥9 minutes), and prior to any analysis artifact in this packet.
- Stage 2: absent, and correctly so — the stage-1 CI does not straddle 0 (§2/§3), so the registered trigger for Stage 2 was never met.

On the face of the packet's own timestamps, no confirmatory run predates the prereg, and the epsilon amendment postdates all runs but predates analysis, exactly as the prereg narrates.

**Important limitation (see MAJOR findings):** these timestamps are explicitly labeled in the manifest as *"asserted metadata, not raw record content"* — I could not cross-check them against anything independently generated inside the JSONL files (the `run_start`/`run_end` records carry no wall-clock timestamps of their own). The same applies to the preregistration text itself: `preregistration.md` as shipped is the **post-amendment** document; I have no way to confirm from this packet alone what the pre-run, pre-amendment prereg actually said, since only commit hashes (`0a917c3812a6`, `224a3eb59eff`) are cited, with nothing in the packet to verify them against. The entire optional-stopping story is therefore **assertion, not proof**, from what's in front of me.

One micro-anomaly consistent with this: `B-seed2.jsonl end=2026-08-18T17:05:03+00:00` lacks the microsecond field present on every other timestamp in the manifest block — suggestive of hand-transcription rather than a uniform machine export.

---

## 5. ROBUSTNESS PROBES (reported, not adjudicated)

**Sign test:** 5/5 pairs have d > 0 (unanimous direction, consistent with H1). Exact one-sided binomial p under p=0.5 is 1/32 = 0.03125. Note: with n=5, the smallest achievable one-sided sign-test p-value is 0.03125, which **cannot** reach the registered per-look one-sided alpha (1.25%). The sign test is directionally unanimous but structurally underpowered to confirm significance at the registered stringency — it corroborates direction only, nothing more.

**Leave-one-out sensitivity** (recomputed CI dropping each pair in turn, n=4 remaining):

At the registered per-look stringency (t₀.₉₈₇₅,₃ = 4.1765, very conservative for n=4):
| dropped seed | mean(d) | CI | excludes 0? |
|---|---|---|---|
| 1 | 0.005497 | [0.000123, 0.010870] | yes |
| 2 | 0.005661 | [0.000937, 0.010384] | yes |
| 3 | 0.004555 | [−0.001470, 0.010580] | **no** |
| 4 | 0.004575 | [−0.001482, 0.010632] | **no** |
| 5 | 0.004164 | [−0.000758, 0.009085] | **no** |

At a conventional 95% two-sided threshold (t₀.₉₇₅,₃ = 3.1824, less conservative):
| dropped seed | CI | excludes 0? |
|---|---|---|
| 1 | [0.001402, 0.009591] | yes |
| 2 | [0.002062, 0.009260] | yes |
| 3 | [−0.000036, 0.009146] | **no** (barely) |
| 4 | [−0.000040, 0.009191] | **no** (barely) |
| 5 | [0.000414, 0.007913] | yes |

No leave-one-out subset flips the sign or produces a CI entirely below 0. But at conventional 95%, dropping seed 3 or seed 4 alone is enough to make the CI straddle 0 (marginally). The full-sample "excludes 0" result is not strongly robust to single-pair removal — it depends on having all 5 pairs.

**Per-pair spread:** the 5 differences cluster into two groups — {seed 1, seed 2} ≈ 0.0018–0.0025, and {seed 3, seed 4, seed 5} ≈ 0.0062–0.0078 — roughly a 3–4× gap between groups, driving both the sd(d) and the leave-one-out fragility above. With n=5 this pattern cannot be attributed to anything beyond noise, but it should not be waved away either.

**Anomaly scan:** no non-finite values anywhere; no truncated series (all 10 runs reach step 20000 with a `run_end` record); eval cadence is exactly every 500 steps (40 evals, 500→20000) for all 10 runs, no irregularities; no duplicate `data_order_digest` across different seeds (seeds were genuinely varied); kappa telemetry is absent from every final eval record in all 10 files uniformly (not an inconsistency — it's just never populated, so no differential telemetry gap between arms).

---

## 6. VERDICT

**Registered criterion (verbatim):** *"CI entirely above 0 -> H1 supported. CI entirely below +epsilon -> no practically relevant improvement."*

My recomputed Stage-1 CI is **[0.00081, 0.00897]** (per-look 97.5%, t₀.₉₈₇₅,₄).

- This CI is **entirely above 0** → by the letter of the registered rule, **H1 is SUPPORTED**: arm A shows a statistically confident (at the registered per-look alpha) lower final nats/byte than arm B.
- The **same CI is also entirely below epsilon = 0.01** → the registered practical-relevance clause fires simultaneously: **"no practically relevant improvement."**

These two clauses answer different questions (direction/confidence vs. magnitude) and are not mutually exclusive as written; together they describe one coherent, determinate joint outcome: **a real but small effect, confidently below the pre-declared bar for practical relevance.**

### VERDICT: **SUPPORTED** (statistically, per the literal registered CI-above-0 criterion) — **conditional on the packet's asserted registration/timing metadata**, and **with the registered epsilon clause simultaneously establishing no practically relevant improvement.** This is not "arm A is meaningfully better"; it is "arm A is confidently, marginally lower, by less than the study's own pre-declared threshold for mattering."

---

### MAJOR concerns

1. **Epsilon was fixed post-run, ~9 minutes after the last confirmatory run ended, and it directly governs the practical-relevance verdict that the observed effect fails to clear.** The amendment's "no access to any confirmatory outcome of this comparison" is a bare assertion I cannot verify from anything in this packet — there is no cryptographic or process evidence (no diff of an actual pre-amendment prereg, no access log) that the amender didn't see or infer the outcome before committing epsilon. This is exactly the kind of post-hoc-threshold risk optional-stopping audits exist to catch, and here it lands squarely on the boundary: the observed CI upper bound (0.00897) is below epsilon (0.01) by a margin (0.0010) smaller than the SE (0.0012) — i.e., epsilon chosen even slightly lower would have flipped the practical-relevance reading, and it was chosen after the numbers already existed.
2. **All temporal-ordering evidence for the optional-stopping story is operator-asserted, not independently verifiable from this packet.** The run-record JSONL contains no internal timestamps; the manifest explicitly disclaims its own timestamp block as "asserted metadata, not raw record content." Likewise, `preregistration.md` as shipped is the post-amendment text — there is no artifact in this packet that lets me confirm what the pre-run prereg said, only a cited commit hash. My "no confirmatory run predates the prereg" and "epsilon postdates all runs" findings in §4 are therefore conditional on trusting these assertions, which is exactly the trust this review is supposed to withhold by default.

### MINOR concerns

3. Leave-one-out sensitivity: the "CI excludes 0" result depends on all 5 pairs; at a conventional 95% threshold, dropping either seed 3 or seed 4 alone makes the CI straddle 0 (marginally). No subset reverses direction, but the margin is thin.
4. The 5 per-pair differences show a visible two-cluster pattern ({1,2} small, {3,4,5} larger, ~3–4×gap) that is unexplained and, at n=5, statistically inconclusive as to cause.
5. The `data_order_digest` (Appendix A) attests only the first 32 of the run's batches; identity of data order for the remaining ~19,968 steps in a pair is by design/inference, not attestation.
6. `B-seed2` end timestamp lacks the microsecond precision present on all other timestamp fields in the manifest's operator-supplied block — a small internal inconsistency in that metadata's provenance.
7. Sign test (5/5 positive, exact one-sided p = 0.03125) is directionally corroborating but mathematically cannot reach the registered per-look alpha at n=5; it should not be read as an independent confirmation of significance.

### Specific additional evidence that would change my verdict

- **A verifiable, pre-amendment copy of the preregistration** (or a diff against commit `0a917c3812a6` from a source I can check independently of this packet) confirming epsilon was not present, and confirming no other silent changes accompanied "amendment 1."
- **Independently generated timestamps** for run start/end (e.g., signed tracking-server logs, or timestamps embedded in the JSONL records themselves) rather than a hand-asserted manifest block, to actually close the optional-stopping audit rather than take it on faith.
- **Evidence bearing on the epsilon amender's information access** at commit time (e.g., that the analysis/aggregation script did not exist or was not run before 17:19:15Z) — this is the single biggest lever on whether the "no practically relevant improvement" reading should be trusted.
- **A larger or replicated seed set** (e.g., actually running Stage 2, or an independent replication) to resolve whether the seed-3/4/5 vs seed-1/2 clustering and the leave-one-out fragility reflect real heterogeneity or are sampling noise at n=5.
- Any evidence that the {1,2} vs {3,4,5} split correlates with an unregistered covariate (e.g., hardware, launch batch, wall-clock order) would elevate the clustering from a MINOR to a MAJOR concern.
