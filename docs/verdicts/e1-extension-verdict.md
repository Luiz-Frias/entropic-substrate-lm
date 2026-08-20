# E1 independent analysis — verdict artifact (fresh session, packet-only)

## 0. Header — what this is

**What this is.** An independent, from-scratch confirmatory analysis of rung **e1** (embedding-initialisation comparison: `e1-fisher` = √p Fisher–Rao init vs `e1-random` = uniform-random unit-vector init), performed in an isolated sandbox with **no prior context** and no access to any repository narrative, review artifact, verdict file, or tracking server. Every number below labelled *(measured)* was computed in this session from the packet's raw JSONL telemetry; every number labelled *(quoted)* is transcribed from a registration document in `docs/`. No review artifact (`docs/reviews/*`) was present in the packet, and none was read.

**Data analysed** (all files in `data/`; md5 computed here, *measured* — the packet contained **no** `manifest.txt`, so these are my own content hashes, not a cross-check against a filed capture manifest):

| file | md5 (measured) | run | seed | embedding_init |
|---|---|---|---|---|
| `e0-byte-4620594a-e0-byte.jsonl` | `ecfc0ee3b2489331b95ab9eefd610b71` | e0-byte | 42 | default |
| `e0-bpe-aa909040-e0-bpe.jsonl` | `94a7e7355ffdae3b0e8d4d1e6d43da59` | e0-bpe | 42 | default |
| `e1-fisher-bd0fb0d3-e1-fisher-s1.jsonl` | `f9e3498ab923b27ae233aa9e1b85c2bc` | e1-fisher | 1 | fisher_sqrt |
| `e1-fisher-56d129a1-e1-fisher-s2.jsonl` | `e57e834fc356ef35b9618c15ae151af8` | e1-fisher | 2 | fisher_sqrt |
| `e1-fisher-7d84caf7-e1-fisher-s3.jsonl` | `ba7d61140cc2b7cc281aa881d2f67e35` | e1-fisher | 3 | fisher_sqrt |
| `e1-fisher-6015d8d1-e1-fisher-s4.jsonl` | `8269945108deacfaa8f0adcce83c61bf` | e1-fisher | 4 | fisher_sqrt |
| `e1-fisher-1fca0b15-e1-fisher-s5.jsonl` | `8c8e8cf48db0e939bd39ca9241d64831` | e1-fisher | 5 | fisher_sqrt |
| `e1-fisher-52658db6-e1-fisher-s6.jsonl` | `c5a123f4f71c5d1a6a3dd5aa21b76d9a` | e1-fisher | 6 | fisher_sqrt |
| `e1-fisher-44c9e748-e1-fisher-s7.jsonl` | `5760439fc8ff478c3244f515e800e46c` | e1-fisher | 7 | fisher_sqrt |
| `e1-fisher-798de418-e1-fisher-s8.jsonl` | `a498901024a4de139979a28eba753942` | e1-fisher | 8 | fisher_sqrt |
| `e1-fisher-6a92ca08-e1-fisher-s9.jsonl` | `3e68debaa9329164681b8343aba077a1` | e1-fisher | 9 | fisher_sqrt |
| `e1-random-7bc2a00f-e1-random-s1.jsonl` | `534ee3be5293f89b72f2fbd521e5a4a9` | e1-random | 1 | random_sphere |
| `e1-random-d2cbbc78-e1-random-s2.jsonl` | `f8df7e6ae83b47ada84ad5146ede1aa3` | e1-random | 2 | random_sphere |
| `e1-random-2cc15d7b-e1-random-s3.jsonl` | `5af2c2a8a21fb707d83ad406ec4ed44c` | e1-random | 3 | random_sphere |
| `e1-random-3629cca4-e1-random-s4.jsonl` | `f34cad979e70bfc08decf4bab5456411` | e1-random | 4 | random_sphere |
| `e1-random-5ec7c790-e1-random-s5.jsonl` | `6cf44777014435a450b7d8d4e5a8368c` | e1-random | 5 | random_sphere |
| `e1-random-0030b0a0-e1-random-s6.jsonl` | `e66189d576336e89018bedcbc88c7768` | e1-random | 6 | random_sphere |
| `e1-random-b2febf05-e1-random-s7.jsonl` | `ea7a1f32efd427359c4271e75a732028` | e1-random | 7 | random_sphere |
| `e1-random-0a1c00bf-e1-random-s8.jsonl` | `7874649781308b0b6bea9fb08a68e8e0` | e1-random | 8 | random_sphere |
| `e1-random-58b39b28-e1-random-s9.jsonl` | `55538a2b2732a494464b9f6770028cba` | e1-random | 9 | random_sphere |

Each e1 file *(measured)*: 1 `run_start`, 1000 `train`, 40 `eval` (steps 500…20000, `eval_interval` 500, `max_steps` 20000), 1 `run_end`; `params` = 85,150,464; `tokenizer` = `byte`; `train.resume` = `null` (all fresh runs).

**Registrations governing this analysis** (both read in full, both in `docs/`):
- `docs/e1-preregistration.md` — stage-1 binding design, incl. **Amendment 1** (ε = 0.01 nats/byte) and **Amendment 2** (ε ratified + information-access disclosure).
- `docs/e1-extension-preregistration.md` — seeds 6–9, **self-declared outcome-aware**, with its own registered 3σ rule and its own declared clause precedence.
- `docs/rung-lifecycle-protocol.md` — process context only; supplies no estimator not already in the two preregistrations.

**Endpoint reader.** The registrations specify the primary metric is read "exclusively via `entropic_substrate.telemetry.final_eval_metric(path, "nats_per_byte")`". That module is **not** in the packet, so I re-implemented its documented contract *(measured, and flagged as a deviation of instrument, not of definition)*: **the `nats_per_byte` field of the last `eval` record in the file**. In every one of the 20 files the last `eval` record is at step 20000, i.e. the terminal evaluation of a completed 20000-step run, so the reading is unambiguous.

---

## 1. Validity findings

### 1.1 Pairing by seed
All 18 e1 runs pair exactly: seeds {1,…,9} × {`e1-fisher`, `e1-random`}, seed taken from `run_start.config.train.seed` and cross-checked against `run_start.run` and the filename *(measured)*. **No unpaired run. No seed appears twice within an arm.**

### 1.2 `data_order_digest` (the registered pairing evidence)
Every run emits `data_order_digest` **exactly once** *(measured)*, on the `train` record at **step 40**, together with `data_order_batches` = **32** — consistent with Appendix A (digest over the first 32 training batches, emitted once at/after the window, fresh runs only; `resume` = `null` everywhere, so no run is a non-attesting resume).

| seed | `data_order_digest` (identical in both arms, measured) | within-pair match |
|---|---|---|
| 1 | `3f05412324b70543fba0e6537d130ecf240b1339355ea4a366d0db4f18626e27` | ✅ |
| 2 | `8a39105f5aa9f1ed382a5103765f3dcd106be70404044231f9805d10a6d0f91e` | ✅ |
| 3 | `4f26fe9084deda26957f274ac7147c84df6f91d8f9e7daf0d40d10c208c67a26` | ✅ |
| 4 | `6e4e11046c3f652c28e813d31aad31b10dd76d1d84623439ae9b38bb11c8cb5b` | ✅ |
| 5 | `cc450a490e8d99a4fcee2a41141035d2694a89f495d1b524ee088b00fe7b007b` | ✅ |
| 6 | `bb8387791f7e1b0a262c3c6d895bf7c2160b24eab8b6ce15d23634d00bae5162` | ✅ |
| 7 | `13d3221ee13657ac49622b5975dba8a2387e1d31b42932f92df4451824940a15` | ✅ |
| 8 | `5dfb9709713bf6d999fbb46645fa64d40367b03be6e192cb1ee9514a05335553` | ✅ |
| 9 | `c4bdbf9702178825fa7ef1a7b0f57b27eed6c12cd64edb944101c8107ec9ed76` | ✅ |

All 9 digests are distinct across seeds *(measured)* — i.e. the seeds really do induce different data orders, and the within-pair identity is not a degenerate constant. **Zero digest mismatches ⇒ zero pairs invalidated on the registered pairing criterion.**

### 1.3 Registered exclusion rule ("only this one": non-finite final loss)
All 18 final eval `loss`/`nats_per_byte` values are finite and in [0.763245, 0.778289] *(measured)*; no `NaN`/`Inf` anywhere in any eval or train record of any e1 file *(measured)*. **No run and no pair is excluded.** I applied **no** exclusion beyond the registered one, and I exercised no discretionary exclusion of any kind.

### 1.4 "Identical except `embedding_init`" (design claim, verified)
Flattened `run_start.config` compared within each pair *(measured)*: the **only** differing keys are `name` (`e1-fisher`/`e1-random`), `model.embedding_init` (`fisher_sqrt`/`random_sphere`) and `model.fisher_init_path` (`data/fisher_init_d768.pt` / `null`) — i.e. the manipulated variable and its own path argument. Compared **across** seeds, all 18 configs are byte-identical except `train.seed` and those same three keys *(measured)*. All 18 report `params` = 85,150,464 and `train_tokens` = 95,000,000 / `val_tokens` = 5,000,000 *(measured)*.

### 1.5 Pilot
The declared pilot (`e1-fisher`, run id `c6025e…`, default config seed) is *(quoted)* "never part of the confirmatory set". **No file in `data/` carries that run id or a default/42 seed in an e1 arm** *(measured)* — the confirmatory set in the packet is exactly seeds 1–9 in both arms. The two e0 files (seed 42) are baselines, not e1 arms, and enter only the exploratory convergence target (§4a).

**Validity verdict: 9 of 9 pairs valid; 0 unpaired; 0 invalid; 0 excluded.** Stage 1 analyses seeds {1,2,3,4,5} (n = 5, df = 4); the extension analyses {6,7,8,9} and the cumulative set {1,…,9} (n = 9, df = 8) — exactly the registered seed sets.

---

## 2. Primary endpoint — stage 1, `docs/e1-preregistration.md` as written

**Direction convention** *(quoted)*: `d_s = y_random,s − y_fisher,s`; H1 predicts `mean(d) > 0`.
**Estimator** *(quoted)*: `CI = mean(d) ± t_{0.9875, n−1} · sd(d)/√n`, total α = 5% split Bonferroni over two looks (2.5% per look), with `t_{0.9875,4} ≈ 3.50`.
**Stage-1 seed set** *(quoted)*: {1,2,3,4,5}, df = 4, "analyzed once".

### 2.1 Per-seed values (measured)

| seed | y_random (nats/byte) | y_fisher (nats/byte) | d = y_random − y_fisher |
|---|---|---|---|
| 1 | 0.772992 | 0.770528 | +0.00246400 |
| 2 | 0.769449 | 0.767641 | +0.00180800 |
| 3 | 0.772037 | 0.765805 | +0.00623200 |
| 4 | 0.772891 | 0.766741 | +0.00615000 |
| 5 | 0.773662 | 0.765865 | +0.00779700 |

All five paired differences are positive (5/5) *(measured)*.

### 2.2 Registered statistics (all measured)

| quantity | value |
|---|---|
| n | 5 |
| mean(d) | **0.00489020000000000** |
| sd(d) (sample, ddof = 1) | **0.00260880302054407** |
| se = sd/√5 | **0.00116669217876866** |
| t = mean/se (df = 4) | **4.19150834212428** |
| registered constant t_{0.9875,4} *(quoted)* | 3.50 |
| CI half-width = 3.50·se | **0.00408342262569032** |
| **registered CI** | **[0.000806777374309681, 0.00897362262569032]** |

Cross-checks *(measured, exploratory only — not the registered interval)*: the exact quantile is t_{0.9875,4} = 3.49540593251698, giving CI = [0.000812137236910860, 0.00896826276308914] — the registration's rounded 3.50 is very slightly the more conservative choice, and the conclusion is identical. One-sided paired-t p = 0.00689487530717408; dz = mean/sd = 1.87449951624947; exact sign test with 5/5 positive gives one-sided p = 0.03125.

### 2.3 Applying the registered verdict rule verbatim

The rule *(quoted)*: "CI entirely above 0 → H1 supported. CI entirely below +ε → no practically relevant improvement (ε reported alongside). Otherwise → inconclusive (stage 2 if not yet run; else reported as inconclusive)."

- **Clause A (direction).** Lower bound 0.000806777374309681 > 0 ⇒ CI entirely above 0 ⇒ **clause A fires: H1 supported.**
- **Clause B (practical relevance), ε = 0.01 nats/byte** *(quoted, Amendment 1, ratified Amendment 2)*. Upper bound 0.00897362262569032 < 0.01 ⇒ CI entirely below +ε ⇒ **clause B fires: no practically relevant improvement.** Margin to ε *(measured)*: 0.01 − 0.00897362262569032 = 0.00102637737430968.
- **Clause C (inconclusive / stage-2 trigger).** Not reached: the CI does not straddle 0, so the registered stage-2 trigger *(quoted: "run ONLY if the stage-1 CI straddles 0")* is **not met**.

### 2.4 Stage-1 verdict label, in the prereg's own vocabulary

> **STAGE-1 VERDICT (registered vocabulary): `SUPPORTED`** — direction clause fires, CI entirely above 0.
> **MANDATORY ε RIDER (registered vocabulary): `no practically relevant improvement` at ε = 0.01 nats/byte** — the same CI lies entirely below +ε.

Both clauses fire simultaneously; see §5 for the precedence application. The registered stage-2 look was **not** triggered, so the Bonferroni α-split was spent on a single look — the filed stage-1 interval is therefore *conservative* relative to a one-look 5% design *(measured observation, no registered consequence)*. Per the prereg, stage 1 is "analyzed once": this document re-derives that single registered analysis; it does not add a look.

---

## 3. Extension + cumulative (n = 9) — `docs/e1-extension-preregistration.md` as written

### 3.0 Registered status of these seeds (quoted)
The extension document states in its own text that it "is **NOT** the registered stage 2 of `docs/e1-preregistration.md`: that trigger (stage-1 CI straddling 0) was NOT met"; that the original registration's "confirmatory phase is CLOSED and its filed verdict stands untouched"; and that the extension is "**outcome-aware by construction**". My §2.3 independently confirms the trigger was not met *(measured)*. Consequently: **nothing in §3 can alter §2**, and the extension's label is *(quoted)* "reported *alongside* — never replacing — the stage-1 filed verdict."

### 3.1 Extension seeds {6,7,8,9} (measured)

| seed | y_random | y_fisher | d = y_random − y_fisher |
|---|---|---|---|
| 6 | 0.766413 | 0.763245 | +0.00316800 |
| 7 | 0.776814 | 0.773541 | +0.00327300 |
| 8 | 0.778289 | 0.768590 | +0.00969900 |
| 9 | 0.773521 | 0.772754 | +0.00076700 |

Extension-only statistics *(measured, descriptive — the registered primary analysis is cumulative, not extension-only)*: n = 4; mean = **0.00422675000000000**; sd = **0.00382735682292625**; se = **0.00191367841146312**; t = **2.20870443784146**. All 4 differences positive.

### 3.2 Cumulative paired analysis over {1,…,9} (the registered primary analysis)

Same reader, same estimator family, same sign convention `d_s = y_random,s − y_fisher,s` *(quoted)*.

| quantity | value (measured) |
|---|---|
| n | 9 |
| mean(d) | **0.00459533333333333** |
| sd(d) (ddof = 1) | **0.00300307217695479** |
| se = sd/√9 | **0.00100102405898493** |
| **test statistic t = mean/se (df = 8)** | **4.59063225512597** |
| registered 3σ threshold t_{0.99865,8} *(quoted)* | **4.277** |
| one-sided 99.865% lower confidence bound = mean − 4.277·se | **0.000313953433054792** |
| all 9 differences positive? | yes (9/9) |

Threshold comparison *(measured)*: 4.59063225512597 **>** 4.277 ⇒ the registered inequality is satisfied; equivalently the one-sided 99.865% LCB, 0.000313953433054792, **exceeds 0**. Margin *(measured)*: t − threshold = **0.313632255125968** (7.33% above the bar).

Cross-check on the quoted constant *(measured, exploratory)*: the exact quantile is t_{0.99865,8} = 4.27657679957102, so the registration's 4.277 is a faithful (very slightly stricter) rounding; the decision is unchanged either way. Exploratory adjuncts: one-sided paired-t p = 0.000888547234116854; dz = 1.53021075170866; exact sign test 9/9 one-sided p = 0.001953125.

### 3.3 Practical-relevance rider at n = 9 (registered as mandatory and co-equal)

The extension registers *(quoted)*: "Practical relevance is read against the ratified ε = 0.01 nats/byte …, same clause wording", and the stage-1 clause wording is *(quoted)* "CI entirely below +ε". The extension's own registered interval is **one-sided** (a lower bound only), and a one-sided interval [LCB, +∞) can never be "entirely below +ε" — the clause is vacuous on it. I therefore report every non-vacuous reading of the rider that uses only registered constants, all *measured*:

| rider reading | interval | entirely below ε = 0.01? |
|---|---|---|
| point estimate mean(d) | 0.00459533333333333 | yes |
| two-sided CI with the original prereg's cumulative constant t_{0.9875,8} = 2.75 *(quoted)* | [0.00184251717112478, 0.00734814949554189] | **yes** (0.00734815 < 0.01) |
| two-sided interval using the extension's own 3σ constant 4.277 *(quoted)* | [0.000313953433054792, 0.00887671323361188] | **yes** (0.00887671 < 0.01) |
| one-sided 3σ interval as literally registered | [0.000313953433054792, +∞) | clause vacuous / undefined |

**All non-vacuous readings agree.** Largest observed upper bound, 0.00887671323361188, is below ε with margin 0.00112328676638812 *(measured)*.

### 3.4 Extension verdict label, in its own vocabulary

> **EXTENSION / CUMULATIVE VERDICT (n = 9, registered vocabulary): `STRONGLY_SUPPORTED`** — t = 4.59063225512597 > t_{0.99865,8} = 4.277; one-sided 99.865% LCB = 0.000313953433054792 > 0.
> **MANDATORY CO-EQUAL RIDER: `no practically relevant improvement` at the ratified ε = 0.01 nats/byte** — every non-vacuous reading of the cumulative interval lies entirely below +ε.
> Reported **alongside**, never replacing, the stage-1 `SUPPORTED` + ε-rider verdict of §2.4.

Per the registration's own words *(quoted)*, "Quoting the direction label without its rider is a misquotation of this registration."

**Which of the two projected worlds obtained** *(the extension's registered power context, quoted: stage-1-like distribution ⇒ t ≈ 5.62 passes; sd doubling ⇒ t ≈ 2.81 fails; minimum homogeneous detectable effect ≈ 0.0037 nats/byte)*. Measured outcome: cumulative sd = 0.00300307217695479, i.e. **1.15112×** the quoted stage-1 sd of 0.00261 — closer to the homogeneous world than to the sd-doubling world, but with a mean (0.00459533333333333) that fell **below** the quoted stage-1 mean of 0.00489, so the realised t = 4.59063 landed between the two projections and above the bar. The measured cumulative mean also exceeds the quoted minimum detectable effect 0.0037 *(0.00459533 > 0.0037)*.

---

## 4. Secondary / exploratory analyses

> **Every result in §4 is EXPLORATORY.** None of it can change, qualify, or reweight the registered verdicts in §2.4 and §3.4. Items (a) and the spread/overfit/LOO probes correspond to the extension's "**Predeclared probes (report, never decide)**" *(quoted)*; items (b) and (c) are not in any registration and are exploratory a fortiori.

### 4a. Convergence speed (extension predeclared probe 4 — "directional readout, no decision rule")

**Target, computed from the packet** *(measured)*: terminal (step-20000) eval `nats_per_byte` of `e0-byte-4620594a-e0-byte.jsonl` = **0.773782** nats/byte (= 1.116190 bits/byte as recorded in that record; the registrations *quote* this baseline rounded, as "0.7738"). For reference the e0-bpe terminal value is 0.965620 nats/byte *(measured)*; it is not used here. Criterion applied: **first eval record with `nats_per_byte` ≤ 0.773782**.

| seed | e1-fisher: first step (value at it) | e1-random: first step (value at it) | random/fisher step ratio |
|---|---|---|---|
| 1 | 16000 (0.773597) | 18000 (0.771887) | 1.125000 |
| 2 | 16000 (0.769376) | 16500 (0.772428) | 1.031250 |
| 3 | 15000 (0.772479) | 19500 (0.772882) | 1.300000 |
| 4 | 16000 (0.773481) | 17500 (0.773399) | 1.093750 |
| 5 | 16000 (0.770644) | 18500 (0.772517) | 1.156250 |
| 6 | 14500 (0.773236) | 16000 (0.772758) | 1.103448 |
| 7 | 15000 (0.770156) | **never reached** (final 0.776814) | censored |
| 8 | 16000 (0.771165) | **never reached** (final 0.778289) | censored |
| 9 | 17000 (0.773513) | 19000 (0.773668) | 1.117647 |

Summary *(measured)*:
- `e1-fisher`: reaches the target in **9/9** runs; steps sorted = 14500, 15000, 15000, 16000, 16000, 16000, 16000, 16000, 17000; **median = 16000** (n = 9); median over stage-1 seeds only = 16000.
- `e1-random`: reaches the target in **7/9** runs (seeds 7 and 8 are **right-censored at > 20000**); sorted = 16000, 16500, 17500, 18000, 18500, 19000, 19500, >20000, >20000; the **median = 18500** is still exactly determined despite the censoring (the 5th of 9 order statistics is uncensored); median over stage-1 seeds only = 18000.
- **Median-step ratio random/fisher = 18500/16000 = 1.15625** (n = 9); stage-1-seeds-only ratio = 18000/16000 = **1.125**; inverse (fisher/random) = 0.864864864864865. Median of the 7 uncensored per-seed paired ratios = **1.11764705882353**.
- Direction: the Fisher-init arm reaches the e0-byte terminal quality **earlier in 9/9 pairs**, by a median ≈ 2500 steps ≈ **13.5%** of the 18500-step random-arm median. Because 2 random runs never reach the target, **1.15625 is a lower bound** on the true median ratio.

### 4b. Gap trajectory and diminishing returns (exploratory)

Paired eval gap at each of the 40 common eval steps, averaged over all 9 valid pairs: `gap(step) = mean_s[ y_random,s(step) − y_fisher,s(step) ]` *(all measured)*. "Cumulative share" = running sum of `gap(step)` divided by the total over all 40 checkpoints, Σgap = **0.296273666666667** (mean per checkpoint 0.00740684166666666). This is a *shape* diagnostic, not an integral of anything registered.

| step | mean gap (nats/byte) | sd across 9 pairs | cumulative gap | cumulative share | pairs with gap > 0 |
|---|---|---|---|---|---|
| 500 | +0.0142646666666667 | 0.0205960 | 0.0142646666666667 | 0.0481469272215667 | 8/9 |
| 1000 | +0.00583311111111110 | 0.00976272 | 0.0200977777777778 | 0.0678351809119422 | 7/9 |
| 2000 | +0.00618588888888886 | 0.00573288 | 0.0326756666666667 | 0.110288798306971 | 8/9 |
| 4000 | +0.00704333333333334 | 0.00370286 | 0.0619977777777777 | 0.209258482116572 | 9/9 |
| 6000 | +0.00961766666666667 | 0.00414216 | 0.0979854444444444 | 0.330726134208500 | 9/9 |
| 8000 | +0.00776499999999999 | 0.00507073 | 0.129028333333333 | 0.435503886609340 | 9/9 |
| 10000 | +0.00735700000000001 | 0.00420566 | 0.161966777777778 | 0.546679627656562 | 8/9 |
| 12000 | +0.00663955555555555 | 0.00454688 | 0.190666111111111 | 0.643547275923199 | 9/9 |
| 14000 | +0.00646311111111112 | 0.00425704 | 0.219133555555555 | 0.739632239412285 | 9/9 |
| 16000 | +0.00700600000000001 | 0.00321489 | 0.249021222222222 | 0.840510819013802 | 9/9 |
| 18000 | +0.00622566666666667 | 0.00415775 | 0.274458777777778 | 0.926369126442032 | 9/9 |
| 20000 | +0.00459533333333333 | 0.00300307 | 0.296273666666667 | 1.00000000000000 | 9/9 |

Quarter-window view *(measured)*:

| window (eval steps) | mean gap over window | share of Σgap |
|---|---|---|
| 500–5000 (evals 1–10) | 0.00764914444444444 | 0.258178343370975 |
| 5500–10000 (evals 11–20) | 0.00854753333333333 | 0.288501284285587 |
| 10500–15000 (evals 21–30) | 0.00729046666666666 | 0.246072043752342 |
| 15500–20000 (evals 31–40) | 0.00614022222222222 | 0.207248328591096 |

Reading *(exploratory)*: the gap is largest at the **first** evaluation (0.0142646666666667 at step 500, with by far the largest cross-seed sd, 0.0205960 — an early-training artefact, and the only checkpoint where a pair is negative in 2 of 9 seeds), then settles into a broad plateau near 0.0065–0.0096 through mid-training, and **decays monotonically in the last quarter to its minimum at the terminal step (0.00459533333333333, the registered endpoint)**. The terminal gap is **32.2%** of the step-500 gap and **62.0%** of the peak-window (5500–10000) mean. The last quarter of training contributes only 20.7% of the summed gap versus 25.8%/28.9% for the first two quarters. In short: **the Fisher-init advantage is real at every checkpoint from step 3000 onward (9/9 pairs positive at 28 of 40 checkpoints) but is shrinking as training proceeds** — i.e. the registered endpoint measures the *smallest* gap seen in the run, and extrapolating this arm-difference to longer budgets is not supported by these data.

### 4c. Endpoint-noise sensitivity (exploratory; k = 1 is the registered reading)

Re-defining each run's endpoint as the mean of its **last k eval records** (k = 3 → steps 19000/19500/20000; k = 5 → 18000…20000), then re-running both registered computations unchanged *(all measured)*:

| k | stage-1 mean(d) | stage-1 sd(d) | stage-1 CI (t = 3.50) | CI above 0? | CI below ε? |
|---|---|---|---|---|---|
| **1 (registered)** | **0.00489020000000000** | **0.00260880302054407** | **[0.000806777374309681, 0.00897362262569032]** | **yes** | **yes** |
| 3 | 0.00496760000000003 | 0.00304081149987596 | [0.000207977145660014, 0.00972722285434004] | yes | yes |
| 5 | 0.00490132000000003 | 0.00305662760767481 | [0.000116941021135631, 0.00968569897886443] | yes | yes |

| k | cumulative mean(d) | cumulative sd(d) | t (df = 8) | > 4.277? | one-sided LCB | two-sided 2.75-CI upper < ε? |
|---|---|---|---|---|---|---|
| **1 (registered)** | **0.00459533333333333** | **0.00300307217695479** | **4.59063225512597** | **yes** | **0.000313953433054792** | **yes (0.00734814949554189)** |
| 3 | 0.00532592592592592 | 0.00299296581252411 | 5.33844312919262 | yes | 0.00105895433253737 | yes (0.00806947792073969) |
| 5 | 0.00560811111111110 | 0.00312772171893713 | 5.37910173768611 | yes | 0.00114902251381307 | yes (0.00847518935347013) |

Per-seed d under smoothing *(measured)*: k = 3 → s1 0.00178933333333320, s2 0.00188666666666681, s3 0.00685000000000002, s4 0.00861733333333337, s5 0.00569466666666674, s6 0.00303533333333328, s7 0.00782733333333319, s8 0.00937833333333327, s9 0.00285433333333340. k = 5 → s1 0.00144359999999999, s2 0.00173820000000013, s3 0.00710280000000008, s4 0.00774299999999994, s5 0.00647900000000001, s6 0.00232759999999987, s7 0.00869579999999992, s8 0.00983279999999998, s9 0.00511019999999995. **All 9 differences remain positive at every k.**

**Does any registered conclusion change? No.**
- Stage-1 `SUPPORTED` (CI above 0): holds at k = 1, 3, 5 — though the lower bound is visibly noise-sensitive, shrinking from 0.000806777 (k=1) to 0.000116941 (k=5), i.e. within 1.2×10⁻⁴ of zero.
- Stage-1 ε rider (CI below 0.01): holds at k = 1, 3, 5.
- Cumulative `STRONGLY_SUPPORTED` (t > 4.277): holds at k = 1, 3, 5, with the margin *widening* under smoothing (4.591 → 5.338 → 5.379).
- Cumulative ε rider: holds at k = 1, 3, 5 under the point estimate and under the two-sided 2.75 reading. **One caveat, reported for completeness:** at k = 5 the *symmetric* 4.277 interval's upper end, 0.0100671997084091, exceeds ε by 6.7×10⁻⁵ — the only sign flip anywhere in §4c. It occurs in a reading that the extension does not register (the registered 3σ interval is one-sided), at a k the registration does not use, and the registered readings are all unambiguous; it changes nothing, and is noted only so that no reader discovers it later and thinks it was suppressed.

### 4d. Other extension predeclared probes reportable from JSONL alone (report, never decide)

- **Probe 2 — spread, {1–5} vs {6–9}, "does the two-cluster pattern replicate?"** *(measured)*: stage-1 mean 0.00489020000000000, sd 0.00260880302054407, range [0.00180800, 0.00779700]; extension mean 0.00422675000000000, sd 0.00382735682292625, range [0.00076700, 0.00969900]. Extension sd is **1.46672×** stage-1 sd and its range is *wider* on both ends; the difference of means is small (0.000663450000000 in favour of stage 1; descriptive Welch-type t = 0.296013894819817). The stage-1 "{1,2} low vs {3,4,5} high" split does **not** reproduce as a clean two-cluster structure: the extension seeds interleave (0.000767, 0.003168, 0.003273, 0.009699), i.e. the extension both extends the low tail (s9 = 0.000767, the smallest of all nine) and the high tail (s8 = 0.009699, the largest of all nine). Read directionally, this looks like a **single heavy-ish-tailed seed distribution with a positive mean**, not two regimes — but n = 4 cannot settle that, and this probe decides nothing.
- **Probe 3 — train-vs-eval gap (overfit/saturation check)** *(measured; using the mean of each run's last 10 `train` `nats_per_byte` records vs its final eval value)*: `e1-fisher` mean (eval − train) = **0.0621515000000000**; `e1-random` = **0.0553165000000000**; positive in 18/18 runs, per-run range 0.0380002–0.102003. So both arms sit ≈0.04–0.10 nats/byte above their training loss at the end of ≈3.45 passes, and the **fisher arm carries the slightly larger train→eval gap** (its lower eval loss is *not* explained by less overfitting; if anything it fits the training stream harder). Directional only.
- **Probe 5 — leave-one-out at n = 9 against the 3σ bar** *(measured; k = 1 endpoints)*:

| dropped seed | n | mean(d) | sd(d) | t | > 4.277 (bar as quoted, df = 8) | > 4.52991 (df-correct 3σ constant at n = 8) |
|---|---|---|---|---|---|---|
| s1 | 8 | 0.00486174999999999 | 0.00309462980707824 | 4.44353813896652 | yes | no |
| s2 | 8 | 0.00494374999999999 | 0.00300967866106286 | 4.64602310501344 | yes | yes |
| s3 | 8 | 0.00439074999999999 | 0.00314265646274885 | 3.95172572795839 | **no** | no |
| s4 | 8 | 0.00440100000000000 | 0.00314934070197920 | 3.95254402554323 | **no** | no |
| s5 | 8 | 0.00419512500000001 | 0.00294267946108887 | 4.03224527122308 | **no** | no |
| s6 | 8 | 0.00477374999999999 | 0.00315901430150970 | 4.27418260825991 | **no** (by 0.0026) | no |
| s7 | 8 | 0.00476062500000000 | 0.00316634994625402 | 4.25255613223512 | **no** | no |
| s8 | 8 | 0.00395737499999999 | 0.00247398463660201 | 4.52433965320258 | yes | no |
| s9 | 8 | 0.00507387500000001 | 0.00281981222463468 | 5.08937636068365 | yes | yes |

**The 3σ pass is fragile to single-pair removal**: dropping any one of seeds 3, 4, 5, 6, or 7 puts the statistic below the registered 4.277 bar (and if one insists on the *correct* 3σ constant for n = 8, t_{0.99865,7} = 4.52991, only 2 of 9 leave-one-out subsets clear it). The registered analysis is the n = 9 one, which passes; this probe is registered as "report, never decide" and I do not let it decide. For symmetry, the stage-1 LOO at n = 4 with the quoted 3.50 constant *(measured)*: dropping s3 gives CI [−0.000494323881148008, 0.00960382388114800] and dropping s4 gives [−0.000500709668603666, 0.00965120966860367] — both straddle 0; dropping s1, s2, or s5 keeps the CI above 0 (s5: lower bound 3.93430280642391×10⁻⁵). **Both registered verdicts sit close to their thresholds and neither survives arbitrary single-pair deletion.**

---

## 5. Explicit clause-precedence application

1. **Stage 1 (`docs/e1-preregistration.md`).** Its verdict rule contains two clauses that, on the observed interval [0.000806777374309681, 0.00897362262569032], **both fire**: "entirely above 0" (⇒ H1 supported) and "entirely below +ε = 0.01" (⇒ no practically relevant improvement). **The stage-1 document declares no precedence between them.** I therefore do not invent one: I report both, and I label the stage-1 verdict with the direction clause's vocabulary (`SUPPORTED`) *and* attach the ε clause verbatim as a rider. The stage-1 text's own ordering (direction first, practical relevance second with "ε reported alongside") and Amendment 2's framing (*quoted*: "the direction verdict is ε-invariant … nothing about the SUPPORTED verdict depended on where ε fell", while "the practical-relevance reading DOES depend on ε") are consistent with that reporting shape — the direction reading is the verdict, the ε reading is a co-travelling qualifier — but I flag that this is a *reading of an under-specified rule*, not a rule.
2. **Extension (`docs/e1-extension-preregistration.md`, §"Clause precedence", declared 2026-08-18 before any extension analysis).** This is the **only** explicit precedence declaration in the packet *(quoted)*: "the two clauses answer DIFFERENT questions and are BOTH always reported — direction (STRONGLY_SUPPORTED / NOT_STRONGLY_SUPPORTED per the 3σ rule) is the primary verdict label; practical relevance vs ε is a mandatory co-equal rider that travels with every quotation of the verdict." I apply it exactly and literally to §3.4: primary label `STRONGLY_SUPPORTED`; mandatory rider `no practically relevant improvement at ε = 0.01`. Its scope is self-limited ("For this extension"), so I do **not** back-propagate it to §2 as if the stage-1 registration had said it; I merely note the two readings coincide in reporting shape.
3. **Cross-document precedence.** The extension is *self-declared* outcome-aware and *self-declared* not the registered stage 2; it registers that its label is reported "alongside — never replacing" the stage-1 verdict. Accordingly the two verdicts are **stacked, not merged**: §2.4 stands as the confirmatory stage-1 verdict of the original registration, and §3.4 stands as a separately-labelled, weaker-evidential-standing (because outcome-aware) cumulative reading under a stricter bar. Neither is quoted here without its ε rider.
4. **α accounting.** The original design split 5% across two looks; the second registered look never occurred (trigger unmet, §2.3), so stage 1's interval is conservative. The extension's 3σ bar (per-look ≈0.135% one-sided) is **not** a Bonferroni continuation of that split but an independently registered, deliberately stricter bar *(quoted: "deliberately stricter than stage 1's")*; I did not pool or re-split α across documents, and I performed no additional look beyond the two registered ones.
5. **Exclusions.** Only the single registered exclusion rule (non-finite final loss ⇒ drop the whole pair, always reported) was available, and it fired zero times (§1.3). No other exclusion, winsorisation, or outlier handling was applied anywhere in §2 or §3 — including to seed 9 (smallest d) and seed 8 (largest d), both of which are retained in full.

---

## 6. Limitations

1. **Instrument substitution.** `entropic_substrate.telemetry.final_eval_metric` was not in the packet; I re-implemented its documented contract (last `eval` record's `nats_per_byte`). All 20 files end with a step-20000 eval in a completed 20000-step run, so the reading is unambiguous — but this is a deviation of *instrument* and is disclosed as such.
2. **Integrity chain only partially checkable.** The packet contains no `manifest.txt`, no `capture-results` md5 list, no `config_hash`/`git_sha`/`data/enwik8.dvc` values inside the JSONLs, and no container-image digest (which the extension requires to be identical across all 8 extension pods). I verified what the telemetry actually carries — identical within-pair `data_order_digest`, byte-identical configs modulo the manipulation, `resume` = null, finite losses, params/token counts — and computed my own md5s. The registered digest→config_hash→git_sha→artifact chain is therefore **asserted by the documents, not verified here**.
3. **Both registered verdicts are threshold-adjacent and single-pair fragile.** Stage 1's CI lower bound is 8.07×10⁻⁴ (and 1.17×10⁻⁴ under k = 5 smoothing); the cumulative t exceeds the 3σ bar by 0.3136 (7.33%). Removing any one of seeds 3, 4, 5, 6, 7 drops the cumulative statistic below 4.277; removing seed 3 or 4 makes the stage-1 CI straddle 0 (§4d). The direction of the effect is nonetheless unanimous — 9/9 pairs positive at the endpoint, 9/9 pairs faster to the e0-byte target — which is the more robust fact in this packet than either t-statistic.
4. **The extension is outcome-aware by its own declaration**, motivated by stage-1 robustness findings. Its 3σ bar mitigates but does not erase selection concerns: the seed set, the bar, and the motivation were all chosen with stage-1 results in hand. Its `STRONGLY_SUPPORTED` label should not be read as if it were a blind pre-planned confirmation.
5. **ε provenance.** ε = 0.01 nats/byte was fixed *after* the stage-1 runs completed (Amendment 1) and Amendment 2 discloses that the operator had arm-level dashboard access before requesting it. The direction verdicts are ε-independent by construction; the practical-relevance rider is entirely ε-dependent, and the filed ε-sensitivity map that Amendment 2 relies on is **not in this packet**, so I could not check it. A reader who rejects a post-run ε should read §2/§3 as intervals only: [0.000806777374309681, 0.00897362262569032] at n = 5 and one-sided LCB 0.000313953433054792 (or [0.00184251717112478, 0.00734814949554189] two-sided) at n = 9.
6. **The rider's registered form is ill-posed at n = 9.** "CI entirely below +ε" cannot be evaluated on a one-sided bound; §3.3 reports all non-vacuous readings rather than picking one, and they agree — but the agreement is a fact about these data, not a repair of the registration.
7. **Practical-relevance conclusion is scale-fragile.** mean(d) ≈ 0.0046–0.0056 nats/byte is roughly half of ε, and the k = 5 symmetric-4.277 reading already touches ε from below (§4c). A modestly larger effect, a different smoothing choice, or a slightly smaller ε would flip the rider even while the direction verdict stands. The honest one-line summary: **a real, consistently-signed, sub-ε improvement.**
8. **Narrow evidential base.** One dataset (enwik8, byte-level), one scale (85.15M params), one 20000-step budget, one eval protocol (20 eval batches per eval; single terminal value per run), 9 seeds. The gap trajectory (§4b) shows the arm difference *shrinking* into the terminal checkpoint, so extrapolation to longer training, other tokenizers (the e0-bpe baseline at 0.965620 nats/byte is not comparable), or other corpora is unsupported. The convergence probe is right-censored for 2 of 9 random runs, making 1.15625 a lower bound on the median step ratio. The registration's queued FineWeb-Edu single-pass arm and the embedding-drift/substitutability probes cannot be addressed from JSONL captures at all.
9. **Scope of this document.** It re-derives registered quantities from raw telemetry; it is not a review, it read no review artifact, and it makes no claim about *why* the Fisher init helps — the packet contains no checkpoint or embedding-table data that could support a mechanism claim.

---

### Verdict summary (both labels always travel with their riders)

| stage | n | mean(d) (measured) | sd(d) (measured) | registered test (constant quoted) | result (measured) | **label (registered vocabulary)** |
|---|---|---|---|---|---|---|
| Stage 1, seeds {1–5} | 5 | 0.00489020000000000 | 0.00260880302054407 | CI = mean ± 3.50·se | [0.000806777374309681, 0.00897362262569032] | **`SUPPORTED`** + rider **`no practically relevant improvement`** (ε = 0.01) |
| Extension cumulative, seeds {1–9} | 9 | 0.00459533333333333 | 0.00300307217695479 | t = mean/se > 4.277 (3σ, one-sided) | t = 4.59063225512597; LCB = 0.000313953433054792 | **`STRONGLY_SUPPORTED`** + mandatory co-equal rider **`no practically relevant improvement`** (ε = 0.01) |

Pairs valid: 9/9 *(measured)*. Pairs excluded: 0 *(registered exclusion rule; no other exclusion applied)*. Registered stage-2 trigger: **not met** *(measured, per quoted rule)* — seeds 6–9 are the self-declared outcome-aware extension, reported alongside and never replacing the stage-1 verdict.
