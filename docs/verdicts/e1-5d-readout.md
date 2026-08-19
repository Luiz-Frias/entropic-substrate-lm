# E1.5-D endpoint readout — embedding geometry after 20k steps

Exploratory diagnostic per `docs/e1_5-instruction.md` §E1.5-D: **selects the
next rung's design, gates nothing.** Computed 2026-08-19 in the isolated
`e15d-readout` worktree from the 18 captured final checkpoints
(`models/captured.dvc`, stripped, step 20000, 9 seed-pairs) via
`entropic-substrate embedding-geometry`. Raw per-seed output:
`e1-5d-readout.json` (committed beside this file). All numbers below are
measured; mean ± sd over the 9 seed-pairs.

## Headline numbers

| quantity | e1-fisher (arm A) | e1-random (arm B) |
|---|---|---|
| angular drift from init, per-row mean (rad) | 1.1530 ± 0.0063 | 1.0088 ± 0.0014 |
| Gram retention vs own init, Pearson | **0.8401 ± 0.0026** | **0.0170 ± 0.0009** |
| Gram retention vs own init, Spearman | 0.7227 ± 0.0205 | 0.0261 (seed-1 value; same order) |

| cross-arm quantity (per seed pair) | value |
|---|---|
| final-Gram disagreement ‖G_A − G_B‖_F² | 417.96 ± 46.72 (per-entry RMS ≈ 0.080 cosine) |
| orthogonal-Procrustes residual (‖E‖_F = 16 for 256 unit rows) | 7.22 ± 1.46 (≈ 45% relative) |
| functional divergence, JS (nats, bound ln 2) | 0.0258 ± 0.0003 |
| functional divergence, KL A→B / B→A | 0.1266 ± 0.0018 / ≈ symmetric |

## Reading (against the §E1.5-D outcome map)

1. **The Fisher relational geometry substantially SURVIVES 20k steps of
   unconstrained training.** Individual rows drift enormously (mean 1.15 rad
   ≈ 66°), yet the pairwise structure of the Fisher init retains Pearson
   0.840 with the final cosine geometry — while the random init's structure
   is fully erased (0.017, i.e. nothing to keep and nothing kept). The
   fisher arm's rows move MORE than random's (1.153 vs 1.009 rad) while
   preserving relations better: the motion is coordinated, not destructive.
2. This is **not** the "Fisher geometry erased while performance better"
   cell (pure optimization conditioning). The induced geometry is durable —
   consistent with the field-not-lookup thesis: the model keeps the
   relational structure the √p substrate supplied.
3. The arms do NOT converge to one attractor: final Grams disagree
   (RMS ≈ 0.08/entry), embeddings are not rotation-related (Procrustes
   residual ≈ 45%), and predictions differ slightly but consistently
   (JS 0.0258 ± 0.0003 — tight across all 9 pairs). Combined with E1's
   persistent sub-ε endpoint gap: **weak "different basin" reading** —
   different geometry, slightly different function, small stable gap.

## Design implications (exploratory; operator judges)

- **E1.5a variant priority**: freeze-both is the sharpest first test — if
  84% of the relational structure survives anyway, freezing asks whether
  the remaining 16% adaptation was necessary. Input-only second;
  output-only third.
- **E1.5b is well-motivated**: unconstrained training keeps 84% naturally,
  so λ·‖G(E_t) − G(E_init)‖² regularization probes whether the "lost" 16%
  was noise (retention → 1 harmless or helpful) or necessary adaptation
  (performance cost).
- **E1.5c**: the durability of even a STATIC induced geometry strengthens
  the case that a live calibrated field is worth testing (four-arm
  contract, §E1.5c).

## Provenance and caveats

- **Seed-9 random checkpoint**: the pod never uploaded a `-final.pt`
  (zombie-restart casualty, run `58b39b28`); its `best.pt` was MEASURED to
  be step 20000 == max_steps and was stripped into the capture set with
  `models/captured/PROVENANCE.md` documenting the substitution. All 18
  other files are true final captures (plus 2 e0 baselines in the set).
- **Init-table reconstruction**: fisher init tables were rebuilt from the
  DVC-pinned enwik8 stream (the CLI's documented fallback, warned per
  invocation) — deterministic, but not bit-guaranteed identical to the
  tables the pods built if the corpus source ever changed. `data/enwik8`
  digest is DVC-pinned, so in practice the rebuild is exact.
- **FOR-920 measured live**: the fallback rebuild cost ~7.5 min × 9 fisher
  checkpoints in the JSON pass (~68 min of identical work); generating
  `data/fisher_init_d768.pt` once dropped the charts pass to ~1 min.
  That is the cache-issue evidence FOR-920 asked for.
- Endpoint-only: trajectory versions of these diagnostics require a rerun
  with denser `save_interval` (decision deferred per the instruction doc).
