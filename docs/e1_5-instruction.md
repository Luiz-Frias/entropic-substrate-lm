# E-1.5 Instruction — from Fisher initialization to Fisher representation

Operator pivot, 2026-08-18. E1 as run tests a narrower hypothesis than the
thesis that motivated it: it warm-starts a static `nn.Embedding` table from
Fisher–Rao coordinates and lets an unconstrained transformer train it freely.
The original thesis is **state construction**: the symbol's representation is
its current predictive distribution mapped onto the statistical manifold —
"the encoding is a field" — with the byte as the observation entering the
system, never the representation used by it. The measured E1 outcome (real,
confidently-signed, ~0.005 nats/byte terminal gap below ε; advantage peaking
+0.0174 at step 500 and decaying ~3.5×; ~10% fewer steps to the e0-byte
terminal loss) is a **warm-start result** — preliminary evidence that even a
static, order-1 snapshot of the corpus field points optimization in a useful
direction. E1.5 asks what happens when the field stays live, causal,
contextual, and geometrically preserved.

The decomposition this series completes (E1 result does not change; no rung
is edited retroactively):

| Box | Question | Rung |
|---|---|---|
| Initialization geometry | where does a symbol begin? | E1 (done: stage 1 + extension) |
| **State construction** | fixed point, or context-conditioned distribution? | **E1.5 (this doc)** |
| State dynamics | does the model stay in the induced geometry? | E2 |
| Relational kernel | what geometry compares states? | E3 |
| Coarse-graining | how are variable-scale symbols built? | E4 |

## E1.5-D — diagnostics on existing artifacts (exploratory; runs first)

No new training. Computed from the 18 paired-run final checkpoints (MLflow
artifacts) plus deterministically reconstructed init tables — fisher: the
saved table (`data/fisher_init_d768.pt`) or the exact corpus rebuild;
random_sphere: **fixed seed 0** (CORRECTED 2026-08-18: `backbone.py:109`
calls `random_sphere_init` with its default seed, so every e1-random run
shares ONE init table regardless of `train.seed` — reconstruction follows
the code, and cross-seed drift on the random arm shares a single origin).
No decision rules — these select the next rung's design, they do not gate it.

1. **Angular drift from init**: mean over the 256 rows of
   `arccos(cos(e_i^(0), e_i^(final)))`, per arm per seed.
2. **Pairwise-geometry retention**: correlation of the init Fisher–Rao
   distance matrix with the final embedding cosine-distance matrix.
3. **Cross-arm geometry agreement**: compare 256×256 Gram matrices between
   paired arms (rotation-invariant; never compare raw coordinates — latent
   spaces are not identifiable).
4. **Orthogonal Procrustes alignment** before residual comparison.
5. **Functional equivalence**: KL / Jensen–Shannon between the paired arms'
   next-byte distributions on shared held-out contexts.

Outcome map: same Gram + same predictions → shorter path to a common
attractor. Different embeddings + same predictions → non-identifiability
(coordinates changed, function did not). Different Gram + different
predictions + persistent gap → different basin. Fisher geometry erased while
performance better → the benefit was optimization conditioning, not retained
semantics.

**Constraint (named up front):** only final/best checkpoints exist for the
E1 runs, so E1.5-D is an ENDPOINT diagnostic. Trajectory versions of 1–5
require a rerun with denser `save_interval` — decide after the endpoint
readout, not before.

## E1.5a — frozen Fisher table

Freeze the Fisher geometry while the backbone trains. Because embeddings are
tied (`tie_embeddings: true`), the E1 init seeded BOTH the input code and the
256-way output classifier — so three variants attribute the effect:

- freeze input table only;
- freeze output classifier only (untie for this variant);
- freeze both.

Readout: frozen-Fisher performs well → the geometry carries durable
predictive structure. Frozen fails where trainable-Fisher won early → good
launch ramp, insufficient substrate. Frozen Fisher and frozen random both
fail → higher-order contextual construction dominates static symbol geometry.

## E1.5b — geometry-preserving regularization

Table stays trainable; penalize divergence from the init's relational
geometry: `L = L_CE + λ‖G(E_t) − G(E_fisher)‖_F²` with `G` the normalized
Gram matrix (stronger than row normalization, which preserves radius, not
pairwise Fisher relations). λ sweep distinguishes helpful inductive bias /
harmful overconstraint / transient warm-start information.

## E1.5c — online contextual Fisher state (the original thesis)

At each position, a causal density estimator produces
`p_φ(b_{t+1} | x_{≤t})`; the state entering the backbone is
`z_t = Q·√p_φ(· | x_{≤t})` (same fixed semi-orthogonal lift as E1). The byte
is the observation; `z_t` is the representation.

**Estimator ladder (ascending cost; pure-torch first, per the E4b toolchain
quarantine — `mamba_ssm` builds only in the container lane):**

- **v0**: order-k sliding-window count model (no learned params) — isolates
  "contextual at all" from "learned estimator".
- **v1** (the first registered rung): tiny causal transformer `p_φ`
  (~2 layers, d≈128, 256-way softmax) — runs on the current image.
- **v2**: Mamba2/SSM `p_φ` — same interface, built in the E4 container lane.

**Registration-time pins:**

1. **Anti-leakage**: `p_φ` consumes `x_{≤t}` only, never the target byte —
   pinned by a unit test the way the data-order digest is.
2. **Param parity**: `p_φ` adds parameters; report matched-total-param AND
   matched-backbone variants with param/FLOP counts (E0-bpe precedent — the
   asymmetry is part of the measurement).
3. **Combination rule**: `z_t` alone vs `z_t ⊕ E[b_t]` is a variable; v1
   registers exactly one (default: `z_t` alone — the stronger form of the
   thesis) and the other is a declared follow-up arm.
4. Primary endpoint declared at registration: E1's primary stays final
   nats/byte; if E1.5c registers **steps-to-target** as primary (the
   engineering claim the E1 evidence points at), that is declared ex-ante
   with the target level named (e0-byte terminal, 0.7738).

## Multi-scale field (E1.5c″ — horizon, feeds E4)

`z_t` as a product of scales:
`√p₁(·|b_t) × √p₂(·|b_{t−1:t}) × √p₄(·|b_{t−3:t}) × …` — each factor a
different conditional scale. Reconnects the character → word → sentence zoom:
E1 supplies one static order-1 field; E1.5c a live local field; E4 learned
coarse-graining over field states. Not registered until E1.5c v1 concludes.

## Where E2/E3 stand under the pivot

Unchanged and still one-variable: E2 tests whether respecting spherical
geometry downstream preserves/magnifies the substrate advantage; E3 tests
the relational kernel. Both currently carry the STATIC table; once an E1.5
winner exists, the ladder's carry-forward rule applies to it like any other
settled component.

## Process

Per `docs/rung-lifecycle-protocol.md`: E1.5-D is exploratory and runs on
captured artifacts immediately. E1.5a/b/c each get their own preregistration
committed AFTER the E1 extension verdict files and BEFORE their first
confirmatory run, drawing stage sizes from the measured σ_seed. Clause
precedence is declared in every new registration (direction primary,
practical-relevance rider mandatory) per the stage-1 cloud review's MAJOR-1.
The CLAUDE.md ladder description gains the E1.5 box when this doc merges —
operator ratification required (north-star amendment).
