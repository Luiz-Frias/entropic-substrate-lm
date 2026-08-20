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

**DESIGN CORRECTION, 2026-08-19 (operator-ratified; supersedes the two-arm
v1 sketch this section previously carried).** The FieldEncoder machinery
merged in PR #69 has the right object — causal encoder, √p map, exact lift,
FP32 island, state replacement — but as merged it does NOT yet test the
thesis: `p_φ = softmax(field logits)` is optimized only through the backbone
loss, so nothing forces it to remain an honest next-byte predictive
distribution. Unconstrained, it can degenerate into an arbitrary 256-category
latent code that merely passes through softmax and sqrt — an experiment on a
**simplex bottleneck**, not a Fisher predictive state. The repairs and the
full four-arm contract below are the direction now.

### Thesis

A symbol enters the model not as a fixed learned identity vector but as a
causal predictive distribution over continuations. A lightweight causal
estimator produces `p_φ,t = p_φ(b_{t+1} | b_{≤t})`; the backbone receives
`z_t = √p_φ,t · Q` with `Q ∈ R^{256×d}` semi-orthogonal. `‖√p‖₂ = 1`, and
`z_t·z_sᵀ` is the Bhattacharyya coefficient between the two predictions —
angular separation IS distinguishability of predictions, not arbitrary
ID distance. The discrete byte remains the **observation** consumed by
`p_φ`; it is no longer the **state** supplied to the backbone.

Four separable value claims, never compressed into one "faster" claim:
**statistical efficiency** (lower held-out nats/byte at the same source
bytes — the core scientific claim), **optimization efficiency** (fewer
bytes/steps to a registered target — the claim the E1 evidence points at),
**compute efficiency** (fewer FLOPs/GPU-seconds — NOT guaranteed; the field
adds compute), and **architectural efficiency** (a smaller backbone matches
a larger static one — a later width/depth experiment).

### Repair 1 — calibrated field (the critical one)

An explicit calibration objective keeps `p_φ` an honest predictor:
`L_total = L_backbone + λ_field · L_field` with
`L_field = −log p_φ(b_{t+1} | b_{≤t})`. This does not leak the target into
the input: `p_φ` still consumes `x_{≤t}` only; the target scores the
prediction during training exactly as the main LM loss does. Two registered
training regimes:

- **Joint calibrated field** — `p_φ` and backbone train jointly with the
  auxiliary field CE;
- **Pretrained/frozen field** — `p_φ` trains independently as a small causal
  LM, freezes, then the backbone trains over `√p_φ·Q`. The frozen variant is
  the cleanest test: it prevents the field from becoming a private code
  negotiated with the backbone.

### Repair 2 — the four-arm table (replaces the two-arm design)

A single table-vs-field comparison cannot identify where a gain came from
(the encoder adds parameters, compute, depth, and nonlinear preprocessing).

| Arm | State entering backbone | What it isolates |
|---|---|---|
| **T** | `E_fisher[b_t]` (static Fisher table) | the E1 reference |
| **H** | `W·h_φ,t` (contextual hidden state) | contextual preprocessing without probability geometry |
| **P** | `p_φ,t·Q` (raw probability state) | predictive distribution without the √ map |
| **F** | `√p_φ,t·Q` (Fisher state) | the full thesis |

H/P/F share the same `p_φ`, parameter budget, context, and data. Critical
contrasts: **H−T** (contextual at all), **P−H** (explicitly predictive),
**F−P** (Fisher geometry beyond exposing probabilities), **F−T** (headline).
Budget-limited start: T/H/F, with P as the declared follow-up (P is the
cleanest control for the √ map itself).

### Comparison lanes (both run; they answer different questions)

- **Matched-backbone**: 85M backbone unchanged + `p_φ` added — does the
  field provide useful information to an otherwise identical model?
- **Matched-total-parameter**: shrink the backbone so
  `N_backbone + N_field ≈ N_baseline` — is predictive-state construction a
  better use of a fixed budget? (the architecture-value test)

Report estimated training FLOPs and measured GPU-seconds in both lanes;
steps alone are not a fair compute comparison once the encoder exists.

### Endpoints (registered ex-ante in the prereg)

- **Scientific primary — learning-curve estimand**: paired gap
  `d_B = L_control(B) − L_field(B)` at preregistered source-byte budgets
  (e.g. `B ∈ {50M, 100M, 200M, 300M}`), and the primary statistic is the
  mean over budgets `d̄ = (1/K)·Σ_k d_{B_k}` — a discrete area-under-the-
  learning-curve. E1 showed a front-loaded effect; a contextual-state
  architecture is expected to change the CURVE, so a single terminal value
  is not primary. ε_L = 0.01 nats/byte retained for continuity with E1.
- **Engineering co-primary — steps/bytes-to-target**: one target every arm
  can reach; `L* = 0.80` nats/byte (NOT e0-byte terminal 0.7738, which sits
  too near the horizon — E1's convergence probe right-censored 2/9 random
  runs there). Per paired seed, `S = (steps_T − steps_F)/steps_T`;
  ε_S = 10%. Decision language: convergence improvement (LCB of S > 0);
  practically relevant (point estimate ≥ 10%); strong support (LCB > 10%).
  Bytes/step is constant across byte arms, so steps- and bytes-to-target
  coincide; report both.
- **Guardrail — terminal noninferiority**:
  `L_F(B_max) − L_T(B_max) < δ = 0.005` nats/byte, so "reaches useful
  quality sooner" cannot hide "plateaus worse".
- **Compute/systems readouts, reported but never merged into the primary**:
  FLOPs-to-target, GPU-seconds-to-target, peak VRAM, tokens/s, field share
  of params and FLOPs. A 10% step reduction is not a compute win if each
  step costs 25% more.

### Hierarchical decision rule

**Thesis supported** (all three): (1) F beats T on the learning-curve
estimand, CI entirely above 0; (2) F noninferior at final budget within δ;
(3) the field stays calibrated on its own held-out next-byte CE +
calibration metrics. **Practically valuable** (additionally, any one):
≥10% fewer bytes/steps to L*; ≥0.01 lower nats/byte at a fixed budget;
matched-total F matches/beats the 85M static baseline. **Compute-efficient**
(separately): fewer FLOPs- or GPU-seconds-to-target.

### Calibration, geometry, and context-sensitivity diagnostics

- **Predictive calibration** (logged for `p_φ` throughout): field nats/byte,
  Brier score, expected calibration error, entropy `H(p_φ,t)`, top-1
  next-byte accuracy. Field CE worsening while backbone CE improves is the
  private-code signature.
- **Geometry**: `θ(z_t, z_s) = arccos(z_t·z_sᵀ)` against Hellinger and
  Jensen–Shannon between the predictions; nearby states should have similar
  continuation distributions; corruption robustness. `Q` semi-orthogonal ⇒
  these correspondences are preserved numerically — tests prove it.
- **Context sensitivity**: for the SAME current byte in different contexts,
  the distribution of pairwise state angles. A static table gives zero
  within-byte variation; a genuine contextual field separates contexts in
  ways correlated with different futures — the clearest demonstration that
  the state is no longer "byte encoding".

### Anti-leakage: structural pin + behavioral test

The structural guarantee (`forward(idx)` takes no targets) is necessary but
insufficient. Registered behavioral test: two sequences identical through
position `t`, divergent after; assert `z_{≤t}^(A) = z_{≤t}^(B)` within
tolerance AND that post-divergence states can differ; separately, changing
the target tensor with `idx` fixed must not alter any `z_t`.

**Estimator ladder (unchanged):** v0 count model → **v1 tiny causal
transformer (this contract; pure-torch, current image)** → v2 Mamba2/SSM in
the shared E4 container lane. The `z_t ⊕ E[b_t]` combination stays a
declared follow-up arm; all four registered arms use state replacement.

### Implementation status vs this contract (2026-08-19)

Merged (PR #69): F-arm state path, causality tests, FP32 island, byte-only
refusals, param reporting. Open (PR #74, this branch, not yet merged): the
`λ_field` calibration seam
(`train.py`'s `_field_loss_terms`, folded into the backward `total` at the
same seam `geometry/gram.py`'s Gram penalty uses, NOT `aux_loss`) +
frozen-field mode (`FieldEncoder`'s `field_mode="frozen"` /
`field_ckpt_path`, guard-teeth in `tests/test_field_calibration.py`) — CE-only
telemetry (`field_ce`) at this stage, not yet the fuller Brier/ECE/entropy/
top-1 surface. Landed (FOR-924, this branch): H and P arm variants sharing
one `p_φ` — `field_state: Literal["sqrt_prob","prob","hidden"]` in
`config.py`, `configs/e1-5c-hidden.yaml` + `configs/e1-5c-prob.yaml`,
`FieldEncoder` branching in `field_encoder.py` (F/P share the FP32 softmax
island against the fixed semi-orthogonal lift; H is a plain trainable
`Linear(field_d_model, d_model)` over the encoder's hidden state, the only
arm with a trainable lift), and per-arm construction + causality guard-teeth
in `tests/`. Landed (P8, FOR-925, this branch): the remaining field
calibration telemetry — Brier/ECE/entropy/top-1, via a streaming
`FieldCalibrationAccumulator`, computed at eval time whenever a field
encoder exists at all (decoupled from `field_loss_lambda`, so a frozen field
still reports them) — surfaced through `telemetry.py`'s `EvalRecord`/
`read_eval_series` and forwarded by `sinks.py`; the behavioral anti-leakage
proofs (prefix-identical/suffix-divergent sequences share `z_<=t` and
diverge after; a forward hook proves targets never alter the field
encoder's own output) and the geometry-correspondence proofs (arccos(z·z)
reproduces the Bhattacharyya coefficient exactly under the semi-orthogonal
lift, θ and Jensen–Shannon move together along a mixture path, and a
context-sensitivity probe against the static-table zero control); and three
matched-total-parameter configs
(`configs/e1-5c-{field,hidden,prob}-matched-total.yaml`, this section's
second comparison lane) shrinking the backbone so `N_backbone + N_field`
lands in the shared 80–100M band — code-ready, not yet registered in
`EXPERIMENTS` (execution stays prereg-time scope, same
posture as `e3-anchor`/`e3-vantage`). Still to build before the prereg's
pods: none from this list — the prereg gate is the registration/execution
step itself, not further implementation.

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
settled component. If F wins, the ladder reads as one pipeline: discrete
observations → causal predictive field → Fisher state → spherical dynamics
(E2: does the backbone preserve the contextual field?) → curvature-aware
relations (E3: do Fisher states benefit from κ-kernels?) → learned
coarse-graining (E4: when to merge adjacent predictive states).

## Process

Per `docs/rung-lifecycle-protocol.md`: E1.5-D is exploratory and runs on
captured artifacts immediately. E1.5a/b/c each get their own preregistration
committed AFTER the E1 extension verdict files and BEFORE their first
confirmatory run, drawing stage sizes from the measured σ_seed. Clause
precedence is declared in every new registration (direction primary,
practical-relevance rider mandatory) per the stage-1 cloud review's MAJOR-1.
The CLAUDE.md ladder description gains the E1.5 box when this doc merges —
operator ratification required (north-star amendment).
