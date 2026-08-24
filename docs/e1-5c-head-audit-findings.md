# E1.5c frozen-head audit — exploratory findings (D0)

**EXPLORATORY.** This is an endpoint diagnostic on the captured E1.5c frozen
checkpoint — no training, CPU only, no registered gate. It feeds judgment
about the frozen verdict's causal ladder (`docs/verdicts/e1-5c-frozen-verdict.md`
§3, P−H **MEASURED** −0.0508 nats/byte); it decides nothing. Filed 2026-08-23.

## Question

Before attributing the H→P loss to *compression* (a 128-dim hidden state
squeezed through a 256-way head), the smallest honest diagnostic: is the
frozen head injective in the softmax quotient, well-conditioned, and
invertible **in production precision** on the states the runs actually
produced?

## Inputs and integrity

| item | value | evidence class |
|---|---|---|
| Checkpoint | `models/field-pphi-v1.pt`, MD5 `edc3e3e55ccc340bf565e14ac547618c` | **MEASURED**; matches the verdict's §2.2 integrity chain |
| Config | `configs/e1-5c-field.yaml` via `load_config` with overlay `{model: {field_mode: frozen, field_ckpt_path: models/field-pphi-v1.pt}}` | the same frozen overrides the captured runs report in `run_start` |
| Head | tied `head.weight = wte.weight`, (256, 128), bias-free | **MEASURED** `torch.equal` between `load_frozen_head` and the live module |
| Eval positions | first 8 batches of the deterministic held-out grid (`SequenceSampler(train=False)`, batch 16 × seq 1024) = 131,072 positions | same grid construction as `analysis.embedding_geometry.functional_divergence` |
| Machinery | `entropic_substrate.analysis.field_head_audit` (branch `field-head-audit`, test-pinned) | driver snippet in §6 |

## 1. Head spectrum — **MEASURED**

All spectra fp64; "centered" = `(I − 11ᵀ/256) W`, the matrix that governs
`h → softmax(Wh)` after the softmax shift-invariance quotient.

| readout | raw W | centered W |
|---|---:|---:|
| rank (of 128) | 128 | **128** |
| σ_max | 27.596 | 21.106 |
| σ_min | 0.2084 | 0.2006 |
| condition number | 132.4 | **105.2** |
| entropy effective rank | 74.96 | 83.67 |
| stable rank (‖·‖²_F/σ²_max, derived in driver) | — | 1.387 |

The head is **injective in the softmax quotient**: no rank deficiency, no
all-ones-direction pathology, σ_min two-tenths — not a numerically dead
direction. The stable rank near 1 alongside entropy effective rank ~84 says
one direction dominates the energy (σ 21.1 vs a bulk in ~0.2–2) with a long,
alive tail.

## 2. Reconstruction from production probabilities — **MEASURED**

`ĥ = pinv(centered W) · center(log p)`, where `p` is constructed exactly as
`field_encoder._encode` produces it for the P/F arms: fp32 softmax, then
`clamp_min(1e-12)`. So this measures the information budget the P and F arms
actually received.

| readout | value |
|---|---:|
| positions | 131,072 |
| clamped fraction (≥1 byte at the 1e-12 floor) | **0.2662** |
| mean clamped bytes per position (all positions) | 14.45 |
| max clamped bytes in one position | 62 |
| median relative error | 1.14e-6 |
| mean relative error | 0.0870 |
| p99 relative error | 0.6949 |
| max relative error | 0.8409 |
| median angular error | 6.5e-5 ° |
| p99 / max angular error | 37.0° / 42.8° |

### Conditional split — the error mass sits exactly on the clamp

| group | n | median rel err | p99 rel err | max rel err | median field entropy |
|---|---:|---:|---:|---:|---:|
| unclamped | 96,182 | 1.03e-6 | 1.66e-6 | **2.96e-6** | 1.427 nats |
| clamped | 34,890 | **0.3198** | 0.7557 | 0.8409 | **0.601 nats** |

Positions with relative error > 1%: **0** unclamped, 33,901 clamped (97.2%
of the clamped group). The separation is total.

## 3. Realized hidden-state spectrum — **MEASURED**

| readout | value |
|---|---:|
| numerical rank (of 128) | 127 |
| entropy effective rank | 101.1 |
| σ top-5 | 3357.8, 2325.3, 1868.6, 1732.3, 1448.1 |
| σ bottom-5 | 194.6, 192.3, 187.3, 185.6, **8.6e-5** |

One direction is structurally dead, not learned-dead: the post-`norm_f`
states of a LayerNorm carry a zero-feature-mean constraint, so realized
states live in (approximately) the 127-dim hyperplane orthogonal to `1_128`
— consistent with the single σ at 8.6e-5 against a floor of ~185 for every
other direction. The data excites the remaining space broadly (effective
rank 101 of 127): **no low-dimensional collapse of the field state.**

## 4. What this answers (and what it does not)

**The compression-in-principle story is dead.** The head is injective in the
softmax quotient, moderately conditioned (κ ≈ 105), and on 73.4% of eval
positions the hidden state is recoverable from the production probabilities
to fp32 noise (max 3.0e-6). No information is discarded by the 128→256
projection as a map.

**The production loss is real, and it is the floor.** All measured
information loss is confined to the 26.6% of positions where the 1e-12 clamp
fired — where the frozen field is *most confident* (median entropy 0.60 vs
1.43 nats). There, a median 32% of the hidden state (by relative norm) is
unrecoverable, up to 43° of angular error. The P/F arms received a state
selectively corrupted on exactly the most predictable positions — the
positions where a byte-level model earns most of its nats/byte.

**Interpretation, not measurement:** the clamp is where the loss becomes
countable, but the underlying mechanism is amplitude-domain squashing —
distinctions in the deep log-tail (logit gaps beyond ~27.6 nats) map to
amplitude differences of order ≤1e-6 in `p` (≤1e-6 in `√p`), which the
downstream L2/angular geometry cannot resolve regardless of where the floor
sits. Lowering the floor would restore *log-space* recoverability in this
audit without necessarily restoring *amplitude-space* usability to the
backbone. The √ map halves the log-scale compression (consistent with the
verdict's F−P **MEASURED** +0.0138) but cannot undo it (F−H **MEASURED**
−0.0369). Untested here; a matched-adapter probe (D1) would separate it from
the fixed-lift-vs-trainable-adapter confound, which this audit does not
touch.

## 5. Bearing on the causal ladder

The H→P regression (−0.0508) now decomposes into at least: (a) a measured,
position-selective information loss on the low-entropy quarter of positions;
(b) an unmeasured interface/conditioning component (κ ≈ 105 coordinates,
fixed `Q` vs H's trainable `state_lift`). Neither alone is established as
dominant. The audit rules out only the uniform-bottleneck reading.

## 6. Reproduction

Driver (run from repo root, `PYTHONPATH=src`, checkpoint and enwik8 pulled
via DVC):

```python
from entropic_substrate.analysis.field_head_audit import audit_field_head, load_frozen_head
from entropic_substrate.config import load_config
from entropic_substrate.data.batching import SequenceSampler
from entropic_substrate.data.build import build_corpus
from entropic_substrate.models.field_encoder import FieldEncoder

cfg = load_config("configs/e1-5c-field.yaml", overlay={
    "model": {"field_mode": "frozen", "field_ckpt_path": "models/field-pphi-v1.pt"}})
encoder = FieldEncoder(cfg.model)
corpus = build_corpus(cfg.data, cfg.model.tokenizer)
sampler = SequenceSampler(corpus.val, cfg.data.batch_size, cfg.data.seq_len,
                          train=False, seed=cfg.train.seed + 1, device="cpu")
audit = audit_field_head(encoder, [x for x, _y in sampler.iter_batches(8)])
```

The conditional split (§2) derives from the same capture via
`collect_field_states` + `reconstruct_hidden` with the clamp mask
`probs.min(dim=-1).values <= PROB_FLOOR`.
