# E-4 Instruction — dynamic chunking hierarchy (deferred phase)

E4 replaces fixed sequence granularity with learned, content-dependent segmentation
([H-Net](https://arxiv.org/abs/2507.07955), pinned at `references/hnet`). It is deliberately
last: chunking and geometry are separate variables, and E4 only means something measured
against the settled E0–E3 winner.

**Precondition:** E3 concluded (positive or negative) with 3-seed nats/byte numbers. Carry
forward the winning init (E1) and block/norm stack (E2/E3) unchanged.

## Toolchain delta (the reason E4 is quarantined)

H-Net wants PyTorch ≥ 2.5.1 and `mamba_ssm` + `causal_conv1d` — prebuilt wheels exist for
common torch/CUDA/python combos (cp312 included); otherwise it is an nvcc source build
([goombalab/hnet](https://github.com/goombalab/hnet)). Do this in the container, not on the
host:

```dockerfile
# E4 image layer (extend the base Containerfile)
RUN pip install causal-conv1d mamba-ssm --no-build-isolation \
 || (git clone https://github.com/state-spaces/mamba && cd mamba && pip install .)
```

Escape hatch: the dynamic-chunking mechanism itself (routing module → boundary probability,
smoothing module, downsampling-ratio auxiliary loss, straight-through estimators) is pure
differentiable PyTorch. At ~85M scale, small transformer encoders in place of Mamba-2
encoder/decoder stages are acceptable for a first E4 — losing some efficiency but zero CUDA
build risk. Only reach for real `mamba_ssm` if the pure-torch variant shows signal worth
scaling.

## E4a — fixed-W chunking control

Before learned boundaries, run the HoloByte-style control: fixed W=8 chunk superposition on the
E3-winner stack (`references/holobyte/holobyte.py` is the reference for the
bind/unbind + micro-decoder pattern). This isolates "hierarchy at all" from "learned
boundaries," and its byte-receptive-field advantage (W× more bytes per macro-step at equal
sequence length) must be reported separately.

## E4b — learned dynamic chunking

Port H-Net's DC interface between the byte encoder and the macro backbone:

- Routing module: boundary probability from adjacent-representation similarity.
- Smoothing module: interpolate uncertain boundaries (this is what makes it learnable — do not
  skip it).
- Ratio loss targeting ~4.5–5 bytes/chunk average (H-Net's observed BPE-like equilibrium).
- One stage first; recurse to two stages only if stage-one beats E4a.

## Readouts

1. **nats/byte vs E4a and E3** at matched params *and* matched bytes-seen (chunking changes
   the byte/step ratio — control for it explicitly).
2. **Boundary inspection**: dump learned boundaries on held-out text; H-Net finds
   morpheme/word-like units without supervision. Boundaries that track whitespace only =
   the router degenerated to a space detector (compare against a whitespace-baseline chunker).
3. **Where tokenizers are weakest** (the H-Net result predicts the largest wins): add one
   non-space-delimited eval — Chinese FineWeb-2 slice or code (The Stack smol) — and report the
   delta vs the E0-bpe arm there specifically.
4. **Geometry × chunking interaction** (the novel question this repo adds): do κ trajectories
   and gate usage in the E3 layer shift when granularity becomes learned? Prediction: coarser
   learned chunks push κ more negative (chunk trees are deeper hierarchies than byte streams).

## Failure triage

| Symptom | First suspect |
|---|---|
| Ratio loss fights CE, loss oscillates | ratio target too aggressive; anneal it in |
| All boundaries at spaces | router capacity too low / boundary temperature too high |
| E4b ≈ E4a exactly | smoothing module missing or STE gradient blocked — check router grads |
| OOM at same params as E3 | dechunk/upsample path holding full-resolution activations; checkpoint them |
| mamba_ssm build failure | match wheel to torch/CUDA/cp312 exactly, or use the pure-torch escape hatch |

## Exit criteria

E4 ends the ladder. The write-up compares: substrate init (E1) × backbone geometry (E2/E3) ×
granularity (E4) on nats/byte, robustness-to-corruption, and the κ/gate diagnostics — that
factorial table is the contribution, not any single row.
