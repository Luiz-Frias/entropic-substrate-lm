# entropic-substrate-lm

Experiment repo testing substrate-level symbol encoding for language models. It combines
three ideas: (1) **√p Fisher-Rao initialization** — byte identity vectors initialized from
corpus next-byte transition distributions, sqrt-mapped onto the positive orthant of the unit
hypersphere (Fisher-Rao metric = angular metric), then lifted 256→`d_model` via a fixed
semi-orthogonal matrix; (2) an **nGPT-style hypersphere-native backbone** (normalization-as-
retraction, no manifold-aware optimizer, plain AdamW); and (3) a custom **gated
κ-stereographic normalization layer** ("MoE-over-geometries"): `logmap0` → sparse gating over
`K` constant-curvature components (each with a learnable, clamped curvature κ) → `expmap0` /
`projx` recombine, with a load-balancing auxiliary loss. Geometry math is always forced to
FP32 inside autocast-disabled blocks, regardless of the surrounding AMP precision. Prior art is
pinned as git submodules under [`references/`](references/) (holobyte, hnet, ngpt, geoopt).

## Quickstart

```bash
# 1. Trust and install pinned tool versions (python 3.12, uv, just, prek, ...)
mise trust
mise install

# 2. Sync the Python environment
uv sync --all-groups

# 3. Run the full quality gate (fmt-check + lint + typecheck + unit tests)
just check

# 4. Run an experiment (see the experiment table below for valid `exp` values)
just train exp=e1-fisher

# 5. Build + run the training container for a single H100 (RunPod / Vast.ai friendly)
just container-build
podman run --gpus all \
  -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results \
  entropic-substrate-lm:latest train --exp e3
```

Other useful commands:

```bash
just download-data      # fetch enwik8 into data/ (then `dvc add data/enwik8`)
just test                # unit tests only (CPU, <60s)
just test-all            # unit + integration tests
just eval ckpt=models/e3-latest.pt
```

## Experiment ladder

| ID | Description                                                                                  | Metric    |
| -- | --------------------------------------------------------------------------------------------- | --------- |
| E0 | Matched-param baselines: byte-level vanilla transformer **and** GPT-2-BPE transformer (~80–100M params) | nats/byte (BPE loss converted at the measured bytes/token on the eval split, not assumed 4.0) |
| E1 | E0 byte model with embedding init swapped: random-on-sphere vs. √p Fisher-Rao init. Identical arch/seed. | nats/byte |
| E2 | Hypersphere-native backbone (nGPT-style Norm-as-retraction blocks, SLERP/LERP residual with learnable eigen-LRs α) | nats/byte |
| E3 | E2 with LayerNorm/Norm replaced by the gated κ-stereographic layer (K=3 components: init κ = −1, 0, +1; learnable; sparse top-1 or top-2 gate + load-balance loss) | nats/byte + learned κ / gate-usage logs |

Datasets: [enwik8](http://mattmahoney.net/dc/textdata.html) (downloaded via
`just download-data` into `data/`, tracked with `dvc add data/enwik8`) and
[FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (streamed
via 🤗 `datasets`).

## Project layout

```text
entropic-substrate-lm/
├── src/entropic_substrate/   # package: data loaders, geometry, layers, models, train.py, cli.py
├── configs/                  # per-experiment YAML (e0..e3, tiny.yaml)
├── tests/                    # unit + integration tests (pytest markers: unit, integration)
├── scripts/                  # data download / utility scripts
├── docs/                     # design notes, ADRs, experiment writeups
├── references/               # pinned upstream submodules (holobyte, hnet, ngpt, geoopt)
├── data/ models/ results/    # DVC-tracked working dirs (empty dir placeholders only in git)
└── notebooks/                # scratch analysis notebooks (not tracked beyond .gitkeep)
```

## References

- Prior art is vendored as git submodules under [`references/`](references/): HoloByte,
  H-Net, nGPT, and [geoopt](https://github.com/geoopt/geoopt) (also used as the dev/test
  oracle that this repo's own κ-stereographic ops are checked against).
- See [`docs/`](docs/) for design notes, ADRs, and experiment write-ups.
- geoopt is installed directly from GitHub (`geoopt @ git+https://github.com/geoopt/geoopt.git`)
  per its own README's recommendation, ahead of its PyPI release cadence.
