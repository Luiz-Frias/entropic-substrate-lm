# Deploy — H100 on RunPod / Vast.ai

The `Containerfile` builds a single image (base: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`,
CUDA 12.4 covers H100 / sm_90). Everything below assumes the image is pushed to a registry both
platforms can pull from (GHCR or Docker Hub).

## Build & push

```bash
just container-build                       # podman/docker build -t entropic-substrate-lm .
docker tag entropic-substrate-lm ghcr.io/luiz-frias/entropic-substrate-lm:latest
docker push ghcr.io/luiz-frias/entropic-substrate-lm:latest
```

## RunPod

- Pod type: GPU Pod, H100 80GB (PCIe is fine at this scale; SXM unnecessary for ~85M params).
- Container image: `ghcr.io/luiz-frias/entropic-substrate-lm:latest`.
- Volume: mount a persistent volume at `/app/models` and `/app/results` — pods are ephemeral.
- Env: `HF_TOKEN` (only needed for gated datasets; FineWeb-Edu is public), optional
  `HF_HOME=/app/.cache/hf` pointed at the volume so streaming shards cache across restarts.
- Run command examples:

```bash
python -m entropic_substrate.cli train --exp e1-fisher --device cuda
python -m entropic_substrate.cli train --exp e3 --device cuda
bash scripts/run_ladder.sh --device cuda   # gated ladder E1 → E2 → E3, dvc push per rung
```

## Vast.ai

- Filter: H100, CUDA ≥ 12.4 driver, verified hosts, ≥ 200GB disk if caching FineWeb shards.
- Image: same GHCR image; on-start script:

```bash
cd /app && python -m entropic_substrate.cli train --exp e3 --device cuda \
  2>&1 | tee results/e3-console.log
```

- Vast instances die without warning — checkpointing is on by default; resume with
  `--resume models/<exp>-latest.pt` (checkpoints are written as
  `models/<name>-latest.pt` / `models/<name>-final.pt`).

## Sizing expectations

- E0–E3 at ~85M params, seq 1024, micro-batch 4–16 w/ grad accumulation: a single H100 is
  overkill on memory (< 20GB) and each 20k-step run is hours, not days. bf16 autocast is on by
  default; the geometry stays FP32 internally regardless (see `architecture.md`).
- Multi-seed sweeps: run seeds sequentially on one pod rather than multi-pod — data streaming
  is the bottleneck, and the HF cache is shared on-volume.

## Sync results home

The DVC remote is committed in `.dvc/config` (Cloudflare R2, bucket `entropic-substrate-lm`,
the embedding-conversion convention: topology in git, identity in env). A pod only needs
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` from a bucket-scoped R2 Object Read & Write
token injected as env vars, then `dvc push`. For the tiny `results/` JSONL files,
`rsync`/`runpodctl send` remains a credential-free alternative.
