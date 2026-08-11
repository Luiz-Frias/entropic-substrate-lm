# syntax=docker/dockerfile:1
# =============================================================================
# entropic-substrate-lm training image.
#
# DEVIATION from the doc-lora-training two-stage pattern (Containerfile.devenv +
# thin app layer): this project ships a SINGLE-FILE, single-stage build instead.
# There is no shared devenv base image to cache across sibling repos here; the
# expensive part (torch/CUDA base image + third-party deps) is kept in early
# layers so Docker/Podman layer caching still gives "rebuild fast when only
# src/ changes", without a second image to publish and version.
#
# Target: single H100 (sm_90). CUDA 12.4 + cuDNN9 runtime is compatible with H100
# and matches the pinned torch>=2.6.0 wheel. Also runs CPU-only for `just test`.
#
# Build:  podman build -t entropic-substrate-lm:latest -f Containerfile .
# Run (H100, RunPod/Vast.ai): see CMD notes at the bottom of this file.
# =============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS base

WORKDIR /app

# git is required for the geoopt git+https dependency; the pytorch runtime
# base does not guarantee it.
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# uv — fast resolver/installer, used instead of pip for reproducible installs.
RUN pip install --no-cache-dir "uv>=0.5"

# Dependency layer: install third-party deps WITHOUT the project itself, so
# this layer stays cached across src/ changes. (Installing `.` here would fail
# anyway: the setuptools src-layout build needs src/ present.)
COPY pyproject.toml ./
RUN uv pip compile pyproject.toml -o /tmp/requirements.txt \
 && uv pip install --system --no-cache -r /tmp/requirements.txt

# Source layer: invalidated on every src/configs/scripts change, but cheap to
# rebuild since deps are already installed above.
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN uv pip install --system --no-cache --no-deps .

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# No ENTRYPOINT: an ENTRYPOINT would compose with the sleep CMD below into
# `cli sleep infinity` and crash on argparse. CMD-only keeps both modes clean.
#
# Default CMD is sleep-friendly for RunPod/Vast.ai pods: the container stays up
# so you can `exec` in and launch training interactively, rather than exiting
# immediately after a one-shot run. Override CMD to run training directly:
#
# Example direct-run on an H100 pod:
#   podman run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
#     -v $(pwd)/results:/app/results entropic-substrate-lm:latest \
#     python -m entropic_substrate.cli train --exp e3
CMD ["sleep", "infinity"]
