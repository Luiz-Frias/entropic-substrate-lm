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
# Target: single H100 (sm_90). CUDA 12.6 + cuDNN9 runtime is compatible with H100
# and matches the pinned torch==2.13.0 wheel. Also runs CPU-only for `just test`.
#
# Build:  podman build -t entropic-substrate-lm:latest -f Containerfile .
# Run (H100, RunPod/Vast.ai): see CMD notes at the bottom of this file.
# =============================================================================

FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS base

WORKDIR /app

# git is required by the DVC/repository workflow used to publish rung artifacts;
# the pytorch runtime base does not guarantee it.
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# uv — standalone, immutable amd64 binary from Astral's official distroless
# image. RETRACTION (2026-08-17): this comment previously claimed the copy
# "respects the base image's PEP 668 system-Python boundary" — wrong; copying
# a binary neither respects nor violates that boundary, which bites at
# INSTALL time. See the ENV below for the measured failure and the fix.
COPY --from=ghcr.io/astral-sh/uv:0.11.6@sha256:43cb71695fcad1516c2fbe0f56e500184c42d8bce838d9f64593b8aff2c16298 /uv /uvx /bin/
# MEASURED 2026-09-16 in the pinned project image: PATH selected a distinct
# /usr/local/bin/uv (SHA-256 1cb9cd0a...), shadowing /bin/uv (dc407e8e...).
# The explicit /bin/uv reported 0.11.6 and exported the current lock offline.
# Invoke that copied tool explicitly; COPY alone does not control PATH lookup.

# The base image's system Python 3.12 is Debian "externally managed"
# (PEP 668). MEASURED on the FIRST real build of this file (Docker Build
# Cloud, native amd64, 2026-08-17): the dependency layer failed exit 2 with
# uv refusing `uv pip install --system` against the marker ("Virtual
# environments were not considered due to the `--system` flag"). Inside this
# image the marker guards an OS Python that IS the payload interpreter —
# torch itself already sits in that site-packages, put there by the base
# image's own build — so the opt-out is deliberate, not a workaround.
# ENV form rather than a per-RUN flag so BOTH install layers below inherit
# it and cannot drift apart. Pinned by tests/test_deploy_config.py.
ENV UV_BREAK_SYSTEM_PACKAGES=1

# Dependency layer: export the audited runtime + deploy graph from uv.lock,
# excluding the dev-only test oracle and quality tools. Torch is pruned with its
# dependency closure because the official base owns the CUDA-matched wheel and
# libraries. This layer stays cached across src/ changes. (Installing `.` here
# would fail anyway: the setuptools src-layout build needs src/ present.)
COPY pyproject.toml uv.lock ./
RUN /bin/uv export --frozen --no-dev --group deploy --prune torch --no-emit-project --no-hashes \
      --format requirements-txt --output-file /tmp/requirements.txt \
 && /bin/uv pip install --system --no-cache -r /tmp/requirements.txt

# Source layer: invalidated on every src/configs/scripts change, but cheap to
# rebuild since deps are already installed above.
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
# The D1 pod-side worker composes its checkpoint and recovery transports from
# this committed, credential-free topology. Runtime credentials arrive only
# through RunPod secret references; no .env or local DVC config enters the
# remote build context.
COPY .dvc/config ./.dvc/config
# E1.5c frozen field: the pretrained p_phi (2.5MB) the frozen arms load via
# field_ckpt_path. Baked because pods are credential-free — and baking is
# what puts these weights under the image digest the launch record pins.
COPY models/field-pphi-v1.pt ./models/field-pphi-v1.pt
RUN /bin/uv pip install --system --no-cache --no-deps .

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
#     python -m entropic_substrate.composition.entrypoints.cli train --exp e3
CMD ["sleep", "infinity"]
