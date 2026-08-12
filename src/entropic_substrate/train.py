"""Config-driven training loop.

Everything the experiment ladder needs lives here: seeding, AdamW with the
usual decay/no-decay split, cosine schedule with warmup, bf16 autocast (with
all manifold math forced to FP32 *inside* the layers), periodic held-out eval
reported in **nats/byte**, JSONL metrics, checkpointing and resume.

nats/byte is the only metric on which the byte and BPE arms of E0 are
comparable::

    nats_per_byte = cross_entropy_nats_per_token / bytes_per_token

with ``bytes_per_token`` *measured* on the eval split (see
``data/build.py``), never assumed to be 4.0.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from entropic_substrate.checkpoint import CHECKPOINT_FORMAT_VERSION, normalize_checkpoint
from entropic_substrate.config import Config
from entropic_substrate.data.batching import Corpus, SequenceSampler
from entropic_substrate.data.build import build_corpus, corpus_byte_stream
from entropic_substrate.logging import logger
from entropic_substrate.models.backbone import LanguageModel, LMOutput, build_model
from entropic_substrate.sinks import JsonlSink, MetricSink

__all__ = [
    "EvalResult",
    "Trainer",
    "cosine_lr",
    "resolve_device",
    "resolve_results_path",
    "seed_everything",
    "train_from_config",
]


def resolve_results_path(cfg: Config, results_path: str | os.PathLike[str] | None) -> Path:
    """The JSONL path a run writes to: ``results_path`` if given, else ``{results_dir}/{name}.jsonl``.

    Shared by :class:`Trainer` and ``entropic_substrate.cli`` (which needs the
    resolved path BEFORE constructing a ``Trainer``, to build its metric sink)
    — a single source of truth so the two can never drift apart.
    """
    if results_path is not None:
        return Path(results_path)
    return Path(cfg.train.results_dir) / f"{cfg.name}.jsonl"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def resolve_device(spec: str) -> torch.device:
    """Turn ``'auto' | 'cpu' | 'cuda' | 'cuda:1' | 'mps'`` into a device.

    ``'auto'`` honours the ``ENTROPIC_SUBSTRATE_DEVICE`` environment variable
    (used by CI to force CPU) before probing hardware.
    """
    if spec == "auto":
        env_spec = os.environ.get("ENTROPIC_SUBSTRATE_DEVICE", "").strip()
        if env_spec:
            return torch.device(env_spec)
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_lr(step: int, *, base_lr: float, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup then cosine decay to ``base_lr * min_ratio``."""
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return base_lr * min_ratio
    progress = (step - warmup) / max(total - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * coeff)


def _require_loss(out: LMOutput, phase: str) -> torch.Tensor:
    """Narrow ``out.loss`` to a tensor, failing loud when the model returned none.

    An ``assert`` here is stripped by ``python -O``, which turns a contract
    violation into a ``backward()`` on ``None`` (train) or a ``float(None)``
    deep inside the loop (eval). The eval path is a measurement path: it must
    fail loud, never fail open.
    """
    if out.loss is None:
        msg = f"{phase}: model returned no loss — forward was called without targets"
        raise RuntimeError(msg)
    return out.loss


@dataclass(slots=True)
class EvalResult:
    """Held-out evaluation summary."""

    loss: float
    nats_per_byte: float
    bits_per_byte: float
    perplexity: float
    kappa: list[list[float]]
    gate_fraction: list[list[float]]
    router_entropy: list[float]


class Trainer:
    """Owns the model, data, optimiser and metric sinks for one run."""

    def __init__(
        self,
        cfg: Config,
        *,
        corpus: Corpus | None = None,
        model: LanguageModel | None = None,
        device: torch.device | str | None = None,
        results_path: str | os.PathLike[str] | None = None,
        sink: MetricSink | None = None,
    ) -> None:
        self.cfg = cfg
        seed_everything(cfg.train.seed)
        self.device = (
            torch.device(device) if device is not None else resolve_device(cfg.train.device)
        )

        self.corpus = corpus if corpus is not None else build_corpus(cfg.data, cfg.model.tokenizer)
        self.meta = self.corpus.meta

        if model is None:
            fisher_source = None
            if cfg.model.embedding_init == "fisher_sqrt" and (
                cfg.model.fisher_init_path is None or not Path(cfg.model.fisher_init_path).exists()
            ):
                fisher_source = b"".join(corpus_byte_stream(cfg.data, 20_000_000))
            model = build_model(cfg.model, fisher_source=fisher_source)
        # Annotated, not inferred: `torch.compile`'s stub returns a bare Callable,
        # so without this the attribute widens to `LanguageModel | Callable` and
        # every later `.parameters()` / `.train()` / `.eval()` fails the strict
        # gate. The compiled wrapper is an nn.Module at runtime (`raw_model`
        # unwraps `_orig_mod`), so nn.Module is the honest type for both branches.
        self.model: nn.Module = model.to(self.device)
        if cfg.train.compile:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        self.train_sampler = SequenceSampler(
            self.corpus.train,
            cfg.data.batch_size,
            cfg.data.seq_len,
            train=True,
            seed=cfg.train.seed,
            device=self.device,
        )
        self.val_sampler = SequenceSampler(
            self.corpus.val,
            cfg.data.batch_size,
            cfg.data.seq_len,
            train=False,
            seed=cfg.train.seed + 1,
            device=self.device,
        )

        self.optimizer = self._build_optimizer()
        self.step = 0
        self.results_path = resolve_results_path(cfg, results_path)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        self.sink: MetricSink = sink if sink is not None else JsonlSink(self.results_path)
        self.ckpt_dir = Path(cfg.train.models_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_nats_per_byte = math.inf

        if cfg.train.resume:
            self.load_checkpoint(cfg.train.resume)

    # ----------------------------------------------------------- internals
    @property
    def raw_model(self) -> LanguageModel:
        """The underlying module (unwrapping ``torch.compile``)."""
        inner = getattr(self.model, "_orig_mod", self.model)
        return inner  # type: ignore[return-value]

    def _build_optimizer(self) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            # No weight decay on 1-D tensors: biases, norm gains, nGPT alpha/
            # sqk/suv/sz, and the learnable curvatures (decaying kappa toward 0
            # would silently bias the geometry MoE toward Euclidean).
            (no_decay if param.ndim < 2 else decay).append(param)
        groups: list[dict[str, Any]] = [
            {"params": decay, "weight_decay": self.cfg.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=self.cfg.train.lr, betas=self.cfg.train.betas, eps=1e-8)

    def _autocast(self):
        dtype = self.cfg.train.amp_dtype
        if dtype == "off" or self.device.type == "cpu":
            return torch.autocast(device_type=self.device.type, enabled=False)
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        return torch.autocast(device_type=self.device.type, dtype=torch_dtype)

    def _set_lr(self) -> float:
        lr = cosine_lr(
            self.step,
            base_lr=self.cfg.train.lr,
            warmup=self.cfg.train.warmup_steps,
            total=self.cfg.train.max_steps,
            min_ratio=self.cfg.train.min_lr_ratio,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def log(self, record: dict[str, Any]) -> None:
        """Fan one metric record out through :attr:`sink` (JSONL always, MLflow if wired)."""
        record = {"run": self.cfg.name, "step": self.step, **record}
        self.sink.log(record)

    # --------------------------------------------------------------- steps
    def train_step(self) -> dict[str, float]:
        """One optimisation step; returns scalar metrics."""
        self.model.train()
        lr = self._set_lr()
        x, y = self.train_sampler.batch()
        with self._autocast():
            out = self.model(x, y)
        loss = _require_loss(out, "train_step")
        aux = out.aux_loss
        total = loss if aux is None else loss + aux
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        self.raw_model.normalize_weights()
        self.step += 1

        loss = float(out.loss.detach())
        return {
            "loss": loss,
            "aux_loss": float(aux.detach()) if aux is not None else 0.0,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "nats_per_byte": loss / self.meta.bytes_per_token,
        }

    @torch.no_grad()
    def evaluate(self, num_batches: int | None = None) -> EvalResult:
        """Deterministic held-out evaluation, reported in nats/byte."""
        self.model.eval()
        self.val_sampler.reset_eval()
        n = num_batches if num_batches is not None else self.cfg.data.num_eval_batches
        losses: list[float] = []
        kappas: list[list[float]] = []
        gates: list[list[float]] = []
        entropies: list[float] = []
        for x, y in self.val_sampler.iter_batches(n):
            with self._autocast():
                out = self.model(x, y)
            losses.append(float(_require_loss(out, "evaluate")))
            if out.kappa_stats:
                kappas = [s.kappa for s in out.kappa_stats]
                gates = [s.gate_fraction for s in out.kappa_stats]
                entropies = [s.entropy for s in out.kappa_stats]
        mean_loss = float(np.mean(losses))
        npb = mean_loss / self.meta.bytes_per_token
        self.model.train()
        return EvalResult(
            loss=mean_loss,
            nats_per_byte=npb,
            bits_per_byte=npb / math.log(2.0),
            perplexity=float(math.exp(min(mean_loss, 20.0))),
            kappa=kappas,
            gate_fraction=gates,
            router_entropy=entropies,
        )

    # ----------------------------------------------------------- lifecycle
    def state_dict(self) -> dict[str, Any]:
        """Full run state for exact resume."""
        return {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": self.step,
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "sampler": self.train_sampler.state_dict(),
            "config": self.cfg.to_dict(),
        }

    def save_checkpoint(self, tag: str = "latest") -> Path:
        """Write ``models/<run>-<tag>.pt``."""
        path = self.ckpt_dir / f"{self.cfg.name}-{tag}.pt"
        torch.save(self.state_dict(), path)
        return path

    def load_checkpoint(self, path: str | os.PathLike[str]) -> None:
        """Restore model/optimiser/sampler/step from a checkpoint file."""
        # weights_only=True matches the eval path in cli.py: checkpoints are
        # self-produced and hold only tensors and primitives, so the secure
        # default loads them fine (bandit B614).
        state = normalize_checkpoint(
            torch.load(Path(path), map_location=self.device, weights_only=True),
            source=str(path),
            for_resume=True,
        )
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.train_sampler.load_state_dict(state["sampler"])
        self.step = int(state["step"])

    def _eval_record(self, result: EvalResult, wall: float) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": "eval",
            "loss": round(result.loss, 6),
            "nats_per_byte": round(result.nats_per_byte, 6),
            "bits_per_byte": round(result.bits_per_byte, 6),
            "perplexity": round(result.perplexity, 4),
            "bytes_per_token": round(self.meta.bytes_per_token, 6),
            "wall_s": round(wall, 3),
        }
        if result.kappa:
            record["kappa"] = result.kappa
            record["gate_usage"] = result.gate_fraction
            record["router_entropy"] = [round(e, 6) for e in result.router_entropy]
        return record

    def fit(self, max_steps: int | None = None, verbose: bool = True) -> list[dict[str, Any]]:
        """Run the training loop; returns the list of per-step train metrics."""
        total = max_steps if max_steps is not None else self.cfg.train.max_steps
        history: list[dict[str, Any]] = []
        start = time.time()

        self.log(
            {
                "event": "run_start",
                "config": self.cfg.to_dict(),
                "params": self.raw_model.num_parameters(),
                "device": str(self.device),
                "tokenizer": self.meta.tokenizer,
                "bytes_per_token": self.meta.bytes_per_token,
                "train_tokens": self.meta.train_tokens,
                "val_tokens": self.meta.val_tokens,
            }
        )

        while self.step < total:
            metrics = self.train_step()
            history.append(metrics)
            if self.step % self.cfg.train.log_interval == 0 or self.step == total:
                record = {
                    "event": "train",
                    **{k: round(v, 6) for k, v in metrics.items()},
                    "wall_s": round(time.time() - start, 3),
                }
                self.log(record)
                if verbose:
                    logger.info(
                        f"step {self.step:>6} loss {metrics['loss']:.4f} "
                        f"nats/byte {metrics['nats_per_byte']:.4f} "
                        f"lr {metrics['lr']:.2e}"
                    )
            if self.cfg.train.eval_interval > 0 and (
                self.step % self.cfg.train.eval_interval == 0 or self.step == total
            ):
                result = self.evaluate()
                record = self._eval_record(result, time.time() - start)
                self.log(record)
                if result.nats_per_byte < self.best_nats_per_byte:
                    self.best_nats_per_byte = result.nats_per_byte
                    best_path = self.save_checkpoint("best")
                    self.sink.log_artifact(best_path, artifact_path="checkpoints")
                if verbose:
                    kappa = record.get("kappa")
                    extra = f" kappa {kappa}" if kappa else ""
                    logger.info(
                        f"  eval  loss {result.loss:.4f} "
                        f"nats/byte {result.nats_per_byte:.4f} "
                        f"bits/byte {result.bits_per_byte:.4f}{extra}"
                    )
            if self.cfg.train.save_interval > 0 and self.step % self.cfg.train.save_interval == 0:
                self.save_checkpoint()

        final_path = self.save_checkpoint("final")
        self.sink.log_artifact(final_path, artifact_path="checkpoints")
        self.log({"event": "run_end", "wall_s": round(time.time() - start, 3)})
        self.sink.finish()
        return history

    def iter_history(self) -> Iterator[dict[str, Any]]:
        """Re-read this run's JSONL records."""
        if not self.results_path.exists():
            return
        with self.results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def train_from_config(cfg: Config, **kwargs: Any) -> Trainer:
    """Build a :class:`Trainer` from ``cfg``, run it, and return it."""
    trainer = Trainer(cfg, **kwargs)
    trainer.fit()
    return trainer
