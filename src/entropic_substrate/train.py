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
import torch.nn.functional as F
from torch import nn

from entropic_substrate.checkpoint import CHECKPOINT_FORMAT_VERSION, normalize_checkpoint
from entropic_substrate.config import Config
from entropic_substrate.data.batching import (
    DATA_ORDER_DIGEST_BATCHES,
    Corpus,
    SequenceSampler,
)
from entropic_substrate.data.build import build_corpus, corpus_byte_stream
from entropic_substrate.geometry.gram import gram_frobenius_sq, normalized_gram
from entropic_substrate.logging import logger
from entropic_substrate.models.backbone import LanguageModel, LMOutput, build_model
from entropic_substrate.models.field_encoder import FieldCalibrationAccumulator
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
    # E1.5b: None when gram_lambda == 0.0 (disabled) — never 0.0-filled, so a
    # run without the penalty carries no gram_* keys at all (same "absent
    # channel is absent" convention `kappa`/`gate_fraction` already use).
    gram_frob_sq: float | None = None
    gram_penalty: float | None = None
    # E1.5c Repair 1: None when field_loss_lambda == 0.0 (disabled) — same
    # "absent channel is absent" convention as the gram_* pair above. Mean
    # calibration CE over this eval's batches; NEVER folded into `loss`/
    # `nats_per_byte` above (train.py's registered likelihood metric stays
    # pure backbone CE, test-pinned in test_field_calibration.py).
    field_ce: float | None = None
    # P8 (FOR-925): predictive-calibration diagnostics for p_phi, "logged
    # THROUGHOUT" (docs/e1_5-instruction.md) — present whenever a field
    # encoder exists at all, DECOUPLED from field_loss_lambda/
    # _field_loss_enabled above (which gate only the train-loss calibration
    # seam, P6/FOR-923, unchanged). None under state_source="table" (no
    # field encoder to diagnose) — same absent-channel convention as
    # gram_frob_sq/field_ce.
    field_nats_per_byte: float | None = None
    field_brier: float | None = None
    field_ece: float | None = None
    field_entropy: float | None = None
    field_top1: float | None = None


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

        # E1.5b: snapshot the Gram-penalty reference BEFORE any resume load.
        # Init is deterministic (embedding_init draws from a fixed generator
        # seed — see init/fisher.py), so this snapshot is identical whether
        # or not the run resumes; snapshotting from the model's CURRENT
        # (fresh) weights here, rather than re-deriving it after a resumed
        # load, is what makes "preserve the init geometry" mean the same
        # thing on a resumed run as on a fresh one — the checkpoint is never
        # consulted for this reference, so nothing needs to persist it.
        # `gram_lambda == 0.0` (every arm except e1-5b-gram) skips the
        # snapshot entirely: no reference, no seam, no extra compute.
        self._gram_ref: torch.Tensor | None = None
        if cfg.model.gram_lambda != 0.0:
            with torch.no_grad():
                self._gram_ref = normalized_gram(self.raw_model.wte.weight).detach().clone()

        # E1.5c Repair 1: field_loss_lambda != 0.0 requires an actual field
        # encoder to calibrate — a table-arm run (state_source="table") has
        # raw_model.field_encoder is None, and reaching
        # `_field_loss_terms` on it would surface as an AttributeError deep
        # inside train_step/evaluate rather than a clear construction-time
        # refusal (CLAUDE.md Errors principle: fail fast on invalid state).
        self._field_loss_enabled = cfg.model.field_loss_lambda != 0.0
        if self._field_loss_enabled and self.raw_model.field_encoder is None:
            msg = (
                "field_loss_lambda != 0.0 requires state_source='field' — "
                "no field encoder exists to calibrate under state_source='table'"
            )
            raise ValueError(msg)
        if self._field_loss_enabled and cfg.model.field_mode == "frozen":
            msg = (
                "field_mode='frozen' requires field_loss_lambda == 0.0 — "
                "frozen field weights receive no gradient, so the calibration "
                "term cannot train anything; frozen-field CE telemetry is "
                "the eval-time job (FOR-925, docs/e1_5-instruction.md §E1.5c Repair 1)"
            )
            raise ValueError(msg)

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

    def _gram_penalty_terms(self) -> tuple[torch.Tensor, torch.Tensor]:
        """E1.5b Gram-preserving penalty: ``(frob_sq, gram_lambda * frob_sq)``.

        Only called when :attr:`_gram_ref` is set (``gram_lambda != 0.0`` at
        construction) — callers guard that themselves. FP32 throughout, inside
        an explicit ``autocast(enabled=False)`` block: this runs OUTSIDE
        ``train_step``'s own ``_autocast()`` context (the forward pass has
        already returned by the time this is called), but the geometry-
        numerics invariant (CLAUDE.md) binds unconditionally, not only when
        the caller happens to already be under bf16 — matches
        ``geometry/kappa.py``'s scalar reductions.
        """
        if self._gram_ref is None:
            msg = "_gram_penalty_terms called with the Gram seam disabled (gram_lambda == 0.0)"
            raise RuntimeError(msg)
        with torch.autocast(device_type=self.device.type, enabled=False):
            gram_t = normalized_gram(self.raw_model.wte.weight)
            frob_sq = gram_frobenius_sq(gram_t, self._gram_ref)
            penalty = self.cfg.model.gram_lambda * frob_sq
        return frob_sq, penalty

    def _field_loss_terms(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """E1.5c Repair 1 calibration seam: ``(L_field, field_loss_lambda * L_field)``.

        Only called when :attr:`_field_loss_enabled` (``field_loss_lambda !=
        0.0``, validated + raised at construction if no field encoder
        exists — callers do not need to guard that themselves). ``p_phi``
        still consumes ``x`` only (``FieldEncoder.field_logits`` takes no
        ``targets`` parameter, the same structural anti-leakage pin
        ``forward`` carries); ``y`` scores the prediction here exactly as
        the backbone's own CE does in ``LanguageModel.forward`` — target
        scores, never feeds. ``y[:, t] == x[:, t+1]`` by construction
        (``data/batching.py``'s ``SequenceSampler.batch``), which is exactly
        the byte ``z_t = f(x_<=t)`` already predicts (verified by
        ``test_field_encoder.py``'s causality suite) — so this scores
        ``field_logits`` against ``y`` ELEMENTWISE, with no slice, the same
        shape ``LanguageModel.forward`` uses for its own CE.

        FP32 throughout, inside an explicit ``autocast(enabled=False)``
        block — mirrors ``_gram_penalty_terms`` EXACTLY: this runs OUTSIDE
        ``train_step``'s own ``_autocast()`` context (the forward pass has
        already returned by the time this is called), but the geometry/
        numerics invariant (CLAUDE.md) binds unconditionally, not only when
        the caller happens to already be under bf16. NOT via ``aux_loss``
        (that channel is the dead ``GatedKappaNorm`` load-balance term).
        """
        field_encoder = self.raw_model.field_encoder
        if field_encoder is None:
            # Construction-time validation (Trainer.__init__) already
            # refuses field_loss_lambda != 0.0 without a field encoder, so
            # reaching here means that guard was bypassed (e.g. a caller
            # mutating `_field_loss_enabled` by hand) — fail loud rather
            # than silently `assert`, which `python -O` strips (mirrors
            # `_require_loss`'s documented rationale for the same choice).
            msg = "_field_loss_terms called with no field encoder on the model (state_source != 'field')"
            raise RuntimeError(msg)
        with torch.autocast(device_type=self.device.type, enabled=False):
            logits = field_encoder.field_logits(x)
            field_loss = self._field_ce_from_logits(logits, y)
            weighted = self.cfg.model.field_loss_lambda * field_loss
        return field_loss, weighted

    def _field_ce_from_logits(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Field cross-entropy from already-computed FP32 field logits.

        Extracted (CodeRabbit finding, P8 review round) so the
        ``autocast(enabled=False)`` + ``F.cross_entropy`` island is defined
        once rather than duplicated verbatim between :meth:`_field_loss_terms`
        (train seam) and :meth:`evaluate` (eval-time calibration diagnostic,
        P8/FOR-925) — both need the identical FP32 field CE, one weighted by
        ``field_loss_lambda`` for the backward pass, one plain for
        telemetry. Nesting this inside a caller's own already-active
        ``autocast(enabled=False)`` block (as :meth:`_field_loss_terms`
        does) is a harmless no-op — the context managers compose.
        """
        with torch.autocast(device_type=self.device.type, enabled=False):
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

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
        gram_metrics: dict[str, float] = {}
        if self._gram_ref is not None:
            gram_frob_sq, gram_penalty = self._gram_penalty_terms()
            # NOT via aux_loss: that channel is the dead GatedKappaNorm
            # load-balance term (CLAUDE.md architecture section). This is a
            # second, independent term added directly to the optimized total.
            total = total + gram_penalty
            gram_metrics = {
                "gram_frob_sq": float(gram_frob_sq.detach()),
                "gram_penalty": float(gram_penalty.detach()),
            }
        field_metrics: dict[str, float] = {}
        if self._field_loss_enabled:
            field_loss, field_weighted = self._field_loss_terms(x, y)
            # NOT via aux_loss — same rationale as the gram term above.
            total = total + field_weighted
            field_metrics = {"field_ce": float(field_loss.detach())}
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
            **gram_metrics,
            **field_metrics,
        }

    def evaluate(self, num_batches: int | None = None) -> EvalResult:
        """Deterministic held-out evaluation, reported in nats/byte."""
        with torch.no_grad():
            self.model.eval()
            self.val_sampler.reset_eval()
            n = num_batches if num_batches is not None else self.cfg.data.num_eval_batches
            losses: list[float] = []
            kappas: list[list[float]] = []
            gates: list[list[float]] = []
            entropies: list[float] = []
            field_losses: list[float] = []
            # P8 (FOR-925): decoupled from `_field_loss_enabled` — whenever a
            # field encoder exists at all (state_source == "field"), even
            # with the calibration seam off or frozen (field_loss_lambda ==
            # 0.0 either way), its own predictive-calibration diagnostics
            # still accumulate here ("logged for p_phi THROUGHOUT",
            # docs/e1_5-instruction.md). See FieldCalibrationAccumulator's
            # docstring for why this streams scalar reductions per batch
            # rather than concatenating logits across the whole eval pass.
            field_encoder = self.raw_model.field_encoder
            field_calib = FieldCalibrationAccumulator() if field_encoder is not None else None
            for x, y in self.val_sampler.iter_batches(n):
                with self._autocast():
                    out = self.model(x, y)
                losses.append(float(_require_loss(out, "evaluate")))
                if out.kappa_stats:
                    kappas = [s.kappa for s in out.kappa_stats]
                    gates = [s.gate_fraction for s in out.kappa_stats]
                    entropies = [s.entropy for s in out.kappa_stats]
                # E1.5c Repair 1: `field_loss` is accumulated into its OWN
                # list, never `losses` above — the registered likelihood
                # metric (`mean_loss`/`npb` below) must stay pure backbone
                # CE, test-pinned in test_field_calibration.py.
                #
                # The field encoder runs ONCE per eval batch, inside the
                # `self.model(x, y)` call above: `LanguageModel.forward`
                # carries its next-byte logits out on `LMOutput.field_logits`
                # (CR PR #79), and both consumers below (the CE loss and the
                # calibration accumulator) read them from there.
                #
                # RETRACTION (2026-08-20) of the note that stood here: it
                # claimed `field_logits(x)` was "computed at most ONCE per
                # eval batch" — true of the two consumers BELOW, but false of
                # the encoder as a whole, which ran its full forward twice
                # per batch (once for `z` inside `LanguageModel.forward`,
                # once here). MEASURED before the fix, by counting
                # `FieldEncoder._encode` calls across `evaluate(num_batches=
                # 3)`: 6 calls, now 3 (tests/test_field_calibration.py::
                # test_evaluate_runs_the_field_encoder_exactly_once_per_batch).
                #
                # This also drops the `autocast(enabled=False)` island that
                # used to wrap the second call: the encoder now runs in the
                # AMBIENT dtype on the eval path, like every other forward.
                # MEASURED no-op where the tests run (`amp_dtype="off"`/CPU
                # makes `_autocast()` itself `enabled=False`, so the two
                # shapes are bit-identical — pinned by
                # test_deduped_eval_reproduces_the_two_pass_field_metrics_
                # exactly). Under a real bf16 autocast it is NOT a no-op: the
                # calibration diagnostics now describe p_phi as the model
                # actually runs it, rather than an FP32 re-run of it. The CE
                # reduction itself stays FP32 either way —
                # `_field_ce_from_logits` supplies its own island, and
                # `FieldCalibrationAccumulator.update` refuses non-fp32 input
                # loudly.
                field_logits: torch.Tensor | None = None
                # Annotated local, not `out.field_logits` read inline: the
                # model attribute comes back through `nn.Module.__call__`,
                # typed `Any`, and an `Any` assigned into a declared
                # `Tensor | None` slot stays `Tensor | None` no matter what
                # check precedes it (basedpyright: "reshape is not a known
                # attribute of None"). Binding it once here gives the
                # `is None` check below something it can actually narrow.
                encoded_field_logits: torch.Tensor | None = out.field_logits
                if field_encoder is not None:
                    if encoded_field_logits is None:
                        # Same fail-loud posture as the `_field_loss_enabled`
                        # branch below: a field encoder exists, so the
                        # forward pass MUST have populated this channel.
                        # Reaching here means backbone and trainer disagree
                        # about the contract — never silently skip the
                        # diagnostics (a measurement path fails loud, it
                        # does not fail open).
                        msg = (
                            "model has a field encoder but forward returned no "
                            "field_logits — LMOutput.field_logits is the eval-time "
                            "calibration channel"
                        )
                        raise RuntimeError(msg)
                    # `.float()` here, not in `LanguageModel.forward`: the
                    # channel is ambient-dtype by design (a training forward
                    # never reads it and must not pay a (B, T, 256) fp32 copy
                    # per step), and `FieldCalibrationAccumulator.update`
                    # requires fp32.
                    field_logits = encoded_field_logits.float()
                    if field_calib is not None:
                        field_calib.update(
                            field_logits.reshape(-1, field_logits.shape[-1]), y.reshape(-1)
                        )
                if self._field_loss_enabled:
                    if field_logits is None:
                        # Same fail-loud contract as `_field_loss_terms`
                        # (construction-time validation already refuses
                        # field_loss_lambda != 0.0 without a field encoder;
                        # reaching here means that guard was bypassed, e.g.
                        # a caller mutating `_field_loss_enabled` by hand).
                        msg = (
                            "_field_loss_enabled but no field encoder on the model "
                            "(state_source != 'field')"
                        )
                        raise RuntimeError(msg)
                    # Shared with `_field_loss_terms` — see
                    # `_field_ce_from_logits`'s own docstring for why this
                    # was previously duplicated inline here.
                    field_loss = self._field_ce_from_logits(field_logits, y)
                    field_losses.append(float(field_loss.detach()))
            mean_loss = float(np.mean(losses))
            npb = mean_loss / self.meta.bytes_per_token
            field_ce: float | None = float(np.mean(field_losses)) if field_losses else None
            field_nats_per_byte: float | None = None
            field_brier: float | None = None
            field_ece: float | None = None
            field_entropy: float | None = None
            field_top1: float | None = None
            if field_calib is not None:
                diag = field_calib.finalize()
                field_nats_per_byte = diag["ce"] / self.meta.bytes_per_token
                field_brier = diag["brier"]
                field_ece = diag["ece"]
                field_entropy = diag["entropy"]
                field_top1 = diag["top1"]
            # E1.5b: recomputed here (no_grad, already inside this method's
            # outer torch.no_grad()) from the CURRENT table only — it does
            # not depend on the eval batches above and must never perturb
            # `mean_loss`/`npb`: the penalty is a training regularizer, not
            # part of the measured likelihood.
            gram_frob_sq: float | None = None
            gram_penalty: float | None = None
            if self._gram_ref is not None:
                frob_sq, penalty = self._gram_penalty_terms()
                gram_frob_sq = float(frob_sq)
                gram_penalty = float(penalty)
            self.model.train()
            return EvalResult(
                loss=mean_loss,
                nats_per_byte=npb,
                bits_per_byte=npb / math.log(2.0),
                perplexity=float(math.exp(min(mean_loss, 20.0))),
                kappa=kappas,
                gate_fraction=gates,
                router_entropy=entropies,
                gram_frob_sq=gram_frob_sq,
                gram_penalty=gram_penalty,
                field_ce=field_ce,
                field_nats_per_byte=field_nats_per_byte,
                field_brier=field_brier,
                field_ece=field_ece,
                field_entropy=field_entropy,
                field_top1=field_top1,
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
        if result.gram_frob_sq is not None and result.gram_penalty is not None:
            record["gram_frob_sq"] = round(result.gram_frob_sq, 6)
            record["gram_penalty"] = round(result.gram_penalty, 6)
        if result.field_ce is not None:
            record["field_ce"] = round(result.field_ce, 6)
        field_nats_per_byte = result.field_nats_per_byte
        field_brier = result.field_brier
        field_ece = result.field_ece
        field_entropy = result.field_entropy
        field_top1 = result.field_top1
        field_calib_values = (
            field_nats_per_byte,
            field_brier,
            field_ece,
            field_entropy,
            field_top1,
        )
        if any(v is not None for v in field_calib_values):
            # P8 (FOR-925): the five calibration-diagnostic keys are always
            # computed together (one FieldCalibrationAccumulator.finalize()
            # call in evaluate()) — guarding on "any set" and requiring "all
            # set" is equivalent to guarding on all five, in BOTH directions.
            # `assert` is not used here (stripped under `python -O`,
            # CLAUDE.md's documented rationale for `_require_loss`/
            # `_field_loss_terms`'s own control-flow checks): a caller that
            # constructs an EvalResult with any one of the five set but
            # another left None (evaluate() never does this — they come from
            # one dict) must fail loud here, not silently write a partial —
            # or, previously, an entirely empty — record.
            #
            # Checked as five explicit `is None` arms (not a loop/`any()`
            # over the tuple above) so basedpyright narrows each LOCAL to
            # `float` past this point — the tuple-membership form cannot
            # narrow, which is why the round() calls below previously each
            # needed a `# type: ignore[arg-type]` (CodeRabbit finding, P8
            # review round).
            if (
                field_nats_per_byte is None
                or field_brier is None
                or field_ece is None
                or field_entropy is None
                or field_top1 is None
            ):
                msg = (
                    "EvalResult carries some but not all of field_nats_per_byte/"
                    "field_brier/field_ece/field_entropy/field_top1 — the five "
                    "calibration keys must be set together"
                )
                raise RuntimeError(msg)
            record["field_nats_per_byte"] = round(field_nats_per_byte, 6)
            record["field_brier"] = round(field_brier, 6)
            record["field_ece"] = round(field_ece, 6)
            record["field_entropy"] = round(field_entropy, 6)
            record["field_top1"] = round(field_top1, 6)
        return record

    def fit(self, max_steps: int | None = None, verbose: bool = True) -> list[dict[str, Any]]:
        """Run the training loop; returns the list of per-step train metrics."""
        total = max_steps if max_steps is not None else self.cfg.train.max_steps
        history: list[dict[str, Any]] = []
        start = time.time()
        # Pairing evidence for the E1 preregistration: attest the data order
        # ONCE, on the first logged train record at/after the digest window
        # (or the final record of a shorter run) — and only for a FRESH run:
        # a resumed run has not observed batches 1..N, so it must not attest
        # to them (absence of the field = not attested, never "same order").
        fresh_start = self.step == 0
        order_attested = False

        run_start_record: dict[str, Any] = {
            "event": "run_start",
            "config": self.cfg.to_dict(),
            "params": self.raw_model.num_parameters(),
            "device": str(self.device),
            "tokenizer": self.meta.tokenizer,
            "bytes_per_token": self.meta.bytes_per_token,
            "train_tokens": self.meta.train_tokens,
            "val_tokens": self.meta.val_tokens,
        }
        # E1.5c: absent from the record entirely unless state_source ==
        # "field" (same "absent channel is absent" convention E1.5b's
        # gram_* eval keys use) — this trainer always builds the
        # matched-BACKBONE lane (whatever d_model the resolved config
        # names); the matched-TOTAL-parameter twins are separate, shrunk
        # configs (configs/e1-5c-*-matched-total.yaml, P8/FOR-925) that
        # report their own field_encoder_params the same way. RUNNING either
        # lane for real stays prereg-time scope (docs/e1_5-instruction.md
        # Phase D) — only the configs themselves are built here.
        field_encoder_params = self.raw_model.field_encoder_parameter_count()
        if field_encoder_params is not None:
            run_start_record["field_encoder_params"] = field_encoder_params
        self.log(run_start_record)

        while self.step < total:
            metrics = self.train_step()
            history.append(metrics)
            if self.step % self.cfg.train.log_interval == 0 or self.step == total:
                record = {
                    "event": "train",
                    **{k: round(v, 6) for k, v in metrics.items()},
                    "wall_s": round(time.time() - start, 3),
                }
                if (
                    fresh_start
                    and not order_attested
                    and (self.step >= DATA_ORDER_DIGEST_BATCHES or self.step == total)
                ):
                    attestation = self.train_sampler.data_order_digest()
                    if attestation is not None:
                        record["data_order_digest"], record["data_order_batches"] = attestation
                        order_attested = True
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
