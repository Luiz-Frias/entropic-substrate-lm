"""Standalone ``p_phi`` pretraining — the producer for E1.5c's frozen field.

``docs/e1_5-instruction.md`` §E1.5c Repair 1 names two regimes for the
contextual Fisher state: ``field_mode="joint"`` (``p_phi`` trains with the
backbone under the calibration CE) and ``field_mode="frozen"`` (``p_phi`` is
loaded from a checkpoint and frozen before the backbone ever trains over
``sqrt(p_phi).Q`` — the cleaner test, since a frozen field cannot become a
private code negotiated with the backbone). The frozen regime needs a
checkpoint that some OTHER run produced. This module is that run.

The wire format is not invented here: it is DEFINED by
:meth:`entropic_substrate.models.field_encoder.FieldEncoder._load_frozen_weights`
— ``{"format_version": CHECKPOINT_FORMAT_VERSION, "model":
<FieldEncoder.state_dict()>}``, read back through ``normalize_checkpoint``
with ``for_resume=False`` and ``weights_only=True``. This writer adds
``step`` and ``config`` (the same two extra keys
``Trainer.state_dict`` carries, for provenance); the loader reads only
``model``, and ``normalize_checkpoint`` passes the rest through untouched.

WHAT THIS LANE IS, in one line: ``FieldEncoder`` trained as its own small
causal byte LM, on the SAME corpus/split/batch machinery ``Trainer`` uses,
against the SAME loss the joint arm's calibration seam applies
(``Trainer._field_loss_terms``) — so a frozen ``p_phi`` and a jointly
calibrated one differ in WHEN they were trained, not in what they were
trained to do. The backbone is never built here: nothing in this lane
touches ``LanguageModel``.

The readout is field CE in nats. Under the byte tokenizer that IS
nats/byte (``bytes_per_token == 1.0`` by construction in
``data/build.py``), which is why this lane is byte-only twice over — once
because ``p_phi`` is defined over the 256-symbol alphabet, once because the
readout would otherwise need a conversion factor the lane never measures.

Non-negotiable numerics (CLAUDE.md): the CE runs in FP32 inside an explicit
``autocast(enabled=False)`` island, exactly as ``Trainer._field_loss_terms``
does — see :func:`field_cross_entropy`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from entropic_substrate.core.corpus.batching import SequenceSampler
from entropic_substrate.core.substrate.field_encoder import FieldEncoder
from entropic_substrate.historical.e_series.train import cosine_lr, resolve_device, seed_everything
from entropic_substrate.kernel.runtime.checkpoint import CHECKPOINT_FORMAT_VERSION
from entropic_substrate.kernel.runtime.config import EXPERIMENTS, Config, resolve_experiment_config
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.capture.adapters.jsonl import JsonlSink
from entropic_substrate.ladder.capture.custody.build import build_corpus

__all__ = [
    "FIELD_PRETRAIN_EVAL",
    "FIELD_PRETRAIN_RUN_START",
    "FieldPretrainEval",
    "FieldPretrainGap",
    "FieldPretrainResult",
    "add_field_pretrain_parser",
    "cmd_field_pretrain",
    "field_cross_entropy",
    "read_field_pretrain_series",
    "run_field_pretrain",
]

#: ``type`` discriminators for this lane's JSONL records. DELIBERATELY not
#: ``event`` (the key ``results/*.jsonl`` uses): ``telemetry.py`` is the only
#: admissibility predicate for BACKBONE run telemetry, and its ``EvalRecord``
#: schema requires ``loss``/``nats_per_byte`` keys this lane does not produce.
#: A different discriminator key means a field-pretrain file cannot be
#: half-parsed by those readers — it reads as zero eval records, and each
#: reader refuses an empty series loudly rather than returning one.
FIELD_PRETRAIN_RUN_START: str = "field_pretrain_run_start"
FIELD_PRETRAIN_EVAL: str = "field_pretrain_eval"

#: The tail window :attr:`FieldPretrainResult.last5_eval_ce_mean` averages.
#: Matches the last-k readout convention adopted after the E1 extension
#: verdict (a single final-step reading is threshold-fragile).
LAST_K_EVALS: int = 5


class FieldPretrainGap(Exception):
    """This lane's readout is absent or malformed.

    Its own type rather than ``telemetry.TelemetryGap``: that module is the
    single validated reader for BACKBONE ``results/*.jsonl`` and owns that
    boundary's vocabulary. Same posture though — deliberately not a value,
    so nothing downstream can mistake absence for a measurement.
    """


@dataclass(frozen=True, slots=True)
class FieldPretrainEval:
    """One admissible ``field_pretrain_eval`` record."""

    step: int
    #: Mean field cross-entropy over the eval batches, in nats. Byte
    #: tokenizer, so this is also nats/byte (see the module docstring).
    field_ce: float


@dataclass(frozen=True, slots=True)
class FieldPretrainResult:
    """What one pretrain run produced.

    ``encoder`` is the LIVE trained module, not a re-read of ``out_path``:
    a caller (and ``tests/test_field_pretrain.py``'s roundtrip test) can
    then compare what training produced against what the frozen loader
    reconstructs, which a file-vs-file comparison cannot do.
    """

    final_eval_ce: float
    #: Mean over the last :data:`LAST_K_EVALS` evals, or over ALL of them
    #: when the run produced fewer (a 20-step smoke has two).
    last5_eval_ce_mean: float
    steps: int
    out_path: Path
    encoder: FieldEncoder


def _refuse_incompatible(cfg: Config) -> None:
    """Refuse the two configurations this lane cannot mean anything for.

    Both are also refused one layer down (``FieldEncoder.__init__`` rejects
    a non-byte tokenizer; ``field_mode="frozen"`` without a path, or with
    one, would load weights this run is supposed to be producing) — but
    they are surfaced HERE, before ``build_corpus``, because that call is
    the expensive one: a gpt2 corpus build materialises and BPE-encodes the
    whole blob for a run that cannot exist.
    """
    if cfg.model.tokenizer != "byte":
        msg = (
            "field pretrain is defined over the 256-symbol byte alphabet; "
            f"tokenizer={cfg.model.tokenizer!r} has no 256-way next-byte target for "
            "p_phi to predict (FieldEncoder refuses it too, one layer down)"
        )
        raise ValueError(msg)
    if cfg.model.field_mode == "frozen":
        msg = (
            "field_mode='frozen' cannot be pretrained — a frozen field loads a "
            "checkpoint of a p_phi that a run like THIS one already produced, so "
            "pretraining one is a contradiction; set field_mode='joint' (and clear "
            "field_ckpt_path) on the config this lane trains from"
        )
        raise ValueError(msg)


def autocast_context(
    device: torch.device, amp_dtype: Literal["bf16", "fp16", "off"]
) -> torch.autocast:
    """Ambient autocast for one step — mirrors ``Trainer._autocast`` exactly.

    Including the cpu-disables-amp rule: CPU autocast is bf16-only and the
    CPU path here exists for smoke runs, where the extra dtype churn buys
    nothing. Mirrored rather than shared because ``Trainer._autocast`` is a
    method on a trainer that owns a backbone, a corpus and an optimiser —
    none of which this lane builds.

    Nothing measurable rides on it TODAY: the fp32 island in
    :func:`field_cross_entropy` nests inside this context and, in this
    lane, that island covers the entire gradient path by construction (the
    encoder forward AND the CE), so ``amp_dtype`` cannot change this run's
    numerics. It is kept because the step's dtype policy must be the
    trainer's, not an accident of what this lane happens to compute — the
    same reason ``_field_loss_terms``' island binds unconditionally rather
    than only under bf16.
    """
    if amp_dtype == "off" or device.type == "cpu":
        return torch.autocast(device_type=device.type, enabled=False)
    torch_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=torch_dtype)


def field_cross_entropy(
    encoder: FieldEncoder, x: torch.Tensor, y: torch.Tensor, *, device_type: str
) -> torch.Tensor:
    """``CE(field_logits(x), y)`` — the SAME loss the joint arm calibrates with.

    Convention copied from ``Trainer._field_loss_terms`` /
    ``_field_ce_from_logits``, deliberately down to the details: ``p_phi``
    consumes ``x`` only (``field_logits`` takes no ``targets`` parameter —
    the structural anti-leakage pin), ``y`` scores it ELEMENTWISE with no
    slice (``y[:, t] == x[:, t+1]`` by ``SequenceSampler``'s construction,
    which is exactly the byte ``p_phi(.|x_<=t)`` predicts), and the whole
    thing runs FP32 inside an explicit ``autocast(enabled=False)`` island
    regardless of the caller's ambient autocast (CLAUDE.md non-negotiable
    numerics). Nesting inside an already-disabled context is a harmless
    no-op — the context managers compose.

    Module-level, not a private method, for two reasons: it is the seam
    ``tests/test_field_pretrain.py`` poisons to prove the non-finite guards
    are load-bearing, and it is the one place a future divergence from
    ``_field_loss_terms`` would have to be written down.
    """
    with torch.autocast(device_type=device_type, enabled=False):
        logits = encoder.field_logits(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def _require_finite(loss: torch.Tensor, phase: str, step: int) -> float:
    """Scalar value of ``loss``, or raise.

    A measurement path fails LOUD, never open (CLAUDE.md): a NaN train loss
    poisons every parameter on the next optimiser step, and a NaN eval CE
    is a readout that compares false against every threshold while looking
    like a number. ``phase`` is part of the message on purpose — the two
    guards are distinguishable in a test's ``pytest.raises(match=...)``, so
    deleting one cannot be masked by the other firing later in the run.
    """
    value = float(loss.detach())
    if not math.isfinite(value):
        msg = f"field pretrain: non-finite {phase} loss {value!r} at step {step}"
        raise RuntimeError(msg)
    return value


def _build_optimizer(encoder: FieldEncoder, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with ``Trainer._build_optimizer``'s decay/no-decay split.

    Same rule, same reason: no weight decay on 1-D tensors (biases and norm
    gains). ``parameters()`` de-duplicates the tied ``head.weight`` /
    ``wte.weight`` pair by identity, so the tied table is decayed once.
    """
    decay, no_decay = [], []
    for param in encoder.parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 else decay).append(param)
    groups: list[dict[str, Any]] = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.train.lr, betas=cfg.train.betas, eps=1e-8)


def _set_lr(optimizer: torch.optim.Optimizer, cfg: Config, step: int) -> float:
    """Cosine schedule with warmup — ``train.cosine_lr``, imported not copied.

    ``step`` is the PRE-increment (0-based) step, matching
    ``Trainer._set_lr``'s ordering so a pretrain run and a backbone run of
    the same length see the same LR at the same step index.
    """
    lr = cosine_lr(
        step,
        base_lr=cfg.train.lr,
        warmup=cfg.train.warmup_steps,
        total=cfg.train.max_steps,
        min_ratio=cfg.train.min_lr_ratio,
    )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _evaluate(
    encoder: FieldEncoder,
    sampler: SequenceSampler,
    *,
    num_batches: int,
    device: torch.device,
    amp_dtype: Literal["bf16", "fp16", "off"],
    step: int,
) -> float:
    """Mean field CE over ``num_batches`` deterministic held-out batches.

    ``reset_eval()`` first, exactly as ``Trainer.evaluate`` does: the eval
    sampler walks a fixed grid, so rewinding it makes every eval in a run
    (and across runs) score the same bytes.
    """
    encoder.eval()
    sampler.reset_eval()
    losses: list[float] = []
    with torch.no_grad():
        for x, y in sampler.iter_batches(num_batches):
            with autocast_context(device, amp_dtype):
                loss = field_cross_entropy(encoder, x, y, device_type=device.type)
            losses.append(_require_finite(loss, "eval", step))
    encoder.train()
    return float(np.mean(losses))


def run_field_pretrain(
    cfg: Config,
    *,
    out_path: Path,
    results_path: Path,
    device: str = "auto",
) -> FieldPretrainResult:
    """Train ``cfg``'s :class:`FieldEncoder` as its own byte LM and save it.

    Args:
        cfg: the experiment config. Only the ``field_*`` knobs, ``d_model``
            (the lift's output width — a frozen consumer with a different
            ``d_model`` gets a shape error from ``load_state_dict``, which
            is the intended refusal), and the data/train blocks are read;
            ``state_source`` is NOT consulted, since this lane builds the
            encoder directly and never builds a backbone to inject into.
        out_path: checkpoint destination, in the format
            ``FieldEncoder._load_frozen_weights`` defines.
        results_path: JSONL destination for this lane's own records (see
            :data:`FIELD_PRETRAIN_EVAL`).
        device: ``"auto"`` defers to ``cfg.train.device`` (itself possibly
            ``"auto"``, which probes ``ENTROPIC_SUBSTRATE_DEVICE`` then
            hardware); anything else overrides the config.

    Evaluation runs every ``cfg.train.eval_interval`` steps AND at the final
    step — including when ``eval_interval == 0``, which in ``Trainer.fit``
    disables eval entirely. Deliberate divergence: a backbone run's product
    is the checkpoint and eval is telemetry, whereas THIS run's product is a
    checkpoint whose only quality readout is that CE, so a run without one
    would be unusable by the arm that consumes it.
    """
    _refuse_incompatible(cfg)
    seed_everything(cfg.train.seed)
    resolved = resolve_device(cfg.train.device if device == "auto" else device)

    corpus = build_corpus(cfg.data, cfg.model.tokenizer)
    if corpus.meta.bytes_per_token != 1.0:
        # Unreachable given the byte-tokenizer refusal above (build_corpus
        # pins 1.0 for the byte path) — restated as a real control-flow
        # check rather than a comment, because this lane reports CE in nats
        # and calls it nats/byte. A `assert` would vanish under `python -O`
        # (bandit B101, and this repo's `_require_loss` rationale).
        msg = (
            f"field pretrain: bytes_per_token is {corpus.meta.bytes_per_token}, not 1.0 — "
            "this lane reports field CE in nats AS nats/byte, which only holds for "
            "byte tokenization"
        )
        raise RuntimeError(msg)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = FieldEncoder(cfg.model).to(resolved)
    train_sampler = SequenceSampler(
        corpus.train,
        cfg.data.batch_size,
        cfg.data.seq_len,
        train=True,
        seed=cfg.train.seed,
        device=resolved,
    )
    val_sampler = SequenceSampler(
        corpus.val,
        cfg.data.batch_size,
        cfg.data.seq_len,
        train=False,
        seed=cfg.train.seed + 1,
        device=resolved,
    )
    optimizer = _build_optimizer(encoder, cfg)
    sink = JsonlSink(results_path)
    amp_dtype = cfg.train.amp_dtype
    total = cfg.train.max_steps
    field_encoder_params = sum(p.numel() for p in encoder.parameters())

    sink.log(
        {
            "type": FIELD_PRETRAIN_RUN_START,
            "run": cfg.name,
            "config": cfg.to_dict(),
            "seed": cfg.train.seed,
            "device": str(resolved),
            "max_steps": total,
            "field_encoder_params": field_encoder_params,
            "bytes_per_token": corpus.meta.bytes_per_token,
            "out_path": str(out_path),
        }
    )
    logger.info(
        f"[{cfg.name}] field pretrain: {field_encoder_params / 1e6:.3f}M p_phi params on "
        f"{resolved}; {total} steps over {corpus.meta.train_tokens} train bytes"
    )

    eval_ces: list[float] = []
    start = time.time()
    step = 0
    encoder.train()
    while step < total:
        lr = _set_lr(optimizer, cfg, step)
        x, y = train_sampler.batch()
        with autocast_context(resolved, amp_dtype):
            loss = field_cross_entropy(encoder, x, y, device_type=resolved.type)
        step += 1
        loss_value = _require_finite(loss, "train", step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(encoder.parameters(), cfg.train.grad_clip)
        optimizer.step()

        if step % cfg.train.log_interval == 0 or step == total:
            logger.info(
                f"step {step:>6} field_ce {loss_value:.4f} "
                f"lr {lr:.2e} grad_norm {float(grad_norm):.3f}"
            )
        if (cfg.train.eval_interval > 0 and step % cfg.train.eval_interval == 0) or step == total:
            eval_ce = _evaluate(
                encoder,
                val_sampler,
                num_batches=cfg.data.num_eval_batches,
                device=resolved,
                amp_dtype=amp_dtype,
                step=step,
            )
            eval_ces.append(eval_ce)
            # NOT rounded, unlike `Trainer._eval_record`'s display rounding:
            # this file is the lane's only readout, and a consumer comparing
            # a returned FieldPretrainResult against the series must see the
            # same float the run measured, not a 6-dp rendering of it.
            sink.log({"type": FIELD_PRETRAIN_EVAL, "step": step, "field_ce": eval_ce})
            logger.info(f"  eval  step {step:>6} field_ce {eval_ce:.4f} (nats/byte)")

    if not eval_ces:
        # Unreachable: the loop always evaluates at `step == total`. Stated
        # as control flow anyway, because the alternative is a `mean([])`
        # NaN reaching the caller as a readout (same fail-loud posture as
        # `FieldCalibrationAccumulator.finalize`'s empty-accumulator check).
        msg = f"field pretrain: {total} steps produced no eval — nothing to report"
        raise RuntimeError(msg)

    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "step": step,
            "model": encoder.state_dict(),
            "config": cfg.to_dict(),
        },
        out_path,
    )
    tail = eval_ces[-LAST_K_EVALS:]
    logger.info(
        f"[{cfg.name}] field pretrain done in {time.time() - start:.1f}s; "
        f"final field_ce {eval_ces[-1]:.4f} nats/byte -> {out_path}"
    )
    return FieldPretrainResult(
        final_eval_ce=eval_ces[-1],
        last5_eval_ce_mean=float(np.mean(tail)),
        steps=step,
        out_path=out_path,
        encoder=encoder,
    )


# ------------------------------------------------------------------- reader
def _eval_step(path: Path, record: dict[str, Any], lineno: int) -> int:
    value = record.get("step")
    # bool is an int subclass, so a JSON `true` would otherwise read as
    # step 1 (telemetry.py's own rationale for the same check).
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{path}:{lineno}: eval record 'step' is not an int: {value!r}"
        raise FieldPretrainGap(msg)
    return value


def _eval_field_ce(path: Path, record: dict[str, Any], lineno: int) -> float:
    if "field_ce" not in record:
        msg = f"{path}:{lineno}: eval record carries no 'field_ce'"
        raise FieldPretrainGap(msg)
    value = record["field_ce"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{path}:{lineno}: 'field_ce' is not numeric: {value!r}"
        raise FieldPretrainGap(msg)
    if not math.isfinite(value):
        # `json` parses bare NaN/Infinity back without complaint, so a
        # numerically failed run would otherwise read as a valid CE.
        msg = f"{path}:{lineno}: non-finite 'field_ce': {value!r}"
        raise FieldPretrainGap(msg)
    if value < 0.0:
        # Cross-entropy is non-negative; a finite-but-impossible reading is
        # corruption the same way NaN is (telemetry.py's documented-domain
        # rule), just not caught by the finiteness check alone.
        msg = f"{path}:{lineno}: 'field_ce' {value!r} is below its documented minimum 0.0"
        raise FieldPretrainGap(msg)
    return float(value)


def read_field_pretrain_series(path: str | Path) -> list[FieldPretrainEval]:
    """Every admissible ``field_pretrain_eval`` record in ``path``, in file order.

    The validated read side of this lane, mirroring ``telemetry.py``'s
    posture for backbone runs: an inadmissible series is a
    :class:`FieldPretrainGap`, never a value — no empty list on a missing
    file, no skipped record on a malformed one. A series with zero eval
    records is inadmissible too: a run that never evaluated produced no
    readout, and returning ``[]`` would let a caller average nothing.
    """
    path = Path(path)
    if not path.exists():
        msg = f"{path}: does not exist — no field-pretrain series to read"
        raise FieldPretrainGap(msg)
    records: list[FieldPretrainEval] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"{path}:{lineno}: not valid JSON ({exc})"
                raise FieldPretrainGap(msg) from exc
            if not isinstance(record, dict):
                msg = f"{path}:{lineno}: record is {type(record).__name__}, not a mapping"
                raise FieldPretrainGap(msg)
            if record.get("type") != FIELD_PRETRAIN_EVAL:
                continue
            records.append(
                FieldPretrainEval(
                    step=_eval_step(path, record, lineno),
                    field_ce=_eval_field_ce(path, record, lineno),
                )
            )
    if not records:
        msg = f"{path}: no {FIELD_PRETRAIN_EVAL} records — the run never evaluated"
        raise FieldPretrainGap(msg)
    return records


# ---------------------------------------------------------------------- cli
def add_field_pretrain_parser(
    # argparse exports no public type for what `add_subparsers()` returns;
    # MEASURED 2026-08-20 (basedpyright 1.37, strict): naming it raises
    # `"_SubParsersAction" is private and used outside of the module in which
    # it is declared`. Suppressed rather than widened to `Any` — the ignore
    # keeps `sub.add_parser` type-checked at the one call site below.
    sub: argparse._SubParsersAction[argparse.ArgumentParser],  # pyright: ignore[reportPrivateUsage]
) -> argparse.ArgumentParser:
    """Register the ``field-pretrain`` subcommand on ``sub``.

    Lives here rather than inline in ``cli.py`` so the subcommand and its
    handler stay in one file with the lane they drive — ``cli.py`` wires it
    with one call and one handler-table entry.
    """
    parser = sub.add_parser(
        "field-pretrain",
        help="train a standalone p_phi (FieldEncoder) and save it for field_mode='frozen'",
    )
    parser.add_argument("--exp", choices=EXPERIMENTS, help="experiment ladder rung")
    parser.add_argument("--config", type=Path, help="explicit YAML config path")
    parser.add_argument("--steps", type=int, default=None, help="override max_steps")
    parser.add_argument("--seed", type=int, default=None, help="override the seed")
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:N | mps | auto")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="checkpoint destination (the path a frozen arm's field_ckpt_path points at)",
    )
    parser.add_argument(
        "--results", type=Path, required=True, help="JSONL destination for the CE series"
    )
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    """``--steps``/``--seed``/``--device`` as a config overlay.

    A local copy of ``cli._overrides`` minus ``--resume`` (this lane has no
    resume path) — importing ``cli`` here would be circular the moment
    ``cli`` imports this module.
    """
    train: dict[str, Any] = {}
    if args.device is not None:
        train["device"] = args.device
    if args.steps is not None:
        train["max_steps"] = args.steps
    if args.seed is not None:
        train["seed"] = args.seed
    return {"train": train} if train else {}


def cmd_field_pretrain(args: argparse.Namespace) -> int:
    """Handle ``field-pretrain``."""
    if args.exp is None and args.config is None:
        logger.error("one of --exp or --config is required")
        return 2
    cfg = resolve_experiment_config(args.exp, args.config, overrides=_overrides(args))
    result = run_field_pretrain(cfg, out_path=args.out, results_path=args.results)
    logger.bind(report=True).info(
        f"field pretrain: {result.steps} steps, final field_ce "
        f"{result.final_eval_ce:.6f} nats/byte (last-{LAST_K_EVALS} mean "
        f"{result.last5_eval_ce_mean:.6f}) -> {result.out_path}"
    )
    return 0
