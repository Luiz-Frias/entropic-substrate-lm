"""Exploratory E1.5c head audit: is the frozen field head's 128 -> 256 map
injective on paper, and is it invertible in practice on the probabilities the
P arm actually consumed?

``docs/verdicts/e1-5c-frozen-verdict.md`` §3 measures P-H at -0.0508
nats/byte. The compression reading of that gap — "P hands the backbone a
256-way distribution squeezed out of a 128-dim hidden state, H hands it the
state itself, so P must have lost something" — is a hypothesis about the
frozen head, and this module is the smallest diagnostic that can bear on it.
It answers three separable questions and never merges them:

* :func:`head_spectrum` — the head's own conditioning. Softmax is
  shift-invariant (``softmax(l) == softmax(l + c*1)``), so injectivity of
  ``h -> softmax(W @ h)`` is governed by the ROW-CENTERED head
  ``(I - 11^T/256) @ W`` (:func:`center_logit_columns`), never by raw ``W``:
  a ``W`` carrying ``1_256`` in its column span is full rank and still has a
  one-dimensional kernel in the softmax quotient. Both spectra are reported,
  and every injectivity readout (``centered_rank``, ``sigma_min_centered``,
  ``condition_number_centered``) is the centered one.
* :func:`reconstruction_report` — practical invertibility. Analytic
  injectivity says nothing about a pipeline that computes softmax in FP32 and
  then applies ``clamp_min(1e-12)`` (``field_encoder._encode``); the clamp
  overwrites every tail logit it touches. The report therefore carries the
  clamped-position fraction beside the reconstruction error, so a small error
  is never read as "invertible" when the clamp never fired, nor a large one as
  "the head is rank-deficient" when it did.
* :func:`realized_spectrum` — how many state directions the DATA actually
  excites. "128 dimensions through a 256-way head" overstates the squeeze if
  the realized states live on a far lower-dimensional set.

Exploratory posture (CLAUDE.md, E1.5-D lane): checkpoints and CPU only, no
training, no registered gate — this feeds judgment about the verdict's causal
ladder, it does not decide anything.

Non-negotiable numerics (CLAUDE.md): every reduction here runs in FP64 inside
an explicit ``torch.autocast(enabled=False)`` block regardless of ambient
autocast, mirroring ``analysis/embedding_geometry.py``. This module runs
offline, so the precision costs nothing and buys headroom for two steep
operations: the ``log`` of a probability sitting on the 1e-12 floor, and a
least-squares solve whose smallest centered singular value may be near zero.

Fail-loud posture: this is a measurement path, so an empty or mismatched
input is refused rather than reduced to a silent ``nan``/``0.0``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from entropic_substrate.checkpoint import CheckpointFormatError, normalize_checkpoint
from entropic_substrate.models.field_encoder import (
    _PROB_EPS,  # pyright: ignore[reportPrivateUsage]  # see PROB_FLOOR below
    FieldEncoder,
)

__all__ = [
    "PROB_FLOOR",
    "FieldHeadAudit",
    "FieldStates",
    "HeadSpectrum",
    "RealizedSpectrum",
    "ReconstructionReport",
    "audit_field_head",
    "center_logit_columns",
    "collect_field_states",
    "head_spectrum",
    "load_frozen_head",
    "realized_spectrum",
    "reconstruct_hidden",
    "reconstruction_report",
]

# The pipeline's own post-softmax floor, IMPORTED rather than restated: clamp
# detection below asks "did this position hit the floor the encoder clamps
# with", and a local copy of the literal would keep answering for a floor the
# encoder no longer uses if `field_encoder._PROB_EPS` ever moves.
PROB_FLOOR: float = _PROB_EPS

# Rank tolerance scale. FP32, not FP64, even though every reduction here runs
# in FP64: the tensors entering this module are fp32 checkpoint weights and
# fp32 activations, so their trailing singular values carry fp32 rounding
# (~1e-5 on a 256x16 outer product) that an fp64-eps tolerance (~1e-12) would
# count as signal. The tolerance is the numpy/LAPACK convention
# `sigma_max * max(m, n) * eps` with eps taken at the INPUT's precision.
_RANK_EPS: float = torch.finfo(torch.float32).eps

# Relative-error denominator floor. A position whose hidden state is exactly
# zero has no scale to be relative to; its numerator is zero as well, so this
# floor reports 0.0 there rather than 0/0. It is NOT a general degeneracy
# guard — an empty or mismatched input is refused outright below.
_NORM_EPS: float = 1e-12

_N_BYTES: int = 256


def center_logit_columns(head: torch.Tensor) -> torch.Tensor:
    """``(I - 11^T/n) @ W`` in FP64: the head as the softmax actually sees it.

    Adding any multiple of ``1_n`` to a logit vector is a softmax no-op, so
    the component of each column along ``1_n`` is unobservable downstream.
    Projecting it out is what makes "rank" and "condition number" statements
    about ``W`` statements about the map ``h -> softmax(W @ h)``.
    """
    if head.ndim != 2:
        msg = f"center_logit_columns: expected a 2-D (n_symbols, d) head, got {tuple(head.shape)}"
        raise ValueError(msg)
    with torch.autocast(device_type=head.device.type, enabled=False):
        w = head.detach().double()
        return w - w.mean(dim=0, keepdim=True)


def _rank_and_effective_rank(singular_values: torch.Tensor) -> tuple[int, float, int]:
    """``(rank, entropy effective rank, n_values)`` for a descending spectrum.

    Effective rank is ``exp`` of the spectral entropy over the l1-normalized
    singular values — the count itself when the spectrum is flat, and lower
    the more mass concentrates in the leading directions. ``torch.special.entr``
    rather than ``p * p.log()``: a rank-deficient input normalizes exact zeros,
    and ``0 * log(0)`` is ``nan`` while ``entr(0)`` is ``0``.
    """
    n = int(singular_values.numel())
    sigma_max = float(singular_values[0]) if n else 0.0
    tol = sigma_max * max(n, _N_BYTES) * _RANK_EPS
    rank = int((singular_values > tol).sum())
    total = singular_values.sum()
    if float(total) <= 0.0:
        return 0, 0.0, n
    p = singular_values / total
    return rank, float(torch.exp(torch.special.entr(p).sum())), n


def _condition_number(singular_values: torch.Tensor) -> float:
    """``sigma_max / sigma_min``, divided as TENSORS on purpose: an exactly
    singular input (the ones-direction head below is one) makes this ``inf``,
    the honest value, where a Python-float division raises ZeroDivisionError.
    """
    return float(singular_values[0] / singular_values[-1])


@dataclasses.dataclass(frozen=True, slots=True)
class HeadSpectrum:
    """Raw and softmax-quotient spectra of a tied 256-way head.

    The ``_centered`` fields are the ones that speak to injectivity of
    ``h -> softmax(W @ h)``; their raw counterparts are reported beside them
    precisely so the two can be seen to disagree (they do, whenever the head
    carries a component along the all-ones direction).
    """

    singular_values: torch.Tensor
    centered_singular_values: torch.Tensor
    rank: int
    centered_rank: int
    effective_rank: float
    effective_rank_centered: float
    sigma_min: float
    sigma_min_centered: float
    condition_number: float
    condition_number_centered: float


def head_spectrum(head: torch.Tensor) -> HeadSpectrum:
    """SVD the head both raw and row-centered, in FP64, descending.

    ``torch.linalg.svdvals`` returns singular values in descending order, so
    ``singular_values[0]``/``[-1]`` are the extremes without a sort.
    """
    centered = center_logit_columns(head)
    with torch.autocast(device_type=head.device.type, enabled=False):
        raw_s = torch.linalg.svdvals(head.detach().double())
        centered_s = torch.linalg.svdvals(centered)
    rank, eff, _ = _rank_and_effective_rank(raw_s)
    centered_rank, centered_eff, _ = _rank_and_effective_rank(centered_s)
    return HeadSpectrum(
        singular_values=raw_s,
        centered_singular_values=centered_s,
        rank=rank,
        centered_rank=centered_rank,
        effective_rank=eff,
        effective_rank_centered=centered_eff,
        sigma_min=float(raw_s[-1]),
        sigma_min_centered=float(centered_s[-1]),
        condition_number=_condition_number(raw_s),
        condition_number_centered=_condition_number(centered_s),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class RealizedSpectrum:
    """Spectrum of the hidden states the data actually produced.

    Uncentered: the head consumes the state vectors themselves, not their
    deviations from a batch mean, so the subspace this measures is the one the
    head had to preserve.
    """

    singular_values: torch.Tensor
    rank: int
    effective_rank: float
    n_positions: int


def realized_spectrum(hidden: torch.Tensor) -> RealizedSpectrum:
    """FP64 spectrum of an ``(n_positions, d)`` stack of hidden states."""
    if hidden.ndim != 2:
        msg = f"realized_spectrum: expected a 2-D (n_positions, d) stack, got {tuple(hidden.shape)}"
        raise ValueError(msg)
    n = int(hidden.shape[0])
    if n < 2:
        msg = f"realized_spectrum: need at least 2 positions to span anything, got {n}"
        raise ValueError(msg)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        s = torch.linalg.svdvals(hidden.detach().double())
    rank, eff, _ = _rank_and_effective_rank(s)
    return RealizedSpectrum(singular_values=s, rank=rank, effective_rank=eff, n_positions=n)


def _check_probs(probs: torch.Tensor, head: torch.Tensor, *, fn: str) -> None:
    if probs.ndim != 2:
        msg = f"{fn}: expected 2-D (n_positions, n_symbols) probabilities, got {tuple(probs.shape)}"
        raise ValueError(msg)
    if probs.shape[1] != head.shape[0]:
        msg = (
            f"{fn}: probabilities are over {probs.shape[1]} symbols but the head emits "
            f"{head.shape[0]} — these are not the same distribution"
        )
        raise ValueError(msg)


def reconstruct_hidden(probs: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    """Least-squares ``h`` from ``p``, through the CENTERED head, in FP64.

    ``log p = W h - logsumexp(W h)``, and the ``logsumexp`` term is a multiple
    of ``1_n``, so row-centering the log-probabilities kills it exactly:
    ``center(log p) = (I - 11^T/n) W h``. The pseudo-inverse of the centered
    head then recovers ``h`` up to whatever the fp32 softmax and the 1e-12
    clamp destroyed — which is the quantity :func:`reconstruction_report`
    exists to size. ``pinv`` rather than ``lstsq``: it returns the
    minimum-norm solution on a rank-deficient centered head instead of a
    driver-dependent one.
    """
    _check_probs(probs, head, fn="reconstruct_hidden")
    centered = center_logit_columns(head)
    with torch.autocast(device_type=probs.device.type, enabled=False):
        log_p = probs.detach().double().clamp_min(PROB_FLOOR).log()
        target = log_p - log_p.mean(dim=-1, keepdim=True)
        return target @ torch.linalg.pinv(centered).T


@dataclasses.dataclass(frozen=True, slots=True)
class ReconstructionReport:
    """How well ``h`` comes back out of the probabilities, and how much of the
    input had already hit the pipeline's clamp before the question was asked.

    The two must be read together: ``median_rel_error`` alone cannot tell an
    ill-conditioned head from a well-conditioned one whose tail logits were
    overwritten by ``clamp_min(PROB_FLOOR)``, and ``clamped_fraction`` alone
    says nothing about how much information the clamp actually cost.
    """

    n_positions: int
    clamped_fraction: float
    median_rel_error: float
    mean_rel_error: float
    max_rel_error: float


def reconstruction_report(
    hidden: torch.Tensor, probs: torch.Tensor, head: torch.Tensor
) -> ReconstructionReport:
    """Reconstruct ``hidden`` from ``probs`` and report the error + clamp rate.

    Raises:
        ValueError: ``hidden`` and ``probs`` disagree on the position count
            (they would not be the same positions), or there are zero
            positions — a measurement path reports nothing rather than a
            silent ``nan`` over an empty reduction (CLAUDE.md failure
            semantics).
    """
    if hidden.ndim != 2:
        msg = (
            "reconstruction_report: expected a 2-D (n_positions, d) hidden stack, "
            f"got {tuple(hidden.shape)}"
        )
        raise ValueError(msg)
    _check_probs(probs, head, fn="reconstruction_report")
    if hidden.shape[0] != probs.shape[0]:
        msg = (
            f"reconstruction_report: {hidden.shape[0]} hidden positions vs "
            f"{probs.shape[0]} probability positions — not the same positions"
        )
        raise ValueError(msg)
    n = int(hidden.shape[0])
    if n == 0:
        msg = (
            "reconstruction_report: zero positions — there is nothing to measure, and "
            "an empty reduction would report nan/0.0 as if something had been measured"
        )
        raise ValueError(msg)

    h_hat = reconstruct_hidden(probs, head)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        h = hidden.detach().double()
        # Inclusive comparison: the clamp writes the floor VALUE, so a clamped
        # entry sits AT it, not below it.
        clamped = (probs.detach().double().min(dim=-1).values <= PROB_FLOOR).double()
        rel = (h_hat - h).norm(dim=1) / h.norm(dim=1).clamp_min(_NORM_EPS)
    return ReconstructionReport(
        n_positions=n,
        clamped_fraction=float(clamped.mean()),
        median_rel_error=float(rel.median()),
        mean_rel_error=float(rel.mean()),
        max_rel_error=float(rel.max()),
    )


def load_frozen_head(path: str | Path) -> torch.Tensor:
    """The tied ``(256, field_d_model)`` head out of a field checkpoint.

    ``FieldEncoder`` ties ``head.weight`` to ``wte.weight`` (one storage, two
    names), so a payload may carry either key or both. Both-and-different is
    refused: the tie is what makes "the head" a well-defined object, and
    picking one tensor of a broken pair would audit a head no run ever used.

    Routed through :func:`~entropic_substrate.checkpoint.normalize_checkpoint`
    like every other checkpoint reader in this repo, so a malformed payload
    fails as the same typed refusal rather than an ``AttributeError`` further
    down.

    Raises:
        CheckpointFormatError: not loadable as a torch checkpoint, or refused
            by the validated reader.
        RuntimeError: the payload carries no head at all, an untied
            ``head.weight``/``wte.weight`` pair, or a non-mapping ``model``.
    """
    path = Path(path)
    try:
        # map_location="cpu" / weights_only=True mirror Trainer.load_checkpoint:
        # a checkpoint captured on a training pod must load on an analysis
        # machine with no GPU, and self-produced checkpoints hold only tensors
        # and primitives (bandit B614).
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except CheckpointFormatError:
        raise
    except Exception as exc:  # any load failure, re-raised as a named refusal
        msg = f"{path}: could not be loaded as a torch checkpoint ({exc})"
        raise CheckpointFormatError(msg) from exc
    state = normalize_checkpoint(raw, source=str(path), for_resume=False)
    model_state = state.get("model")
    if not isinstance(model_state, dict):
        msg = f"{path}: checkpoint 'model' is not a state dict"
        raise RuntimeError(msg)
    head = model_state.get("head.weight")
    wte = model_state.get("wte.weight")
    if head is not None and wte is not None and not torch.equal(head, wte):
        msg = (
            f"{path}: head.weight and wte.weight differ — FieldEncoder's head is tied to "
            "its embedding (one storage), so this payload is not a tied field head and "
            "auditing either tensor would audit a head no run used"
        )
        raise RuntimeError(msg)
    weight = head if head is not None else wte
    if weight is None:
        msg = (
            f"{path}: checkpoint carries neither 'head.weight' nor 'wte.weight' — "
            "there is no field head in it to audit"
        )
        raise RuntimeError(msg)
    return weight


@dataclasses.dataclass(frozen=True, slots=True)
class FieldStates:
    """The head's own input and output, over a set of evaluation batches.

    ``hidden`` is POST-``norm_f`` — the exact tensor ``field_encoder._encode``
    hands to ``self.head`` (and, on the H leg, to ``state_lift``), so
    ``hidden @ head.weight.T`` reproduces the encoder's logits. ``probs`` is
    reconstructed the way ``_encode`` does it for the F/P legs: FP32 softmax,
    then ``clamp_min(PROB_FLOOR)``.
    """

    hidden: torch.Tensor
    probs: torch.Tensor


def collect_field_states(encoder: FieldEncoder, batches: Iterable[torch.Tensor]) -> FieldStates:
    """Run ``encoder`` over ``batches`` and capture ``(hidden, probs)``.

    The hidden state is captured with a forward hook on ``encoder.norm_f``
    rather than recomputed: ``_encode`` is private and its body is the thing
    under audit, so re-deriving ``h`` here would audit this module's
    reimplementation instead of the encoder's own pass. The hook fires because
    ``FieldEncoder.forward`` reaches ``norm_f`` through ``nn.Module.__call__``
    (the same property ``tests/test_field_encoder.py``'s anti-leakage hook test
    depends on).

    ``encoder.eval()``: a diagnostic must be deterministic, and a
    dropout-carrying config would otherwise give a different ``h`` on every
    call.

    Raises:
        ValueError: no batches, or a batch that is not ``(B, T)`` byte ids —
            an empty capture would hand every downstream reduction a
            zero-position input.
    """
    batch_list: Sequence[torch.Tensor] = list(batches)
    if not batch_list:
        msg = (
            "collect_field_states: no batches — there is nothing to capture, and an "
            "empty capture would silently produce a zero-position audit"
        )
        raise ValueError(msg)

    captured: list[torch.Tensor] = []
    hidden_chunks: list[torch.Tensor] = []
    prob_chunks: list[torch.Tensor] = []

    def _hook(_module: torch.nn.Module, _args: object, output: torch.Tensor) -> None:
        captured.append(output.detach())

    was_training = encoder.training
    encoder.eval()
    handle = encoder.norm_f.register_forward_hook(_hook)  # pyright: ignore[reportUnknownMemberType]
    try:
        with torch.no_grad():
            for idx in batch_list:
                if idx.ndim != 2:
                    msg = (
                        "collect_field_states: expected (B, T) byte ids, got shape "
                        f"{tuple(idx.shape)}"
                    )
                    raise ValueError(msg)
                captured.clear()
                logits = encoder(idx).logits
                if len(captured) != 1:
                    msg = (
                        "collect_field_states: the norm_f hook fired "
                        f"{len(captured)} times for one forward pass — the capture is "
                        "not the head's own input"
                    )
                    raise RuntimeError(msg)
                # Same FP32-island construction as field_encoder._encode's F/P
                # legs, and the same row-major flatten `logits.reshape(-1, 256)`
                # uses, so hidden row i and prob row i are one position.
                with torch.autocast(device_type=idx.device.type, enabled=False):
                    probs = torch.softmax(logits.float(), dim=-1).clamp_min(PROB_FLOOR)
                hidden_chunks.append(captured[0].float().reshape(-1, encoder.field_d_model))
                prob_chunks.append(probs.reshape(-1, probs.shape[-1]))
    finally:
        handle.remove()
        encoder.train(was_training)
    return FieldStates(hidden=torch.cat(hidden_chunks), probs=torch.cat(prob_chunks))


@dataclasses.dataclass(frozen=True, slots=True)
class FieldHeadAudit:
    """The three diagnostics side by side, over one set of positions.

    Deliberately not collapsed into a verdict: "P lost information to the
    head" needs a rank-deficient CENTERED spectrum, a large reconstruction
    error, AND a realized spectrum that fills the state's dimensions. Any one
    of the three alone has a benign reading.
    """

    spectrum: HeadSpectrum
    reconstruction: ReconstructionReport
    realized: RealizedSpectrum
    n_positions: int


def audit_field_head(encoder: FieldEncoder, batches: Iterable[torch.Tensor]) -> FieldHeadAudit:
    """Capture states once, then run all three diagnostics against them.

    The head is read off the live module (``encoder.head.weight``) rather than
    a checkpoint, so the matrix audited is the one that produced these very
    probabilities; :func:`load_frozen_head` is the checkpoint-side entry point
    for auditing a head without instantiating its encoder.
    """
    states = collect_field_states(encoder, batches)
    head = encoder.head.weight.detach()
    return FieldHeadAudit(
        spectrum=head_spectrum(head),
        reconstruction=reconstruction_report(states.hidden, states.probs, head),
        realized=realized_spectrum(states.hidden),
        n_positions=int(states.hidden.shape[0]),
    )
