"""Checkpoint format versioning, and the ``tau`` semantics change it exists for.

The E3 attention kernel scales its logits by a per-head temperature. That
temperature used to be stored directly (``scale = tau``); it is now stored as an
unconstrained pre-activation read through softplus (``scale = softplus(tau)``),
so the temperature cannot cross zero and invert the kernel.

Both eras write the same tensor, under the same name, in checkpoints that carry
**no version stamp** — so the meaning of an unversioned ``tau`` cannot be
recovered from the file. A trained temperature of 0.54 and a trained
pre-activation of 0.54 are the same bytes, and guessing between them silently
changes the effective temperature by ~30% either way (``softplus(1) = 1.313``,
``softplus(0.539) = 0.999``). Value-sniffing here would be a guess wearing the
costume of a check.

So: from now on every checkpoint carries ``format_version``. An unversioned
checkpoint that contains no ``tau`` at all (every E0/E1/E2 artifact) is
unambiguous and loads. An unversioned checkpoint that *does* contain ``tau`` is
refused, and the operator declares the era explicitly via
``ENTROPIC_CKPT_TAU_SEMANTICS``:

``softplus``
    written after the change — load as-is (the overwhelmingly likely case for
    anything produced by this repo's own runs).
``temperature``
    written before it — convert with ``inverse_softplus`` so that
    ``softplus(new tau)`` equals the old temperature exactly, preserving the
    forward pass. Eval only: the AdamW moments saved alongside ``tau`` are in
    the pre-conversion parameterisation, so a converted resume is not the exact
    resume :meth:`Trainer.state_dict` promises, and a run that changed
    parameterisation mid-flight is not a valid rung of a controlled ablation.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "TAU_SEMANTICS_ENV",
    "CheckpointFormatError",
    "inverse_softplus",
    "normalize_checkpoint",
]

# 1: `tau` is a softplus pre-activation. Unversioned (implicitly 0) is ambiguous
# whenever `tau` is present -- see the module docstring.
CHECKPOINT_FORMAT_VERSION: int = 1

TAU_SEMANTICS_ENV: str = "ENTROPIC_CKPT_TAU_SEMANTICS"
_TAU_ERAS: tuple[str, ...] = ("softplus", "temperature")


class CheckpointFormatError(RuntimeError):
    """A checkpoint cannot be loaded without changing what it means."""


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`torch.nn.functional.softplus`.

    ``softplus`` has range ``(0, inf)``, so only a strictly positive input has a
    preimage. Computed as ``y + log(-expm1(-y))`` in float64 — the naive
    ``log(exp(y) - 1)`` loses the low end to cancellation and overflows the high
    end — then cast back to the input dtype.
    """
    if not bool(torch.all(x > 0)):
        offenders = x[x <= 0].tolist()
        msg = f"inverse_softplus requires strictly positive inputs; got {offenders}"
        raise ValueError(msg)
    y = x.to(torch.float64)
    return (y + torch.log(-torch.expm1(-y))).to(x.dtype)


def _tau_keys(model_state: dict[str, Any]) -> list[str]:
    return sorted(k for k in model_state if k == "tau" or k.endswith(".tau"))


def _declared_era() -> str | None:
    era = os.environ.get(TAU_SEMANTICS_ENV)
    if era is None or era == "":
        return None
    if era not in _TAU_ERAS:
        msg = f"{TAU_SEMANTICS_ENV}={era!r} is not a known tau era; choose from {list(_TAU_ERAS)}"
        raise CheckpointFormatError(msg)
    return era


def _ambiguity_error(source: str, keys: list[str]) -> CheckpointFormatError:
    return CheckpointFormatError(
        f"\n{'=' * 79}\n"
        f"{source}: checkpoint carries no format_version, and holds kappa-attention\n"
        f"temperatures ({len(keys)} tau tensors, e.g. {keys[0]}). `tau` changed meaning\n"
        "from an effective temperature to a softplus pre-activation, and BOTH eras\n"
        "wrote unversioned checkpoints — so what these values mean cannot be read off\n"
        "the file, and loading either way silently changes the temperature.\n"
        "\n"
        "Declare the era explicitly (or re-run the experiment, which is cheap for a\n"
        "smoke checkpoint and removes the ambiguity for good):\n"
        f"  {TAU_SEMANTICS_ENV}=softplus     written AFTER the softplus change (load as-is)\n"
        f"  {TAU_SEMANTICS_ENV}=temperature  written BEFORE it (converted; eval only)\n"
        f"{'=' * 79}"
    )


def _convert_legacy_tau(model_state: dict[str, Any], keys: list[str], source: str) -> None:
    for key in keys:
        tensor = model_state[key]
        try:
            model_state[key] = inverse_softplus(tensor)
        except ValueError as exc:
            msg = (
                f"{source}: {key} holds a non-positive legacy temperature "
                f"({tensor.tolist()}). softplus has range (0, inf), so an inverted "
                "kernel has no representation under the current parameterisation — "
                "that run cannot be reproduced by this build."
            )
            raise CheckpointFormatError(msg) from exc


def normalize_checkpoint(
    state: dict[str, Any], *, source: str, for_resume: bool = False
) -> dict[str, Any]:
    """Return ``state`` at the current format version, or refuse.

    Args:
        state: a checkpoint mapping as written by :meth:`Trainer.state_dict`.
        source: path (or label) named in error messages.
        for_resume: ``True`` when the optimiser state will also be restored,
            which forbids the legacy-``tau`` conversion.
    """
    version = state.get("format_version")
    if version == CHECKPOINT_FORMAT_VERSION:
        return state
    if version is not None:
        msg = (
            f"{source}: checkpoint format_version {version!r} is newer than this build "
            f"understands ({CHECKPOINT_FORMAT_VERSION}) — update the code, do not "
            "downgrade the file"
        )
        raise CheckpointFormatError(msg)

    model_state = state.get("model", {})
    keys = _tau_keys(model_state)
    if not keys:
        # No kappa kernel: nothing in this checkpoint changed meaning.
        return {**state, "format_version": CHECKPOINT_FORMAT_VERSION}

    era = _declared_era()
    if era is None:
        raise _ambiguity_error(source, keys)
    if era == "temperature":
        if for_resume:
            msg = (
                f"{source}: {TAU_SEMANTICS_ENV}=temperature converts tau, but the saved "
                "AdamW moments for it are in the pre-conversion parameterisation — the "
                "resume would not be the exact resume the checkpoint promises. Evaluate "
                "this checkpoint instead, or start a fresh run."
            )
            raise CheckpointFormatError(msg)
        model_state = dict(model_state)
        _convert_legacy_tau(model_state, keys, source)
    print(
        f"{source}: unversioned checkpoint loaded as {TAU_SEMANTICS_ENV}={era} "
        f"({'tau converted' if era == 'temperature' else 'tau taken as-is'}, "
        f"{len(keys)} tensors) — declared by the operator, not inferred.",
        file=sys.stderr,
    )
    return {**state, "model": model_state, "format_version": CHECKPOINT_FORMAT_VERSION}
