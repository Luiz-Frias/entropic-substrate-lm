r"""Gated :math:`\kappa`-stereographic normalisation ("MoE over geometries").

A drop-in replacement for ``LayerNorm`` / hyperspherical ``Norm`` in which the
*geometry itself* is the mixture-of-experts dimension.

For each token the layer

1. casts to FP32 and disables autocast (all manifold math is FP32, always);
2. projects the token into each of ``K`` constant-curvature charts
   (:func:`projx`) and lifts it to the tangent space at the origin of that
   chart (:func:`logmap0`) - each component owns a *learnable* curvature
   :math:`\kappa_i`, clamped to ``[-5, 5]``;
3. routes the token to the top-``k`` components with a linear gate and mixes
   their tangent vectors with renormalised gate weights;
4. maps the mixed tangent vector back with :func:`expmap0` at the
   gate-weighted effective curvature and retracts onto the unit sphere.

A Switch-Transformer style load-balancing auxiliary loss is exposed on
``self.aux_loss`` (plus router statistics on ``self.stats``) and collected by
the trainer each step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from entropic_substrate.geometry.kappa import (
    KAPPA_MAX,
    KAPPA_MIN,
    clamp_kappa,
    expmap0,
    logmap0,
    projx,
)

__all__ = ["GatedKappaNorm", "KappaNormStats", "UnitNorm"]


class UnitNorm(nn.Module):
    """Affine-free L2 retraction onto the unit hypersphere (nGPT ``Norm``)."""

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / x.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


@dataclass(slots=True)
class KappaNormStats:
    """Router / curvature telemetry for one forward pass."""

    kappa: list[float]
    gate_fraction: list[float]
    router_prob: list[float]
    aux_loss: float
    entropy: float


class GatedKappaNorm(nn.Module):
    r"""Sparse mixture over ``n_components`` constant-curvature geometries.

    Parameters
    ----------
    d_model:
        Feature dimension.
    n_components:
        Number of curvature experts ``K``. Curvatures are initialised evenly
        over ``kappa_init`` (default ``[-1, 0, +1]`` for ``K == 3``).
    top_k:
        1 or 2. Number of components each token is routed to.
    aux_loss_weight:
        Scale applied to the Switch load-balancing loss when the trainer adds
        it to the task loss.
    gate_pool:
        ``"token"`` (default) routes each token independently.  ``"sequence"``
        mean-pools features over time before routing - only valid for
        non-causal use, since it leaks future context.
    router_noise:
        Std of Gaussian logit noise during training (0 disables), a cheap
        exploration aid that keeps dead experts from staying dead.
    """

    kappa: nn.Parameter

    def __init__(
        self,
        d_model: int,
        n_components: int = 3,
        top_k: int = 1,
        *,
        aux_loss_weight: float = 0.01,
        kappa_init: tuple[float, ...] | list[float] | None = None,
        gate_pool: str = "token",
        router_noise: float = 0.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if top_k not in (1, 2):
            msg = f"top_k must be 1 or 2, got {top_k}"
            raise ValueError(msg)
        if top_k > n_components:
            msg = f"top_k={top_k} exceeds n_components={n_components}"
            raise ValueError(msg)
        if gate_pool not in ("token", "sequence"):
            msg = f"gate_pool must be 'token' or 'sequence', got {gate_pool!r}"
            raise ValueError(msg)

        self.d_model = d_model
        self.n_components = n_components
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.gate_pool = gate_pool
        self.router_noise = router_noise
        self.eps = eps

        if kappa_init is None:
            if n_components == 1:
                values = [0.0]
            else:
                step = 2.0 / (n_components - 1)
                values = [-1.0 + step * i for i in range(n_components)]
        else:
            values = list(kappa_init)
        if len(values) != n_components:
            msg = f"kappa_init has {len(values)} entries, expected {n_components}"
            raise ValueError(msg)

        self.kappa = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.gate = nn.Linear(d_model, n_components, bias=False, dtype=torch.float32)
        nn.init.normal_(self.gate.weight, std=d_model**-0.5)

        # Populated on every forward; consumed by the trainer.
        self.aux_loss: torch.Tensor = torch.zeros((), dtype=torch.float32)
        self.stats: KappaNormStats = KappaNormStats(
            kappa=values,
            gate_fraction=[0.0] * n_components,
            router_prob=[0.0] * n_components,
            aux_loss=0.0,
            entropy=0.0,
        )

    # ---------------------------------------------------------------- utils
    def current_kappa(self) -> torch.Tensor:
        """Curvatures as actually used by the geometry (post-clamp), detached."""
        return clamp_kappa(self.kappa.detach())

    def _router_input(self, x: torch.Tensor) -> torch.Tensor:
        """Feature vector fed to the linear gate, RMS-normalised for scale safety."""
        if self.gate_pool == "sequence" and x.dim() >= 3:
            pooled = x.mean(dim=-2, keepdim=True).expand_as(x)
        else:
            pooled = x
        rms = pooled.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return pooled * rms

    # -------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` (``..., d_model``) through the geometry mixture."""
        out_dtype = x.dtype
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            x32 = x.float()
            # Retract onto the sphere first: the backbone lives on S^{d-1}, so
            # this fixes the input radius and keeps every chart well-conditioned.
            x32 = x32 / x32.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)

            logits = self.gate(self._router_input(x32))
            if self.training and self.router_noise > 0.0:
                logits = logits + torch.randn_like(logits) * self.router_noise
            probs = torch.softmax(logits, dim=-1)

            top_vals, top_idx = torch.topk(probs, self.top_k, dim=-1)
            weights = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            # Dense one-hot dispatch mask: K is tiny (<= 8), so computing every
            # chart and masking is far cheaper than gather/scatter, and keeps
            # the graph static for torch.compile.
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, top_idx, weights)

            kappa = clamp_kappa(self.kappa).to(x32.dtype)

            tangents = []
            for i in range(self.n_components):
                k_i = kappa[i]
                tangents.append(logmap0(projx(x32, k_i), k_i))
            tangent_stack = torch.stack(tangents, dim=-2)  # (..., K, d)

            gate_w = mask.unsqueeze(-1)  # (..., K, 1)
            mixed = (tangent_stack * gate_w).sum(dim=-2)
            kappa_eff = (mask * kappa).sum(dim=-1, keepdim=True)

            y = expmap0(mixed, kappa_eff)
            y = y / y.norm(p=2, dim=-1, keepdim=True).clamp_min(self.eps)

            self._update_stats(probs, mask, kappa)
        return y.to(out_dtype)

    # ---------------------------------------------------------------- stats
    def _update_stats(self, probs: torch.Tensor, mask: torch.Tensor, kappa: torch.Tensor) -> None:
        """Switch-style load balance: ``K * sum_i f_i * P_i``."""
        flat_probs = probs.reshape(-1, self.n_components)
        flat_mask = (mask.reshape(-1, self.n_components) > 0).to(flat_probs.dtype)

        # f_i: fraction of dispatch slots that went to component i.
        load = flat_mask.mean(dim=0) * (self.n_components / max(self.top_k, 1))
        # P_i: mean router probability mass on component i.
        importance = flat_probs.mean(dim=0)
        aux = (load * importance).sum() * self.aux_loss_weight
        self.aux_loss = aux

        with torch.no_grad():
            fraction = flat_mask.mean(dim=0)
            entropy = -(flat_probs.clamp_min(1e-9).log() * flat_probs).sum(-1).mean()
            self.stats = KappaNormStats(
                kappa=[round(float(v), 6) for v in kappa.detach().cpu()],
                gate_fraction=[round(float(v), 6) for v in fraction.cpu()],
                router_prob=[round(float(v), 6) for v in importance.detach().cpu()],
                aux_loss=float(aux.detach()),
                entropy=float(entropy),
            )

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_components={self.n_components}, "
            f"top_k={self.top_k}, kappa_range=[{KAPPA_MIN}, {KAPPA_MAX}], "
            f"gate_pool={self.gate_pool!r}"
        )
