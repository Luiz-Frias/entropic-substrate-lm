"""Transformer blocks: a pre-LN baseline and an nGPT-style hyperspherical block.

The nGPT block (Loshchilov et al., "nGPT: Normalized Transformer with
Representation Learning on the Hypersphere", arXiv:2410.01131; reference impl.
vendored at ``references/ngpt/model.py``) treats the residual stream as living
on :math:`S^{d-1}`:

* every weight matrix is unit-norm along its embedding dimension (enforced by
  :meth:`normalize_weights`, called after each optimiser step);
* there is no LayerNorm - the ``Norm`` retraction *is* the normalisation;
* the residual update is a learnable interpolation along the sphere,
  :math:`h \\leftarrow \\mathrm{Norm}(h + \\alpha(\\mathrm{Norm}(h_A) - h))`,
  where :math:`\\alpha` is a learned per-dimension "eigen learning rate".

For E3 the attention *similarity kernel* is swapped (``attn_kernel: kappa``):
per-head logits go from cosine to the κ-stereographic squared distance
:math:`-d_\\kappa(q, k)^2` with one learnable clamped curvature per head — E2
is the exact :math:`\\kappa = 0` member of the family. (The earlier plan to
swap the ``Norm`` retraction is radially annihilated — design record in
``docs/e-3-instruction.md``.)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from entropic_substrate.config import ModelConfig
from entropic_substrate.geometry.kappa import (
    clamp_kappa,
    dist_k_from_cos,
    dist_k_from_inner,
    mobius_add,
    projx,
)
from entropic_substrate.layers.kappa_norm import GatedKappaNorm, UnitNorm

__all__ = [
    "MLP",
    "BaselineBlock",
    "CausalSelfAttention",
    "NGPTBlock",
    "apply_rotary",
    "build_block",
    "build_norm",
    "rotary_embeddings",
]

# inverse_softplus(1) = ln(e - 1): the tau init that makes softplus(tau) == 1.
_INV_SOFTPLUS_1: float = math.log(math.e - 1.0)


# --------------------------------------------------------------------- rope
def rotary_embeddings(
    seq_len: int, head_dim: int, device: torch.device, base: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(cos, sin)`` of shape ``(seq_len, head_dim // 2)``."""
    if head_dim % 2 != 0:
        msg = f"head_dim must be even for RoPE, got {head_dim}"
        raise ValueError(msg)
    inv_freq = torch.exp(
        torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        * (-math.log(base) / head_dim)
    )
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    angles = pos * inv_freq.unsqueeze(0)
    return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` of shape ``(B, n_head, T, head_dim)``."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.to(x.dtype).view(1, 1, cos.shape[0], cos.shape[1])
    sin = sin.to(x.dtype).view(1, 1, sin.shape[0], sin.shape[1])
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# ---------------------------------------------------------------------- norm
def build_norm(cfg: ModelConfig) -> nn.Module:
    """Instantiate the normalisation layer selected by ``cfg.norm``."""
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.d_model, bias=cfg.bias)
    if cfg.norm == "ngpt_norm":
        return UnitNorm()
    if cfg.norm == "kappa_moe":
        return GatedKappaNorm(
            cfg.d_model,
            n_components=cfg.n_components,
            top_k=cfg.top_k,
            aux_loss_weight=cfg.aux_loss_weight,
            kappa_init=cfg.kappa_init,
            router_noise=cfg.router_noise,
        )
    msg = f"unknown norm {cfg.norm!r}"
    raise ValueError(msg)


# ----------------------------------------------------------------- submodules
class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with RoPE and (optionally) nGPT q/k scaling."""

    # Each is registered as None on the paths that don't use it.
    sqk: nn.Parameter | None
    kappa: nn.Parameter | None
    tau: nn.Parameter | None
    p_q: nn.Parameter | None
    p_k: nn.Parameter | None

    def __init__(self, cfg: ModelConfig, *, hypersphere: bool) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            msg = f"d_model={cfg.d_model} not divisible by n_heads={cfg.n_heads}"
            raise ValueError(msg)
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.hypersphere = hypersphere
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

        if hypersphere:
            # Per-dimension query/key scale, restoring the unit-norm q,k to a
            # learned length (nGPT `sqk`).
            self.sqk_scale = cfg.d_model**-0.5
            self.sqk = nn.Parameter(torch.full((cfg.d_model,), self.sqk_scale))
        else:
            self.register_parameter("sqk", None)

        if cfg.attn_kernel not in ("cosine", "kappa"):
            msg = f"unknown attn_kernel {cfg.attn_kernel!r}"
            raise ValueError(msg)
        self.attn_kernel = cfg.attn_kernel
        if cfg.anchor_mode not in ("none", "global", "per_head"):
            msg = f"unknown anchor_mode {cfg.anchor_mode!r}"
            raise ValueError(msg)
        if cfg.kappa_granularity not in ("per_head", "global"):
            msg = f"unknown kappa_granularity {cfg.kappa_granularity!r}"
            raise ValueError(msg)
        if cfg.anchor_mode != "none" and cfg.attn_kernel != "kappa":
            msg = "anchor_mode requires attn_kernel='kappa'"
            raise ValueError(msg)
        self.anchor_mode = cfg.anchor_mode

        if cfg.attn_kernel == "kappa":
            # Learnable curvature — per head (kappa_init tiled) or one shared
            # knob (the vantage ablation). sqk is deliberately NOT used on this
            # path: a per-dimension scale inside a distance kernel is a learned
            # diagonal metric, which would confound anisotropy with the
            # curvature under test. Logit shaping is a per-head temperature.
            init = list(cfg.kappa_init) if cfg.kappa_init else [-1.0, 0.0, 1.0]
            if cfg.kappa_granularity == "global":
                values = [init[0]]
            else:
                values = [init[i % len(init)] for i in range(cfg.n_heads)]
            self.kappa = nn.Parameter(torch.tensor(values, dtype=torch.float32))
            # Temperature is stored unconstrained and read through softplus: a
            # tau that crossed zero would invert the kernel (nearest becomes
            # farthest). inverse_softplus(1) starts it at exactly 1.
            self.tau = nn.Parameter(
                torch.full((cfg.n_heads,), _INV_SOFTPLUS_1, dtype=torch.float32)
            )
        else:
            self.register_parameter("kappa", None)
            self.register_parameter("tau", None)

        if cfg.attn_kernel == "kappa" and cfg.anchor_mode != "none":
            # Asymmetric Mobius anchors (p_q != p_k breaks isometry-invariance;
            # symmetric anchors are provably null). Zero init = the kernel
            # starts exactly at the anchor-free point and learns away from it.
            n_anchor = 1 if cfg.anchor_mode == "global" else cfg.n_heads
            shape = (1, n_anchor, 1, self.head_dim)
            self.p_q = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
            self.p_k = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        else:
            self.register_parameter("p_q", None)
            self.register_parameter("p_k", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, D) -> (B, T, D)``."""
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = rotary_embeddings(t, self.head_dim, x.device)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

        if self.attn_kernel == "kappa" and self.kappa is not None:
            y = self._kappa_attention(q, k, v)
            y = y.transpose(1, 2).contiguous().view(b, t, d)
            return self.out_proj(y)

        if self.hypersphere and self.sqk is not None:
            sqk = (self.sqk / self.sqk_scale).view(1, self.n_heads, 1, self.head_dim)
            q = sqk * F.normalize(q, p=2, dim=-1)
            k = sqk * F.normalize(k, p=2, dim=-1)
            # q,k are unit vectors => logits in [-1, 1]; nGPT rescales by
            # sqrt(head_dim) instead of the usual 1/sqrt(head_dim).
            scale = float(self.head_dim) ** 0.5
        else:
            scale = float(self.head_dim) ** -0.5

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=scale,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(y)

    def _kappa_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """E3 kernel: per-head logits ``−d_κ(q̂, k̂)² · softplus(τ)·√head_dim / 8``.

        At κ = 0 and softplus(τ) = 1 this reproduces the cosine path at init exactly:
        ``d_0² = 8 − 8cos`` for unit vectors, so the logits are
        ``√hd·(cos − 1)·softplus(τ)`` and softmax kills the constant. All geometry runs
        in FP32 with autocast disabled (non-negotiable numerics).
        """
        if self.kappa is None or self.tau is None:
            msg = "kappa attention requires attn_kernel='kappa' parameters"
            raise RuntimeError(msg)
        t = q.shape[-2]
        with torch.autocast(device_type=q.device.type, enabled=False):
            q32 = F.normalize(q.float(), p=2, dim=-1)
            k32 = F.normalize(k.float(), p=2, dim=-1)
            kappa = clamp_kappa(self.kappa.float()).view(1, -1, 1, 1)
            # Operating radius for κ < 0: a bounded FRACTION of the hyperbolic
            # ball, not its edge — at the boundary, near-antipodal pairs push
            # artanh into its quasi-divergent regime (d/du = 1/(1−u²)), which
            # both amplifies fp32 noise ~1e5x and saturates gradients. At
            # f = 0.75 of the ball, antipodal u = 2f/(1+f²) ≤ 0.96: conditioned.
            # κ ≥ 0 charts are unbounded; radius stays 1 (E2 equivalence).
            ball = 0.75 * kappa.abs().clamp_min(1e-12).rsqrt()
            radius = torch.where(kappa < 0, ball.clamp_max(1.0), torch.ones_like(ball))
            if self.p_q is not None and self.p_k is not None:
                # Anchored arms: Mobius-translate q and k by their own learned
                # anchors before measuring. The anchors are unconstrained
                # parameters, so for kappa < 0 they can drift out of the ball
                # where mobius_add's rational form is undefined — both operands
                # are projected in (a no-op for radius*q32, inside by
                # construction). Translated points lose the common radius, so
                # the general three-scalar reduction is used.
                q_in = projx(radius * q32, kappa)
                k_in = projx(radius * k32, kappa)
                q_pts = projx(mobius_add(projx(-self.p_q.float(), kappa), q_in, kappa), kappa)
                k_pts = projx(mobius_add(projx(-self.p_k.float(), kappa), k_in, kappa), kappa)
                dot = q_pts @ k_pts.transpose(-2, -1)
                q_sq = q_pts.pow(2).sum(-1, keepdim=True)
                k_sq = k_pts.pow(2).sum(-1, keepdim=True).transpose(-2, -1)
                dist = dist_k_from_inner(dot, q_sq, k_sq, kappa)
            else:
                cos = q32 @ k32.transpose(-2, -1)
                dist = dist_k_from_cos(cos, kappa, radius)
            scale = F.softplus(self.tau.float()).view(1, -1, 1, 1) * float(self.head_dim) ** 0.5
            logits = -dist.square() * (scale / 8.0)
            causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
            logits = logits.masked_fill(~causal, float("-inf"))
            weights = torch.softmax(logits, dim=-1)
            if self.training and self.dropout > 0.0:
                weights = F.dropout(weights, p=self.dropout)
            y32 = weights @ v.float()
        return y32.to(v.dtype)


class MLP(nn.Module):
    """GELU MLP (baseline) or SwiGLU MLP with ``suv`` scaling (nGPT)."""

    def __init__(self, cfg: ModelConfig, *, hypersphere: bool) -> None:
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.d_model)
        self.hypersphere = hypersphere
        self.dropout = nn.Dropout(cfg.dropout)
        if hypersphere:
            self.fc = nn.Linear(cfg.d_model, 2 * hidden, bias=cfg.bias)
            self.proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
            self.suv_scale = 1.0
            self.suv = nn.Parameter(torch.ones(2 * hidden))
            self.d_model = cfg.d_model
        else:
            self.fc = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
            self.proj = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, D) -> (B, T, D)``."""
        h = self.fc(x)
        if self.hypersphere:
            h = h * (self.suv * (self.d_model**0.5 / self.suv_scale))
            u, v = torch.chunk(h, 2, dim=-1)
            h = u * F.silu(v)
        else:
            h = F.gelu(h, approximate="tanh")
        return self.dropout(self.proj(h))


# -------------------------------------------------------------------- blocks
class BaselineBlock(nn.Module):
    """Standard pre-normalisation transformer block (E0/E1 baselines)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = build_norm(cfg)
        self.norm_mlp = build_norm(cfg)
        self.attn = CausalSelfAttention(cfg, hypersphere=False)
        self.mlp = MLP(cfg, hypersphere=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``(B, T, D) -> (B, T, D)``."""
        h = h + self.resid_dropout(self.attn(self.norm_attn(h)))
        return h + self.resid_dropout(self.mlp(self.norm_mlp(h)))

    def normalize_weights(self) -> None:
        """No-op: the baseline block has no unit-norm weight constraint."""
        return


class NGPTBlock(nn.Module):
    r"""Hyperspherical block: Norm-as-retraction + learnable :math:`\alpha` LERP.

    ``h <- Norm(h + alpha * (Norm(h_sub) - h))`` with ``Norm`` either the plain
    unit retraction (E2) or :class:`GatedKappaNorm` (E3). ``alpha`` is stored
    scaled by ``1/sqrt(d_model)`` (nGPT's ``base_scale`` trick) so AdamW sees a
    parameter of the same magnitude as everything else.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(cfg, hypersphere=True)
        self.mlp = MLP(cfg, hypersphere=True)
        self.norm_attn = build_norm(cfg)
        self.norm_mlp = build_norm(cfg)
        # Retraction of the residual stream itself is always the plain unit
        # retraction: the stream must stay exactly on S^{d-1}.
        self.retract = UnitNorm()

        self.alpha_scale = cfg.d_model**-0.5
        self.alpha_init = cfg.alpha_init
        self.attn_alpha = nn.Parameter(torch.full((cfg.d_model,), self.alpha_scale))
        self.mlp_alpha = nn.Parameter(torch.full((cfg.d_model,), self.alpha_scale))

    def _lerp(self, h: torch.Tensor, update: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        lr = (alpha * (self.alpha_init / self.alpha_scale)).abs()
        a = self.retract(h)
        b = self.retract(update)
        return self.retract(a + lr * (b - a))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``(B, T, D) -> (B, T, D)``, output exactly unit-norm along ``D``."""
        h = self._lerp(h, self.attn(self.norm_attn(h)), self.attn_alpha)
        return self._lerp(h, self.mlp(self.norm_mlp(h)), self.mlp_alpha)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        """Project every projection matrix back to unit norm along the embedding axis.

        Follows the nGPT reference (``references/ngpt/train.py``): matrices that
        *consume* the d_model stream (q/k/v, mlp.fc) are normalised along their
        input axis (``dim=1`` for ``nn.Linear``'s ``(out, in)`` layout), while
        matrices that *produce* the d_model stream (attn.out_proj, mlp.proj —
        nGPT's ``c_proj`` pair) are normalised along their output axis
        (``dim=0``). Either way the unit constraint lives on the embedding
        dimension, so every dot product against the stream is a cosine.
        """
        for linear, dim in (
            (self.attn.q_proj, 1),
            (self.attn.k_proj, 1),
            (self.attn.v_proj, 1),
            (self.attn.out_proj, 0),
            (self.mlp.fc, 1),
            (self.mlp.proj, 0),
        ):
            w = linear.weight
            w.copy_(w / w.norm(p=2, dim=dim, keepdim=True).clamp_min(1e-12))


def build_block(cfg: ModelConfig) -> nn.Module:
    """Instantiate the block type selected by ``cfg.block``."""
    if cfg.block == "baseline":
        return BaselineBlock(cfg)
    if cfg.block == "ngpt":
        return NGPTBlock(cfg)
    msg = f"unknown block {cfg.block!r}"
    raise ValueError(msg)
