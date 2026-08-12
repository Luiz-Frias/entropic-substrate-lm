"""Forward/backward tests for the transformer blocks and backbones."""

from __future__ import annotations

import pytest
import torch

from entropic_substrate.models.backbone import build_model, vocab_size_for
from entropic_substrate.models.blocks import (
    BaselineBlock,
    CausalSelfAttention,
    NGPTBlock,
    apply_rotary,
    build_block,
    build_norm,
    rotary_embeddings,
)
from helpers import tiny_model_config

pytestmark = pytest.mark.unit


def test_rotary_embeddings_preserve_norm() -> None:
    cos, sin = rotary_embeddings(16, 8, torch.device("cpu"))
    assert cos.shape == (16, 4)
    x = torch.randn(2, 3, 16, 8)
    y = apply_rotary(x, cos, sin)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=0)


def test_rotary_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="even"):
        rotary_embeddings(4, 7, torch.device("cpu"))


@pytest.mark.parametrize("norm", ["layernorm", "ngpt_norm", "kappa_moe"])
def test_build_norm_dispatch(norm: str) -> None:
    module = build_norm(tiny_model_config(norm=norm))
    assert module(torch.randn(2, 3, 32)).shape == (2, 3, 32)


def test_build_norm_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown norm"):
        build_norm(tiny_model_config(norm="banana"))


def test_build_block_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown block"):
        build_block(tiny_model_config(block="banana"))


def test_attention_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        build_block(tiny_model_config(d_model=30, n_heads=4))


@pytest.mark.parametrize(
    ("block", "norm"),
    [("baseline", "layernorm"), ("ngpt", "ngpt_norm"), ("ngpt", "kappa_moe")],
)
def test_block_forward_backward(block: str, norm: str) -> None:
    cfg = tiny_model_config(block=block, norm=norm)
    module = build_block(cfg)
    x = torch.randn(2, 5, cfg.d_model, requires_grad=True)
    y = module(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, param in module.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad).all(), name


def test_baseline_block_is_residual() -> None:
    cfg = tiny_model_config(block="baseline")
    module = BaselineBlock(cfg)
    with torch.no_grad():
        module.attn.out_proj.weight.zero_()
        module.mlp.proj.weight.zero_()
    x = torch.randn(2, 4, cfg.d_model)
    torch.testing.assert_close(module(x), x, atol=1e-6, rtol=0)


@pytest.mark.parametrize("norm", ["ngpt_norm", "kappa_moe"])
def test_ngpt_block_output_lives_on_the_unit_sphere(norm: str) -> None:
    cfg = tiny_model_config(block="ngpt", norm=norm)
    module = NGPTBlock(cfg)
    x = torch.randn(3, 6, cfg.d_model) * 4.0
    y = module(x)
    torch.testing.assert_close(y.norm(dim=-1), torch.ones(3, 6), atol=1e-5, rtol=0)


def test_ngpt_normalize_weights_unit_norms_every_projection() -> None:
    """Unit constraint must live on the EMBEDDING axis, matching the nGPT reference.

    references/ngpt/train.py normalizes q/k/v and mlp.fc along the input axis
    (dim=1 in ``nn.Linear``'s (out, in) layout) and the c_proj pair
    (attn.out_proj, mlp.proj) along the output axis (dim=0). Regression test
    for review finding F5.
    """
    cfg = tiny_model_config(block="ngpt", norm="ngpt_norm")
    module = NGPTBlock(cfg)
    with torch.no_grad():
        module.attn.q_proj.weight.mul_(37.0)
        module.attn.out_proj.weight.mul_(19.0)
        module.mlp.proj.weight.mul_(11.0)
    module.normalize_weights()
    for linear, dim in (
        (module.attn.q_proj, 1),
        (module.attn.k_proj, 1),
        (module.attn.v_proj, 1),
        (module.attn.out_proj, 0),
        (module.mlp.fc, 1),
        (module.mlp.proj, 0),
    ):
        norms = linear.weight.norm(dim=dim)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)
    # Non-square matrices make an axis mix-up impossible to pass silently:
    # out_proj/mlp.proj normalized along dim=0 must NOT be unit-norm along dim=1
    # (and vice versa) unless the matrix is degenerate.
    if module.mlp.proj.weight.shape[0] != module.mlp.proj.weight.shape[1]:
        wrong_axis = module.mlp.proj.weight.norm(dim=1)
        assert not torch.allclose(wrong_axis, torch.ones_like(wrong_axis), atol=1e-3)


def test_ngpt_alpha_is_learnable_and_interpolates() -> None:
    cfg = tiny_model_config(block="ngpt", norm="ngpt_norm")
    module = NGPTBlock(cfg)
    assert module.attn_alpha.requires_grad and module.mlp_alpha.requires_grad
    # alpha = 0 => the block is the identity retraction of its input.
    with torch.no_grad():
        module.attn_alpha.zero_()
        module.mlp_alpha.zero_()
    x = torch.randn(2, 4, cfg.d_model)
    expected = x / x.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(module(x), expected, atol=1e-6, rtol=0)


def test_baseline_normalize_weights_is_a_noop() -> None:
    module = BaselineBlock(tiny_model_config())
    before = module.attn.q_proj.weight.clone()
    module.normalize_weights()
    assert torch.equal(module.attn.q_proj.weight, before)


# ------------------------------------------------------------------ backbone
def test_vocab_size_for_byte_and_override() -> None:
    assert vocab_size_for(tiny_model_config(tokenizer="byte")) == 256
    assert vocab_size_for(tiny_model_config(vocab_size=1234)) == 1234


@pytest.mark.parametrize(
    ("block", "norm"),
    [("baseline", "layernorm"), ("ngpt", "ngpt_norm"), ("ngpt", "kappa_moe")],
)
def test_language_model_forward_backward(block: str, norm: str) -> None:
    cfg = tiny_model_config(block=block, norm=norm)
    model = build_model(cfg)
    idx = torch.randint(0, 256, (2, 16))
    out = model(idx, idx)
    assert out.logits.shape == (2, 16, 256)
    assert out.loss is not None and torch.isfinite(out.loss)
    total = out.loss if out.aux_loss is None else out.loss + out.aux_loss
    total.backward()
    assert model.wte.weight.grad is not None


def test_kappa_moe_model_reports_stats_and_aux_loss() -> None:
    model = build_model(tiny_model_config(block="ngpt", norm="kappa_moe"))
    out = model(torch.randint(0, 256, (2, 12)))
    assert len(model.kappa_modules()) == 2 * model.cfg.n_layers
    assert len(out.kappa_stats) == len(model.kappa_modules())
    assert out.aux_loss is not None and torch.isfinite(out.aux_loss)


def test_model_without_kappa_moe_has_no_aux_loss() -> None:
    model = build_model(tiny_model_config(block="baseline", norm="layernorm"))
    out = model(torch.randint(0, 256, (2, 12)))
    assert out.aux_loss is None
    assert float(model.collect_aux_loss()) == 0.0


def test_random_sphere_embedding_init_is_applied() -> None:
    model = build_model(tiny_model_config(embedding_init="random_sphere"))
    norms = model.wte.weight.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


def test_fisher_embedding_init_is_applied(sample_bytes: bytes) -> None:
    model = build_model(tiny_model_config(embedding_init="fisher_sqrt"), fisher_source=sample_bytes)
    norms = model.wte.weight.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


def test_fisher_init_requires_a_source_or_cached_table() -> None:
    with pytest.raises(ValueError, match="fisher_sqrt init needs"):
        build_model(tiny_model_config(embedding_init="fisher_sqrt"))


def test_fisher_init_rejects_bpe_tokenizer(sample_bytes: bytes) -> None:
    cfg = tiny_model_config(embedding_init="fisher_sqrt", tokenizer="gpt2", vocab_size=300)
    with pytest.raises(ValueError, match="byte alphabet"):
        build_model(cfg, fisher_source=sample_bytes)


def test_unknown_embedding_init_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown embedding_init"):
        build_model(tiny_model_config(embedding_init="quantum"))


def test_model_rejects_sequences_longer_than_block_size() -> None:
    cfg = tiny_model_config(block_size=8)
    model = build_model(cfg)
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 256, (1, 9)))


def test_ngpt_model_weights_are_unit_norm_at_init() -> None:
    model = build_model(tiny_model_config(block="ngpt", norm="ngpt_norm"))
    norms = model.wte.weight.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


def test_generate_extends_the_sequence() -> None:
    model = build_model(tiny_model_config())
    idx = torch.randint(0, 256, (1, 4))
    out = model.generate(idx, 5, temperature=0.8, top_k=10)
    assert out.shape == (1, 9)
    assert bool(((out >= 0) & (out < 256)).all())


def test_parameter_counting_helpers() -> None:
    model = build_model(tiny_model_config())
    assert model.num_parameters() > model.num_parameters(non_embedding=True)


# --------------------------------------------------------------------------
# kappa-attention (E3): curvature in the comparison kernel
# --------------------------------------------------------------------------


def _kappa_attn(**overrides: object) -> CausalSelfAttention:
    cfg = tiny_model_config(block="ngpt", norm="ngpt_norm", attn_kernel="kappa", **overrides)
    return CausalSelfAttention(cfg, hypersphere=True)


def test_kappa_attention_gradient_reaches_kappa_at_exactly_zero() -> None:
    """The invariant the annihilated norm layer could not satisfy, resurrected
    where it is structurally satisfiable: between distinct q/k points curvature
    has first-order effect, even at the Taylor seam."""
    torch.manual_seed(0)
    attn = _kappa_attn(kappa_init=[0.0])
    x = torch.randn(2, 8, 32)
    attn(x).square().sum().backward()
    assert attn.kappa.grad is not None
    assert float(attn.kappa.grad.abs().sum()) > 1e-6


def test_kappa_attention_at_zero_curvature_matches_cosine_at_init() -> None:
    """E2 is the exact kappa=0 member of the E3 kernel family (softmax
    shift-invariance kills the constant), pinned at initialization where
    sqk == sqk_scale makes the E2 kernel exactly cosine * sqrt(head_dim)."""
    torch.manual_seed(0)
    cos_attn = CausalSelfAttention(
        tiny_model_config(block="ngpt", norm="ngpt_norm"), hypersphere=True
    )
    kap_attn = _kappa_attn(kappa_init=[0.0])
    kap_attn.load_state_dict(cos_attn.state_dict(), strict=False)
    x = torch.randn(2, 12, 32)
    torch.testing.assert_close(kap_attn(x), cos_attn(x), atol=1e-5, rtol=0)


def test_kappa_attention_is_causal() -> None:
    torch.manual_seed(0)
    attn = _kappa_attn()
    x = torch.randn(1, 10, 32)
    base = attn(x)
    bumped = x.clone()
    bumped[:, -1] += 3.0
    torch.testing.assert_close(attn(bumped)[:, :-1], base[:, :-1], atol=1e-6, rtol=0)


def test_kappa_attention_kappa_is_per_head_and_finite_under_autocast() -> None:
    torch.manual_seed(0)
    attn = _kappa_attn()
    assert attn.kappa.shape == (4,)
    x = torch.randn(2, 8, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = attn(x)
    assert torch.isfinite(out.float()).all()


def test_kappa_attention_default_init_spreads_kappa_over_heads() -> None:
    attn = _kappa_attn()
    assert attn.kappa.detach().tolist() == [-1.0, 0.0, 1.0, -1.0]


def test_build_model_with_kappa_attention_reports_per_head_kappa_stats() -> None:
    cfg = tiny_model_config(block="ngpt", norm="ngpt_norm", attn_kernel="kappa")
    model = build_model(cfg)
    x = torch.randint(0, 256, (2, 16))
    out = model(x, x)
    assert out.kappa_stats
    assert len(out.kappa_stats[0].kappa) == cfg.n_heads


def test_kappa_attention_rejects_unknown_kernel() -> None:
    with pytest.raises(ValueError, match="unknown attn_kernel"):
        CausalSelfAttention(tiny_model_config(attn_kernel="banana"), hypersphere=True)


def test_anchored_kernel_at_zero_anchors_matches_e3_prime() -> None:
    """Anchors init at zero: the anchored arm starts exactly at the E3 point."""
    torch.manual_seed(0)
    base = _kappa_attn()
    anchored = _kappa_attn(anchor_mode="global")
    anchored.load_state_dict(base.state_dict(), strict=False)
    x = torch.randn(2, 10, 32)
    torch.testing.assert_close(anchored(x), base(x), atol=1e-6, rtol=0)


def test_anchor_gradients_alive_at_the_e3_prime_point() -> None:
    """The anchors must be learnable FROM the zero init, not just after it."""
    torch.manual_seed(0)
    attn = _kappa_attn(anchor_mode="global")
    x = torch.randn(2, 8, 32)
    attn(x).square().sum().backward()
    assert attn.p_q is not None and attn.p_q.grad is not None
    assert attn.p_k is not None and attn.p_k.grad is not None
    assert float(attn.p_q.grad.abs().sum()) > 1e-8
    assert float(attn.p_k.grad.abs().sum()) > 1e-8


def test_symmetric_anchors_are_null_by_isometry() -> None:
    """Mobius translation is an isometry: p_q = p_k leaves the distance kernel
    (hence the output) exactly invariant. The trap that motivated the
    asymmetric design, pinned as a theorem (docs/e-3-instruction.md)."""
    torch.manual_seed(0)
    attn = _kappa_attn(anchor_mode="global")
    x = torch.randn(2, 8, 32)
    base = attn(x)
    assert attn.p_q is not None and attn.p_k is not None
    with torch.no_grad():
        shared = torch.randn_like(attn.p_q) * 0.05
        attn.p_q.copy_(shared)
        attn.p_k.copy_(shared)
    torch.testing.assert_close(attn(x), base, atol=1e-5, rtol=0)


def test_asymmetric_anchors_break_isometry_invariance() -> None:
    torch.manual_seed(0)
    attn = _kappa_attn(anchor_mode="global")
    x = torch.randn(2, 8, 32)
    base = attn(x)
    assert attn.p_q is not None and attn.p_k is not None
    with torch.no_grad():
        attn.p_q.add_(0.05)
        attn.p_k.sub_(0.05)
    assert not torch.allclose(attn(x), base, atol=1e-4)


def test_vantage_arm_global_kappa_per_head_anchors() -> None:
    """The vantage ablation arm: one shared curvature, per-head anchors."""
    torch.manual_seed(0)
    attn = _kappa_attn(anchor_mode="per_head", kappa_granularity="global", kappa_init=[0.0])
    assert attn.kappa is not None and attn.kappa.shape == (1,)
    assert attn.p_q is not None and attn.p_q.shape[1] == 4
    out = attn(torch.randn(2, 8, 32))
    assert torch.isfinite(out).all()
