"""Q7: no hidden normalization on the magnitude-preserving paths, proven by mutation.

On the development widths, the magnitude sentence interface and the magnitude spine carrier
must emit states whose norms are not 1 and equal an independent unnormalized reference; the
sphere interface and carrier must emit unit norms (the control showing the instrument can
read a normalization). The sentence reference is the raw encoder projection applied to the
projection's own captured input (a forward hook), so it is independent of everything after
that projection; a normalization placed before the projection is outside it. The spine
reference is written out from the raw weights. Held on a fresh model, on a model prepared as
the runner prepares it (``normalize_weights``), in training mode, and on a trained, saved and
reloaded state. The mutation tests inject an F.normalize or an RMS overwrite into each of the codec,
numerics and spine magnitude paths by monkeypatch (never an edit of a reader member) and
require the detector to fire, so a green run is not "the hazard never reached the guard".
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import entropic_substrate.core.sentence.model as model_module
import entropic_substrate.core.sentence.numerics as numerics_module
from entropic_substrate.core.sentence.boundary import SentenceByteCodec
from entropic_substrate.core.sentence.codec import CodecSpec, SentenceEncoder
from entropic_substrate.core.sentence.law import LawSpec
from entropic_substrate.core.sentence.model import SentencePredictor
from entropic_substrate.core.sentence.spine import CarrierBlock, SentenceSpine, SpineSpec

pytestmark = pytest.mark.unit

UNIT_TOL = 1e-4  # a norm is "1" iff it is within this of 1; sphere paths hold to ~1e-7
# A magnitude sentence norm is "constant" iff its relative spread is within this: a clean
# magnitude arm measures 0.2865, an RMS-pinned one 9.7e-08 (probed).
SPREAD_TOL = 1e-3
REFERENCE_TOL = 1e-3  # relative and absolute, FP32 on CPU
CELLS = {
    "MM": ("magnitude", "magnitude"),
    "SM": ("sphere", "magnitude"),
    "MS": ("magnitude", "sphere"),
    "SS": ("sphere", "sphere"),
}
SENTENCE_END, PAD = SentenceByteCodec.sentence_end, SentenceByteCodec.pad


def build(
    mode: str, carrier: str, seed: int = 7, *, runner_regime: bool = False
) -> SentencePredictor:
    """The development widths (384/4 codec, 768x12 spine, context 64), pinned explicitly.

    ``runner_regime`` applies ``spine.normalize_weights()`` as ``ladder/run/execution.py`` does
    at construction.
    """
    torch.manual_seed(seed)
    model = SentencePredictor(
        CodecSpec(384, 4, 384, 4, 64, 4096),
        SpineSpec(768, 12, 64, 64, carrier),  # pyright: ignore[reportArgumentType]
        LawSpec(384, mode),  # pyright: ignore[reportArgumentType]
        consumer="isotropic",
    )
    if runner_regime:
        model.spine.normalize_weights()
    return model.eval()


def history() -> torch.Tensor:
    """Two documents of three complete sentences; ragged lengths, contiguous pad suffix."""
    generator = torch.Generator().manual_seed(1)
    rows = torch.full((2, 3, 14), PAD, dtype=torch.int64)
    for batch in range(2):
        for index in range(3):
            length = 6 + 3 * index + batch
            rows[batch, index, :length] = torch.randint(0, 256, (length,), generator=generator)
            rows[batch, index, length] = SENTENCE_END
    return rows


def _view(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True)


def reference_magnitude_states(model: SentencePredictor, sentences: torch.Tensor) -> torch.Tensor:
    """The unnormalized magnitude carrier, written out independently of ``CarrierBlock``.

    Only the sublayer and attention views are normalized (as the construction allows); the
    carrier itself is ``current + step`` with no retraction and no final overwrite. The entry
    is also written from the raw weights, so a normalization or overwrite of the projection
    or position outputs is not shared with the reference.
    """
    spine: SentenceSpine = model.spine
    batch, time, _ = sentences.shape
    start = spine.bos.view(1, 1, -1).expand(batch, 1, -1)
    states = torch.cat((start, F.linear(sentences, spine.projection.weight)), dim=1)
    states = states + spine.positions.weight[: time + 1]
    width, head_width = spine.spec.width, spine.spec.head_width
    heads = width // head_width
    causal = torch.ones(time + 1, time + 1, dtype=torch.bool).triu(1)

    def residual(
        current: torch.Tensor, proposal: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        gain = (alpha * (spine.spec.alpha_init / width**-0.5)).abs()
        return current + gain * (_view(proposal) - _view(current))

    for block in spine.blocks:
        assert isinstance(block, CarrierBlock)
        qkv = block.qkv(_view(states)).reshape(batch, time + 1, 3, heads, head_width)
        q, k, v = (part.transpose(1, 2) for part in qkv.unbind(dim=2))
        q = _view(q) * block.qk_scale.abs().view(1, -1, 1, 1)
        scores = (q @ _view(k).transpose(-1, -2)).masked_fill(causal, -torch.inf)
        attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, time + 1, width)
        states = residual(states, block.attn_out(attended), block.attn_alpha)
        proposal = block.down(F.gelu(block.up(_view(states))))
        states = residual(states, proposal, block.mlp_alpha)
    return states


def is_unit(norms: torch.Tensor) -> bool:
    return bool(((norms - 1).abs() <= UNIT_TOL).all())


def is_not_unit(norms: torch.Tensor) -> bool:
    return bool(((norms - 1).abs() > UNIT_TOL).all())


@torch.no_grad()
def findings(model: SentencePredictor, mode: str, carrier: str) -> list[str]:
    """Every violated Q7 claim for one cell; empty means the cell is as constructed.

    A refusal by the model's own guards is reported as ``predict_refused`` rather than raised,
    so the sentence checks, which run first, are read even when the model refuses to continue.
    """
    tokens = history()
    projection = model.codec.encoder.projection
    captured: list[torch.Tensor] = []
    handle = projection.register_forward_hook(lambda _m, args, _out: captured.append(args[0]))
    try:
        sentences = model.encode_history(tokens)
    finally:
        handle.remove()
    # The unnormalized encoder output, from the raw weight and the projection's own input: it
    # shares nothing with the module's forward after the projection.
    raw = F.linear(torch.cat(captured), projection.weight)
    found: list[str] = []
    flat = sentences.flatten(0, 1)
    if mode == "magnitude":
        if not is_not_unit(flat.norm(dim=-1)):
            found.append("sentence_norm_is_one")
        sentence_norms = flat.norm(dim=-1)
        if float(sentence_norms.std() / sentence_norms.mean()) <= SPREAD_TOL:
            found.append("sentence_norm_constant")
        if not torch.allclose(flat, raw, rtol=REFERENCE_TOL, atol=REFERENCE_TOL):
            found.append("sentence_differs_from_unnormalized_reference")
    else:
        if not is_unit(flat.norm(dim=-1)):
            found.append("sentence_sphere_not_unit")
        if not torch.allclose(flat, _view(raw), rtol=REFERENCE_TOL, atol=REFERENCE_TOL):
            found.append("sentence_differs_from_normalized_reference")
    try:
        states = model.predict(tokens).states
    except ValueError:
        return [*found, "predict_refused"]
    norms = states.norm(dim=-1)
    if carrier == "magnitude":
        if not is_not_unit(norms):
            found.append("spine_norm_is_one")
        reference = reference_magnitude_states(model, sentences)
        if not torch.allclose(states, reference, rtol=REFERENCE_TOL, atol=REFERENCE_TOL):
            found.append("spine_differs_from_unnormalized_reference")
    elif not is_unit(norms):
        found.append("spine_sphere_not_unit")
    return found


@pytest.mark.parametrize("cell", list(CELLS))
def test_magnitude_paths_are_unnormalized_and_sphere_paths_are_unit(cell: str) -> None:
    mode, carrier = CELLS[cell]
    assert findings(build(mode, carrier), mode, carrier) == []


@pytest.mark.parametrize("cell", list(CELLS))
def test_the_claims_hold_in_the_runners_regime_after_weight_normalization(cell: str) -> None:
    """``normalize_weights`` is applied at runner construction; the claims survive it."""
    mode, carrier = CELLS[cell]
    assert findings(build(mode, carrier, runner_regime=True), mode, carrier) == []


@pytest.mark.parametrize("cell", list(CELLS))
def test_the_claims_hold_in_training_mode(cell: str) -> None:
    mode, carrier = CELLS[cell]
    assert findings(build(mode, carrier).train(), mode, carrier) == []


def test_the_instrument_reads_magnitudes_not_a_constant() -> None:
    """Control: the magnitude arms' norms are large and varied, so 'not 1' is not vacuous."""
    model = build("magnitude", "magnitude")
    with torch.no_grad():
        sentence = model.encode_history(history()).norm(dim=-1)
        states = model.predict(history()).states.norm(dim=-1)
    assert float(sentence.min()) > 1.5 and float(states.min()) > 5.0
    assert float(states.std()) > 0


@pytest.mark.parametrize("cell", list(CELLS))
def test_the_claims_hold_on_a_trained_saved_and_reloaded_state(cell: str, tmp_path: Path) -> None:
    mode, carrier = CELLS[cell]
    trained = build(mode, carrier, seed=7)
    with torch.no_grad():
        untrained = trained.predict(history()).states.clone()
    trained.train()
    optimizer = torch.optim.AdamW(trained.parameters(), lr=1e-2)
    prediction = trained.predict(history())
    # A coordinate loss, not a squared norm: a norm is constant on the sphere cells, so its
    # gradient is zero there and the step would move nothing.
    (prediction.states[..., 0].mean() + prediction.sentences[..., 0].mean()).backward()
    optimizer.step()
    trained.eval()
    with torch.no_grad():
        before = trained.predict(history()).states.clone()
    # Control: the step moved the state, so the saved state is not the initialization.
    assert not torch.allclose(before, untrained, rtol=REFERENCE_TOL, atol=REFERENCE_TOL)
    path = tmp_path / "state.pt"
    torch.save(trained.state_dict(), path)
    del trained, optimizer
    restored = build(mode, carrier, seed=99)  # a different initialization, then the saved state
    with torch.no_grad():
        initial = restored.predict(history()).states.clone()
    assert not torch.allclose(initial, before, rtol=REFERENCE_TOL, atol=REFERENCE_TOL)
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        after = restored.predict(history()).states
    assert torch.allclose(after, before, rtol=REFERENCE_TOL, atol=REFERENCE_TOL)
    assert findings(restored, mode, carrier) == []


# Mutations, applied by monkeypatch only. Each injects a normalization into one magnitude
# path of a reader-member module without editing it.


def _mutate_codec_f_normalize(patch: pytest.MonkeyPatch) -> None:
    original = SentenceEncoder.forward
    patch.setattr(
        SentenceEncoder,
        "forward",
        lambda self, tokens: F.normalize(original(self, tokens), dim=-1),
    )


def _mutate_numerics_f_normalize(patch: pytest.MonkeyPatch) -> None:
    original = numerics_module.apply_interface

    def mutant(values: torch.Tensor, mode: str) -> torch.Tensor:
        result = original(values, mode)  # pyright: ignore[reportArgumentType]
        return F.normalize(result, dim=-1) if mode == "magnitude" else result

    patch.setattr(numerics_module, "apply_interface", mutant)
    patch.setattr(model_module, "apply_interface", mutant)


def _mutate_spine_rms_overwrite(patch: pytest.MonkeyPatch) -> None:
    original = SentenceSpine.forward

    def mutant(
        self: SentenceSpine, sentences: torch.Tensor, *, consumer: object = None
    ) -> torch.Tensor:
        states = original(self, sentences, consumer=consumer)  # pyright: ignore[reportArgumentType]
        return states * states.square().mean(dim=-1, keepdim=True).rsqrt()

    patch.setattr(SentenceSpine, "forward", mutant)


def _mutate_codec_rms_overwrite(patch: pytest.MonkeyPatch) -> None:
    original = SentenceEncoder.forward

    def mutant(self: SentenceEncoder, tokens: torch.Tensor) -> torch.Tensor:
        result = original(self, tokens)
        return result * result.square().mean(dim=-1, keepdim=True).rsqrt()

    patch.setattr(SentenceEncoder, "forward", mutant)


def _mutate_spine_residual_f_normalize(patch: pytest.MonkeyPatch) -> None:
    original = CarrierBlock._residual

    def mutant(
        self: CarrierBlock, current: torch.Tensor, proposal: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        result = original(self, current, proposal, alpha)
        return F.normalize(result, dim=-1) if self.spec.carrier == "magnitude" else result

    patch.setattr(CarrierBlock, "_residual", mutant)


def _squash(values: torch.Tensor) -> torch.Tensor:
    """A soft norm squash: neither F.normalize nor an RMS overwrite, yet magnitude is lost."""
    return values / (1 + values.norm(dim=-1, keepdim=True) / 10)


def _power_norm(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).sqrt()


def _mutate_codec_soft_squash(patch: pytest.MonkeyPatch) -> None:
    original = SentenceEncoder.forward
    patch.setattr(SentenceEncoder, "forward", lambda self, tokens: _squash(original(self, tokens)))


def _mutate_codec_power_norm(patch: pytest.MonkeyPatch) -> None:
    original = SentenceEncoder.forward
    patch.setattr(
        SentenceEncoder, "forward", lambda self, tokens: _power_norm(original(self, tokens))
    )


def _mutate_numerics_soft_squash(patch: pytest.MonkeyPatch) -> None:
    original = numerics_module.apply_interface

    def mutant(values: torch.Tensor, mode: str) -> torch.Tensor:
        result = original(values, mode)  # pyright: ignore[reportArgumentType]
        return _squash(result) if mode == "magnitude" else result

    patch.setattr(numerics_module, "apply_interface", mutant)
    patch.setattr(model_module, "apply_interface", mutant)


MUTATIONS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], str]] = {
    "codec_f_normalize": (_mutate_codec_f_normalize, "sentence_norm_is_one"),
    "codec_rms_overwrite": (_mutate_codec_rms_overwrite, "sentence_norm_constant"),
    "numerics_f_normalize": (_mutate_numerics_f_normalize, "sentence_norm_is_one"),
    "codec_soft_squash": (
        _mutate_codec_soft_squash,
        "sentence_differs_from_unnormalized_reference",
    ),
    "codec_power_norm": (_mutate_codec_power_norm, "sentence_differs_from_unnormalized_reference"),
    "numerics_soft_squash": (
        _mutate_numerics_soft_squash,
        "sentence_differs_from_unnormalized_reference",
    ),
    "spine_rms_overwrite": (
        _mutate_spine_rms_overwrite,
        "spine_differs_from_unnormalized_reference",
    ),
    "spine_residual_f_normalize": (_mutate_spine_residual_f_normalize, "spine_norm_is_one"),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_an_injected_normalization_is_detected(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model = build("magnitude", "magnitude")
    assert findings(model, "magnitude", "magnitude") == []  # control: clean before the mutation
    apply, expected = MUTATIONS[name]
    apply(monkeypatch)
    assert expected in findings(model, "magnitude", "magnitude")


def _entry_projection_f_normalize(model: SentencePredictor, patch: pytest.MonkeyPatch) -> None:
    original = model.spine.projection.forward
    patch.setattr(model.spine.projection, "forward", lambda x: F.normalize(original(x), dim=-1))


def _entry_projection_rms_overwrite(model: SentencePredictor, patch: pytest.MonkeyPatch) -> None:
    original = model.spine.projection.forward

    def mutant(x: torch.Tensor) -> torch.Tensor:
        out = original(x)
        return out * out.square().mean(dim=-1, keepdim=True).rsqrt()

    patch.setattr(model.spine.projection, "forward", mutant)


def _entry_positions_f_normalize(model: SentencePredictor, patch: pytest.MonkeyPatch) -> None:
    original = model.spine.positions.forward
    patch.setattr(model.spine.positions, "forward", lambda x: F.normalize(original(x), dim=-1))


ENTRY_MUTATIONS: dict[str, Callable[[SentencePredictor, pytest.MonkeyPatch], None]] = {
    "projection_f_normalize": _entry_projection_f_normalize,
    "projection_rms_overwrite": _entry_projection_rms_overwrite,
    "positions_f_normalize": _entry_positions_f_normalize,
}


@pytest.mark.parametrize("name", list(ENTRY_MUTATIONS))
def test_an_injected_entry_overwrite_is_detected(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The entry (projection and position adds) is checked against raw weights, not the module."""
    model = build("magnitude", "magnitude")
    assert findings(model, "magnitude", "magnitude") == []  # control: clean before the mutation
    ENTRY_MUTATIONS[name](model, monkeypatch)
    assert "spine_differs_from_unnormalized_reference" in findings(model, "magnitude", "magnitude")


def test_the_rms_overwrite_keeps_norms_far_from_one_yet_still_trips_the_reference() -> None:
    """The RMS mutation defeats a bare 'norm is not 1' check, so the reference check is load-bearing."""
    model = build("magnitude", "magnitude")
    with pytest.MonkeyPatch.context() as patch:
        _mutate_spine_rms_overwrite(patch)
        found = findings(model, "magnitude", "magnitude")
    assert "spine_norm_is_one" not in found
    assert "spine_differs_from_unnormalized_reference" in found


# Sphere-control and training-mode branches: each branch of the detector must fire on a
# mutation that only that branch can read.


def _sphere_interface_unnormalized(patch: pytest.MonkeyPatch) -> None:
    original = numerics_module.apply_interface

    def mutant(values: torch.Tensor, mode: str) -> torch.Tensor:
        return values.to(torch.float32) if mode == "sphere" else original(values, mode)  # pyright: ignore[reportArgumentType]

    patch.setattr(numerics_module, "apply_interface", mutant)
    patch.setattr(model_module, "apply_interface", mutant)


def _sphere_interface_unit_but_rotated(patch: pytest.MonkeyPatch) -> None:
    original = numerics_module.apply_interface

    def mutant(values: torch.Tensor, mode: str) -> torch.Tensor:
        result = original(values, mode)  # pyright: ignore[reportArgumentType]
        return F.normalize(result + 0.5, dim=-1) if mode == "sphere" else result

    patch.setattr(numerics_module, "apply_interface", mutant)
    patch.setattr(model_module, "apply_interface", mutant)


def _sphere_spine_scaled(patch: pytest.MonkeyPatch) -> None:
    original = SentenceSpine.forward

    def mutant(
        self: SentenceSpine, sentences: torch.Tensor, *, consumer: object = None
    ) -> torch.Tensor:
        states = original(self, sentences, consumer=consumer)  # pyright: ignore[reportArgumentType]
        return states * 1.5 if self.spec.carrier == "sphere" else states

    patch.setattr(SentenceSpine, "forward", mutant)


SPHERE_MUTATIONS: dict[str, tuple[Callable[[pytest.MonkeyPatch], None], str, str, str]] = {
    "interface_unnormalized": (
        _sphere_interface_unnormalized,
        "sentence_sphere_not_unit",
        "sphere",
        "magnitude",
    ),
    "interface_unit_but_rotated": (
        _sphere_interface_unit_but_rotated,
        "sentence_differs_from_normalized_reference",
        "sphere",
        "magnitude",
    ),
    "spine_scaled": (_sphere_spine_scaled, "spine_sphere_not_unit", "magnitude", "sphere"),
}


@pytest.mark.parametrize("name", list(SPHERE_MUTATIONS))
def test_a_broken_sphere_control_is_detected(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    apply, expected, mode, carrier = SPHERE_MUTATIONS[name]
    model = build(mode, carrier)
    assert findings(model, mode, carrier) == []  # control: clean before the mutation
    apply(monkeypatch)
    assert expected in findings(model, mode, carrier)


def test_a_rotated_unit_sentence_trips_only_the_reference_not_the_unit_check() -> None:
    """The reference branch is load-bearing: the unit check alone would pass this mutation."""
    model = build("sphere", "magnitude")
    with pytest.MonkeyPatch.context() as patch:
        _sphere_interface_unit_but_rotated(patch)
        found = findings(model, "sphere", "magnitude")
    assert "sentence_sphere_not_unit" not in found
    assert "sentence_differs_from_normalized_reference" in found


def _mutate_spine_training_gated_f_normalize(patch: pytest.MonkeyPatch) -> None:
    original = SentenceSpine.forward

    def mutant(
        self: SentenceSpine, sentences: torch.Tensor, *, consumer: object = None
    ) -> torch.Tensor:
        states = original(self, sentences, consumer=consumer)  # pyright: ignore[reportArgumentType]
        return F.normalize(states, dim=-1) if self.training else states

    patch.setattr(SentenceSpine, "forward", mutant)


def _mutate_codec_training_gated_f_normalize(patch: pytest.MonkeyPatch) -> None:
    original = SentenceEncoder.forward
    patch.setattr(
        SentenceEncoder,
        "forward",
        lambda self, tokens: (
            F.normalize(original(self, tokens), dim=-1) if self.training else original(self, tokens)
        ),
    )


@pytest.mark.parametrize(
    ("apply", "expected"),
    [
        (_mutate_spine_training_gated_f_normalize, "spine_norm_is_one"),
        (_mutate_codec_training_gated_f_normalize, "sentence_norm_is_one"),
    ],
)
def test_a_training_gated_normalization_is_seen_only_in_training_mode(
    apply: Callable[[pytest.MonkeyPatch], None], expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = build("magnitude", "magnitude")
    apply(monkeypatch)
    assert findings(model.eval(), "magnitude", "magnitude") == []  # invisible in eval mode
    assert expected in findings(model.train(), "magnitude", "magnitude")
