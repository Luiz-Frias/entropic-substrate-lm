"""Executable, seed-separated initialization contract for D1 state branches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import nn

from entropic_substrate.kernel.identity.d1_primitives import (
    EncoderInitializationIdentity,
    default_d1_encoder_initialization_identity,
)

__all__ = ["D1InitializationAttestation", "SeededD1BranchInitializer"]


def _parameter_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class D1InitializationAttestation:
    """Measured tensor identity produced by one registered initialization."""

    identity: EncoderInitializationIdentity
    seed: int
    online_parameter_digest: str
    target_parameter_digest: str

    def __post_init__(self) -> None:
        if type(self.identity) is not EncoderInitializationIdentity:
            msg = "D1 initialization attestation requires an initialization identity"
            raise TypeError(msg)
        if type(self.seed) is not int or self.seed < 0:
            msg = "D1 initialization attestation seed must be a nonnegative integer"
            raise ValueError(msg)
        if self.online_parameter_digest != self.target_parameter_digest:
            msg = "D1 initialized online and target branch parameter digests differ"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SeededD1BranchInitializer:
    """Materialize D1 v1 weights from a registered algorithm and run seed.

    The algorithm identity is construction-level evidence. ``seed`` remains a
    per-run execution identity, so paired arms share tensor initialization
    without changing their construction digest.
    """

    seed: int

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            msg = "D1 initialization seed must be a nonnegative integer"
            raise ValueError(msg)

    @property
    def identity(self) -> EncoderInitializationIdentity:
        return default_d1_encoder_initialization_identity()

    def require_identity(self, identity: EncoderInitializationIdentity) -> None:
        """Refuse a declared construction this executable does not implement."""
        if type(identity) is not EncoderInitializationIdentity:
            msg = "D1 initializer identity must be an EncoderInitializationIdentity"
            raise TypeError(msg)
        if identity != self.identity:
            msg = (
                f"D1 initializer {self.identity.name!r} does not materialize "
                f"declared identity {identity.name!r}/{identity.construction_digest}"
            )
            raise ValueError(msg)

    def initialize_pair(self, online: nn.Module, target: nn.Module) -> D1InitializationAttestation:
        """Initialize ``online`` once and copy it into independent target storage."""
        self._validate_pair(online, target)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        handled: set[int] = set()
        with torch.no_grad():
            for module in online.modules():
                if isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)
                    handled.add(id(module.weight))
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=generator)
                    handled.add(id(module.weight))
                    linear_bias: object = getattr(module, "bias", None)
                    if isinstance(linear_bias, nn.Parameter):
                        nn.init.zeros_(linear_bias)
                        handled.add(id(linear_bias))
                elif isinstance(module, nn.LayerNorm):
                    layer_norm_weight: object = getattr(module, "weight", None)
                    if isinstance(layer_norm_weight, nn.Parameter):
                        nn.init.ones_(layer_norm_weight)
                        handled.add(id(layer_norm_weight))
                    layer_norm_bias: object = getattr(module, "bias", None)
                    if isinstance(layer_norm_bias, nn.Parameter):
                        nn.init.zeros_(layer_norm_bias)
                        handled.add(id(layer_norm_bias))
            unhandled = [
                name for name, value in online.named_parameters() if id(value) not in handled
            ]
            if unhandled:
                msg = f"D1 initialization has no registered rule for parameters {unhandled}"
                raise ValueError(msg)
            target.load_state_dict(online.state_dict(), strict=True)
        self._validate_pair(online, target, require_equal=True)
        return D1InitializationAttestation(
            identity=self.identity,
            seed=self.seed,
            online_parameter_digest=_parameter_digest(online),
            target_parameter_digest=_parameter_digest(target),
        )

    @staticmethod
    def _validate_pair(
        online: nn.Module, target: nn.Module, *, require_equal: bool = False
    ) -> None:
        if type(online) is not type(target):
            msg = "D1 online and target initialization modules must have the same type"
            raise ValueError(msg)
        online_parameters = dict(online.named_parameters())
        target_parameters = dict(target.named_parameters())
        if online_parameters.keys() != target_parameters.keys():
            msg = "D1 online and target initialization parameter maps differ"
            raise ValueError(msg)
        for name, online_parameter in online_parameters.items():
            target_parameter = target_parameters[name]
            if (
                online_parameter.shape != target_parameter.shape
                or online_parameter.dtype != target_parameter.dtype
                or online_parameter.device != target_parameter.device
            ):
                msg = f"D1 online and target initialization parameter mismatch for {name!r}"
                raise ValueError(msg)
            if online_parameter.device.type != "cpu":
                msg = "D1 branch initialization must occur on CPU before device placement"
                raise ValueError(msg)
            if online_parameter.data_ptr() == target_parameter.data_ptr():
                msg = f"D1 online and target initialization storage is shared for {name!r}"
                raise ValueError(msg)
            if require_equal and not torch.equal(online_parameter, target_parameter):
                msg = f"D1 online and target initialized values differ for {name!r}"
                raise ValueError(msg)
