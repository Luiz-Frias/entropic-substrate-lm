"""Review regressions for explicit devices and the numerical consumer boundary."""

import pytest
import torch
import yaml

from d1_runtime_fixtures import runtime_batch, runtime_system
from entropic_substrate.kernel.identity.d1_primitives import D1Arm
from entropic_substrate.ladder.register.d1_config import d1_configs_dir, load_d1_profile

pytestmark = pytest.mark.unit


def test_cpu_tiny_cannot_silently_select_an_accelerator(tmp_path) -> None:
    payload = yaml.safe_load((d1_configs_dir() / "d1-tiny.yaml").read_text())
    payload["train"]["device"] = "auto"
    overlay = tmp_path / "tiny.yaml"
    overlay.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="CPU"):
        load_d1_profile(d1_configs_dir() / "d1-both.yaml", tiny_overlay_path=overlay)


@pytest.mark.parametrize("dtype", [torch.int64, torch.float64])
def test_residual_callback_cannot_change_the_numeric_contract(dtype) -> None:
    system = runtime_system(D1Arm.BOTH)
    encoder = system.online_branch.encoder
    inputs = system.online_branch.adapter(runtime_batch().online_observations, system.draft.codec)
    with pytest.raises(ValueError, match="dtype"):
        encoder(inputs, final_residual=lambda _state, residual: residual.to(dtype))


def test_residual_callback_can_promote_bf16_to_the_fp32_island() -> None:
    system = runtime_system(D1Arm.BOTH).to(torch.bfloat16)
    encoder = system.online_branch.encoder
    inputs = system.online_branch.adapter(runtime_batch().online_observations, system.draft.codec)
    result = encoder(inputs, final_residual=lambda _state, residual: residual.float())
    assert result.values.dtype == torch.float32
