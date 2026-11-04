"""Unexpected runtime precision is a defect, never an implicit policy change."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import torch
from tests.latent_usage_support import POSTERIOR_MEAN
from tests.test_sentence_execution import engine
from tests.test_sentence_profile import profile
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.core.sentence import runtime_precision as precision_module
from entropic_substrate.core.sentence.numerics import UndefinedDirection
from entropic_substrate.core.sentence.runtime_precision import require_execution_precision
from entropic_substrate.ladder.run import execution as runtime
from entropic_substrate.ladder.run.execution import SentenceExecution
from entropic_substrate.ladder.run.runner import run_probe_training
from entropic_substrate.ladder.run.worker_execution import _runtime

pytestmark = pytest.mark.unit


def test_precision_admission_is_read_only_and_accepts_explicit_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    before = (torch.get_default_dtype(), torch.get_float32_matmul_precision())
    observed = require_execution_precision()
    assert observed["effective_tf32"] is False
    assert before == (torch.get_default_dtype(), torch.get_float32_matmul_precision())


@pytest.mark.parametrize(
    "setting",
    ["dtype", "matmul", "tf32_env", "nvidia_env", "effective_tf32", "generic", "cuda_matmul"],
)
def test_precision_guard_refuses_ambient_changes(
    monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    if setting == "dtype":
        monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float64)
    elif setting == "matmul":
        monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high")
    elif setting == "tf32_env":
        monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    elif setting == "nvidia_env":
        monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "1")
    else:
        matmul = SimpleNamespace(fp32_precision="none", allow_tf32=False)
        backends = SimpleNamespace(fp32_precision="none", cuda=SimpleNamespace(matmul=matmul))
        if setting == "effective_tf32":
            matmul.allow_tf32 = True
        elif setting == "generic":
            backends.fp32_precision = "tf32"
        else:
            matmul.fp32_precision = "tf32"
        monkeypatch.setattr(torch, "backends", backends)
    with pytest.raises(ValueError, match="precision"):
        require_execution_precision()


def test_initialization_refuses_before_rng_change_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = engine(tmp_path)
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float64)
    with pytest.raises(ValueError, match="precision"):
        SentenceExecution(template.profile, template.source_root, max_source_bytes=1_000_000)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(ValueError, match="precision"):
        run_probe_training(
            template.profile,
            template.source_root,
            output,
            max_source_bytes=1_000_000,
            max_artifact_bytes=10_000_000,
            latent_usage=POSTERIOR_MEAN,
        )
    failure = json.loads(next(output.glob("*-failure.json")).read_text())
    assert failure["phase"] == "initialization" and failure["step"] == 0
    assert failure["run_class"] == "local_tiny" and failure["scientific_authority"] == "none"
    assert failure["status"] == "execution_defect"
    assert failure["error_code"] == "execution_precision_contract"


def test_step_precision_drift_halts_before_optimizer_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = engine(tmp_path)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "high")
    with pytest.raises(ValueError, match="precision"):
        run.advance()
    assert run.failed and run.step == 0 and not run.optimizer.state


@pytest.mark.parametrize("where", ["scoring", "sampler", "timer_start", "timer_stop"])
def test_evaluation_defect_halts_and_retains_partial_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()
    run.advance()

    def fail(*args: object, **kwargs: object) -> NoReturn:
        raise RuntimeError("injected evaluation defect")

    if where == "scoring":
        monkeypatch.setattr(runtime, "evaluate_proposals", fail)
    elif where == "sampler":
        monkeypatch.setattr(run.validation_sampler, "restore", fail)
    else:

        class Timer:
            def start(self, phase: object) -> object:
                if where == "timer_start":
                    fail()
                return object()

            def stop(self, ticket: object, phase: object) -> float:
                fail()

        run.timer = Timer()
    with pytest.raises(RuntimeError, match="injected evaluation defect"):
        run.evaluate(POSTERIOR_MEAN)
    assert run.failed and run.last_evaluation["status"] == "execution_defect"
    if where in ("scoring", "timer_stop"):
        assert run.last_evaluation["current_source_examples"]
    assert run.model.training
    with pytest.raises(ValueError, match="failed"):
        run.advance()


def test_evaluation_zero_direction_remains_a_scientific_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = engine(tmp_path)
    run.advance()
    run.advance()
    run.begin_prediction()

    def gap(*args: object, **kwargs: object) -> NoReturn:
        raise UndefinedDirection(2)

    monkeypatch.setattr(runtime, "evaluate_proposals", gap)
    with pytest.raises(UndefinedDirection):
        run.evaluate(POSTERIOR_MEAN)
    assert run.last_evaluation["status"] == "scientific_gap"
    assert run.last_evaluation["affected_positions"] == 2


def test_training_driver_retains_evaluation_gap_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = engine(tmp_path)
    output = tmp_path / "run"
    output.mkdir()

    def gap(*args: object, **kwargs: object) -> NoReturn:
        raise UndefinedDirection(2)

    monkeypatch.setattr(runtime, "evaluate_proposals", gap)
    with pytest.raises(UndefinedDirection):
        run_probe_training(
            template.profile,
            template.source_root,
            output,
            max_source_bytes=1_000_000,
            max_artifact_bytes=10_000_000,
            latent_usage=POSTERIOR_MEAN,
        )
    failure = json.loads(next(output.glob("*-failure.json")).read_text())
    assert failure["status"] == "scientific_gap"
    assert failure["error_type"] == "UndefinedDirection"
    assert failure["error_code"] == "undefined_zero_norm"
    assert failure["last_evaluation"]["affected_positions"] == 2


# --- FOR-1672: the OBSERVED autocast state beside the declared precision ---------------------
#
# MEASURED 2026-10-22 on torch 2.13.0 (CPU host): inside torch.autocast("cpu",
# dtype=torch.bfloat16, enabled=True) torch.is_autocast_enabled("cpu") is True and
# torch.get_autocast_dtype("cpu") is torch.bfloat16. Inside enabled=False the enabled flag is
# False and the dtype STILL reads torch.bfloat16 (autocast.__enter__ stores the requested dtype
# unconditionally), so `enabled` is the only independent observation.

_BASE_KEYS = {
    "default_dtype",
    "matmul_precision",
    "generic_fp32_precision",
    "cuda_matmul_fp32_precision",
    "effective_tf32",
    "torch_tf32_override",
    "nvidia_tf32_override",
}


def _observe(**changes: object) -> dict[str, object]:
    return precision_module.require_observed_execution_precision(replace(profile(), **changes))  # type: ignore[arg-type]


def test_bf16_cpu_profile_records_what_torch_reports_for_cpu_bf16_autocast() -> None:
    record = _observe(precision="bf16")
    # Measured value, not assumed: torch 2.13.0 reports CPU bf16 autocast ENABLED.
    assert record["autocast"] == {"enabled": True, "dtype": "torch.bfloat16", "device_type": "cpu"}
    assert record["declared_precision"] == "bf16"


def test_fp32_profile_records_autocast_disabled_beside_the_declared_value() -> None:
    record = _observe(precision="fp32")
    autocast = record["autocast"]
    assert isinstance(autocast, dict) and autocast["enabled"] is False
    # Measured: the dtype echoes the requested bf16 even though autocast is disabled.
    assert autocast == {"enabled": False, "dtype": "torch.bfloat16", "device_type": "cpu"}
    assert record["declared_precision"] == "fp32"


def test_observed_record_keeps_the_seven_guard_keys_and_adds_exactly_two() -> None:
    record = _observe(precision="bf16")
    assert set(record) == _BASE_KEYS | {"declared_precision", "autocast"}
    assert {k: record[k] for k in _BASE_KEYS} == require_execution_precision()
    autocast = record["autocast"]
    assert isinstance(autocast, dict) and set(autocast) == {"enabled", "dtype", "device_type"}
    assert json.loads(json.dumps(record)) == record


def test_observed_form_refuses_exactly_as_the_guard_and_before_opening_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float64)
    with pytest.raises(ValueError, match="precision") as bare:
        require_execution_precision()

    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("autocast opened before the precision refusal")

    monkeypatch.setattr(torch, "autocast", forbidden)
    with pytest.raises(ValueError, match="precision") as observed:
        _observe(precision="bf16")
    assert str(observed.value) == str(bare.value)
    assert type(observed.value) is type(bare.value)


def test_cuda_profile_whose_autocast_torch_disables_records_declared_beside_observed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deterministic on any host: torch's own "CUDA AMP unavailable" branch is forced.
    monkeypatch.setattr(torch.cuda.amp.common, "amp_definitely_not_available", lambda: True)
    with pytest.warns(UserWarning, match="Disabling autocast"):
        record = _observe(device="cuda", precision="bf16")
    assert record["declared_precision"] == "bf16"
    assert record["autocast"] == {
        "enabled": False,
        "dtype": "torch.bfloat16",
        "device_type": "cuda",
    }


def test_probe_failure_propagates_torch_text_and_is_never_recorded_as_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda.amp.common, "amp_definitely_not_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="does not support bfloat16"):
        _observe(device="cuda", precision="bf16")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_probe_restores_ambient_autocast_state(precision: str) -> None:
    before = (torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"))
    nesting = torch.autocast_increment_nesting()
    torch.autocast_decrement_nesting()
    _observe(precision=precision)
    assert before == (torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"))
    again = torch.autocast_increment_nesting()
    torch.autocast_decrement_nesting()
    assert nesting == again


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_run_start_runtime_record_carries_declared_and_observed_precision(
    tmp_path: Path, precision: str
) -> None:
    request, root = prepared(tmp_path)
    request = replace(request, profile=replace(request.profile, precision=precision))  # type: ignore[arg-type]
    runtime_record = _runtime(request, root, None)
    recorded = runtime_record["precision"]
    assert isinstance(recorded, dict)
    assert recorded["declared_precision"] == precision
    autocast = recorded["autocast"]
    assert isinstance(autocast, dict)
    assert autocast["enabled"] is (precision == "bf16") and autocast["device_type"] == "cpu"
    assert {k: recorded[k] for k in _BASE_KEYS} == require_execution_precision()
