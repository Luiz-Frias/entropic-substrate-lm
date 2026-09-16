"""Unexpected runtime precision is a defect, never an implicit policy change."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn

import pytest
import torch
from tests.test_sentence_execution import engine

from entropic_substrate.sentence_prediction import execution as runtime
from entropic_substrate.sentence_prediction.execution import SentenceExecution
from entropic_substrate.sentence_prediction.numerics import UndefinedDirection
from entropic_substrate.sentence_prediction.runner import run_probe_training
from entropic_substrate.sentence_prediction.runtime_precision import require_execution_precision

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
        monkeypatch.setattr(runtime, "importance_evaluate", fail)
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
        run.evaluate()
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

    monkeypatch.setattr(runtime, "importance_evaluate", gap)
    with pytest.raises(UndefinedDirection):
        run.evaluate()
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

    monkeypatch.setattr(runtime, "importance_evaluate", gap)
    with pytest.raises(UndefinedDirection):
        run_probe_training(
            template.profile,
            template.source_root,
            output,
            max_source_bytes=1_000_000,
            max_artifact_bytes=10_000_000,
        )
    failure = json.loads(next(output.glob("*-failure.json")).read_text())
    assert failure["status"] == "scientific_gap"
    assert failure["error_type"] == "UndefinedDirection"
    assert failure["error_code"] == "undefined_zero_norm"
    assert failure["last_evaluation"]["affected_positions"] == 2
