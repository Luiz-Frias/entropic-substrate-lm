"""End-to-end CPU smoke tests for the trainer and CLI.

The headline test is :func:`test_five_step_tiny_train_decreases_loss`: five
steps on a tiny synthetic corpus, on CPU, in well under a minute, with the loss
required to actually go down.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pytest
import torch

from entropic_substrate.cli import main
from entropic_substrate.config import resolve_experiment_config
from entropic_substrate.train import Trainer, cosine_lr, resolve_device, seed_everything
from helpers import tiny_config

pytestmark = pytest.mark.unit


def test_seed_everything_is_reproducible() -> None:
    seed_everything(3)
    a = torch.randn(4)
    seed_everything(3)
    assert torch.equal(a, torch.randn(4))


def test_resolve_device_honours_explicit_spec() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


def test_cosine_schedule_warms_up_then_decays() -> None:
    kwargs = {"base_lr": 1.0, "warmup": 10, "total": 100, "min_ratio": 0.1}
    assert cosine_lr(0, **kwargs) == pytest.approx(0.1)
    assert cosine_lr(9, **kwargs) == pytest.approx(1.0)
    assert cosine_lr(10, **kwargs) == pytest.approx(1.0, abs=1e-6)
    assert cosine_lr(55, **kwargs) < 1.0
    assert cosine_lr(100, **kwargs) == pytest.approx(0.1)
    assert cosine_lr(500, **kwargs) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("block", "norm"),
    [("baseline", "layernorm"), ("ngpt", "ngpt_norm"), ("ngpt", "kappa_moe")],
)
def test_five_step_tiny_train_decreases_loss(tmp_path: Path, block: str, norm: str) -> None:
    cfg = tiny_config(tmp_path, block=block, norm=norm)
    cfg.name = f"smoke-{block}-{norm}"
    started = time.time()
    trainer = Trainer(cfg)
    history = trainer.fit(verbose=False)
    elapsed = time.time() - started

    assert len(history) == 5
    losses = [h["loss"] for h in history]
    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0], losses
    assert elapsed < 60.0, f"{elapsed:.1f}s"


def test_nats_per_byte_is_the_loss_for_byte_models(tmp_path: Path) -> None:
    trainer = Trainer(tiny_config(tmp_path))
    metrics = trainer.train_step()
    assert trainer.meta.bytes_per_token == 1.0
    assert metrics["nats_per_byte"] == pytest.approx(metrics["loss"])


def test_nats_per_byte_divides_by_measured_bytes_per_token(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, tokenizer="gpt2")
    cfg.data.synthetic_bytes = 60_000
    trainer = Trainer(cfg)
    bpt = trainer.meta.bytes_per_token
    assert bpt > 1.0 and bpt != 4.0
    result = trainer.evaluate(1)
    assert result.nats_per_byte == pytest.approx(result.loss / bpt)
    assert result.bits_per_byte == pytest.approx(result.nats_per_byte / math.log(2))


def test_eval_is_deterministic(tmp_path: Path) -> None:
    trainer = Trainer(tiny_config(tmp_path))
    first = trainer.evaluate(2)
    second = trainer.evaluate(2)
    assert first.loss == pytest.approx(second.loss, rel=1e-6)


def test_ngpt_weights_stay_unit_norm_after_optimiser_steps(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, block="ngpt", norm="ngpt_norm")
    trainer = Trainer(cfg)
    for _ in range(3):
        trainer.train_step()
    weight = trainer.raw_model.blocks[0].attn.q_proj.weight
    norms = weight.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


def test_checkpoint_roundtrip_resumes_exactly(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    trainer = Trainer(cfg)
    for _ in range(3):
        trainer.train_step()
    path = trainer.save_checkpoint("test")
    expected = trainer.train_step()["loss"]

    revived = Trainer(tiny_config(tmp_path))
    revived.load_checkpoint(path)
    assert revived.step == 3
    assert revived.train_step()["loss"] == pytest.approx(expected, rel=1e-5)


def test_jsonl_records_kappa_and_gate_usage(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, block="ngpt", norm="kappa_moe")
    cfg.name = "smoke-e3-metrics"
    cfg.train.eval_interval = 2
    trainer = Trainer(cfg)
    trainer.fit(verbose=False)

    records = list(trainer.iter_history())
    evals = [r for r in records if r["event"] == "eval"]
    assert evals, records
    last = evals[-1]
    assert len(last["kappa"]) == 2 * cfg.model.n_layers
    assert all(len(k) == cfg.model.n_components for k in last["kappa"])
    assert all(sum(g) == pytest.approx(cfg.model.top_k, abs=1e-4) for g in last["gate_usage"])
    assert all(math.isfinite(e) for e in last["router_entropy"])
    assert last["bytes_per_token"] == 1.0

    trains = [r for r in records if r["event"] == "train"]
    assert all(r["aux_loss"] > 0.0 for r in trains)


def test_curvature_is_actually_learned(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path, block="ngpt", norm="kappa_moe")
    cfg.train.max_steps = 8
    trainer = Trainer(cfg)
    before = trainer.raw_model.kappa_modules()[0].kappa.detach().clone()
    trainer.fit(verbose=False)
    after = trainer.raw_model.kappa_modules()[0].kappa.detach()
    assert not torch.equal(before, after)
    assert float(after.min()) >= -5.0 and float(after.max()) <= 5.0


def test_run_start_record_captures_the_config(tmp_path: Path) -> None:
    trainer = Trainer(tiny_config(tmp_path))
    trainer.fit(verbose=False)
    records = list(trainer.iter_history())
    start = next(r for r in records if r["event"] == "run_start")
    assert start["config"]["model"]["d_model"] == 32
    assert start["params"] > 0
    assert json.dumps(start)  # round-trips


# ------------------------------------------------------------------ CLI
def test_cli_train_tiny_e3_end_to_end(tmp_path: Path) -> None:
    results = tmp_path / "e3.jsonl"
    code = main(
        [
            "train",
            "--exp",
            "e3",
            "--tiny",
            "--device",
            "cpu",
            "--steps",
            "4",
            "--results",
            str(results),
            "--quiet",
        ]
    )
    assert code == 0
    records = [json.loads(line) for line in results.read_text().splitlines() if line]
    assert any(r["event"] == "eval" and "kappa" in r for r in records)


def test_cli_init_fisher_writes_a_table(tmp_path: Path) -> None:
    out = tmp_path / "fisher.pt"
    assert (
        main(["init-fisher", "--corpus", "synthetic", "--d-model", "320", "--out", str(out)]) == 0
    )
    table = torch.load(out, weights_only=True)
    assert table.shape == (256, 320)
    torch.testing.assert_close(table.norm(dim=1), torch.ones(256), atol=1e-5, rtol=0)


def test_cli_eval_reads_a_checkpoint(tmp_path: Path, capsys) -> None:
    cfg = tiny_config(tmp_path, block="ngpt", norm="kappa_moe")
    trainer = Trainer(cfg)
    trainer.train_step()
    ckpt = trainer.save_checkpoint("cli")

    assert main(["eval", "--ckpt", str(ckpt), "--device", "cpu", "--batches", "1", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["nats_per_byte"] > 0
    assert payload["kappa"]


def test_cli_train_requires_exp_or_config(capsys) -> None:
    assert main(["train", "--quiet"]) == 2
    assert "--exp or --config" in capsys.readouterr().err


def test_cli_parser_exposes_the_whole_ladder() -> None:
    from entropic_substrate.cli import build_parser
    from entropic_substrate.config import EXPERIMENTS

    parser = build_parser()
    args = parser.parse_args(["train", "--exp", "e3", "--tiny"])
    assert args.exp == "e3" and args.tiny
    for exp in EXPERIMENTS:
        assert parser.parse_args(["train", "--exp", exp]).exp == exp


# --------------------------------------------- anchored-arm preflight (E3'')
def _anchored_argv(tmp_path: Path) -> list[str]:
    return [
        "train",
        "--exp",
        "e3-anchor",
        "--tiny",
        "--device",
        "cpu",
        "--steps",
        "1",
        "--results",
        str(tmp_path / "out.jsonl"),
        "--quiet",
    ]


def _e3_eval_record() -> str:
    record = {
        "run": "e3-tiny",
        "step": 50,
        "event": "eval",
        "nats_per_byte": 1.5,
        "kappa": [[-0.9, 0.1, 1.2, -0.4], [-0.2, 0.0, 0.3, 0.1]],
    }
    return json.dumps(record) + "\n"


def test_anchored_arm_refuses_to_launch_without_the_e3_readout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Sanity, not judgment: the arm is gated on the E3 kappa readout existing
    at all. Whether the divergence is big enough stays a human call."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENTROPIC_ALLOW_UNGATED", raising=False)
    assert main(_anchored_argv(tmp_path)) == 2
    assert "DECISION-NEEDED" in capsys.readouterr().err


def test_anchored_arm_launches_once_the_e3_readout_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENTROPIC_ALLOW_UNGATED", raising=False)
    results = tmp_path / "results"
    results.mkdir()
    (results / "e3-tiny.jsonl").write_text(_e3_eval_record(), encoding="utf-8")
    assert main(_anchored_argv(tmp_path)) == 0
    assert "head spread" in capsys.readouterr().out


def test_anchored_arm_override_is_loud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENTROPIC_ALLOW_UNGATED", "1")
    assert main(_anchored_argv(tmp_path)) == 0
    assert "ENTROPIC_ALLOW_UNGATED" in capsys.readouterr().err


def test_tiny_configs_for_every_rung_build_a_trainer(tmp_path: Path) -> None:
    """Smoke: every ladder rung must construct and take one step on CPU."""
    for exp in ("e0-byte", "e1-fisher", "e1-random", "e2", "e3"):
        cfg = resolve_experiment_config(
            exp,
            tiny=True,
            overrides={
                "data": {"synthetic_bytes": 40_000, "seq_len": 32, "batch_size": 2},
                "train": {
                    "max_steps": 1,
                    "results_dir": str(tmp_path / "results"),
                    "models_dir": str(tmp_path / "models"),
                    "save_interval": 0,
                },
            },
        )
        trainer = Trainer(cfg)
        metrics = trainer.train_step()
        assert math.isfinite(metrics["loss"]), exp
