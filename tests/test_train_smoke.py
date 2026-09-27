"""End-to-end CPU smoke tests for the trainer and CLI.

The headline test is :func:`test_five_step_tiny_train_decreases_loss`: five
steps on a tiny synthetic corpus, on CPU, in well under a minute, with the loss
required to actually go down.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import pytest
import torch

from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.core.substrate.field_encoder import FieldEncoder
from entropic_substrate.historical.e_series.train import (
    Trainer,
    cosine_lr,
    resolve_device,
    seed_everything,
)
from entropic_substrate.kernel.runtime.config import configs_dir, resolve_experiment_config
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


def test_missing_loss_fails_loud_not_silently() -> None:
    """The invariant used to be an ``assert``, which ``python -O`` strips: the
    eval path is a measurement path and must never fail open."""
    from entropic_substrate.core.substrate.backbone import LMOutput
    from entropic_substrate.historical.e_series.train import _require_loss

    out = LMOutput(logits=torch.zeros(1, 1, 2))
    with pytest.raises(RuntimeError, match="without targets"):
        _require_loss(out, "evaluate")
    with_loss = LMOutput(logits=torch.zeros(1, 1, 2), loss=torch.tensor(1.5))
    assert float(_require_loss(with_loss, "evaluate")) == pytest.approx(1.5)


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


def test_run_start_record_attests_the_frozen_parameter_mask(tmp_path: Path) -> None:
    """Unconditional, control arms included (review finding MAJOR-1).

    A key emitted only by frozen arms would itself be the arm signal the
    review packet exists to remove, so an unfrozen run attests the EMPTY
    mask rather than omitting the field.
    """
    trainer = Trainer(tiny_config(tmp_path))
    trainer.fit(verbose=False)
    start = next(r for r in trainer.iter_history() if r["event"] == "run_start")
    assert start["frozen_param_count"] == 0
    assert start["trainable_param_count"] == start["params"]
    assert start["frozen_mask_digest"] == hashlib.sha256(b"").hexdigest()


# --------------------------------------------------------------- helpers
def _run_tiny_cli(
    exp: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, object]]:
    # chdir FIRST. `--results` redirects only the metrics sink; `models_dir`
    # stays the relative default "models", so without an isolated cwd this
    # helper writes models/<exp>-tiny-final.pt into the repo working tree.
    monkeypatch.chdir(tmp_path)
    results = tmp_path / f"{exp}.jsonl"
    code = main(
        [
            "train",
            "--exp",
            exp,
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
    return [json.loads(line) for line in results.read_text().splitlines() if line]


def _tiny_trainer_overrides(tmp_path: Path, *, max_steps: int) -> dict[str, object]:
    return {
        "data": {"synthetic_bytes": 40_000, "seq_len": 32, "batch_size": 2},
        "train": {
            "max_steps": max_steps,
            "results_dir": str(tmp_path / "results"),
            "models_dir": str(tmp_path / "models"),
            "save_interval": 0,
        },
    }


# ------------------------------------------------------------------ CLI
def test_cli_train_tiny_e3_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = _run_tiny_cli("e3", tmp_path, monkeypatch)
    assert any(r["event"] == "eval" and "kappa" in r for r in records)
    assert (tmp_path / "models" / "e3-tiny-final.pt").exists(), "checkpoint escaped tmp_path"


def test_cli_train_tiny_e1_5c_field_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P5 pod-ready acceptance: `train --exp e1-5c-field --tiny` runs on CPU
    end to end through the real CLI, and the run_start record proves the
    tiny overlay actually reached the field encoder (field_d_model: 32)."""
    records = _run_tiny_cli("e1-5c-field", tmp_path, monkeypatch)
    start = next(r for r in records if r["event"] == "run_start")
    assert start["config"]["model"]["field_d_model"] == 32
    # Equality against an independently constructed tiny encoder, not `> 0`:
    # the config echo above only proves the CONFIG carried 32 — a build that
    # ignored field_* and constructed the full-size default would still pass
    # it. MEASURED 2026-08-19: tiny 32/1/2 encoder = 20,576 params vs
    # 426,624 at the full 128/2/4 — a 20x gap this equality cannot miss.
    expected_params = sum(
        p.numel()
        for p in FieldEncoder(
            resolve_experiment_config("e1-5c-field", tiny=True).model
        ).parameters()
    )
    assert start["field_encoder_params"] == expected_params
    assert (tmp_path / "models" / "e1-5c-field-tiny-final.pt").exists(), (
        "checkpoint escaped tmp_path"
    )


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
    from entropic_substrate.composition.entrypoints.cli import build_parser
    from entropic_substrate.kernel.runtime.config import EXPERIMENTS

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


def _write_e3_readout(tmp_path: Path, payload: str) -> None:
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    (results / "e3-tiny.jsonl").write_text(payload, encoding="utf-8")


@pytest.mark.parametrize(
    "kappa",
    ["[[]]", '[["a","b"]]', "[[NaN, NaN]]", "[[Infinity, 1.0]]", "[[0.1,0.2],[]]"],
)
def test_anchored_arm_refuses_malformed_kappa_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, kappa: str
) -> None:
    """A numerically failed run writes NaN that ``json`` parses back happily,
    and a killed one writes short rows. Either way the gate returns a
    controlled refusal — it never crashes, and never opens on a NaN spread."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENTROPIC_ALLOW_UNGATED", raising=False)
    _write_e3_readout(tmp_path, f'{{"event":"eval","nats_per_byte":1.5,"kappa":{kappa}}}\n')
    assert main(_anchored_argv(tmp_path)) == 2
    err = capsys.readouterr().err
    assert "DECISION-NEEDED" in err
    assert "ENTROPIC_ALLOW_UNGATED" in err


def test_malformed_telemetry_override_names_the_defect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The override covers malformed as well as missing — same premise, launch
    without a valid readout — but must not report it as merely missing."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENTROPIC_ALLOW_UNGATED", "1")
    _write_e3_readout(tmp_path, '{"event":"eval","nats_per_byte":1.5,"kappa":[[NaN]]}\n')
    assert main(_anchored_argv(tmp_path)) == 0
    assert "non-finite" in capsys.readouterr().err


# ---------------------------------------------- the gate reads what RUNS
def _config_argv(tmp_path: Path, config: Path, *, exp: str | None = None) -> list[str]:
    argv = ["train", "--config", str(config), "--tiny", "--device", "cpu", "--steps", "1"]
    if exp is not None:
        argv[1:1] = ["--exp", exp]
    return [*argv, "--results", str(tmp_path / "out.jsonl"), "--quiet"]


def test_anchored_config_path_is_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """``--config configs/e3-anchor.yaml`` is the supported invocation form and
    leaves ``args.exp`` None; gating off the raw flag skipped it entirely."""
    anchor_cfg = configs_dir() / "e3-anchor.yaml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENTROPIC_ALLOW_UNGATED", raising=False)
    assert main(_config_argv(tmp_path, anchor_cfg)) == 2
    assert "DECISION-NEEDED" in capsys.readouterr().err


def test_non_anchor_exp_cannot_launder_an_anchored_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """``--config`` wins over ``--exp``, so what runs is the anchored arm."""
    anchor_cfg = configs_dir() / "e3-anchor.yaml"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENTROPIC_ALLOW_UNGATED", raising=False)
    assert main(_config_argv(tmp_path, anchor_cfg, exp="e2")) == 2
    assert "DECISION-NEEDED" in capsys.readouterr().err


def test_anchored_exp_with_an_ungated_config_is_not_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The inverse: ``--exp e3-anchor --config configs/e3.yaml`` runs plain E3,
    so it must NOT be gated — the gate follows the resolved config, both ways."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENTROPIC_ALLOW_UNGATED", raising=False)
    assert main(_config_argv(tmp_path, configs_dir() / "e3.yaml", exp="e3-anchor")) == 0


def test_tiny_configs_for_every_rung_build_a_trainer(tmp_path: Path) -> None:
    """Smoke: every ladder rung must construct and take one step on CPU."""
    for exp in (
        "e0-byte",
        "e1-fisher",
        "e1-random",
        "e2",
        "e3",
        "e1-5a-input",
        "e1-5a-output",
        "e1-5a-both",
        "e1-5b-gram",
        "e1-5c-field",
    ):
        cfg = resolve_experiment_config(
            exp,
            tiny=True,
            overrides=_tiny_trainer_overrides(tmp_path, max_steps=1),
        )
        trainer = Trainer(cfg)
        metrics = trainer.train_step()
        assert math.isfinite(metrics["loss"]), exp


def test_tiny_overlay_shrinks_the_field_encoder() -> None:
    """P5: configs/tiny.yaml gains field_* shrink keys so an e1-5c-field CPU
    smoke does not pay the full 128-wide/2-layer p_phi estimator. Shrink-only
    - the arm's defining axis (state_source) must survive untouched."""
    tiny = resolve_experiment_config("e1-5c-field", tiny=True)
    assert tiny.model.field_d_model == 32
    assert tiny.model.field_n_layers == 1
    assert tiny.model.field_n_heads == 2
    assert tiny.model.state_source == "field"
    # Config -> construction: the assertions above only pin the resolved
    # CONFIG; these pin that FieldEncoder's constructor actually consumes
    # each field_* value (head count leaves param counts unchanged, so the
    # CLI e2e's param-equality alone cannot cover n_heads).
    encoder = FieldEncoder(tiny.model)
    assert len(encoder.blocks) == 1
    assert encoder.wte.weight.shape[1] == 32
    assert encoder.blocks[0].attn.n_heads == 2


def test_tiny_overlay_preserves_e1_5_arm_axes() -> None:
    """Pin: the --tiny overlay must never touch the field(s) that make an
    E1.5 arm what it is (embedding_freeze/tie_embeddings/gram_lambda/
    state_source). RED proof (run manually, not committed): add
    `state_source: table` under `model:` in configs/tiny.yaml and this test
    fails for e1-5c-field, since the overlay would silently turn the field
    arm back into a table-lookup arm under --tiny while the full run stays
    "field" - exactly the divergence between what is smoke-tested and what
    ships that this test exists to catch.

    Scope: iterates every entry in test_config._E1_5_ARM_AXES — all five
    E1.5 arms at time of writing — so a new arm added there is covered
    here with no edit."""
    from test_config import _E1_5_ARM_AXES  # tests/ is on sys.path (conftest.py)

    for exp, axis in _E1_5_ARM_AXES.items():
        full = resolve_experiment_config(exp).model
        tiny = resolve_experiment_config(exp, tiny=True).model
        for field_name in axis:
            assert getattr(tiny, field_name) == getattr(full, field_name), (exp, field_name)


@pytest.mark.parametrize(
    "exp",
    [
        "e1-5a-input",
        "e1-5a-output",
        "e1-5a-both",
        "e1-5b-gram",
        "e1-5c-field",
        "e1-5c-hidden",
        "e1-5c-prob",
    ],
)
def test_five_step_tiny_train_decreases_loss_for_e1_5_arms(tmp_path: Path, exp: str) -> None:
    """Mirrors test_five_step_tiny_train_decreases_loss but drives the REAL
    shipped configs/<exp>.yaml through the REAL configs/tiny.yaml overlay
    (the CLI's own path via --exp/--tiny), one parametrized case per new
    mechanism: input/output/both freeze, the Gram penalty, and the field
    encoder. `metrics["loss"]` is the CE-only loss (train.py:319) — the Gram
    penalty is added only to the optimized `total`, never to the reported
    metric — so the decrease assertion is comparable across every arm here.
    """
    cfg = resolve_experiment_config(
        exp,
        tiny=True,
        overrides=_tiny_trainer_overrides(tmp_path, max_steps=5),
    )
    started = time.monotonic()
    trainer = Trainer(cfg)
    history = trainer.fit(verbose=False)
    elapsed = time.monotonic() - started

    assert len(history) == 5
    losses = [h["loss"] for h in history]
    assert all(math.isfinite(v) for v in losses), (exp, losses)
    assert losses[-1] < losses[0], (exp, losses)
    assert elapsed < 60.0, f"{exp}: {elapsed:.1f}s"


def test_train_records_attest_a_paired_reproducible_data_order(tmp_path: Path) -> None:
    """The pairing evidence the E1 preregistration relies on: two runs with the
    same seed must RECORD the same data-order digest in their JSONL — exactly
    once, on a train record, with the attested batch count. Additive field
    only; every pre-existing record field is untouched.
    """

    def digest_record(workdir: Path) -> dict[str, object]:
        cfg = tiny_config(workdir)
        trainer = Trainer(cfg)
        trainer.fit(verbose=False)
        lines = Path(trainer.results_path).read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        carrying = [r for r in records if "data_order_digest" in r]
        assert len(carrying) == 1, f"expected exactly one attesting record, got {len(carrying)}"
        assert carrying[0]["event"] == "train"
        return carrying[0]

    first = digest_record(tmp_path / "a")
    second = digest_record(tmp_path / "b")
    assert first["data_order_digest"] == second["data_order_digest"]
    # tiny runs are 5 steps — shorter than the 32-batch window, so the record
    # attests the actual count rather than pretending to a full window.
    assert first["data_order_batches"] == second["data_order_batches"] == 5
