"""E1.5c Repair 1: the standalone ``p_phi`` pretrain lane (producer side).

``FieldEncoder._load_frozen_weights`` DEFINES the wire format the frozen
regime loads (``{"format_version": ..., "model": <FieldEncoder.state_dict()>}``
through ``normalize_checkpoint``); until now nothing in the repo wrote one.
``entropic_substrate.field_pretrain`` is that producer, and this module pins
the contract between the two halves.

RED-first convention (mirrors ``test_field_calibration.py``'s module
docstring): before ``entropic_substrate.field_pretrain`` exists, every test
here fails at COLLECTION on ``ModuleNotFoundError`` rather than on a wrong
value — that absence IS the RED proof for the seam itself. The two
guard-teeth tests below carry their own, sharper RED proof: each names the
exact line it kills, and the two fail-loud messages are deliberately
distinguishable ("train" vs "eval") so deleting one guard cannot be masked
by the other one firing later in the same run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pytest
import torch
import yaml

from entropic_substrate.checkpoint import CHECKPOINT_FORMAT_VERSION, normalize_checkpoint
from entropic_substrate.config import Config
from entropic_substrate.field_pretrain import (
    FieldPretrainGap,
    add_field_pretrain_parser,
    cmd_field_pretrain,
    read_field_pretrain_series,
    run_field_pretrain,
)
from entropic_substrate.models.field_encoder import FieldEncoder
from helpers import assert_bitwise_equal, tiny_config

pytestmark = pytest.mark.unit


def _pretrain_config(tmp_path: Path, **overrides: object) -> Config:
    """A CPU-sized config for the pretrain lane (field knobs kept tiny)."""
    defaults: dict[str, object] = {
        "state_source": "field",
        "field_d_model": 32,
        "field_n_layers": 1,
        "field_n_heads": 2,
    }
    defaults.update(overrides)
    cfg = tiny_config(tmp_path, **defaults)
    cfg.train.max_steps = 3
    return cfg


def _frozen_encoder(cfg: Config, ckpt: Path) -> FieldEncoder:
    """A fresh encoder in the frozen regime, loading ``ckpt`` at construction."""
    model_cfg = dataclasses.replace(cfg.model, field_mode="frozen", field_ckpt_path=str(ckpt))
    return FieldEncoder(model_cfg)


# --------------------------------------------------------------- roundtrip
def test_checkpoint_roundtrips_into_a_frozen_field_encoder_bitwise(tmp_path: Path) -> None:
    """THE contract test: trained weights -> file -> frozen load, bit for bit.

    Compared against the LIVE trained encoder (``result.encoder``), not
    against a second read of the same file — a file-vs-file comparison would
    prove only that the loader applied *something*, not that it applied what
    training produced. Buffers are compared too: ``lift`` is E1's fixed
    semi-orthogonal map, and a frozen field that silently re-drew it would
    put ``z_t`` in a different chart than the checkpoint's.
    """
    cfg = _pretrain_config(tmp_path)
    out = tmp_path / "field-pphi.pt"
    result = run_field_pretrain(
        cfg, out_path=out, results_path=tmp_path / "field-pretrain.jsonl", device="cpu"
    )

    assert result.steps == 3
    assert result.out_path == out
    assert out.exists()

    frozen = _frozen_encoder(cfg, out)
    trained_state = result.encoder.state_dict()
    frozen_state = frozen.state_dict()
    assert set(frozen_state) == set(trained_state)
    for name, tensor in trained_state.items():
        assert_bitwise_equal(frozen_state[name], tensor)
    assert not any(p.requires_grad for p in frozen.parameters()), (
        "field_mode='frozen' must leave every parameter frozen after the load"
    )


def test_saved_checkpoint_is_admissible_to_normalize_checkpoint(tmp_path: Path) -> None:
    """The stamp is what keeps the loader from having to guess (checkpoint.py)."""
    cfg = _pretrain_config(tmp_path)
    out = tmp_path / "field-pphi.pt"
    run_field_pretrain(
        cfg, out_path=out, results_path=tmp_path / "field-pretrain.jsonl", device="cpu"
    )

    raw = torch.load(out, map_location="cpu", weights_only=True)
    assert raw["format_version"] == CHECKPOINT_FORMAT_VERSION
    state = normalize_checkpoint(raw, source=str(out), for_resume=False)
    assert set(state) >= {"format_version", "step", "model", "config"}
    assert state["step"] == 3
    assert state["config"]["model"]["field_d_model"] == 32


# ------------------------------------------------------------------ signal
def test_field_ce_decreases_over_a_short_run(tmp_path: Path) -> None:
    """Smoke: the lane actually optimises p_phi, it does not merely spin."""
    cfg = _pretrain_config(tmp_path)
    cfg.train.max_steps = 20
    cfg.train.eval_interval = 10
    results = tmp_path / "field-pretrain.jsonl"
    result = run_field_pretrain(
        cfg, out_path=tmp_path / "field-pphi.pt", results_path=results, device="cpu"
    )

    series = read_field_pretrain_series(results)
    assert [record.step for record in series] == [10, 20]
    assert series[-1].field_ce < series[0].field_ce
    assert result.final_eval_ce == series[-1].field_ce
    # Fewer than five evals: the last-k mean is taken over what exists.
    assert result.last5_eval_ce_mean == pytest.approx(sum(r.field_ce for r in series) / len(series))


def test_same_seed_reproduces_the_eval_ce_series(tmp_path: Path) -> None:
    """CPU determinism — the precondition for any paired-seed comparison."""
    first_results = tmp_path / "a.jsonl"
    second_results = tmp_path / "b.jsonl"
    for results in (first_results, second_results):
        cfg = _pretrain_config(tmp_path)
        cfg.train.max_steps = 4
        cfg.train.eval_interval = 2
        run_field_pretrain(
            cfg, out_path=tmp_path / "field-pphi.pt", results_path=results, device="cpu"
        )

    first = [(r.step, r.field_ce) for r in read_field_pretrain_series(first_results)]
    second = [(r.step, r.field_ce) for r in read_field_pretrain_series(second_results)]
    assert first == second


def test_run_start_record_carries_the_config_and_seed(tmp_path: Path) -> None:
    cfg = _pretrain_config(tmp_path)
    results = tmp_path / "field-pretrain.jsonl"
    run_field_pretrain(cfg, out_path=tmp_path / "field-pphi.pt", results_path=results, device="cpu")

    records = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
    start = records[0]
    assert start["type"] == "field_pretrain_run_start"
    assert start["seed"] == cfg.train.seed
    assert start["config"]["model"]["field_n_layers"] == 1
    assert start["field_encoder_params"] > 0
    assert all(r["type"] == "field_pretrain_eval" for r in records[1:])


# ------------------------------------------------------------ guard teeth
def test_nonfinite_train_loss_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kills the train-loop line ``_require_finite(loss, "train", step)``.

    The poison stays DIFFERENTIABLE (``real_loss * nan``) so the NaN reaches
    the parameters exactly as a genuine numerical failure would. With the
    train guard deleted, the run continues, the eval at the final step then
    raises its OWN "non-finite eval loss" — a different message this
    ``match=`` does not accept, so the test goes RED rather than passing on
    the wrong guard.
    """
    import entropic_substrate.field_pretrain as field_pretrain

    real = field_pretrain.field_cross_entropy

    def poisoned(
        encoder: FieldEncoder, x: torch.Tensor, y: torch.Tensor, *, device_type: str
    ) -> torch.Tensor:
        return real(encoder, x, y, device_type=device_type) * float("nan")

    monkeypatch.setattr(field_pretrain, "field_cross_entropy", poisoned)
    cfg = _pretrain_config(tmp_path)
    with pytest.raises(RuntimeError, match="non-finite train loss"):
        run_field_pretrain(
            cfg,
            out_path=tmp_path / "field-pphi.pt",
            results_path=tmp_path / "field-pretrain.jsonl",
            device="cpu",
        )


def test_nonfinite_eval_loss_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kills the eval line ``_require_finite(loss, "eval", step)``.

    Poisons ONLY the no-grad (eval) call, so the training steps stay finite
    and this test cannot pass on the train guard by accident. With the eval
    guard deleted, the run returns a NaN ``final_eval_ce`` and writes it to
    the JSONL — no exception at all, so ``pytest.raises`` goes RED.
    """
    import entropic_substrate.field_pretrain as field_pretrain

    real = field_pretrain.field_cross_entropy

    def poisoned(
        encoder: FieldEncoder, x: torch.Tensor, y: torch.Tensor, *, device_type: str
    ) -> torch.Tensor:
        loss = real(encoder, x, y, device_type=device_type)
        return loss if torch.is_grad_enabled() else loss * float("nan")

    monkeypatch.setattr(field_pretrain, "field_cross_entropy", poisoned)
    cfg = _pretrain_config(tmp_path)
    with pytest.raises(RuntimeError, match="non-finite eval loss"):
        run_field_pretrain(
            cfg,
            out_path=tmp_path / "field-pphi.pt",
            results_path=tmp_path / "field-pretrain.jsonl",
            device="cpu",
        )


# -------------------------------------------------------------- refusals
def test_refuses_a_non_byte_tokenizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kills ``_refuse_incompatible``'s ``tokenizer != "byte"`` arm.

    ``build_corpus`` is poisoned to prove the refusal is EARLY, which is the
    whole point of restating a check ``FieldEncoder.__init__`` also makes:
    with that arm deleted, this run would fetch and run the GPT-2 BPE
    encoder over the corpus before failing one layer down — so the poisoned
    call fires and the test goes RED on the wrong exception.
    """
    import entropic_substrate.field_pretrain as field_pretrain

    def poisoned(*_args: object, **_kwargs: object) -> None:
        msg = "build_corpus ran for a tokenizer this lane cannot train"
        raise RuntimeError(msg)

    monkeypatch.setattr(field_pretrain, "build_corpus", poisoned)
    cfg = _pretrain_config(tmp_path, tokenizer="gpt2")
    with pytest.raises(ValueError, match="tokenizer"):
        run_field_pretrain(
            cfg,
            out_path=tmp_path / "field-pphi.pt",
            results_path=tmp_path / "field-pretrain.jsonl",
            device="cpu",
        )


def test_refuses_frozen_field_mode(tmp_path: Path) -> None:
    """Pretraining an already-frozen field is a contradiction, not a no-op."""
    cfg = _pretrain_config(tmp_path, field_mode="frozen", field_ckpt_path="/nonexistent.pt")
    with pytest.raises(ValueError, match="field_mode='frozen'"):
        run_field_pretrain(
            cfg,
            out_path=tmp_path / "field-pphi.pt",
            results_path=tmp_path / "field-pretrain.jsonl",
            device="cpu",
        )


# -------------------------------------------------------------------- cli
def _parser() -> argparse.ArgumentParser:
    """A throwaway root parser carrying only this lane's subcommand.

    ``cli.py`` is owned elsewhere this sprint and wires the same two
    functions with three lines; registering them on a local parser pins the
    seam without depending on that wiring having landed.
    """
    parser = argparse.ArgumentParser(prog="test")
    sub = parser.add_subparsers(dest="command", required=True)
    add_field_pretrain_parser(sub)
    return parser


def test_cli_subcommand_runs_a_pretrain_end_to_end(tmp_path: Path) -> None:
    cfg = _pretrain_config(tmp_path)
    config_path = tmp_path / "field-pretrain.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    out = tmp_path / "cli-pphi.pt"
    results = tmp_path / "cli.jsonl"

    args = _parser().parse_args(
        [
            "field-pretrain",
            "--config",
            str(config_path),
            "--steps",
            "2",
            "--seed",
            "7",
            "--device",
            "cpu",
            "--out",
            str(out),
            "--results",
            str(results),
        ]
    )
    assert cmd_field_pretrain(args) == 0

    assert [r.step for r in read_field_pretrain_series(results)] == [2]
    start = json.loads(results.read_text(encoding="utf-8").splitlines()[0])
    assert start["seed"] == 7, "--seed must reach the run as a config override"
    assert start["max_steps"] == 2
    assert torch.load(out, map_location="cpu", weights_only=True)["step"] == 2


def test_cli_requires_an_experiment_or_config(tmp_path: Path) -> None:
    args = _parser().parse_args(
        [
            "field-pretrain",
            "--out",
            str(tmp_path / "o.pt"),
            "--results",
            str(tmp_path / "r.jsonl"),
        ]
    )
    assert cmd_field_pretrain(args) == 2


def test_cli_requires_out_and_results(tmp_path: Path) -> None:
    """A pretrain whose checkpoint destination is implicit would be a run
    whose only product lands wherever the cwd happens to be."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["field-pretrain", "--exp", "e1-5c-field"])


# ---------------------------------------------------------------- reader
def _write_series(path: Path, *lines: str) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def test_reader_raises_on_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FieldPretrainGap, match="does not exist"):
        read_field_pretrain_series(tmp_path / "absent.jsonl")


def test_reader_raises_on_a_series_with_no_eval_records(tmp_path: Path) -> None:
    """No fail-open empty return: a run that never evaluated has no readout."""
    path = _write_series(tmp_path / "s.jsonl", '{"type": "field_pretrain_run_start", "seed": 42}')
    with pytest.raises(FieldPretrainGap, match="no field_pretrain_eval"):
        read_field_pretrain_series(path)


def test_reader_raises_on_a_missing_field(tmp_path: Path) -> None:
    path = _write_series(tmp_path / "s.jsonl", '{"type": "field_pretrain_eval", "step": 1}')
    with pytest.raises(FieldPretrainGap, match="field_ce"):
        read_field_pretrain_series(path)


def test_reader_raises_on_a_nonfinite_field(tmp_path: Path) -> None:
    """``json`` parses bare ``NaN`` back without complaint — a numerically
    failed run must not read as a valid, merely-unusual CE."""
    path = _write_series(
        tmp_path / "s.jsonl", '{"type": "field_pretrain_eval", "step": 1, "field_ce": NaN}'
    )
    with pytest.raises(FieldPretrainGap, match="non-finite"):
        read_field_pretrain_series(path)


def test_reader_raises_on_a_negative_field_ce(tmp_path: Path) -> None:
    """Cross-entropy is non-negative; a finite-but-impossible reading is
    corruption the same way NaN is (telemetry.py's documented-domain rule)."""
    path = _write_series(
        tmp_path / "s.jsonl", '{"type": "field_pretrain_eval", "step": 1, "field_ce": -0.5}'
    )
    with pytest.raises(FieldPretrainGap, match="minimum"):
        read_field_pretrain_series(path)


def test_reader_raises_on_a_non_integer_step(tmp_path: Path) -> None:
    """``bool`` is an ``int`` subclass, so a JSON ``true`` would otherwise
    read as step 1 (telemetry.py's own rationale for the same check)."""
    path = _write_series(
        tmp_path / "s.jsonl", '{"type": "field_pretrain_eval", "step": true, "field_ce": 5.5}'
    )
    with pytest.raises(FieldPretrainGap, match="step"):
        read_field_pretrain_series(path)
