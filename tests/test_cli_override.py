"""``train --override SECTION.KEY=VALUE``: the generic sweep surface.

The E1.5 registrations sweep single config knobs that have no dedicated CLI
flag — E1.5b's gram_lambda grid ("sweeps via CLI overrides", the config's
own comment), E1.5c's λ_field pilot, and the LR-sensitivity diagnostic
(docs/e1-5c-preregistration.md §Pilots). The plumbing already existed
(``resolve_experiment_config(overrides=...)``); this flag is its missing
CLI surface. Values parse via ``yaml.safe_load`` — the SAME typing rules
the config files themselves use, so ``0.3`` is a float, ``frozen`` a
string, ``null`` is None.

Precedence (declared in the flag's help): the dedicated flags
(--steps/--seed/--device/--resume) WIN over --override — they are the more
specific, more visible spelling of the same intent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import entropic_substrate.config as config_mod
from entropic_substrate.cli import UNGATED_ENV, main

pytestmark = pytest.mark.unit


def _probe_yaml(tmp_path: Path) -> Path:
    doc = {
        "name": "override-probe",
        "model": {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "block_size": 32,
            "tokenizer": "byte",
        },
        "data": {
            "source": "synthetic",
            "synthetic_bytes": 40_000,
            "seq_len": 32,
            "batch_size": 2,
            "num_eval_batches": 1,
        },
        "train": {
            "max_steps": 1,
            "eval_interval": 1,
            "warmup_steps": 1,
            "device": "cpu",
            "save_interval": 0,
            "results_dir": str(tmp_path / "results"),
            "models_dir": str(tmp_path / "models"),
        },
    }
    path = tmp_path / "override-probe.yaml"
    path.write_text(json.dumps(doc))
    return path


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> list[dict[str, object]]:
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "override-probe.jsonl"
    code = main(
        [
            "train",
            "--config",
            str(_probe_yaml(tmp_path)),
            "--device",
            "cpu",
            "--results",
            str(results),
            "--quiet",
            *extra,
        ]
    )
    assert code == 0
    return [json.loads(line) for line in results.read_text().splitlines() if line]


def _start_config(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    start = next(r for r in records if r["event"] == "run_start")
    config = start["config"]
    assert isinstance(config, dict)
    return config


def test_override_reaches_the_resolved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _run(tmp_path, monkeypatch, "--override", "model.dropout=0.05")
    assert _start_config(records)["model"]["dropout"] == 0.05


def test_override_values_are_yaml_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """3e-4 must arrive as a float, not the string "3e-4" — a string lr
    would explode inside the optimizer far from the flag that caused it."""
    records = _run(
        tmp_path,
        monkeypatch,
        "--override",
        "train.lr=3e-4",
        "--override",
        "data.num_eval_batches=2",
    )
    config = _start_config(records)
    assert config["train"]["lr"] == pytest.approx(3e-4)
    assert config["data"]["num_eval_batches"] == 2


def test_dedicated_flags_win_over_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = _run(tmp_path, monkeypatch, "--steps", "1", "--override", "train.max_steps=99")
    assert _start_config(records)["train"]["max_steps"] == 1


def test_malformed_override_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No '='; no dotted section — both refused by argparse itself (exit 2)
    before any config resolution."""
    for bad in ("model.gram_lambda", "gram_lambda=0.5"):
        with pytest.raises(SystemExit) as excinfo:
            main(["train", "--config", "irrelevant.yaml", "--override", bad])
        assert excinfo.value.code == 2


def test_unknown_key_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo'd knob must refuse (config_from_dict's unknown-key check),
    never silently train the default it failed to override."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unknown keys"):
        main(
            [
                "train",
                "--config",
                str(_probe_yaml(tmp_path)),
                "--override",
                "model.bogus_knob=1",
            ]
        )


def test_override_cannot_sneak_past_the_prereg_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The E15 gate reads the RESOLVED config, so an axis activated via
    --override is gated exactly like one activated in a yaml."""
    monkeypatch.setitem(config_mod.E15_PREREGISTRATIONS, "gram_lambda", None)
    monkeypatch.delenv(UNGATED_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "override-probe.jsonl"
    code = main(
        [
            "train",
            "--config",
            str(_probe_yaml(tmp_path)),
            "--device",
            "cpu",
            "--results",
            str(results),
            "--quiet",
            "--override",
            "model.gram_lambda=0.1",
        ]
    )
    assert code == 2
    assert not results.exists()
