"""FOR-914: CLI execution gate for unregistered E1.5 arms.

E1.5 arms live in ``EXPERIMENTS`` (P0, PR #66) but the rung-lifecycle
flywheel (docs/rung-lifecycle-protocol.md) holds each arm's CONFIRMATORY
execution behind its committed preregistration. The gate keys off the
RESOLVED config's axes — never ``--exp`` — for the same reason
``is_execution_gated`` does: ``--config`` outranks ``--exp`` and a
hand-rolled config never names an experiment.

Two enforcement sites, split by failure domain (the Containerfile copies
``src/ configs/ scripts/`` but NOT ``docs/``, so a pod cannot stat the
preregistration file):

1. RUNTIME (works in the docs-less image): ``config.E15_PREREGISTRATIONS``
   maps each E1.5 axis to its committed preregistration path, or ``None``
   while unregistered; ``cli._e15_prereg_preflight`` refuses an active axis
   mapped to ``None``.
2. CI (full repo checkout): the tests here pin every non-``None`` map entry
   to a file that actually exists, so the in-code assertion and the
   artifact cannot drift apart.

``--tiny`` runs are exempt BY DESIGN: the capture lane excludes ``-tiny``
runs by construction (capture_results), so a tiny smoke cannot produce a
confirmatory-looking artifact — and the CPU smoke tests (test_train_smoke)
must stay hermetic while later stack branches register their axes one by
one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import entropic_substrate.config as config_mod
from entropic_substrate.cli import UNGATED_ENV, main
from entropic_substrate.config import (
    E15_PREREGISTRATIONS,
    Config,
    required_preregistrations,
    resolve_experiment_config,
)
from helpers import tiny_model_config

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------- axis detection
@pytest.mark.parametrize(
    ("overrides", "axis"),
    [
        ({"embedding_freeze": "both"}, "embedding_freeze"),
        ({"embedding_freeze": "input", "tie_embeddings": False}, "embedding_freeze"),
        ({"gram_lambda": 0.1}, "gram_lambda"),
        (
            {
                "state_source": "field",
                "field_d_model": 32,
                "field_n_layers": 1,
                "field_n_heads": 2,
            },
            "state_source",
        ),
    ],
)
def test_required_preregistrations_fires_per_axis(overrides: dict[str, object], axis: str) -> None:
    cfg = Config(name="gate-probe", model=tiny_model_config(**overrides))
    assert [a for a, _ in required_preregistrations(cfg)] == [axis]


def test_required_preregistrations_empty_off_axis() -> None:
    """A table-arm config (no E1.5 axis active) requires no registration."""
    assert required_preregistrations(resolve_experiment_config("e1-fisher")) == ()
    assert required_preregistrations(resolve_experiment_config("e3")) == ()


def test_required_preregistrations_collects_multiple_axes() -> None:
    """Freeze + Gram together (not refused at config parse — the cross-axis
    refusals live at model build) must list BOTH axes, not stop at the
    first: a refusal message naming one missing registration when two are
    missing sends the operator around the loop twice."""
    cfg = Config(
        name="gate-probe",
        model=tiny_model_config(embedding_freeze="both", gram_lambda=0.1),
    )
    assert [a for a, _ in required_preregistrations(cfg)] == [
        "embedding_freeze",
        "gram_lambda",
    ]


# ------------------------------------------------- map <-> artifact pins
def test_e1_5a_axis_registered_to_this_stacks_prereg() -> None:
    assert E15_PREREGISTRATIONS["embedding_freeze"] == "docs/e1-5a-preregistration.md"


def test_registered_prereg_paths_exist_in_repo() -> None:
    """CI half of the gate: an axis may only claim a registration that is
    actually committed. (The runtime half cannot stat these paths — the
    image carries no docs/ — which is exactly why this test exists.)"""
    for axis, path in E15_PREREGISTRATIONS.items():
        if path is None:
            continue
        assert (_REPO_ROOT / path).exists(), f"{axis} claims missing registration {path}"


def test_every_e1_5_axis_is_in_the_map() -> None:
    """The map must carry an explicit entry (path or None) for every E1.5
    axis the detector knows — a silently absent key would read as
    'unregistered' at runtime, which is correct-but-illegible; the map is
    the single place a reader checks registration state."""
    assert set(E15_PREREGISTRATIONS) == {"embedding_freeze", "gram_lambda", "state_source"}


# ------------------------------------------------------------ CLI gate
def _gate_probe_yaml(tmp_path: Path, model_overrides: dict[str, object]) -> Path:
    """A hand-rolled --config probe (json is valid yaml): tiny dims so the
    RED state of these tests (gate absent => the run actually trains) costs
    one CPU step, not an 85M-param build."""
    doc = {
        "name": "gate-probe",
        "model": {
            "d_model": 32,
            "n_layers": 1,
            "n_heads": 2,
            "block_size": 32,
            "tokenizer": "byte",
            **model_overrides,
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
    path = tmp_path / "gate-probe.yaml"
    path.write_text(json.dumps(doc))
    return path


def _run_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **model_overrides: object) -> int:
    monkeypatch.chdir(tmp_path)
    cfg_path = _gate_probe_yaml(tmp_path, dict(model_overrides))
    results = tmp_path / "gate-probe.jsonl"
    return main(
        [
            "train",
            "--config",
            str(cfg_path),
            "--device",
            "cpu",
            "--results",
            str(results),
            "--quiet",
        ]
    )


def test_cli_refuses_unregistered_axis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate's teeth: an active E1.5 axis whose map entry is None must
    refuse BEFORE any training side effect (exit 2, no results file)."""
    monkeypatch.setitem(config_mod.E15_PREREGISTRATIONS, "embedding_freeze", None)
    monkeypatch.delenv(UNGATED_ENV, raising=False)
    code = _run_probe(tmp_path, monkeypatch, embedding_freeze="both")
    assert code == 2
    assert not (tmp_path / "gate-probe.jsonl").exists(), "gate fired after training started"


def test_cli_override_env_allows_deliberate_ungated_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(config_mod.E15_PREREGISTRATIONS, "embedding_freeze", None)
    monkeypatch.setenv(UNGATED_ENV, "1")
    code = _run_probe(tmp_path, monkeypatch, embedding_freeze="both")
    assert code == 0
    assert (tmp_path / "gate-probe.jsonl").exists()


def test_cli_registered_axis_proceeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the axis registered (the real map, untouched), the same probe
    trains — the gate holds unregistered arms, it does not tax registered
    ones."""
    monkeypatch.delenv(UNGATED_ENV, raising=False)
    code = _run_probe(tmp_path, monkeypatch, embedding_freeze="both")
    assert code == 0


def test_tiny_runs_are_exempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--tiny is the smoke lane: capture excludes -tiny runs by
    construction, so an unregistered tiny run cannot contaminate a
    confirmatory set — and the CPU smoke suite must keep passing while
    stack branches register axes one at a time."""
    monkeypatch.setitem(config_mod.E15_PREREGISTRATIONS, "state_source", None)
    monkeypatch.delenv(UNGATED_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    results = tmp_path / "e1-5c-field.jsonl"
    code = main(
        [
            "train",
            "--exp",
            "e1-5c-field",
            "--tiny",
            "--device",
            "cpu",
            "--steps",
            "2",
            "--results",
            str(results),
            "--quiet",
        ]
    )
    assert code == 0
