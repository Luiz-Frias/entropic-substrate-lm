"""The public tiny command exercises real mechanisms without scientific authority."""

import json
from pathlib import Path

import pytest

from entropic_substrate.cli import main

pytestmark = pytest.mark.unit


def test_d_probe_runs_real_tiny_and_emits_smoke_receipt(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "d-probe",
                "--exp",
                "d2-l",
                "--tiny",
                "true",
                "--steps",
                "1",
                "--out-dir",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["run_class"] == "smoke"
    assert receipt["step"] == 1
    assert receipt["scientific_authority"] == "none"
    assert Path(receipt["checkpoint"]).is_file()


def test_d_probe_refuses_non_tiny_without_creating_artifacts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local tiny"):
        main(["d-probe", "--exp", "d2-l", "--tiny", "false", "--out-dir", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()
