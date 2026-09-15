"""D1 convenience probes retain the existing manifest and trainer boundary."""

import pytest

from entropic_substrate.cli import main
from entropic_substrate.d1_cli import D1CliExitCode

from .test_d1_trainer_cli_services import _manifest

pytestmark = pytest.mark.unit


def test_d1_probe_dispatches_the_real_existing_trainer(tmp_path, monkeypatch, capsys) -> None:
    manifest, request = _manifest(tmp_path)
    monkeypatch.setattr(
        "entropic_substrate.d1_probe_cli.load_d1_profile_catalog",
        lambda **_kwargs: (request.plan.profile,),
    )
    assert (
        main(
            [
                "d-probe",
                "--exp",
                "d1-both",
                "--manifest",
                str(manifest),
                "--out-dir",
                str(tmp_path),
            ]
        )
        == D1CliExitCode.SUCCESS
    )
    assert (tmp_path / "run/checkpoint.pt").is_file()
    assert "fresh_completion" in capsys.readouterr().out


def test_d1_probe_does_not_invent_runtime_provenance(tmp_path) -> None:
    with pytest.raises(ValueError, match="manifest"):
        main(["d-probe", "--exp", "d1-both", "--out-dir", str(tmp_path)])
