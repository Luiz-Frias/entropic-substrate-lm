"""Documented key=value probes reach the same validated tiny construction."""

import pytest

import entropic_substrate.historical.d_ladder.d_ladder_execution as execution
from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.ladder.register.ladder import Treatment

pytestmark = pytest.mark.unit


def test_named_convenience_matches_flag_request(tmp_path, monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(
        execution,
        "run_local_smoke",
        lambda request, root, **kwargs: observed.append((request, root)) or {},
    )
    assert main(["d-probe", "exp=d2-g", "tiny=true", "steps=1", f"out={tmp_path}"]) == 0
    assert (
        main(
            [
                "d-probe",
                "--exp",
                "d2-g",
                "--tiny",
                "true",
                "--steps",
                "1",
                "--out-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert observed[0] == observed[1]
    assert observed[0][0].spec.treatment is Treatment.G
    capsys.readouterr()


@pytest.mark.parametrize("extra", ["exp=d2-s", "--exp=d2-s", "unknown=value", "steps=oops"])
def test_convenience_rejects_conflicting_or_invalid_fields(tmp_path, extra):
    with pytest.raises((ValueError, SystemExit)):
        main(["d-probe", "exp=d2-g", f"out={tmp_path / 'absent'}", extra])
    assert not (tmp_path / "absent").exists()
