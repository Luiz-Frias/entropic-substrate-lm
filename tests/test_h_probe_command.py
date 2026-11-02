"""Host command composition is distinct from paid or scientific execution proof."""

import io
import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest
from tests.test_h_probe_launch import PILOT_CEILING, launch, pilot_launch, pilot_launch_v4

from entropic_substrate.ladder.run.h_probe_launch import H0ProbeLaunch
from entropic_substrate.ladder.run.h_probe_process import WatchdogProcessConfig
from entropic_substrate.ladder.run.worker import h_probe_command
from entropic_substrate.ladder.run.worker.h_probe_command import run_probe_command
from entropic_substrate.ladder.run.worker.h_probe_controller import ProbeControllerFailure

pytestmark = pytest.mark.unit


def arguments(tmp_path: Path) -> list[str]:
    (tmp_path / "launch.json").write_bytes(launch().payload)
    (tmp_path / "workspace").mkdir(mode=0o700)
    return [
        "--launch-root",
        str(tmp_path),
        "--launch-file",
        "launch.json",
        "--workspace",
        str(tmp_path / "workspace"),
        "--repo-root",
        str(tmp_path),
        "--expected-account-id",
        "fixture-account",
        "--runpodctl",
        "/fixture/runpodctl",
        "--provider-timeout",
        "30",
        "--max-provider-bytes",
        "1000000",
        "--max-launch-bytes",
        "1000000",
        "--max-control-bytes",
        "1000000",
        "--poll-seconds",
        "5",
        "--startup-timeout",
        "120",
        "--shutdown-timeout",
        "30",
        "--max-hourly-rate",
        "0.6",
    ]


@pytest.mark.parametrize(
    "failure,release_failure", [(False, False), (True, False), (False, True), (True, True)]
)
def test_command_connects_one_manifest_to_guard_and_controller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: bool,
    release_failure: bool,
) -> None:
    argv = arguments(tmp_path)
    calls: list[str] = []
    provider, objects, guard = object(), object(), object()
    report = {
        "status": "controller_failed" if failure else "terminal_collected",
        "workload_verified": not failure,
        "termination_verified": True,
        "scientific_authority": "none",
    }

    def object_factory(repo: Path, environ: Mapping[str, str], **limits: int) -> object:
        assert repo == tmp_path and environ == {"FIXTURE": "not-credentials"}
        assert limits["max_object_bytes"] == launch().request.limits.max_object_bytes
        calls.append("objects")
        return objects

    @contextmanager
    def supervise(
        root: Path, spec: H0ProbeLaunch, config: WatchdogProcessConfig, *, expected_account_id: str
    ) -> Generator[object, None, None]:
        assert spec == launch() and expected_account_id == "fixture-account"
        assert root == tmp_path / "workspace/control" and root.is_dir()
        assert config.executable == Path("/fixture/runpodctl")
        calls.append("guard_enter")
        try:
            yield guard
        finally:
            calls.append("guard_exit")
            if release_failure:
                raise RuntimeError("private release detail")

    def controller(
        actual_guard: object,
        actual_provider: object,
        actual_objects: object,
        destination: Path,
        **limits: float,
    ) -> dict[str, object]:
        assert (actual_guard, actual_provider, actual_objects) == (guard, provider, objects)
        assert destination == tmp_path / "workspace/capture" and destination.is_dir()
        assert limits == {"poll_seconds": 5.0, "max_hourly_rate": 0.6}
        calls.append("controller")
        if failure:
            raise ProbeControllerFailure(report, (ValueError("private detail"),))
        return report

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.HRunpodTransport",
        lambda **_: provider,
    )
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.build_worker_objects", object_factory
    )
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.supervise_probe_watchdog", supervise
    )
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.run_probe_controller", controller
    )
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_probe_command(
        argv, environ={"FIXTURE": "not-credentials"}, stdout=stdout, stderr=stderr
    )
    failed = failure or release_failure
    assert code == (1 if failed else 0)
    result = json.loads(stderr.getvalue() if failed else stdout.getvalue())
    assert (result["evidence"] if failed else result) == report
    assert "private detail" not in stderr.getvalue()
    assert "private release detail" not in stderr.getvalue()
    assert calls == ["objects", "guard_enter", "controller", "guard_exit"]


@pytest.mark.parametrize(
    "invalid", ["occupied", "public", "bad_manifest", "missing_flag", "pilot_over_cap"]
)
def test_invalid_launch_never_reaches_provider_or_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    argv = arguments(tmp_path)
    if invalid == "occupied":
        (tmp_path / "workspace/existing").write_text("keep")
    elif invalid == "public":
        (tmp_path / "workspace").chmod(0o755)
    elif invalid == "bad_manifest":
        (tmp_path / "launch.json").write_bytes(b"{}")
    elif invalid == "pilot_over_cap":
        # 37 USD/h over the 3900 s watchdog is 40.08 USD: past the authority's 40 USD cap
        (tmp_path / "launch.json").write_bytes(pilot_launch_v4().payload)
        argv[-1] = "37"
    else:
        argv = argv[:-2]

    reached: list[str] = []

    def forbidden(**_kwargs: object) -> None:
        # Recorded, not pytest.fail: the command's own ``except BaseException`` would
        # swallow a test failure raised here and still exit nonzero.
        reached.append("provider")
        raise RuntimeError("invalid input reached provider construction")

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.HRunpodTransport", forbidden
    )
    stdout, stderr = io.StringIO(), io.StringIO()
    code = run_probe_command(argv, environ={}, stdout=stdout, stderr=stderr)
    assert code != 0
    if invalid == "missing_flag":
        assert code == 2  # the parent tip's argparse refusal (exit 2), pinned byte-for-byte below
    assert stdout.getvalue() == ""
    assert not (tmp_path / "workspace/control").exists()
    assert reached == []


def test_a_pilot_launch_within_its_cap_reaches_the_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = arguments(tmp_path)
    (tmp_path / "launch.json").write_bytes(pilot_launch_v4().payload)
    argv[-1] = "36.9"
    reached: list[float] = []

    @contextmanager
    def supervise(*_: object, **__: object) -> Generator[object, None, None]:
        yield object()

    def controller(*_: object, **limits: float) -> dict[str, object]:
        reached.append(limits["max_hourly_rate"])
        return {"status": "terminal_collected"}

    for name, value in (
        ("HRunpodTransport", lambda **_: object()),
        ("build_worker_objects", lambda *_, **__: object()),
        ("supervise_probe_watchdog", supervise),
        ("run_probe_controller", controller),
    ):
        monkeypatch.setattr(f"entropic_substrate.ladder.run.worker.h_probe_command.{name}", value)
    stdout, stderr = io.StringIO(), io.StringIO()
    assert run_probe_command(argv, environ={}, stdout=stdout, stderr=stderr) == 0
    assert reached == [36.9]


def test_main_configures_the_substrate_before_it_runs_the_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C10: ``main`` is the entry point; ``configure_logging()`` strictly precedes the run and
    the run receives the argv it was given, so the command's own contract is unchanged."""
    order: list[object] = []
    monkeypatch.setattr(h_probe_command, "configure_logging", lambda: order.append("logging"))

    def run(argv: object, *, environ: object) -> int:
        order.append(("run", list(argv)))  # type: ignore[call-overload]
        return 7

    monkeypatch.setattr(h_probe_command, "run_probe_command", run)
    assert h_probe_command.main(["--flag", "x"]) == 7
    assert order == ["logging", ("run", ["--flag", "x"])]


# ---------------------------------------------------------------------------
# FOR-1694 (launch v5): the command DERIVES the ceiling from a sealed launch.
# ---------------------------------------------------------------------------
def run_sealed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str | None
) -> tuple[int, list[float | None], list[str]]:
    argv = arguments(tmp_path)
    (tmp_path / "launch.json").write_bytes(pilot_launch().payload)
    argv = argv[:-2] if flag is None else [*argv[:-1], flag]
    reached: list[float | None] = []
    provider: list[str] = []

    @contextmanager
    def supervise(*_: object, **__: object) -> Generator[object, None, None]:
        yield object()

    def controller(*_: object, **limits: float) -> dict[str, object]:
        reached.append(limits["max_hourly_rate"])
        return {"status": "terminal_collected"}

    def transport(**_: object) -> object:
        provider.append("provider")
        return object()

    for name, value in (
        ("HRunpodTransport", transport),
        ("build_worker_objects", lambda *_, **__: object()),
        ("supervise_probe_watchdog", supervise),
        ("run_probe_controller", controller),
    ):
        monkeypatch.setattr(f"entropic_substrate.ladder.run.worker.h_probe_command.{name}", value)
    code = run_probe_command(argv, environ={}, stdout=io.StringIO(), stderr=io.StringIO())
    return code, reached, provider


@pytest.mark.parametrize("flag", [None, str(PILOT_CEILING)])
def test_a_sealed_launch_supplies_the_ceiling_with_or_without_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str | None
) -> None:
    code, reached, _ = run_sealed(tmp_path, monkeypatch, flag)
    assert code == 0 and reached == [PILOT_CEILING]


@pytest.mark.parametrize("flag", ["37", "36.8", "0.6", "36.900000001"])
def test_a_flag_that_disagrees_with_the_sealed_ceiling_is_refused_before_any_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    argv = arguments(tmp_path)
    (tmp_path / "launch.json").write_bytes(pilot_launch().payload)
    stderr = io.StringIO()
    reached: list[str] = []
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.HRunpodTransport",
        lambda **_: reached.append("provider"),
    )
    code = run_probe_command([*argv[:-1], flag], environ={}, stdout=io.StringIO(), stderr=stderr)
    assert code == 1 and reached == []
    assert not (tmp_path / "workspace/control").exists()  # no workspace was touched
    assert json.loads(stderr.getvalue())["error_type"] == "ValueError"


def test_an_unsealed_launch_without_the_flag_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Smoke-class refusal, byte-identical to the parent tip (b429635e): argparse usage error,
    exit 2, on the process stderr, before any workspace or provider call."""
    argv = arguments(tmp_path)[:-2]  # the smoke launch seals nothing
    reached: list[str] = []
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.HRunpodTransport",
        lambda **_: reached.append("provider"),
    )
    code = run_probe_command(argv, environ={}, stdout=io.StringIO(), stderr=io.StringIO())
    err = capsys.readouterr().err
    assert code == 2 and reached == [] and not (tmp_path / "workspace/control").exists()
    assert err.startswith("usage: ")
    assert err.endswith("error: the following arguments are required: --max-hourly-rate\n")
