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
        assert code == 2  # the argparse refusal (exit 2), pinned by its usage and error lines below
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
# C-L4c-4: the flag is REQUIRED for every launch class. A sealed launch's flag must equal the
# sealed ceiling exactly; an absent flag is refused by the argparse text before any read.
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


def test_a_sealed_launch_with_the_matching_flag_reaches_the_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, reached, _ = run_sealed(tmp_path, monkeypatch, str(PILOT_CEILING))
    assert code == 0 and reached == [PILOT_CEILING]


def normalized(text: str) -> str:
    """Argparse wraps usage across lines; compare the words, not the line breaks."""
    return " ".join(text.split())


def test_a_sealed_launch_without_the_flag_is_refused_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sealed launch still requires the flag: exit 2 with the argparse text, and no launch
    read, workspace or provider call happens before that refusal."""
    argv = arguments(tmp_path)[:-2]
    (tmp_path / "launch.json").write_bytes(pilot_launch().payload)
    reads: list[str] = []

    def read(*_: object, **__: object) -> None:
        reads.append("launch read")
        raise RuntimeError("the launch was read before the flag was parsed")

    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.read_d1_artifact_bytes", read
    )
    reached: list[str] = []
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.HRunpodTransport",
        lambda **_: reached.append("provider"),
    )
    stderr = io.StringIO()
    code = run_probe_command(argv, environ={}, stdout=io.StringIO(), stderr=stderr)
    err = normalized(capsys.readouterr().err)
    assert code == 2 and reads == [] and reached == []
    assert stderr.getvalue() == "" and not (tmp_path / "workspace/control").exists()
    assert err.endswith("error: the following arguments are required: --max-hourly-rate")
    assert "--max-hourly-rate MAX_HOURLY_RATE" in err
    assert "[--max-hourly-rate" not in err


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


def test_an_unsealed_launch_without_the_flag_is_refused_by_the_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Smoke-class refusal through the parser: exit 2, argparse usage on the process stderr with
    the flag REQUIRED (no brackets), before any launch read, workspace or provider call.

    RETRACTED: the earlier claim that this text is byte-identical to the parent tip. That was
    measured false at 1697db41 and is not pinned here; the usage and error text are pinned.
    """
    argv = arguments(tmp_path)[:-2]  # the smoke launch seals nothing
    reached: list[str] = []
    monkeypatch.setattr(
        "entropic_substrate.ladder.run.worker.h_probe_command.HRunpodTransport",
        lambda **_: reached.append("provider"),
    )
    stderr = io.StringIO()
    code = run_probe_command(argv, environ={}, stdout=io.StringIO(), stderr=stderr)
    err = capsys.readouterr().err
    assert code == 2 and reached == [] and not (tmp_path / "workspace/control").exists()
    assert stderr.getvalue() == ""
    assert err.startswith("usage: ")
    assert err.endswith("error: the following arguments are required: --max-hourly-rate\n")
    assert "--max-hourly-rate MAX_HOURLY_RATE" in normalized(err)
    assert "[--max-hourly-rate" not in normalized(err)


def test_the_flag_refusal_precedes_a_missing_launch_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ordering pin: parse before read. A missing launch root with the flag absent must be the
    argparse refusal (exit 2, no JSON on the command's stderr), not a launch-read failure."""
    argv = arguments(tmp_path)[:-2]
    argv[argv.index("--launch-root") + 1] = str(tmp_path / "absent")
    stderr = io.StringIO()
    code = run_probe_command(argv, environ={}, stdout=io.StringIO(), stderr=stderr)
    assert code == 2 and stderr.getvalue() == ""
    assert (
        "error: the following arguments are required: --max-hourly-rate" in capsys.readouterr().err
    )
