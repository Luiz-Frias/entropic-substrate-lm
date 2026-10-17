"""The public command executes and inspects the same file-bound training path."""

import json
from pathlib import Path

import pytest
from tests.test_sentence_execution import engine

from entropic_substrate.composition.entrypoints.cli import main
from entropic_substrate.ladder.run.journal import artifact_reference, write_json

pytestmark = pytest.mark.unit


def test_public_training_and_inspection_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = engine(tmp_path)
    (tmp_path / "run").mkdir()
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    command = {
        "schema": "h-probe-train-command-v1",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "max_source_bytes": 1_000_000,
        "max_artifact_bytes": 10_000_000,
        "resume": None,
        "pause_after": None,
    }
    write_json(tmp_path, "train.json", command, max_bytes=1_000_000)
    assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", "train.json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "training_complete" and result["scientific_authority"] == "none"
    inspect = {key: value for key, value in command.items() if key not in ("resume", "pause_after")}
    inspect.update(schema="h-probe-inspect-command-v1", receipt=result["receipt"])
    write_json(tmp_path, "inspect.json", inspect, max_bytes=1_000_000)
    assert main(["h-inspect-training", "--root", str(tmp_path), "--manifest", "inspect.json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["successful_optimizer_steps"] == 6


def test_command_rejects_unknown_fields_and_source_escape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = engine(tmp_path)
    (tmp_path / "run").mkdir()
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    for index, source in enumerate(("../outside", "source")):
        command = {
            "schema": "h-probe-train-command-v1",
            "profile": artifact_reference(tmp_path, profile),
            "source_root": source,
            "output_root": "run",
            "max_source_bytes": 1_000_000,
            "max_artifact_bytes": 10_000_000,
            "resume": None,
            "pause_after": None,
        }
        if index:
            command["steps_override"] = 1
        name = f"bad-{index}.json"
        write_json(tmp_path, name, command, max_bytes=1_000_000)
        assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", name]) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
    assert not list((tmp_path / "run").iterdir())


def test_public_pause_resume_preserves_the_training_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = engine(tmp_path)
    (tmp_path / "run").mkdir()
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    command = {
        "schema": "h-probe-train-command-v1",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "max_source_bytes": 1_000_000,
        "max_artifact_bytes": 10_000_000,
        "resume": None,
        "pause_after": {"phase": "prediction", "step": 2},
    }
    write_json(tmp_path, "pause.json", command, max_bytes=1_000_000)
    assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", "pause.json"]) == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["status"] == "paused"
    command.update(resume=paused["receipt"], pause_after=None)
    write_json(tmp_path, "resume.json", command, max_bytes=1_000_000)
    assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", "resume.json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "training_complete"


def test_pause_at_an_unsealed_step_is_not_silently_ignored(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    template = engine(tmp_path)
    (tmp_path / "run").mkdir()
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    command = {
        "schema": "h-probe-train-command-v1",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "max_source_bytes": 1_000_000,
        "max_artifact_bytes": 10_000_000,
        "resume": None,
        "pause_after": {"phase": "prediction", "step": 1},
    }
    write_json(tmp_path, "bad-pause.json", command, max_bytes=1_000_000)
    assert main(["h-train-probe", "--root", str(tmp_path), "--manifest", "bad-pause.json"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
    assert not list((tmp_path / "run").iterdir())


def test_public_generation_pause_resume_and_inspection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_sentence_generation import trained
    from tests.test_sentence_rollout import scripted_tokens

    template, output, training = trained(tmp_path)
    profile = write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000)
    command = {
        "schema": "h-probe-generate-command-v1",
        "profile": artifact_reference(tmp_path, profile),
        "source_root": "source",
        "output_root": "run",
        "max_source_bytes": 1_000_000,
        "max_artifact_bytes": 10_000_000,
        "training": artifact_reference(output, training),
        "resume": None,
        "pause_after": 1,
    }
    scripted_tokens(monkeypatch, [65, 46, 257])
    write_json(tmp_path, "generate.json", command, max_bytes=1_000_000)
    assert main(["h-generate-probe", "--root", str(tmp_path), "--manifest", "generate.json"]) == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["status"] == "paused"
    command.update(resume=paused["receipt"], pause_after=None)
    write_json(tmp_path, "resume-generation.json", command, max_bytes=1_000_000)
    assert (
        main(["h-generate-probe", "--root", str(tmp_path), "--manifest", "resume-generation.json"])
        == 0
    )
    complete = json.loads(capsys.readouterr().out)
    assert complete["status"] == "generation_complete"
    inspect = {key: value for key, value in command.items() if key not in ("resume", "pause_after")}
    inspect.update(schema="h-probe-inspect-generation-command-v1", receipt=complete["receipt"])
    write_json(tmp_path, "inspect-generation.json", inspect, max_bytes=1_000_000)
    assert (
        main(
            [
                "h-inspect-generation",
                "--root",
                str(tmp_path),
                "--manifest",
                "inspect-generation.json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["summary"]["sentences"] == ["A."]


# ------------------------------------------------------------------------------------------
# FOR-1644: the local probe is tracked. It builds its sink through build_sink_for_run (the
# one H construction seam), writes step rows and evaluation reports to a results file OUTSIDE
# the probe's output root, writes NO lifecycle records (the probe is not the worker), and
# finishes its sink exactly once: FINISHED on a return, FAILED on every raised exit.


class ProbeSetup:
    """One tmp_path prepared once: profile, source and an empty ``run`` output root."""

    def __init__(self, tmp_path: Path) -> None:
        template = engine(tmp_path)
        (tmp_path / "run").mkdir(exist_ok=True)
        self.root = tmp_path
        self.profile = template.profile
        self._profile_ref = artifact_reference(
            tmp_path,
            write_json(tmp_path, "profile.json", template.profile.manifest, max_bytes=1_000_000),
        )

    def train(self, name: str = "train.json", **overrides: object) -> list[str]:
        command: dict[str, object] = {
            "schema": "h-probe-train-command-v1",
            "profile": self._profile_ref,
            "source_root": "source",
            "output_root": "run",
            "max_source_bytes": 1_000_000,
            "max_artifact_bytes": 10_000_000,
            "resume": None,
            "pause_after": None,
        }
        command.update(overrides)
        write_json(self.root, name, command, max_bytes=1_000_000)
        return ["h-train-probe", "--root", str(self.root), "--manifest", name]


class _RecordingSink:
    """A MetricSink double that records every call, including the finish status."""

    def __init__(self) -> None:
        self.records: list[object] = []
        self.finishes: list[str] = []

    def log(self, record: object) -> None:
        self.records.append(record)

    def log_artifact(self, path: Path, *, artifact_path: str | None = None) -> None:
        raise AssertionError("the probe never uploads artifacts")

    def annotate(self, tags: object) -> None:
        raise AssertionError("the probe never annotates")

    def finish(self, *, status: str = "FINISHED") -> None:
        self.finishes.append(status)


@pytest.fixture
def mlflow_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """The mise-loaded ``.env`` may carry a tracking URI: the probe tests never rely on it."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("ENTROPIC_MLFLOW_TINY", raising=False)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_the_probe_writes_step_rows_to_a_results_file_beside_not_inside_its_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mlflow_off: None
) -> None:
    """K3 + M3: the REAL ``build_sink_for_run`` (MLflow off, so a bare JsonlSink) through a
    fresh run. K1: the results file is a sibling of the output root, because the runner
    refuses a fresh run whose root is non-empty. K2: no ``h-run-lifecycle-v1`` record."""
    assert main(ProbeSetup(tmp_path).train()) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "training_complete"
    metrics = tmp_path / "run.observability" / "metrics.jsonl"
    rows = _records(metrics)
    training = [r for r in rows if r.get("schema") == "h-training-step-v2"]
    assert [r["step"] for r in training] == [1, 2, 1, 2, 3, 4]
    assert not [r for r in rows if r.get("schema") == "h-run-lifecycle-v1"]
    assert all("schema" in r for r in rows)  # no annotate rows: the probe seals no R2 checkpoint
    assert not any(path.name == "metrics.jsonl" for path in (tmp_path / "run").rglob("*"))


def test_the_probe_finishes_its_sink_finished_once_on_a_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from entropic_substrate.composition import cli as probe_cli

    sink = _RecordingSink()
    seen: list[dict[str, object]] = []

    def build(descriptor: object, **kwargs: object) -> _RecordingSink:
        seen.append({"descriptor": descriptor, **kwargs})
        return sink

    monkeypatch.setattr(probe_cli, "build_sink_for_run", build)
    assert main(ProbeSetup(tmp_path).train()) == 0
    capsys.readouterr()
    assert sink.finishes == ["FINISHED"]
    assert [r["step"] for r in sink.records if r["schema"] == "h-training-step-v2"] == [  # type: ignore[index]
        1,
        2,
        1,
        2,
        3,
        4,
    ]
    (call,) = seen
    results_path = call["results_path"]
    assert isinstance(results_path, Path)
    assert tmp_path in results_path.parents and (tmp_path / "run") not in results_path.parents
    assert set(call) == {"descriptor", "results_path", "metric_extractor"}


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt(), SystemExit(3)])
def test_the_probe_finishes_failed_on_every_raised_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exc: BaseException,
) -> None:
    """M5-probe: an ``except Exception: failed = True`` shape would finish an interrupted
    run FINISHED. The original exception keeps its existing mapping (RuntimeError is
    the handler's rc 1; the BaseExceptions propagate)."""
    from entropic_substrate.composition import cli as probe_cli

    sink = _RecordingSink()
    monkeypatch.setattr(probe_cli, "build_sink_for_run", lambda *_a, **_k: sink)

    def explode(*_a: object, **_k: object) -> object:
        raise exc

    monkeypatch.setattr(probe_cli, "run_probe_training", explode)
    args = ProbeSetup(tmp_path).train()
    if isinstance(exc, RuntimeError):
        assert main(args) == 1
        assert json.loads(capsys.readouterr().out)["status"] == "execution_defect"
    else:
        with pytest.raises(type(exc)):
            main(args)
    assert sink.finishes == ["FAILED"]


def test_a_refused_request_builds_no_sink_and_creates_no_results_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """K4: the sink is built only on the training branch, after every validation
    refusal, so a refused manifest leaves no run and no file behind."""
    from entropic_substrate.composition import cli as probe_cli

    built: list[object] = []
    monkeypatch.setattr(
        probe_cli, "build_sink_for_run", lambda *a, **k: built.append((a, k)) or _RecordingSink()
    )
    args = ProbeSetup(tmp_path).train(pause_after={"phase": "prediction", "step": 1})
    assert main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
    assert built == [] and not (tmp_path / "run.observability").exists()


def test_a_sink_construction_failure_is_an_execution_defect_never_invalid_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """G10: ``MlflowStartupError`` is a RuntimeError; it must be reported past the
    validation stage (rc 1, execution_defect), not as a refused manifest (rc 2)."""
    from entropic_substrate.composition import cli as probe_cli
    from entropic_substrate.ladder.capture.reader.sinks import MlflowStartupError

    def unreachable(*_a: object, **_k: object) -> object:
        raise MlflowStartupError("tracking server unreachable")

    monkeypatch.setattr(probe_cli, "build_sink_for_run", unreachable)
    assert main(ProbeSetup(tmp_path).train()) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "execution_defect" and report["error_type"] == "MlflowStartupError"
    assert not list((tmp_path / "run").iterdir())


def test_a_fresh_probe_never_appends_to_a_previous_runs_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mlflow_off: None
) -> None:
    """G10: JsonlSink appends. A fresh run (``resume`` null) over a results file that
    already holds a previous run's records is refused loudly, before anything runs."""
    stale = tmp_path / "run.observability" / "metrics.jsonl"
    stale.parent.mkdir()
    stale.write_text('{"schema": "h-training-step-v2", "stale": true}\n', encoding="utf-8")
    assert main(ProbeSetup(tmp_path).train()) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid_request"
    assert stale.read_text(encoding="utf-8") == '{"schema": "h-training-step-v2", "stale": true}\n'
    assert not list((tmp_path / "run").iterdir())


def test_a_resumed_probe_continues_the_same_results_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mlflow_off: None
) -> None:
    """The pause->resume pin: the resume appends to the paused invocation's file, each
    committed (phase, step) exactly once, and still no lifecycle record."""
    setup = ProbeSetup(tmp_path)
    pause_args = setup.train("pause.json", pause_after={"phase": "prediction", "step": 2})
    assert main(pause_args) == 0
    paused = json.loads(capsys.readouterr().out)
    assert paused["status"] == "paused"
    metrics = tmp_path / "run.observability" / "metrics.jsonl"
    first = _records(metrics)
    assert first, "the paused invocation already wrote its committed rows"

    resume_args = setup.train("resume.json", resume=paused["receipt"])
    assert main(resume_args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "training_complete"
    rows = _records(metrics)
    assert rows[: len(first)] == first
    committed = [
        (r["phase"], r["step"])
        for r in rows
        if r.get("schema") == "h-training-step-v2" and r.get("status") == "committed"
    ]
    assert committed == [
        ("codec_warmup", 1),
        ("codec_warmup", 2),
        ("prediction", 1),
        ("prediction", 2),
        ("prediction", 3),
        ("prediction", 4),
    ]
    assert not [r for r in rows if r.get("schema") == "h-run-lifecycle-v1"]
