"""FOR-1637 (ruling D2): the worker's lines on a REAL process stderr.

``python -m ...worker_command run`` always builds the real R2 adapter, which
exits 2 at ``object_client`` without credentials and, with them, writes a full
tiny run under the production ``h-probes/v1/<nonce>/`` prefix -- not
admissible from a unit test. So the child here does exactly what
``worker_command.main`` does (``configure_logging()`` and then
``run_worker_command(...)``) over the in-memory ``_ObjectStore``, and the test
asserts on the child's REAL stderr: the stream a RunPod container log shows.

Child hermeticity (G15): ``sys.executable``, a tmp cwd (``main()`` writes the
cwd-relative default ``logs/entropic-substrate.jsonl``), the repo root,
``src`` and ``tests`` on ``PYTHONPATH`` rather than relying on cwd, stdin closed, and an
explicit environment with every MLflow and R2 variable removed so no gate env
can reach a live service.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared

import entropic_substrate
from entropic_substrate.kernel.runtime.logging import configure_logging
from entropic_substrate.ladder.run.worker_command import run_worker_command

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
# The src tree THIS process imported (the checkout's own src in a normal run),
# so the child provably runs the same code the parent was handed.
SRC = Path(entropic_substrate.__file__).resolve().parents[1]

# MEASURED 2026-10-12 on the development Mac, idle host, three consecutive runs:
# the whole child (interpreter + torch import + one tiny run + exit) took
# 19.8 s, 19.9 s and 19.8 s wall. 300 s is a HANG guard (about 15x the measured
# runtime, covering the 3-lane / `-n auto` load the pre-commit suite adds), not
# a tuned budget: it is never to be raised to make this test pass (house rule 3).
_CHILD_TIMEOUT_S = 300

_CHILD = """
import sys
from pathlib import Path

from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.kernel.runtime.logging import configure_logging
from entropic_substrate.ladder.run.worker_command import run_worker_command
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

root = Path(sys.argv[1])
configure_logging()  # exactly what worker_command.main() does first
request, inputs = prepared(root)
objects = _ObjectStore()
marker = publish_probe_request(objects, request, inputs)
workspace = root / "child-worker"
workspace.mkdir()
raise SystemExit(
    run_worker_command(
        [
            "run", "--repo-root", str(root), "--max-object-bytes", "10000000",
            "--max-objects", "100", "--workspace-root", str(workspace),
            "--request-key", marker.key, "--request-sha256", marker.sha256,
            "--request-size", str(marker.size_bytes), "--local-tiny",
        ],
        environ={},
        objects=objects,
    )
)
"""


def _child_env() -> dict[str, str]:
    scrubbed = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MLFLOW_", "ENTROPIC_MLFLOW", "AWS_"))
    }
    # tests/ joins the child's path because the child reuses this suite's
    # fixtures (``tests/conftest.py`` does that for in-process runs only).
    scrubbed["PYTHONPATH"] = os.pathsep.join([str(REPO), str(SRC), str(REPO / "tests")])
    scrubbed["PYTHONUNBUFFERED"] = "1"
    return scrubbed


def test_a_real_child_process_stderr_carries_each_worker_line_exactly_once(tmp_path: Path) -> None:
    work = tmp_path / "child"
    work.mkdir()
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD, str(work)],
        cwd=work,
        env=_child_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=_CHILD_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    # The JSON response channel is untouched: one object on stdout (G6).
    (response,) = [line for line in completed.stdout.splitlines() if line.strip()]
    assert json.loads(response)["status"] == "worker_returned"

    lines = completed.stderr.splitlines()

    def messages(event: str) -> list[str]:
        # "<local ts> | LEVEL    | <message>": the message leads with the event name.
        return [
            line.split(" | ", 2)[2]
            for line in lines
            if line.count(" | ") >= 2 and line.split(" | ", 2)[2].startswith(f"{event} ")
        ]

    # configure_logging() runs a SECOND time inside run_worker_command (G5):
    # nothing may be printed twice.
    for once in ("worker.start", "worker.end", "completion.written"):
        assert len(messages(once)) == 1, (once, completed.stderr[-3000:])
    phases = messages("phase.start")
    assert (
        len(phases) == 2 and "phase=codec_warmup" in phases[0] and "phase=prediction" in phases[1]
    )
    for event in ("checkpoint.seal", "evaluation.end", "publish.end", "sink.finish.end"):
        assert messages(event), (event, completed.stderr[-3000:])
    unique_events = [
        line
        for line in lines
        if any(f" | {e} " in line for e in ("worker.start", "worker.end", "phase.start"))
        or " | checkpoint.seal " in line
    ]
    assert len(unique_events) == len(set(unique_events)), "a worker line was printed twice"
    # The observability root exists and holds the structured file.
    assert (work / "child-worker.observability" / "worker-log.jsonl").is_file()


def test_host_publish_verb_stderr_now_carries_publish_lines_and_stays_otherwise_clean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """G7: ``publish_verified_object`` also runs host-side (the ``publish`` verb,
    the stop-marker put). Its lines reach the verb's process stderr; the injected
    ``stdout``/``stderr`` channels keep their machine-parsed contract."""
    # Bound in the BODY (see `bind_logs` in test_h_worker_log_lines.py): sinks
    # bound by a fixture write into a closed stream on pytest 9.1.1, which
    # would make the assertion below match loguru's "Logging error" dump text.
    configure_logging(json_path=None)
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    out, err = StringIO(), StringIO()
    status = run_worker_command(
        [
            "publish",
            "--repo-root",
            str(tmp_path),
            "--max-object-bytes",
            "10000000",
            "--max-objects",
            "100",
            "--artifact-root",
            str(inputs),
            "--request-file",
            "request.json",
        ],
        environ={},
        objects=objects,
        stdout=out,
        stderr=err,
    )
    assert status == 0 and err.getvalue() == ""
    assert json.loads(out.getvalue())["status"] == "request_published"
    human = capsys.readouterr().err
    assert "Logging error" not in human
    # The line SHAPE, not a substring a dump could also satisfy: the substrate's
    # `<ts> | LEVEL | message` format (the host verbs keep it; moving them to a
    # JSON sink is filed as a follow-up, not done here).
    start = re.compile(
        rf"^\S+ \S+ \S+ \| INFO\s+\| publish\.start key={re.escape(request.prefix)}/request\.json bytes=\d+$",
        re.M,
    )
    end = re.compile(
        rf"^\S+ \S+ \S+ \| INFO\s+\| publish\.end key={re.escape(request.prefix)}/request\.json bytes=\d+ seconds=\d+\.\d{{3}}$",
        re.M,
    )
    assert start.search(human), human
    assert end.search(human), human
