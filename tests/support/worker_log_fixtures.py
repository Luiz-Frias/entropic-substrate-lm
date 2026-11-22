"""Worker log capture helpers shared by the worker log tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from entropic_substrate.kernel.runtime.logging import configure_logging


def bind_logs(path: Path) -> None:
    """Re-bind the substrate's sinks to THIS test: current stderr + a JSON file.

    Called in the test BODY, never from a fixture: MEASURED 2026-10-12 on
    pytest 9.1.1, ``sys.stderr`` at fixture setup is not the object ``capsys``
    reads at call time, so sinks bound by the autouse conftest fixture (or by
    any fixture) write into a closed stream (the same reason
    ``tests/unit/kernel/runtime/test_logging.py`` configures inside its tests).
    """
    configure_logging(json_path=path)


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]
