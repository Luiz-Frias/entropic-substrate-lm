"""FOR-1659: the three host components emit through ONE helper, ``h_probe_host_log.emit``.

Slice 5 left three private ``_emit`` copies (controller, stop, monitor). The helper is a
library: no ``main``, no ``print``, not an entrypoint row, not a reader-tuple member.
What the exact-stderr pins cannot see (the structured fields, the ``logger`` attribution of
the JSON row, the order of the rendered fields) is pinned here, by name.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import entropic_substrate
from entropic_substrate.composition.entrypoints import h_wave_monitor
from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.run.worker import (
    h_probe_controller,
    h_probe_host_log,
    h_probe_log_events,
    h_probe_stop,
)
from entropic_substrate.ladder.run.worker.h_probe_host_log import emit

pytestmark = pytest.mark.unit

PKG = Path(entropic_substrate.__file__).resolve().parent
WORKER_DIR = PKG / "ladder/run/worker"
MONITOR = PKG / "composition/entrypoints/h_wave_monitor.py"
HELPER = WORKER_DIR / "h_probe_host_log.py"


@contextmanager
def recorded() -> Iterator[list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []

    def sink(message: Any) -> None:
        record = message.record
        rows.append(
            {
                "level": record["level"].name,
                "message": record["message"],
                "name": record["name"],
                "extra": {k: v for k, v in record["extra"].items() if k != "local_ts"},
            }
        )

    handler = logger.add(sink, level="DEBUG")
    try:
        yield rows
    finally:
        logger.remove(handler)


# ------------------------------------------------------------ what one emission is
def test_emit_binds_the_event_and_every_field_and_renders_the_one_sentence() -> None:
    with recorded() as rows:
        emit("warning", "host.poll.tick", pod_name="p", step=3, ratio=0.5)
    (row,) = rows
    assert row["level"] == "WARNING"
    assert row["extra"] == {"event": "host.poll.tick", "pod_name": "p", "step": 3, "ratio": 0.5}
    assert row["message"] == h_probe_log_events.render(
        "host.poll.tick", pod_name="p", step=3, ratio=0.5
    )
    assert row["message"] == "host.poll.tick pod_name=p step=3 ratio=0.500"


def test_emit_with_no_fields_binds_only_the_event() -> None:
    with recorded() as rows:
        emit("info", "host.monitor.end")
    (row,) = rows
    assert row["extra"] == {"event": "host.monitor.end"} and row["message"] == "host.monitor.end"


def test_a_brace_in_a_field_value_never_reaches_a_formatter() -> None:
    with recorded() as rows:
        emit("info", "host.poll.tick", note="{x} {0}")
    assert rows[0]["message"] == "host.poll.tick note={x} {0}"


def test_an_unknown_level_is_a_hard_error_not_a_silent_drop() -> None:
    with pytest.raises(AttributeError):
        emit("shout", "host.poll.tick")


def test_the_record_is_attributed_to_the_module_that_called_emit() -> None:
    """Measured with a control: ``getattr(logger.bind(...), level)(...)`` called from inside
    the helper leaves stderr identical but rewrites the JSON row's ``logger`` key (loguru's
    ``record['name']``) to the helper's module; ``opt(depth=1)`` keeps the caller's."""
    with recorded() as rows:
        emit("info", "host.poll.tick")
    (row,) = rows
    assert row["name"] == __name__
    assert row["name"] != h_probe_host_log.__name__


# --------------------------------------------------- the callers' field order and DRY
def _emit_calls(path: Path) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "emit"
    ]


@pytest.mark.parametrize("name", ["h_probe_controller.py", "h_probe_stop.py"])
def test_every_pod_scoped_call_renders_pod_name_first_ahead_of_any_expansion(name: str) -> None:
    """``{"pod_name": pod_name, **fields}`` was the slice-5 bound dict: pod_name leads every
    multi-field sentence. ``emit(..., **fields, pod_name=pod_name)`` would reorder them all."""
    calls = _emit_calls(WORKER_DIR / name)
    assert calls, f"{name} makes no emit call; the scan is mis-rooted"
    for call in calls:
        first = call.keywords[0]
        assert first.arg == "pod_name", ast.unparse(call)
        assert isinstance(first.value, ast.Name) and first.value.id == "pod_name"
        assert [k.arg for k in call.keywords].count("pod_name") == 1, ast.unparse(call)


def _private_emit_definitions(path: Path) -> list[str]:
    return [
        f"{path.name}:{node.lineno}"
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_emit"
    ]


def test_no_host_component_defines_a_private_emit() -> None:
    scanned = [*sorted(WORKER_DIR.glob("*.py")), MONITOR]
    names = {p.name for p in scanned}
    assert {"h_probe_controller.py", "h_probe_stop.py", "h_wave_monitor.py"} <= names
    assert [d for path in scanned for d in _private_emit_definitions(path)] == []


def test_the_three_components_share_the_one_helper_and_the_one_renderer() -> None:
    assert h_probe_controller.emit is emit
    assert h_probe_stop.emit is emit
    assert h_wave_monitor.emit is emit
    assert h_probe_host_log.render is h_probe_log_events.render


# ------------------------------------------------------- the helper is a plain library
def test_the_helper_is_a_library_with_a_closed_import_set_and_no_io() -> None:
    tree = ast.parse(HELPER.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name for a in node.names}
        if isinstance(node, ast.ImportFrom):
            modules.add(("." * node.level) + (node.module or ""))
    assert modules <= {
        "__future__",
        "entropic_substrate.kernel.runtime.logging",
        "entropic_substrate.ladder.run.worker.h_probe_log_events",
    }
    assert "mlflow" not in " ".join(modules)
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert defined == {"emit"}
    calls = {
        getattr(n.func, "id", getattr(n.func, "attr", ""))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert not calls & {"print", "open", "getenv", "system"}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert "environ" not in names and "__name__" not in names  # no env read, no __main__ guard
    assert not any(
        isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", "") == "object"
        for n in tree.body
    ), "a module-level Name = object() sentinel would join the frozen sentinel census"
