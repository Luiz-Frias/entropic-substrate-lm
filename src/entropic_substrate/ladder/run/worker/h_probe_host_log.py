"""The one emission path of the H host components (FOR-1659).

The controller, the operator stop command and the wave monitor each narrate through
``emit``: structured fields travel through ``logger.bind`` and the human sentence is the
vocabulary leaf's ``render`` of the same fields, so stderr and the JSON record come from
one call. This is a library: no ``main``, no environment read, no I/O at import; it is not
an entrypoint and not a member of the reader identity. The leaf ``h_probe_log_events``
stays standard-library only, which is why the logger-bearing helper lives beside it and
not in it.
"""

from __future__ import annotations

from entropic_substrate.kernel.runtime.logging import logger
from entropic_substrate.ladder.run.worker.h_probe_log_events import render

__all__ = ["emit"]


def emit(level: str, event: str, **fields: object) -> None:
    """One host event: structured fields through ``bind``, one human sentence.

    Callers that scope an event to a pod pass ``pod_name=pod_name`` as the FIRST keyword,
    ahead of any ``**fields``, because ``render`` lists fields in keyword order.
    """
    # depth=1 attributes the record to the CALLER's module. Measured 2026-10-16 (loguru,
    # two-module control): a plain bind(...).info(...) inside this helper made
    # record["name"] this module (the JSON row's "logger" key), opt(depth=1) kept the
    # calling module's; stderr was identical in both.
    getattr(logger.bind(event=event, **fields).opt(depth=1), level)(render(event, **fields))
