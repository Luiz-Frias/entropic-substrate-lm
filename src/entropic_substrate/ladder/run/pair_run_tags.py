"""The two DH-50 lifecycle run tags of a pair's MLflow runs and their closed vocabularies.

A LEAF module: standard library only, names only. It is deliberately NOT a member of
``checkpoint._READER_MODULES`` (precedent: ``worker_log_events``): the writers
(``worker_command._PairCellSinks``, driven by ``object_worker._PairObserver`` and
``worker_execution._execute_pair``) are members and the reader (``h_wave_monitor``) is not, so
the one vocabulary both sides import lives outside the reader identity.

Ruling DH-50 (operator, 2026-11-03):

* ``esl.lifecycle`` on EACH cell's run: ``live`` at run start (every begin, so every resume,
  which opens a run of its own); ``paused`` or ``stopped`` when the cell pauses or stops;
  ``completed`` at the cell's end. A crash never resumed keeps ``live``.
* ``esl.pair_outcome`` on the DONOR's run: ``handed_off`` once the recipient's run is open;
  ``stopped_before_handoff`` when the pair stops with no hand-off; ``paused_before_handoff``
  when the donor pauses before the hand-off.

The wave monitor retires a recipient arm whose run does not exist iff the donor's run carries
``esl.pair_outcome = stopped_before_handoff``; every other value, or none, keeps it pending.
"""

from __future__ import annotations

from typing import Final, Literal, get_args

__all__ = [
    "CELL_LIFECYCLES",
    "LIFECYCLE_TAG",
    "PAIR_OUTCOMES",
    "PAIR_OUTCOME_TAG",
    "CellLifecycle",
    "PairOutcome",
]

LIFECYCLE_TAG: Final = "esl.lifecycle"
PAIR_OUTCOME_TAG: Final = "esl.pair_outcome"

CellLifecycle = Literal["live", "paused", "stopped", "completed"]
PairOutcome = Literal["handed_off", "stopped_before_handoff", "paused_before_handoff"]

CELL_LIFECYCLES: Final[frozenset[str]] = frozenset(get_args(CellLifecycle))
PAIR_OUTCOMES: Final[frozenset[str]] = frozenset(get_args(PairOutcome))
