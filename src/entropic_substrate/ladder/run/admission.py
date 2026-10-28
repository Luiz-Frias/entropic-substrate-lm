"""The one admission port for H run classes: which class may run where, at which stage.

Rung A (operator ruling 2026-10-27) gathers the class gates that were scattered across
the worker request, runner, worker command, execution/completion receipts and launch
into this table, without changing a single verdict (pinned by
``tests/test_h_run_admission_sites.py``). The policy is a constant: rung B replaces
``RUN_ADMISSION`` with one derived from a sealed ``RunAuthority`` artifact.

Predicates only. Every gate keeps its own refusal text, so the port never raises a
gate message; it answers questions and the site decides what to say.

Every gate reads this table: the launch (ladder/run/h_probe_launch.py) and the five
``h-checkpoint-reader-v2`` members worker_request, runner, worker_command,
worker_execution and worker_completion. This module is itself a member of that reader set
(door amendment ruled 2026-10-27), so the policy the members enforce is bound into the
checkpoint identity and changing it re-pins.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

type Surface = Literal["local_worker", "remote_worker", "launch"]
type WorkerSurface = Literal["local_worker", "remote_worker"]

H_STAGES = ("H0", "H1", "H2", "H3", "H4", "H5")
SURFACES: tuple[Surface, ...] = ("local_worker", "remote_worker", "launch")
WORKER_SURFACES: tuple[WorkerSurface, ...] = ("local_worker", "remote_worker")

# The ``kind`` an execution receipt records for each worker surface, and that the
# completion admission demands back. The wire values predate the port.
RUNTIME_KIND: Mapping[WorkerSurface, str] = MappingProxyType(
    {"local_worker": "local_tiny", "remote_worker": "remote_tiny"}
)


@dataclass(frozen=True, slots=True)
class RunAdmission:
    """``(run_class, stage) -> surfaces``, each class's one worker surface, its cost guards."""

    surfaces: Mapping[tuple[str, str], frozenset[Surface]]
    worker_surfaces: Mapping[str, WorkerSurface]
    cost_guards: Mapping[str, frozenset[str]]

    def __post_init__(self) -> None:
        for (run_class, stage), admitted in self.surfaces.items():
            if stage not in H_STAGES or not admitted <= frozenset(SURFACES):
                raise ValueError("admission table names an unknown stage or surface")
            workers = admitted & frozenset(WORKER_SURFACES)
            if workers and (
                len(workers) != 1 or self.worker_surfaces.get(run_class) not in workers
            ):
                raise ValueError("a run class is admitted on exactly its one worker surface")
        if not set(self.worker_surfaces.values()) <= set(WORKER_SURFACES):
            raise ValueError("admission table names an unknown worker surface")
        for run_class in self.cost_guards:
            if not any(
                c == run_class and "launch" in admitted
                for (c, _), admitted in self.surfaces.items()
            ):
                raise ValueError("cost guards belong only to a class admitted at launch")
        object.__setattr__(self, "surfaces", MappingProxyType(dict(self.surfaces)))
        object.__setattr__(self, "worker_surfaces", MappingProxyType(dict(self.worker_surfaces)))
        object.__setattr__(self, "cost_guards", MappingProxyType(dict(self.cost_guards)))

    def worker_surface(self, run_class: str) -> WorkerSurface | None:
        """The one worker surface a class executes on; ``None`` when it has none."""
        return self.worker_surfaces.get(run_class)

    def admits_class(self, run_class: str) -> bool:
        """Whether the class may hold a worker request or receipt at any stage."""
        return self.worker_surface(run_class) is not None

    def admits(self, run_class: str, stage: str, surface: Surface | None) -> bool:
        """Whether ``run_class`` at ``stage`` is admitted on ``surface``."""
        return surface is not None and surface in self.surfaces.get((run_class, stage), ())

    def admits_cost_guard(self, run_class: str, cost_guard: str) -> bool:
        """Whether a launch of ``run_class`` may declare ``cost_guard``.

        The guard arrives from a parsed manifest, so a non-string (possibly unhashable)
        value is refused here rather than raising ``TypeError`` past the site's message.
        """
        return type(cost_guard) is str and cost_guard in self.cost_guards.get(run_class, ())


RUN_ADMISSION = RunAdmission(
    surfaces={
        **{("local_tiny", stage): frozenset({"local_worker"}) for stage in H_STAGES},
        ("remote_tiny", "H0"): frozenset({"remote_worker", "launch"}),
    },
    worker_surfaces={"local_tiny": "local_worker", "remote_tiny": "remote_worker"},
    cost_guards={"remote_tiny": frozenset({"host_watchdog_tiny_only"})},
)
