"""The one admission port for H run classes: which class may run where, at which stage.

Rung A (operator ruling 2026-10-27) gathers the class gates that were scattered across
the worker request, runner, worker command, execution/completion receipts and launch
into this table, without changing a single verdict (pinned by
``tests/test_h_run_admission_sites.py``).

Rung B (operator ruling 2026-10-27, "option B") admits the scientific classes ``pilot``
and ``confirmatory`` at H1-H5 on the remote worker and the launch, but only under a
sealed ``RunAuthority`` (``ladder/run/run_authority.py``) whose stage and class match
the run and whose seal still verifies. The table says where a class may run; the
authority says that this run may. Smoke classes carry no authority and keep every
rung-A verdict and message.

Predicates only. Every gate keeps its own refusal text, so the port never raises a
gate message; it answers questions and the site decides what to say. The authority
refusals are new text with no legacy site wording, so the port names them and every
site raises the name verbatim.

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

from entropic_substrate.ladder.run.run_authority import SCIENTIFIC_RUN_CLASSES, RunAuthority

type Surface = Literal["local_worker", "remote_worker", "launch"]
type WorkerSurface = Literal["local_worker", "remote_worker"]

H_STAGES = ("H0", "H1", "H2", "H3", "H4", "H5")
SURFACES: tuple[Surface, ...] = ("local_worker", "remote_worker", "launch")
WORKER_SURFACES: tuple[WorkerSurface, ...] = ("local_worker", "remote_worker")

SCIENTIFIC_STAGES = ("H1", "H2", "H3", "H4", "H5")
TINY_GUARD = "host_watchdog_tiny_only"
BUDGETED_GUARD = "host_watchdog_budgeted"

# The ``kind`` an execution receipt records for each run class, and that the completion
# admission demands back. The smoke wire values predate the port (rung A keyed them by
# surface, which coincided with the class); rung B keys them by class so a pilot on the
# remote worker is never recorded as ``remote_tiny``.
RUNTIME_KIND: Mapping[str, str] = MappingProxyType(
    {
        "local_tiny": "local_tiny",
        "remote_tiny": "remote_tiny",
        "pilot": "remote_pilot",
        "confirmatory": "remote_confirmatory",
    }
)

SMOKE_AUTHORITY = "smoke run classes carry no run authority"
NO_AUTHORITY = "scientific run requires a sealed run authority"
UNSEALED = "run authority fails its seal"
OTHER_CLASS = "run authority names another run class"
OTHER_STAGE = "run authority names another stage"


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
        for run_class, guards in self.cost_guards.items():
            if not any(
                c == run_class and "launch" in admitted
                for (c, _), admitted in self.surfaces.items()
            ):
                raise ValueError("cost guards belong only to a class admitted at launch")
            scientific = run_class in SCIENTIFIC_RUN_CLASSES
            if not scientific and BUDGETED_GUARD in guards:
                raise ValueError("smoke classes keep the tiny-only guard")
            if scientific and guards - {BUDGETED_GUARD}:
                raise ValueError("scientific classes take only the budgeted guard")
        for run_class, surface in self.worker_surfaces.items():
            if run_class in SCIENTIFIC_RUN_CLASSES and surface != "remote_worker":
                raise ValueError("scientific classes run only on the remote worker")
        object.__setattr__(self, "surfaces", MappingProxyType(dict(self.surfaces)))
        object.__setattr__(self, "worker_surfaces", MappingProxyType(dict(self.worker_surfaces)))
        object.__setattr__(self, "cost_guards", MappingProxyType(dict(self.cost_guards)))

    def worker_surface(self, run_class: str) -> WorkerSurface | None:
        """The one worker surface a class executes on; ``None`` when it has none."""
        return self.worker_surfaces.get(run_class)

    def admits_class(self, run_class: str) -> bool:
        """Whether the class may hold a worker request or receipt at any stage."""
        return self.worker_surface(run_class) is not None

    def authority_refusal(self, run_class: str, stage: str, authority: object) -> str | None:
        """The named reason ``authority`` cannot carry this run, or ``None`` when it can.

        An unknown class is the class gate's to refuse, so this abstains. A smoke class
        must carry none. A scientific class must carry a ``RunAuthority`` that still
        verifies against its seal (re-checked here, not trusted from construction) and
        names this class and stage.
        """
        if run_class not in self.worker_surfaces:
            return None
        if run_class not in SCIENTIFIC_RUN_CLASSES:
            return None if authority is None else SMOKE_AUTHORITY
        if type(authority) is not RunAuthority:
            return NO_AUTHORITY
        if not authority.verify():
            return UNSEALED
        if authority.run_class != run_class:
            return OTHER_CLASS
        if authority.stage != stage:
            return OTHER_STAGE
        return None

    def admits(
        self,
        run_class: str,
        stage: str,
        surface: Surface | None,
        authority: RunAuthority | None = None,
    ) -> bool:
        """Whether ``run_class`` at ``stage`` is admitted on ``surface`` under ``authority``."""
        return (
            surface is not None
            and surface in self.surfaces.get((run_class, stage), ())
            and self.authority_refusal(run_class, stage, authority) is None
        )

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
        **{
            (run_class, stage): frozenset({"remote_worker", "launch"})
            for run_class in SCIENTIFIC_RUN_CLASSES
            for stage in SCIENTIFIC_STAGES
        },
    },
    worker_surfaces={
        "local_tiny": "local_worker",
        "remote_tiny": "remote_worker",
        **dict.fromkeys(SCIENTIFIC_RUN_CLASSES, "remote_worker"),
    },
    cost_guards={
        "remote_tiny": frozenset({TINY_GUARD}),
        **dict.fromkeys(SCIENTIFIC_RUN_CLASSES, frozenset({BUDGETED_GUARD})),
    },
)
