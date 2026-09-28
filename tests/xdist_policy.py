"""Thread policy for the unit suite under pytest-xdist (FOR-1618 P4b).

``tests/conftest.py`` pins ``torch.set_num_threads`` for every test. The
pinned value used to be a hardcoded 2. Under ``-n auto`` on a 2-vCPU runner
that is 2 workers x 2 intra-op threads on 2 CPUs before any subprocess a
test spawns, i.e. oversubscription with spin-waiting OpenMP pools. xdist
sets ``PYTEST_XDIST_WORKER`` (``gw0``, ``gw1``, ...) in every worker's
environment and never in a serial (``-n 0`` / no xdist) session, so the
presence of that variable is the exact signal for "I share the CPUs with
other workers". Presence, not value: an empty string still means a worker
set it.
"""

from __future__ import annotations

from collections.abc import Mapping

XDIST_WORKER_ENV = "PYTEST_XDIST_WORKER"
THREADS_SERIAL = 2
THREADS_PER_XDIST_WORKER = 1


def torch_threads_for(env: Mapping[str, str]) -> int:
    """Intra-op thread count for a test process with environment ``env``."""
    if XDIST_WORKER_ENV in env:
        return THREADS_PER_XDIST_WORKER
    return THREADS_SERIAL
