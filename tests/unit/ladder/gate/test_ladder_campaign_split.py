"""Pytest for the p5b split: `evidence.ladder_campaign`'s five artifact
functions move into `kernel.artifacts` (RED-first, brief p5b-ladder-campaign,
ACCEPTANCE item 1).

RETIRED 2026-09-26 (p13-sweep, B2 row 12, forwarder-identity assertions
retired): the two identity arms this file used to carry -- the `__legacy`
remainder's own re-export, and the consumer-facing old-dotted-name path
through the typed alias shim (`entropic_substrate.evidence.ladder_campaign`)
-- both resolved through machinery this phase deletes (`__legacy/` and the
`evidence/` facade directory). Both paths now land at the SAME real module,
`entropic_substrate.ladder.gate.ladder_campaign`, so the two arms would be
degenerate duplicates of each other and of the direct landed-module check
below, never distinguishing anything. What remains load-bearing is that the
five moved names live in `kernel.artifacts`, not in the module they moved
out of.
"""

from __future__ import annotations

import pytest

import entropic_substrate.kernel.artifacts as _kernel_artifacts

pytestmark = pytest.mark.unit

_MOVED_NAMES = (
    "artifact_mapping",
    "artifact_from_mapping",
    "read_bound_artifact",
    "object_fields",
    "read_json_artifact",
)


@pytest.mark.parametrize("name", _MOVED_NAMES)
def test_moved_name_module_is_kernel_artifacts(name: str) -> None:
    obj = getattr(_kernel_artifacts, name)
    assert obj.__module__ == "entropic_substrate.kernel.artifacts"


@pytest.mark.parametrize("name", _MOVED_NAMES)
def test_gate_module_reexports_the_kernel_object_by_identity(name: str) -> None:
    """The gate module the five names moved OUT of still re-exports them
    (its surviving consumers reach these names through it) -- by identity,
    never a parallel re-definition."""
    import entropic_substrate.ladder.gate.ladder_campaign as _gate_module

    assert getattr(_gate_module, name) is getattr(_kernel_artifacts, name)
