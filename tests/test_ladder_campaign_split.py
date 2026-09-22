"""Pytest for the p5b split: `evidence.ladder_campaign`'s five artifact
functions move into `kernel.artifacts` (RED-first, brief p5b-ladder-campaign,
ACCEPTANCE item 1).

Two identity arms, both required (architect constraint, design gate
2026-09-22): the `__legacy` attribute path (the remainder's own re-export)
AND the consumer-facing old-dotted-name path through the typed alias shim
(`entropic_substrate.evidence.ladder_campaign`), because after this phase no
landed module imports the remainder directly -- every surviving `__legacy`
consumer and every test reaches these five names through the shim, so an
arm that inspects only the `__legacy` attribute would leave the path a
mis-targeted fragment row actually breaks unpinned.

Mutation (house rule 5): defining one of these names directly in the
remainder (shadowing its re-export) must turn the identity arm RED -- the
shadowed name's `__module__` becomes `entropic_substrate.__legacy.evidence
.ladder_campaign`, and the `is` comparison against `kernel.artifacts`'s own
object fails.
"""

from __future__ import annotations

import pytest

import entropic_substrate.__legacy.evidence.ladder_campaign as _legacy_remainder
import entropic_substrate.evidence.ladder_campaign as _shim
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
def test_legacy_remainder_attribute_is_the_kernel_object(name: str) -> None:
    """The `__legacy` remainder re-exports the moved name by identity, not
    by a second definition."""
    assert getattr(_legacy_remainder, name) is getattr(_kernel_artifacts, name)


@pytest.mark.parametrize("name", _MOVED_NAMES)
def test_old_dotted_shim_path_is_the_kernel_object(name: str) -> None:
    """The consumer-facing path every surviving `__legacy` importer and test
    actually uses: `entropic_substrate.evidence.ladder_campaign.<name>` (the
    typed alias shim onto the `__legacy` remainder) resolves to the SAME
    object as `entropic_substrate.kernel.artifacts.<name>`."""
    assert getattr(_shim, name) is getattr(_kernel_artifacts, name)
