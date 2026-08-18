"""Static contract: .coderabbit.yaml conforms to CodeRabbit's own schema.

The schema at ``tests/fixtures/coderabbit-schema.v2.json`` is a VENDORED
SNAPSHOT of https://coderabbit.ai/integrations/schema.v2.json, fetched
2026-08-17 (JSON carries no comments, so the date lives here). MEASURED the
same day on the live PR banner: CodeRabbit silently IGNORED three
unrecognized top-level keys — ``ignore``, ``abort_on_changes_requested``,
``abort_on_close`` — meaning the path-ignore list and the abort behavior this
repo thought it had configured were not in effect at all.

The schema is strict at the root (``additionalProperties: false``) but LAX
inside ``reviews`` (no additionalProperties), so pure schema validation
cannot catch a key placed one level too deep. Hence the second half: explicit
pins for the misplacements found on 2026-08-17 — they fail closed even where
the schema shrugs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = Path(__file__).parent / "fixtures" / "coderabbit-schema.v2.json"


def _config() -> dict[str, Any]:
    with (REPO_ROOT / ".coderabbit.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_coderabbit_config_validates_against_the_vendored_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(_config(), schema)


def test_no_silently_ignored_top_level_keys() -> None:
    """The exact three keys the 2026-08-17 PR banner reported as unrecognized.

    Redundant with the root's additionalProperties today — kept explicit so
    the regression stays named even if a future schema snapshot relaxes the
    root the way ``reviews`` already is.
    """
    config = _config()
    for key in ("ignore", "abort_on_changes_requested", "abort_on_close"):
        assert key not in config, (
            f"top-level {key!r} is not a CodeRabbit config key — it is silently "
            f"ignored (measured on the PR banner 2026-08-17); see this file's "
            f"docstring for where each intent actually lives"
        )


def test_abort_on_close_lives_under_reviews() -> None:
    """Valid key, wrong level was the original defect — pin the right level."""
    assert "abort_on_close" in _config()["reviews"]


def test_path_filters_live_under_reviews_not_auto_review() -> None:
    """Schema places ``path_filters`` at ``reviews.*``; ``auto_review`` has no
    such property, and the schema's laxity inside ``reviews`` means a
    misplaced copy validates anyway — this pin is the only thing that fails.
    """
    config = _config()
    assert "path_filters" not in config["reviews"].get("auto_review", {}), (
        "path_filters under auto_review is not in the schema and is silently "
        "dropped — it belongs at reviews.path_filters"
    )
    filters = config["reviews"].get("path_filters", [])
    assert filters, "reviews.path_filters must carry the exclusion globs"
    # The never-review intent from the retired top-level `ignore:` block —
    # spot-pin the entries whose loss would quietly re-enable review of
    # vendored submodules and DVC-tracked bulk.
    for required in ("!**/references/**", "!**/data/**", "!**/results/**"):
        assert required in filters, f"exclusion glob {required!r} lost in the migration"
