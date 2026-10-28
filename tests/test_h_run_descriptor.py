"""The H run surface (FOR-1642, FOR-1643, FOR-1644): what every H sink site declares.

Every H producer that mirrors to MLflow (the pod worker, the local tiny tracer and
the local ``h-train-probe``) builds its run through ``build_sink_for_run`` with one
descriptor shape:

    params = {exp, seed, config_hash, git_sha, run_class, precision, image}
    tags   = {schema, exp, config_hash}   (the pod worker alone adds request_sha256, FOR-1657)

``config_hash`` is ``h_config_hash(profile)``: ``sha256(canonical manifest minus the
sweep ``seed``)[:12]``, the key ``mlflow_aggregate`` groups seed runs by. ``git_sha`` and
``image`` are OMITTED where there is no verified value (local producers have no image;
a ``local_tiny`` worker request has neither), never ``None`` and never a placeholder.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from tests.test_d1_trainer_object_transport import _ObjectStore
from tests.test_h_tiny_tracer import LIMIT, _Builder, _local_profile
from tests.test_run_authority import authority as sealed_authority
from tests.test_sentence_cli import ProbeSetup
from tests.test_sentence_profile import profile
from tests.test_sentence_worker_execution import prepared

from entropic_substrate.composition.entrypoints.cli import main as cli_main
from entropic_substrate.composition.wiring.h_tiny_tracer import run_tiny_trace
from entropic_substrate.ladder.capture.ports.run_descriptor import TrackingRunDescriptor
from entropic_substrate.ladder.capture.reader.sinks import (
    CONFIG_HASH_TAG,
    EXP_TAG,
    git_sha,
    h_config_hash,
)
from entropic_substrate.ladder.run import worker_command
from entropic_substrate.ladder.run.run_authority import RunAuthority
from entropic_substrate.ladder.run.worker_command import run_worker_command
from entropic_substrate.ladder.run.worker_transport import publish_probe_request

pytestmark = pytest.mark.unit

# Golden value for profile() (tests/test_sentence_profile.py) so a silent change of the
# serialization, which would split every seed group across image builds, is a RED test.
# RE-PINNED 2026-10-20 (FOR-1664 door A (ii), DN-1664-GOLDEN): a15d83dd37e6 -> f56a5ac2049d.
# Cause: the manifest schema literal became h-run-profile-v2 and schedule gained
# diagnostic_samples, so seed groups split across the bump by design. New value = the
# documented recipe's output on the staged profile().
# RE-PINNED 2026-10-23 (FOR-1674, operator ruling D1): f56a5ac2049d -> 596bea7c6262.
# Cause: the manifest schema literal became h-run-profile-v3 and schedule gained
# generation_continuations (2 on the tiny smoke profile), so seed groups split across the
# bump by design. New value = the documented recipe's output on the staged profile().
GOLDEN_CONFIG_HASH = "596bea7c6262"  # pragma: allowlist secret


# ------------------------------------------------------------------ h_config_hash
def test_config_hash_is_twelve_lowercase_hex_and_stable() -> None:
    value = h_config_hash(profile())
    assert len(value) == 12 and set(value) <= set("0123456789abcdef")
    assert h_config_hash(profile()) == value


def test_config_hash_equals_the_documented_recipe() -> None:
    manifest = {k: v for k, v in profile().manifest.items() if k != "seed"}
    recipe = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    assert h_config_hash(profile()) == recipe == GOLDEN_CONFIG_HASH


def test_config_hash_is_invariant_under_the_sweep_seed() -> None:
    """M1: the seed is the one axis a sweep varies; hashing it would give every seed its
    own group of one, the failure ``mlflow_aggregate`` cannot recover from."""
    base = profile()
    assert h_config_hash(replace(base, seed=base.seed + 1)) == h_config_hash(base)
    assert replace(base, seed=base.seed + 1).identity != base.identity  # control


def test_config_hash_separates_everything_else_including_source_seed() -> None:
    base = profile()
    assert h_config_hash(replace(base, source_seed=base.source_seed + 1)) != h_config_hash(base)
    assert h_config_hash(replace(base, spine=replace(base.spine, carrier="sphere"))) != (
        h_config_hash(base)
    )
    assert h_config_hash(replace(base, consumer="isotropic")) != h_config_hash(base)


def test_the_reader_sinks_module_still_loads_without_torch() -> None:
    """G8: ``h_config_hash`` takes the profile structurally; importing the reader's sinks
    module must not pull in torch (``core.sentence.profile`` does)."""
    code = (
        "import sys; import entropic_substrate.ladder.capture.reader.sinks; "
        "assert 'torch' not in sys.modules, 'sinks imported torch'"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# ----------------------------------------------------------------- the tracer site
def test_the_tracer_declares_the_h_run_surface(tmp_path: Path) -> None:
    local, source = _local_profile(tmp_path)
    builder = _Builder()
    run_tiny_trace(
        local,
        source,
        tmp_path / "trace",
        max_source_bytes=LIMIT,
        max_artifact_bytes=LIMIT,
        sink_builder=builder,
    )
    (call,) = builder.calls
    descriptor: TrackingRunDescriptor = call["descriptor"]
    chash = h_config_hash(local)
    # TRACER-PARTITION: the -tiny suffix keeps this lifecycle-less producer off the H lane.
    assert descriptor.run_name == f"h-tiny-trace-{local.identity}-tiny"
    assert dict(descriptor.params) == {
        EXP_TAG: local.name,
        "seed": local.seed,
        CONFIG_HASH_TAG: chash,
        "git_sha": git_sha(),
        "run_class": local.run_class,
        "precision": local.precision,
    }
    assert dict(descriptor.tags) == {
        "schema": "h-tiny-trace-v1",
        EXP_TAG: local.name,
        CONFIG_HASH_TAG: chash,
    }
    assert "request_sha256" not in descriptor.tags  # a tracer has no request (FOR-1657)
    assert "image" not in descriptor.params
    assert descriptor.config_hash is None  # the citation key stays E-only (K7)
    assert call["results_path"] == tmp_path / "trace" / "observability" / "metrics.jsonl"


# ------------------------------------------------------------------ the worker site
class _Stop(Exception):
    """Raised by the descriptor-capturing builder: the descriptor is all these tests need."""


def _command_args(workspace: Path, repo: Path, marker: Any) -> list[str]:
    return [
        "run",
        "--repo-root",
        str(repo),
        "--max-object-bytes",
        "10000000",
        "--max-objects",
        "100",
        "--workspace-root",
        str(workspace),
        "--request-key",
        marker.key,
        "--request-sha256",
        marker.sha256,
        "--request-size",
        str(marker.size_bytes),
        "--local-tiny",
    ]


def _worker_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    image: str | None = None,
    commit: str | None = None,
    authority: RunAuthority | None = None,
) -> tuple[Any, TrackingRunDescriptor, dict[str, Any]]:
    request, inputs = prepared(tmp_path)
    objects = _ObjectStore()
    marker = publish_probe_request(objects, request, inputs)
    if image is not None or commit is not None or authority is not None:
        # A remote request carries both; set past the dataclass validation, which
        # refuses them on a local_tiny request (the tests/test_h_worker_log_lines.py idiom).
        # A scientific request carries its sealed authority the same way (rung B).
        object.__setattr__(request, "image", image)
        object.__setattr__(request, "code_commit", commit)
        object.__setattr__(request, "authority", authority)
        monkeypatch.setattr(
            worker_command, "read_published_probe_request", lambda *_a, **_k: request
        )
    seen: dict[str, Any] = {}

    def capture(descriptor: TrackingRunDescriptor, **kwargs: Any) -> Any:
        seen["descriptor"], seen["kwargs"] = descriptor, kwargs
        raise _Stop

    monkeypatch.setattr(worker_command, "build_sink_for_run", capture)
    workspace = tmp_path / "descriptor-worker"
    workspace.mkdir()
    status = run_worker_command(
        _command_args(workspace, tmp_path, marker),
        environ={},
        objects=objects,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert status == 2 and "descriptor" in seen
    return request, seen["descriptor"], seen["kwargs"]


def test_a_local_tiny_worker_omits_git_sha_and_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G1: a local_tiny request carries no image or commit; MLflow params cannot be
    None, and a local ``git_sha()`` would mix an unverified HEAD into a verified key."""
    request, descriptor, _ = _worker_descriptor(tmp_path, monkeypatch)
    chash = h_config_hash(request.profile)
    assert request.image is None and request.code_commit is None
    assert dict(descriptor.params) == {
        EXP_TAG: request.profile.name,
        "seed": request.profile.seed,
        CONFIG_HASH_TAG: chash,
        "run_class": request.profile.run_class,
        "precision": request.profile.precision,
    }
    assert dict(descriptor.tags) == {
        "schema": "h-probe-worker-result-v1",
        EXP_TAG: request.profile.name,
        CONFIG_HASH_TAG: chash,
        # FOR-1657: the launch-bound identity the wave monitor binds an arm by
        "request_sha256": request.digest,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", descriptor.tags["request_sha256"])
    assert "request_sha256" not in descriptor.params  # a tag only; the E-level params stay
    assert descriptor.run_name == f"h-worker-{request.profile.identity}"  # unchanged
    assert descriptor.config_hash is None and descriptor.tiny is True


def test_a_remote_shaped_worker_carries_the_request_commit_and_image_unswapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = "ghcr.io/luiz-frias/entropic-substrate-lm@sha256:" + "d" * 64
    commit = "c" * 40
    request, descriptor, kwargs = _worker_descriptor(
        tmp_path, monkeypatch, image=image, commit=commit
    )
    assert descriptor.params["git_sha"] == commit and descriptor.params["image"] == image
    assert descriptor.params[CONFIG_HASH_TAG] == h_config_hash(request.profile)
    assert descriptor.tags[CONFIG_HASH_TAG] == descriptor.params[CONFIG_HASH_TAG]
    assert descriptor.tags["request_sha256"] == request.digest  # FOR-1657
    assert re.fullmatch(r"[0-9a-f]{64}", descriptor.tags["request_sha256"])
    assert descriptor.run_name == f"h-worker-{request.profile.identity}"  # the name is unchanged
    # G4: the call keeps its keyword shape (descriptor positional, three keywords).
    assert {"results_path", "log_path", "metric_extractor"} <= set(kwargs)


def test_a_scientific_worker_is_tracked_without_the_tiny_opt_in_and_tags_its_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rung B: a pilot's eight-key template carries no ``ENTROPIC_MLFLOW_TINY``, so its
    sink must not be ``tiny`` (``sinks.mlflow_enabled`` would drop the mirror silently),
    and its params name the sealed authority it runs under."""
    sealed = sealed_authority()
    _, descriptor, _ = _worker_descriptor(tmp_path, monkeypatch, authority=sealed)
    assert descriptor.tiny is False
    assert descriptor.params["run_authority"] == sealed.digest


# ------------------------------------------------------------------- the probe site
def test_the_local_probe_declares_the_h_run_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from entropic_substrate.composition import cli as probe_cli

    seen: dict[str, Any] = {}

    def capture(descriptor: TrackingRunDescriptor, **kwargs: Any) -> Any:
        seen["descriptor"], seen["kwargs"] = descriptor, kwargs
        raise _Stop

    monkeypatch.setattr(probe_cli, "build_sink_for_run", capture)
    setup = ProbeSetup(tmp_path)
    args = setup.train()
    # _Stop is not one of the handler's tuple, so it propagates: the descriptor is all
    # this test reads.
    with pytest.raises(_Stop):
        cli_main(args)
    capsys.readouterr()
    descriptor: TrackingRunDescriptor = seen["descriptor"]
    template = setup
    chash = h_config_hash(template.profile)
    assert descriptor.run_name == f"h-train-probe-{template.profile.identity}-tiny"
    assert descriptor.tiny is True and descriptor.config_hash is None
    assert dict(descriptor.params) == {
        EXP_TAG: template.profile.name,
        "seed": template.profile.seed,
        CONFIG_HASH_TAG: chash,
        "git_sha": git_sha(),
        "run_class": template.profile.run_class,
        "precision": template.profile.precision,
    }
    assert dict(descriptor.tags) == {
        "schema": "h-train-probe-v1",
        EXP_TAG: template.profile.name,
        CONFIG_HASH_TAG: chash,
    }
    assert "request_sha256" not in descriptor.tags  # the local probe has no request (FOR-1657)
