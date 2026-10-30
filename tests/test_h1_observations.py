"""The H1 observation adapter (FOR-1691): captured H runs to paired pilot contrasts.

Two evidence classes, kept apart on purpose. The ``unit`` tests build SYNTHETIC captured runs
on disk under real H1 pilot profiles validated through ``H1PreservationBlock`` (the captured
``run_start`` carries each profile's identity and ``run_class`` ``pilot``), so the exposure
flows through ``read_h_rows`` -> ``h1_exposure_bytes`` exactly as for a real capture. The
``integration`` tests read the real final-probe bundle ``results/h0-final-probe-1k-v4-92c9db``
(``dvc pull results/h0-final-probe-1k-v4-92c9db.dvc``) as a per-cell k <= 8 DIAGNOSTIC: it has
no construction digests and no pilot class (it is ``remote_tiny``), so it is never fed to
``plan_seed_count``. They RAISE when the payload is absent; a skip would be a false green.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
from tests.test_sentence_profile import profile

from entropic_substrate.core.sentence.profile import HProfile
from entropic_substrate.ladder.analyze import h1_observations, h_aggregate
from entropic_substrate.ladder.analyze.h1_observations import (
    ENDPOINT_METRIC_KEY,
    ENDPOINT_WINDOW,
    ESS_THRESHOLD,
    GATE_METRIC_KEY,
    H1BlockCapture,
    H1BlockObservation,
    H1EvaluationReading,
    H1ExclusionFinding,
    observe_h1_blocks,
    pilot_inputs,
    pilot_inputs_by_threshold,
    project_h1_cell,
)
from entropic_substrate.ladder.analyze.h_aggregate import HRun, load_h_run
from entropic_substrate.ladder.analyze.h_resumed_run import HResumedRun, load_h_resumed_run
from entropic_substrate.ladder.analyze.seed_sizing import (
    ContrastHypothesis,
    PairedSeed,
    PilotContrast,
    PilotVarianceUnknown,
    dual_confidence_intervals,
    plan_seed_count,
)
from entropic_substrate.ladder.register.h_preservation import (
    H1ExclusionReason,
    H1PreservationBlock,
    h1_construction_digest,
)

FAMILY = sha256(b"sealed-design-family").hexdigest()
UNITS = "nats_per_canonical_byte"
CONTRASTS = ("sentence_normalization", "spine_normalization", "interaction")
AXES = (
    ("magnitude", "magnitude"),
    ("sphere", "magnitude"),
    ("magnitude", "sphere"),
    ("sphere", "sphere"),
)
LABELS = tuple(f"h1-{sentence}-{spine}-pilot" for sentence, spine in AXES)
# window evaluations of a default synthetic cell sit at global steps 4, 6, 8, 10, 12 with
# 2, 4, 6, 8, 10 committed prediction steps (warm-up 2, evaluation interval 2)
STANDARD_STEPS = (2, 4, 6, 8, 10)
BASE = (-2.5, -2.0, -2.25, -1.5)  # one asymmetric endpoint per cell (mm, sm, ms, ss)
K8_KEYS = {"endpoint_key": "eval/log_marginal_per_byte_k8", "gate_key": "eval/ess_ratio_median_k8"}


# ----------------------------------------------------------------- synthetic capture builder
@dataclass(frozen=True)
class CellSpec:
    """One synthetic cell. ``endpoints[j]`` is the k = 64 per-byte value of evaluation j
    (``None`` omits k = 64 from that evaluation); ``ratios[j]`` its per-row ESS/k at k = 64.
    ``gaps`` holds ``(evaluation index, k)`` pairs whose FIRST importance row at that k is a
    ``zero_importance_weight_mass`` gap row (three nulls), when that k is present."""

    endpoints: tuple[float | None, ...]
    ratios: tuple[tuple[float, ...], ...] | None = None
    row_bytes: int = 128
    gaps: tuple[tuple[int, int], ...] = ()


def _spec(
    endpoints: Sequence[float | None], *, ratios: tuple[float, ...] = (0.5, 0.5, 0.5)
) -> CellSpec:
    return CellSpec(tuple(endpoints), tuple(ratios for _ in endpoints))


def _entries(k: int, per_byte: float, ratios: Sequence[float], row_bytes: int) -> list[Any]:
    return [
        {
            "effective_sample_size": ratio * k,
            "gap": None,
            "likelihood_relative_mcse": 0.5,
            "log_estimate": per_byte * row_bytes,
        }
        for ratio in ratios
    ]


def _eval_row(identity: str, step: int, spec: CellSpec, index: int) -> dict[str, Any]:
    ratios = (spec.ratios or tuple((0.5, 0.5, 0.5) for _ in spec.endpoints))[index]
    value = spec.endpoints[index]
    by_count: dict[str, Any] = {"8": _entries(8, -3.0, (0.5,) * len(ratios), spec.row_bytes)}
    if value is not None:
        by_count["64"] = _entries(64, value, ratios, spec.row_bytes)
    for gap_index, k in spec.gaps:
        if gap_index == index and str(k) in by_count:
            by_count[str(k)][0] = {
                "effective_sample_size": None,
                "gap": "zero_importance_weight_mass",
                "likelihood_relative_mcse": None,
                "log_estimate": None,
            }
    return {
        "checkpoint_seal": {"path": f"seal-{step}.json", "sha256": sha256(b"seal").hexdigest()},
        "cuda_seconds": 0.25,
        "examples": [
            {
                "boundary_symbols": [1] * len(ratios),
                "source_bytes": [spec.row_bytes] * len(ratios),
                "by_sample_count": by_count,
            }
        ],
        "meaning": "importance_weighted_log_marginal_estimate",
        "run_identity": identity,
        "schema": "h-predictive-evaluation-v1",
        "step": step,
        "wall_seconds": 0.75,
    }


def _step_row(identity: str, phase: str, step: int, source_bytes: int) -> dict[str, Any]:
    return {
        "aggregate_norm": 1.0,
        "decoder_ce": 5.0,
        "future_nll_per_dimension": None if phase == "codec_warmup" else 1.0,
        "objective": 6.0,
        "phase": phase,
        "run_identity": identity,
        "schema": "h-training-step-v2",
        "source_bytes": source_bytes,
        "status": "committed",
        "step": step,
        "wall_seconds": 0.5,
    }


def _log(event: str, **fields: Any) -> dict[str, Any]:
    return {"event": event, "level": "INFO", "message": event, **fields}


def write_run(
    directory: Path,
    identity: str,
    spec: CellSpec,
    *,
    step_bytes: int,
    tag: str,
    run_class: str = "pilot",
) -> tuple[Path, Path]:
    warmup, steps = 2, 2 * len(spec.endpoints)
    request = sha256(f"request-{tag}".encode()).hexdigest()
    metrics: list[dict[str, Any]] = [
        {
            "code_commit": "c" * 40,
            "event": "run_start",
            "image": "example.invalid/image@sha256:" + sha256(b"image").hexdigest(),
            "profile_identity": identity,
            "request_sha256": request,
            "run_class": run_class,
            "runtime": {"accelerator": {"count": 1, "model": "synthetic"}, "pod_id": "pod"},
            "schema": "h-run-lifecycle-v1",
        }
    ]
    log = [_log("worker.start", request_sha256=request)]
    metrics.extend(_step_row(identity, "codec_warmup", s, 7) for s in range(1, warmup + 1))
    log.append(_log("checkpoint.seal", phase="codec_warmup", step=warmup, global_step=warmup))
    for p in range(1, steps + 1):
        metrics.append(_step_row(identity, "prediction", p, step_bytes))
        if p % 2 == 0:
            metrics.append(_eval_row(identity, p, spec, p // 2 - 1))
            log.append(_log("checkpoint.seal", phase="prediction", step=p, global_step=warmup + p))
            log.append(_log("evaluation.start", global_step=warmup + p))
    metrics.append(
        {
            "event": "run_end",
            "result_status": "probe_workload_complete",
            "schema": "h-run-lifecycle-v1",
            "status": "ok",
            "successful_optimizer_steps": warmup + steps,
            "wall_seconds": 12.5,
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    paths = (directory / f"{tag}-metrics.jsonl", directory / f"{tag}-worker-log.jsonl")
    for path, rows in zip(paths, (metrics, log), strict=True):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return paths


def _edit_metrics(
    metrics: Path, edit: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
) -> None:
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    metrics.write_text("".join(json.dumps(r) + "\n" for r in edit(rows)), encoding="utf-8")


def _before_run_end(row: dict[str, Any]) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    def edit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        index = next(i for i, r in enumerate(rows) if r.get("event") == "run_end")
        return [*rows[:index], row, *rows[index:]]

    return edit


def h1_profiles(seed: int, *, encoder_layers: int | None = None) -> tuple[HProfile, ...]:
    """Four real H1 pilot profiles: development curve, CUDA/bf16, no Fisher, no future law."""
    base = profile()
    codec = base.codec if encoder_layers is None else replace(base.codec, encoder_layers=2)
    base = replace(
        base,
        stage="H1",
        name="h1-base-pilot",
        run_class="pilot",
        device="cuda",
        precision="bf16",
        seed=seed,
        source_seed=seed,
        consumer="isotropic",
        codec=codec,
        optimization=replace(base.optimization, future_weight=0),
        schedule=replace(
            base.schedule,
            steps=10,
            importance_samples=(2, 4, 8, 16, 32, 64),
            generation_continuations=5,
            diagnostic_samples=256,
        ),
    )
    return tuple(
        replace(
            base,
            name=f"h1-{sentence}-{spine}-pilot",
            law=replace(base.law, mode=sentence),
            spine=replace(base.spine, carrier=spine),
        )
        for sentence, spine in AXES
    )


def build_block(
    root: Path,
    seed: int,
    specs: Sequence[CellSpec],
    *,
    step_bytes: int = 100,
    profiles: tuple[HProfile, ...] | None = None,
    run_class: str = "pilot",
    independent_warmup: tuple[str, ...] = (),
) -> H1BlockCapture:
    block = H1PreservationBlock(profiles or h1_profiles(seed))
    cells: dict[str, HRun] = {}
    for member, spec in zip(block.profiles, specs, strict=True):
        paths = write_run(
            root / f"seed-{seed}",
            member.identity,
            spec,
            step_bytes=step_bytes,
            tag=member.name,
            run_class=run_class,
        )
        cells[member.name] = load_h_run(*paths)
    return H1BlockCapture(block, cells, independent_warmup)


def standard_specs(index: int, *, noise: bool = True) -> tuple[CellSpec, ...]:
    """Five dyadic window values per cell; cell-specific noise so contrasts have variance."""
    out: list[CellSpec] = []
    for c, base in enumerate(BASE):
        shift = ((index * 7 + c * 3) % 8 - 4) / 32 if noise else 0.0
        out.append(_spec([base + shift + j / 16 for j in range(5)]))
    return tuple(out)


def standard_endpoint(index: int, c: int, *, noise: bool = True) -> float:
    shift = ((index * 7 + c * 3) % 8 - 4) / 32 if noise else 0.0
    return BASE[c] + shift + sum(j / 16 for j in range(5)) / 5


def eight_blocks(root: Path, *, noise: bool = True) -> list[H1BlockCapture]:
    """The synthetic 8-block pilot: canonical byte totals DIFFER across blocks (step_bytes
    100, 110, ...) and agree within each; every block commits the same 10 prediction steps."""
    return [
        build_block(root, 101 + i, standard_specs(i, noise=noise), step_bytes=100 + 10 * i)
        for i in range(8)
    ]


def hypotheses() -> tuple[ContrastHypothesis, ...]:
    return tuple(
        ContrastHypothesis(name, UNITS, "superiority", 0.1, 0.0, None) for name in CONTRASTS
    )


def inputs_for(observations: Sequence[H1BlockObservation], **kwargs: Any) -> Any:
    return pilot_inputs(
        observations, family_digest=FAMILY, units=UNITS, expected_exposure=10, **kwargs
    )


# ------------------------------------------------------------------------ the endpoint
@pytest.mark.unit
class TestEndpoint:
    def test_endpoint_is_the_mean_of_the_last_five_evaluations(self, tmp_path: Path) -> None:
        # six evaluations: the first and the last differ from the window mean, so neither
        # "last evaluation" nor "all evaluations" can produce the asserted value
        values = (-9.0, -2.5, -2.25, -2.0, -1.75, -1.0)
        paths = write_run(tmp_path, "a" * 64, _spec(values), step_bytes=100, tag="c")
        cell = project_h1_cell(load_h_run(*paths))
        assert [r.global_step for r in cell.window_readings] == [6, 8, 10, 12, 14]
        assert cell.endpoint == pytest.approx(sum(values[1:]) / 5)
        assert cell.endpoint != pytest.approx(values[-1])
        assert cell.endpoint_reason is None

    def test_fewer_than_five_evaluations_is_missing_endpoint_and_never_shrinks(
        self, tmp_path: Path
    ) -> None:
        paths = write_run(
            tmp_path, "a" * 64, _spec((-2.5, -2.0, -1.5, -1.0)), step_bytes=1, tag="c"
        )
        cell = project_h1_cell(load_h_run(*paths))
        assert cell.endpoint is None and cell.window_readings == ()
        assert cell.endpoint_reason is H1ExclusionReason.MISSING_ENDPOINT
        assert len(cell.readings) == 4  # readings are still reported as captured

    def test_an_absent_key_inside_the_window_is_missing_not_skipped(self, tmp_path: Path) -> None:
        values: tuple[float | None, ...] = (-2.5, -2.0, None, -1.5, -1.0)
        cell = project_h1_cell(
            load_h_run(*write_run(tmp_path, "a" * 64, _spec(values), step_bytes=1, tag="c"))
        )
        assert cell.endpoint is None
        assert cell.endpoint_reason is H1ExclusionReason.MISSING_ENDPOINT
        assert cell.readings[2].endpoint_value is None  # absence, never 0.0 or nan

    def test_an_absent_key_before_the_window_does_not_matter(self, tmp_path: Path) -> None:
        values: tuple[float | None, ...] = (None, -2.5, -2.0, -1.5, -1.0, -0.5)
        cell = project_h1_cell(
            load_h_run(*write_run(tmp_path, "a" * 64, _spec(values), step_bytes=1, tag="c"))
        )
        assert cell.endpoint == pytest.approx(-1.5)
        assert cell.endpoint_reason is None

    def test_a_non_finite_reduction_is_nonfinite_endpoint(self, tmp_path: Path) -> None:
        # five finite evaluations whose sum overflows: fsum raises OverflowError
        big = 1.7e308
        spec = CellSpec((big,) * 5, ratios=((0.5,),) * 5, row_bytes=1)
        cell = project_h1_cell(
            load_h_run(*write_run(tmp_path, "a" * 64, spec, step_bytes=1, tag="c"))
        )
        assert cell.endpoint is None
        assert cell.endpoint_reason is H1ExclusionReason.NONFINITE_ENDPOINT

    @pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
    def test_every_non_finite_outcome_of_the_reduction_is_mapped(self, bad: float) -> None:
        readings = tuple(
            H1EvaluationReading(s, s, s, bad if s == 3 else -1.0, 0.5) for s in range(1, 6)
        )
        _, endpoint, reason, _ = h1_observations._endpoint(readings, 5)  # pyright: ignore[reportPrivateUsage]
        assert endpoint is None and reason is H1ExclusionReason.NONFINITE_ENDPOINT

    def test_inf_minus_inf_is_the_value_error_path(self) -> None:
        values = (math.inf, -math.inf, -1.0, -1.0, -1.0)
        readings = tuple(H1EvaluationReading(i + 1, 1, 1, v, 0.5) for i, v in enumerate(values))
        _, endpoint, reason, _ = h1_observations._endpoint(readings, 5)  # pyright: ignore[reportPrivateUsage]
        assert endpoint is None and reason is H1ExclusionReason.NONFINITE_ENDPOINT

    def test_metric_keys_are_selectable_at_invocation(self, tmp_path: Path) -> None:
        run = load_h_run(*write_run(tmp_path, "a" * 64, _spec((-2.5,) * 5), step_bytes=1, tag="c"))
        default = project_h1_cell(run)
        k8 = project_h1_cell(run, **K8_KEYS)
        assert default.endpoint == pytest.approx(-2.5)
        assert k8.endpoint == pytest.approx(-3.0)  # the k = 8 rows carry -3.0, never a fallback
        assert all(r.gate_value == 0.5 for r in k8.readings)

    def test_a_missing_k64_never_falls_back_to_another_k(self, tmp_path: Path) -> None:
        spec = _spec((None,) * 5)
        cell = project_h1_cell(
            load_h_run(*write_run(tmp_path, "a" * 64, spec, step_bytes=1, tag="c"))
        )
        assert cell.endpoint is None
        assert all(r.endpoint_value is None and r.gate_value is None for r in cell.readings)

    def test_registration_defaults_are_documented_parameters(self) -> None:
        assert (ENDPOINT_METRIC_KEY, GATE_METRIC_KEY) == (
            "eval/log_marginal_per_byte_k64",
            "eval/ess_ratio_median_k64",
        )
        assert (ENDPOINT_WINDOW, ESS_THRESHOLD) == (5, 0.25)

    @pytest.mark.parametrize("window", [0, -1, True, 2.0])
    def test_window_must_be_a_positive_integer(self, tmp_path: Path, window: Any) -> None:
        run = load_h_run(*write_run(tmp_path, "a" * 64, _spec((-2.5,) * 5), step_bytes=1, tag="c"))
        with pytest.raises(ValueError, match="window"):
            project_h1_cell(run, window=window)


# ------------------------------------------------------------------------------ exposure
@pytest.mark.unit
class TestExposure:
    def test_evaluation_exposure_is_the_inclusive_prediction_prefix(self, tmp_path: Path) -> None:
        run = load_h_run(
            *write_run(tmp_path, "a" * 64, _spec((-2.5,) * 5), step_bytes=100, tag="c")
        )
        cell = project_h1_cell(run)
        assert [(r.global_step, r.committed_steps, r.source_bytes) for r in cell.readings] == [
            (4, 2, 200),
            (6, 4, 400),
            (8, 6, 600),
            (10, 8, 800),
            (12, 10, 1000),
        ]  # warm-up rows (7 bytes each) never count

    def test_a_stitched_run_read_through_its_final_pod_alone_is_refused(
        self, tmp_path: Path
    ) -> None:
        run = load_h_run(*write_run(tmp_path, "a" * 64, _spec((-2.5,) * 5), step_bytes=1, tag="c"))
        # a stitched resumed run holds more prediction entries than its final pod's file
        short = write_run(tmp_path, "a" * 64, _spec((-2.5,) * 4), step_bytes=1, tag="d")
        stitched = replace(run, metrics_path=short[0])
        with pytest.raises(ValueError, match="differ from the run's training entries"):
            project_h1_cell(stitched)

    def test_a_non_committed_prediction_row_without_bytes_fails_loud(self, tmp_path: Path) -> None:
        metrics, log = write_run(tmp_path, "a" * 64, _spec((-2.5,) * 5), step_bytes=1, tag="c")
        run = load_h_run(metrics, log)
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        target = next(r for r in rows if r.get("phase") == "prediction")
        target["source_bytes"] = None
        metrics.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        with pytest.raises(ValueError, match="source_bytes"):
            project_h1_cell(run)  # never caught into 0 or a partial sum

    def test_h1_exposure_bytes_is_consumed_not_reimplemented(self) -> None:
        tree = ast.parse(Path(h1_observations.__file__).read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert "h1_exposure_bytes" in names
        # no second status/phase filter: the module never compares a row's source_bytes
        assert "source_bytes" not in {
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }

    def test_exposure_unequal_inside_a_block_raises(self, tmp_path: Path) -> None:
        specs = list(standard_specs(0))
        capture = build_block(tmp_path, 101, specs)
        member = capture.block.profiles[1]
        # rewrite one cell with different canonical bytes per step: an execution defect
        other = write_run(
            tmp_path / "other", member.identity, specs[1], step_bytes=101, tag=member.name
        )
        cells = {**capture.cells, member.name: load_h_run(*other)}
        with pytest.raises(ValueError, match="canonical exposure bytes"):
            observe_h1_blocks([replace(capture, cells=cells)])

    def test_a_pre_window_byte_mismatch_inside_a_block_raises(self, tmp_path: Path) -> None:
        # six evaluations: cell 2 carries 101 then 99 canonical bytes at prediction steps 2 and 3,
        # so its exposure differs at the first evaluation (global 4, before the window opens at 6)
        # and re-converges from global 6; the window slice alone cannot see it
        specs = tuple(_spec([BASE[c] + j / 16 for j in range(6)]) for c in range(4))
        capture = build_block(tmp_path, 101, specs)
        member = capture.block.profiles[2]
        metrics, log = write_run(
            tmp_path / "other", member.identity, specs[2], step_bytes=100, tag=member.name
        )

        def shift_bytes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            for row in rows:
                if row.get("schema") == "h-training-step-v2" and row.get("phase") == "prediction":
                    if row["step"] == 2:
                        row["source_bytes"] = 101
                    if row["step"] == 3:
                        row["source_bytes"] = 99
            return rows

        _edit_metrics(metrics, shift_bytes)
        cells = {**capture.cells, member.name: load_h_run(metrics, log)}
        with pytest.raises(ValueError, match="canonical exposure bytes"):
            observe_h1_blocks([replace(capture, cells=cells)])

    def test_a_pre_window_committed_count_mismatch_inside_a_block_raises(
        self, tmp_path: Path
    ) -> None:
        # six evaluations, cell 2 never commits prediction step 1 (global 3). Committed counts are
        # cumulative, so the mismatch persists into the window; this pins the full-readings check
        specs = tuple(_spec([BASE[c] + j / 16 for j in range(6)]) for c in range(4))
        capture = build_block(tmp_path, 101, specs)
        member = capture.block.profiles[2]
        metrics, log = write_run(
            tmp_path / "other", member.identity, specs[2], step_bytes=100, tag=member.name
        )

        def uncommit_step_one(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            for row in rows:
                if row.get("schema") == "h-training-step-v2" and row.get("phase") == "prediction":
                    if row["step"] == 1:
                        row["status"] = "started"
            return rows

        _edit_metrics(metrics, uncommit_step_one)
        cells = {**capture.cells, member.name: load_h_run(metrics, log)}
        with pytest.raises(ValueError, match="different committed step counts"):
            observe_h1_blocks([replace(capture, cells=cells)])

    def test_window_evaluation_steps_unequal_inside_a_block_raise(self, tmp_path: Path) -> None:
        specs = list(standard_specs(0))
        specs[2] = _spec([BASE[2] + j / 16 for j in range(6)])  # one more evaluation
        with pytest.raises(ValueError, match="window evaluation steps"):
            observe_h1_blocks([build_block(tmp_path, 101, specs)])

    def test_pairing_exposure_is_committed_prediction_steps_not_bytes(self, tmp_path: Path) -> None:
        observations = observe_h1_blocks(eight_blocks(tmp_path))
        pilots = inputs_for(observations).contrasts
        assert {p.left_exposure for c in pilots for p in c.pairs} == {10}
        assert {p.right_exposure for c in pilots for p in c.pairs} == {10}
        # canonical bytes are reported per block as a diagnostic and DIFFER across blocks
        totals = {o.exposure.window_source_bytes[-1] for o in observations if o.exposure}
        assert len(totals) == 8

    def test_blocks_with_a_different_window_schedule_raise(self, tmp_path: Path) -> None:
        longer = tuple(_spec([BASE[c] + j / 16 for j in range(6)]) for c in range(4))
        with pytest.raises(ValueError, match="committed step counts at the endpoint window"):
            observe_h1_blocks(
                [build_block(tmp_path, 101, standard_specs(0)), build_block(tmp_path, 102, longer)]
            )


# --------------------------------------------------------------------------- the ESS gate
@pytest.mark.unit
class TestEssGate:
    def test_a_median_below_the_threshold_excludes_even_when_the_mean_passes(
        self, tmp_path: Path
    ) -> None:
        specs = list(standard_specs(0))
        straddle = [(0.5, 0.5, 0.5)] * 4 + [(0.10, 0.20, 0.80)]  # median 0.20, mean 0.3667
        specs[1] = CellSpec(specs[1].endpoints, tuple(straddle))
        observations = observe_h1_blocks([build_block(tmp_path, 101, specs)])
        result = inputs_for(observations)
        (exclusion,) = result.exclusions
        assert exclusion.reasons == (H1ExclusionReason.INADMISSIBLE_ESS,)
        assert exclusion.cells == (LABELS[1],)
        assert result.admitted_seeds == ()

    def test_the_gate_compares_greater_or_equal(self, tmp_path: Path) -> None:
        specs = tuple(CellSpec(s.endpoints, ((0.25, 0.25, 0.25),) * 5) for s in standard_specs(0))
        observations = observe_h1_blocks([build_block(tmp_path, 101, specs)])
        assert inputs_for(observations).admitted_seeds == (101,)

    @pytest.mark.parametrize("position", [0, 2, 4])
    def test_every_evaluation_of_the_window_is_gated_on_its_own(
        self, tmp_path: Path, position: int
    ) -> None:
        specs = list(standard_specs(0))
        weak = [(0.9, 0.9, 0.9)] * 5  # the average of the five stays at 0.74
        weak[position] = (0.1, 0.1, 0.1)
        specs[3] = CellSpec(specs[3].endpoints, tuple(weak))
        observations = observe_h1_blocks([build_block(tmp_path, 101, specs)])
        assert inputs_for(observations).exclusions[0].cells == (LABELS[3],)

    def test_an_evaluation_before_the_window_is_not_gated(self, tmp_path: Path) -> None:
        values = [-9.0] + [BASE[0] + j / 16 for j in range(5)]
        weak = ((0.02, 0.02, 0.02),) + ((0.5, 0.5, 0.5),) * 5
        specs = [CellSpec(tuple(values), weak)] + [
            _spec([BASE[c] + j / 16 for j in range(6)]) for c in range(1, 4)
        ]
        observations = observe_h1_blocks([build_block(tmp_path, 101, specs)])
        assert inputs_for(observations).admitted_seeds == (101,)

    def test_an_absent_gate_key_never_passes(self, tmp_path: Path) -> None:
        observations = observe_h1_blocks(
            [build_block(tmp_path, 101, standard_specs(0))], gate_key="eval/ess_ratio_median_k2"
        )
        (exclusion,) = inputs_for(observations).exclusions
        assert exclusion.reasons == (H1ExclusionReason.INADMISSIBLE_ESS,)
        assert exclusion.cells == LABELS

    def test_the_gate_key_is_selectable_at_invocation(self, tmp_path: Path) -> None:
        # k = 8 ratios are 0.5 everywhere: the k = 8 gate admits what a k = 64 gate refuses
        specs = tuple(CellSpec(s.endpoints, ((0.02, 0.02, 0.02),) * 5) for s in standard_specs(0))
        capture = build_block(tmp_path, 101, specs)
        assert inputs_for(observe_h1_blocks([capture])).admitted_seeds == ()
        assert inputs_for(observe_h1_blocks([capture], **K8_KEYS)).admitted_seeds == (101,)

    def test_every_exclusion_names_the_scope_and_the_trigger(self, tmp_path: Path) -> None:
        specs = tuple(CellSpec(s.endpoints, ((0.02, 0.02, 0.02),) * 5) for s in standard_specs(0))
        (exclusion,) = inputs_for(observe_h1_blocks([build_block(tmp_path, 101, specs)])).exclusions
        assert "every evaluation of the endpoint window" in exclusion.findings[0].detail
        assert exclusion.pairing_id and exclusion.seed == 101

    @pytest.mark.parametrize("threshold", [0, 0.0, -0.1, 1.5, math.nan, True, "0.25"])
    def test_a_threshold_outside_the_ratio_range_is_refused(
        self, tmp_path: Path, threshold: Any
    ) -> None:
        observations = observe_h1_blocks([build_block(tmp_path, 101, standard_specs(0))])
        with pytest.raises(ValueError, match="threshold"):
            inputs_for(observations, ess_threshold=threshold)

    def test_the_default_threshold_excludes_a_block_below_it(self, tmp_path: Path) -> None:
        specs = list(standard_specs(0))
        specs[0] = CellSpec(specs[0].endpoints, ((0.10, 0.10, 0.10),) * 5)
        observations = observe_h1_blocks([build_block(tmp_path, 101, specs)])
        assert inputs_for(observations).admitted_seeds == ()
        assert inputs_for(observations, ess_threshold=0.1).admitted_seeds == (101,)


# ------------------------------------------------------------------ whole-block exclusion
@pytest.mark.unit
class TestGapRows:
    """A ``zero_importance_weight_mass`` gap row at the selected k inside the endpoint window
    would make the window reduction a mean over the finite rows only. D1 parity: such a gap
    makes the endpoint unavailable, never a silent partial mean."""

    def test_a_gap_row_in_the_window_is_missing_endpoint_not_a_mean_of_the_finite_rows(
        self, tmp_path: Path
    ) -> None:
        spec = replace(_spec((-2.5,) * 5), gaps=((4, 64),))  # the last window evaluation
        cell = project_h1_cell(
            load_h_run(*write_run(tmp_path, "a" * 64, spec, step_bytes=1, tag="c"))
        )
        assert cell.endpoint is None
        assert cell.endpoint_reason is H1ExclusionReason.MISSING_ENDPOINT
        assert "step 12" in cell.endpoint_detail and "k=64" in cell.endpoint_detail

    def test_a_gap_row_before_the_window_is_not_gated(self, tmp_path: Path) -> None:
        values = (-9.0, -2.5, -2.25, -2.0, -1.75, -1.0)
        spec = replace(_spec(values), gaps=((0, 64),))  # the first, outside the window
        cell = project_h1_cell(
            load_h_run(*write_run(tmp_path, "a" * 64, spec, step_bytes=1, tag="c"))
        )
        assert cell.endpoint_reason is None
        assert cell.endpoint == pytest.approx(sum(values[1:]) / 5)

    def test_a_gap_at_k64_is_not_gated_when_the_k8_keys_are_selected(self, tmp_path: Path) -> None:
        spec = replace(_spec((-2.5,) * 5), gaps=((4, 64),))
        cell = project_h1_cell(
            load_h_run(*write_run(tmp_path, "a" * 64, spec, step_bytes=1, tag="c")), **K8_KEYS
        )
        assert cell.endpoint == pytest.approx(-3.0)
        assert cell.endpoint_reason is None

    def test_a_gap_at_the_selected_k8_in_the_window_is_missing_endpoint(
        self, tmp_path: Path
    ) -> None:
        spec = replace(_spec((-2.5,) * 5), gaps=((2, 8),))
        cell = project_h1_cell(
            load_h_run(*write_run(tmp_path, "a" * 64, spec, step_bytes=1, tag="c")), **K8_KEYS
        )
        assert cell.endpoint is None
        assert cell.endpoint_reason is H1ExclusionReason.MISSING_ENDPOINT

    def test_a_gap_row_in_the_window_excludes_the_whole_block(self, tmp_path: Path) -> None:
        specs = list(standard_specs(0))
        specs[1] = replace(specs[1], gaps=((3, 64),))
        observations = observe_h1_blocks([build_block(tmp_path, 101, specs)])
        (exclusion,) = inputs_for(observations).exclusions
        assert exclusion.reasons == (H1ExclusionReason.MISSING_ENDPOINT,)
        assert exclusion.cells == (LABELS[1],)


@pytest.mark.unit
class TestPinnedGuards:
    """Guard-teeth pins: each test goes RED when the named guard is removed or inverted."""

    def test_committed_count_mismatch_with_equal_bytes_raises(self, tmp_path: Path) -> None:
        specs = list(standard_specs(0))
        capture = build_block(tmp_path, 101, specs)
        member = capture.block.profiles[1]
        metrics, log = write_run(
            tmp_path / "other", member.identity, specs[1], step_bytes=100, tag=member.name
        )

        def skip_step_one(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            # step 1 is not committed; step 2 carries the bytes it would have, so the
            # canonical totals agree at every window evaluation while the counts do not
            for row in rows:
                if row.get("schema") == "h-training-step-v2" and row.get("phase") == "prediction":
                    if row["step"] == 1:
                        row["status"] = "started"
                    if row["step"] == 2:
                        row["source_bytes"] = 200
            return rows

        _edit_metrics(metrics, skip_step_one)
        cells = {**capture.cells, member.name: load_h_run(metrics, log)}
        with pytest.raises(ValueError, match="different committed step counts"):
            observe_h1_blocks([replace(capture, cells=cells)])

    def test_one_short_cell_is_missing_endpoint_and_never_reaches_the_exposure_check(
        self, tmp_path: Path
    ) -> None:
        specs = list(standard_specs(0))
        specs[2] = _spec([BASE[2] + j / 16 for j in range(4)])
        observations = observe_h1_blocks([build_block(tmp_path, 101, specs)])
        (observation,) = observations
        assert observation.exposure is None
        (exclusion,) = inputs_for(observations).exclusions
        assert exclusion.reasons == (H1ExclusionReason.MISSING_ENDPOINT,)
        assert exclusion.cells == (LABELS[2],)

    def test_a_resumed_lineage_cell_enters_the_block_path_as_its_global_run(
        self, tmp_path: Path
    ) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0))
        victim = capture.cells[LABELS[0]]
        assert type(victim) is HRun
        wrapped = {**capture.cells, LABELS[0]: HResumedRun(run=victim, segments=(victim,))}
        plain = inputs_for(observe_h1_blocks([capture]))
        resumed = inputs_for(observe_h1_blocks([replace(capture, cells=wrapped)]))
        assert resumed == plain

    def test_every_evaluation_is_a_run_evaluation_not_the_window_only(self, tmp_path: Path) -> None:
        specs = tuple(_spec([BASE[c] + j / 16 for j in range(6)]) for c in range(4))
        result = inputs_for(observe_h1_blocks([build_block(tmp_path, 101, specs)]))
        assert [e.step for e in result.runs[0].evals] == [4, 6, 8, 10, 12, 14]

    def test_runs_label_every_evaluation_with_the_selected_endpoint_key(
        self, tmp_path: Path
    ) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0))
        result = inputs_for(observe_h1_blocks([capture], **K8_KEYS))
        assert result.runs
        for run in result.runs:
            for evaluation in run.evals:
                assert {key for key, _ in evaluation.metrics} == {K8_KEYS["endpoint_key"]}

    def test_observation_and_admission_order_is_seed_order_whatever_the_input_order(
        self, tmp_path: Path
    ) -> None:
        blocks = eight_blocks(tmp_path)
        for position in (2, 5):  # seeds 103 and 106 fail the gate
            specs = list(standard_specs(position))
            specs[0] = CellSpec(specs[0].endpoints, ((0.05, 0.05, 0.05),) * 5)
            blocks[position] = build_block(
                tmp_path / f"low-{position}",
                101 + position,
                specs,
                step_bytes=100 + 10 * position,
            )
        result = inputs_for(observe_h1_blocks(list(reversed(blocks))))
        admitted = (101, 102, 104, 105, 107, 108)
        assert result.admitted_seeds == admitted
        assert [e.seed for e in result.exclusions] == [103, 106]
        assert [r.seed for r in result.runs] == [s for s in admitted for _ in range(4)]

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_text_is_refused_at_every_text_boundary(self, tmp_path: Path, blank: str) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0))
        observations = observe_h1_blocks([capture])
        with pytest.raises(ValueError, match="nonblank"):
            pilot_inputs(observations, family_digest=blank, units=UNITS, expected_exposure=10)
        with pytest.raises(ValueError, match="nonblank"):
            pilot_inputs(observations, family_digest=FAMILY, units=blank, expected_exposure=10)
        with pytest.raises(ValueError, match="nonblank"):
            observe_h1_blocks([capture], endpoint_key=blank)
        with pytest.raises(ValueError, match="nonblank"):
            observe_h1_blocks([capture], gate_key=blank)

    def test_a_non_capture_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="H1BlockCapture"):
            observe_h1_blocks(["not a capture"])  # type: ignore[list-item]

    def test_a_capture_without_an_exact_block_is_a_type_error(self) -> None:
        with pytest.raises(TypeError, match="H1PreservationBlock"):
            observe_h1_blocks([H1BlockCapture("not a block", {})])  # type: ignore[arg-type]

    def test_a_cell_that_is_not_a_captured_run_is_a_type_error(self, tmp_path: Path) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0))
        cells = {**capture.cells, LABELS[0]: "not a run"}
        with pytest.raises(TypeError, match="captured HRun or HResumedRun"):
            observe_h1_blocks([replace(capture, cells=cells)])  # type: ignore[dict-item]

    def test_projection_refuses_a_non_run(self) -> None:
        with pytest.raises(TypeError, match="captured HRun or HResumedRun"):
            project_h1_cell("not a run")  # type: ignore[arg-type]

    def test_an_arm_label_bound_to_two_constructions_across_blocks_raises(
        self, tmp_path: Path
    ) -> None:
        # the digest ignores names, so swapping two arm names keeps every per-axis digest
        # consistent: only the per-label check sees the change
        first = build_block(tmp_path, 101, standard_specs(0))
        profiles = h1_profiles(102)
        swapped = (
            replace(profiles[0], name=profiles[1].name),
            replace(profiles[1], name=profiles[0].name),
            *profiles[2:],
        )
        second = build_block(tmp_path, 102, standard_specs(1), profiles=swapped)
        with pytest.raises(ValueError, match="construction digest"):
            observe_h1_blocks([first, second])

    def test_a_non_committed_prediction_row_contributes_nothing(self, tmp_path: Path) -> None:
        identity = "a" * 64
        metrics, log = write_run(tmp_path, identity, _spec((-2.5,) * 5), step_bytes=100, tag="c")
        run = load_h_run(metrics, log)
        baseline = project_h1_cell(run)
        started = {**_step_row(identity, "prediction", 11, 100), "status": "started"}
        _edit_metrics(metrics, _before_run_end(started))
        assert project_h1_cell(run).readings == baseline.readings

    def test_an_unrecognized_status_after_the_last_evaluation_fails_loud(
        self, tmp_path: Path
    ) -> None:
        identity = "a" * 64
        metrics, log = write_run(tmp_path, identity, _spec((-2.5,) * 5), step_bytes=100, tag="c")
        run = load_h_run(metrics, log)
        bogus = {**_step_row(identity, "prediction", 11, 100), "status": "bogus"}
        _edit_metrics(metrics, _before_run_end(bogus))
        with pytest.raises(ValueError, match="unrecognized phase or status"):
            project_h1_cell(run)

    @pytest.mark.parametrize("bad", [0, -1, "3", True, 2.0])
    def test_a_step_row_needs_a_positive_integer_step(self, tmp_path: Path, bad: Any) -> None:
        metrics, log = write_run(tmp_path, "a" * 64, _spec((-2.5,) * 5), step_bytes=100, tag="c")
        run = load_h_run(metrics, log)

        def corrupt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            row = next(
                r
                for r in rows
                if r.get("schema") == "h-training-step-v2" and r.get("phase") == "prediction"
            )
            row["step"] = bad
            return rows

        _edit_metrics(metrics, corrupt)
        with pytest.raises(ValueError, match="positive int step"):
            project_h1_cell(run)

    def test_a_bool_metric_value_is_refused_not_read_as_a_number(self, tmp_path: Path) -> None:
        run = load_h_run(*write_run(tmp_path, "a" * 64, _spec((-2.5,) * 5), step_bytes=1, tag="c"))
        step, elm = run.elm[0]
        forged = {**elm, "metrics": {**elm["metrics"], ENDPOINT_METRIC_KEY: True}}
        with pytest.raises(ValueError, match="not a number"):
            project_h1_cell(replace(run, elm=((step, forged), *run.elm[1:])))

    def test_a_nan_gate_value_is_inadmissible_not_passing(self, tmp_path: Path) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0))
        victim = capture.cells[LABELS[0]]
        assert type(victim) is HRun
        step, elm = victim.elm[2]
        forged = {**elm, "metrics": {**elm["metrics"], GATE_METRIC_KEY: math.nan}}
        forged_run = replace(victim, elm=(*victim.elm[:2], (step, forged), *victim.elm[3:]))
        cells = {**capture.cells, LABELS[0]: forged_run}
        (exclusion,) = inputs_for(observe_h1_blocks([replace(capture, cells=cells)])).exclusions
        assert exclusion.reasons == (H1ExclusionReason.INADMISSIBLE_ESS,)
        assert exclusion.cells == (LABELS[0],)


@pytest.mark.unit
class TestMutationSurvivorPins:
    """Pins for guards that survived a mutation sweep of the adapter with no failing test."""

    @pytest.mark.parametrize(
        ("keys", "gap_k"),
        [
            ({"endpoint_key": ENDPOINT_METRIC_KEY, "gate_key": K8_KEYS["gate_key"]}, 8),
            ({"endpoint_key": ENDPOINT_METRIC_KEY, "gate_key": K8_KEYS["gate_key"]}, 64),
            ({"endpoint_key": K8_KEYS["endpoint_key"], "gate_key": GATE_METRIC_KEY}, 8),
            ({"endpoint_key": K8_KEYS["endpoint_key"], "gate_key": GATE_METRIC_KEY}, 64),
        ],
    )
    def test_a_gap_at_either_selected_k_is_missing_endpoint_under_mixed_keys(
        self, tmp_path: Path, keys: dict[str, str], gap_k: int
    ) -> None:
        # the endpoint key and the gate key name different k: a gap at EITHER k in the last
        # window evaluation must still refuse the endpoint (DH-32 mixed-key invocation)
        spec = replace(_spec((-2.5,) * 5), gaps=((4, gap_k),))
        cell = project_h1_cell(
            load_h_run(*write_run(tmp_path, "a" * 64, spec, step_bytes=1, tag="c")), **keys
        )
        assert cell.endpoint is None
        assert cell.endpoint_reason is H1ExclusionReason.MISSING_ENDPOINT

    @pytest.mark.parametrize(
        "key",
        ["eval/log_marginal_per_byte", "eval/x_k0", "eval/x_k-3", "eval/x_k08", "eval/x_k", "_k8"],
    )
    def test_a_metric_key_without_a_selected_k_is_refused(self, tmp_path: Path, key: str) -> None:
        run = load_h_run(*write_run(tmp_path, "a" * 64, _spec((-2.5,) * 5), step_bytes=1, tag="c"))
        with pytest.raises(ValueError, match="names no k"):
            project_h1_cell(run, endpoint_key=key)

    def test_a_confirmatory_block_keeps_its_class_and_the_planner_refuses_it(
        self, tmp_path: Path
    ) -> None:
        # seed_sizing refuses non-pilot data only if the adapter carries the captured class
        blocks = [
            build_block(
                tmp_path,
                101 + i,
                standard_specs(i),
                step_bytes=100 + 10 * i,
                profiles=tuple(
                    replace(member, run_class="confirmatory") for member in h1_profiles(101 + i)
                ),
                run_class="confirmatory",
            )
            for i in range(8)
        ]
        result = inputs_for(observe_h1_blocks(blocks))
        assert {run.run_class for run in result.runs} == {"confirmatory"}
        assert {c.run_class for c in result.contrasts} == {"confirmatory"}
        with pytest.raises(ValueError, match="pilot"):
            plan_seed_count(result.contrasts, hypotheses(), max_seeds=500)

    def test_pilot_inputs_order_by_seed_whatever_the_observation_order(
        self, tmp_path: Path
    ) -> None:
        observations = observe_h1_blocks(eight_blocks(tmp_path))
        forward = inputs_for(observations)
        backward = inputs_for(tuple(reversed(observations)))
        assert backward == forward
        assert backward.admitted_seeds == tuple(range(101, 109))
        assert [p.seed for p in backward.contrasts[0].pairs] == list(range(101, 109))

    def test_exclusions_are_listed_in_seed_order_whatever_the_observation_order(
        self, tmp_path: Path
    ) -> None:
        blocks = eight_blocks(tmp_path)
        observations = observe_h1_blocks(blocks)
        flagged = tuple(
            replace(
                item, findings=(H1ExclusionFinding(H1ExclusionReason.INDEPENDENT_WARMUP, "x", ""),)
            )
            if item.seed in (103, 106)
            else item
            for item in observations
        )
        result = inputs_for(tuple(reversed(flagged)))
        assert [e.seed for e in result.exclusions] == [103, 106]
        assert result.admitted_seeds == (101, 102, 104, 105, 107, 108)

    def test_admitted_blocks_of_different_captured_classes_are_refused(
        self, tmp_path: Path
    ) -> None:
        # unreachable through observe_h1_blocks (the per-label construction digest differs
        # first); this guards hand-built observations
        observations = observe_h1_blocks(eight_blocks(tmp_path))
        first = observations[0]
        cells = tuple(
            replace(cell, run_class="confirmatory") if cell is not None else None
            for cell in first.cells
        )
        with pytest.raises(ValueError, match="different captured run classes"):
            inputs_for((replace(first, cells=cells), *observations[1:]))


@pytest.mark.unit
class TestWholeBlockExclusion:
    def test_a_bad_cell_drops_the_whole_block_from_all_three_contrasts(
        self, tmp_path: Path
    ) -> None:
        blocks = eight_blocks(tmp_path)
        bad = list(standard_specs(3))
        bad[2] = CellSpec(bad[2].endpoints, ((0.05, 0.05, 0.05),) * 5)
        blocks[3] = build_block(tmp_path / "bad", 104, bad, step_bytes=130)
        result = inputs_for(observe_h1_blocks(blocks))
        seeds = [101, 102, 103, 105, 106, 107, 108]
        assert result.admitted_seeds == tuple(seeds)
        assert [len(c.pairs) for c in result.contrasts] == [7, 7, 7]
        assert all([p.seed for p in c.pairs] == seeds for c in result.contrasts)
        assert len(result.runs) == 4 * 7 and 104 not in {r.seed for r in result.runs}
        (exclusion,) = result.exclusions
        assert exclusion.seed == 104 and exclusion.cells == (LABELS[2],)
        assert exclusion.pairing_id == blocks[3].block.identity

    def test_one_pairing_id_per_seed(self, tmp_path: Path) -> None:
        observations = observe_h1_blocks(eight_blocks(tmp_path))
        assert len({o.pairing_id for o in observations}) == 8
        assert [o.seed for o in observations] == list(range(101, 109))

    def test_independent_warmup_maps_to_exclusion(self, tmp_path: Path) -> None:
        capture = build_block(
            tmp_path, 101, standard_specs(0), independent_warmup=(LABELS[1], LABELS[3])
        )
        (exclusion,) = inputs_for(observe_h1_blocks([capture])).exclusions
        assert exclusion.reasons == (H1ExclusionReason.INDEPENDENT_WARMUP,)
        assert exclusion.cells == (LABELS[1], LABELS[3])
        assert all(type(f.reason) is H1ExclusionReason for f in exclusion.findings)

    def test_a_warmup_flag_on_a_foreign_cell_is_refused(self, tmp_path: Path) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0), independent_warmup=("h1-nope",))
        with pytest.raises(ValueError, match="outside the block"):
            observe_h1_blocks([capture])

    def test_a_missing_endpoint_excludes_the_block(self, tmp_path: Path) -> None:
        specs = list(standard_specs(0))
        specs[0] = _spec([BASE[0] + j / 16 for j in range(4)])
        specs[1:] = [_spec([BASE[c] + j / 16 for j in range(4)]) for c in range(1, 4)]
        (exclusion,) = inputs_for(observe_h1_blocks([build_block(tmp_path, 101, specs)])).exclusions
        assert exclusion.reasons == (H1ExclusionReason.MISSING_ENDPOINT,)
        assert exclusion.cells == LABELS

    def test_a_non_finite_endpoint_excludes_the_block(self, tmp_path: Path) -> None:
        specs = list(standard_specs(0))
        specs[2] = CellSpec((1.7e308,) * 5, ((0.5,),) * 5, row_bytes=1)
        (exclusion,) = inputs_for(observe_h1_blocks([build_block(tmp_path, 101, specs)])).exclusions
        assert exclusion.reasons == (H1ExclusionReason.NONFINITE_ENDPOINT,)
        assert exclusion.cells == (LABELS[2],)

    def test_a_reader_refusal_excludes_the_block_and_carries_its_text(self, tmp_path: Path) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0))
        victim = capture.cells[LABELS[0]]
        assert type(victim) is HRun
        victim.metrics_path.write_text("this is not json\n", encoding="utf-8")
        (exclusion,) = inputs_for(observe_h1_blocks([capture])).exclusions
        assert exclusion.reasons == (H1ExclusionReason.READER_REFUSAL,)
        assert str(victim.metrics_path) in exclusion.findings[0].detail
        assert exclusion.cells == (LABELS[0],)

    def test_a_non_reader_failure_is_not_swallowed_as_a_reason(self, tmp_path: Path) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0))
        victim = capture.cells[LABELS[0]]
        assert type(victim) is HRun
        short = write_run(
            tmp_path / "x", victim.profile_identity, _spec((-2.5,) * 4), step_bytes=1, tag="x"
        )
        cells = {**capture.cells, LABELS[0]: replace(victim, metrics_path=short[0])}
        with pytest.raises(ValueError, match="training entries"):
            observe_h1_blocks([replace(capture, cells=cells)])

    def test_findings_refuse_an_unregistered_reason(self) -> None:
        with pytest.raises(TypeError):
            h1_observations.H1ExclusionFinding("inadmissible_ess", "cell", "x")  # type: ignore[arg-type]


# ------------------------------------------------------------------------ binding defects
@pytest.mark.unit
class TestBinding:
    def test_a_profile_identity_that_differs_from_the_capture_raises(self, tmp_path: Path) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0))
        other = build_block(tmp_path / "o", 102, standard_specs(0))
        cells = {**capture.cells, LABELS[0]: other.cells[LABELS[0]]}
        with pytest.raises(ValueError, match="profile identity"):
            observe_h1_blocks([replace(capture, cells=cells)])

    def test_a_captured_class_that_is_not_the_profiles_raises(self, tmp_path: Path) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0), run_class="remote_tiny")
        with pytest.raises(ValueError, match="run_class"):
            observe_h1_blocks([capture])

    def test_cells_must_equal_the_blocks_arms(self, tmp_path: Path) -> None:
        capture = build_block(tmp_path, 101, standard_specs(0))
        cells = dict(capture.cells)
        del cells[LABELS[3]]
        with pytest.raises(ValueError, match="must equal the block's arms"):
            observe_h1_blocks([replace(capture, cells=cells)])

    def test_a_duplicate_seed_raises(self, tmp_path: Path) -> None:
        first = build_block(tmp_path / "a", 101, standard_specs(0))
        second = build_block(tmp_path / "b", 101, standard_specs(1))
        with pytest.raises(ValueError, match="duplicate seed"):
            observe_h1_blocks([first, second])

    def test_a_construction_that_changes_between_blocks_raises(self, tmp_path: Path) -> None:
        first = build_block(tmp_path, 101, standard_specs(0))
        changed = build_block(
            tmp_path, 102, standard_specs(1), profiles=h1_profiles(102, encoder_layers=2)
        )
        with pytest.raises(ValueError, match="construction digest"):
            observe_h1_blocks([first, changed])

    def test_a_construction_change_under_renamed_cells_raises(self, tmp_path: Path) -> None:
        # every cell name is seen once across the two blocks, so only the (sentence, spine)
        # axis key can see the change: RED with the by_axis guard removed
        def renamed(seed: int, **kwargs: Any) -> tuple[HProfile, ...]:
            return tuple(
                replace(member, name=f"{member.name}-renamed-{seed}")
                for member in h1_profiles(seed, **kwargs)
            )

        first = build_block(tmp_path, 101, standard_specs(0), profiles=renamed(101))
        changed = build_block(
            tmp_path, 102, standard_specs(1), profiles=renamed(102, encoder_layers=2)
        )
        with pytest.raises(ValueError, match="construction digest"):
            observe_h1_blocks([first, changed])

    def test_no_blocks_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one block"):
            observe_h1_blocks([])


# ------------------------------------------------------------------------ the contrasts
@pytest.mark.unit
class TestContrasts:
    def test_contrast_values_match_hand_computation_and_signs(self, tmp_path: Path) -> None:
        # mm=-2.5, sm=-2.0, ms=-2.25, ss=-1.5 with a flat window (every evaluation equal)
        specs = [_spec([value] * 5) for value in BASE]
        result = inputs_for(observe_h1_blocks([build_block(tmp_path, 101, specs)]))
        by_name = {c.contrast_id: c.pairs[0].left for c in result.contrasts}
        assert tuple(by_name) == CONTRASTS
        assert by_name["sentence_normalization"] == pytest.approx(0.625)  # +0.5(sm+ss-mm-ms)
        assert by_name["spine_normalization"] == pytest.approx(0.375)  # +0.5(ms+ss-mm-sm)
        assert by_name["interaction"] == pytest.approx(0.25)  # mm-sm-ms+ss
        assert all(c.pairs[0].right == 0.0 for c in result.contrasts)

    def test_endpoints_enter_exactly_as_captured_with_no_negation(self, tmp_path: Path) -> None:
        run = inputs_for(observe_h1_blocks([build_block(tmp_path, 101, standard_specs(0))])).runs[0]
        assert [dict(e.metrics)[ENDPOINT_METRIC_KEY] for e in run.evals] == pytest.approx(
            [BASE[0] + j / 16 + (((0 * 7 + 0) % 8 - 4) / 32) for j in range(5)]
        )
        assert all(v < 0 for e in run.evals for _, v in e.metrics)

    def test_pairs_are_ordered_by_seed_whatever_the_input_order(self, tmp_path: Path) -> None:
        blocks = eight_blocks(tmp_path)
        forward = inputs_for(observe_h1_blocks(blocks))
        backward = inputs_for(observe_h1_blocks(list(reversed(blocks))))
        assert forward.contrasts == backward.contrasts
        assert [p.seed for p in forward.contrasts[0].pairs] == list(range(101, 109))

    def test_runs_carry_the_declared_identity_fields(self, tmp_path: Path) -> None:
        blocks = eight_blocks(tmp_path)
        result = inputs_for(observe_h1_blocks(blocks))
        assert len(result.runs) == 32
        first = result.runs[0]
        member = blocks[0].block.profiles[0]
        assert (first.arm, first.seed, first.run_class) == (
            member.name,
            101,
            "pilot",
        )
        assert first.family_digest == FAMILY and first.data_order is None
        assert first.construction_digest == h1_construction_digest(member)
        assert [(e.step, e.source_bytes) for e in first.evals] == [
            (4, 200),
            (6, 400),
            (8, 600),
            (10, 800),
            (12, 1000),
        ]
        by_arm: dict[str, set[str]] = {}
        for run in result.runs:
            by_arm.setdefault(run.arm, set()).add(run.construction_digest)
        assert all(len(digests) == 1 for digests in by_arm.values())  # invariant across blocks
        assert len({next(iter(d)) for d in by_arm.values()}) == 4  # distinct within a block

    def test_runs_arm_is_the_profile_name_the_pilot_generator_gives(self, tmp_path: Path) -> None:
        # the generator names an arm h1-{sentence}-{spine}-pilot: it recurs across the eight
        # seed blocks and the seed lives in the profile only, so the profile name is the arm
        blocks = eight_blocks(tmp_path)
        result = inputs_for(observe_h1_blocks(blocks))
        assert len(result.runs) == 32
        assert {run.arm for run in result.runs} == set(LABELS)
        for arm in LABELS:
            seeds = sorted(run.seed for run in result.runs if run.arm == arm)
            assert seeds == list(range(101, 109))
        names = {profile.name for capture in blocks for profile in capture.block.profiles}
        assert names == set(LABELS)

    def test_the_contrast_carries_the_captured_class_and_the_declared_metadata(
        self, tmp_path: Path
    ) -> None:
        result = inputs_for(observe_h1_blocks(eight_blocks(tmp_path)))
        for contrast in result.contrasts:
            assert (contrast.units, contrast.run_class) == (UNITS, "pilot")
            assert (contrast.regime_id, contrast.expected_exposure) == (FAMILY, 10)

    def test_no_admitted_block_gives_an_empty_family_that_the_planner_refuses(
        self, tmp_path: Path
    ) -> None:
        specs = tuple(CellSpec(s.endpoints, ((0.02, 0.02, 0.02),) * 5) for s in standard_specs(0))
        result = inputs_for(observe_h1_blocks([build_block(tmp_path, 101, specs)]))
        assert result.contrasts == () and len(result.exclusions) == 1
        with pytest.raises(ValueError):
            plan_seed_count(result.contrasts, hypotheses(), max_seeds=50)


# ------------------------------------------------------------------------ the sensitivity
@pytest.mark.unit
class TestThresholdSweep:
    THRESHOLDS = (0.10, 0.15, 0.25, 0.35, 0.50)

    def _blocks(self, tmp_path: Path) -> list[H1BlockCapture]:
        medians = (0.12, 0.2, 0.25, 0.4, 0.55, 0.6, 0.7, 0.8)
        out = []
        for i, median in enumerate(medians):
            specs = list(standard_specs(i))
            specs[0] = CellSpec(specs[0].endpoints, ((median,) * 3,) * 5)
            out.append(build_block(tmp_path, 101 + i, specs, step_bytes=100 + 10 * i))
        return out

    def test_admitted_blocks_per_reading_and_exclusions_are_reported(self, tmp_path: Path) -> None:
        observations = observe_h1_blocks(self._blocks(tmp_path))
        sweep = pilot_inputs_by_threshold(
            observations, self.THRESHOLDS, family_digest=FAMILY, units=UNITS, expected_exposure=10
        )
        assert [len(r.admitted_seeds) for r in sweep] == [8, 7, 6, 5, 4]
        assert [r.ess_threshold for r in sweep] == list(self.THRESHOLDS)
        assert [len(r.exclusions) for r in sweep] == [0, 1, 2, 3, 4]
        assert sweep[2].exclusions[0].seed == 101 and sweep[2].exclusions[1].seed == 102

    def test_the_result_at_a_threshold_equals_the_single_threshold_call(
        self, tmp_path: Path
    ) -> None:
        observations = observe_h1_blocks(self._blocks(tmp_path))
        sweep = pilot_inputs_by_threshold(
            observations, self.THRESHOLDS, family_digest=FAMILY, units=UNITS, expected_exposure=10
        )
        for threshold, result in zip(self.THRESHOLDS, sweep, strict=True):
            assert result == inputs_for(observations, ess_threshold=threshold)

    @pytest.mark.parametrize("thresholds", [(), (0.25, 0.25)])
    def test_a_degenerate_threshold_list_is_refused(
        self, tmp_path: Path, thresholds: tuple[float, ...]
    ) -> None:
        observations = observe_h1_blocks([build_block(tmp_path, 101, standard_specs(0))])
        with pytest.raises(ValueError, match="thresholds"):
            pilot_inputs_by_threshold(
                observations, thresholds, family_digest=FAMILY, units=UNITS, expected_exposure=10
            )


# ----------------------------------------------------------------------------- sizing
@pytest.mark.unit
class TestSizing:
    def test_plan_seed_count_plans_n_from_the_synthetic_eight_block_pilot(
        self, tmp_path: Path
    ) -> None:
        result = inputs_for(observe_h1_blocks(eight_blocks(tmp_path)))
        plan = plan_seed_count(result.contrasts, hypotheses(), max_seeds=500)
        assert plan.status == "planned"
        assert plan.required_seeds is not None and plan.required_seeds >= 2
        assert [c.contrast_id for c in plan.contrasts] == list(CONTRASTS)

    def test_expected_pilot_blocks_stays_the_declared_roster_not_the_admitted_count(
        self, tmp_path: Path
    ) -> None:
        blocks = eight_blocks(tmp_path)
        bad = list(standard_specs(7))
        bad[0] = CellSpec(bad[0].endpoints, ((0.05, 0.05, 0.05),) * 5)
        blocks[7] = build_block(tmp_path / "bad", 108, bad, step_bytes=170)
        result = inputs_for(observe_h1_blocks(blocks))
        with pytest.raises(ValueError, match="declared pilot roster size"):
            plan_seed_count(result.contrasts, hypotheses(), max_seeds=500)

    def test_dual_confidence_intervals_consume_the_pairs_unchanged(self, tmp_path: Path) -> None:
        result = inputs_for(observe_h1_blocks(eight_blocks(tmp_path)))
        report = dual_confidence_intervals(
            result.contrasts[0].pairs, family_size=3, sealed_data_id="sealed-1"
        )
        values = [
            0.5 * (standard_endpoint(i, 1) + standard_endpoint(i, 3))
            - 0.5 * (standard_endpoint(i, 0) + standard_endpoint(i, 2))
            for i in range(8)
        ]
        assert report.mean == pytest.approx(sum(values) / 8)
        assert report.df == 7 and report.seed_ids == tuple(range(101, 109))

    def test_zero_variance_is_refused_by_the_planner(self, tmp_path: Path) -> None:
        flat = eight_blocks(tmp_path, noise=False)
        result = inputs_for(observe_h1_blocks(flat))
        with pytest.raises(PilotVarianceUnknown):
            plan_seed_count(result.contrasts, hypotheses(), max_seeds=500)

    def test_pairing_on_canonical_bytes_would_be_refused_for_the_right_reason(
        self, tmp_path: Path
    ) -> None:
        observations = observe_h1_blocks(eight_blocks(tmp_path))
        result = inputs_for(observations)
        by_seed = {o.seed: o.exposure.window_source_bytes[-1] for o in observations if o.exposure}
        rebuilt = tuple(
            PilotContrast(
                c.contrast_id,
                c.units,
                c.run_class,
                c.regime_id,
                by_seed[101],
                tuple(
                    PairedSeed(p.seed, p.left, p.right, by_seed[p.seed], by_seed[p.seed])
                    for p in c.pairs
                ),
            )
            for c in result.contrasts
        )
        with pytest.raises(ValueError, match="seed exposure mismatch"):
            plan_seed_count(rebuilt, hypotheses(), max_seeds=500)

    def test_a_declared_exposure_off_by_one_is_refused_by_the_planner(self, tmp_path: Path) -> None:
        observations = observe_h1_blocks(eight_blocks(tmp_path))
        result = pilot_inputs(observations, family_digest=FAMILY, units=UNITS, expected_exposure=11)
        with pytest.raises(ValueError, match="pilot exposure differs from declared endpoint"):
            plan_seed_count(result.contrasts, hypotheses(), max_seeds=500)


# ------------------------------------------------------------------------- module contract
@pytest.mark.unit
class TestModuleContract:
    SOURCE = Path(h1_observations.__file__)

    def test_the_module_is_pure(self) -> None:
        tree = ast.parse(self.SOURCE.read_text())
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        ]
        assert not any(".composition" in m or ".ladder.run" in m for m in imports)
        assert not any("mlflow" in m or "requests" in m or "boto3" in m for m in imports)
        calls = {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        assert "print" not in calls
        sentinels = [
            n
            for n in tree.body
            if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id == "object"
        ]
        assert sentinels == []

    def test_no_threshold_literal_lives_in_the_source_beyond_the_default(self) -> None:
        tree = ast.parse(self.SOURCE.read_text())
        floats = {
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and type(n.value) is float
        }
        assert floats <= {0.0, 0.25}
        text = self.SOURCE.read_text()
        assert not any(token in text for token in ("0.10", "0.15", "0.35", "0.50"))

    def test_the_step_schema_literal_equals_the_aggregates(self) -> None:
        assert h1_observations._STEP_SCHEMA == h_aggregate._STEP_SCHEMA  # pyright: ignore[reportPrivateUsage]

    def test_the_prelude_is_consumed_not_redefined(self) -> None:
        text = self.SOURCE.read_text()
        for name in (
            "class H1ExclusionReason",
            "def h1_construction_digest",
            "def h1_exposure_bytes",
        ):
            assert name not in text


# ------------------------------------------------------------ integration: real captures
FIXTURE = Path(__file__).resolve().parents[1] / "results" / "h0-final-probe-1k-v4-92c9db"
ARM_A = (
    "5ece5d1b160d593cebfe3a7bbefc0af40f56e59feb6c7db91777c68dd4004880"  # pragma: allowlist secret
)
ARM_B = (
    "3390f577bcb3df123a68d6ba4c51bfc735a39216343ca20fd08c13cca3238b66"  # pragma: allowlist secret
)
# (global evaluation step, committed prediction steps, cumulative canonical bytes), MEASURED
# from the captured rows of all three lineages (the resume pair equals the single-pod arm)
PROBE_READINGS = ((500, 250, 103639), (750, 500, 205158), (1000, 750, 303434))


def _probe(identity: str, suffix: str) -> tuple[Path, Path]:
    if not FIXTURE.is_dir():
        raise AssertionError(
            f"{FIXTURE} is absent: run `dvc pull results/h0-final-probe-1k-v4-92c9db.dvc` "
            "(this test fails rather than skips: a skip is a false green)"
        )
    metrics = FIXTURE / f"h-worker-{identity}-{suffix}-metrics.jsonl"
    first = [json.loads(line) for line in metrics.read_text().splitlines()]
    start = next(r for r in first if r.get("event") == "run_start")
    assert start["profile_identity"] == identity
    return metrics, metrics.with_name(metrics.name.replace("-metrics.jsonl", "-worker-log.jsonl"))


def _lineage() -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    return _probe(ARM_A, "f25744b7"), _probe(ARM_A, "5e76f21c")


@pytest.mark.integration
class TestFinalProbeDiagnostics:
    @pytest.mark.parametrize(("identity", "suffix"), [(ARM_A, "14637e75"), (ARM_B, "38c25d26")])
    def test_a_complete_arm_projects_as_a_k8_diagnostic(self, identity: str, suffix: str) -> None:
        cell = project_h1_cell(load_h_run(*_probe(identity, suffix)), **K8_KEYS)
        assert [(r.global_step, r.committed_steps, r.source_bytes) for r in cell.readings] == list(
            PROBE_READINGS
        )
        assert cell.run_class == "remote_tiny" and cell.warmup_steps == 250
        assert cell.endpoint is None and cell.window_readings == ()
        assert cell.endpoint_reason is H1ExclusionReason.MISSING_ENDPOINT  # 3 < 5, no shrink
        assert all(r.endpoint_value is not None and r.endpoint_value < 0 for r in cell.readings)
        assert all(r.gate_value is not None for r in cell.readings)

    def test_the_default_k64_keys_are_absent_and_never_fall_back_to_k8(self) -> None:
        cell = project_h1_cell(load_h_run(*_probe(ARM_A, "14637e75")))
        assert all(r.endpoint_value is None and r.gate_value is None for r in cell.readings)
        assert cell.endpoint_reason is H1ExclusionReason.MISSING_ENDPOINT

    def test_the_resume_lineage_is_read_from_every_segment(self) -> None:
        resumed = load_h_resumed_run(list(_lineage()))
        cell = project_h1_cell(resumed, **K8_KEYS)
        assert [(r.global_step, r.committed_steps, r.source_bytes) for r in cell.readings] == list(
            PROBE_READINGS
        )
        assert len(cell.segment_metrics) == 2
        single = project_h1_cell(load_h_run(*_probe(ARM_A, "14637e75")), **K8_KEYS)
        assert [r.source_bytes for r in cell.readings] == [r.source_bytes for r in single.readings]

    def test_the_stitched_run_alone_would_undercount_and_is_refused(self) -> None:
        resumed = load_h_resumed_run(list(_lineage()))
        # the stitched HRun's metrics_path is the final pod's alone (199795 bytes at its
        # second evaluation against the lineage's 303434): refused, never undercounted
        with pytest.raises(ValueError, match="differ from the run's training entries"):
            project_h1_cell(resumed.run, **K8_KEYS)

    def test_a_window_that_the_capture_can_form_is_projected_by_parameter(self) -> None:
        run = load_h_run(*_probe(ARM_A, "14637e75"))
        cell = project_h1_cell(run, window=3, **K8_KEYS)
        expected = math.fsum(
            r.endpoint_value for r in cell.readings if r.endpoint_value is not None
        )
        assert cell.endpoint == pytest.approx(expected / 3)
        assert cell.endpoint_reason is None
