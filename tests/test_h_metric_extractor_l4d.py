"""FOR-1704 (L4d): the extractor projects the v2 evaluation instruments.

Hand-built records only (tests/evaluation_v2_fixtures.py): the extractor is not a reader member
and these tests need no engine bytes. Each case is a guard with a named mutation arm.
"""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Any

import pytest
from tests import evaluation_v2_fixtures as fx

from entropic_substrate.ladder.run.h_metric_extractor import (
    EVALUATION_SCHEMAS,
    LATENT_OPERANDS,
    PROPOSALS,
    HTrainingStepExtractor,
    project_evaluation,
)

pytestmark = pytest.mark.unit

# the closed non-k eval keys (C9, amended by A5(b): the latent-usage key carries its operand)
_NON_K = (
    re.compile(r"eval/(prior|q)_log_radius_mean"),
    re.compile(r"eval/prior_q_cosine_mean"),
    re.compile(r"eval/latent_usage_(posterior_mean|draw_average)_nats_per_byte"),
    re.compile(r"eval/(wall|cuda)_seconds"),
)


def _full_example(**over: Any) -> dict[str, Any]:
    """Two rows (2 and 6 canonical bytes), k in {2, 4}, all three proposals."""
    base: dict[str, Any] = {
        "source_bytes": [2, 6],
        "q": {
            2: [fx.row(-4.0, 1.5), fx.row(-12.0, 1.0)],
            4: [fx.row(-4.0, 3.0), fx.row(-12.0, 2.0)],
        },
        "prior": {
            2: [fx.row(-5.0, 2.0), fx.row(-18.0, 2.0)],
            4: [fx.row(-5.0, 4.0), fx.row(-18.0, 4.0)],
        },
        "mixture": {
            2: [fx.row(-4.5, 2.0), fx.row(-13.0, 1.5)],
            4: [fx.row(-4.5, 3.5), fx.row(-13.0, 3.0)],
        },
        "sd": {
            "q": {2: [0.5, 1.5], 4: [0.7, 2.5]},
            "prior": {2: [0.1, 0.3], 4: [0.2, 0.4]},
            "mixture": {2: [0.2, 0.6], 4: [0.3, 0.9]},
        },
        "prior_log_radius": [0.1, 0.3],
        "q_log_radius": [0.2, 0.4],
        "cosine": [0.5, 0.9],
        "usage": ([-3.0, -9.0], [-4.0, -8.0]),
    }
    base.update(over)
    return fx.example(**base)


def _metrics(event: dict[str, Any]) -> dict[str, float]:
    out = HTrainingStepExtractor(warmup_steps=0).extract(event)
    assert out is not None
    return out.metrics


def test_every_new_key_is_emitted_with_its_proposal_segment_and_the_q_keys_stay_bare() -> None:
    m = _metrics(fx.event([_full_example()]))
    for k in (2, 4):
        for name, base in (
            ("q", ""),
            ("prior", "_prior"),
            ("mixture", "_mixture"),
        ):
            for stem in (
                "log_marginal",
                "ess",
                "gap_rows",
                "log_marginal_per_byte_ratio_of_sums",
                "log_weight_sd",
            ):
                assert f"eval/{stem}{base}_k{k}" in m, (stem, name, k)
    for key in (
        "eval/prior_log_radius_mean",
        "eval/q_log_radius_mean",
        "eval/prior_q_cosine_mean",
        "eval/latent_usage_posterior_mean_nats_per_byte",
    ):
        assert key in m, key
    # the bare names are the q values, and the prior values do not overwrite them
    assert m["eval/ess_k2"] == pytest.approx(1.25)
    assert m["eval/ess_prior_k2"] == pytest.approx(2.0)
    assert m["eval/ess_mixture_k2"] == pytest.approx(1.75)
    assert m["eval/log_marginal_prior_k4"] == pytest.approx(-11.5)


def test_the_key_family_is_k_suffixed_except_a_closed_allowlist() -> None:
    """FOR-1664 guard extended to the whole family (C9). A bare ``eval/ess_prior`` fails it."""
    m = _metrics(fx.event([_full_example()], cuda_seconds=1.0))
    for key in m:
        assert key.startswith("eval/"), key
        assert re.search(r"_k[0-9]+$", key) or any(p.fullmatch(key) for p in _NON_K), key
    assert not {"eval/ess_prior", "eval/ess_mixture", "eval/log_marginal_prior"} & set(m)
    assert "eval/latent_usage_nats_per_byte" not in m  # the bare latent-usage key is never emitted


def test_ratio_of_sums_weights_by_bytes_where_the_mean_of_ratios_does_not() -> None:
    m = _metrics(fx.event([_full_example()]))
    # q at k=2: (-4 + -12) / (2 + 6) = -2.0 per byte; the mean of ratios would be (-2 + -2)/2
    # for this pair, so use the prior: (-5 + -18) / 8 = -2.875 versus (-2.5 + -3) / 2 = -2.75
    assert m["eval/log_marginal_per_byte_ratio_of_sums_k2"] == pytest.approx(-2.0)
    assert m["eval/log_marginal_per_byte_ratio_of_sums_prior_k2"] == pytest.approx(-23 / 8)
    assert m["eval/log_marginal_per_byte_ratio_of_sums_prior_k2"] != pytest.approx(-2.75)
    assert m["eval/log_marginal_per_byte_ratio_of_sums_mixture_k4"] == pytest.approx(-17.5 / 8)


def test_a_gap_rows_bytes_and_the_boundary_symbols_stay_out_of_the_denominator() -> None:
    gap = _full_example(
        source_bytes=[2, 6],
        prior={2: [fx.row(-5.0, 2.0), fx.gap_row()], 4: [fx.row(-5.0, 4.0), fx.gap_row()]},
    )
    m = _metrics(fx.event([gap]))
    # only row 0 is finite for the prior: -5 / 2, NOT -5 / (2 + 6) and NOT -5 / (2 + 1)
    assert m["eval/log_marginal_per_byte_ratio_of_sums_prior_k2"] == pytest.approx(-2.5)
    assert m["eval/gap_rows_prior_k2"] == 1.0
    assert m["eval/gap_rows_k2"] == 0.0


def test_the_ratio_key_is_absent_never_zero_when_no_row_of_the_proposal_is_finite() -> None:
    all_gap = _full_example(
        prior={2: [fx.gap_row(), fx.gap_row()], 4: [fx.gap_row(), fx.gap_row()]}
    )
    m = _metrics(fx.event([all_gap]))
    assert "eval/log_marginal_per_byte_ratio_of_sums_prior_k2" not in m
    assert "eval/log_marginal_prior_k2" not in m and "eval/ess_prior_k2" not in m
    assert m["eval/gap_rows_prior_k2"] == 2.0
    assert "eval/log_marginal_per_byte_ratio_of_sums_k2" in m  # q is unaffected


def test_a_v1_record_gains_only_the_q_derivable_ratio_and_never_an_invented_key() -> None:
    example = fx.example(source_bytes=[2, 6], q={2: [fx.row(-4.0, 1.5), fx.row(-12.0, 1.0)]})
    out = _metrics(fx.event([example], schema=fx.V1, operand=None))
    assert set(out) == {
        "eval/log_marginal_k2",
        "eval/ess_k2",
        "eval/gap_rows_k2",
        "eval/log_marginal_per_byte_ratio_of_sums_k2",
        "eval/wall_seconds",
    }
    assert out["eval/log_marginal_k2"] == pytest.approx(-8.0)  # the v1 mean is unchanged


def test_a_sphere_cell_has_no_log_radius_keys_and_a_magnitude_cell_has_both() -> None:
    sphere = _full_example(prior_log_radius=None, q_log_radius=None)
    m = _metrics(fx.event([sphere]))
    assert "eval/prior_log_radius_mean" not in m and "eval/q_log_radius_mean" not in m
    assert "eval/prior_q_cosine_mean" in m
    magnitude = _metrics(fx.event([_full_example()]))
    assert magnitude["eval/prior_log_radius_mean"] == pytest.approx(0.2)
    assert magnitude["eval/q_log_radius_mean"] == pytest.approx(0.3)
    assert magnitude["eval/prior_q_cosine_mean"] == pytest.approx(0.7)


def test_latent_usage_is_signed_never_clamped_and_names_its_operand() -> None:
    helps = _metrics(fx.event([_full_example(usage=([-3.0, -9.0], [-4.0, -12.0]))]))
    # (1 + 3) nats over 8 bytes
    assert helps["eval/latent_usage_posterior_mean_nats_per_byte"] == pytest.approx(0.5)
    hurts = _metrics(fx.event([_full_example(usage=([-5.0, -13.0], [-4.0, -12.0]))]))
    assert hurts["eval/latent_usage_posterior_mean_nats_per_byte"] == pytest.approx(-0.25)
    blind = _metrics(fx.event([_full_example(usage=([-4.0, -12.0], [-4.0, -12.0]))]))
    assert blind["eval/latent_usage_posterior_mean_nats_per_byte"] == 0.0
    drawn = _metrics(fx.event([_full_example()], operand="draw_average"))
    assert "eval/latent_usage_draw_average_nats_per_byte" in drawn
    assert "eval/latent_usage_posterior_mean_nats_per_byte" not in drawn


def test_latent_usage_without_a_recorded_operand_is_a_malformed_record() -> None:
    assert project_evaluation(fx.event([_full_example()], operand=None)) is None
    assert project_evaluation(fx.event([_full_example()], operand="q_draw")) is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: e["proposals"]["prior"].update(by_sample_count="x"),
        lambda e: e["proposals"]["mixture"].update(proposal="prior"),
        lambda e: e["proposals"].update(extra={}),
        # The two cases below are well-formed blocks under a name that is not prior or mixture:
        # `extra={}` above is refused earlier by the block_rows guard, so only these reach (and
        # pin) the closed-name guard. MEASURED 2026-11-05 with that guard removed: a block named
        # q PROJECTED and folded a second copy into the q tally, and foo raised KeyError.
        lambda e: e["proposals"].update(q={**e["proposals"]["prior"], "proposal": "q"}),
        lambda e: e["proposals"].update(extra={**e["proposals"]["prior"], "proposal": "extra"}),
        lambda e: e["diagnostics"].update(prior_q_cosine=["nan"]),
        lambda e: e["diagnostics"]["log_weight_sd"]["q"].update({"2": "x"}),
        lambda e: e["latent_usage"].update(null_decoder_log_prob=[-1.0]),
        lambda e: e["latent_usage"].update(q_decoder_log_prob=[float("inf"), -1.0]),
    ],
)
def test_a_malformed_proposal_block_voids_the_whole_projection(mutate: Any) -> None:
    """A8(c): the extractor never emits the q keys while silently omitting a bad block."""
    example = _full_example()
    mutate(example)
    assert project_evaluation(fx.event([example])) is None


def test_a_block_on_some_examples_and_not_others_is_malformed() -> None:
    plain = fx.example(source_bytes=[2, 6], q={2: [fx.row(-4.0, 1.5), fx.row(-12.0, 1.0)]})
    assert project_evaluation(fx.event([_full_example(), plain])) is None


@pytest.mark.parametrize("dropped", ["all", "prior", "mixture"])
def test_a_proposal_block_on_some_examples_and_not_others_is_refused_by_its_own_guard(
    dropped: str,
) -> None:
    """E16 (L3 review): ``plain`` above also lacks the diagnostics and latent usage, so the
    sibling guards refuse it and the proposal guard is never the one tested (removing it left 159
    tests GREEN). Here the second example differs from the first ONLY in its proposal blocks."""
    second = _full_example()
    if dropped == "all":
        del second["proposals"]
    else:
        del second["proposals"][dropped]
    assert project_evaluation(fx.event([_full_example(), second])) is None


@pytest.mark.parametrize(
    "absent",
    ["log_weight_sd", "latent_usage", "prior_log_radius_mean", "q_log_radius_mean"],
)
def test_a_diagnostic_on_some_examples_and_not_others_is_refused_by_its_own_guard(
    absent: str,
) -> None:
    """X15 (L3 review): removing the whole partial-presence guard, or any one of its flags, left
    every test GREEN, because the proposal guard above already refuses the sibling fixtures. Here
    the second example is a COMPLETE example that differs from the first ONLY in the one named
    diagnostic, so the proposal guard cannot be the one that fires."""
    second = _full_example()
    if absent == "latent_usage":
        del second["latent_usage"]
    elif absent == "log_weight_sd":
        second["diagnostics"]["log_weight_sd"] = None
    else:
        second["diagnostics"][absent] = None
    assert set(second["proposals"]) == {"prior", "mixture"}
    assert project_evaluation(fx.event([_full_example(), second])) is None
    # the control: two complete examples project, so the refusal above is the one absence
    assert project_evaluation(fx.event([_full_example(), _full_example()])) is not None


def test_log_weight_sd_skips_rows_without_an_sd_and_never_invents_one() -> None:
    sd = {"q": {2: [None, 1.5], 4: [None, None]}}
    m = _metrics(fx.event([_full_example(sd=sd)]))
    assert m["eval/log_weight_sd_k2"] == pytest.approx(1.5)
    assert "eval/log_weight_sd_k4" not in m
    assert "eval/log_weight_sd_prior_k2" not in m


def test_the_extractor_vocabularies_are_the_closed_sets() -> None:
    assert PROPOSALS == ("q", "prior", "mixture")
    assert LATENT_OPERANDS == ("posterior_mean", "draw_average")
    assert EVALUATION_SCHEMAS == ("h-predictive-evaluation-v1", "h-predictive-evaluation-v2")
    assert fx.PROPOSALS == PROPOSALS and fx.OPERANDS == LATENT_OPERANDS


def test_the_global_step_axis_is_unchanged_for_a_v2_report() -> None:
    out = HTrainingStepExtractor(warmup_steps=250).extract(fx.event([_full_example()], step=500))
    assert out is not None and out.step == 750


def test_the_in_memory_report_with_tuple_byte_counts_projects_like_its_json_form() -> None:
    """The runner mirrors the report BEFORE any JSON round trip, where counts are tuples."""
    event = fx.event([_full_example()])
    tupled = {
        **event,
        "examples": [
            {**e, "source_bytes": tuple(e["source_bytes"]), "boundary_symbols": (1, 1)}
            for e in event["examples"]
        ],
    }
    assert project_evaluation(tupled) == project_evaluation(event)
    assert project_evaluation(tupled) is not None


# ------------------------------------------------------------------------------ type guards
# One malformed input per ``return None`` type guard in the extractor (L4d review, round 5).
# A MappingProxyType or a tuple is a well-formed mapping or sequence of the wrong type: with the
# guard removed the projection would read it and emit metrics; a list or a bool where a number
# belongs crashes later instead of returning None. Keys name the guarded line at 0c1d3a35.
# MEASURED 2026-11-05: each guard's ``return None`` replaced by ``pass`` turned its own case RED.
# Silent admission (metrics emitted) at 232, 288, 310, 325, 335, 341, 370, 376, 389 and 190;
# a later crash at 193 and 359 (TypeError), 329 (AttributeError) and 383 (TypeError in the mean).
# The zero-byte row at 232 put -16/2 = -8.0 into the q ratio of sums instead of voiding it.
def _sd(example: dict[str, Any]) -> dict[str, Any]:
    return example["diagnostics"]["log_weight_sd"]


def _set(example: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    node: Any = example
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


_PROJECTION_GUARDS: dict[str, Any] = {
    "bytes-zero-232": lambda ev: _set(ev["examples"][0], ("source_bytes",), [2, 0]),
    "bytes-bool-232": lambda ev: _set(ev["examples"][0], ("source_bytes",), [True, 6]),
    "finite-list-tuple-288": lambda ev: _set(
        ev["examples"][0], ("diagnostics", "prior_q_cosine"), (0.5, 0.9)
    ),
    "examples-tuple-310": lambda ev: _set(ev, ("examples",), tuple(ev["examples"])),
    "example-mapping-325": lambda ev: _set(
        ev, ("examples",), [MappingProxyType(ev["examples"][0])]
    ),
    "q-rows-list-329": lambda ev: _set(ev["examples"][0], ("by_sample_count",), []),
    "proposals-mapping-335": lambda ev: _set(
        ev["examples"][0], ("proposals",), MappingProxyType(ev["examples"][0]["proposals"])
    ),
    "block-mapping-341": lambda ev: _set(
        ev["examples"][0],
        ("proposals", "prior"),
        MappingProxyType(ev["examples"][0]["proposals"]["prior"]),
    ),
    "radius-bool-359": lambda ev: _set(
        ev["examples"][0], ("diagnostics", "q_log_radius_mean"), [0.2, True]
    ),
    "sd-mapping-370": lambda ev: _set(
        ev["examples"][0],
        ("diagnostics", "log_weight_sd"),
        MappingProxyType(_sd(ev["examples"][0])),
    ),
    "sd-tuple-376": lambda ev: _sd(ev["examples"][0])["q"].update({"2": (0.5, 1.5)}),
    "sd-bool-383": lambda ev: _sd(ev["examples"][0])["q"].update({"2": [0.5, True]}),
    "usage-mapping-389": lambda ev: _set(
        ev["examples"][0], ("latent_usage",), MappingProxyType(ev["examples"][0]["latent_usage"])
    ),
}


@pytest.mark.parametrize("guard", list(_PROJECTION_GUARDS))
def test_each_type_guard_voids_the_projection_of_a_wrong_typed_field(guard: str) -> None:
    event = fx.event([_full_example()])
    assert project_evaluation(event) is not None  # control: the well-typed record projects
    _PROJECTION_GUARDS[guard](event)
    assert project_evaluation(event) is None


def test_an_evaluation_with_a_bool_step_is_refused_not_placed_on_the_axis() -> None:
    """Guard 190: ``True`` is an int to arithmetic, so without the guard it lands on step 1."""
    extractor = HTrainingStepExtractor(warmup_steps=0)
    assert extractor.extract(fx.event([_full_example()], step=1000)) is not None  # control
    assert extractor.extract(fx.event([_full_example()], step=True)) is None


def test_a_malformed_evaluation_is_refused_by_extract_not_wrapped() -> None:
    """Guard 193: a record whose projection is ``None`` yields no ``ExtractedMetrics`` at all."""
    extractor = HTrainingStepExtractor(warmup_steps=0)
    event = fx.event([_full_example()])
    assert extractor.extract(event) is not None  # control
    event["examples"] = "not a list"
    assert extractor.extract(event) is None
