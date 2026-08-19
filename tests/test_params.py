"""Tests for the parameter split, prior box, and config loading (SPEC §4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from culturesim.config import load_config
from culturesim.model.params import (
    DEFAULT_PRIOR_RANGES,
    FREE_PARAM_NAMES,
    FreeParams,
    ModelParams,
    NetworkParams,
    PriorBox,
    SimulationParams,
)


def test_there_are_exactly_eight_free_parameters() -> None:
    """SPEC §4.4 fixes this at 8; the SBI posterior has that dimensionality."""
    assert len(FREE_PARAM_NAMES) == 8
    assert set(FREE_PARAM_NAMES) == set(DEFAULT_PRIOR_RANGES)


def test_vector_round_trip_preserves_order() -> None:
    params = FreeParams(
        p_conn=0.1, w_e=1.0, g=5.0, rate_bg=2.0, tau_m=25.0, U=0.3, tau_rec=900.0, b=2.0
    )
    vector = params.to_vector()
    assert vector.tolist() == [0.1, 1.0, 5.0, 2.0, 25.0, 0.3, 900.0, 2.0]
    assert FreeParams.from_vector(vector) == params


def test_from_vector_rejects_the_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected 8 free parameters"):
        FreeParams.from_vector(np.zeros(7))


def test_replace_rejects_a_non_free_parameter() -> None:
    """Reassigning a fixed parameter through the free interface is a category error."""
    with pytest.raises(KeyError, match="not free parameters"):
        FreeParams().replace(v_th=-45.0)


def test_shipped_defaults_lie_inside_the_prior() -> None:
    params = ModelParams.load("model_default.yaml")
    assert params.prior.contains(params.free), "the hand-tuned start must be a legal draw"


def test_config_priors_agree_with_the_spec() -> None:
    """The YAML and SPEC §4.4 must not drift apart silently."""
    config_priors = load_config("model_default.yaml")["priors"]
    for name, (low, high) in DEFAULT_PRIOR_RANGES.items():
        assert tuple(config_priors[name]) == pytest.approx((low, high))


def test_prior_sampling_stays_in_the_box() -> None:
    prior = PriorBox.default()
    rng = np.random.default_rng(0)
    for draw in prior.sample(rng, n=200):
        assert prior.contains(draw)


def test_prior_bounds_are_in_free_param_order() -> None:
    prior = PriorBox.default()
    assert prior.bounds == [DEFAULT_PRIOR_RANGES[n] for n in FREE_PARAM_NAMES]
    np.testing.assert_array_less(prior.low, prior.high)


def test_prior_clip_pulls_a_draw_back_into_the_box() -> None:
    prior = PriorBox.default()
    wild = FreeParams(p_conn=99.0, w_e=-5.0)
    clipped = prior.clip(wild)
    assert prior.contains(clipped)
    assert clipped.p_conn == pytest.approx(DEFAULT_PRIOR_RANGES["p_conn"][1])
    assert clipped.w_e == pytest.approx(DEFAULT_PRIOR_RANGES["w_e"][0])


def test_prior_rejects_an_inverted_range() -> None:
    ranges = dict(DEFAULT_PRIOR_RANGES)
    ranges["g"] = (12.0, 1.0)
    with pytest.raises(ValueError, match="must have low < high"):
        PriorBox(ranges=ranges)


def test_config_round_trip() -> None:
    original = ModelParams.load("model_default.yaml")
    assert ModelParams.from_config(original.to_config()) == original


def test_unknown_config_key_is_rejected() -> None:
    """A typo must not silently fall back to a default (SPEC §11)."""
    with pytest.raises(ValueError, match="unknown keys for FreeParams"):
        ModelParams.from_config({"free": {"w_ee": 1.0}})


def test_derived_quantities() -> None:
    params = ModelParams.load("model_default.yaml")
    assert params.lambda_conn_um == pytest.approx(params.network.sheet_width_um / 3.0, rel=1e-6)
    assert params.w_i == pytest.approx(params.free.g * params.free.w_e)


def test_network_defaults_meet_the_spec() -> None:
    """SPEC §4.3: 1000 neurons, 80/20 split; burst statistics break below ~800."""
    network = ModelParams.load("model_default.yaml").network
    assert network.n_neurons == 1000
    assert network.n_excitatory == 800
    assert network.n_inhibitory == 200


def test_invalid_network_params_are_rejected() -> None:
    with pytest.raises(ValueError, match="excitatory_fraction"):
        NetworkParams(excitatory_fraction=1.0)
    with pytest.raises(ValueError, match="n_neurons must be positive"):
        NetworkParams(n_neurons=0)


def test_simulation_total_duration_includes_the_transient() -> None:
    simulation = SimulationParams(duration_s=300.0, transient_s=20.0)
    assert simulation.total_duration_s == pytest.approx(320.0)


def test_fixed_params_are_physically_consistent() -> None:
    from culturesim.model.params import FixedParams

    with pytest.raises(ValueError, match="threshold must be above"):
        FixedParams(v_th=-80.0)
    with pytest.raises(ValueError, match="reset must be below threshold"):
        FixedParams(v_reset=-40.0)


def test_static_synapse_ablation_is_off_by_default() -> None:
    """The ablation exists for the Task 1 acceptance test, not as a default."""
    assert ModelParams.load("model_default.yaml").simulation.static_synapses is False


def test_poisson_background_is_the_scientific_default() -> None:
    """Diffusion was a failed speed experiment; SPEC §4.3 is independent Poisson drive."""
    simulation = ModelParams.load("model_default.yaml").simulation
    assert simulation.background_mode == "poisson"
    assert SimulationParams().background_mode == "poisson"


def test_timestep_is_the_measured_budget_decision() -> None:
    """0.2 ms is what puts 300 s biological well under the 60 s wall-clock budget."""
    simulation = ModelParams.load("model_default.yaml").simulation
    assert simulation.dt_ms == pytest.approx(0.2)
    assert SimulationParams().dt_ms == pytest.approx(0.2)
