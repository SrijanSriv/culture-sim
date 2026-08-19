"""Tests for the virtual MEA layouts and config (SPEC §5).

The detection pipeline itself is Task 2; what is testable now is the geometry and the
config, and those matter -- a layout whose electrode count silently disagrees with the
real array invalidates every comparison downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from culturesim.observation.virtual_mea import (
    ElectrodeLayout,
    ObservationConfig,
    detection_radius_um,
    observe,
)


def test_mcs_60_layout_has_60_electrodes() -> None:
    config = ObservationConfig.load("observation.yaml", layout_name="mcs_60")
    assert config.layout.n_electrodes == 60
    # 8x8 at 200 um pitch spans 1400 um between outermost electrodes.
    width, height = config.layout.extent_um
    assert width == pytest.approx(1400.0)
    assert height == pytest.approx(1400.0)


def test_hd_mea_layout_has_1024_electrodes() -> None:
    config = ObservationConfig.load("observation.yaml", layout_name="hd_mea_1024")
    assert config.layout.n_electrodes == 1024


def test_layouts_are_centred_on_the_sheet() -> None:
    for name in ("mcs_60", "hd_mea_1024"):
        layout = ObservationConfig.load("observation.yaml", layout_name=name).layout
        assert layout.x_um.mean() == pytest.approx(0.0, abs=1e-9)
        assert layout.y_um.mean() == pytest.approx(0.0, abs=1e-9)


def test_corner_omission_removes_exactly_four() -> None:
    full = ElectrodeLayout.grid("full", 8, 8, 200.0)
    trimmed = ElectrodeLayout.grid("trimmed", 8, 8, 200.0, omit_corners=True)
    assert full.n_electrodes - trimmed.n_electrodes == 4
    corners = {(-700.0, -700.0), (-700.0, 700.0), (700.0, -700.0), (700.0, 700.0)}
    present = set(zip(trimmed.x_um.tolist(), trimmed.y_um.tolist(), strict=True))
    assert not (corners & present)


def test_declared_electrode_count_must_match_the_geometry() -> None:
    """Downstream configs and figures assume the declared count."""
    with pytest.raises(ValueError, match="declares n_electrodes=99"):
        ElectrodeLayout.from_config(
            "broken",
            {"kind": "grid", "n_rows": 8, "n_cols": 8, "pitch_um": 200.0, "n_electrodes": 99},
        )


def test_unsupported_layout_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported layout kind"):
        ElectrodeLayout.from_config("weird", {"kind": "hexagonal", "n_rows": 4, "n_cols": 4})


def test_unknown_layout_name_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="not in observation config"):
        ObservationConfig.load("observation.yaml", layout_name="nonexistent")


def test_layout_metadata_carries_the_geometry() -> None:
    """Distance-dependent statistics need geometry to travel with the recording."""
    layout = ObservationConfig.load("observation.yaml").layout
    metadata = layout.to_metadata()
    assert metadata["n_electrodes"] == layout.n_electrodes
    assert len(metadata["x_um"]) == layout.n_electrodes
    assert metadata["pitch_um"] == pytest.approx(layout.pitch_um)


# -- amplitude model ------------------------------------------------------


def test_detection_radius_matches_the_amplitude_model() -> None:
    """At the returned radius, A = A_0 / (1 + (d/d_0)**2) equals the threshold."""
    A_0, d_0, threshold = 150.0, 25.0, 15.0
    radius = detection_radius_um(A_0, d_0, threshold)
    assert A_0 / (1.0 + (radius / d_0) ** 2) == pytest.approx(threshold)


def test_detection_radius_is_zero_when_nothing_is_detectable() -> None:
    assert detection_radius_um(10.0, 25.0, 50.0) == 0.0


def test_default_detection_radius_is_physically_plausible() -> None:
    """Real MEA electrodes see somata within roughly 50-100 um."""
    config = ObservationConfig.load("observation.yaml")
    assert 40.0 < config.detection_radius_um < 120.0


def test_threshold_is_k_times_the_noise_rms() -> None:
    config = ObservationConfig.load("observation.yaml")
    assert config.threshold_uv == pytest.approx(config.threshold_k * config.noise_rms_uv)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("noise_rms_uv", 0.0, "noise RMS must be positive"),
        ("threshold_k", -1.0, "threshold_k must be positive"),
        ("dead_electrode_fraction", 1.0, "dead_electrode_fraction"),
        ("dead_time_ms", -1.0, "dead_time_ms"),
        ("A_0_uv", 0.0, "amplitude parameters must be positive"),
    ],
)
def test_invalid_observation_config_is_rejected(field: str, value: float, match: str) -> None:
    layout = ElectrodeLayout.grid("g", 4, 4, 200.0)
    with pytest.raises(ValueError, match=match):
        ObservationConfig(layout=layout, **{field: value})


def test_observe_output_writes_cl_h5_that_sdk_can_open(tmp_path) -> None:
    from culturesim.interop.cl_analysis import recording_view

    recording = observe(
        np.array([0.1, 0.2, 0.3]),
        np.array([0, 0, 0]),
        np.array([0.0]),
        np.array([0.0]),
        1.0,
        ObservationConfig.load("observation.yaml"),
        np.random.default_rng(0),
    )
    path = recording.to_hdf5(tmp_path / "observed.h5")
    with recording_view(path) as view:
        assert view.attributes.channel_count == 64
        assert len(view.spikes) == recording.n_spikes


def test_dead_electrodes_emit_no_spikes() -> None:
    """SPEC §13 Task 2: a marked-dead site records nothing, even with a soma on it."""
    layout = ElectrodeLayout.grid("tiny", 4, 4, 200.0)
    config = ObservationConfig(
        layout=layout,
        per_neuron_sigma=0.0,
        dead_electrode_fraction=0.0,
        dead_electrode_indices=(0, 1),
    )
    # Neuron 0 sits on dead electrode 0; neuron 1 sits on live electrode 5.
    recording = observe(
        np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60]),
        np.array([0, 0, 0, 1, 1, 1]),
        np.array([layout.x_um[0], layout.x_um[5]]),
        np.array([layout.y_um[0], layout.y_um[5]]),
        1.0,
        config,
        np.random.default_rng(0),
    )
    assert recording.metadata["dead_electrodes"] == [0, 1]
    assert 0 not in recording.channels
    assert 1 not in recording.channels
    assert 5 in set(recording.channels.tolist())


def test_neuron_sheet_shares_the_electrode_origin() -> None:
    """A sheet in [0, W] against electrodes centred at 0 only observes one corner."""
    from culturesim.model.network import place_neurons
    from culturesim.model.params import ModelParams

    params = ModelParams.load("model_default.yaml")
    positions = place_neurons(params, np.random.default_rng(0))
    half_w = params.network.sheet_width_um / 2.0
    half_h = params.network.sheet_height_um / 2.0
    assert positions.x_um.min() >= -half_w
    assert positions.x_um.max() <= half_w
    assert positions.y_um.min() >= -half_h
    assert positions.y_um.max() <= half_h
    for name in ("mcs_60", "hd_mea_1024"):
        layout = ObservationConfig.load("observation.yaml", layout_name=name).layout
        assert layout.x_um.min() >= -half_w
        assert layout.x_um.max() <= half_w
        assert layout.y_um.min() >= -half_h
        assert layout.y_um.max() <= half_h


def test_sixty_and_hd_mea_observations_yield_different_fingerprints() -> None:
    """SPEC §13 Task 2: identical neuron spikes, different arrays, different vectors."""
    from dataclasses import replace

    from culturesim.stats.branching import naive_branching_ratio
    from culturesim.stats.fingerprint import FingerprintSpec, compute_fingerprint
    from culturesim.stats.rates import rate_stats

    rng = np.random.default_rng(7)
    n_neurons, duration_s, rate_hz = 80, 8.0, 8.0
    counts = rng.poisson(rate_hz * duration_s, size=n_neurons)
    times = np.concatenate([rng.uniform(0.0, duration_s, size=c) for c in counts])
    neurons = np.repeat(np.arange(n_neurons), counts)
    order = np.argsort(times, kind="stable")
    times, neurons = times[order], neurons[order]
    x_um = rng.uniform(-900.0, 900.0, size=n_neurons)
    y_um = rng.uniform(-900.0, 900.0, size=n_neurons)

    recordings = {}
    for name in ("mcs_60", "hd_mea_1024"):
        config = replace(
            ObservationConfig.load("observation.yaml", layout_name=name),
            dead_electrode_fraction=0.0,
            dead_electrode_indices=(),
        )
        recordings[name] = observe(
            times, neurons, x_um, y_um, duration_s, config, np.random.default_rng(11)
        )

    rec_60, rec_1024 = recordings["mcs_60"], recordings["hd_mea_1024"]
    assert rec_60.n_channels == 60
    assert rec_1024.n_channels == 1024
    assert rec_60.n_spikes != rec_1024.n_spikes

    rates_60, rates_1024 = rate_stats(rec_60), rate_stats(rec_1024)
    assert rates_60.per_electrode_rates.size == 60
    assert rates_1024.per_electrode_rates.size == 1024
    # Denser arrays typically see a *higher* active fraction (more sites near a soma),
    # so do not assert a direction -- only that the two observations disagree.
    assert rates_60.active_electrode_fraction != pytest.approx(rates_1024.active_electrode_fraction)

    spec = FingerprintSpec.load("fingerprint.yaml")
    fp_60 = compute_fingerprint(rec_60, spec)
    fp_1024 = compute_fingerprint(rec_1024, spec)
    assert fp_60.names == fp_1024.names
    assert not np.allclose(fp_60.values, fp_1024.values, equal_nan=True)
    # Delegated burst analysis cannot ingest 1024 channels in cl-sdk==1.0.0.
    assert np.isnan(fp_1024["burst_rate_per_min"])
    assert np.isfinite(fp_60["rate_mean"])
    # The naive branching estimator is the statistic that *should* change with
    # electrode count; the figure documents that bias.
    assert naive_branching_ratio(rec_60) != pytest.approx(naive_branching_ratio(rec_1024))
