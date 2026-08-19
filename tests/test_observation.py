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
