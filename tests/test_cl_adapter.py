"""Tests for the CL SDK recording adapter (SPEC §6.0.1)."""

from __future__ import annotations

import numpy as np
import pytest

from culturesim.interop.cl_adapter import (
    CL_COMMON_GROUND_CHANNELS,
    cl_channel_mapping,
    from_cl_h5,
    to_cl_h5,
)
from culturesim.interop.cl_analysis import analyse_network_bursts, recording_view
from culturesim.stats.spiketrains import SpikeRecording


def test_sixty_channel_recordings_map_onto_non_ground_cl_channels() -> None:
    mapping = cl_channel_mapping(60)
    assert mapping.size == 60
    assert not set(mapping.tolist()) & set(CL_COMMON_GROUND_CHANNELS)
    assert mapping.max() == 62


def test_cl_h5_round_trip_is_identity(tmp_path, poisson_recording: SpikeRecording) -> None:
    path = to_cl_h5(poisson_recording, tmp_path / "recording.h5")
    assert from_cl_h5(path) == poisson_recording


def test_cl_recording_loads_in_sdk_and_runs_burst_analysis(
    tmp_path,
    poisson_recording: SpikeRecording,
) -> None:
    path = to_cl_h5(poisson_recording, tmp_path / "recording.h5")
    with recording_view(path) as view:
        assert view.attributes.channel_count == 64
        assert len(view.spikes) == poisson_recording.n_spikes
        result = view.analyse_network_bursts(
            bin_size_sec=0.05,
            onset_freq_hz=3.0,
            offset_freq_hz=1.0,
            min_active_channels=3,
        )
    assert result.bin_size_sec == pytest.approx(0.05)


def test_project_wrapper_reaches_delegated_burst_analysis(
    poisson_recording: SpikeRecording,
) -> None:
    result = analyse_network_bursts(poisson_recording)
    assert result.network_burst_count >= 0


def test_hd_mea_channel_ids_are_not_losslessly_supported_by_cl_1_0(tmp_path) -> None:
    recording = SpikeRecording(
        times=np.array([0.1]),
        channels=np.array([1023]),
        n_channels=1024,
        duration=1.0,
        source="synthetic-hd",
    )
    with pytest.raises(ValueError, match="uint8"):
        to_cl_h5(recording, tmp_path / "hd.h5")
