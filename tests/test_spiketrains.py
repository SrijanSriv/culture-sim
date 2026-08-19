"""Tests for the canonical spike data structure (SPEC §12).

Covers the HDF5 round-trip identity and the property-based invariants hypothesis is
required for: sortedness, channel bounds, duration consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import spike_recordings
from hypothesis import given
from hypothesis import strategies as st

from culturesim.stats.spiketrains import SpikeRecording, load_recording, save_recording

# -- validation -----------------------------------------------------------


def test_rejects_unsorted_times() -> None:
    with pytest.raises(ValueError, match="sorted ascending"):
        SpikeRecording(
            times=np.array([1.0, 0.5]),
            channels=np.array([0, 1]),
            n_channels=2,
            duration=10.0,
            source="test",
        )


def test_rejects_channel_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"channels must lie in"):
        SpikeRecording(
            times=np.array([1.0]),
            channels=np.array([5]),
            n_channels=2,
            duration=10.0,
            source="test",
        )


def test_rejects_spike_after_duration() -> None:
    with pytest.raises(ValueError, match="exceeds duration"):
        SpikeRecording(
            times=np.array([11.0]),
            channels=np.array([0]),
            n_channels=2,
            duration=10.0,
            source="test",
        )


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        SpikeRecording(
            times=np.array([1.0, 2.0]),
            channels=np.array([0]),
            n_channels=2,
            duration=10.0,
            source="test",
        )


@pytest.mark.parametrize("duration", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_bad_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration must be"):
        SpikeRecording(
            times=np.array([], dtype=np.float64),
            channels=np.array([], dtype=np.int32),
            n_channels=2,
            duration=duration,
            source="test",
        )


def test_arrays_are_immutable(poisson_recording: SpikeRecording) -> None:
    """A frozen dataclass whose arrays are writable is not actually frozen."""
    with pytest.raises(ValueError):
        poisson_recording.times[0] = 999.0
    with pytest.raises(ValueError):
        poisson_recording.channels[0] = 999


def test_input_arrays_are_copied() -> None:
    """Mutating the caller's array must not change the recording."""
    times = np.array([1.0, 2.0])
    recording = SpikeRecording(
        times=times, channels=np.array([0, 0]), n_channels=1, duration=10.0, source="test"
    )
    times[0] = 5.0
    assert recording.times[0] == 1.0


# -- derived quantities ---------------------------------------------------


def test_rates_use_declared_duration(single_channel_recording: SpikeRecording) -> None:
    """Rate is spikes / duration, not spikes / observed span."""
    expected = single_channel_recording.n_spikes / single_channel_recording.duration
    assert single_channel_recording.mean_rate == pytest.approx(expected)
    rates = single_channel_recording.channel_rates()
    assert rates.size == single_channel_recording.n_channels
    assert rates[3] == pytest.approx(expected)
    assert np.count_nonzero(rates) == 1


def test_by_channel_covers_silent_channels(single_channel_recording: SpikeRecording) -> None:
    per_channel = single_channel_recording.by_channel()
    assert len(per_channel) == single_channel_recording.n_channels
    assert per_channel[3].size == single_channel_recording.n_spikes
    assert all(per_channel[c].size == 0 for c in range(60) if c != 3)


def test_time_slice_rezeroes_and_preserves_channels(poisson_recording: SpikeRecording) -> None:
    sliced = poisson_recording.time_slice(100.0, 200.0)
    assert sliced.duration == pytest.approx(100.0)
    assert sliced.n_channels == poisson_recording.n_channels
    assert sliced.times.min() >= 0.0
    assert sliced.times.max() <= sliced.duration


def test_drop_channels_keeps_channel_indices(poisson_recording: SpikeRecording) -> None:
    """Electrode identity is geometric, so indices must survive blanking."""
    dropped = poisson_recording.drop_channels(np.array([0, 1, 2]))
    assert dropped.n_channels == poisson_recording.n_channels
    assert dropped.spike_counts()[:3].sum() == 0
    assert dropped.spike_counts()[3] == poisson_recording.spike_counts()[3]


# -- HDF5 round-trip (SPEC §12) -------------------------------------------


@pytest.mark.parametrize(
    "fixture_name",
    ["poisson_recording", "regular_recording", "empty_recording", "single_spike_recording"],
)
def test_hdf5_round_trip_is_identity(fixture_name: str, tmp_path, request) -> None:
    original: SpikeRecording = request.getfixturevalue(fixture_name)
    path = save_recording(original, tmp_path / "recording.h5")
    restored = load_recording(path)

    assert restored == original
    np.testing.assert_array_equal(restored.times, original.times)
    np.testing.assert_array_equal(restored.channels, original.channels)
    assert restored.times.dtype == np.float64
    assert restored.channels.dtype == np.int32
    assert restored.duration == original.duration
    assert restored.metadata == original.metadata


def test_hdf5_native_write_uses_cl_recording_schema(tmp_path, poisson_recording) -> None:
    h5py = pytest.importorskip("h5py")
    path = poisson_recording.to_hdf5(tmp_path / "recording.h5")
    with h5py.File(path, "r") as handle:
        assert "spikes" in handle
        assert "channel_count" in handle.attrs
        assert "spike_recording" not in handle
        assert int(handle.attrs["culture_sim_original_n_channels"]) == poisson_recording.n_channels


# -- property-based tests (SPEC §12) --------------------------------------


@given(recording=spike_recordings())
def test_invariants_hold_for_any_recording(recording: SpikeRecording) -> None:
    assert np.all(np.diff(recording.times) >= 0), "times must stay sorted"
    assert np.all(recording.channels >= 0)
    assert np.all(recording.channels < recording.n_channels)
    assert recording.times.size == recording.channels.size == recording.n_spikes
    if recording.n_spikes:
        assert recording.times[-1] <= recording.duration
    assert recording.spike_counts().sum() == recording.n_spikes
    assert recording.spike_counts().size == recording.n_channels


@given(recording=spike_recordings())
def test_round_trip_is_identity_for_any_recording(
    recording: SpikeRecording, tmp_path_factory
) -> None:
    path = tmp_path_factory.mktemp("h5") / "recording.h5"
    assert load_recording(save_recording(recording, path)) == recording


@given(recording=spike_recordings())
def test_by_channel_partitions_the_spikes(recording: SpikeRecording) -> None:
    per_channel = recording.by_channel()
    assert sum(t.size for t in per_channel) == recording.n_spikes
    for times in per_channel:
        assert np.all(np.diff(times) >= 0)


@given(recording=spike_recordings(), fraction=st.floats(min_value=0.1, max_value=0.9))
def test_time_slice_never_loses_or_invents_spikes(
    recording: SpikeRecording, fraction: float
) -> None:
    cut = recording.duration * fraction
    first = recording.time_slice(0.0, cut)
    second = recording.time_slice(cut, recording.duration)
    assert first.n_spikes + second.n_spikes == recording.n_spikes
