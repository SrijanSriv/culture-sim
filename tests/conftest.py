"""Shared fixtures and hypothesis strategies.

The edge-case recordings here exist because SPEC §12 requires every statistic to
return without crashing on empty, single-spike and single-active-electrode input, with
documented sentinel values. They are fixtures rather than inline constructions so that
each new statistic in Task 3 is tested against all of them.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

from culturesim.stats.spiketrains import SpikeRecording

# Writing HDF5 inside a hypothesis example is slower than the default deadline
# allows, and a timing-dependent failure here would be noise, not a bug.
settings.register_profile(
    "culturesim",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    max_examples=50,
)
settings.load_profile("culturesim")

MASTER_SEED = 20250819


@pytest.fixture
def rng() -> np.random.Generator:
    from culturesim.rng import generator

    return generator(MASTER_SEED, "tests")


@pytest.fixture
def poisson_recording(rng: np.random.Generator) -> SpikeRecording:
    """Homogeneous Poisson spikes on 60 channels at 5 Hz each (SPEC §12).

    Analytically known answers: ISI CV -> 1.0, MR branching ratio -> 0.
    """
    n_channels, duration, rate_hz = 60, 300.0, 5.0
    counts = rng.poisson(rate_hz * duration, size=n_channels)
    times = np.concatenate([rng.uniform(0.0, duration, size=c) for c in counts])
    channels = np.repeat(np.arange(n_channels), counts)
    order = np.argsort(times, kind="stable")
    return SpikeRecording(
        times=times[order],
        channels=channels[order],
        n_channels=n_channels,
        duration=duration,
        source="synthetic-poisson",
        metadata={"process": "homogeneous_poisson", "rate_hz": rate_hz},
    )


@pytest.fixture
def regular_recording() -> SpikeRecording:
    """Perfectly periodic spikes on every channel; ISI CV -> 0 (SPEC §12)."""
    n_channels, duration, interval = 8, 100.0, 0.1
    per_channel = np.arange(interval, duration, interval)
    times = np.tile(per_channel, n_channels)
    channels = np.repeat(np.arange(n_channels), per_channel.size)
    order = np.argsort(times, kind="stable")
    return SpikeRecording(
        times=times[order],
        channels=channels[order],
        n_channels=n_channels,
        duration=duration,
        source="synthetic-regular",
        metadata={"process": "regular", "interval_s": interval},
    )


@pytest.fixture
def empty_recording() -> SpikeRecording:
    """No spikes at all -- a dead culture, or a bad parameter draw during SBI."""
    return SpikeRecording(
        times=np.array([], dtype=np.float64),
        channels=np.array([], dtype=np.int32),
        n_channels=60,
        duration=300.0,
        source="synthetic-empty",
        metadata={},
    )


@pytest.fixture
def single_spike_recording() -> SpikeRecording:
    return SpikeRecording(
        times=np.array([12.5]),
        channels=np.array([7]),
        n_channels=60,
        duration=300.0,
        source="synthetic-single-spike",
        metadata={},
    )


@pytest.fixture
def single_channel_recording(rng: np.random.Generator) -> SpikeRecording:
    """One active electrode, 59 silent -- pairwise statistics are undefined here."""
    times = np.sort(rng.uniform(0.0, 300.0, size=500))
    return SpikeRecording(
        times=times,
        channels=np.full(times.size, 3, dtype=np.int32),
        n_channels=60,
        duration=300.0,
        source="synthetic-single-channel",
        metadata={},
    )


@pytest.fixture
def edge_case_recordings(
    empty_recording: SpikeRecording,
    single_spike_recording: SpikeRecording,
    single_channel_recording: SpikeRecording,
) -> dict[str, SpikeRecording]:
    """The SPEC §12 degenerate cases, for parametrising statistic tests."""
    return {
        "empty": empty_recording,
        "single_spike": single_spike_recording,
        "single_channel": single_channel_recording,
    }


@st.composite
def spike_recordings(draw: st.DrawFn) -> SpikeRecording:
    """Arbitrary valid recordings, for the property tests."""
    n_channels = draw(st.integers(min_value=1, max_value=32))
    duration = draw(st.floats(min_value=0.1, max_value=1000.0, allow_nan=False))
    n_spikes = draw(st.integers(min_value=0, max_value=200))
    times = sorted(
        draw(
            st.lists(
                st.floats(min_value=0.0, max_value=duration, allow_nan=False, allow_infinity=False),
                min_size=n_spikes,
                max_size=n_spikes,
            )
        )
    )
    channels = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_channels - 1),
            min_size=n_spikes,
            max_size=n_spikes,
        )
    )
    metadata = draw(
        st.dictionaries(
            st.text(min_size=1, max_size=8),
            st.one_of(st.integers(), st.booleans(), st.text(max_size=8)),
            max_size=3,
        )
    )
    return SpikeRecording(
        times=np.array(times, dtype=np.float64),
        channels=np.array(channels, dtype=np.int32),
        n_channels=n_channels,
        duration=duration,
        source=draw(st.sampled_from(["simulation", "wagenaar2006", "synthetic"])),
        metadata=metadata,
    )
