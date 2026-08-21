"""Unit tests for Task 7 validation helpers (no Brian2)."""

from __future__ import annotations

import numpy as np
import pytest

from culturesim.stats.spiketrains import SpikeRecording
from culturesim.validate.cross_culture import marginal_overlaps, posterior_overlap
from culturesim.validate.perturbation import StimulusProtocol, evoked_response


def test_posterior_overlap_identical_is_near_one():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(800, 3))
    b = a + rng.normal(scale=0.05, size=a.shape)
    assert posterior_overlap(a, b) > 0.5


def test_posterior_overlap_far_apart_is_near_zero():
    rng = np.random.default_rng(1)
    a = rng.normal(loc=0.0, size=(800, 2))
    b = rng.normal(loc=10.0, size=(800, 2))
    assert posterior_overlap(a, b) < 0.05


def test_marginal_overlaps_keys():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(100, 2))
    b = rng.normal(size=(100, 2))
    overlaps = marginal_overlaps(a, b, names=("x", "y"))
    assert set(overlaps) == {"x", "y"}


def test_evoked_response_empty_stimuli():
    recording = SpikeRecording(
        times=np.array([0.1, 0.2]),
        channels=np.array([0, 1], dtype=np.int32),
        n_channels=2,
        duration=1.0,
        source="test",
    )
    result = evoked_response(recording, [], amplitude=0.5)
    assert result.n_trials == 0
    assert result.response_probability == 0.0


def test_evoked_response_detects_locked_spikes():
    stim = np.array([1.0, 2.0, 3.0])
    # One spike 10 ms after each stimulus on channel 0.
    times = stim + 0.010
    recording = SpikeRecording(
        times=times,
        channels=np.zeros(times.size, dtype=np.int32),
        n_channels=4,
        duration=5.0,
        source="test",
    )
    result = evoked_response(recording, stim, amplitude=1.0, blanking_ms=2.0)
    assert result.n_trials == 3
    assert result.response_probability == pytest.approx(1.0)
    assert result.mean_latency_s == pytest.approx(0.010, abs=1e-6)
    assert result.psth_hz.sum() > 0


def test_stimulus_protocol_duration():
    protocol = StimulusProtocol(
        electrode=0, amplitudes=(0.5, 1.0), n_pulses_per_amplitude=5, inter_stimulus_interval_s=2.0
    )
    assert protocol.total_duration_s == pytest.approx(20.0)
