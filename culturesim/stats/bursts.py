"""Network burst detection and inter-burst intervals (SPEC §6.2).

Task 3.

Detection: bin array-wide spike counts at 25 ms; a burst starts when the count
exceeds a threshold derived from an ISI-shuffled surrogate null and ends when it
drops back below, subject to a minimum duration and a minimum electrode
participation. The threshold comes from the surrogate rather than from a fixed
multiple of the mean, because a fixed multiple makes the detector's sensitivity a
function of the firing rate -- which is itself one of the fitted parameters.

The full inter-burst-interval distribution matters more than any burst scalar:
real IBIs span roughly 1-300 s, and that range is hard to reproduce by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spiketrains import SpikeRecording

__all__ = ["BurstStats", "Bursts", "detect_bursts", "burst_stats", "surrogate_threshold"]

BIN_WIDTH_S = 0.025  # SPEC §6.2
MIN_BURST_DURATION_S = 0.05
MIN_PARTICIPATING_ELECTRODES = 3
SURROGATE_PERCENTILE = 99.0  # threshold percentile of the shuffled-ISI null
N_SURROGATES = 100


@dataclass(frozen=True)
class Bursts:
    """Detected network bursts. All times in seconds."""

    starts: np.ndarray
    stops: np.ndarray
    n_spikes: np.ndarray
    participation: np.ndarray  # fraction of electrodes contributing per burst
    threshold: float  # spikes/bin, from the surrogate null

    @property
    def durations(self) -> np.ndarray:
        return self.stops - self.starts

    @property
    def n_bursts(self) -> int:
        return int(self.starts.size)

    def inter_burst_intervals(self) -> np.ndarray:
        """Onset-to-onset intervals; empty for fewer than two bursts."""
        return np.diff(self.starts)


@dataclass(frozen=True)
class BurstStats:
    burst_rate_per_min: float
    burst_duration_mean: float
    burst_duration_std: float
    burst_size_mean: float
    burst_size_std: float
    burst_participation_mean: float
    ibi_seconds: np.ndarray


def surrogate_threshold(
    recording: SpikeRecording,
    rng: np.random.Generator,
    *,
    bin_width_s: float = BIN_WIDTH_S,
    percentile: float = SURROGATE_PERCENTILE,
    n_surrogates: int = N_SURROGATES,
) -> float:
    """Burst-onset threshold in spikes/bin from an ISI-shuffled surrogate null.

    Shuffling ISIs per electrode destroys cross-electrode coincidence while
    preserving each electrode's rate and ISI distribution, so what remains is the
    count a bin reaches by chance alone.
    """
    raise NotImplementedError("Task 3 (SPEC §6.2)")


def detect_bursts(
    recording: SpikeRecording,
    rng: np.random.Generator,
    *,
    bin_width_s: float = BIN_WIDTH_S,
    min_duration_s: float = MIN_BURST_DURATION_S,
    min_electrodes: int = MIN_PARTICIPATING_ELECTRODES,
) -> Bursts:
    raise NotImplementedError("Task 3 (SPEC §6.2)")


def burst_stats(recording: SpikeRecording, rng: np.random.Generator) -> BurstStats:
    """Burst scalars plus the full IBI array for the fingerprint's histogram."""
    raise NotImplementedError("Task 3 (SPEC §6.2)")
