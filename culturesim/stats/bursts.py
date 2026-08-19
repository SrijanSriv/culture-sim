"""Delegated network burst analysis and inter-burst intervals (SPEC §6.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interop import cl_analysis
from .spiketrains import SpikeRecording

__all__ = ["BurstStats", "Bursts", "detect_bursts", "burst_stats", "surrogate_threshold"]

BIN_WIDTH_S = cl_analysis.DEFAULT_BURST_BIN_SIZE_S
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
    """Return the CL onset threshold expressed as spikes/bin."""
    del rng, percentile, n_surrogates
    return cl_analysis.DEFAULT_BURST_ONSET_HZ * recording.n_channels * bin_width_s


def detect_bursts(
    recording: SpikeRecording,
    rng: np.random.Generator,
    *,
    bin_width_s: float = BIN_WIDTH_S,
    min_duration_s: float = MIN_BURST_DURATION_S,
    min_electrodes: int = MIN_PARTICIPATING_ELECTRODES,
) -> Bursts:
    del rng, min_duration_s, min_electrodes
    if (
        recording.n_spikes < MIN_PARTICIPATING_ELECTRODES
        or np.unique(recording.channels).size < MIN_PARTICIPATING_ELECTRODES
    ):
        return _empty_bursts()
    result = cl_analysis.analyse_network_bursts(
        recording,
        bin_size_sec=bin_width_s,
        min_active_channels=MIN_PARTICIPATING_ELECTRODES,
    )
    dump = result.model_dump()
    sampling_frequency = float(dump["metadata"]["sampling_frequency"])
    starts = np.asarray([burst["start_frame"] for burst in dump["bursts"]], dtype=np.float64)
    stops = np.asarray([burst["end_frame"] for burst in dump["bursts"]], dtype=np.float64)
    starts /= sampling_frequency
    stops /= sampling_frequency
    n_spikes = np.asarray(dump["network_burst_spike_counts"], dtype=np.int64)
    participation = _participation(recording, starts, stops)
    return Bursts(
        starts=starts,
        stops=stops,
        n_spikes=n_spikes,
        participation=participation,
        threshold=surrogate_threshold(recording, np.random.default_rng(0), bin_width_s=bin_width_s),
    )


def burst_stats(recording: SpikeRecording, rng: np.random.Generator) -> BurstStats:
    """Burst scalars plus the full IBI array for the fingerprint's histogram."""
    bursts = detect_bursts(recording, rng)
    durations = bursts.durations
    if bursts.n_bursts == 0:
        return BurstStats(
            burst_rate_per_min=0.0,
            burst_duration_mean=float("nan"),
            burst_duration_std=float("nan"),
            burst_size_mean=float("nan"),
            burst_size_std=float("nan"),
            burst_participation_mean=float("nan"),
            ibi_seconds=np.array([], dtype=np.float64),
        )
    return BurstStats(
        burst_rate_per_min=float(bursts.n_bursts / recording.duration * 60.0),
        burst_duration_mean=float(np.mean(durations)),
        burst_duration_std=float(np.std(durations)),
        burst_size_mean=float(np.mean(bursts.n_spikes)),
        burst_size_std=float(np.std(bursts.n_spikes)),
        burst_participation_mean=float(np.mean(bursts.participation)),
        ibi_seconds=bursts.inter_burst_intervals(),
    )


def _empty_bursts() -> Bursts:
    return Bursts(
        starts=np.array([], dtype=np.float64),
        stops=np.array([], dtype=np.float64),
        n_spikes=np.array([], dtype=np.int64),
        participation=np.array([], dtype=np.float64),
        threshold=float("nan"),
    )


def _participation(recording: SpikeRecording, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    if starts.size == 0:
        return np.array([], dtype=np.float64)
    values = []
    for start, stop in zip(starts, stops, strict=True):
        in_burst = (recording.times >= start) & (recording.times < stop)
        values.append(np.unique(recording.channels[in_burst]).size / recording.n_channels)
    return np.asarray(values, dtype=np.float64)
