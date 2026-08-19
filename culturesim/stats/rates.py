"""Firing rate and ISI statistics (SPEC §6.1).

Task 3. The across-electrode heterogeneity is a fitting target, not noise: real
arrays have a few hot electrodes and many quiet ones, so a simulation with a
uniform rate distribution has failed even if its mean rate is right.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spiketrains import SpikeRecording

__all__ = ["RateStats", "rate_stats", "isi_cv", "active_electrode_fraction"]

# SPEC §6.1: an electrode counts as active above this rate.
ACTIVE_RATE_THRESHOLD_HZ = 0.01


@dataclass(frozen=True)
class RateStats:
    """Result of :func:`rate_stats`. NaN where a statistic is undefined."""

    rate_mean: float
    rate_std: float
    rate_p10: float
    rate_p50: float
    rate_p90: float
    isi_cv_pooled: float
    isi_cv_electrode_mean: float
    isi_cv_electrode_std: float
    active_electrode_fraction: float
    per_electrode_rates: np.ndarray


def rate_stats(recording: SpikeRecording) -> RateStats:
    """Mean/std/percentiles of the across-electrode rate distribution, plus ISI CVs."""
    per_electrode_rates = recording.channel_rates()
    per_channel_cv = np.asarray(
        [isi_cv(times) for times in recording.by_channel()],
        dtype=np.float64,
    )
    finite_cv = per_channel_cv[np.isfinite(per_channel_cv)]
    return RateStats(
        rate_mean=float(np.mean(per_electrode_rates)),
        rate_std=float(np.std(per_electrode_rates)),
        rate_p10=float(np.percentile(per_electrode_rates, 10)),
        rate_p50=float(np.percentile(per_electrode_rates, 50)),
        rate_p90=float(np.percentile(per_electrode_rates, 90)),
        isi_cv_pooled=isi_cv(recording.times),
        isi_cv_electrode_mean=float(np.mean(finite_cv)) if finite_cv.size else float("nan"),
        isi_cv_electrode_std=float(np.std(finite_cv)) if finite_cv.size else float("nan"),
        active_electrode_fraction=active_electrode_fraction(recording),
        per_electrode_rates=per_electrode_rates,
    )


def isi_cv(spike_times: np.ndarray) -> float:
    """Coefficient of variation of inter-spike intervals.

    Sentinel: NaN for fewer than 3 spikes (fewer than 2 intervals), where the CV
    is not defined. A regular train gives ~0 and a Poisson train ~1 (SPEC §12).
    """
    spike_times = np.asarray(spike_times, dtype=np.float64)
    if spike_times.size < 3:
        return float("nan")
    intervals = np.diff(spike_times)
    if intervals.size < 2:
        return float("nan")
    mean = float(np.mean(intervals))
    if mean <= 0.0:
        return float("nan")
    return float(np.std(intervals) / mean)


def active_electrode_fraction(
    recording: SpikeRecording,
    threshold_hz: float = ACTIVE_RATE_THRESHOLD_HZ,
) -> float:
    """Fraction of electrodes firing above ``threshold_hz``."""
    return float(np.mean(recording.channel_rates() > threshold_hz))
