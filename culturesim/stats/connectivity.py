"""Functional connectivity from pairwise cross-correlation (SPEC §6.5).

Task 3.

Significance is judged against a jitter-corrected surrogate null rather than an
absolute correlation threshold: spike jitter within a window destroys precise
timing while preserving each electrode's slow rate covariation, so what survives
is genuine short-latency coupling rather than shared burst envelope. Without that
correction, every electrode pair in a bursting culture looks connected.

Note this is a *functional* graph. It is not the model's synaptic connectivity and
must never be compared to it directly -- that comparison is only meaningful with
the same observation model applied to both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spiketrains import SpikeRecording

__all__ = [
    "ConnectivityStats",
    "cross_correlation_matrix",
    "functional_graph",
    "connectivity_stats",
]

DEFAULT_BIN_WIDTH_S = 0.005
DEFAULT_JITTER_WINDOW_S = 0.020
N_SURROGATES = 100
SIGNIFICANCE_PERCENTILE = 99.0


@dataclass(frozen=True)
class ConnectivityStats:
    fc_mean_degree: float
    fc_degree_skew: float
    fc_clustering_coefficient: float
    adjacency: np.ndarray  # bool, (n_channels, n_channels), symmetric, no self-loops
    degrees: np.ndarray


def cross_correlation_matrix(
    recording: SpikeRecording,
    bin_width_s: float = DEFAULT_BIN_WIDTH_S,
) -> np.ndarray:
    """Zero-lag Pearson correlation between binned electrode spike trains.

    Sentinel: NaN entries for electrode pairs where either train has zero variance.
    """
    raise NotImplementedError("Task 3 (SPEC §6.5)")


def functional_graph(
    recording: SpikeRecording,
    rng: np.random.Generator,
    *,
    bin_width_s: float = DEFAULT_BIN_WIDTH_S,
    jitter_window_s: float = DEFAULT_JITTER_WINDOW_S,
    n_surrogates: int = N_SURROGATES,
    percentile: float = SIGNIFICANCE_PERCENTILE,
) -> np.ndarray:
    """Binary adjacency matrix of pairs exceeding the jitter-corrected null."""
    raise NotImplementedError("Task 3 (SPEC §6.5)")


def connectivity_stats(recording: SpikeRecording, rng: np.random.Generator) -> ConnectivityStats:
    raise NotImplementedError("Task 3 (SPEC §6.5)")
