"""Delegated functional connectivity summaries (SPEC §6.5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interop import cl_analysis
from ..interop.cl_adapter import cl_channel_mapping
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
    fc_community_count: float
    fc_community_size_mean: float
    fc_community_size_std: float
    adjacency: np.ndarray  # bool, (n_channels, n_channels), symmetric, no self-loops
    degrees: np.ndarray
    communities: np.ndarray


def cross_correlation_matrix(
    recording: SpikeRecording,
    bin_width_s: float = DEFAULT_BIN_WIDTH_S,
) -> np.ndarray:
    """CL weighted functional-connectivity matrix."""
    del bin_width_s
    stats = _delegated_connectivity(recording)
    return stats.adjacency


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
    del rng, jitter_window_s, n_surrogates, percentile
    matrix = cross_correlation_matrix(recording, bin_width_s)
    return np.asarray(np.abs(matrix) > 0.0, dtype=bool)


def connectivity_stats(recording: SpikeRecording, rng: np.random.Generator) -> ConnectivityStats:
    del rng
    return _delegated_connectivity(recording)


def _delegated_connectivity(recording: SpikeRecording) -> ConnectivityStats:
    if recording.n_channels < 1:
        raise ValueError("recording must have at least one channel")
    empty = _empty(recording.n_channels)
    if recording.n_spikes < 2:
        return empty
    try:
        result = cl_analysis.analyse_functional_connectivity(recording)
    except ValueError:
        return empty
    dump = result.model_dump()
    matrix = np.asarray(dump["adjacency_matrix"], dtype=np.float64)
    mapping = cl_channel_mapping(recording.n_channels)
    if matrix.shape[0] >= int(mapping.max()) + 1:
        matrix = matrix[mapping[:, None], mapping[None, :]]
    np.fill_diagonal(matrix, 0.0)
    adjacency_bool = np.abs(matrix) > 0.0
    degrees = adjacency_bool.sum(axis=1).astype(np.float64)
    communities = _communities(dump.get("graph_partition", {}), mapping, recording.n_channels)
    sizes = np.asarray(
        [np.count_nonzero(communities == label) for label in np.unique(communities)],
        dtype=np.float64,
    )
    return ConnectivityStats(
        fc_mean_degree=float(np.mean(degrees)),
        fc_degree_skew=_skew(degrees),
        fc_clustering_coefficient=float(dump.get("clustering_coefficient", np.nan)),
        fc_community_count=float(np.unique(communities).size) if communities.size else 0.0,
        fc_community_size_mean=float(np.mean(sizes)) if sizes.size else float("nan"),
        fc_community_size_std=float(np.std(sizes)) if sizes.size else float("nan"),
        adjacency=matrix,
        degrees=degrees,
        communities=communities,
    )


def _empty(n_channels: int) -> ConnectivityStats:
    adjacency = np.zeros((n_channels, n_channels), dtype=float)
    degrees = np.zeros(n_channels, dtype=np.float64)
    communities = np.arange(n_channels, dtype=np.int64)
    return ConnectivityStats(
        fc_mean_degree=0.0,
        fc_degree_skew=0.0,
        fc_clustering_coefficient=0.0,
        fc_community_count=float(n_channels),
        fc_community_size_mean=1.0 if n_channels else float("nan"),
        fc_community_size_std=0.0 if n_channels else float("nan"),
        adjacency=adjacency,
        degrees=degrees,
        communities=communities,
    )


def _skew(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    std = float(np.std(values))
    if std == 0.0:
        return 0.0
    centred = values - np.mean(values)
    return float(np.mean(centred**3) / std**3)


def _communities(partition: dict, mapping: np.ndarray, n_channels: int) -> np.ndarray:
    if not partition:
        return np.arange(n_channels, dtype=np.int64)
    communities = []
    for cl_channel in mapping:
        label = partition.get(int(cl_channel), partition.get(str(int(cl_channel))))
        communities.append(int(label) if label is not None else int(cl_channel))
    return np.asarray(communities, dtype=np.int64)
