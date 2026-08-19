"""Thin wrappers around CL SDK analysis (SPEC §6.0.2)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from ..stats.spiketrains import SpikeRecording
from .cl_adapter import to_cl_h5

DEFAULT_BURST_BIN_SIZE_S = 0.05
DEFAULT_BURST_ONSET_HZ = 3.0
DEFAULT_BURST_OFFSET_HZ = 1.0
DEFAULT_MIN_ACTIVE_CHANNELS = 3
DEFAULT_CRITICALITY_BIN_SIZE_S = 0.05
DEFAULT_CRITICALITY_PERCENTILE = 0.95
DEFAULT_CONNECTIVITY_BIN_SIZE_S = 0.05
DEFAULT_CONNECTIVITY_THRESHOLD = 0.6

__all__ = [
    "DEFAULT_BURST_BIN_SIZE_S",
    "DEFAULT_BURST_ONSET_HZ",
    "DEFAULT_BURST_OFFSET_HZ",
    "DEFAULT_MIN_ACTIVE_CHANNELS",
    "DEFAULT_CRITICALITY_BIN_SIZE_S",
    "DEFAULT_CRITICALITY_PERCENTILE",
    "DEFAULT_CONNECTIVITY_BIN_SIZE_S",
    "DEFAULT_CONNECTIVITY_THRESHOLD",
    "recording_view",
    "analyse_network_bursts",
    "analyse_criticality",
    "analyse_functional_connectivity",
]


def recording_view(path: str | Path):
    """Open a CL recording view; caller is responsible for closing it."""
    from cl.util.recording_view import RecordingView

    return RecordingView(str(path))


def _with_recording_path(recording: SpikeRecording, fn):
    with tempfile.TemporaryDirectory(prefix="culturesim_cl_") as tmp:
        path = to_cl_h5(recording, Path(tmp) / "recording.h5")
        with recording_view(path) as view:
            return fn(view)


def analyse_network_bursts(
    recording: SpikeRecording,
    *,
    bin_size_sec: float = DEFAULT_BURST_BIN_SIZE_S,
    onset_freq_hz: float = DEFAULT_BURST_ONSET_HZ,
    offset_freq_hz: float = DEFAULT_BURST_OFFSET_HZ,
    min_active_channels: int | None = DEFAULT_MIN_ACTIVE_CHANNELS,
) -> Any:
    """Run CL network burst analysis."""

    def run(view):
        # DELEGATED: cl.util.recording_view.RecordingView.analyse_network_bursts
        return view.analyse_network_bursts(
            bin_size_sec=bin_size_sec,
            onset_freq_hz=onset_freq_hz,
            offset_freq_hz=offset_freq_hz,
            min_active_channels=min_active_channels,
        )

    return _with_recording_path(recording, run)


def analyse_criticality(
    recording: SpikeRecording,
    *,
    bin_size_sec: float = DEFAULT_CRITICALITY_BIN_SIZE_S,
    percentile_threshold: float = DEFAULT_CRITICALITY_PERCENTILE,
    n_bootstraps: int = 100,
    random_seed: int = 42,
) -> Any:
    """Run CL criticality analysis and return raw avalanche distributions."""

    def run(view):
        # DELEGATED: cl.util.recording_view.RecordingView.analyse_criticality
        return view.analyse_criticality(
            bin_size_sec=bin_size_sec,
            percentile_threshold=percentile_threshold,
            n_bootstraps=n_bootstraps,
            random_seed=random_seed,
        )

    return _with_recording_path(recording, run)


def analyse_functional_connectivity(
    recording: SpikeRecording,
    *,
    bin_size_sec: float = DEFAULT_CONNECTIVITY_BIN_SIZE_S,
    correlation_threshold: float = DEFAULT_CONNECTIVITY_THRESHOLD,
) -> Any:
    """Run CL functional connectivity analysis."""

    def run(view):
        # DELEGATED: cl.util.recording_view.RecordingView.analyse_functional_connectivity
        return view.analyse_functional_connectivity(
            bin_size_sec=bin_size_sec,
            correlation_threshold=correlation_threshold,
        )

    return _with_recording_path(recording, run)
