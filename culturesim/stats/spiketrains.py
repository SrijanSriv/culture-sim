"""Canonical spike data structure (SPEC §3).

Every source of spikes -- Brian2 simulation output after the virtual MEA, and
every real dataset loader -- is converted to :class:`SpikeRecording` before any
statistic is computed. Nothing downstream of this module is allowed to know
whether it is looking at a simulation or a culture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SpikeRecording",
    "save_recording",
    "load_recording",
]

HDF5_SCHEMA_VERSION = 1


def _frozen_array(values: Any, dtype: np.dtype | str) -> np.ndarray:
    """Return a contiguous read-only copy, so a 'frozen' dataclass really is."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {arr.shape}")
    arr = arr.copy()
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class SpikeRecording:
    """Canonical spike data. Units: seconds.

    Attributes
    ----------
    times:
        Spike times in seconds, float64, sorted ascending.
    channels:
        Electrode index per spike, int32, in ``[0, n_channels)``.
    n_channels:
        Number of electrodes in the recording, including silent and dead ones.
    duration:
        Recording length in seconds. Rates are always computed against this,
        never against ``times[-1] - times[0]``.
    source:
        ``'simulation'`` or a dataset identifier.
    metadata:
        Free-form provenance: DIV, culture id, sampling rate, electrode
        geometry, etc. Must be JSON-serialisable so it survives the HDF5
        round-trip.
    """

    times: np.ndarray
    channels: np.ndarray
    n_channels: int
    duration: float
    source: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", _frozen_array(self.times, np.float64))
        object.__setattr__(self, "channels", _frozen_array(self.channels, np.int32))
        object.__setattr__(self, "n_channels", int(self.n_channels))
        object.__setattr__(self, "duration", float(self.duration))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", dict(self.metadata))

        if self.times.size != self.channels.size:
            raise ValueError(
                f"times and channels must be the same length, got "
                f"{self.times.size} and {self.channels.size}"
            )
        if self.n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {self.n_channels}")
        if not np.isfinite(self.duration) or self.duration <= 0:
            raise ValueError(f"duration must be finite and positive, got {self.duration}")
        if self.times.size:
            if not np.all(np.isfinite(self.times)):
                raise ValueError("times must all be finite")
            if np.any(np.diff(self.times) < 0):
                raise ValueError("times must be sorted ascending")
            if self.times[0] < 0.0:
                raise ValueError(f"times must be non-negative, got {self.times[0]}")
            if self.times[-1] > self.duration:
                raise ValueError(
                    f"last spike at {self.times[-1]} s exceeds duration {self.duration} s"
                )
            if np.any(self.channels < 0) or np.any(self.channels >= self.n_channels):
                raise ValueError(
                    f"channels must lie in [0, {self.n_channels}), got range "
                    f"[{self.channels.min()}, {self.channels.max()}]"
                )

    # -- basic properties -------------------------------------------------

    @property
    def n_spikes(self) -> int:
        return int(self.times.size)

    @property
    def mean_rate(self) -> float:
        """Array-wide spikes per second, pooled over all channels."""
        return self.n_spikes / self.duration

    def spike_counts(self) -> np.ndarray:
        """Spike count per channel, shape ``(n_channels,)``."""
        return np.bincount(self.channels, minlength=self.n_channels).astype(np.int64)

    def channel_rates(self) -> np.ndarray:
        """Firing rate in Hz per channel, shape ``(n_channels,)``."""
        return self.spike_counts() / self.duration

    def times_of(self, channel: int) -> np.ndarray:
        """Sorted spike times for one channel."""
        if not 0 <= channel < self.n_channels:
            raise IndexError(f"channel {channel} out of range [0, {self.n_channels})")
        return self.times[self.channels == channel]

    def by_channel(self) -> list[np.ndarray]:
        """Spike times split per channel; silent channels give empty arrays."""
        order = np.argsort(self.channels, kind="stable")
        sorted_channels = self.channels[order]
        sorted_times = self.times[order]
        bounds = np.searchsorted(sorted_channels, np.arange(self.n_channels + 1))
        return [sorted_times[bounds[c] : bounds[c + 1]] for c in range(self.n_channels)]

    # -- derived recordings -----------------------------------------------

    def time_slice(self, t_start: float, t_stop: float) -> SpikeRecording:
        """A new recording covering ``[t_start, t_stop)``, times re-zeroed.

        The interval is half-open so that adjacent slices partition the spikes, except
        at the end of the recording, where it closes: a spike at exactly ``duration``
        is legal, and a half-open final window would silently discard it.
        """
        if not 0.0 <= t_start < t_stop:
            raise ValueError(f"require 0 <= t_start < t_stop, got {t_start}, {t_stop}")
        lo = int(np.searchsorted(self.times, t_start, side="left"))
        hi = int(
            np.searchsorted(self.times, t_stop, side="right" if t_stop >= self.duration else "left")
        )
        return replace(
            self,
            times=self.times[lo:hi] - t_start,
            channels=self.channels[lo:hi],
            duration=min(t_stop, self.duration) - t_start,
        )

    def drop_channels(self, dead: np.ndarray) -> SpikeRecording:
        """Blank the given channels while keeping ``n_channels`` unchanged.

        Channel indices are preserved deliberately: electrode identity is
        geometric, so re-indexing after removing dead electrodes would corrupt
        every distance-dependent statistic.
        """
        dead_mask = np.zeros(self.n_channels, dtype=bool)
        dead_mask[np.asarray(dead, dtype=np.int64)] = True
        keep = ~dead_mask[self.channels]
        return replace(self, times=self.times[keep], channels=self.channels[keep])

    # -- persistence (SPEC §12: round-trip must be the identity) -----------

    def to_hdf5(self, path: str | Path, *, compression: str | None = "gzip") -> Path:
        """Write the recording using the CL recording H5 format.

        ``compression`` is accepted for the legacy API but ignored: the CL SDK writer
        owns the physical layout of the native on-disk representation.
        """
        del compression
        from ..interop.cl_adapter import to_cl_h5

        return to_cl_h5(self, path)

    @classmethod
    def from_hdf5(cls, path: str | Path) -> SpikeRecording:
        import h5py

        with h5py.File(Path(path), "r") as handle:
            if "spikes" in handle and "channel_count" in handle.attrs:
                from ..interop.cl_adapter import from_cl_h5

                return from_cl_h5(path)
            group = handle["spike_recording"]
            version = int(group.attrs.get("schema_version", -1))
            if version != HDF5_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported SpikeRecording schema version {version}, "
                    f"expected {HDF5_SCHEMA_VERSION}"
                )
            source = group.attrs["source"]
            return cls(
                times=group["times"][:],
                channels=group["channels"][:],
                n_channels=int(group.attrs["n_channels"]),
                duration=float(group.attrs["duration"]),
                source=source.decode() if isinstance(source, bytes) else str(source),
                metadata=json.loads(group.attrs["metadata_json"]),
            )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SpikeRecording):
            return NotImplemented
        return (
            self.n_channels == other.n_channels
            and self.duration == other.duration
            and self.source == other.source
            and self.metadata == other.metadata
            and np.array_equal(self.times, other.times)
            and np.array_equal(self.channels, other.channels)
        )

    def __repr__(self) -> str:
        return (
            f"SpikeRecording(source={self.source!r}, n_spikes={self.n_spikes}, "
            f"n_channels={self.n_channels}, duration={self.duration:.3f}s, "
            f"mean_rate={self.mean_rate:.3f}Hz)"
        )


def save_recording(recording: SpikeRecording, path: str | Path) -> Path:
    return recording.to_hdf5(path)


def load_recording(path: str | Path) -> SpikeRecording:
    return SpikeRecording.from_hdf5(path)
