"""Virtual MEA: neuron-level spikes -> electrode-level recording (SPEC §5).

This module is the scientific crux of the project. Statistics must be computed on
simulated data that has passed through the same observational bottleneck as the real
data. Computing statistics over all 1000 simulated neurons and comparing them to
statistics from 60 electrodes is invalid, and it is the single most common error in
this literature (SPEC §5, §14).

The pipeline, in order:

1. Place electrodes on a regular grid matching the target array's geometry.
2. Each neuron's spike produces an amplitude at each electrode, ``A = A_0 / (1 + (d/d_0)**2)``.
3. Add Gaussian recording noise at the real system's RMS.
4. Detect a spike on an electrode when ``A > k * noise_rms``.
5. Apply a per-electrode dead time for detector refractoriness.
6. Optionally mark a fraction of electrodes dead -- real arrays always have some.

Consequences worth stating, because they are the point rather than side effects: one
neuron near several electrodes is counted several times, one electrode near several
neurons merges them, and quiet or distant neurons are never seen at all. That is what
a real MEA does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..stats.spiketrains import SpikeRecording

__all__ = [
    "ElectrodeLayout",
    "ObservationConfig",
    "observe",
    "detection_radius_um",
]


@dataclass(frozen=True)
class ElectrodeLayout:
    """Electrode positions in micrometres, centred on the culture sheet."""

    name: str
    x_um: np.ndarray
    y_um: np.ndarray
    pitch_um: float
    description: str = ""

    def __post_init__(self) -> None:
        for attr in ("x_um", "y_um"):
            arr = np.ascontiguousarray(np.asarray(getattr(self, attr), dtype=np.float64))
            arr.setflags(write=False)
            object.__setattr__(self, attr, arr)
        if self.x_um.shape != self.y_um.shape:
            raise ValueError("electrode x and y arrays must have the same shape")
        if self.x_um.size == 0:
            raise ValueError("layout has no electrodes")

    @property
    def n_electrodes(self) -> int:
        return int(self.x_um.size)

    @property
    def extent_um(self) -> tuple[float, float]:
        return (
            float(self.x_um.max() - self.x_um.min()),
            float(self.y_um.max() - self.y_um.min()),
        )

    @classmethod
    def grid(
        cls,
        name: str,
        n_rows: int,
        n_cols: int,
        pitch_um: float,
        *,
        omit_corners: bool = False,
        center: tuple[float, float] = (0.0, 0.0),
        description: str = "",
    ) -> ElectrodeLayout:
        """A regular grid, optionally without its four corners (the MCS 60 layout)."""
        if n_rows < 1 or n_cols < 1:
            raise ValueError(f"grid needs positive dimensions, got {n_rows}x{n_cols}")
        if pitch_um <= 0:
            raise ValueError(f"pitch must be positive, got {pitch_um}")
        rows = (np.arange(n_rows) - (n_rows - 1) / 2.0) * pitch_um + center[1]
        cols = (np.arange(n_cols) - (n_cols - 1) / 2.0) * pitch_um + center[0]
        xx, yy = np.meshgrid(cols, rows, indexing="xy")
        # Row-major so electrode index maps to the array's own channel ordering.
        x_flat, y_flat = xx.ravel(), yy.ravel()
        if omit_corners:
            if n_rows < 2 or n_cols < 2:
                raise ValueError("cannot omit corners from a grid narrower than 2x2")
            keep = np.ones(x_flat.size, dtype=bool)
            for row in (0, n_rows - 1):
                for col in (0, n_cols - 1):
                    keep[row * n_cols + col] = False
            x_flat, y_flat = x_flat[keep], y_flat[keep]
        return cls(
            name=name,
            x_um=x_flat,
            y_um=y_flat,
            pitch_um=float(pitch_um),
            description=description,
        )

    @classmethod
    def from_config(cls, name: str, entry: Mapping[str, Any]) -> ElectrodeLayout:
        kind = str(entry.get("kind", "grid"))
        if kind != "grid":
            raise ValueError(f"unsupported layout kind {kind!r}; only 'grid' is implemented")
        layout = cls.grid(
            name=name,
            n_rows=int(entry["n_rows"]),
            n_cols=int(entry["n_cols"]),
            pitch_um=float(entry["pitch_um"]),
            omit_corners=bool(entry.get("omit_corners", False)),
            description=str(entry.get("description", "")),
        )
        declared = entry.get("n_electrodes")
        if declared is not None and int(declared) != layout.n_electrodes:
            raise ValueError(
                f"layout {name!r} declares n_electrodes={declared} but its geometry "
                f"yields {layout.n_electrodes}; the declared count is what downstream "
                "configs and figures assume, so this must be fixed rather than ignored"
            )
        return layout

    def to_metadata(self) -> dict[str, Any]:
        """Geometry recorded into the SpikeRecording metadata.

        Every distance-dependent statistic depends on this, so it travels with the
        recording rather than being looked up again later.
        """
        return {
            "layout": self.name,
            "description": self.description,
            "n_electrodes": self.n_electrodes,
            "pitch_um": self.pitch_um,
            "x_um": self.x_um.tolist(),
            "y_um": self.y_um.tolist(),
        }


@dataclass(frozen=True)
class ObservationConfig:
    """Parsed ``configs/observation.yaml`` (SPEC §5)."""

    layout: ElectrodeLayout
    A_0_uv: float = 150.0
    d_0_um: float = 25.0
    per_neuron_sigma: float = 0.5
    noise_rms_uv: float = 3.0
    threshold_k: float = 5.0
    dead_time_ms: float = 2.0
    dead_electrode_fraction: float = 0.05
    dead_electrode_indices: tuple[int, ...] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.A_0_uv <= 0 or self.d_0_um <= 0:
            raise ValueError("amplitude parameters must be positive")
        if self.noise_rms_uv <= 0:
            raise ValueError("noise RMS must be positive; a noiseless MEA is not the model")
        if self.threshold_k <= 0:
            raise ValueError("threshold_k must be positive")
        if not 0.0 <= self.dead_electrode_fraction < 1.0:
            raise ValueError(
                f"dead_electrode_fraction must be in [0, 1), got {self.dead_electrode_fraction}"
            )
        if self.dead_time_ms < 0:
            raise ValueError("dead_time_ms must be non-negative")
        if self.dead_electrode_indices is not None:
            object.__setattr__(
                self, "dead_electrode_indices", tuple(int(i) for i in self.dead_electrode_indices)
            )

    @property
    def threshold_uv(self) -> float:
        return self.threshold_k * self.noise_rms_uv

    @property
    def detection_radius_um(self) -> float:
        """Distance at which a mean-amplitude spike falls to the detection threshold."""
        return detection_radius_um(self.A_0_uv, self.d_0_um, self.threshold_uv)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        layout_name: str | None = None,
    ) -> ObservationConfig:
        name = layout_name or str(config["active_layout"])
        layouts = config.get("layouts", {})
        if name not in layouts:
            raise KeyError(f"layout {name!r} not in observation config; have {sorted(layouts)}")
        amplitude = config.get("amplitude", {})
        noise = config.get("noise", {})
        detector = config.get("detector", {})
        dead_indices = detector.get("dead_electrode_indices")
        return cls(
            layout=ElectrodeLayout.from_config(name, layouts[name]),
            A_0_uv=float(amplitude.get("A_0_uv", 150.0)),
            d_0_um=float(amplitude.get("d_0_um", 25.0)),
            per_neuron_sigma=float(amplitude.get("per_neuron_sigma", 0.5)),
            noise_rms_uv=float(noise.get("rms_uv", 3.0)),
            threshold_k=float(noise.get("threshold_k", 5.0)),
            dead_time_ms=float(detector.get("dead_time_ms", 2.0)),
            dead_electrode_fraction=float(detector.get("dead_electrode_fraction", 0.05)),
            dead_electrode_indices=None if dead_indices is None else tuple(dead_indices),
            raw=dict(config),
        )

    @classmethod
    def load(
        cls,
        path: str | Path = "observation.yaml",
        layout_name: str | None = None,
    ) -> ObservationConfig:
        return cls.from_config(load_config(path), layout_name)


def detection_radius_um(A_0_uv: float, d_0_um: float, threshold_uv: float) -> float:
    """Distance where ``A_0 / (1 + (d/d_0)**2)`` equals ``threshold_uv``.

    Infinite if the threshold is below... nothing: if ``threshold >= A_0`` the spike is
    never detectable even at zero distance, and this returns 0.
    """
    if threshold_uv >= A_0_uv:
        return 0.0
    return float(d_0_um * np.sqrt(A_0_uv / threshold_uv - 1.0))


def observe(
    spike_times_s: np.ndarray,
    spike_neuron_ids: np.ndarray,
    neuron_x_um: np.ndarray,
    neuron_y_um: np.ndarray,
    duration_s: float,
    config: ObservationConfig,
    rng: np.random.Generator,
    *,
    source: str = "simulation",
    metadata: Mapping[str, Any] | None = None,
) -> SpikeRecording:
    """Push neuron-level spikes through the observation model (SPEC §5).

    Returns a recording with ``n_channels == config.layout.n_electrodes``. Everything
    downstream sees only this.
    """
    raise NotImplementedError("Task 2 (SPEC §5)")
