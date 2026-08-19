"""Figure generation (SPEC §11: every figure regenerable from a single command).

Figures are registered by name here rather than living in ad-hoc scripts, so that
``culture-sim report`` can regenerate all of them and so no figure in the report can
come from code that is no longer in the repo.

Each figure function takes whatever inputs it needs and returns a
:class:`matplotlib.figure.Figure`; :func:`save_figure` handles the output path and
records the git commit in the image metadata.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .config import REPO_ROOT
from .stats.spiketrains import SpikeRecording

__all__ = [
    "DEFAULT_FIGURE_DIR",
    "FIGURES",
    "register",
    "save_figure",
    "apply_style",
    "raster_panel",
    "population_rate_panel",
]

DEFAULT_FIGURE_DIR = REPO_ROOT / "figures"

FIGURES: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        if name in FIGURES:
            raise ValueError(f"figure {name!r} is already registered")
        FIGURES[name] = function
        return function

    return decorator


def apply_style() -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display in a subprocess or on CI
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save_figure(figure: Any, name: str, directory: Path | None = None) -> Path:
    """Write a figure to ``figures/<name>.png`` with provenance in its metadata."""
    from .manifest import git_commit

    directory = Path(directory) if directory is not None else DEFAULT_FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    figure.savefig(path, metadata={"Software": f"culture-sim @ {git_commit()}"})
    return path


# -- reusable panels ------------------------------------------------------


def raster_panel(
    axis: Any,
    recording: SpikeRecording,
    *,
    max_channels: int = 200,
    max_spikes: int = 120_000,
    title: str = "",
    marker_size: float = 0.35,
) -> None:
    """Spike raster. Subsamples channels and spikes so the PNG stays legible.

    Subsampling is by channel rather than by time, so the burst structure -- which is
    what the figure is for -- survives intact.
    """
    channels, times = recording.channels, recording.times
    if recording.n_channels > max_channels:
        step = int(np.ceil(recording.n_channels / max_channels))
        keep = (channels % step) == 0
        channels, times = channels[keep] // step, times[keep]
    if times.size > max_spikes:
        stride = int(np.ceil(times.size / max_spikes))
        channels, times = channels[::stride], times[::stride]

    axis.plot(times, channels, "|", markersize=marker_size * 8, markeredgewidth=0.4, color="0.15")
    axis.set_ylabel("channel")
    axis.set_xlim(0, recording.duration)
    axis.grid(False)
    if title:
        axis.set_title(title, loc="left")


def population_rate_panel(
    axis: Any,
    recording: SpikeRecording,
    *,
    bin_s: float = 0.025,
    color: str = "0.15",
    label: str = "",
) -> np.ndarray:
    """Array-wide firing rate in Hz per bin. Returns the binned counts."""
    n_bins = max(1, int(np.ceil(recording.duration / bin_s)))
    index = np.minimum((recording.times / bin_s).astype(np.int64), n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    edges = np.arange(n_bins) * bin_s
    axis.fill_between(edges, counts / bin_s, step="post", color=color, alpha=0.85, label=label)
    axis.set_ylabel("population rate (Hz)")
    axis.set_xlim(0, recording.duration)
    return counts
