"""Fingerprint vector and its frozen order (SPEC §3, §6.6).

The fingerprint is a fixed-order vector of summary statistics. Its order is
declared in ``configs/fingerprint.yaml`` and expanded here deterministically:
per group, scalars first, then quantiles, then histogram bins.

SPEC §3: the ``names`` tuple freezes once Task 3 is complete. After that, adding
a statistic invalidates every fit made against the old vector, so the only legal
move is to bump ``version`` and re-run. ``names_sha256`` in the config makes an
accidental edit a test failure rather than a silent invalidation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from .spiketrains import SpikeRecording

__all__ = [
    "Fingerprint",
    "FingerprintSpec",
    "HistogramSpec",
    "compute_fingerprint",
]


@dataclass(frozen=True)
class HistogramSpec:
    """Fixed bin edges for a distributional component of the fingerprint."""

    stat: str
    edges: np.ndarray
    scale: str
    normalize: str

    @classmethod
    def from_dict(cls, entry: Mapping[str, Any]) -> HistogramSpec:
        scale = str(entry.get("scale", "log10"))
        n_bins = int(entry["n_bins"])
        start, stop = float(entry["start"]), float(entry["stop"])
        if n_bins < 1:
            raise ValueError(f"histogram {entry['stat']} needs at least one bin")
        if stop <= start:
            raise ValueError(f"histogram {entry['stat']} needs stop > start")
        if scale == "log10":
            edges = np.logspace(start, stop, n_bins + 1)
        elif scale == "linear":
            edges = np.linspace(start, stop, n_bins + 1)
        else:
            raise ValueError(f"unknown histogram scale {scale!r}, expected log10 or linear")
        edges.setflags(write=False)
        return cls(
            stat=str(entry["stat"]),
            edges=edges,
            scale=scale,
            normalize=str(entry.get("normalize", "density")),
        )

    @property
    def n_bins(self) -> int:
        return int(self.edges.size - 1)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"{self.stat}_hist_{i:02d}" for i in range(self.n_bins))


@dataclass(frozen=True)
class FingerprintSpec:
    """The frozen statistic order, weights, and histogram binning."""

    version: str
    frozen: bool
    names: tuple[str, ...]
    weights: np.ndarray
    groups: tuple[str, ...]
    group_of: Mapping[str, str]
    histograms: tuple[HistogramSpec, ...]
    quantile_levels: Mapping[str, tuple[float, ...]]
    undefined_value: float
    declared_sha256: str | None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> FingerprintSpec:
        default_levels = tuple(float(q) for q in config.get("default_quantile_levels", ()))
        names: list[str] = []
        weights: list[float] = []
        group_names: list[str] = []
        group_of: dict[str, str] = {}
        histograms: list[HistogramSpec] = []
        quantile_levels: dict[str, tuple[float, ...]] = {}

        for group in config.get("groups", ()):
            group_name = str(group["name"])
            group_names.append(group_name)
            group_weight = float(group.get("weight", 1.0))

            entries: list[str] = [str(s) for s in group.get("scalars", ())]

            for quantile_entry in group.get("quantiles", ()):
                stat = str(quantile_entry["stat"])
                levels = tuple(float(q) for q in quantile_entry.get("levels", default_levels))
                if not levels:
                    raise ValueError(f"quantile entry {stat} has no levels and no default")
                quantile_levels[stat] = levels
                entries.extend(f"{stat}_p{int(round(q)):02d}" for q in levels)

            for hist_entry in group.get("histograms", ()):
                histogram = HistogramSpec.from_dict(hist_entry)
                histograms.append(histogram)
                entries.extend(histogram.names)

            for name in entries:
                if name in group_of:
                    raise ValueError(f"duplicate fingerprint statistic {name!r}")
                group_of[name] = group_name
                names.append(name)
                weights.append(group_weight)

        if not names:
            raise ValueError("fingerprint config declares no statistics")

        weight_array = np.asarray(weights, dtype=np.float64)
        weight_array.setflags(write=False)
        declared = config.get("names_sha256")
        return cls(
            version=str(config.get("version", "unversioned")),
            frozen=bool(config.get("frozen", False)),
            names=tuple(names),
            weights=weight_array,
            groups=tuple(group_names),
            group_of=group_of,
            histograms=tuple(histograms),
            quantile_levels=quantile_levels,
            undefined_value=float(config.get("undefined_value", np.nan)),
            declared_sha256=None if declared is None else str(declared),
        )

    @classmethod
    def load(cls, path: str | Path = "fingerprint.yaml") -> FingerprintSpec:
        spec = cls.from_config(load_config(path))
        spec.check_freeze()
        return spec

    def __len__(self) -> int:
        return len(self.names)

    @property
    def names_sha256(self) -> str:
        return hashlib.sha256("\n".join(self.names).encode("utf-8")).hexdigest()

    def check_freeze(self) -> None:
        """Refuse to run against a frozen spec whose order has been edited."""
        if not self.frozen:
            return
        if self.declared_sha256 is None:
            raise ValueError(
                f"fingerprint {self.version} is marked frozen but declares no "
                "names_sha256; run scripts/freeze_fingerprint.py"
            )
        if self.declared_sha256 != self.names_sha256:
            raise ValueError(
                f"fingerprint {self.version} order has changed: config declares "
                f"{self.declared_sha256[:12]} but the expanded names hash to "
                f"{self.names_sha256[:12]}. Editing a frozen fingerprint "
                "invalidates every fit made against it (SPEC §3) -- bump the "
                "version and re-run, do not silently re-freeze."
            )

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"{name!r} is not a fingerprint statistic") from exc

    def group_mask(self, group: str) -> np.ndarray:
        """Boolean mask selecting one group's entries -- used by held-out validation."""
        if group not in self.groups:
            raise KeyError(f"unknown fingerprint group {group!r}, have {self.groups}")
        return np.array([self.group_of[n] == group for n in self.names], dtype=bool)

    def histogram_for(self, stat: str) -> HistogramSpec:
        for histogram in self.histograms:
            if histogram.stat == stat:
                return histogram
        raise KeyError(f"no histogram declared for {stat!r}")

    def empty_values(self) -> np.ndarray:
        """A vector of the undefined sentinel, for statistics that cannot be computed."""
        return np.full(len(self.names), self.undefined_value, dtype=np.float64)


@dataclass(frozen=True)
class Fingerprint:
    """Fixed-order summary statistic vector."""

    values: np.ndarray
    names: tuple[str, ...]
    version: str = "unversioned"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.ascontiguousarray(np.asarray(self.values, dtype=np.float64)).copy()
        if values.ndim != 1:
            raise ValueError(f"fingerprint values must be 1-D, got shape {values.shape}")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", tuple(str(n) for n in self.names))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if values.size != len(self.names):
            raise ValueError(f"fingerprint has {values.size} values but {len(self.names)} names")
        if len(set(self.names)) != len(self.names):
            raise ValueError("fingerprint names must be unique")

    def __len__(self) -> int:
        return int(self.values.size)

    def __getitem__(self, name: str) -> float:
        try:
            return float(self.values[self.names.index(name)])
        except ValueError as exc:
            raise KeyError(f"{name!r} is not in this fingerprint") from exc

    def __iter__(self) -> Iterator[tuple[str, float]]:
        return zip(self.names, (float(v) for v in self.values), strict=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "names": list(self.names),
            # None rather than NaN: NaN is not valid JSON and json.dump would
            # emit a bare `NaN` token that strict parsers reject.
            "values": [None if not np.isfinite(v) else float(v) for v in self.values],
            "non_finite": {
                name: _describe_non_finite(value)
                for name, value in zip(self.names, self.values, strict=True)
                if not np.isfinite(value)
            },
            "metadata": self.metadata,
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def read_json(cls, path: str | Path) -> Fingerprint:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        non_finite = payload.get("non_finite", {})
        names = [str(n) for n in payload["names"]]
        if len(names) != len(payload["values"]):
            raise ValueError(f"{path} has {len(payload['values'])} values for {len(names)} names")
        values = [
            _parse_non_finite(non_finite.get(name, "nan")) if raw is None else float(raw)
            for name, raw in zip(names, payload["values"], strict=True)
        ]
        return cls(
            values=np.asarray(values, dtype=np.float64),
            names=tuple(names),
            version=str(payload.get("version", "unversioned")),
            metadata=payload.get("metadata", {}),
        )

    @property
    def n_undefined(self) -> int:
        return int(np.count_nonzero(~np.isfinite(self.values)))

    def matches(self, spec: FingerprintSpec) -> bool:
        return self.names == spec.names and self.version == spec.version

    def require_match(self, spec: FingerprintSpec) -> None:
        if self.names != spec.names:
            raise ValueError(
                "fingerprint order does not match the spec: comparing these "
                "vectors would compare different statistics to each other"
            )
        if self.version != spec.version:
            raise ValueError(
                f"fingerprint version mismatch: vector is {self.version!r}, "
                f"spec is {spec.version!r}"
            )

    def __repr__(self) -> str:
        return (
            f"Fingerprint(version={self.version!r}, n_stats={len(self)}, "
            f"n_undefined={self.n_undefined})"
        )


def _describe_non_finite(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _parse_non_finite(token: str) -> float:
    return {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[token]


def compute_fingerprint(
    recording: SpikeRecording,
    spec: FingerprintSpec | None = None,
) -> Fingerprint:
    """Assemble the full fingerprint from a recording (SPEC §6.6)."""
    from .avalanche import avalanche_stats
    from .branching import mr_branching_ratio
    from .bursts import burst_stats
    from .connectivity import connectivity_stats
    from .rates import rate_stats

    spec = FingerprintSpec.load() if spec is None else spec
    if recording.metadata.get("observation") == "none":
        raise ValueError(
            "fingerprint requires electrode-level data; this recording is tagged "
            "observation='none' (neuron-level). Pass it through the virtual MEA "
            "first (SPEC §5, §14)."
        )
    rng = np.random.default_rng(0)

    rates = rate_stats(recording)
    bursts = burst_stats(recording, rng)
    avalanches = avalanche_stats(recording)
    branching = mr_branching_ratio(recording)
    connectivity = connectivity_stats(recording, rng)

    scalars: dict[str, float] = {}
    distributions: dict[str, np.ndarray] = {
        "ibi_seconds": bursts.ibi_seconds,
        "avalanche_size": avalanches.avalanches.sizes,
        "avalanche_duration": avalanches.avalanches.durations,
    }
    for stats_obj in (rates, bursts, avalanches, branching, connectivity):
        for key, value in vars(stats_obj).items():
            if key in distributions or isinstance(value, np.ndarray):
                continue
            try:
                scalars[key] = float(value)
            except (TypeError, ValueError):
                continue

    values = []
    for name in spec.names:
        if name in scalars:
            values.append(scalars[name])
            continue
        quantile = _quantile_name(name, spec.quantile_levels)
        if quantile is not None:
            stat, level = quantile
            values.append(_quantile(distributions.get(stat), level, spec.undefined_value))
            continue
        histogram = _histogram_name(name, spec.histograms)
        if histogram is not None:
            stat, index = histogram
            values.append(
                _histogram_value(distributions.get(stat), spec.histogram_for(stat), index)
            )
            continue
        values.append(spec.undefined_value)

    return Fingerprint(
        values=np.asarray(values, dtype=np.float64),
        names=spec.names,
        version=spec.version,
        metadata={"source": recording.source, "n_spikes": recording.n_spikes},
    )


def _quantile_name(
    name: str,
    quantile_levels: Mapping[str, tuple[float, ...]],
) -> tuple[str, float] | None:
    for stat, levels in quantile_levels.items():
        prefix = f"{stat}_p"
        if not name.startswith(prefix):
            continue
        suffix = name.removeprefix(prefix)
        for level in levels:
            if suffix == f"{int(round(level)):02d}":
                return stat, level
    return None


def _quantile(values: np.ndarray | None, level: float, undefined: float) -> float:
    if values is None:
        return undefined
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, level)) if values.size else undefined


def _histogram_name(
    name: str,
    histograms: tuple[HistogramSpec, ...],
) -> tuple[str, int] | None:
    for histogram in histograms:
        prefix = f"{histogram.stat}_hist_"
        if name.startswith(prefix):
            return histogram.stat, int(name.removeprefix(prefix))
    return None


def _histogram_value(values: np.ndarray | None, histogram: HistogramSpec, index: int) -> float:
    if values is None:
        return float("nan")
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return float("nan")
    density = histogram.normalize == "density"
    counts, _ = np.histogram(values, bins=histogram.edges, density=density)
    return float(counts[index])
