"""Coarse fitting: grid search then local optimisation (SPEC §8.2).

Task 5.

The grid stage over ``w_e``, ``g`` and ``tau_rec`` is not optional. It answers a
question no optimiser can: whether the model can produce the target behaviour *at
all*. If nothing in the grid bursts, that is a structural bug in the network -- and an
optimiser handed a structural bug returns a converged-looking point estimate at the
edge of the parameter box, which reads exactly like a hard fitting problem. The grid
also produces the distance-landscape figure required for Task 5 acceptance.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..model.params import FREE_PARAM_NAMES, FreeParams, ModelParams
from ..stats.fingerprint import Fingerprint, FingerprintSpec, compute_fingerprint
from .distance import ScaleReference, distance

__all__ = [
    "GRID_PARAMETERS",
    "DEFAULT_GRID",
    "COARSE_DURATION_S",
    "GridResult",
    "CoarseFitResult",
    "grid_search",
    "local_optimize",
    "coarse_fit",
    "plot_distance_landscape",
    "scale_from_wagenaar",
]

# SPEC §8.2: the three parameters that dominate whether bursting happens at all.
GRID_PARAMETERS = ("w_e", "g", "tau_rec")

# Coarse-fit biological duration. 300 s is the Task 1 budget target; the grid plus
# Nelder-Mead is O(10^2) simulations, so 60 s is the duration that actually finishes
# on a laptop. Burst/avalanche bins are 50 ms and Task 1 IBIs are ~8 s, so 60 s still
# contains several network events. The 50% acceptance comparison uses this duration
# on both the baseline and the fitted point -- mixing 60 s draws with a 300 s
# baseline would make the improvement look better than it is.
COARSE_DURATION_S = 60.0

# 4 x 3 x 3 = 36 cells. Concentrated where the Task 1 hand-tune actually bursts
# (g must stay low; tau_rec sets IBI; w_e ignites). The prior box is wider; that
# is SBI's job, not the coarse grid's.
DEFAULT_GRID: dict[str, tuple[float, ...]] = {
    "w_e": (0.7, 1.5, 2.2, 3.0),
    "g": (1.5, 2.0, 3.0),
    "tau_rec": (600.0, 2500.0, 4500.0),
}


@dataclass(frozen=True)
class GridResult:
    """Distances over the parameter grid, and the landscape figure's data."""

    parameter_names: tuple[str, ...]
    grid_values: tuple[np.ndarray, ...]
    distances: np.ndarray  # shape = tuple(len(v) for v in grid_values)
    fingerprints: dict[tuple[int, ...], Fingerprint] = field(default_factory=dict)
    n_failed: int = 0

    def __post_init__(self) -> None:
        distances = np.ascontiguousarray(np.asarray(self.distances, dtype=np.float64)).copy()
        distances.setflags(write=False)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "parameter_names", tuple(self.parameter_names))
        object.__setattr__(
            self,
            "grid_values",
            tuple(np.asarray(v, dtype=np.float64) for v in self.grid_values),
        )

    @property
    def best_index(self) -> tuple[int, ...]:
        if not np.any(np.isfinite(self.distances)):
            raise ValueError("grid has no finite distances")
        flat_best = int(np.nanargmin(self.distances))
        return tuple(int(i) for i in np.unravel_index(flat_best, self.distances.shape))

    @property
    def best_distance(self) -> float:
        return float(self.distances[self.best_index])

    @property
    def best_params(self) -> dict[str, float]:
        index = self.best_index
        return {
            name: float(values[i])
            for name, values, i in zip(self.parameter_names, self.grid_values, index, strict=True)
        }

    @property
    def any_finite(self) -> bool:
        """False means every cell crashed or returned NaN -- a structural bug."""
        return bool(np.any(np.isfinite(self.distances)))

    @property
    def any_bursting(self) -> bool:
        """Alias kept for the stub's name; finite distance, not a burst detector."""
        return self.any_finite

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter_names": list(self.parameter_names),
            "grid_values": [v.tolist() for v in self.grid_values],
            "distances": self.distances.tolist(),
            "n_failed": self.n_failed,
            "best_params": self.best_params if self.any_finite else None,
            "best_distance": self.best_distance if self.any_finite else None,
        }


@dataclass(frozen=True)
class CoarseFitResult:
    best: FreeParams
    best_distance: float
    baseline_distance: float  # hand-tuned Task 1 parameters, for the >=50% acceptance test
    grid: GridResult | None
    optimizer: str
    n_evaluations: int
    history: list[tuple[FreeParams, float]] = field(default_factory=list)
    duration_s: float = COARSE_DURATION_S
    scale_n_cultures: int = 0

    @property
    def improvement_fraction(self) -> float:
        if not np.isfinite(self.baseline_distance) or self.baseline_distance == 0:
            return float("nan")
        return 1.0 - self.best_distance / self.baseline_distance

    def to_dict(self) -> dict[str, Any]:
        return {
            "best": self.best.to_dict(),
            "best_distance": self.best_distance,
            "baseline_distance": self.baseline_distance,
            "improvement_fraction": self.improvement_fraction,
            "meets_50_percent": bool(
                np.isfinite(self.improvement_fraction) and self.improvement_fraction >= 0.5
            ),
            "optimizer": self.optimizer,
            "n_evaluations": self.n_evaluations,
            "duration_s": self.duration_s,
            "scale_n_cultures": self.scale_n_cultures,
            "grid": None if self.grid is None else self.grid.to_dict(),
            "history": [{"params": p.to_dict(), "distance": d} for p, d in self.history],
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path


def grid_search(
    objective: Callable[[FreeParams], float],
    base: ModelParams,
    grid: Mapping[str, Sequence[float]],
) -> GridResult:
    """Exhaustive search over 2-3 parameters (SPEC §8.2)."""
    names = tuple(grid)
    unknown = [name for name in names if name not in FREE_PARAM_NAMES]
    if unknown:
        raise KeyError(f"not free parameters: {unknown}")
    values = tuple(np.asarray(grid[name], dtype=np.float64) for name in names)
    shape = tuple(int(v.size) for v in values)
    distances = np.full(shape, np.nan, dtype=np.float64)
    n_failed = 0
    for index in np.ndindex(shape):
        changes = {names[axis]: float(values[axis][index[axis]]) for axis in range(len(names))}
        params = base.free.replace(**changes)
        try:
            distances[index] = float(objective(params))
        except Exception:  # noqa: BLE001 - a crashed cell is a NaN, not a pipeline abort
            distances[index] = float("nan")
            n_failed += 1
    return GridResult(
        parameter_names=names,
        grid_values=values,
        distances=distances,
        n_failed=n_failed,
    )


def local_optimize(
    objective: Callable[[FreeParams], float],
    start: FreeParams,
    base: ModelParams,
    *,
    method: str = "nelder-mead",
    max_evaluations: int = 40,
) -> CoarseFitResult:
    """Nelder-Mead over all 8 free parameters for a point estimate.

    Parameters are mapped to the unit hypercube of the prior box before the
    optimiser sees them: ``tau_rec`` is thousands of ms and ``U`` is O(0.1), and
    an unscaled simplex walks only along the largest coordinate.

    A point estimate is a starting point for SBI, not a result. SPEC §8.3: the
    posterior is the deliverable.
    """
    from scipy.optimize import minimize

    if method.lower() not in {"nelder-mead", "nelder_mead"}:
        raise NotImplementedError(f"optimizer {method!r} is not implemented; use nelder-mead")

    prior = base.prior
    span = prior.high - prior.low
    history: list[tuple[FreeParams, float]] = []

    def unpack(unit: np.ndarray) -> FreeParams:
        return prior.clip(FreeParams.from_vector(prior.low + np.clip(unit, 0.0, 1.0) * span))

    def packed(params: FreeParams) -> np.ndarray:
        return (params.to_vector() - prior.low) / span

    def fun(unit: np.ndarray) -> float:
        params = unpack(unit)
        value = float(objective(params))
        if not np.isfinite(value):
            value = 1.0e6
        history.append((params, value))
        return value

    result = minimize(
        fun,
        packed(start),
        method="Nelder-Mead",
        options={"maxfev": int(max_evaluations), "xatol": 0.02, "fatol": 0.01, "disp": False},
    )
    best_params = unpack(np.asarray(result.x, dtype=np.float64))
    best_distance = float(result.fun)
    return CoarseFitResult(
        best=best_params,
        best_distance=best_distance,
        baseline_distance=float("nan"),
        grid=None,
        optimizer="nelder-mead",
        n_evaluations=len(history),
        history=history,
    )


def coarse_fit(
    target: Fingerprint,
    base: ModelParams,
    *,
    scale: ScaleReference,
    spec: FingerprintSpec | None = None,
    grid: Mapping[str, Sequence[float]] | None = None,
    duration_s: float = COARSE_DURATION_S,
    max_evaluations: int = 40,
    evaluate: Callable[[FreeParams], float] | None = None,
    observation_config: Mapping[str, Any] | None = None,
    figure_path: Path | None = None,
) -> CoarseFitResult:
    """Grid stage then local stage, in that order."""
    spec = spec if spec is not None else FingerprintSpec.load()
    target.require_match(spec)
    grid = dict(DEFAULT_GRID if grid is None else grid)
    sim_base = replace(
        base,
        simulation=replace(base.simulation, duration_s=float(duration_s)),
    )

    if evaluate is None:
        grid_result, baseline_distance = _simulated_grid_and_baseline(
            sim_base,
            target,
            scale,
            spec,
            dict(observation_config) if observation_config is not None else None,
            grid,
        )
        evaluate = _make_simulator(
            sim_base,
            target,
            scale,
            spec,
            observation_config,
            start_index=1 + int(np.prod(grid_result.distances.shape)),
        )
    else:
        baseline_distance = float(evaluate(sim_base.free))
        grid_result = grid_search(evaluate, sim_base, grid)
    if not grid_result.any_finite:
        raise RuntimeError(
            "every grid cell returned a non-finite distance; the model cannot "
            "produce a comparable fingerprint anywhere in the coarse grid "
            "(SPEC §8.2 -- this is a structural bug, not an optimiser failure)"
        )
    start = sim_base.free.replace(**grid_result.best_params)
    print(
        f"coarse fit: grid best {grid_result.best_distance:.3g} at {grid_result.best_params}; "
        f"baseline {baseline_distance:.3g}; Nelder-Mead {max_evaluations} evals",
        flush=True,
    )
    local = local_optimize(
        evaluate,
        start,
        sim_base,
        max_evaluations=max_evaluations,
    )
    best = local.best
    best_distance = local.best_distance
    if grid_result.best_distance < best_distance:
        # Nelder-Mead is allowed to wander; never return a point worse than the grid.
        best = start
        best_distance = grid_result.best_distance

    result = CoarseFitResult(
        best=best,
        best_distance=best_distance,
        baseline_distance=baseline_distance,
        grid=grid_result,
        optimizer=local.optimizer,
        n_evaluations=1 + int(np.prod(grid_result.distances.shape)) + local.n_evaluations,
        history=local.history,
        duration_s=float(duration_s),
        scale_n_cultures=scale.n_cultures,
    )
    if figure_path is not None:
        figure_path = Path(figure_path)
        figure = plot_distance_landscape(grid_result)
        from ..figures import save_figure

        save_figure(figure, figure_path.stem, directory=figure_path.parent)
    return result


def _simulated_grid_and_baseline(
    base: ModelParams,
    target: Fingerprint,
    scale: ScaleReference,
    spec: FingerprintSpec,
    observation_config: dict[str, Any] | None,
    grid: Mapping[str, Sequence[float]],
) -> tuple[GridResult, float]:
    """Run the Task 1 baseline and every grid cell in one ``run_many`` pool."""
    from ..config import load_config
    from ..model.runner import run_free_params

    if observation_config is None:
        observation_config = dict(load_config("observation.yaml"))

    names = tuple(grid)
    values = tuple(np.asarray(grid[name], dtype=np.float64) for name in names)
    shape = tuple(int(v.size) for v in values)
    draws = [base.free]
    indices: list[tuple[int, ...]] = []
    for index in np.ndindex(shape):
        changes = {names[axis]: float(values[axis][index[axis]]) for axis in range(len(names))}
        draws.append(base.free.replace(**changes))
        indices.append(index)

    print(
        f"coarse fit: {len(draws)} simulations (1 baseline + {len(indices)} grid) "
        f"at {base.simulation.duration_s:g} s biological",
        flush=True,
    )
    results = run_free_params(
        draws,
        base,
        observation_config=observation_config,
        start_index=0,
        on_error="skip",
    )
    by_index = {result.run_index: result for result in results}

    def _distance_of(run_index: int) -> float:
        result = by_index.get(run_index)
        if result is None:
            return float("nan")
        fingerprint = compute_fingerprint(result.recording, spec)
        return distance(fingerprint, target, scale=scale, spec=spec)

    baseline = _distance_of(0)
    distances = np.full(shape, np.nan, dtype=np.float64)
    n_failed = 0
    for offset, index in enumerate(indices):
        value = _distance_of(1 + offset)
        distances[index] = value
        if not np.isfinite(value):
            n_failed += 1
    return (
        GridResult(
            parameter_names=names,
            grid_values=values,
            distances=distances,
            n_failed=n_failed,
        ),
        float(baseline),
    )


def _make_simulator(
    base: ModelParams,
    target: Fingerprint,
    scale: ScaleReference,
    spec: FingerprintSpec,
    observation_config: Mapping[str, Any] | None,
    start_index: int = 0,
) -> Callable[[FreeParams], float]:
    """One Brian2 process per call; the grid batches separately would be faster.

    Sequential eval is what Nelder-Mead needs. ``run_one`` is already
    subprocess-isolated.
    """
    from ..config import load_config
    from ..model.runner import RunRequest, SimulationError, run_one

    observation = (
        dict(observation_config)
        if observation_config is not None
        else dict(load_config("observation.yaml"))
    )
    counter = {"n": int(start_index)}

    def evaluate(params: FreeParams) -> float:
        counter["n"] += 1
        request = RunRequest(
            params=base.with_free(params),
            run_index=counter["n"],
            observation_config=observation,
        )
        try:
            result = run_one(request)
        except SimulationError:
            return float("nan")
        fingerprint = compute_fingerprint(result.recording, spec)
        return distance(fingerprint, target, scale=scale, spec=spec)

    return evaluate


def scale_from_wagenaar(*, fetch: bool = False) -> ScaleReference:
    """Across-culture scale from the cached DIV-14 dense Wagenaar recordings.

    Does not download unless ``fetch=True``, so a missing cache is a usage error
    rather than a surprise network call in tests.
    """
    from ..data.loaders import (
        WAGENAAR_SCALE_RELATIVES,
        fetch_wagenaar,
        load_wagenaar,
        wagenaar_cache_path,
    )

    fingerprints: list[Fingerprint] = []
    missing: list[str] = []
    for relative in WAGENAAR_SCALE_RELATIVES:
        path = wagenaar_cache_path(relative)
        if not path.exists():
            if fetch:
                path = fetch_wagenaar(relative=relative)
            else:
                missing.append(relative)
                continue
        print(f"scale: fingerprinting {path.name}", flush=True)
        try:
            fingerprints.append(compute_fingerprint(load_wagenaar(path)))
        except Exception as exc:  # noqa: BLE001 - one bad file must not abort the scale
            print(f"scale: skipping {path.name}: {type(exc).__name__}: {exc}", flush=True)
    if len(fingerprints) < 2:
        raise FileNotFoundError(
            "across-culture scale needs at least two Wagenaar recordings in "
            "data/raw/wagenaar2006 (SPEC §8.1). Found "
            f"{len(fingerprints)}; missing {missing}. "
            "Download with `.venv/bin/python scripts/fetch_wagenaar.py --scale`."
        )
    return ScaleReference.from_fingerprints(fingerprints)


def plot_distance_landscape(grid: GridResult, *, extra_index: int | None = None) -> Any:
    """2-D slice of the coarse grid: ``w_e`` vs ``tau_rec`` when those axes exist."""
    from ..figures import apply_style

    apply_style()
    import matplotlib.pyplot as plt

    names = list(grid.parameter_names)
    if "w_e" in names and "tau_rec" in names:
        x_name, y_name = "w_e", "tau_rec"
    else:
        x_name, y_name = names[0], names[-1]
    x_axis = names.index(x_name)
    y_axis = names.index(y_name)
    distances = np.asarray(grid.distances, dtype=np.float64)
    title = "distance landscape"

    if distances.ndim == 1:
        raise ValueError("a 1-D grid has no landscape")
    if distances.ndim == 2:
        plane = np.moveaxis(distances, (x_axis, y_axis), (1, 0))
    else:
        extra_axes = [i for i in range(distances.ndim) if i not in {x_axis, y_axis}]
        extra_axis = extra_axes[0]
        if extra_index is None:
            extra_index = int(grid.best_index[extra_axis]) if grid.any_finite else 0
        plane3 = np.take(distances, extra_index, axis=extra_axis)
        # After take, remaining axes keep their relative order.
        remaining = [i for i in range(distances.ndim) if i != extra_axis]
        new_x = remaining.index(x_axis)
        new_y = remaining.index(y_axis)
        plane = np.moveaxis(plane3, (new_x, new_y), (1, 0))
        extra_name = names[extra_axis]
        extra_value = float(grid.grid_values[extra_axis][extra_index])
        title = f"distance landscape at {extra_name} = {extra_value:g}"

    x = grid.grid_values[x_axis]
    y = grid.grid_values[y_axis]
    figure, axis = plt.subplots(figsize=(5.6, 4.2))
    mesh = axis.pcolormesh(x, y, plane, shading="nearest", cmap="viridis_r")
    figure.colorbar(mesh, ax=axis, label="fingerprint distance")
    if grid.any_finite:
        best = grid.best_params
        axis.plot(
            best[x_name],
            best[y_name],
            "o",
            color="white",
            markersize=7,
            markeredgecolor="0.1",
        )
    axis.set_xlabel(x_name)
    axis.set_ylabel(y_name)
    axis.set_title(title, loc="left")
    figure.tight_layout()
    return figure
