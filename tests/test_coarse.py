"""Coarse fit: grid then Nelder-Mead (SPEC §8.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from culturesim.fit.coarse import (
    DEFAULT_GRID,
    GRID_PARAMETERS,
    coarse_fit,
    grid_search,
    local_optimize,
    plot_distance_landscape,
)
from culturesim.fit.distance import ScaleReference
from culturesim.model.params import FreeParams, ModelParams
from culturesim.stats.fingerprint import Fingerprint, FingerprintSpec


def _bowl(params: FreeParams) -> float:
    """Quadratic bowl centred on the Task 1 hand-tune."""
    return float(
        ((params.w_e - 1.5) / 1.0) ** 2
        + ((params.g - 2.0) / 1.0) ** 2
        + ((params.tau_rec - 2500.0) / 1000.0) ** 2
        + ((params.rate_bg - 4.98) / 1.0) ** 2
    )


def test_grid_parameters_match_the_spec() -> None:
    assert GRID_PARAMETERS == ("w_e", "g", "tau_rec")
    assert set(DEFAULT_GRID) == set(GRID_PARAMETERS)


def test_grid_search_recovers_the_bowl_minimum() -> None:
    base = ModelParams()
    result = grid_search(lambda p: _bowl(p), base, DEFAULT_GRID)
    assert result.any_finite
    assert result.best_params["w_e"] == pytest.approx(1.5)
    assert result.best_params["g"] == pytest.approx(2.0)
    assert result.best_params["tau_rec"] == pytest.approx(2500.0)


def test_nelder_mead_improves_a_bad_start() -> None:
    base = ModelParams()
    start = base.free.replace(w_e=0.7, g=3.0, tau_rec=600.0, rate_bg=3.0)
    fitted = local_optimize(_bowl, start, base, max_evaluations=80)
    assert fitted.best_distance < _bowl(start)
    assert fitted.n_evaluations > 1


def test_coarse_fit_halves_the_baseline_on_a_known_bowl(tmp_path: Path) -> None:
    spec = FingerprintSpec.load()
    dummy = Fingerprint(
        values=np.ones(len(spec), dtype=np.float64),
        names=spec.names,
        version=spec.version,
    )
    other = Fingerprint(
        values=np.ones(len(spec), dtype=np.float64) * 1.1,
        names=spec.names,
        version=spec.version,
    )
    scale = ScaleReference.from_fingerprints([dummy, other], spec=spec)
    start = FreeParams(w_e=0.7, g=3.0, tau_rec=600.0, rate_bg=3.0)
    base = ModelParams(free=start)
    result = coarse_fit(
        target=dummy,
        base=base,
        scale=scale,
        spec=spec,
        evaluate=_bowl,
        max_evaluations=40,
        figure_path=tmp_path / "task5_distance_landscape.png",
    )
    assert result.improvement_fraction >= 0.5
    assert result.grid is not None
    assert (tmp_path / "task5_distance_landscape.png").exists()


def test_landscape_plot_accepts_a_2d_grid() -> None:
    base = ModelParams()
    grid = grid_search(
        _bowl,
        base,
        {"w_e": (0.7, 1.5, 3.0), "tau_rec": (600.0, 2500.0, 4500.0)},
    )
    figure = plot_distance_landscape(grid)
    assert figure.axes
