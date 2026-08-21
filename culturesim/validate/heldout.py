"""Held-out statistics validation (SPEC §9.1).

Task 7.

Fit using only the rate and burst groups of the fingerprint, then check whether the
avalanche exponents and the crackling-noise scaling relation come out right *without
having been fitted*. If they do, the model captured mechanism rather than curve-fitting
-- and that distinction is the difference between a test bench and a lookup table.

Report the result either way (SPEC §9).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..fit.coarse import local_optimize
from ..fit.distance import UNDEFINED_Z, ScaleReference, distance
from ..model.params import FreeParams, ModelParams
from ..stats.fingerprint import Fingerprint, FingerprintSpec, compute_fingerprint

__all__ = [
    "HeldoutResult",
    "FITTED_GROUPS",
    "HELDOUT_GROUPS",
    "group_weights",
    "heldout_z_scores",
    "run_heldout",
]

FITTED_GROUPS = ("rates", "bursts")
HELDOUT_GROUPS = ("avalanches", "branching", "connectivity")

# Pass if at least this fraction of finite held-out |z| values are below the cap.
PASS_FRACTION = 0.5
PASS_Z = 2.0


@dataclass(frozen=True)
class HeldoutResult:
    fitted_groups: tuple[str, ...]
    heldout_groups: tuple[str, ...]
    predicted: Fingerprint
    observed: Fingerprint
    # Per-statistic z-scores on the held-out groups, in across-culture units.
    heldout_z_scores: dict[str, float] = field(default_factory=dict)
    fitted_params: FreeParams | None = None
    fitted_distance: float = float("nan")
    passed: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted_groups": list(self.fitted_groups),
            "heldout_groups": list(self.heldout_groups),
            "heldout_z_scores": self.heldout_z_scores,
            "fitted_params": None if self.fitted_params is None else self.fitted_params.to_dict(),
            "fitted_distance": self.fitted_distance,
            "passed": self.passed,
            "notes": self.notes,
            "n_heldout_finite": sum(1 for v in self.heldout_z_scores.values() if np.isfinite(v)),
            "fraction_within_z": _fraction_within(self.heldout_z_scores, PASS_Z),
        }


def group_weights(
    spec: FingerprintSpec,
    groups: tuple[str, ...],
    *,
    on: float = 1.0,
    off: float = 0.0,
) -> dict[str, float]:
    """Per-statistic weights that zero out every group outside ``groups``."""
    allowed = set(groups)
    return {name: float(on if spec.group_of[name] in allowed else off) for name in spec.names}


def heldout_z_scores(
    predicted: Fingerprint,
    observed: Fingerprint,
    *,
    scale: ScaleReference,
    spec: FingerprintSpec,
    groups: tuple[str, ...] = HELDOUT_GROUPS,
) -> dict[str, float]:
    """Across-culture z for each scalar in the held-out groups (histograms skipped)."""
    allowed = set(groups)
    bin_names = scale.histogram_bin_names
    out: dict[str, float] = {}
    for i, name in enumerate(predicted.names):
        if name in bin_names:
            continue
        if spec.group_of[name] not in allowed:
            continue
        sim = float(predicted.values[i])
        real = float(observed.values[i])
        s = float(scale.scale[i])
        if not np.isfinite(real):
            continue
        if not np.isfinite(sim):
            out[name] = float(UNDEFINED_Z)
            continue
        if not np.isfinite(s) or s <= 0:
            out[name] = float("nan")
            continue
        out[name] = float((sim - real) / s)
    return out


def run_heldout(
    observed: Fingerprint,
    spec: FingerprintSpec,
    *,
    scale: ScaleReference,
    base: ModelParams,
    start: FreeParams | None = None,
    duration_s: float = 60.0,
    max_evaluations: int = 12,
    observation_config: Mapping[str, Any] | None = None,
    fitted_params: FreeParams | None = None,
    **kwargs: Any,
) -> HeldoutResult:
    """Fit on rates+bursts only, then score avalanche/branching/connectivity.

    When ``fitted_params`` is supplied the optimiser is skipped (useful for tests and
    for replaying a known point). Otherwise a short Nelder–Mead run minimises the
    group-masked fingerprint distance — a held-out *fit*, not a second overnight SBI.
    """
    del kwargs
    observed.require_match(spec)
    weights = group_weights(spec, FITTED_GROUPS)

    if fitted_params is None:
        from dataclasses import replace

        from ..config import load_config
        from ..model.runner import RunRequest, SimulationError, run_one

        sim_base = replace(
            base,
            simulation=replace(base.simulation, duration_s=float(duration_s)),
        )
        observation = (
            dict(observation_config)
            if observation_config is not None
            else dict(load_config("observation.yaml"))
        )
        counter = {"n": 10_000}

        def evaluate(params: FreeParams) -> float:
            counter["n"] += 1
            request = RunRequest(
                params=sim_base.with_free(params),
                run_index=counter["n"],
                observation_config=observation,
            )
            try:
                result = run_one(request)
            except SimulationError:
                return float("nan")
            fingerprint = compute_fingerprint(result.recording, spec)
            return distance(fingerprint, observed, weights=weights, scale=scale, spec=spec)

        start_params = start if start is not None else sim_base.free
        local = local_optimize(
            evaluate,
            start_params,
            sim_base,
            max_evaluations=int(max_evaluations),
        )
        fitted = local.best
        fitted_distance = float(local.best_distance)
        fit_note = (
            f"Nelder–Mead on rates+bursts only ({max_evaluations} evals, "
            f"{duration_s:g} s bio); distance={fitted_distance:.3g}."
        )
    else:
        fitted = fitted_params
        fitted_distance = float("nan")
        fit_note = "Used supplied fitted_params (no re-optimise)."

    predicted, predict_note = _simulate_fingerprint(
        base,
        fitted,
        spec,
        duration_s=duration_s,
        observation_config=observation_config,
    )
    z_scores = heldout_z_scores(predicted, observed, scale=scale, spec=spec)
    fraction = _fraction_within(z_scores, PASS_Z)
    passed = fraction >= PASS_FRACTION and any(np.isfinite(v) for v in z_scores.values())
    notes = (
        f"{fit_note} {predict_note} "
        f"Held-out |z|<{PASS_Z:g} on {fraction:.0%} of finite scalars "
        f"(need >={PASS_FRACTION:.0%})."
    )
    return HeldoutResult(
        fitted_groups=FITTED_GROUPS,
        heldout_groups=HELDOUT_GROUPS,
        predicted=predicted,
        observed=observed,
        heldout_z_scores=z_scores,
        fitted_params=fitted,
        fitted_distance=fitted_distance,
        passed=passed,
        notes=notes,
    )


def _simulate_fingerprint(
    base: ModelParams,
    free: FreeParams,
    spec: FingerprintSpec,
    *,
    duration_s: float,
    observation_config: Mapping[str, Any] | None,
) -> tuple[Fingerprint, str]:
    from dataclasses import replace

    from ..config import load_config
    from ..model.runner import RunRequest, SimulationError, run_one

    sim_base = replace(
        base,
        simulation=replace(base.simulation, duration_s=float(duration_s)),
    )
    observation = (
        dict(observation_config)
        if observation_config is not None
        else dict(load_config("observation.yaml"))
    )
    try:
        result = run_one(
            RunRequest(
                params=sim_base.with_free(free),
                run_index=20_000,
                observation_config=observation,
            )
        )
    except SimulationError as exc:
        # Empty fingerprint of undefined sentinels so callers still report.
        values = np.full(len(spec.names), spec.undefined_value, dtype=np.float64)
        empty = Fingerprint(values=values, names=spec.names, version=spec.version)
        return empty, f"Prediction sim failed ({exc})."
    return compute_fingerprint(result.recording, spec), "Prediction sim ok."


def _fraction_within(z_scores: Mapping[str, float], cap: float) -> float:
    finite = [abs(float(v)) for v in z_scores.values() if np.isfinite(v)]
    if not finite:
        return 0.0
    return float(sum(1 for z in finite if z < cap) / len(finite))
