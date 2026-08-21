"""Run the SPEC §9 validation suite end-to-end."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config
from ..data.loaders import load_wagenaar, wagenaar_cache_path
from ..fit.coarse import scale_from_wagenaar
from ..fit.sbi_fit import SBIResult
from ..model.params import FREE_PARAM_NAMES, FreeParams, ModelParams
from ..stats.fingerprint import FingerprintSpec, compute_fingerprint
from .cross_culture import CrossCultureResult, run_cross_culture
from .heldout import HeldoutResult, run_heldout
from .perturbation import PerturbationResult, StimulusProtocol, run_perturbation

__all__ = ["ValidationReport", "run_validation"]

TESTS = ("heldout", "cross_culture", "perturbation")


@dataclass(frozen=True)
class ValidationReport:
    tests_run: tuple[str, ...]
    heldout: HeldoutResult | None = None
    cross_culture: CrossCultureResult | None = None
    perturbation: PerturbationResult | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tests_run": list(self.tests_run),
            "heldout": None if self.heldout is None else self.heldout.to_dict(),
            "cross_culture": None if self.cross_culture is None else self.cross_culture.to_dict(),
            "perturbation": None if self.perturbation is None else self.perturbation.to_dict(),
            "diagnostics": self.diagnostics,
            "any_passed": any(
                result.passed
                for result in (self.heldout, self.cross_culture, self.perturbation)
                if result is not None
            ),
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_jsonable(self.to_dict()), indent=2, sort_keys=True) + "\n")
        return path


def _jsonable(obj: Any) -> Any:
    """Replace NaN/Inf with None so the validation artefact is valid JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.floating):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def run_validation(
    posterior_path: Path,
    *,
    tests: Sequence[str] = ("all",),
    seed: int = 0,
    duration_s: float = 60.0,
    heldout_max_evaluations: int = 8,
    culture_b_relative: str = "simple-text/daily/spont/dense/1-2-14.spk.txt.bz2",
    stim_relative: str | None = "simple-text/daily/stim/dense/1-1-14.spk.txt.bz2",
    stim_cache: Path | None = None,
    skip_heldout_fit: bool = False,
    fit_culture_b: bool = False,
) -> ValidationReport:
    """Load the Task 6 posterior and run the requested §9 tests."""
    selected = _expand_tests(tests)
    result = SBIResult.load(posterior_path)
    base = ModelParams.load()
    spec = FingerprintSpec.load()
    observed = result.observed_fingerprint
    samples = _posterior_samples(result, n=min(500, max(50, result.n_simulations // 4)), seed=seed)
    median = FreeParams.from_vector(np.median(samples, axis=0))

    heldout_result: HeldoutResult | None = None
    cross_result: CrossCultureResult | None = None
    pert_result: PerturbationResult | None = None
    diagnostics: dict[str, Any] = {
        "posterior": str(posterior_path),
        "n_posterior_samples_used": int(samples.shape[0]),
        "param_names": list(FREE_PARAM_NAMES),
    }

    if "heldout" in selected:
        print("validate: held-out statistics (SPEC §9.1)", flush=True)
        scale = scale_from_wagenaar(fetch=False)
        heldout_result = run_heldout(
            observed,
            spec,
            scale=scale,
            base=base,
            start=median,
            duration_s=duration_s,
            max_evaluations=heldout_max_evaluations,
            fitted_params=median if skip_heldout_fit else None,
        )
        print(
            f"validate: heldout passed={heldout_result.passed} — {heldout_result.notes}",
            flush=True,
        )

    if "cross_culture" in selected:
        print("validate: cross-culture (SPEC §9.2)", flush=True)
        point_b = None
        if fit_culture_b:
            point_b = _coarse_point_for_culture_b(
                culture_b_relative,
                base=base,
                start=median,
                duration_s=duration_s,
            )
        culture_a_key = "1-1-14"
        samples_map = {culture_a_key: samples}
        points = {}
        if point_b is not None:
            points[point_b[0]] = point_b[1]
            diagnostics["culture_b"] = point_b[0]
        cross_result = run_cross_culture(
            samples_by_culture=samples_map,
            point_estimates=points or None,
            prior=base.prior,
        )
        print(
            f"validate: cross_culture passed={cross_result.passed} — {cross_result.notes}",
            flush=True,
        )

    if "perturbation" in selected:
        print("validate: perturbation response (SPEC §9.3)", flush=True)
        # Short protocol so a laptop run finishes in one stim sim (~30–60 s bio).
        protocol = StimulusProtocol(
            electrode=30,
            amplitudes=(0.5, 1.0),
            n_pulses_per_amplitude=5,
            inter_stimulus_interval_s=3.0,
        )
        observed_rec = None
        observed_times = None
        if stim_relative is not None:
            from ..config import REPO_ROOT

            stim_path = (
                Path(stim_cache)
                if stim_cache is not None
                else REPO_ROOT / "data/raw/wagenaar2006/stim" / Path(stim_relative).name
            )
            if stim_path.exists():
                observed_rec = load_wagenaar(stim_path)
                observed_times = observed_rec.metadata.get("stimulus_times_s") or []
                diagnostics["stim_recording"] = str(stim_path)
            else:
                diagnostics["stim_recording_missing"] = str(stim_path)
        free_for_stim = (
            heldout_result.fitted_params
            if heldout_result is not None and heldout_result.fitted_params is not None
            else median
        )
        pert_result = run_perturbation(
            base=base,
            free=free_for_stim,
            protocol=protocol,
            observed_recording=observed_rec,
            observed_stimulus_times_s=observed_times,
        )
        print(
            f"validate: perturbation passed={pert_result.passed} — {pert_result.notes}",
            flush=True,
        )

    return ValidationReport(
        tests_run=selected,
        heldout=heldout_result,
        cross_culture=cross_result,
        perturbation=pert_result,
        diagnostics=diagnostics,
    )


def update_readme_validation(report: ValidationReport, readme: Path | None = None) -> None:
    """Rewrite the Task 7 row and the three result bullets in README.md."""
    from ..config import REPO_ROOT

    readme = Path(readme) if readme is not None else REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    h = report.heldout
    c = report.cross_culture
    p = report.perturbation
    state = "Done (see results)"
    row = f"| 7 | Validation suite | **{state}** |"
    import re

    text = re.sub(r"\| 7 \| Validation suite \|.*?\|", row, text, count=1)

    def verdict(result: Any | None) -> str:
        if result is None:
            return "not run."
        return "**PASS**." if result.passed else "**FAIL**."

    heldout_body = f"{verdict(h)} " + (
        "See `output/validation.json`."
        if h is None
        else (f"Held-out |z|<2 on {_fraction_note(h)}; details in `output/validation.json`.")
    )
    cross_body = f"{verdict(c)} " + (
        "See `output/validation.json`."
        if c is None
        else c.notes.split(" Deferred")[0].rstrip(".") + ". See `docs/SBI_REFIT.md`."
    )
    pert_body = f"{verdict(p)} " + ("See `output/validation.json`." if p is None else p.notes)

    replacements = {
        r"- \*\*Held-out statistics\*\*.*": (
            f"- **Held-out statistics** (SPEC §9.1) — {heldout_body}"
        ),
        r"- \*\*Cross-culture posterior overlap\*\*.*": (
            f"- **Cross-culture posterior overlap** (SPEC §9.2) — {cross_body}"
        ),
        r"- \*\*Whether the model predicts evoked responses\*\*.*": (
            f"- **Whether the model predicts evoked responses** (SPEC §9.3) — {pert_body}"
        ),
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, count=1)
    readme.write_text(text, encoding="utf-8")


def _fraction_note(heldout: HeldoutResult) -> str:
    data = heldout.to_dict()
    return f"{100.0 * float(data.get('fraction_within_z', 0.0)):.0f}% of finite scalars"


def _expand_tests(tests: Sequence[str]) -> tuple[str, ...]:
    if not tests or tests == ("all",) or list(tests) == ["all"]:
        return TESTS
    out: list[str] = []
    for name in tests:
        if name == "all":
            return TESTS
        if name not in TESTS:
            raise ValueError(f"unknown validation test {name!r}; choose from {TESTS}")
        out.append(name)
    return tuple(out)


def _posterior_samples(result: SBIResult, *, n: int, seed: int) -> np.ndarray:
    """Approximate posterior draws from the Task 6 summary (mean/std).

    Re-running MCMC on the pickled posterior is minutes-to-hours for a few dozen
    draws on this machine; the summary already came from 2000 MCMC samples.
    """
    mean = np.asarray(result.summary.mean, dtype=np.float64)
    std = np.asarray(result.summary.std, dtype=np.float64)
    std = np.where(np.isfinite(std) & (std > 0), std, 1e-6)
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=(n, mean.size))


def _coarse_point_for_culture_b(
    relative: str,
    *,
    base: ModelParams,
    start: FreeParams,
    duration_s: float,
) -> tuple[str, FreeParams] | None:
    """Fingerprint culture B and take a few Nelder–Mead steps from A's median."""
    from dataclasses import replace

    from ..fit.coarse import local_optimize
    from ..fit.distance import distance
    from ..model.runner import RunRequest, SimulationError, run_one

    path = wagenaar_cache_path(relative)
    if not path.exists():
        print(f"validate: culture B missing at {path}; skip point estimate", flush=True)
        return None
    key = path.name.split(".")[0]
    print(f"validate: coarse-fitting culture B ({key})", flush=True)
    target = compute_fingerprint(load_wagenaar(path))
    scale = scale_from_wagenaar(fetch=False)
    spec = FingerprintSpec.load()
    observation = dict(load_config("observation.yaml"))
    sim_base = replace(
        base,
        simulation=replace(base.simulation, duration_s=float(duration_s)),
    )
    counter = {"n": 40_000}

    def evaluate(params: FreeParams) -> float:
        counter["n"] += 1
        try:
            run = run_one(
                RunRequest(
                    params=sim_base.with_free(params),
                    run_index=counter["n"],
                    observation_config=observation,
                )
            )
        except SimulationError:
            return float("nan")
        return distance(compute_fingerprint(run.recording, spec), target, scale=scale, spec=spec)

    local = local_optimize(evaluate, start, sim_base, max_evaluations=6)
    return key, local.best
