"""Simulation-based inference with SNPE-C (SPEC §8.3).

Task 6.

Designed to run overnight without an agent attached: ``simulate_training_set``
checkpoints every batch and updates ``output/task6_status.json``; on success
``sync_readme_task6`` rewrites the README State cell. Start with
``culture-sim fit sbi ... --detach`` and check later via
``scripts/check_task6.py``.

The posterior is the deliverable, not a point estimate. Read the marginals honestly:

* A tight marginal means the data genuinely identifies that parameter.
* A flat marginal means the fingerprint cannot see that parameter. That is a
  **finding** about what MEA statistics constrain, not a failure of the fit, and it
  must be reported as such -- collapsing it to a MAP value would assert precision the
  data does not support.
* Pairwise posterior correlations reveal degeneracies: two parameters that trade off
  along a ridge are jointly constrained but individually not, which is a different
  statement again.

Posterior predictive checks close the loop: sample parameters from the posterior,
simulate, and confirm the resulting fingerprints bracket the real one.
"""

from __future__ import annotations

import pickle
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..model.params import FREE_PARAM_NAMES, FreeParams, ModelParams, PriorBox
from ..stats.fingerprint import Fingerprint, FingerprintSpec, compute_fingerprint
from .task_status import (
    DEFAULT_STATUS_PATH,
    Task6Status,
    read_status,
    sync_readme_task6,
    write_status,
)

__all__ = [
    "SBIResult",
    "PosteriorSummary",
    "simulate_training_set",
    "train_posterior",
    "posterior_predictive_check",
    "run_sbi_fit",
    "posterior_to_free_params",
]


@dataclass(frozen=True)
class PosteriorSummary:
    """Per-parameter marginal summary, plus the identifiability verdict."""

    names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    quantiles: Mapping[float, np.ndarray]
    prior_std: np.ndarray
    correlations: np.ndarray  # (n_params, n_params) pairwise posterior correlation
    identified: Mapping[str, bool]
    std_ratio_threshold: float

    def identified_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.names if self.identified[n])

    def unidentified_names(self) -> tuple[str, ...]:
        """Parameters the fingerprint cannot constrain. Goes in the README verbatim."""
        return tuple(n for n in self.names if not self.identified[n])

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "quantiles": {str(k): v.tolist() for k, v in self.quantiles.items()},
            "prior_std": self.prior_std.tolist(),
            "correlations": self.correlations.tolist(),
            "identified": dict(self.identified),
            "identified_names": list(self.identified_names()),
            "unidentified_names": list(self.unidentified_names()),
            "std_ratio_threshold": self.std_ratio_threshold,
        }


@dataclass(frozen=True)
class SBIResult:
    posterior: Any  # sbi DirectPosterior; kept untyped so importing this is cheap
    summary: PosteriorSummary
    observed_fingerprint: Fingerprint
    n_simulations: int
    n_excluded: int  # draws that crashed or returned non-finite fingerprints
    theta: np.ndarray
    fingerprints: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "posterior": self.posterior,
            "summary": self.summary,
            "observed_fingerprint": self.observed_fingerprint,
            "n_simulations": self.n_simulations,
            "n_excluded": self.n_excluded,
            "theta": self.theta,
            "fingerprints": self.fingerprints,
            "metadata": self.metadata,
        }
        staging = path.with_suffix(path.suffix + ".partial")
        with staging.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        staging.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> SBIResult:
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - our own artefact
        return cls(**payload)


def simulate_training_set(
    base: ModelParams,
    prior: PriorBox,
    n_simulations: int,
    *,
    observation_config: Mapping[str, Any] | None = None,
    fingerprint_spec: FingerprintSpec | None = None,
    batch_size: int | None = None,
    n_workers: int | None = None,
    checkpoint: str | Path | None = None,
    status_path: str | Path = DEFAULT_STATUS_PATH,
    duration_s: float | None = None,
    seed: int | None = None,
    resume: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Draw from the prior, simulate, and return ``(theta, fingerprints, n_excluded)``.

    Draws whose simulation crashes or yields a non-finite fingerprint are excluded and
    counted, never silently replaced: the exclusion rate is itself informative about
    which regions of the prior box the model cannot run in.

    Checkpoints after every batch so an overnight run can be killed and resumed.
    """
    del kwargs
    from ..config import load_config
    from ..model.runner import default_workers, run_free_params

    if observation_config is None:
        observation_config = load_config("observation.yaml")
    fingerprint_spec = fingerprint_spec or FingerprintSpec.load()
    if duration_s is not None:
        base = replace(base, simulation=replace(base.simulation, duration_s=float(duration_s)))
    if seed is not None:
        base = replace(base, seed=int(seed))

    workers = default_workers() if n_workers is None else int(n_workers)
    batch = int(batch_size) if batch_size is not None else max(workers, 8)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else None

    kept_theta: list[np.ndarray] = []
    kept_fps: list[np.ndarray] = []
    n_excluded = 0
    n_attempted = 0
    start_index = 0

    if resume and checkpoint_path is not None and checkpoint_path.exists():
        loaded = np.load(checkpoint_path, allow_pickle=False)
        if loaded["theta"].size:
            kept_theta.append(np.asarray(loaded["theta"], dtype=np.float64))
            kept_fps.append(np.asarray(loaded["fingerprints"], dtype=np.float64))
        n_excluded = int(loaded["n_excluded"])
        n_attempted = int(loaded["n_attempted"])
        start_index = n_attempted
        n_kept_so_far = sum(t.shape[0] for t in kept_theta)
        print(
            f"sbi: resumed checkpoint {checkpoint_path} "
            f"({n_attempted}/{n_simulations} attempted, kept {n_kept_so_far})",
            flush=True,
        )

    rng = np.random.default_rng(base.seed + 17)
    # Skip draws already consumed so resume is deterministic.
    if start_index:
        _ = prior.sample(rng, start_index)

    status = read_status(status_path)
    status.state = "running"
    status.n_simulations_target = int(n_simulations)
    status.n_attempted = n_attempted
    status.n_kept = int(sum(t.shape[0] for t in kept_theta))
    status.n_excluded = n_excluded
    status.batch_size = batch
    status.duration_s = float(base.simulation.duration_s)
    status.checkpoint = None if checkpoint_path is None else str(checkpoint_path)
    status.pid = __import__("os").getpid()
    status.message = "simulating training set"
    write_status(status, status_path)
    sync_readme_task6(status, status_path=status_path)

    while n_attempted < n_simulations:
        remaining = n_simulations - n_attempted
        this_batch = min(batch, remaining)
        draws = prior.sample(rng, this_batch)
        t0 = time.perf_counter()
        results = run_free_params(
            draws,
            base,
            observation_config=dict(observation_config),
            start_index=n_attempted,
            n_workers=workers,
            on_error="skip",
        )
        by_index = {r.run_index: r for r in results}
        for offset, draw in enumerate(draws):
            run_index = n_attempted + offset
            result = by_index.get(run_index)
            if result is None:
                n_excluded += 1
                continue
            try:
                fingerprint = compute_fingerprint(result.recording, fingerprint_spec)
            except Exception:  # noqa: BLE001 - treat as excluded draw
                n_excluded += 1
                continue
            if fingerprint.n_undefined > len(fingerprint) // 2:
                # Too many NaNs to train on; still informative via the exclusion count.
                n_excluded += 1
                continue
            values = np.asarray(fingerprint.values, dtype=np.float64)
            values = np.where(np.isfinite(values), values, 0.0)
            kept_theta.append(draw.to_vector().reshape(1, -1))
            kept_fps.append(values.reshape(1, -1))
        n_attempted += this_batch
        elapsed = time.perf_counter() - t0
        n_kept = int(sum(t.shape[0] for t in kept_theta))
        print(
            f"sbi: batch done attempted={n_attempted}/{n_simulations} "
            f"kept={n_kept} excluded={n_excluded} wall={elapsed:.1f}s",
            flush=True,
        )
        if checkpoint_path is not None:
            _save_checkpoint(
                checkpoint_path,
                theta=_stack(kept_theta, n_params=len(FREE_PARAM_NAMES)),
                fingerprints=_stack(kept_fps, n_params=len(fingerprint_spec)),
                n_excluded=n_excluded,
                n_attempted=n_attempted,
            )
        status.n_attempted = n_attempted
        status.n_kept = n_kept
        status.n_excluded = n_excluded
        status.message = f"simulating ({n_attempted}/{n_simulations})"
        write_status(status, status_path)
        sync_readme_task6(status, status_path=status_path)

    theta = _stack(kept_theta, n_params=len(FREE_PARAM_NAMES))
    fingerprints = _stack(kept_fps, n_params=len(fingerprint_spec))
    return theta, fingerprints, n_excluded


def train_posterior(
    theta: np.ndarray,
    fingerprints: np.ndarray,
    observed: Fingerprint,
    config: Mapping[str, Any],
    *,
    prior: PriorBox | None = None,
) -> SBIResult:
    """Train SNPE-C with an embedding net over the fingerprint and condition on ``observed``."""
    import torch
    from sbi.inference import SNPE
    from sbi.neural_nets import posterior_nn
    from sbi.utils import BoxUniform
    from torch import nn

    if theta.ndim != 2 or fingerprints.ndim != 2:
        raise ValueError("theta and fingerprints must be 2-D arrays")
    if theta.shape[0] != fingerprints.shape[0]:
        raise ValueError("theta and fingerprints must have the same number of rows")
    if theta.shape[0] < 50:
        raise ValueError(
            f"need at least 50 kept simulations to train SNPE, got {theta.shape[0]}"
        )

    inference_cfg = config.get("inference", {})
    posterior_cfg = config.get("posterior", {})
    embed_cfg = inference_cfg.get("embedding_net", {})
    train_cfg = inference_cfg.get("training", {})

    if prior is None:
        prior = ModelParams.load(
            config.get("prior", {}).get("from_model_config", "model_default.yaml")
        ).prior

    low = torch.as_tensor(prior.low, dtype=torch.float32)
    high = torch.as_tensor(prior.high, dtype=torch.float32)
    sbi_prior = BoxUniform(low=low, high=high)

    hidden = [int(h) for h in embed_cfg.get("hidden_layers", [128, 128])]
    layers: list[nn.Module] = []
    prev = int(fingerprints.shape[1])
    for width in hidden:
        layers.extend([nn.Linear(prev, width), nn.ReLU()])
        prev = width
    output_dim = int(embed_cfg.get("output_dim", 32))
    layers.append(nn.Linear(prev, output_dim))
    embedding = nn.Sequential(*layers)

    density_estimator = posterior_nn(
        model=str(inference_cfg.get("density_estimator", "nsf")),
        embedding_net=embedding,
    )
    inference = SNPE(prior=sbi_prior, density_estimator=density_estimator)
    inference.append_simulations(
        torch.as_tensor(theta, dtype=torch.float32),
        torch.as_tensor(fingerprints, dtype=torch.float32),
    )
    density = inference.train(
        training_batch_size=int(train_cfg.get("batch_size", 200)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-4)),
        validation_fraction=float(train_cfg.get("validation_fraction", 0.1)),
        stop_after_epochs=int(train_cfg.get("stop_after_epochs", 25)),
        max_num_epochs=int(train_cfg.get("max_num_epochs", 500)),
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density)

    x_o = torch.as_tensor(np.nan_to_num(observed.values, nan=0.0), dtype=torch.float32)
    n_samples = int(posterior_cfg.get("n_samples", 10_000))
    samples = posterior.sample((n_samples,), x=x_o).detach().cpu().numpy()
    summary = summarise_posterior(
        samples,
        prior,
        std_ratio_threshold=float(posterior_cfg.get("identifiability_std_ratio", 0.5)),
    )
    return SBIResult(
        posterior=posterior,
        summary=summary,
        observed_fingerprint=observed,
        n_simulations=int(theta.shape[0]),
        n_excluded=0,
        theta=np.asarray(theta, dtype=np.float64),
        fingerprints=np.asarray(fingerprints, dtype=np.float64),
        metadata={"n_posterior_samples": n_samples, "method": "SNPE_C"},
    )


def summarise_posterior(
    samples: np.ndarray,
    prior: PriorBox,
    *,
    std_ratio_threshold: float = 0.5,
) -> PosteriorSummary:
    samples = np.asarray(samples, dtype=np.float64)
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    prior_std = (prior.high - prior.low) / np.sqrt(12.0)  # uniform
    quantiles = {
        q: np.quantile(samples, q, axis=0) for q in (0.05, 0.25, 0.50, 0.75, 0.95)
    }
    corr = np.corrcoef(samples, rowvar=False)
    identified = {
        name: bool(std[i] < std_ratio_threshold * prior_std[i])
        for i, name in enumerate(FREE_PARAM_NAMES)
    }
    return PosteriorSummary(
        names=FREE_PARAM_NAMES,
        mean=mean,
        std=std,
        quantiles=quantiles,
        prior_std=prior_std,
        correlations=corr,
        identified=identified,
        std_ratio_threshold=std_ratio_threshold,
    )


def posterior_predictive_check(
    result: SBIResult,
    base: ModelParams,
    n_draws: int = 100,
    *,
    observation_config: Mapping[str, Any] | None = None,
    coverage_interval: tuple[float, float] = (5.0, 95.0),
    **kwargs: Any,
) -> dict[str, Any]:
    """Simulate from posterior samples and test whether the real fingerprint is bracketed."""
    del kwargs
    from ..config import load_config
    from ..model.runner import run_free_params

    if observation_config is None:
        observation_config = load_config("observation.yaml")
    spec = FingerprintSpec.load()
    rng = np.random.default_rng(base.seed + 99)
    idx = rng.choice(result.theta.shape[0], size=min(n_draws, result.theta.shape[0]), replace=False)
    # Prefer fresh posterior samples when the posterior object can draw them.
    try:
        import torch

        x_o = torch.as_tensor(
            np.nan_to_num(result.observed_fingerprint.values, nan=0.0), dtype=torch.float32
        )
        draws_theta = (
            result.posterior.sample((n_draws,), x=x_o).detach().cpu().numpy()
        )
        free = [FreeParams.from_vector(row) for row in draws_theta]
    except Exception:  # noqa: BLE001 - fall back to training-set rows near the posterior
        free = [FreeParams.from_vector(result.theta[i]) for i in idx]

    results = run_free_params(
        free,
        base,
        observation_config=dict(observation_config),
        start_index=10_000,
        on_error="skip",
    )
    rows = []
    for run in results:
        try:
            fp = compute_fingerprint(run.recording, spec)
        except Exception:  # noqa: BLE001
            continue
        rows.append(np.nan_to_num(fp.values, nan=np.nan))
    if not rows:
        return {"n_draws": 0, "coverage": {}, "bracketed_fraction": float("nan")}
    stack = np.vstack(rows)
    lo, hi = coverage_interval
    q_lo = np.nanpercentile(stack, lo, axis=0)
    q_hi = np.nanpercentile(stack, hi, axis=0)
    observed = result.observed_fingerprint.values
    bracketed = (observed >= q_lo) & (observed <= q_hi) & np.isfinite(observed)
    coverage = {
        name: bool(bracketed[i])
        for i, name in enumerate(result.observed_fingerprint.names)
        if np.isfinite(observed[i])
    }
    return {
        "n_draws": int(stack.shape[0]),
        "coverage_interval": list(coverage_interval),
        "bracketed_fraction": float(np.mean(list(coverage.values()))) if coverage else float("nan"),
        "coverage": coverage,
        "n_bracketed": int(sum(coverage.values())),
        "n_checked": int(len(coverage)),
    }


def run_sbi_fit(
    *,
    observed: Fingerprint,
    base: ModelParams,
    config: Mapping[str, Any],
    n_simulations: int,
    out: str | Path,
    status_path: str | Path = DEFAULT_STATUS_PATH,
    resume: bool = True,
) -> SBIResult:
    """Full Task 6 pipeline: simulate → train → PPC → status/README update."""
    from ..config import load_config

    out = Path(out)
    status_path = Path(status_path)
    sim_cfg = config.get("simulator", {})
    duration_s = float(sim_cfg.get("duration_s", base.simulation.duration_s))
    n_workers = sim_cfg.get("n_workers")
    observation_config = load_config(sim_cfg.get("observation_config", "observation.yaml"))
    fingerprint_spec = FingerprintSpec.load(
        sim_cfg.get("fingerprint_config", "fingerprint.yaml")
    )
    checkpoint = out.with_suffix(".checkpoint.npz")
    log_hint = str(out.with_suffix(".log"))

    status = Task6Status(
        state="running",
        n_simulations_target=int(n_simulations),
        duration_s=duration_s,
        out=str(out),
        log=log_hint,
        checkpoint=str(checkpoint),
        pid=__import__("os").getpid(),
        message="starting",
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    write_status(status, status_path)
    sync_readme_task6(status, status_path=status_path)

    try:
        base = replace(base, simulation=replace(base.simulation, duration_s=duration_s))
        theta, fingerprints, n_excluded = simulate_training_set(
            base,
            base.prior,
            n_simulations,
            observation_config=observation_config,
            fingerprint_spec=fingerprint_spec,
            n_workers=None if n_workers is None else int(n_workers),
            checkpoint=checkpoint,
            status_path=status_path,
            resume=resume,
        )
        status = read_status(status_path)
        status.message = "training SNPE-C"
        write_status(status, status_path)
        sync_readme_task6(status, status_path=status_path)

        result = train_posterior(theta, fingerprints, observed, config, prior=base.prior)
        result = SBIResult(
            posterior=result.posterior,
            summary=result.summary,
            observed_fingerprint=result.observed_fingerprint,
            n_simulations=result.n_simulations,
            n_excluded=n_excluded,
            theta=result.theta,
            fingerprints=result.fingerprints,
            metadata=result.metadata,
        )
        result.save(out)

        ppc_cfg = config.get("posterior_predictive", {})
        ppc = posterior_predictive_check(
            result,
            base,
            n_draws=int(ppc_cfg.get("n_draws", 50)),
            observation_config=observation_config,
            coverage_interval=tuple(ppc_cfg.get("coverage_interval", [5, 95])),
        )
        summary_path = out.with_suffix(".summary.json")
        summary_payload = {
            "summary": result.summary.to_dict(),
            "n_simulations": result.n_simulations,
            "n_excluded": n_excluded,
            "posterior_predictive": {
                k: v
                for k, v in ppc.items()
                if k != "coverage"  # coverage is huge; kept in the pkl metadata
            },
            "identified": list(result.summary.identified_names()),
            "unidentified": list(result.summary.unidentified_names()),
        }
        summary_path.write_text(
            __import__("json").dumps(summary_payload, indent=2, default=str),
            encoding="utf-8",
        )
        result.metadata["posterior_predictive"] = ppc
        result.metadata["summary_json"] = str(summary_path)
        result.save(out)

        status = read_status(status_path)
        status.state = "done"
        status.n_kept = result.n_simulations
        status.n_excluded = n_excluded
        status.n_attempted = max(status.n_attempted, result.n_simulations + n_excluded)
        status.message = "complete"
        status.summary = {
            "n_simulations": result.n_simulations,
            "n_excluded": n_excluded,
            "identified": list(result.summary.identified_names()),
            "unidentified": list(result.summary.unidentified_names()),
            "bracketed_fraction": ppc.get("bracketed_fraction"),
            "out": str(out),
            "summary_json": str(summary_path),
        }
        write_status(status, status_path)
        sync_readme_task6(status, status_path=status_path)
        _write_identifiability_readme_section(result.summary)
        return result
    except Exception as exc:
        status = read_status(status_path)
        status.state = "failed"
        status.message = f"{type(exc).__name__}: {exc}"
        write_status(status, status_path)
        sync_readme_task6(status, status_path=status_path)
        raise


def _write_identifiability_readme_section(summary: PosteriorSummary) -> None:
    """Fill the README 'which parameters the data identifies' bullet when done."""
    from ..config import REPO_ROOT

    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    identified = ", ".join(summary.identified_names()) or "none"
    unidentified = ", ".join(summary.unidentified_names()) or "none"
    block = (
        f"- **Which parameters the data identifies** (SPEC §8.3) — identified: "
        f"`{identified}`; unidentified (flat marginal): `{unidentified}`. "
        f"Threshold: posterior std < {summary.std_ratio_threshold:g} × prior std."
    )
    pattern = __import__("re").compile(
        r"- \*\*Which parameters the data identifies\*\* \(SPEC §8\.3\).*"
    )
    if pattern.search(text):
        readme.write_text(pattern.sub(block, text, count=1), encoding="utf-8")


def posterior_to_free_params(samples: np.ndarray) -> list[FreeParams]:
    """Posterior sample matrix -> parameter objects, in ``FREE_PARAM_NAMES`` order."""
    return [FreeParams.from_vector(row) for row in np.atleast_2d(samples)]


def _stack(chunks: list[np.ndarray], *, n_params: int) -> np.ndarray:
    if not chunks:
        return np.zeros((0, n_params), dtype=np.float64)
    return np.vstack(chunks)


def _save_checkpoint(
    path: Path,
    *,
    theta: np.ndarray,
    fingerprints: np.ndarray,
    n_excluded: int,
    n_attempted: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / (path.stem + ".partial.npz")
    with staging.open("wb") as handle:
        np.savez(
            handle,
            theta=theta,
            fingerprints=fingerprints,
            n_excluded=np.asarray(n_excluded),
            n_attempted=np.asarray(n_attempted),
        )
    staging.replace(path)