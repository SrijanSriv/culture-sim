#!/usr/bin/env python3
"""Verify the diffusion background against the exact Poisson background.

The diffusion mode matches the first two moments of the Poisson-driven current by
construction. What has to be checked is that the *network* behaves the same, since
spiking depends on threshold crossings and those are sensitive to the shape of the
fluctuations, not only their variance.

    .venv/bin/python scripts/check_background_modes.py --duration 120
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.config import load_config  # noqa: E402
from culturesim.model.params import ModelParams  # noqa: E402
from culturesim.model.runner import RunRequest, run_many  # noqa: E402

BIN_S = 0.025


def summarise(recording, n_neurons: int) -> dict[str, float]:
    n_bins = max(1, int(recording.duration / BIN_S))
    counts = np.bincount(
        np.minimum((recording.times / BIN_S).astype(np.int64), n_bins - 1), minlength=n_bins
    )
    burst_bin = counts >= max(5.0, 0.05 * n_neurons)
    starts = np.flatnonzero(burst_bin & ~np.r_[False, burst_bin[:-1]])
    ibis = np.diff(starts) * BIN_S
    baseline = counts[~burst_bin]
    return {
        "rate_hz_per_neuron": recording.n_spikes / (recording.duration * n_neurons),
        "n_bursts": float(starts.size),
        "burst_rate_per_min": starts.size / (recording.duration / 60.0),
        "median_ibi_s": float(np.median(ibis)) if ibis.size else float("nan"),
        "base_hz": float(baseline.mean() / (BIN_S * n_neurons)) if baseline.size else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    config = load_config("model_default.yaml")
    config["simulation"].update(duration_s=args.duration, transient_s=10.0)
    config["network"]["n_neurons"] = args.neurons
    base = ModelParams.from_config(config)

    requests = []
    labels = []
    for mode in ("poisson", "diffusion"):
        for seed_index in range(args.seeds):
            requests.append(
                RunRequest(
                    params=replace(base, simulation=replace(base.simulation, background_mode=mode)),
                    run_index=seed_index,
                )
            )
            labels.append(mode)

    results = run_many(requests, n_workers=min(args.seeds * 2, 6))
    by_mode: dict[str, list[dict[str, float]]] = {"poisson": [], "diffusion": []}
    wall: dict[str, list[float]] = {"poisson": [], "diffusion": []}
    for label, result in zip(labels, results, strict=True):
        by_mode[label].append(summarise(result.recording, args.neurons))
        wall[label].append(result.wall_clock_s)

    keys = list(by_mode["poisson"][0])
    print(f"{args.duration}s biological, {args.neurons} neurons, {args.seeds} seeds per mode\n")
    print(f"{'metric':22s}{'poisson':>22}{'diffusion':>22}")
    print("-" * 66)
    for key in keys:
        poisson = np.array([s[key] for s in by_mode["poisson"]])
        diffusion = np.array([s[key] for s in by_mode["diffusion"]])
        print(
            f"{key:22s}{np.nanmean(poisson):12.3f} +-{np.nanstd(poisson):7.3f}"
            f"{np.nanmean(diffusion):12.3f} +-{np.nanstd(diffusion):7.3f}"
        )
    print(
        f"\n{'wall clock (s)':22s}{np.mean(wall['poisson']):12.1f}"
        f"{'':10s}{np.mean(wall['diffusion']):12.1f}"
    )
    speedup = np.mean(wall["poisson"]) / np.mean(wall["diffusion"])
    print(f"{'speedup':22s}{'':22s}{speedup:12.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
