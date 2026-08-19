#!/usr/bin/env python3
"""Hand-tune the Task 1 parameter set until the network bursts (SPEC §13).

Sweeps a few parameters in parallel and reports, per draw, the neuron-level firing rate
and a crude burst count. Deliberately crude: the real detector is Task 3, and using it
here would mean tuning against a statistic before it has been tested.

This is a development tool, not part of the pipeline. It exists because SPEC §13 Task 1
requires a hand-tuned set that visibly bursts before any fitting begins.

    .venv/bin/python scripts/tune_task1.py --duration 40
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.config import load_config  # noqa: E402
from culturesim.model.params import ModelParams  # noqa: E402
from culturesim.model.runner import RunRequest, run_many  # noqa: E402
from culturesim.stats.spiketrains import SpikeRecording  # noqa: E402

BIN_S = 0.025


def crude_burst_summary(recording: SpikeRecording, n_neurons: int) -> dict[str, float]:
    """Population-activity summary for the tuning loop.

    Task 1 asks for bursts *separated by quiet periods*, which needs two numbers that
    are easy to conflate: how much of the network a burst recruits, and how active the
    baseline is between bursts. A network firing asynchronously at 1 Hz with one big
    event has the same burst count as a quiet network with one burst, and only the
    second is what the spec asks for.
    """
    n_bins = max(1, int(recording.duration / BIN_S))
    counts = np.bincount(
        np.minimum((recording.times / BIN_S).astype(np.int64), n_bins - 1), minlength=n_bins
    )
    if counts.sum() == 0:
        return {
            "n_bursts": 0.0,
            "median_ibi_s": float("nan"),
            "recruited": 0.0,
            "base_hz": 0.0,
        }

    # A burst bin is one recruiting at least 5% of the network, which is a property of
    # the network rather than of its mean rate.
    burst_bin = counts >= max(5.0, 0.05 * n_neurons)
    starts = np.flatnonzero(burst_bin & ~np.r_[False, burst_bin[:-1]])
    ibis = np.diff(starts) * BIN_S
    baseline_bins = counts[~burst_bin]
    return {
        "n_bursts": float(starts.size),
        "median_ibi_s": float(np.median(ibis)) if ibis.size else float("nan"),
        # Fraction of the network active in the largest bin.
        "recruited": float(counts.max() / n_neurons),
        # Baseline rate per neuron outside burst bins -- the "quiet period" activity.
        "base_hz": float(baseline_bins.mean() / (BIN_S * n_neurons)) if baseline_bins.size else 0.0,
        "quiet_fraction": float(np.mean(counts == 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--transient", type=float, default=5.0)
    parser.add_argument("--neurons", type=int, default=1000)
    parser.add_argument(
        "--depol",
        type=float,
        nargs="*",
        default=None,
        help=(
            "target mean background depolarisation in mV; derives rate_bg per w_e so the "
            "sweep follows a ridge of constant baseline excitability instead of a "
            "rectangle in which w_e and rate_bg fight each other"
        ),
    )
    parser.add_argument("--rate-bg", type=float, nargs="*", default=[3.5, 4.0, 4.5])
    parser.add_argument("--w-e", type=float, nargs="*", default=[0.85, 1.5])
    parser.add_argument("--p-conn", type=float, nargs="*", default=[0.12, 0.25])
    parser.add_argument("--g", type=float, nargs="*", default=[4.0])
    parser.add_argument("--b", type=float, nargs="*", default=[1.5])
    parser.add_argument("--tau-rec", type=float, nargs="*", default=[800.0])
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    config = load_config("model_default.yaml")
    config["simulation"].update(
        duration_s=args.duration, transient_s=args.transient, static_synapses=False
    )
    config["network"]["n_neurons"] = args.neurons
    base = ModelParams.from_config(config)

    if args.depol is None:
        grid = list(
            itertools.product(args.rate_bg, args.w_e, args.p_conn, args.g, args.b, args.tau_rec)
        )
    else:
        from culturesim.model.network import synaptic_current_jumps

        grid = []
        for depol, w_e, p_conn, g, b, tau_rec in itertools.product(
            args.depol, args.w_e, args.p_conn, args.g, args.b, args.tau_rec
        ):
            jump = synaptic_current_jumps(replace(base, free=base.free.replace(w_e=w_e)))
            # <v> = jump * N * rate * tau_e at steady state.
            rate_bg = depol / (
                jump["background"] * base.fixed.n_background_synapses * base.fixed.tau_e / 1000.0
            )
            grid.append((rate_bg, w_e, p_conn, g, b, tau_rec))
    requests = [
        RunRequest(
            params=replace(
                base,
                free=base.free.replace(
                    rate_bg=rate_bg, w_e=w_e, p_conn=p_conn, g=g, b=b, tau_rec=tau_rec
                ),
            ),
            run_index=index,
        )
        for index, (rate_bg, w_e, p_conn, g, b, tau_rec) in enumerate(grid)
    ]

    print(f"{len(requests)} draws, {args.duration}s biological each, {args.neurons} neurons")
    started = time.perf_counter()
    results = run_many(requests, n_workers=args.workers, on_error="skip")
    elapsed = time.perf_counter() - started
    print(f"completed {len(results)}/{len(requests)} in {elapsed:.0f}s\n")

    header = (
        f"{'rate_bg':>8}{'w_e':>7}{'p_conn':>8}{'g':>6}{'b':>6}{'tau_rec':>9}"
        f"{'Hz/neuron':>11}{'bursts':>8}{'medIBI':>8}{'recruit':>9}{'baseHz':>8}{'wall':>7}"
    )
    print(header)
    print("-" * len(header))
    print("  want:  baseHz well under 0.1, recruit over 0.2, medIBI between 1 and 60\n")
    for result in sorted(results, key=lambda r: r.run_index):
        rate_bg, w_e, p_conn, g, b, tau_rec = grid[result.run_index]
        summary = crude_burst_summary(result.recording, args.neurons)
        print(
            f"{rate_bg:8.2f}{w_e:7.2f}{p_conn:8.3f}{g:6.1f}{b:6.2f}{tau_rec:9.0f}"
            f"{result.diagnostics['neuron_mean_rate_hz']:11.3f}"
            f"{summary['n_bursts']:8.0f}{summary['median_ibi_s']:8.2f}"
            f"{summary['recruited']:9.3f}{summary['base_hz']:8.3f}"
            f"{result.wall_clock_s:7.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
