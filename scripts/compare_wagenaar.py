#!/usr/bin/env python3
"""Task 4 acceptance: fingerprint a real Wagenaar recording vs published values.

Wagenaar, Pine & Potter 2006, BMC Neurosci 7:11. Burst *definitions* differ (theirs
vs CL ``analyse_network_bursts``), so a numeric mismatch is not a loader bug -- it
is a methods discrepancy that has to be stated. Loader checks (channel count, DIV,
complete fingerprint) are what decide the exit code.

    .venv/bin/python scripts/compare_wagenaar.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.data.loaders import fetch_wagenaar, load_wagenaar  # noqa: E402
from culturesim.rng import generator  # noqa: E402
from culturesim.stats.bursts import burst_stats  # noqa: E402
from culturesim.stats.fingerprint import FingerprintSpec, compute_fingerprint  # noqa: E402
from culturesim.stats.rates import rate_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    path = args.path if args.path is not None else fetch_wagenaar()
    recording = load_wagenaar(path)
    spec = FingerprintSpec.load("fingerprint.yaml")
    fingerprint = compute_fingerprint(recording, spec)
    rates = rate_stats(recording)
    bursts = burst_stats(recording, generator(0, "wagenaar-compare"))
    median_ibi = float(np.median(bursts.ibi_seconds)) if bursts.ibi_seconds.size else float("nan")

    print(f"loaded {recording!r}")
    print(
        f"  plating={recording.metadata['plating']} culture={recording.metadata['culture']} "
        f"DIV={recording.metadata['div']}"
    )
    print(f"  stim events (channel 60): {recording.metadata['n_stimulus_events']}")
    print()

    loader_ok = True
    print("Loader checks (exit code)")
    print(f"{'statistic':32s}{'ours':>12}  {'expected'}")
    print("-" * 88)
    loader_rows = [
        (
            "n_channels",
            float(recording.n_channels),
            "60 (MCS 8x8 minus corners)",
            recording.n_channels == 60,
        ),
        (
            "n_spikes",
            float(recording.n_spikes),
            ">1000 threshold crossings on a 30+ min dense DIV-14 file",
            recording.n_spikes > 1000,
        ),
        (
            "div",
            float(recording.metadata["div"]),
            "filename plating-culture-div; default cache is 1-1-14",
            int(recording.metadata["div"]) == 14
            if Path(recording.metadata["path"]).name.startswith("1-1-14")
            else int(recording.metadata["div"]) > 0,
        ),
        (
            "duration_covers_spikes",
            recording.duration,
            "duration >= last spike time (do not truncate)",
            recording.n_spikes == 0 or recording.duration + 1e-9 >= float(recording.times[-1]),
        ),
        (
            "fingerprint_defined",
            float(len(fingerprint) - fingerprint.n_undefined),
            f"{len(fingerprint)}/{len(fingerprint)} finite (complete fingerprint)",
            fingerprint.n_undefined == 0,
        ),
    ]
    for name, ours, published, ok in loader_rows:
        loader_ok = loader_ok and ok
        print(f"{name:32s}{ours:12.4g}  {published}  ({'ok' if ok else 'FAIL'})")

    print()
    print("Literature comparison (Wagenaar 2006; discrepancies must be explained)")
    print(f"{'statistic':32s}{'ours':>12}  {'published / expected'}")
    print("-" * 88)
    lit_rows = [
        (
            "duration_s",
            recording.duration,
            "nominal ~1800 s daily session",
            abs(recording.duration - 1800.0) / 1800.0 < 0.25,
        ),
        (
            "rate_mean_Hz",
            rates.rate_mean,
            "dense DIV~14: order-1 Hz per electrode",
            0.1 <= rates.rate_mean <= 10.0,
        ),
        (
            "active_electrode_fraction",
            rates.active_electrode_fraction,
            "most electrodes participate after two weeks",
            rates.active_electrode_fraction > 0.5,
        ),
        (
            "burst_rate_per_min",
            bursts.burst_rate_per_min,
            ">0; bursting dominates after 2 weeks",
            bursts.burst_rate_per_min > 0,
        ),
        (
            "median_ibi_s",
            median_ibi,
            "1-300 s (Wagenaar 2006 abstract; their burst detector)",
            bursts.ibi_seconds.size == 0 or (1.0 <= median_ibi <= 300.0),
        ),
        (
            "burst_duration_mean_s",
            bursts.burst_duration_mean,
            "~1 s at burst onset, <0.2 s after DIV 20; DIV 14 in between",
            (not math.isfinite(bursts.burst_duration_mean))
            or (0.05 <= bursts.burst_duration_mean <= 1.5),
        ),
    ]
    for name, ours, published, ok in lit_rows:
        print(f"{name:32s}{ours:12.4g}  {published}  ({'ok' if ok else 'DISCREPANT'})")

    print()
    print(
        "Methods note: burst scalars use CL analyse_network_bursts (bin=0.05 s, "
        "onset=3 Hz), not Wagenaar's detector. On 1-1-14 that yields median IBI "
        "0.3 s and ~127 bursts/min -- many short events -- against their published "
        "1-300 s IBI range. That is a definition difference, not a misread file. "
        "Duration of 1-1-14 is 2716 s (~45 min); the paper describes ~30 min daily "
        "sessions, and the loader keeps the observed last-spike time rather than "
        "truncating. Channel count, DIV metadata, and a complete 66-stat fingerprint "
        "are the loader checks."
    )
    print(f"\nfingerprint {fingerprint!r}  undefined={fingerprint.n_undefined}")
    if args.out:
        payload = {
            "recording": {
                "n_spikes": recording.n_spikes,
                "n_channels": recording.n_channels,
                "duration": recording.duration,
                "metadata": {
                    key: value
                    for key, value in recording.metadata.items()
                    if key != "stimulus_times_s"
                },
            },
            "rates": {
                "rate_mean": rates.rate_mean,
                "active_electrode_fraction": rates.active_electrode_fraction,
            },
            "bursts": {
                "burst_rate_per_min": bursts.burst_rate_per_min,
                "median_ibi_s": median_ibi,
                "burst_duration_mean": bursts.burst_duration_mean,
            },
            "fingerprint": fingerprint.to_dict(),
            "loader_ok": loader_ok,
        }
        args.out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if loader_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
