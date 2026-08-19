#!/usr/bin/env python3
"""Quick end-to-end smoke run for development. Not part of the test suite.

.venv/bin/python scripts/smoke_run.py --duration 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.config import load_config  # noqa: E402
from culturesim.model.params import ModelParams  # noqa: E402
from culturesim.model.runner import RunRequest, run_one  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--transient", type=float, default=2.0)
    parser.add_argument("--neurons", type=int, default=None)
    parser.add_argument("--observe", action="store_true", help="apply the virtual MEA")
    parser.add_argument("--static", action="store_true", help="static-synapse ablation")
    args = parser.parse_args()

    config = load_config("model_default.yaml")
    config["simulation"]["duration_s"] = args.duration
    config["simulation"]["transient_s"] = args.transient
    config["simulation"]["static_synapses"] = args.static
    if args.neurons:
        config["network"]["n_neurons"] = args.neurons
    params = ModelParams.from_config(config)

    observation = load_config("observation.yaml") if args.observe else None
    started = time.perf_counter()
    result = run_one(RunRequest(params=params, observation_config=observation))
    elapsed = time.perf_counter() - started

    print(f"wall clock       : {elapsed:.1f}s for {args.duration}s biological")
    print(f"recording        : {result.recording!r}")
    for key, value in sorted(result.diagnostics.items()):
        print(f"  {key:22s}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
