#!/usr/bin/env python3
"""Verify the environment can actually do what the spec needs.

Checks the two things that are cheap to test now and expensive to discover later:
Brian2's ``cpp_standalone`` device compiles and runs (SPEC §4.5 depends on it for the
runtime budget), and every hard dependency imports.

    .venv/bin/python scripts/check_environment.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from importlib import import_module
from pathlib import Path

REQUIRED_MODULES = (
    "brian2",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "cl",
    "sbi",
    "torch",
    "powerlaw",
    "yaml",
    "h5py",
    "pynwb",
)


def check_imports() -> list[str]:
    failures = []
    for name in REQUIRED_MODULES:
        try:
            module = import_module(name)
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        print(f"  ok   {name} {getattr(module, '__version__', '?')}")
    return failures


def check_cpp_standalone() -> list[str]:
    """Build and run a trivial standalone simulation in a temp directory.

    SPEC §4.5: every simulation runs in its own subprocess with its own build
    directory, so this mimics one such run rather than reusing the process device.
    """
    import brian2 as b2

    build_dir = Path(tempfile.mkdtemp(prefix="culturesim_envcheck_"))
    try:
        # Brian2's default extra_compile_args carry -std=c++11; overriding them drops
        # the standard flag and the build fails on 'auto'.
        b2.set_device("cpp_standalone", directory=str(build_dir), build_on_run=True)
        group = b2.NeuronGroup(
            100,
            "dv/dt = (1.1 - v) / (10*ms) : 1",
            threshold="v > 1",
            reset="v = 0",
            method="exact",
        )
        monitor = b2.SpikeMonitor(group)
        started = time.perf_counter()
        b2.run(200 * b2.ms)
        elapsed = time.perf_counter() - started
        if monitor.num_spikes == 0:
            return ["cpp_standalone ran but produced no spikes"]
        print(f"  ok   cpp_standalone: {monitor.num_spikes} spikes, {elapsed:.2f}s wall-clock")
        return []
    except Exception as exc:  # noqa: BLE001
        return [
            f"cpp_standalone failed: {type(exc).__name__}: {exc}\n"
            "       A working C++ compiler is required; the runtime target is too slow "
            "for the SPEC §4.5 budget (300 s biological time in under 60 s)."
        ]
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def main() -> int:
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print("imports:")
    failures = check_imports()
    print("brian2 standalone:")
    failures += check_cpp_standalone()

    if failures:
        print("\nFAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nenvironment OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
