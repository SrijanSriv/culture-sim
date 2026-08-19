# AGENTS.md

Guidance for AI agents working in this repository. Read this file and `agent-log.md`
before you touch anything.

## 0. The agent log — do this first and last

**First, read `agent-log.md`.** It records what previous agents did, what they
discovered, what they got wrong, and what is currently half-finished. Several findings in
it are non-obvious and expensive to rediscover, and at least one is a dead end you would
otherwise repeat.

**Last, append to `agent-log.md`.** Before you end a work session, add an entry covering:

- what you changed, and which SPEC task it belongs to
- what you measured, with the numbers (not "it got faster")
- anything you tried that did **not** work, so nobody repeats it
- what you left broken, unfinished, or uncommitted
- open questions and the next obvious step

Write for an agent who has none of your context and cannot see your reasoning.

**Keep the log useful by pruning it.** When an entry stops being relevant to future work
— a fixed typo, a superseded parameter sweep, a bug that no longer exists — move it to
`agent-log-archive.md` under the same heading structure. The archive is append-only
history; `agent-log.md` is the working brief. If `agent-log.md` has grown past roughly
200 lines, that is a signal to archive, not to add a summary at the top.

Never delete log content. Move it.

## 1. What this project is

An in-silico model of a dissociated neuronal culture on a multi-electrode array, built in
Brian2 and fitted by simulation-based inference. `README.md` explains it for humans.

**`SPEC.md` is the contract.** It defines the model, the statistics, the task order, and
the acceptance criterion for every task. When this file and `SPEC.md` disagree,
`SPEC.md` wins. It is not a wish list: §13 gives numbered tasks with acceptance criteria,
and a task is not done until its criterion is demonstrated, not merely coded.

`SPEC.md` has been edited mid-project before. Re-read the sections you depend on rather
than trusting a summary, and check `git diff SPEC.md` and `git log -- SPEC.md`.

## 2. Environment

Python lives in `.venv`. Always invoke it explicitly; there is no activated shell.

```bash
.venv/bin/python -m pytest -q                      # test suite
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
.venv/bin/python scripts/check_environment.py      # imports + Brian2 standalone compiles
.venv/bin/culture-sim --help
```

Dependencies are pinned **exactly** in `pyproject.toml`, per `SPEC.md` §2. Do not relax a
pin to `>=`. A fitted posterior is only reproducible against the versions that produced
it, and every run records them in its manifest.

Brian2 uses the `cpp_standalone` device, which needs a working C++ compiler.

## 3. Rules that invalidate results if broken

These are not style preferences. Each one, if violated, produces numbers that look fine
and are wrong.

**Statistics only ever run on electrode-level data.** The virtual MEA
(`culturesim/observation/virtual_mea.py`) is the observational bottleneck. Comparing
statistics from 1000 simulated neurons against statistics from 60 real electrodes is
invalid and is the most common error in this literature. Neuron-level recordings are
tagged `metadata["observation"] == "none"` so downstream code can refuse them.

**Delegate statistics to `cl.analysis`; never reimplement one it provides.** See
`SPEC.md` §6.0. Every call goes through a wrapper in `culturesim/interop/cl_analysis.py`
marked `# DELEGATED`, and nothing in `stats/` or `fit/` imports `cl.analysis` directly.
If you think an upstream statistic is wrong, write the objection into
`interop/CL_API_PROBE.md` and implement **both** under distinct names. Do not silently
substitute your own.

**All randomness flows from one master seed** through explicit `np.random.Generator`
instances. No `np.random.seed`, no bare `random`. `tests/test_reproducibility.py` parses
the package with an AST walker to enforce this, so a violation fails CI.

**One Brian2 simulation per process.** `cpp_standalone` cannot be reinitialised cleanly
many times in one process. `culturesim/model/runner.py` uses
`multiprocessing.Pool(maxtasksperchild=1)`; that argument is load-bearing, not tuning.

**The fingerprint order freezes at Task 3.** Adding a statistic later invalidates every
fit made against the old vector. `configs/fingerprint.yaml` carries a hash of the
expanded names and CI checks it.

**Verify a dataset actually resolves before writing a loader** (`SPEC.md` §7). Fetch the
URL. If it is gated, say so and stop. Never quietly substitute a different dataset.

## 4. How to work

**Measure; do not assert.** Any claim about speed, correctness, or "bursting now" needs a
number or a figure in the same session. `SPEC.md` §4.5 explicitly requires profiling
rather than guessing when the runtime budget is missed. `scripts/profile_runtime.py`,
`scripts/tune_task1.py`, and `scripts/check_background_modes.py` exist for this.

**Look at the figure, not just the summary statistic.** A parameter set once passed every
scalar burst check while the raster showed continuous asynchronous firing with a single
burst. The plot caught what the numbers hid.

**Report blockers instead of routing around them.** If a dependency is unavailable, a
dataset is gated, or the spec is internally inconsistent, that is a finding to write down
— in `agent-log.md` and to the user — not an obstacle to work around silently. A stub
that raises `NotImplementedError` with a precise message beats a plausible fake.

**Deviating from `SPEC.md` is sometimes correct, but never silent.** Where the spec is
underspecified or unworkable as literally written, implement what works and document the
reasoning at the point of the change, in the config comment *and* in the log. There are
existing examples in `culturesim/model/params.py`.

**Commits:** small, focused, one task's worth of work, present-tense subject. Match
`git log`. Do not commit `figures/`, `.venv`, or run outputs. Only commit when asked.

## 5. Code conventions

- Ruff, line length 100, `select = ["E", "F", "I", "UP", "B", "NPY"]`. Run it after edits.
- Frozen dataclasses for parameters and results; validate in `__post_init__`.
- Type hints throughout; `from __future__ import annotations` at the top.
- Import Brian2 **inside** functions that run in a subprocess, never at module scope, so
  that `culture-sim --help` stays fast and the device is configured per process.
- Comments explain *why*, particularly non-obvious constraints and unit conventions. Do
  not narrate what the code does, and do not explain your change to the reviewer.
- Tests: `pytest`, `hypothesis` for properties, `@pytest.mark.slow` for simulation-backed
  tests. `tests/test_stats_contracts.py` uses `xfail(strict=True)` so a placeholder turns
  into a failure the moment it starts passing — remove the marker, do not leave it.

## 6. Task state

`README.md` has the current status table. Trust `agent-log.md` over it for anything
in-flight; the table is updated at task boundaries, the log continuously.
