# Agent log

Working brief for agents. Read this before starting; append to it before finishing. See
`AGENTS.md` §0 for the protocol. Older or superseded entries live in
`agent-log-archive.md`.

Newest entries at the top.

---

## 2026-08-19 — Task 1 (network + runner), and a blocking spec change

Agent: Claude Opus 5. Branch `main`, last commit `7eec5cb` ("complete basic setup").

### State at end of session: work is UNCOMMITTED, and another agent is editing in parallel

Mine, modified: `configs/model_default.yaml`,
`culturesim/model/{network,params,runner}.py`,
`culturesim/observation/virtual_mea.py`.
Mine, untracked: `culturesim/figures.py`,
`scripts/{smoke_run,tune_task1,profile_runtime,figure_task1_bursts,check_background_modes}.py`,
`figures/task1_bursts_vs_static.{png,json}`.

**Not mine**, changed during the same session by another agent or the user, and not
reviewed by me: `SPEC.md`, `pyproject.toml`, `culturesim/interop/*`,
`culturesim/stats/{avalanche,branching,bursts,connectivity,fingerprint,rates,spiketrains}.py`,
`configs/fingerprint.yaml`, `culturesim/{__init__,manifest}.py`,
`.github/workflows/ci.yml`, `scripts/check_environment.py`. That is ~1000 lines across
`stats/` alone, so Task 0.5/Task 3 work is evidently in flight. Reconcile before assuming
anything in `stats/` matches what this entry describes.

Tests have **not** been run since these changes. `ruff` has not been run. Do that first.

### `SPEC.md` grew mid-session and added a task that blocks Tasks 2 and 3

`SPEC.md` gained 95 lines while I was working (`git diff SPEC.md`). The additions
introduce a dependency the earlier scaffold knew nothing about:

- **Task 0.5 — CL API probe**, which the spec says to do *before anything else*
  (§13). Install `cl-sdk`, empirically determine what `cl.analysis` accepts and returns,
  write `interop/CL_API_PROBE.md`.
- **§6.0 delegation policy**: burst detection, avalanche sizes/durations/shapes, and
  functional connectivity must come **from `cl.analysis`**, not from our own code. Only
  rates/ISI, IBI distribution, power-law fitting, crackling-noise, MR branching, and
  fingerprint assembly stay ours.
- **§6.0.1**: the virtual MEA must write **CL-format H5 as its native output**, and
  `SpikeRecording` must round-trip to and from it.
- **§2**: `cl-sdk` must be pinned exactly, and Python **3.12+** is required.

This substantially reshapes Task 3, adds a requirement to Task 2, and means the existing
stub bodies in `stats/{bursts,avalanche,connectivity}.py` should become thin consumers of
`interop/cl_analysis.py` rather than implementations.

`culturesim/interop/__init__.py` and `culturesim/interop/cl_adapter.py` (~8.5 KB) already
exist on disk, created at 21:31 and not by me. I have not read or verified them.

### `cl-sdk` availability: resolves on PyPI, NOT yet installed or verified

`pip index versions cl-sdk` → `1.0.0` (also 0.29.0, 0.1.0). A `--dry-run` install
resolves cleanly and would pull `tables`, `pydantic`, `python-louvain`, `msgpack`,
`websockets`, `ipykernel` among others. So the package is real and reachable.

`pyproject.toml` has since been updated by the parallel agent to pin `cl-sdk==1.0.0` and
require Python 3.12+ (ruff target `py312`). The venv is Python 3.13.3, so that is
satisfied.

**Still not done as of this entry:** actual install into `.venv`, the license check that
`SPEC.md` §2 requires *before Task 0 completes*, an import test, or any probe of
`cl.analysis`. A pin in `pyproject.toml` is not a verification. Confirm the current state
before assuming Task 0.5 is open or closed.

### Dataset access verification is in flight

A background subagent was dispatched to verify whether the Wagenaar/Potter developmental
MEA dataset and a Braingeneers DANDI organoid dataset are actually downloadable today.
**Its result was not received before this session ended.** Task 4 stays blocked until
someone re-runs that check. Do not write a loader first.

### Task 1: the network now bursts, and three model corrections were needed

The network was silent at every point in the prior box as originally written. Three
separate causes, all worth understanding because each is invisible in the equations:

**1. `rate_bg` needs a background afferent count.** With one afferent per neuron,
crossing the 20 mV rest-to-threshold gap needs ~4.7 kHz of drive — three orders of
magnitude outside the 0.1–20 Hz prior. Added `n_background_synapses = 1000` to
`FixedParams` (Brunel 2000 uses `C_ext = C_E = 1000`).

**2. `w_e` is an EPSP amplitude, not a current jump.** This is the important one. Fed
directly into `I_e`, a `w_e` of 0.85 mV produces a membrane deflection of only ~0.03 mV,
because the current is attenuated by the membrane filter and then scaled by `U`.
Recurrent excitation therefore cannot recruit anything, at any parameter setting. The
peak deflection per unit current jump is
`(tau_syn/(tau_m - tau_syn)) * (exp(-t*/tau_m) - exp(-t*/tau_syn))` — **0.157** at the
defaults, so a synapse must carry ~6.4x its target EPSP amplitude. See
`epsp_peak_ratio()` and `synaptic_current_jumps()` in `model/network.py`. Weights are
also divided by `U` so `w_e` means the amplitude a *rested* synapse delivers and `U`
controls only dynamics.

**3. The background amplitude must be decoupled from `w_e`.** While the background was
driven at `w_e`, baseline membrane noise scaled with recurrent gain: raising `w_e` to get
bursts to ignite simultaneously raised the spontaneous rate into tonic firing, so no
sparse-baseline bursting regime existed anywhere in the prior. Added
`w_background = 0.1 mV` to `FixedParams`. Now `rate_bg` sets how far below threshold the
network idles and `w_e` sets how explosively it recruits.

Both new constants are absent from `SPEC.md` §4.4. Both are required for the model to
function; the reasoning is documented at each field in `model/params.py`.

### Hand-tuned Task 1 parameters (in `configs/model_default.yaml`)

`p_conn 0.20, w_e 1.5, g 2.0, rate_bg 4.98, tau_m 20, U 0.25, tau_rec 2500, b 2.5`

Result at 300 s: 27 bursts (5.4/min), median IBI 10.0 s, baseline **0.007 Hz/neuron**
between bursts. Three of these are sharp — do not nudge them casually:

- `rate_bg` puts the mean background depolarisation at ~15.8 mV against a 20 mV gap. The
  window is narrow: 4.41 Hz is silent, 5.35 Hz is continuously active with no quiet
  periods.
- `g` must stay low. By `g = 4` excitation and inhibition balance, which suppresses
  synchrony into an asynchronous irregular state with **no bursts at all**.
- `tau_rec` sets the IBI, since depleted synaptic resources are what end a burst.

### Static-synapse ablation passes, but not the way you would expect

It does not go silent — it degenerates into a **metronome**: median IBI 1.03 s with
CV 0.11, at 17.7 Hz/neuron versus 0.91 for the STP network. So the acceptance check is
"not culture-like bursting" (degenerate IBI spread or a far higher rate), not "fewer
bursts". A count-based check passes the ablation and is wrong.

### Runtime: 66 s projected for 300 s, against a 60 s budget — OPEN

`scripts/profile_runtime.py`, one run at a time, 1000 neurons:

| component | cost |
|---|---|
| fixed (C++ compile) | 4.0 s |
| bare integration loop | 0.095 s per biological s |
| `PoissonInput` background | 0.105 s per biological s |
| recurrent synapses | 0.015 s per biological s |
| **projected 300 s** | **66 s vs 60 s budget** |

10% over. Options not yet taken: `dt = 0.2 ms` measured 27% faster (would give ~46 s) but
coarsens avalanche timing; OpenMP for solo runs (Apple clang needs libomp). `SPEC.md`
§4.5 asks for profiling before proceeding, which is done — the decision is not.

### Dead end: do not retry the diffusion-approximation background

I replaced `PoissonInput` with an equivalent Ornstein-Uhlenbeck current in `I_e` (moments
matched exactly: mean `J*R*tau_e`, variance `J^2*R*tau_e/2`) expecting a large speedup,
since the background was 6.3 s of a 16.9 s run.

**It was 0.91x — slower, not faster.** Brian2's `xi` noise costs one `randn` per neuron
per timestep, which is no cheaper than the binomial sampler. It also shifted statistics:
rate 0.433 vs 0.565 Hz/neuron, 8.7 vs 11.3 bursts, because the network sits close enough
to threshold to be sensitive to fluctuation shape.

The code is kept as `background_mode = "poisson" | "diffusion"` with `poisson` as the
default, so the approximation stays testable. `scripts/check_background_modes.py`
reproduces the comparison. Do not spend time on this again.

### Task 2 partially done

`observe()` in `observation/virtual_mea.py` is implemented: per-neuron amplitude scatter,
distance decay, detection, per-electrode dead time, dead electrodes. Detection uses
`p = Phi((A - k*rms)/rms)` per (neuron, electrode) pair — analytically identical to
drawing Gaussian noise per spike per electrode, but one evaluation per pair instead of
~90 million draws per 300 s run.

**Not done, and now also needs the CL work:** the 60-vs-1024-electrode fingerprint
comparison figure, and the `interop/cl_adapter.py` H5 round-trip that `SPEC.md` §13
Task 2 now requires.

### Next steps, in order

1. Run `pytest` and `ruff`; several dataclasses gained fields and no test has seen them.
2. **Task 0.5**: install and license-check `cl-sdk`, probe `cl.analysis`, write
   `interop/CL_API_PROBE.md`. It gates Tasks 2 and 3. Read the existing `cl_adapter.py`
   first.
3. Decide the runtime budget question and write the decision down.
4. Re-run dataset access verification for Task 4.
5. Update `README.md`'s status table — it still says Task 1 "Not started".
