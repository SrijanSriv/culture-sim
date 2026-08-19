# Agent log archive

Entries moved out of `agent-log.md` once they stopped being relevant to future work:
superseded parameter sweeps, bugs that no longer exist, dead ends whose conclusion is
already recorded in the working brief.

This file is append-only history. It is not required reading. Consult it only when you
need to know *why* something was tried and rejected, or when a decision recorded in
`agent-log.md` seems arbitrary and you want the evidence behind it.

Newest entries at the top.

---

## 2026-08-19 — commit the parallel Task 0.5–4 work

Agent: Cursor Grok. Branch `main`.

Split the previously uncommitted tree into seven focused commits (see `git log`).
Verified before committing: `ruff check` + `ruff format --check` clean; `pytest -q`
→ **156 passed**. `figures/` stays gitignored.

Commits (newest last among this batch): SPEC/`cl-sdk` pin → CL interop + H5 → Task 1
network → virtual MEA → Task 3 stats → dataset access verification → README/log.

### Still open (unchanged by the commit pass)

Superseded by the Task 1 runtime-budget entry at the top. Remaining: Task 2 60-vs-1024
figure, fingerprint freeze, `load_wagenaar`.

---

## 2026-08-19 — Task 1 (network + runner), and a blocking spec change

Agent: Claude Opus 5. Branch `main`, originated at `7eec5cb`; technical findings below.
Procedural "uncommitted / parallel edits" framing was resolved by the commit pass above.

### Spec change that landed mid-Task-1 (archived detail)

Task 0.5 / §6.0 CL delegation / CL-native H5 / Python 3.12+ were added to `SPEC.md`
while the network work was in flight. Narrative moved to `agent-log-archive.md`; the
live contract is `SPEC.md`.

### `cl-sdk` — installed and probed (see `interop/CL_API_PROBE.md`)

Pin is `cl-sdk==1.0.0`; license recorded as CC BY-NC 4.0. Task 0.5 acceptance is met
by the probe document plus the adapter smoke tests.

### Dataset access: VERIFIED PUBLIC — Task 4 is unblocked

Both `SPEC.md` §7 targets are open, verified on 2026-08-19 by downloading real bytes
rather than by loading a landing page. States and evidence are recorded in
`culturesim/data/loaders.py`; `available_datasets()` now returns both, so the CLI's
access gate no longer refuses them.

- **`wagenaar2006` — PUBLIC, no registration.** Live at
  `neurodatasharing.bme.gatech.edu/development-data/`. Confirmed by downloading
  `simple-text/daily/spont/dense/1-1-14.spk.txt.bz2` (874,682 bytes, 224,547 events).
  Directory indexes are enabled. Citation requirement; bundled code GPL v2.
- **`braingeneers_organoid` — PUBLIC**, DANDI **001603**, OPEN, 111 assets / 322 GB,
  CC-BY-4.0, anonymous byte-range download confirmed. **Draft version only, no immutable
  published version**, so pin asset checksums in the manifest or a refit is not
  reproducible.

**The interesting failure mode here ran opposite to the one `SPEC.md` §7 warns about.**
The spec warns against assuming a dataset is public because a paper says so. In this case
the paper says the opposite of the truth: the BMC Neuroscience text says to email Steve
Potter for access, and the old `potterlab.gatech.edu` host is dead, so the publication
reads as gated while the archive is wide open. Verifying beats trusting the paper in
*either* direction.

Wagenaar gives **threshold crossings, not sorted units**, which is what `SPEC.md` §7
prefers — the virtual MEA models threshold detection, so sorted units would be compared
against something the observation model does not reproduce.

The full file schema needed to write the loader (path scheme, column format, 25 kHz,
0.3335 uV LSB, 8x8-minus-corners geometry, and the fact that **channel 60 is a stimulus
marker rather than an electrode**) is documented in the `load_wagenaar` docstring.

Also verified and recorded in `VERIFIED_ALTERNATIVES`, none of them a substitute without
saying so: **CRCNS hc-8** is GATED (free but manually-reviewed account request; otherwise
the closest match to Wagenaar and sorted); **DANDI 001611** is public, dissociated, 2-D
and the only candidate with an immutable published version, but its protocol is repeated
stimulation rather than spontaneous activity; **G-Node Kapucu** is public but 2.3 TiB and
on Axion plates rather than MCS. `EMPTY_DANDISETS` names four zero-asset dandisets that
rank well in search and waste time.

Remaining Task 4 work: write the loader, then satisfy the §13 acceptance criterion —
print fingerprint values alongside published values for the dataset and explain any
discrepancy.

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

### Runtime: 66 s projection — superseded

The 66 s / 0.1 ms projection and the "OpenMP not tried" note are closed by the
runtime-budget entry at the top of this file. Component costs from that profile are
in `agent-log-archive.md`.

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

Remaining Task 2: the 60-vs-1024-electrode fingerprint comparison figure. CL H5
round-trip for 60 electrodes is done; HD-MEA is blocked by `cl-sdk==1.0.0` uint8
channel ids (see `CL_API_PROBE.md`).

---

## 2026-08-19 — Task 1 runtime profile (superseded by the budget decision)

Historical component costs from `scripts/profile_runtime.py`, 1000 neurons, that led
to the 66 s projection later replaced by `dt = 0.2 ms`:

| component | cost |
|---|---|
| fixed (C++ compile) | 4.0 s |
| bare integration loop | 0.095 s per biological s |
| `PoissonInput` background | 0.105 s per biological s |
| recurrent synapses | 0.015 s per biological s |
| **projected 300 s** | **66 s vs 60 s budget** |

`dt = 0.2 ms` was 27% faster on that machine. The "coarsens avalanche timing" worry
was dropped: analysis bins are 50 ms. OpenMP was not available (Apple clang, no libomp).

---

## 2026-08-19 — mid-session SPEC growth (now committed)

Historical: while Task 1 was in flight, `SPEC.md` gained Task 0.5, §6.0 `cl.analysis`
delegation, CL-native H5 for the virtual MEA, and Python 3.12+ / exact `cl-sdk` pin.
That reshaped Tasks 2–3. The policy is now in `SPEC.md` itself; the interop layer and
stats wrappers landed in the commit pass. Kept only so the timeline of the parallel
edits is not lost.

---

## 2026-08-19 — Task 1 tuning trail (superseded by the final parameter set)

Kept because it maps the shape of the parameter space, which is not obvious and cost four
sweeps to establish. The conclusions are already in `agent-log.md`; these are the
measurements behind them. All runs 1000 neurons unless noted.

### Sweep 1 — the original defaults are silent, and `p_conn` does nothing

`rate_bg 3.5–4.5`, `w_e 0.85/1.5`, `p_conn 0.12/0.25`, `g 4`, `b 1.5`, `tau_rec 800`:

- `rate_bg 3.5, w_e 0.85`: 0.002 Hz/neuron, effectively silent.
- `rate_bg 4.0, w_e 0.85`: 0.565 Hz/neuron, asynchronous, peak/mean 2.0.
- `w_e 1.5` at any `rate_bg`: 18–34 Hz/neuron, tonic saturation, peak/mean 1.1.

The diagnostic finding: `p_conn 0.12` and `p_conn 0.25` gave near-identical firing rates.
Recurrent input was negligible against the background, which is what pointed at the EPSP
unit error rather than at the parameter values.

### Sweep 2 — high `w_e` with low `rate_bg`, before the units were fixed

`w_e 1.5–2.5`, `rate_bg 1.2–2.4`, `p_conn 0.25`. Still no bursting: the transition ran
straight from silent to asynchronous. Two rows showed single huge synchronous events near
the silent boundary (peak/mean 100 and 1600 with the network otherwise dead), which
confirmed the burst *mechanism* worked once triggered and that the problem was the
trigger. This is what motivated computing the membrane attenuation analytically.

### Sweep 3 — after the EPSP conversion, along a constant-depolarisation ridge

Sweeping `rate_bg` and `w_e` independently is misleading once both scale the background,
so `scripts/tune_task1.py --depol` derives `rate_bg` from a target mean depolarisation.
At `depol 14–18`, `w_e 0.5–1.5`, `p_conn 0.15`, `g 4`: rates a healthy 0.4–8.7 Hz/neuron
but peak/mean only 2–4. Lowering `g` from 4 to 2 was what first produced bursts, since
balanced E/I actively suppresses synchrony.

### Sweep 4 — after decoupling the background amplitude

`depol 15.5–16.4`, `w_e 1.5/2.0`, `p_conn 0.30`, `g 2`, `b 2.5`, `tau_rec 2500`, 120 s:

| `rate_bg` | `w_e` | Hz/neuron | bursts | median IBI | baseline Hz |
|---|---|---|---|---|---|
| 4.88 | 1.5 | 0.278 | 5 | 14.8 s | 0.002 |
| 4.98 | 1.5 | 1.035 | 19 | 6.1 s | 0.006 |
| 5.07 | 1.5 | 1.980 | 39 | 3.2 s | 0.020 |
| 5.17 | 1.5 | 2.995 | 63 | 1.9 s | 0.045 |

A clean monotonic family — `rate_bg` moves the burst rate continuously once the units are
right. `p_conn` was then set to 0.20 rather than 0.30 to keep it off the prior boundary,
costing burst rate (5.4/min instead of 9.8/min) for a value an optimiser can move in both
directions.

### Metric that misled, and its replacement

The first tuning metric counted bins exceeding `max(3, 4 x mean)` as bursts. Because it
was relative to the mean, an asynchronous network at 1 Hz/neuron and a quiet network with
one real burst scored the same. Replaced with two absolute numbers: `recruited` (fraction
of the network active in the largest bin) and `base_hz` (rate outside burst bins). Only
the second distinguishes "bursts separated by quiet periods" from "continuous firing with
a spike in it", which is what `SPEC.md` §13 Task 1 actually asks for.

---

## 2026-08-19 — Bugs fixed in the same session they were introduced

No longer present in the code. Recorded only so a future agent recognises the shapes.

**`NetworkOperation` cannot run under `cpp_standalone`.** The first stimulus
implementation (`SPEC.md` §9.3, Task 7) cleared `I_ext` from a Python callback on a
`NetworkOperation`. That silently cannot work in standalone mode. Rewritten to deliver
stimulus pulses into `I_e` through a `SpikeGeneratorGroup` and a fan-out `Synapses`, one
generator per distinct pulse amplitude, so the injected charge decays with `tau_e` and no
Python runs inside the loop.

**Variable shadowing in `scripts/figure_task1_bursts.py`.** A summary dict named `static`
shadowed the `RunResult` named `static`, so the figure block raised
`AttributeError: 'dict' object has no attribute 'recording'` *after* printing the
acceptance results. Renamed to `static_summary`.

**`prefs.codegen.cpp.extra_compile_args` must not be overridden.** From the Task 0
session, repeated here because it recurs: setting it to `["-O2"]` drops Brian2's default
`-std=c++11` and the build fails on `auto`. Leave the defaults alone.
