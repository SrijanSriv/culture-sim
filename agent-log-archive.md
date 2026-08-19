# Agent log archive

Entries moved out of `agent-log.md` once they stopped being relevant to future work:
superseded parameter sweeps, bugs that no longer exist, dead ends whose conclusion is
already recorded in the working brief.

This file is append-only history. It is not required reading. Consult it only when you
need to know *why* something was tried and rejected, or when a decision recorded in
`agent-log.md` seems arbitrary and you want the evidence behind it.

Newest entries at the top.

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
