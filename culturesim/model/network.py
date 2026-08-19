"""Brian2 network builder (SPEC §4).

Task 1. The equations are written out here now because they are fully specified and
because two of them are easy to get subtly wrong:

* The Tsodyks-Markram updates in ``ON_PRE_EXCITATORY`` must execute in exactly the
  order given in SPEC §4.2 -- ``u`` is incremented, then the current is delivered
  using the new ``u`` and the *old* ``x``, then ``x`` is depleted. Any other order
  changes the effective synaptic gain.
* ``(event-driven)`` on the ``u`` and ``x`` equations makes Brian2 integrate them
  lazily at presynaptic spikes instead of every timestep. With 1000 neurons and
  ~10^5 synapses that is the difference between a usable simulator and an unusable
  one (SPEC §4.2).

Short-term plasticity is a structural requirement, not a refinement: a network with
static synapses has no mechanism to terminate a network burst, so it cannot produce
realistic inter-burst intervals at any parameter setting. ``static_synapses: true``
in the simulation config builds that ablation deliberately, for the Task 1
acceptance test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .params import ModelParams

__all__ = [
    "NEURON_EQUATIONS",
    "neuron_equations",
    "background_moments",
    "epsp_peak_ratio",
    "synaptic_current_jumps",
    "sample_connections",
    "SYNAPSE_MODEL_STP",
    "SYNAPSE_MODEL_STATIC",
    "ON_PRE_EXCITATORY",
    "ON_PRE_INHIBITORY",
    "ON_PRE_STATIC_EXCITATORY",
    "ON_PRE_STATIC_INHIBITORY",
    "NeuronPositions",
    "place_neurons",
    "connection_probability",
    "build_network",
]

# SPEC §4.1. Current-based LIF with spike-frequency adaptation; all synaptic
# "currents" are in volts, i.e. already scaled by the membrane resistance.
_NEURON_EQUATIONS_TEMPLATE = """
dv/dt   = (E_L - v + I_e + I_i + I_ext - a) / tau_m : volt (unless refractory)
{excitatory_current}
dI_i/dt = -I_i / tau_i                              : volt
da/dt   = -a / tau_a                                : volt
I_ext                                               : volt
x_pos                                               : meter
y_pos                                               : meter
"""

# Background drive arrives as explicit Poisson events from a PoissonInput object.
_I_E_POISSON = "dI_e/dt = -I_e / tau_e                              : volt"

# Background drive folded into I_e as an Ornstein-Uhlenbeck process. For a Poisson train
# of jumps J at total rate R through an exponential filter of time constant tau_e, the
# stationary current has mean J*R*tau_e and variance J^2*R*tau_e/2; an OU process with
# those two moments matches them exactly. The approximation is in the higher moments
# within a single timestep, which the membrane -- integrating over tau_m = 200 timesteps
# and ~100 background events -- averages away.
_I_E_DIFFUSION = "dI_e/dt = (mu_bg - I_e) / tau_e + sigma_bg * sqrt(2 / tau_e) * xi : volt"


def neuron_equations(background_mode: str = "poisson") -> str:
    """Membrane equations, with the background drive folded in or left explicit."""
    if background_mode == "diffusion":
        return _NEURON_EQUATIONS_TEMPLATE.format(excitatory_current=_I_E_DIFFUSION)
    if background_mode == "poisson":
        return _NEURON_EQUATIONS_TEMPLATE.format(excitatory_current=_I_E_POISSON)
    raise ValueError(f"unknown background_mode {background_mode!r}")


def background_moments(params: ModelParams) -> tuple[float, float]:
    """Stationary mean and standard deviation of the background current, in mV."""
    jump = synaptic_current_jumps(params)["background"]
    total_rate = params.fixed.n_background_synapses * params.free.rate_bg
    tau_e_s = params.fixed.tau_e / 1000.0
    return jump * total_rate * tau_e_s, jump * float(np.sqrt(total_rate * tau_e_s / 2.0))


# Retained for callers that want the default (poisson) form directly.
NEURON_EQUATIONS = neuron_equations()

THRESHOLD = "v > v_th"
RESET = "v = v_reset; a += b"

# SPEC §4.2. Tsodyks-Markram short-term plasticity, integrated lazily.
SYNAPSE_MODEL_STP = """
w                          : volt
du/dt = -u / tau_f         : 1 (event-driven)
dx/dt = (1 - x) / tau_rec  : 1 (event-driven)
"""

SYNAPSE_MODEL_STATIC = """
w : volt
"""

# Order is load-bearing; see the module docstring.
ON_PRE_EXCITATORY = """
u += U * (1 - u)
I_e_post += w * u * x
x -= u * x
"""

ON_PRE_INHIBITORY = """
u += U * (1 - u)
I_i_post -= w * u * x
x -= u * x
"""

# Ablation only (SPEC §13, Task 1): demonstrably fails to terminate bursts.
ON_PRE_STATIC_EXCITATORY = "I_e_post += w"
ON_PRE_STATIC_INHIBITORY = "I_i_post -= w"


def epsp_peak_ratio(tau_syn_ms: float, tau_m_ms: float) -> float:
    """Peak membrane deflection per unit synaptic current jump.

    For ``tau_m dv/dt = -v + A exp(-t/tau_syn)`` the response peaks at
    ``t* = tau_syn tau_m ln(tau_m/tau_syn) / (tau_m - tau_syn)`` with value
    ``A (tau_syn/(tau_m - tau_syn)) (exp(-t*/tau_m) - exp(-t*/tau_syn))``. This returns
    that value divided by ``A``.

    Why this exists: SPEC §4.4 gives ``w_e`` in millivolts over 0.05-3.0, which is the
    range of *unitary EPSP amplitudes* measured at the soma. Feeding ``w_e`` directly
    into ``I_e`` as a current jump instead produces a membrane deflection of only
    ``w_e * ratio * U`` -- about 0.03 mV at the defaults -- so recurrent excitation
    cannot recruit anything and the network never bursts at any point in the prior box.
    Converting through this ratio makes ``w_e`` mean what its units say it means.

    At the defaults (tau_e = 5 ms, tau_m = 20 ms) the ratio is 0.157, so a synapse must
    carry roughly 6.4x its target EPSP amplitude as a current jump.
    """
    tau_syn, tau_m = float(tau_syn_ms), float(tau_m_ms)
    if tau_syn <= 0 or tau_m <= 0:
        raise ValueError(f"time constants must be positive, got {tau_syn}, {tau_m}")
    if abs(tau_m - tau_syn) < 1e-9:
        # Degenerate case: the two exponentials coincide and v(t) = A t/tau_m e^{-t/tau_m},
        # peaking at t = tau_m with value A/e.
        return float(np.exp(-1.0))
    t_peak = tau_syn * tau_m * np.log(tau_m / tau_syn) / (tau_m - tau_syn)
    return float(
        (tau_syn / (tau_m - tau_syn)) * (np.exp(-t_peak / tau_m) - np.exp(-t_peak / tau_syn))
    )


def synaptic_current_jumps(params: ModelParams) -> dict[str, float]:
    """Convert EPSP/IPSP amplitudes in mV into the current jumps Brian2 needs.

    Returns the raw ``w`` values in mV for the excitatory, inhibitory and background
    synapse populations. The excitatory and inhibitory values additionally divide by
    ``U``, so that ``w_e`` is the amplitude a *rested* synapse delivers (``u = U``,
    ``x = 1``) and ``U`` therefore controls only short-term dynamics rather than also
    rescaling the static gain. Background afferents carry no short-term plasticity, so
    they are not divided by ``U``, and they use the fixed ``w_background`` amplitude
    rather than ``w_e`` -- see the comment on that field in ``params.py``.
    """
    free, fixed = params.free, params.fixed
    excitatory_ratio = epsp_peak_ratio(fixed.tau_e, free.tau_m)
    inhibitory_ratio = epsp_peak_ratio(fixed.tau_i, free.tau_m)
    if free.U <= 0:
        raise ValueError(f"U must be positive, got {free.U}")
    return {
        "excitatory": free.w_e / (free.U * excitatory_ratio),
        "inhibitory": params.w_i / (free.U * inhibitory_ratio),
        "background": fixed.w_background / excitatory_ratio,
        "excitatory_peak_ratio": excitatory_ratio,
        "inhibitory_peak_ratio": inhibitory_ratio,
    }


@dataclass(frozen=True)
class NeuronPositions:
    """Neuron placement on the 2-D sheet, in micrometres (SPEC §4.3).

    Handed to the virtual MEA, which needs neuron-to-electrode distances.
    """

    x_um: np.ndarray
    y_um: np.ndarray
    is_excitatory: np.ndarray

    def __post_init__(self) -> None:
        if not (self.x_um.shape == self.y_um.shape == self.is_excitatory.shape):
            raise ValueError("position and type arrays must have the same shape")

    @property
    def n_neurons(self) -> int:
        return int(self.x_um.size)

    def distances_to(self, x_um: np.ndarray, y_um: np.ndarray) -> np.ndarray:
        """Distance matrix ``(n_neurons, n_targets)`` in micrometres."""
        dx = self.x_um[:, None] - np.asarray(x_um, dtype=np.float64)[None, :]
        dy = self.y_um[:, None] - np.asarray(y_um, dtype=np.float64)[None, :]
        return np.sqrt(dx * dx + dy * dy)


def place_neurons(params: ModelParams, rng: np.random.Generator) -> NeuronPositions:
    """Uniform random placement on the sheet, with excitatory/inhibitory labels.

    Uniform rather than lattice: a dissociated culture is not ordered, and a
    lattice would impose a spurious spatial periodicity on the avalanche
    statistics.

    Excitatory neurons occupy the low indices, so ``0:n_excitatory`` and
    ``n_excitatory:`` are contiguous subgroups in Brian2.
    """
    network = params.network
    # Electrode layouts are centred on the origin (see observation.yaml). The sheet
    # must share that origin or the virtual MEA only sees one corner of the culture.
    x_um = rng.uniform(-0.5, 0.5, size=network.n_neurons) * network.sheet_width_um
    y_um = rng.uniform(-0.5, 0.5, size=network.n_neurons) * network.sheet_height_um
    is_excitatory = np.zeros(network.n_neurons, dtype=bool)
    is_excitatory[: network.n_excitatory] = True
    return NeuronPositions(x_um=x_um, y_um=y_um, is_excitatory=is_excitatory)


def connection_probability(distance_um: np.ndarray, lambda_conn_um: float, p_conn: float):
    """``p(d) = p_conn * exp(-d / lambda_conn)`` (SPEC §4.3)."""
    return p_conn * np.exp(-np.asarray(distance_um, dtype=np.float64) / float(lambda_conn_um))


def sample_connections(
    positions: NeuronPositions,
    params: ModelParams,
    rng: np.random.Generator,
    source_slice: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw distance-dependent connections from a subgroup to the whole network.

    Sampled here with an explicit Generator rather than by Brian2's ``connect(p=...)``
    so that topology is reproducible from the master seed independently of Brian2's own
    RNG, and so the same topology can be rebuilt for the static-synapse ablation.
    """
    source_index = np.arange(positions.n_neurons)[source_slice]
    distances = positions.distances_to(positions.x_um, positions.y_um)[source_index, :]
    probability = connection_probability(distances, params.lambda_conn_um, params.free.p_conn)

    if not params.network.allow_autapses:
        # Row r of `probability` is source neuron source_index[r]; zero its own column.
        probability[np.arange(source_index.size), source_index] = 0.0

    connected = rng.random(probability.shape) < probability
    rows, cols = np.nonzero(connected)
    return source_index[rows].astype(np.int32), cols.astype(np.int32)


def build_network(
    params: ModelParams,
    positions: NeuronPositions,
    seed: int,
    *,
    stimulus: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Assemble the Brian2 network.

    Returns the network and a dict of its groups and monitors. Must only be called
    inside a runner subprocess: SPEC §4.5 forbids repeated ``device.reinit()`` in a
    single process under ``cpp_standalone``.

    ``stimulus`` is a :class:`~culturesim.validate.perturbation.StimulusProtocol`-shaped
    mapping used by the Task 7 perturbation test; ``None`` gives spontaneous activity.
    """
    import brian2 as b2

    fixed, free, network_params, simulation = (
        params.fixed,
        params.free,
        params.network,
        params.simulation,
    )
    b2.seed(seed)
    rng = np.random.default_rng(seed)

    namespace = {
        "E_L": fixed.E_L * b2.mV,
        "v_th": fixed.v_th * b2.mV,
        "v_reset": fixed.v_reset * b2.mV,
        "tau_e": fixed.tau_e * b2.ms,
        "tau_i": fixed.tau_i * b2.ms,
        "tau_a": fixed.tau_a * b2.ms,
        "tau_f": fixed.tau_f * b2.ms,
        "tau_m": free.tau_m * b2.ms,
        "tau_rec": free.tau_rec * b2.ms,
        "U": free.U,
        "b": free.b * b2.mV,
    }
    mean_bg, sigma_bg = background_moments(params)
    if simulation.background_mode == "diffusion":
        namespace["mu_bg"] = mean_bg * b2.mV
        namespace["sigma_bg"] = sigma_bg * b2.mV

    neurons = b2.NeuronGroup(
        network_params.n_neurons,
        neuron_equations(simulation.background_mode),
        threshold=THRESHOLD,
        reset=RESET,
        refractory=fixed.t_ref * b2.ms,
        method="euler",
        namespace=namespace,
        name="neurons",
    )
    neurons.x_pos = positions.x_um * b2.umetre
    neurons.y_pos = positions.y_um * b2.umetre
    # Start scattered between reset and threshold so the network does not open with an
    # artefactual synchronous volley that would be mistaken for a network burst.
    neurons.v = (
        fixed.v_reset
        + rng.uniform(0.0, 1.0, network_params.n_neurons) * (fixed.v_th - fixed.v_reset)
    ) * b2.mV

    excitatory = neurons[: network_params.n_excitatory]
    inhibitory = neurons[network_params.n_excitatory :]

    jumps = synaptic_current_jumps(params)

    # Independent background drive per neuron (SPEC §4.3): `n_background_synapses`
    # afferents each firing at `rate_bg`. In diffusion mode this is already inside the
    # membrane equations, so there is no separate object.
    background = None
    if simulation.background_mode == "poisson":
        background = b2.PoissonInput(
            neurons,
            target_var="I_e",
            N=fixed.n_background_synapses,
            rate=free.rate_bg * b2.Hz,
            weight=jumps["background"] * b2.mV,
        )
    else:
        # Start at the stationary mean so the network does not spend a membrane time
        # constant charging up at the beginning of the transient.
        neurons.I_e = mean_bg * b2.mV

    use_stp = not simulation.static_synapses
    synapse_model = SYNAPSE_MODEL_STP if use_stp else SYNAPSE_MODEL_STATIC
    delay_low, delay_high = fixed.synaptic_delay_ms

    # The static ablation keeps the *rested* synapse strength identical to the STP model
    # (u = U, x = 1), so the comparison isolates the loss of depression rather than also
    # changing the gain -- otherwise the ablation would prove nothing (SPEC §13, Task 1).
    static_scale = free.U if simulation.static_synapses else 1.0

    synapse_groups = []
    for label, source, source_slice, on_pre, on_pre_static, weight_mv in (
        (
            "exc",
            excitatory,
            slice(0, network_params.n_excitatory),
            ON_PRE_EXCITATORY,
            ON_PRE_STATIC_EXCITATORY,
            jumps["excitatory"] * static_scale,
        ),
        (
            "inh",
            inhibitory,
            slice(network_params.n_excitatory, network_params.n_neurons),
            ON_PRE_INHIBITORY,
            ON_PRE_STATIC_INHIBITORY,
            jumps["inhibitory"] * static_scale,
        ),
    ):
        synapses = b2.Synapses(
            source,
            neurons,
            model=synapse_model,
            on_pre=on_pre if use_stp else on_pre_static,
            namespace=namespace,
            method="exact",
            name=f"synapses_{label}",
        )
        pre_index, post_index = sample_connections(positions, params, rng, source_slice)
        if pre_index.size == 0:
            # An empty Synapses object is legal in Brian2 and this happens for small
            # p_conn during SBI; connect() with empty arrays would raise.
            synapse_groups.append(synapses)
            continue
        # Synapses over a subgroup index relative to that subgroup's start.
        synapses.connect(i=pre_index - source_slice.start, j=post_index)
        synapses.w = weight_mv * b2.mV
        synapses.delay = rng.uniform(delay_low, delay_high, size=pre_index.size) * b2.ms
        if use_stp:
            synapses.u = 0.0
            synapses.x = 1.0
        synapse_groups.append(synapses)

    spike_monitor = b2.SpikeMonitor(neurons, name="spikes")
    objects = [neurons, *synapse_groups, spike_monitor]
    if background is not None:
        objects.append(background)

    stimulus_generator = None
    if stimulus is not None:
        stimulus_generator = _build_stimulus(neurons, positions, stimulus, b2)
        objects.extend(stimulus_generator)

    components = {
        "neurons": neurons,
        "excitatory": excitatory,
        "inhibitory": inhibitory,
        "background": background,
        "synapses_exc": synapse_groups[0],
        "synapses_inh": synapse_groups[1],
        "spike_monitor": spike_monitor,
        "n_synapses_exc": int(len(synapse_groups[0])),
        "n_synapses_inh": int(len(synapse_groups[1])),
        "stp_enabled": use_stp,
        "current_jumps_mv": jumps,
        "background_mode": simulation.background_mode,
        "background_mean_mv": mean_bg,
        "background_sigma_mv": sigma_bg,
    }
    return b2.Network(*objects), components


def _build_stimulus(neurons: Any, positions: NeuronPositions, stimulus: Any, b2: Any) -> list[Any]:
    """Current injection into neurons near a stimulating electrode (SPEC §9.3).

    Amplitude falls off exponentially with distance from the electrode, so the set of
    neurons a stimulus can recruit resembles the set an electrode can see.

    Delivered into ``I_e`` rather than into ``I_ext``, which makes the injected charge
    decay with ``tau_e`` instead of persisting. Two constraints force this: a
    ``NetworkOperation`` that resets ``I_ext`` in Python cannot run under
    ``cpp_standalone``, and a stimulus current that never decays is not a pulse.

    One generator per distinct amplitude, because a synapse carries a single weight and
    the protocol sweeps amplitude (SPEC §9.3).
    """
    times_s = np.asarray(stimulus["times_s"], dtype=np.float64)
    amplitudes_mv = np.asarray(stimulus["amplitudes_mv"], dtype=np.float64)
    if times_s.size != amplitudes_mv.size:
        raise ValueError(f"stimulus has {times_s.size} times but {amplitudes_mv.size} amplitudes")

    electrode_x = float(stimulus["electrode_x_um"])
    electrode_y = float(stimulus["electrode_y_um"])
    radius_um = float(stimulus.get("radius_um", 100.0))

    distances = np.sqrt((positions.x_um - electrode_x) ** 2 + (positions.y_um - electrode_y) ** 2)
    coupling = np.exp(-distances / radius_um)
    all_targets = np.arange(positions.n_neurons)

    objects: list[Any] = []
    for index, amplitude in enumerate(np.unique(amplitudes_mv)):
        pulse_times = times_s[amplitudes_mv == amplitude]
        generator = b2.SpikeGeneratorGroup(
            1,
            np.zeros(pulse_times.size, dtype=int),
            pulse_times * b2.second,
            name=f"stim_generator_{index}",
        )
        injection = b2.Synapses(
            generator,
            neurons,
            model="w_stim : volt",
            on_pre="I_e_post += w_stim",
            name=f"stim_synapses_{index}",
        )
        injection.connect(i=0, j=all_targets)
        injection.w_stim = (float(amplitude) * coupling) * b2.mV
        objects.extend([generator, injection])
    return objects
