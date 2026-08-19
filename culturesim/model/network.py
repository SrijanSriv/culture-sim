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
from typing import TYPE_CHECKING, Any

import numpy as np

from .params import ModelParams

if TYPE_CHECKING:  # pragma: no cover - Brian2 is imported only inside subprocesses
    import brian2 as b2

__all__ = [
    "NEURON_EQUATIONS",
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
NEURON_EQUATIONS = """
dv/dt   = (E_L - v + I_e + I_i + I_ext - a) / tau_m : volt (unless refractory)
dI_e/dt = -I_e / tau_e                              : volt
dI_i/dt = -I_i / tau_i                              : volt
da/dt   = -a / tau_a                                : volt
I_ext                                               : volt
x_pos                                               : meter
y_pos                                               : meter
"""

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
    """
    raise NotImplementedError("Task 1 (SPEC §4.3)")


def connection_probability(distance_um: np.ndarray, lambda_conn_um: float, p_conn: float):
    """``p(d) = p_conn * exp(-d / lambda_conn)`` (SPEC §4.3)."""
    return p_conn * np.exp(-np.asarray(distance_um, dtype=np.float64) / float(lambda_conn_um))


def build_network(
    params: ModelParams,
    positions: NeuronPositions,
    seed: int,
    *,
    record_spikes: bool = True,
    stimulus: Any | None = None,
) -> tuple[b2.Network, dict[str, Any]]:
    """Assemble the Brian2 network.

    Returns the network and a dict of its monitors and groups. Must only be called
    inside a runner subprocess: SPEC §4.5 forbids repeated ``device.reinit()`` in a
    single process under ``cpp_standalone``.

    ``stimulus`` is unused until Task 7 (SPEC §9.3), where the perturbation test
    injects current into neurons near a stimulating electrode.
    """
    raise NotImplementedError("Task 1 (SPEC §4)")
