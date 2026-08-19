"""Brian2 smoke tests for the equation strings in ``model/network.py``.

The network builder is Task 1, but the equations are already written down, and a typo
in them would not surface until deep into that task. These tests build a tiny network
directly from the module's equation constants so that the strings are known to be valid
Brian2 -- and so that the Tsodyks-Markram update order, which is easy to get subtly
wrong, is checked against a hand-computed value.

Marked slow: Brian2 imports take a few seconds.
"""

from __future__ import annotations

import pytest

from culturesim.model.network import (
    NEURON_EQUATIONS,
    ON_PRE_EXCITATORY,
    RESET,
    SYNAPSE_MODEL_STP,
    THRESHOLD,
)
from culturesim.model.params import ModelParams

pytestmark = pytest.mark.slow

b2 = pytest.importorskip("brian2", reason="brian2 is a hard dependency (SPEC §2)")


@pytest.fixture(autouse=True)
def _isolated_brian2_scope():
    """Brian2 keeps module-level state; reset it around each test."""
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"  # avoid a compiler dependency in CI
    yield
    b2.device.reinit()
    b2.device.activate()


def _namespace(params: ModelParams) -> dict:
    """The Brian2 namespace, with units attached to the unit-free parameters."""
    return {
        "E_L": params.fixed.E_L * b2.mV,
        "v_th": params.fixed.v_th * b2.mV,
        "v_reset": params.fixed.v_reset * b2.mV,
        "tau_e": params.fixed.tau_e * b2.ms,
        "tau_i": params.fixed.tau_i * b2.ms,
        "tau_a": params.fixed.tau_a * b2.ms,
        "tau_f": params.fixed.tau_f * b2.ms,
        "tau_m": params.free.tau_m * b2.ms,
        "tau_rec": params.free.tau_rec * b2.ms,
        "U": params.free.U,
        "b": params.free.b * b2.mV,
    }


def test_neuron_equations_are_valid_brian2() -> None:
    params = ModelParams()
    group = b2.NeuronGroup(
        10,
        NEURON_EQUATIONS,
        threshold=THRESHOLD,
        reset=RESET,
        refractory=params.fixed.t_ref * b2.ms,
        method="euler",
        namespace=_namespace(params),
    )
    group.v = params.fixed.E_L * b2.mV
    b2.run(1 * b2.ms)
    assert group.v[0] / b2.mV == pytest.approx(params.fixed.E_L, abs=0.1)


def test_a_driven_neuron_spikes_and_adapts() -> None:
    """Adaptation must actually raise ``a`` on each spike, or bursts never terminate."""
    params = ModelParams()
    group = b2.NeuronGroup(
        1,
        NEURON_EQUATIONS,
        threshold=THRESHOLD,
        reset=RESET,
        refractory=params.fixed.t_ref * b2.ms,
        method="euler",
        namespace=_namespace(params),
    )
    group.v = params.fixed.E_L * b2.mV
    group.I_ext = 40 * b2.mV  # well above threshold-distance, so it fires repeatedly
    monitor = b2.SpikeMonitor(group)
    b2.run(200 * b2.ms)

    assert monitor.num_spikes > 1, "a strongly driven LIF must spike"
    assert group.a[0] / b2.mV > 0.0, "spike-frequency adaptation must accumulate"


def test_stp_synapse_equations_are_valid_and_deplete() -> None:
    """SPEC §4.2: on a presynaptic spike, u rises and x is depleted."""
    params = ModelParams()
    namespace = _namespace(params)

    source = b2.SpikeGeneratorGroup(1, [0, 0, 0], [5, 10, 15] * b2.ms)
    target = b2.NeuronGroup(
        1,
        NEURON_EQUATIONS,
        threshold=THRESHOLD,
        reset=RESET,
        refractory=params.fixed.t_ref * b2.ms,
        method="euler",
        namespace=namespace,
    )
    target.v = params.fixed.E_L * b2.mV

    synapses = b2.Synapses(
        source,
        target,
        model=SYNAPSE_MODEL_STP,
        on_pre=ON_PRE_EXCITATORY,
        namespace=namespace,
        method="exact",
    )
    synapses.connect(i=0, j=0)
    synapses.w = params.free.w_e * b2.mV
    synapses.u = 0.0
    synapses.x = 1.0

    b2.run(20 * b2.ms)

    assert synapses.u[0] > 0.0, "utilisation must rise on presynaptic spikes"
    assert synapses.x[0] < 1.0, "resources must deplete -- this is what ends a burst"


def test_first_epsp_amplitude_matches_the_hand_computed_value() -> None:
    """The update order in SPEC §4.2 fixes the first EPSP at ``w * U``.

    ``u`` is incremented to ``U`` (from u=0), the current is delivered as ``w * u * x``
    with ``x`` still 1, and only then is ``x`` depleted. Delivering before the ``u``
    update, or after the ``x`` update, gives a different amplitude -- so this value is
    the check on the ordering.
    """
    params = ModelParams()
    namespace = _namespace(params)

    source = b2.SpikeGeneratorGroup(1, [0], [1] * b2.ms)
    target = b2.NeuronGroup(
        1,
        NEURON_EQUATIONS,
        threshold="v > 1e9*volt",  # never fire, so I_e is observable
        reset=RESET,
        method="euler",
        namespace=namespace,
    )
    synapses = b2.Synapses(
        source, target, model=SYNAPSE_MODEL_STP, on_pre=ON_PRE_EXCITATORY, namespace=namespace
    )
    synapses.connect(i=0, j=0)
    synapses.w = params.free.w_e * b2.mV
    synapses.u = 0.0
    synapses.x = 1.0

    monitor = b2.StateMonitor(target, "I_e", record=0)
    b2.run(2 * b2.ms)

    expected = params.free.w_e * params.free.U
    assert max(monitor.I_e[0] / b2.mV) == pytest.approx(expected, rel=0.02)
    assert synapses.x[0] == pytest.approx(1.0 - params.free.U, rel=0.02)
