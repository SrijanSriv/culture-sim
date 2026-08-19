"""Model parameters: free/fixed split and prior bounds (SPEC §4.4).

Parameters are stored as plain floats in the units named in each field's comment,
not as Brian2 quantities. That keeps them trivially YAML-serialisable, hashable
for the run cache, and directly usable as an SBI parameter vector;
:mod:`culturesim.model.network` attaches units at the point of use.

``FREE_PARAM_NAMES`` is the canonical order of the 8-dimensional parameter vector.
Like the fingerprint order it must not be permuted -- an SBI posterior is a
distribution over this exact ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config

__all__ = [
    "FREE_PARAM_NAMES",
    "FreeParams",
    "FixedParams",
    "NetworkParams",
    "SimulationParams",
    "ModelParams",
    "PriorBox",
]

# SPEC §4.4, in table order. Frozen: the SBI posterior is over this ordering.
FREE_PARAM_NAMES: tuple[str, ...] = (
    "p_conn",
    "w_e",
    "g",
    "rate_bg",
    "tau_m",
    "U",
    "tau_rec",
    "b",
)

# Prior ranges from SPEC §4.4. Overridable via `priors:` in model_default.yaml,
# but these are the defaults the priors in that file must agree with.
DEFAULT_PRIOR_RANGES: dict[str, tuple[float, float]] = {
    "p_conn": (0.01, 0.30),
    "w_e": (0.05, 3.0),
    "g": (1.0, 12.0),
    "rate_bg": (0.1, 20.0),
    "tau_m": (10.0, 40.0),
    "U": (0.05, 0.8),
    "tau_rec": (100.0, 5000.0),
    "b": (0.0, 5.0),
}

FREE_PARAM_UNITS: dict[str, str] = {
    "p_conn": "1",
    "w_e": "mV",
    "g": "1",
    "rate_bg": "Hz",
    "tau_m": "ms",
    "U": "1",
    "tau_rec": "ms",
    "b": "mV",
}


@dataclass(frozen=True)
class FreeParams:
    """The 8 fitted parameters (SPEC §4.4)."""

    p_conn: float = 0.20  # 1,  connection probability scale
    w_e: float = 1.5  # mV, EPSP amplitude at a rested synapse (u=U, x=1)
    g: float = 2.0  # 1,  inhibition/excitation weight ratio
    rate_bg: float = 4.98  # Hz, per-afferent rate of the background drive
    tau_m: float = 20.0  # ms, membrane time constant
    U: float = 0.25  # 1,  Tsodyks-Markram utilisation
    tau_rec: float = 2500.0  # ms, Tsodyks-Markram recovery
    b: float = 2.5  # mV, adaptation increment per spike

    def __post_init__(self) -> None:
        for name in FREE_PARAM_NAMES:
            object.__setattr__(self, name, float(getattr(self, name)))

    def to_vector(self) -> np.ndarray:
        """Parameter vector in ``FREE_PARAM_NAMES`` order."""
        return np.array([getattr(self, n) for n in FREE_PARAM_NAMES], dtype=np.float64)

    @classmethod
    def from_vector(cls, vector: Any) -> FreeParams:
        values = np.asarray(vector, dtype=np.float64).ravel()
        if values.size != len(FREE_PARAM_NAMES):
            raise ValueError(
                f"expected {len(FREE_PARAM_NAMES)} free parameters in order "
                f"{FREE_PARAM_NAMES}, got {values.size}"
            )
        return cls(**dict(zip(FREE_PARAM_NAMES, (float(v) for v in values), strict=True)))

    def to_dict(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in FREE_PARAM_NAMES}

    def replace(self, **changes: float) -> FreeParams:
        unknown = set(changes) - set(FREE_PARAM_NAMES)
        if unknown:
            raise KeyError(f"not free parameters: {sorted(unknown)}")
        return replace(self, **changes)


@dataclass(frozen=True)
class FixedParams:
    """Parameters set from the literature (SPEC §4.4).

    Every value carries its source. These are not fitted; changing one is a
    modelling decision that needs its own justification, not a tuning knob.
    """

    # -- Neuron (SPEC §4.1) --
    E_L: float = -70.0  # mV, resting potential, cortical pyramidal; Dayan & Abbott 2001 ch.5
    v_th: float = -50.0  # mV, LIF threshold; Brette & Gerstner 2005, J Neurophysiol 94:3637
    v_reset: float = -60.0  # mV, reset potential; Brunel 2000, J Comput Neurosci 8:183
    t_ref: float = 2.0  # ms, absolute refractory period; Brunel 2000
    tau_e: float = 5.0  # ms, AMPA-mediated decay; Destexhe, Mainen & Sejnowski 1998
    tau_i: float = 10.0  # ms, GABA_A-mediated decay; Destexhe, Mainen & Sejnowski 1998
    tau_a: float = 200.0  # ms, Ca-dependent K adaptation; Benda & Herz 2003, Neural Comput 15:2523
    # -- Synapse (SPEC §4.2) --
    tau_f: float = 100.0  # ms, TM facilitation; Tsodyks & Markram 1997, PNAS 94:719
    # -- Background drive (SPEC §4.3) --
    # External afferents per neuron, each firing at the free parameter `rate_bg`.
    # Brunel 2000 (J Comput Neurosci 8:183) uses C_ext = C_E = 1000 external synapses
    # per neuron for exactly this purpose.
    #
    # These two constants are required for the model to work at all, and their absence
    # from SPEC §4.4 is a gap in the spec rather than a modelling choice.
    #
    # `n_background_synapses`: with a single afferent per neuron, crossing the 20 mV
    # rest-to-threshold gap needs ~4.7 kHz of drive, three orders of magnitude outside
    # `rate_bg`'s 0.1-20 Hz prior, so the network is silent across the whole prior box.
    #
    # `w_background`: the background amplitude must be independent of `w_e`. Driving the
    # background at `w_e` makes the baseline membrane noise scale with the recurrent
    # gain -- raising `w_e` to get bursts to ignite simultaneously raises the spontaneous
    # firing rate -- and no sparse-baseline bursting regime exists anywhere in the prior.
    # Decoupling them lets `rate_bg` set how far below threshold the network idles while
    # `w_e` sets how explosively it recruits. 0.1 mV is a weak distal unitary EPSP; with
    # 1000 afferents the prior spans a mean depolarisation of ~0.3-63 mV.
    n_background_synapses: int = 1000
    w_background: float = 0.1  # mV, EPSP amplitude of one background afferent
    # -- Topology (SPEC §4.3/§4.4) --
    lambda_conn_fraction_of_width: float = 1.0 / 3.0  # SPEC §4.4: one third of array width
    # ms, uniform axonal/synaptic delay range; Wagenaar, Pine & Potter 2006, BMC Neurosci 7:11
    synaptic_delay_ms: tuple[float, float] = (1.0, 3.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "synaptic_delay_ms", tuple(float(d) for d in self.synaptic_delay_ms)
        )
        object.__setattr__(self, "n_background_synapses", int(self.n_background_synapses))
        if self.n_background_synapses < 1:
            raise ValueError(
                f"n_background_synapses must be positive, got {self.n_background_synapses}"
            )
        if self.w_background <= 0:
            raise ValueError(f"w_background must be positive, got {self.w_background}")
        if len(self.synaptic_delay_ms) != 2:
            raise ValueError("synaptic_delay_ms must be a (low, high) pair in ms")
        low, high = self.synaptic_delay_ms
        if not 0.0 <= low <= high:
            raise ValueError(f"require 0 <= low <= high for delays, got {low}, {high}")
        if self.v_th <= self.E_L:
            raise ValueError("threshold must be above the resting potential")
        if self.v_reset >= self.v_th:
            raise ValueError("reset must be below threshold or the neuron cannot recover")


@dataclass(frozen=True)
class NetworkParams:
    """Topology and geometry (SPEC §4.3)."""

    n_neurons: int = 1000  # SPEC §4.3: burst statistics unstable below ~800
    excitatory_fraction: float = 0.8
    sheet_width_um: float = 1800.0
    sheet_height_um: float = 1800.0
    allow_autapses: bool = False
    allow_multapses: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_neurons", int(self.n_neurons))
        if self.n_neurons < 1:
            raise ValueError(f"n_neurons must be positive, got {self.n_neurons}")
        if not 0.0 < self.excitatory_fraction < 1.0:
            raise ValueError(
                f"excitatory_fraction must be in (0, 1), got {self.excitatory_fraction}"
            )
        if self.sheet_width_um <= 0 or self.sheet_height_um <= 0:
            raise ValueError("sheet dimensions must be positive")

    @property
    def n_excitatory(self) -> int:
        return int(round(self.n_neurons * self.excitatory_fraction))

    @property
    def n_inhibitory(self) -> int:
        return self.n_neurons - self.n_excitatory


@dataclass(frozen=True)
class SimulationParams:
    """Run-time settings (SPEC §4.5)."""

    duration_s: float = 300.0
    dt_ms: float = 0.1
    transient_s: float = 20.0  # discarded warm-up; excluded from the output duration
    static_synapses: bool = False  # True gives the Task 1 ablation
    record_neuron_positions: bool = True
    # "diffusion" replaces the explicit Poisson background with an equivalent
    # Ornstein-Uhlenbeck current. Profiling put the explicit version at 6.3 s of a
    # 16.9 s run -- more than the integration loop itself -- because with
    # N*rate*dt ~ 0.5 Brian2's binomial sampler cannot take its normal-approximation
    # fast path. "poisson" is the exact reference, kept so the approximation can be
    # tested rather than assumed (see tests/test_network.py).
    background_mode: str = "diffusion"

    def __post_init__(self) -> None:
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}")
        if self.dt_ms <= 0:
            raise ValueError(f"dt_ms must be positive, got {self.dt_ms}")
        if self.transient_s < 0:
            raise ValueError(f"transient_s must be non-negative, got {self.transient_s}")
        if self.background_mode not in {"diffusion", "poisson"}:
            raise ValueError(
                f"background_mode must be 'diffusion' or 'poisson', got {self.background_mode!r}"
            )

    @property
    def total_duration_s(self) -> float:
        """Biological time actually simulated, including the discarded transient."""
        return self.duration_s + self.transient_s


@dataclass(frozen=True)
class PriorBox:
    """Uniform prior over the free parameters (SPEC §4.4, §8.3)."""

    ranges: Mapping[str, tuple[float, float]]

    def __post_init__(self) -> None:
        coerced = {
            name: (float(self.ranges[name][0]), float(self.ranges[name][1]))
            for name in FREE_PARAM_NAMES
        }
        missing = set(FREE_PARAM_NAMES) - set(self.ranges)
        if missing:
            raise ValueError(f"prior is missing ranges for {sorted(missing)}")
        for name, (low, high) in coerced.items():
            if not low < high:
                raise ValueError(f"prior range for {name} must have low < high, got {low}, {high}")
        object.__setattr__(self, "ranges", coerced)

    @classmethod
    def default(cls) -> PriorBox:
        return cls(ranges=dict(DEFAULT_PRIOR_RANGES))

    @property
    def low(self) -> np.ndarray:
        return np.array([self.ranges[n][0] for n in FREE_PARAM_NAMES], dtype=np.float64)

    @property
    def high(self) -> np.ndarray:
        return np.array([self.ranges[n][1] for n in FREE_PARAM_NAMES], dtype=np.float64)

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """``scipy.optimize``-style bounds in ``FREE_PARAM_NAMES`` order."""
        return [self.ranges[n] for n in FREE_PARAM_NAMES]

    def contains(self, params: FreeParams) -> bool:
        vector = params.to_vector()
        return bool(np.all(vector >= self.low) and np.all(vector <= self.high))

    def clip(self, params: FreeParams) -> FreeParams:
        return FreeParams.from_vector(np.clip(params.to_vector(), self.low, self.high))

    def sample(self, rng: np.random.Generator, n: int = 1) -> list[FreeParams]:
        draws = rng.uniform(self.low, self.high, size=(int(n), len(FREE_PARAM_NAMES)))
        return [FreeParams.from_vector(row) for row in draws]


@dataclass(frozen=True)
class ModelParams:
    """Everything needed to build and run one network."""

    free: FreeParams = FreeParams()
    fixed: FixedParams = FixedParams()
    network: NetworkParams = NetworkParams()
    simulation: SimulationParams = SimulationParams()
    prior: PriorBox = PriorBox.default()
    seed: int = 20250819

    @property
    def lambda_conn_um(self) -> float:
        """Connection length constant: one third of the array width (SPEC §4.4)."""
        return self.network.sheet_width_um * self.fixed.lambda_conn_fraction_of_width

    @property
    def w_i(self) -> float:
        """Inhibitory weight in mV, derived as ``g * w_e`` (SPEC §4.4)."""
        return self.free.g * self.free.w_e

    def with_free(self, free: FreeParams) -> ModelParams:
        return replace(self, free=free)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ModelParams:
        prior_config = config.get("priors") or DEFAULT_PRIOR_RANGES
        return cls(
            free=FreeParams(**_subset(config.get("free", {}), FreeParams)),
            fixed=FixedParams(**_subset(config.get("fixed", {}), FixedParams)),
            network=NetworkParams(**_subset(config.get("network", {}), NetworkParams)),
            simulation=SimulationParams(**_subset(config.get("simulation", {}), SimulationParams)),
            prior=PriorBox(ranges={k: tuple(v) for k, v in prior_config.items()}),
            seed=int(config.get("seed", 20250819)),
        )

    @classmethod
    def load(cls, path: str | Path = "model_default.yaml") -> ModelParams:
        return cls.from_config(load_config(path))

    def to_config(self) -> dict[str, Any]:
        """Round-trips through :meth:`from_config`; this is what the manifest stores."""
        return {
            "seed": self.seed,
            "free": asdict(self.free),
            "fixed": {
                k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self.fixed).items()
            },
            "network": asdict(self.network),
            "simulation": asdict(self.simulation),
            "priors": {k: list(v) for k, v in self.prior.ranges.items()},
        }


def _subset(source: Mapping[str, Any], target: type) -> dict[str, Any]:
    """Keep only keys that are fields of ``target``, rejecting unknown ones loudly.

    A typo in a YAML key must not silently fall back to a default -- that is how a
    fit ends up secretly using different parameters than its config claims.
    """
    valid = {f.name for f in fields(target)}
    unknown = set(source) - valid
    if unknown:
        raise ValueError(
            f"unknown keys for {target.__name__}: {sorted(unknown)}; valid keys are {sorted(valid)}"
        )
    return dict(source)
