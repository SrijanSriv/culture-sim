"""culture-sim: a parameter-calibrated in-silico model of a dissociated neuronal
culture on a multi-electrode array.

Read ``SPEC.md`` §0 before using this for anything. The model reproduces a defined
list of MEA statistics and is explicitly not valid outside them.

Submodules are imported lazily so that ``culture-sim --help`` does not pay the cost
of importing Brian2 and torch.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_LAZY_SUBMODULES = {
    "model",
    "observation",
    "stats",
    "data",
    "fit",
    "interop",
    "validate",
    "report",
    "cli",
    "config",
    "manifest",
    "rng",
}

_LAZY_ATTRS = {
    "SpikeRecording": "culturesim.stats.spiketrains",
    "load_recording": "culturesim.stats.spiketrains",
    "save_recording": "culturesim.stats.spiketrains",
    "Fingerprint": "culturesim.stats.fingerprint",
    "FingerprintSpec": "culturesim.stats.fingerprint",
    "ModelParams": "culturesim.model.params",
    "FreeParams": "culturesim.model.params",
}

if TYPE_CHECKING:  # pragma: no cover
    # Re-exported lazily at runtime via __getattr__; declared here for type checkers.
    from .model.params import FreeParams as FreeParams
    from .model.params import ModelParams as ModelParams
    from .stats.fingerprint import Fingerprint as Fingerprint
    from .stats.fingerprint import FingerprintSpec as FingerprintSpec
    from .stats.spiketrains import SpikeRecording as SpikeRecording
    from .stats.spiketrains import load_recording as load_recording
    from .stats.spiketrains import save_recording as save_recording

__all__ = ["__version__", *sorted(_LAZY_ATTRS), *sorted(_LAZY_SUBMODULES)]


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return import_module(f"culturesim.{name}")
    if name in _LAZY_ATTRS:
        return getattr(import_module(_LAZY_ATTRS[name]), name)
    raise AttributeError(f"module 'culturesim' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
