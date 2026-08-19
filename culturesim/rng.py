"""Seed derivation (SPEC §11).

All randomness flows from a single master seed through explicit
:class:`numpy.random.Generator` instances. There is no global ``np.random.seed``
and no bare ``random`` anywhere in the package; ``tests/test_reproducibility.py``
enforces that by grepping for them.

Subprocess simulations (SPEC §4.5) need per-run seeds that are a deterministic
function of the master seed and the run index, so a fit is reproducible without
depending on the order in which the worker pool happens to finish.
"""

from __future__ import annotations

import numpy as np

__all__ = ["generator", "derive_seed", "spawn_generators"]


def generator(master_seed: int, *stream: int | str) -> np.random.Generator:
    """A Generator for a named substream of ``master_seed``.

    ``stream`` labels the consumer, e.g. ``generator(seed, "topology")`` and
    ``generator(seed, "observation", run_index)``. Different labels give
    independent streams, and the same label always gives the same stream.
    """
    return np.random.default_rng(_entropy(master_seed, stream))


def derive_seed(master_seed: int, *stream: int | str, bits: int = 32) -> int:
    """A plain integer seed for libraries that will not take a Generator.

    Brian2 and torch both need this. ``bits`` defaults to 32 because that is the
    widest value Brian2's ``seed()`` accepts on all platforms.
    """
    if bits <= 0 or bits > 63:
        raise ValueError(f"bits must be in 1..63, got {bits}")
    value = np.random.default_rng(_entropy(master_seed, stream)).integers(
        1, 2**bits, dtype=np.int64
    )
    return int(value)


def spawn_generators(master_seed: int, n: int, *stream: int | str) -> list[np.random.Generator]:
    """``n`` independent Generators, one per parallel run."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    root = np.random.SeedSequence(_entropy(master_seed, stream))
    return [np.random.default_rng(child) for child in root.spawn(n)]


def _entropy(master_seed: int, stream: tuple[int | str, ...]) -> list[int]:
    """Fold the stream labels into extra SeedSequence entropy words."""
    words = [int(master_seed)]
    for label in stream:
        if isinstance(label, str):
            # Deliberately not hash(): Python string hashing is salted per
            # process, which would make runs irreproducible across invocations.
            words.append(int.from_bytes(label.encode("utf-8"), "little") % (2**63))
        else:
            words.append(int(label))
    return words
