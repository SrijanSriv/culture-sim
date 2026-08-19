"""Tests for the reproducibility machinery (SPEC §11).

Includes a source scan for global RNG use. That scan is the only mechanism that
actually keeps SPEC §11's "no global np.random.seed, no bare random" true -- a review
comment will not catch the tenth module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from culturesim.config import (
    PACKAGE_ROOT,
    canonical_json,
    config_hash,
    load_config,
    merge_overrides,
)
from culturesim.data.cache import SimulationCache
from culturesim.manifest import package_versions, record_run
from culturesim.rng import derive_seed, generator, spawn_generators
from culturesim.stats.spiketrains import SpikeRecording

# Constructing an explicit Generator is the approved way to touch numpy.random.
ALLOWED_NUMPY_RANDOM_ATTRS = frozenset(
    {"default_rng", "Generator", "SeedSequence", "BitGenerator", "PCG64"}
)


def _package_sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _dotted_name(node: ast.AST) -> str | None:
    """Reconstruct ``a.b.c`` from an attribute chain, or None if it is not one."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def test_no_global_rng_use_in_the_package() -> None:
    """SPEC §11: no global np.random.seed, no bare random.

    Parsed rather than grepped so that prose mentioning ``np.random.seed`` -- including
    the docstring in ``rng.py`` explaining this very rule -- is not mistaken for code.
    """
    offenders = []
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                names = [module] if module else [alias.name for alias in node.names]
                if any(name == "random" for name in names if name):
                    location = f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}"
                    offenders.append(f"{location}: import random")
                continue
            if not isinstance(node, ast.Attribute):
                continue
            dotted = _dotted_name(node)
            if dotted is None:
                continue
            for prefix in ("np.random.", "numpy.random."):
                if dotted.startswith(prefix):
                    attr = dotted[len(prefix) :].split(".")[0]
                    if attr not in ALLOWED_NUMPY_RANDOM_ATTRS:
                        offenders.append(
                            f"{path.relative_to(PACKAGE_ROOT)}:{node.lineno}: {dotted}"
                        )
    assert not offenders, (
        "SPEC §11: all randomness must flow from an explicit np.random.Generator.\n"
        + "\n".join(sorted(set(offenders)))
    )


# -- seed derivation ------------------------------------------------------


def test_named_streams_are_reproducible_and_independent() -> None:
    first = generator(42, "topology").random(10)
    again = generator(42, "topology").random(10)
    other = generator(42, "observation").random(10)
    np.testing.assert_array_equal(first, again)
    assert not np.allclose(first, other)


def test_string_stream_labels_are_stable_across_processes() -> None:
    """Python's hash() is salted per process; using it here would break replay."""
    import subprocess
    import sys

    code = "from culturesim.rng import derive_seed; print(derive_seed(7, 'topology'))"
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=PACKAGE_ROOT.parent,
    )
    assert int(out.stdout.strip()) == derive_seed(7, "topology")


def test_run_index_gives_distinct_seeds() -> None:
    seeds = [derive_seed(42, "simulation", i) for i in range(500)]
    assert len(set(seeds)) == len(seeds)


def test_derived_seeds_fit_brian2s_accepted_range() -> None:
    for i in range(50):
        seed = derive_seed(42, "simulation", i)
        assert 0 < seed < 2**32


def test_spawned_generators_are_independent() -> None:
    generators = spawn_generators(42, 4, "pool")
    draws = [g.random(20) for g in generators]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.allclose(draws[i], draws[j])


def test_spawn_is_reproducible() -> None:
    first = [g.random(5) for g in spawn_generators(42, 3, "pool")]
    again = [g.random(5) for g in spawn_generators(42, 3, "pool")]
    for a, b in zip(first, again, strict=True):
        np.testing.assert_array_equal(a, b)


# -- config hashing -------------------------------------------------------


def test_config_hash_is_order_independent() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_content() -> None:
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_config_hash_covers_every_config_it_is_given() -> None:
    """A cache key must not ignore the observation config."""
    model = {"free": {"w_e": 1.0}}
    assert config_hash(model, {"layout": "mcs_60"}) != config_hash(model, {"layout": "hd_mea_1024"})


def test_canonical_json_handles_numpy_scalars() -> None:
    assert canonical_json({"a": np.float64(1.5), "b": np.arange(3)}) == '{"a":1.5,"b":[0,1,2]}'


def test_merge_overrides_supports_dotted_keys() -> None:
    config = {"free": {"w_e": 1.0, "g": 4.0}, "seed": 1}
    merged = merge_overrides(config, {"free.w_e": 2.0, "seed": 9})
    assert merged == {"free": {"w_e": 2.0, "g": 4.0}, "seed": 9}
    assert config["free"]["w_e"] == 1.0, "the input config must not be mutated"


def test_load_config_resolves_bare_filenames() -> None:
    assert load_config("model_default.yaml") == load_config("configs/model_default.yaml")


def test_load_config_reports_a_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("no_such_config.yaml")


# -- manifests ------------------------------------------------------------


def test_manifest_records_what_spec_11_requires(tmp_path) -> None:
    import time

    manifest = record_run(
        command="simulate",
        configs={"model": load_config("model_default.yaml")},
        master_seed=20250819,
        started_at=time.time() - 1.0,
    )
    assert manifest.git_commit  # "unknown" outside a checkout, never empty
    assert manifest.master_seed == 20250819
    assert manifest.config_hash
    assert manifest.wall_clock_s >= 1.0
    assert manifest.package_versions["python"]

    from culturesim.manifest import Manifest

    assert Manifest.read(manifest.write(tmp_path / "manifest.json")) == manifest


def test_tracked_package_versions_are_resolvable() -> None:
    versions = package_versions()
    for name in ("numpy", "h5py", "brian2"):
        assert versions[name] != "not-installed", f"{name} is a hard dependency (SPEC §2)"


# -- simulation cache -----------------------------------------------------


def test_cache_round_trip(tmp_path, poisson_recording: SpikeRecording) -> None:
    cache = SimulationCache(tmp_path / "cache")
    key = cache.key({"free": {"w_e": 1.0}}, {"layout": "mcs_60"})
    assert cache.get(key) is None

    cache.put(key, poisson_recording)
    assert cache.contains(key)
    assert cache.get(key) == poisson_recording
    assert len(cache) == 1


def test_cache_key_depends_on_the_config(tmp_path) -> None:
    cache = SimulationCache(tmp_path / "cache")
    assert cache.key({"w_e": 1.0}) != cache.key({"w_e": 1.1})


def test_disabled_cache_stores_nothing(tmp_path, poisson_recording: SpikeRecording) -> None:
    cache = SimulationCache(tmp_path / "cache", enabled=False)
    key = cache.key({"a": 1})
    assert cache.put(key, poisson_recording) is None
    assert cache.get(key) is None


def test_truncated_cache_entry_is_a_miss_not_a_crash(
    tmp_path, poisson_recording: SpikeRecording
) -> None:
    """An interrupted write must not later look like a valid hit."""
    cache = SimulationCache(tmp_path / "cache")
    key = cache.key({"a": 1})
    path = cache.put(key, poisson_recording)
    assert path is not None
    path.write_bytes(b"not an hdf5 file")
    assert cache.get(key) is None
    assert not cache.contains(key)
