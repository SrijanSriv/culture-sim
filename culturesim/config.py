"""YAML config loading and config hashing.

SPEC §11 requires that simulation outputs be cached by config hash and that every
run record its full config. Both need one canonical serialisation of a config, so
it lives here rather than being reimplemented per module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "DEFAULT_CONFIG_DIR",
    "load_config",
    "merge_overrides",
    "config_hash",
    "canonical_json",
]

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. Bare filenames resolve against ``configs/``."""
    candidate = Path(path)
    if not candidate.exists() and not candidate.is_absolute():
        candidate = DEFAULT_CONFIG_DIR / candidate.name
    if not candidate.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with candidate.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config {candidate} must be a YAML mapping, got {type(loaded).__name__}")
    return loaded


def merge_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``config``.

    Supports dotted keys (``"free.w_e": 1.2``) so the CLI can override a single
    leaf without restating the surrounding block.
    """
    merged = json.loads(canonical_json(config))
    for key, value in overrides.items():
        target = merged
        parts = key.split(".")
        for part in parts[:-1]:
            existing = target.get(part)
            if not isinstance(existing, dict):
                existing = {}
                target[part] = existing
            target = existing
        leaf = parts[-1]
        if isinstance(value, dict) and isinstance(target.get(leaf), dict):
            target[leaf] = merge_overrides(target[leaf], value)
        else:
            target[leaf] = value
    return merged


def canonical_json(obj: Any) -> str:
    """Deterministic JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_coerce)


def _coerce(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"cannot serialise {type(obj).__name__} into a config hash")


def config_hash(*objs: Any, length: int = 16) -> str:
    """Stable short hash over one or more configs, used as the cache key."""
    digest = hashlib.sha256()
    for obj in objs:
        digest.update(canonical_json(obj).encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:length]
