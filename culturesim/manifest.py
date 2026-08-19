"""Run manifests (SPEC §11).

Every run writes one of these: git commit hash, full config, master seed, package
versions, wall-clock time. Without it a cached result is unattributable and a
figure in the report cannot be traced back to the code that made it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

from .config import REPO_ROOT, config_hash

__all__ = ["Manifest", "record_run", "git_commit", "package_versions"]

# Recorded for every run; a silent version bump in any of these can move a
# fitted posterior, so the manifest has to name them.
TRACKED_PACKAGES = (
    "culturesim",
    "brian2",
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "cl-sdk",
    "sbi",
    "torch",
    "powerlaw",
    "h5py",
    "pynwb",
)


def git_commit(repo_root: Path | None = None) -> str:
    """Current commit, suffixed ``+dirty`` if the tree has uncommitted changes.

    Returns ``"unknown"`` outside a git checkout rather than raising -- a manifest
    that says it does not know is more useful than a crashed run.
    """
    root = repo_root or REPO_ROOT
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"
    return f"{commit}+dirty" if dirty else commit


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = pkg_version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


@dataclass(frozen=True)
class Manifest:
    """Provenance for a single run."""

    command: str
    git_commit: str
    master_seed: int
    configs: dict[str, Any]
    config_hash: str
    package_versions: dict[str, str]
    platform: str
    started_at: str
    wall_clock_s: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def record_run(
    command: str,
    configs: dict[str, Any],
    master_seed: int,
    started_at: float,
    **extra: Any,
) -> Manifest:
    """Build a manifest for a run that began at ``started_at`` (``time.time()``)."""
    return Manifest(
        command=command,
        git_commit=git_commit(),
        master_seed=int(master_seed),
        configs=configs,
        config_hash=config_hash(configs),
        package_versions=package_versions(),
        platform=platform.platform(),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at)),
        wall_clock_s=round(time.time() - started_at, 3),
        extra=dict(extra),
    )
