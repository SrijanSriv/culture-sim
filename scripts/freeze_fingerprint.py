#!/usr/bin/env python3
"""Freeze the fingerprint order at the end of Task 3 (SPEC §3, §13).

Sets ``frozen: true`` and records the SHA-256 of the expanded ``names`` tuple in
``configs/fingerprint.yaml``. After that, any edit to the statistic list makes
``FingerprintSpec.load`` refuse to run, because such an edit invalidates every fit
made against the old vector.

Editing the file in place as text rather than round-tripping it through PyYAML keeps
the comments, which are the only record of *why* each statistic is in the vector.

    python scripts/freeze_fingerprint.py --check     # CI: verify nothing drifted
    python scripts/freeze_fingerprint.py --bump 1.0.0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.config import load_config  # noqa: E402
from culturesim.stats.fingerprint import FingerprintSpec  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "fingerprint.yaml"


def _replace_scalar(text: str, key: str, value: str) -> str:
    """Rewrite ``key: <value>`` at the top level, keeping any trailing comment."""
    pattern = re.compile(rf"^({re.escape(key)}:[ \t]*)([^\n#]*)(#.*)?$", re.MULTILINE)
    if not pattern.search(text):
        raise SystemExit(f"could not find a top-level `{key}:` in {CONFIG_PATH}")

    def substitute(match: re.Match[str]) -> str:
        comment = f" {match.group(3)}" if match.group(3) else ""
        return f"{match.group(1)}{value}{comment}"

    return pattern.sub(substitute, text, count=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the recorded hash matches the expanded names; do not write",
    )
    parser.add_argument("--bump", metavar="VERSION", help="set a new fingerprint version")
    args = parser.parse_args(argv)

    config = load_config(CONFIG_PATH)
    spec = FingerprintSpec.from_config(config)
    expanded_hash = spec.names_sha256

    if args.check:
        if not spec.frozen:
            print(f"fingerprint {spec.version} is not frozen yet ({len(spec)} statistics)")
            return 0
        if spec.declared_sha256 != expanded_hash:
            print(
                f"FAIL: fingerprint {spec.version} declares {spec.declared_sha256} but the "
                f"expanded names hash to {expanded_hash}.\n"
                "Editing a frozen fingerprint invalidates every fit made against it "
                "(SPEC §3). Bump the version and re-run everything.",
                file=sys.stderr,
            )
            return 1
        print(f"OK: fingerprint {spec.version} frozen with {len(spec)} statistics")
        return 0

    text = CONFIG_PATH.read_text(encoding="utf-8")
    if args.bump:
        text = _replace_scalar(text, "version", f'"{args.bump}"')
    text = _replace_scalar(text, "frozen", "true")
    text = _replace_scalar(text, "names_sha256", expanded_hash)
    CONFIG_PATH.write_text(text, encoding="utf-8")

    frozen = FingerprintSpec.from_config(load_config(CONFIG_PATH))
    frozen.check_freeze()
    print(f"froze fingerprint {frozen.version}: {len(frozen)} statistics, {expanded_hash[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
