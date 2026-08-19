#!/usr/bin/env python3
"""Fetch the default Wagenaar recording into data/raw/ (gitignored).

.venv/bin/python scripts/fetch_wagenaar.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.data.loaders import WAGENAAR_BASE_URL, fetch_wagenaar  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative", default=None, help="archive-relative path to fetch")
    parser.add_argument("--dest", type=Path, default=None)
    args = parser.parse_args()
    kwargs = {}
    if args.relative:
        kwargs["relative"] = args.relative
    if args.dest is not None:
        kwargs["dest"] = args.dest
    path = fetch_wagenaar(**kwargs)
    print(f"wrote {path} ({path.stat().st_size} bytes) from {WAGENAAR_BASE_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
