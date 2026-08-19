#!/usr/bin/env python3
"""Fetch Wagenaar recordings into data/raw/ (gitignored).

.venv/bin/python scripts/fetch_wagenaar.py
.venv/bin/python scripts/fetch_wagenaar.py --scale   # five DIV-14 cultures for §8.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.data.loaders import (  # noqa: E402
    WAGENAAR_BASE_URL,
    WAGENAAR_SCALE_RELATIVES,
    fetch_wagenaar,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relative", default=None, help="archive-relative path to fetch")
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument(
        "--scale",
        action="store_true",
        help="fetch the DIV-14 cultures used as the across-culture distance scale",
    )
    args = parser.parse_args()
    relatives: list[str]
    if args.scale:
        relatives = list(WAGENAAR_SCALE_RELATIVES)
        if args.relative:
            parser.error("--scale and --relative cannot be combined")
    elif args.relative:
        relatives = [args.relative]
    else:
        relatives = [None]  # type: ignore[list-item]

    for relative in relatives:
        kwargs: dict = {}
        if relative is not None:
            kwargs["relative"] = relative
        if args.dest is not None and not args.scale:
            kwargs["dest"] = args.dest
        path = fetch_wagenaar(**kwargs)
        print(f"wrote {path} ({path.stat().st_size} bytes) from {WAGENAAR_BASE_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
