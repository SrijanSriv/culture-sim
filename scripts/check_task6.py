#!/usr/bin/env python3
"""Print Task 6 SBI status and re-sync the README row from the status file.

.venv/bin/python scripts/check_task6.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from culturesim.fit.task_status import (  # noqa: E402
    DEFAULT_STATUS_PATH,
    read_status,
    sync_readme_task6,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS_PATH)
    parser.add_argument(
        "--sync-readme",
        action="store_true",
        default=True,
        help="rewrite the README Task 6 cell from the status file (default)",
    )
    parser.add_argument(
        "--no-sync-readme",
        action="store_false",
        dest="sync_readme",
    )
    args = parser.parse_args()

    status = read_status(args.status)
    print(json.dumps(status.to_dict(), indent=2, sort_keys=True))
    if args.sync_readme and args.status.exists():
        cell = sync_readme_task6(status, status_path=args.status)
        print(f"\nREADME Task 6 cell -> {cell}")
    if status.state == "done":
        return 0
    if status.state == "failed":
        return 1
    if status.state == "running":
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
