#!/usr/bin/env python3
"""Fail unless an executed private notebook completed every code cell cleanly."""

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_ehr.alignment_recovery import validate_executed_notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_executed_notebook(args.notebook)
    print(f"Validated complete executed notebook: {args.notebook.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
