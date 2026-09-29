#!/usr/bin/env python3
"""Serve the private project explorer on localhost only."""

from __future__ import annotations

import argparse
from pathlib import Path

from temporal_ehr.project_explorer import DEFAULT_DATABASE
from temporal_ehr.project_explorer_web import serve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    serve(args.database, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
