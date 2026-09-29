#!/usr/bin/env python3
"""Build the governed train/validation project explorer database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from temporal_ehr.project_explorer import DEFAULT_DATABASE, build_project_explorer, database_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--event-path", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--deep-validate-embeddings", action="store_true")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    database = args.database if args.database.is_absolute() else root / args.database
    result = build_project_explorer(
        repository_root=root,
        event_path=args.event_path,
        database_path=database,
        source_commit=args.source_commit,
        deep_validate_embeddings=args.deep_validate_embeddings,
    )
    result["database_sha256"] = database_sha256(database)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
