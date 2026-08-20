#!/usr/bin/env python3
"""Fail unless an executed private notebook completed every code cell cleanly."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


def validate_executed_notebook(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise ValueError("Executed notebook contains no code cells.")
    incomplete = [index for index, cell in enumerate(code_cells) if cell.execution_count is None]
    errors = [
        index
        for index, cell in enumerate(code_cells)
        if any(output.get("output_type") == "error" for output in cell.get("outputs", []))
    ]
    if incomplete or errors:
        raise RuntimeError(
            f"Executed notebook validation failed; incomplete_code_cells={incomplete}, error_cells={errors}."
        )


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
