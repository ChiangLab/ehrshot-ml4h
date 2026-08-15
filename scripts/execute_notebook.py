#!/usr/bin/env python3
"""Execute a clean source notebook into the governed private output tree."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--timeout", type=int, default=21_600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    destination = args.destination.resolve()
    private_root = (repository_root / "outputs/private/executed_notebooks").resolve()
    if source == destination:
        raise ValueError("Executed notebooks must never overwrite clean source notebooks.")
    if not destination.is_relative_to(private_root):
        raise ValueError("Executed notebooks must be written under outputs/private/executed_notebooks/.")
    notebook = nbformat.read(source, as_version=4)
    if any(cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code"):
        raise ValueError(f"Source notebook contains stored output: {source.name}")
    if any(cell.get("execution_count") is not None for cell in notebook.cells if cell.cell_type == "code"):
        raise ValueError(f"Source notebook contains execution counts: {source.name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(repository_root)}},
        allow_errors=False,
    )
    try:
        client.execute()
    finally:
        nbformat.write(notebook, destination)
    print(f"Executed {source.name} -> outputs/private/executed_notebooks/{destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
