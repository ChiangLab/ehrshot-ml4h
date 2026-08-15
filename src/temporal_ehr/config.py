"""Configuration and path handling without user-specific absolute defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/expected_ehrshot.yaml")


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the expected aggregate EHRSHOT manifest."""
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return loaded


def resolve_path(
    explicit: Path | str | None,
    environment_variable: str,
    *,
    must_be_file: bool = False,
    must_be_dir: bool = False,
) -> Path:
    """Resolve a CLI or environment path and validate its type."""
    raw = str(explicit) if explicit is not None else os.environ.get(environment_variable)
    if not raw:
        raise ValueError(f"Provide a path explicitly or set the {environment_variable} environment variable.")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path.name}")
    if must_be_file and not path.is_file():
        raise ValueError(f"Expected a file: {path.name}")
    if must_be_dir and not path.is_dir():
        raise ValueError(f"Expected a directory: {path.name}")
    return path


def logical_path(path: Path, root: Path | None = None) -> str:
    """Return a report-safe path with no home directory or username."""
    resolved = path.resolve()
    if root is not None:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.name
