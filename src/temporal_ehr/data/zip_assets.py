"""Read governed Parquet members in place from downloaded EHRSHOT ZIP assets."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def locate_member(asset_root: Path, suffix: str) -> tuple[Path, str]:
    """Find exactly one ZIP member by suffix, failing on ambiguity."""
    matches: list[tuple[Path, str]] = []
    for archive in sorted(asset_root.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            matches.extend((archive, name) for name in bundle.namelist() if name.endswith(suffix))
    if not matches:
        raise FileNotFoundError(f"No downloaded EHRSHOT ZIP member ends with {suffix!r}.")
    if len(matches) > 1:
        names = [f"{archive.name}:{member}" for archive, member in matches]
        raise ValueError(f"Ambiguous EHRSHOT ZIP member {suffix!r}: {names}")
    return matches[0]


def read_parquet_member(asset_root: Path, suffix: str, columns: list[str] | None = None) -> pa.Table:
    """Read one Parquet member without extracting a persistent patient-level file."""
    archive, member = locate_member(asset_root, suffix)
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as handle:
        return pq.read_table(pa.BufferReader(handle.read()), columns=columns)
