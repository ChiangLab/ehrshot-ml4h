"""Filename-, type-, and schema-only inventory of local EHRSHOT assets."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from temporal_ehr.config import logical_path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a file without interpreting its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _file_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".parquet": "Apache Parquet",
        ".zip": "ZIP archive",
        ".json": "JSON metadata",
        ".csv": "CSV",
        ".md": "Markdown",
        ".txt": "text",
    }.get(suffix, suffix.removeprefix(".") or "unknown")


def _arrow_schema_summary(parquet_file: pq.ParquetFile) -> dict[str, Any]:
    schema = parquet_file.schema_arrow
    return {
        "row_count": int(parquet_file.metadata.num_rows),
        "row_groups": int(parquet_file.metadata.num_row_groups),
        "columns": [
            {"column": field.name, "type": str(field.type), "nullable": bool(field.nullable)} for field in schema
        ],
    }


def _zip_parquet_schema(archive: Path, member: str) -> dict[str, Any]:
    """Read only a Parquet member's schema/footer, never its record values."""
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as handle:
        buffer = pa.BufferReader(handle.read())
        return _arrow_schema_summary(pq.ParquetFile(buffer))


def _zip_parquet_column_counts(archive: Path, member: str, column: str) -> dict[str, int]:
    """Aggregate one non-identifying categorical metadata column in memory."""
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as handle:
        buffer = pa.BufferReader(handle.read())
        table = pq.read_table(buffer, columns=[column])
    counts = Counter("[missing]" if value is None else str(value) for value in table[column].to_pylist())
    return dict(sorted(counts.items()))


def _local_parquet_schema(path: Path) -> dict[str, Any]:
    return _arrow_schema_summary(pq.ParquetFile(path))


def _schema_names(summary: Mapping[str, Any] | None) -> set[str]:
    if not summary:
        return set()
    return {str(field["column"]) for field in summary.get("columns", [])}


def _asset(
    *,
    name: str,
    status: str,
    evidence: str,
    why_required: str,
    can_continue_without: bool,
    blocks_phase: str,
    likely_official_source: str,
) -> dict[str, Any]:
    return {
        "asset": name,
        "status": status,
        "evidence": evidence,
        "why_required": why_required,
        "can_continue_without": can_continue_without,
        "blocks_phase": blocks_phase,
        "likely_official_source": likely_official_source,
    }


def inventory_assets(
    asset_root: Path,
    data_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Inspect paths, ZIP member names, and Parquet schemas without reading record values."""
    expected_tasks = set(config["assets"]["expected_tasks"])
    official_source = str(config["official_repository"]["restricted_dataset_source"])
    files: list[dict[str, Any]] = []
    virtual_members: list[tuple[str, Path | None, str | None]] = []
    archive_schemas: dict[str, dict[str, Any]] = {}
    dataset_metadata: dict[str, Any] = {}
    archive_details: list[dict[str, Any]] = []

    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    for path in sorted(asset_root.rglob("*")):
        if not path.is_file() or ignored_parts.intersection(path.relative_to(asset_root).parts):
            continue
        relative = logical_path(path, asset_root)
        entry = {
            "path": relative,
            "type": _file_type(relative),
            "size_bytes": int(path.stat().st_size),
        }
        if path.suffix.lower() in {".zip", ".json", ".md", ".txt"} or path == data_path:
            entry["sha256"] = sha256_file(path)
        files.append(entry)
        virtual_members.append((relative, path, None))

        if path.suffix.lower() != ".zip":
            continue
        with zipfile.ZipFile(path) as bundle:
            member_names = sorted(name for name in bundle.namelist() if not name.endswith("/"))
            archive_details.append(
                {
                    "path": relative,
                    "member_count": len(member_names),
                    "members": [{"path": member, "type": _file_type(member)} for member in member_names],
                }
            )
            for member in member_names:
                virtual_members.append((member, path, member))
                lower = member.lower()
                if lower.endswith(".parquet"):
                    archive_schemas[member] = _zip_parquet_schema(path, member)
                elif lower.endswith("metadata/dataset.json"):
                    raw_metadata = json.loads(bundle.read(member))
                    allowlist = {
                        "dataset_name",
                        "dataset_version",
                        "etl_name",
                        "etl_version",
                        "meds_version",
                        "created_at",
                    }
                    dataset_metadata = {key: raw_metadata[key] for key in allowlist if key in raw_metadata}

    all_names = [name for name, _, _ in virtual_members]
    task_members: dict[str, str] = {}
    for name in all_names:
        match = re.search(r"(?:^|/)labels/([^/]+)/labels\.parquet$", name)
        if match:
            task_members[match.group(1)] = name
    found_tasks = set(task_members)
    label_schemas = {task: archive_schemas.get(member) for task, member in sorted(task_members.items())}
    prediction_time_tasks = {
        task for task, schema in label_schemas.items() if "prediction_time" in _schema_names(schema)
    }
    label_subject_key_tasks = {task for task, schema in label_schemas.items() if "subject_id" in _schema_names(schema)}

    split_member = next(
        (name for name in all_names if name.lower().endswith("metadata/subject_splits.parquet")),
        None,
    )
    split_schema = archive_schemas.get(split_member) if split_member else None
    split_schema_names = _schema_names(split_schema)
    split_value_counts: dict[str, int] = {}
    if split_member:
        split_location = next(
            (archive for name, archive, member in virtual_members if name == split_member and member),
            None,
        )
        if split_location is not None and "split" in split_schema_names:
            split_value_counts = _zip_parquet_column_counts(split_location, split_member, "split")
    codes_member = next((name for name in all_names if name.lower().endswith("metadata/codes.parquet")), None)
    codes_schema = archive_schemas.get(codes_member) if codes_member else None

    few_shot_candidates = [
        name for name in all_names if re.search(r"few[-_ ]?shot|shots?/|shot_samples", name, flags=re.IGNORECASE)
    ]
    license_candidates = [name for name in all_names if re.search(r"(?:^|/)(?:dua|license)(?:\.|/|$)", name, re.I)]
    manifest_candidates = [name for name in all_names if re.search(r"checksum|sha256|release[_-]?manifest", name, re.I)]
    baseline_candidates = [
        name
        for name in all_names
        if re.search(r"(?:^|/)(?:features?|results?)(?:/|\.|$)|clmbr|count_features", name, re.I)
    ]

    missing_tasks = sorted(expected_tasks - found_tasks)
    extra_tasks = sorted(found_tasks - expected_tasks)
    labels_complete = not missing_tasks and len(found_tasks) == len(expected_tasks)
    prediction_times_complete = labels_complete and prediction_time_tasks == expected_tasks
    subject_keys_complete = labels_complete and label_subject_key_tasks == expected_tasks
    split_semantics = dict(config["assets"]["split_semantics"])
    expected_splits = set(split_semantics)
    splits_present = bool(
        split_member
        and {"subject_id", "split"}.issubset(split_schema_names)
        and set(split_value_counts) == expected_splits
    )
    codes_present = bool(codes_member and "code" in _schema_names(codes_schema))
    version_present = bool(dataset_metadata.get("dataset_version"))

    assets = [
        _asset(
            name="MEDS event table",
            status="present",
            evidence=(
                f"Apache Parquet event table located as {logical_path(data_path, asset_root)}; "
                "schema and aggregate validity are assessed in audit.json."
            ),
            why_required="Provides the canonical longitudinal event stream for all later phases.",
            can_continue_without=False,
            blocks_phase="All data and modeling phases",
            likely_official_source=official_source,
        ),
        _asset(
            name="Task-label files for all 15 EHRSHOT tasks",
            status="present" if labels_complete and subject_keys_complete else "partial_or_missing",
            evidence=(
                f"Found {len(found_tasks)} expected task members; missing={missing_tasks}; "
                f"unexpected={extra_tasks}; all expose subject_id={subject_keys_complete}."
            ),
            why_required="Defines official task outcomes; labels must never be reconstructed ad hoc.",
            can_continue_without=True,
            blocks_phase="Label validation and all benchmark modeling",
            likely_official_source=official_source,
        ),
        _asset(
            name="Official prediction-time fields",
            status="present" if prediction_times_complete else "partial_or_missing",
            evidence=(
                f"prediction_time is present in {len(prediction_time_tasks)}/{len(expected_tasks)} "
                "expected task schemas."
            ),
            why_required="Required to enforce task-specific censoring and prevent temporal leakage.",
            can_continue_without=True,
            blocks_phase="Censoring validation and all benchmark modeling",
            likely_official_source=official_source,
        ),
        _asset(
            name="Canonical train/validation/test patient splits",
            status="present" if splits_present else "missing",
            evidence=(
                "metadata/subject_splits.parquet is present with subject_id and split columns; "
                "MEDS train/tuning/held_out map to benchmark train/validation/test in the official ETL."
                if splits_present
                else "No split artifact with the required schema was located."
            ),
            why_required="Prevents patient overlap and preserves benchmark comparability.",
            can_continue_without=True,
            blocks_phase="Split validation, baseline reproduction, and modeling",
            likely_official_source=official_source,
        ),
        _asset(
            name="Canonical few-shot samples",
            status="present" if few_shot_candidates else "missing",
            evidence=(
                f"Located {len(few_shot_candidates)} filename candidates."
                if few_shot_candidates
                else "No few-shot or released-shot-sample filenames were found."
            ),
            why_required="Needed to reproduce official 8-shot and 64-shot secondary regimes.",
            can_continue_without=True,
            blocks_phase="Few-shot benchmark analyses only",
            likely_official_source=official_source,
        ),
        _asset(
            name="Code/concept descriptions",
            status="present" if codes_present else "missing",
            evidence=(
                "metadata/codes.parquet is present and exposes a code field; its full schema is recorded."
                if codes_present
                else "No code metadata artifact with a code field was found."
            ),
            why_required="Required for reproducible mapping from structured codes to descriptions.",
            can_continue_without=True,
            blocks_phase="Text serialization and predecessor reproduction",
            likely_official_source=official_source,
        ),
        _asset(
            name="Data-version metadata",
            status="present" if version_present else "missing",
            evidence=(
                "metadata/dataset.json contains allowlisted dataset, ETL, and MEDS version fields."
                if version_present
                else "No parseable dataset-version metadata was located."
            ),
            why_required="Pins the event, label, split, and metadata artifacts to a reproducible release.",
            can_continue_without=True,
            blocks_phase="Release provenance freeze",
            likely_official_source=official_source,
        ),
        _asset(
            name="DUA or license information",
            status="present" if license_candidates else "missing_locally",
            evidence=(
                f"Located {len(license_candidates)} local filename candidates."
                if license_candidates
                else "No local DUA/license file was found; official public DUA.md was reviewed."
            ),
            why_required="Documents authorization, security obligations, and redistribution limits.",
            can_continue_without=True,
            blocks_phase="Governance sign-off before patient-level research use",
            likely_official_source=(
                f"{config['official_repository']['url']}/blob/{config['official_repository']['commit']}/DUA.md"
            ),
        ),
        _asset(
            name="Released checksums or release manifest",
            status="present" if manifest_candidates else "missing",
            evidence=(
                f"Located {len(manifest_candidates)} manifest/checksum filename candidates."
                if manifest_candidates
                else "No released checksum or release-manifest filename was found; local hashes were computed."
            ),
            why_required="Cryptographically binds separately downloaded assets to one official release.",
            can_continue_without=True,
            blocks_phase="Complete release-provenance resolution",
            likely_official_source=official_source,
        ),
        _asset(
            name="Released baseline features or results",
            status="present" if baseline_candidates else "missing",
            evidence=(
                f"Located {len(baseline_candidates)} baseline filename candidates."
                if baseline_candidates
                else "No released feature/result filenames were found."
            ),
            why_required="Supports the required reproduction of one released count-based benchmark.",
            can_continue_without=True,
            blocks_phase="Released-baseline reproduction",
            likely_official_source=official_source,
        ),
    ]

    return {
        "report_kind": "EHRSHOT asset inventory",
        "scope": (
            "Filenames, file types, checksums, JSON version fields, Parquet schemas, and aggregate "
            "split-category counts only"
        ),
        "asset_root": ".",
        "files": files,
        "archives": archive_details,
        "dataset_metadata": dataset_metadata,
        "label_summary": {
            "expected_task_count": len(expected_tasks),
            "found_task_count": len(found_tasks),
            "found_tasks": sorted(found_tasks),
            "missing_tasks": missing_tasks,
            "unexpected_tasks": extra_tasks,
            "all_label_schemas_have_subject_id": subject_keys_complete,
            "all_label_schemas_have_prediction_time": prediction_times_complete,
            "schemas": label_schemas,
        },
        "split_schema": split_schema,
        "split_value_counts": split_value_counts,
        "split_semantic_mapping": split_semantics,
        "code_metadata_schema": codes_schema,
        "assets": assets,
        "official_repository": dict(config["official_repository"]),
        "privacy": {
            "event_or_label_rows_opened": False,
            "split_subject_identifiers_read": False,
            "patient_identifiers_written": False,
            "absolute_source_paths_written": False,
        },
    }
