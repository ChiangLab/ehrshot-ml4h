"""Fail-closed contracts for MEDS-style event data and report privacy."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

REQUIRED_COLUMNS: dict[str, str] = {
    "subject_id": "BIGINT",
    "time": "TIMESTAMP",
    "code": "VARCHAR",
}

OPTIONAL_COLUMNS: dict[str, str] = {
    "numeric_value": "FLOAT",
    "text_value": "VARCHAR",
    "unit": "VARCHAR",
    "omop_table": "VARCHAR",
}

EXPECTED_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS


class ContractError(RuntimeError):
    """Raised when governed data fail a required invariant."""


def normalize_duckdb_type(value: str) -> str:
    """Normalize DuckDB aliases while preserving timestamp precision semantics."""
    normalized = value.upper().strip()
    aliases = {
        "INT64": "BIGINT",
        "STRING": "VARCHAR",
        "REAL": "FLOAT",
        "TIMESTAMP WITHOUT TIME ZONE": "TIMESTAMP",
    }
    return aliases.get(normalized, normalized)


def validate_schema(actual: Mapping[str, str], expected: Mapping[str, str] = EXPECTED_COLUMNS) -> None:
    """Require all event fields with their expected logical types."""
    missing = [column for column in expected if column not in actual]
    mismatched = {
        column: (normalize_duckdb_type(actual[column]), normalize_duckdb_type(expected_type))
        for column, expected_type in expected.items()
        if column in actual and normalize_duckdb_type(actual[column]) != normalize_duckdb_type(expected_type)
    }
    if missing or mismatched:
        raise ContractError(f"Schema contract failed; missing={missing}, mismatched={mismatched}")


def validate_required_null_counts(null_counts: Mapping[str, int]) -> None:
    """Reject any null core MEDS key."""
    failures = {
        column: int(null_counts.get(column, -1)) for column in REQUIRED_COLUMNS if int(null_counts.get(column, -1)) != 0
    }
    if failures:
        raise ContractError(f"Required-key null contract failed: {failures}")


def stable_order_columns(columns: Sequence[str] = tuple(EXPECTED_COLUMNS)) -> tuple[str, ...]:
    """Return the deterministic order key without reducing timestamps to dates."""
    optional_ties = tuple(column for column in OPTIONAL_COLUMNS if column in columns)
    return ("subject_id", "time", "code", *optional_ties)


def deterministic_order_clause(columns: Sequence[str] = tuple(EXPECTED_COLUMNS)) -> str:
    """Build a stable DuckDB ORDER BY expression with explicit null placement."""
    return ", ".join(f'"{column}" NULLS FIRST' for column in stable_order_columns(columns))


_IDENTIFIER_ASSIGNMENT = re.compile(
    r"\b(?:subject|patient)(?:_id| id)?\b\s*[:=]\s*[\"']?\d{2,}[\"']?",
    flags=re.IGNORECASE,
)
_USER_PATH = re.compile(r"/(?:Users|home)/[^/\s]+", flags=re.IGNORECASE)


def privacy_violations(text: str) -> list[str]:
    """Find report patterns that could expose a row-level identifier or home path."""
    violations: list[str] = []
    if _IDENTIFIER_ASSIGNMENT.search(text):
        violations.append("subject-like scalar identifier")
    if _USER_PATH.search(text):
        violations.append("absolute user home path")
    home = str(Path.home())
    if home and home in text and "absolute user home path" not in violations:
        violations.append("absolute user home path")
    return violations


def assert_privacy_safe_report(text: str) -> None:
    """Fail before writing a report containing prohibited identifier/path patterns."""
    violations = privacy_violations(text)
    if violations:
        raise ContractError(f"Privacy contract failed: {', '.join(violations)}")
