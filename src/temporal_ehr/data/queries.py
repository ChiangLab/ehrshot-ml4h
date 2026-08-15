"""Aggregate-only DuckDB queries for the governed Parquet event table."""

from __future__ import annotations

import re
from collections.abc import Sequence

EVENT_SOURCE = "read_parquet(?)"
EVENT_COLUMNS = (
    "subject_id",
    "time",
    "code",
    "numeric_value",
    "text_value",
    "unit",
    "omop_table",
)
QUANTILE_PROBABILITIES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)


def describe_sql() -> str:
    return f"DESCRIBE SELECT * FROM {EVENT_SOURCE}"


def basic_stats_sql() -> str:
    return f"""
        SELECT
            count(*)::BIGINT AS event_count,
            count(DISTINCT subject_id)::BIGINT AS distinct_subject_count,
            count(DISTINCT code)::BIGINT AS distinct_code_count,
            min(time) AS earliest_time,
            max(time) AS latest_time,
            count(numeric_value)::BIGINT AS numeric_valued_events,
            count(text_value)::BIGINT AS text_valued_events,
            count(unit)::BIGINT AS events_with_units,
            count(*) FILTER (
                WHERE numeric_value IS NOT NULL AND text_value IS NOT NULL
            )::BIGINT AS events_with_both_values,
            count(*) FILTER (
                WHERE numeric_value IS NULL AND text_value IS NULL
            )::BIGINT AS events_with_neither_value,
            count(*) FILTER (WHERE omop_table = 'note')::BIGINT AS note_events,
            count(*) FILTER (
                WHERE omop_table = 'note' AND text_value IS NOT NULL
            )::BIGINT AS note_events_with_text
        FROM {EVENT_SOURCE}
    """


def null_counts_sql(columns: Sequence[str] = EVENT_COLUMNS) -> str:
    expressions = [f'count(*) FILTER (WHERE "{column}" IS NULL)::BIGINT AS "{column}"' for column in columns]
    return f"SELECT {', '.join(expressions)} FROM {EVENT_SOURCE}"


def table_counts_sql() -> str:
    return f"""
        SELECT
            coalesce(omop_table, '[missing]') AS omop_table,
            count(*)::BIGINT AS event_count,
            count(DISTINCT subject_id)::BIGINT AS distinct_subject_count,
            count(DISTINCT code)::BIGINT AS distinct_code_count
        FROM {EVENT_SOURCE}
        GROUP BY 1
        ORDER BY event_count DESC, omop_table
    """


def vocabulary_counts_sql(prefixes: Sequence[str]) -> str:
    """Classify codes into an allowlist of public vocabulary names plus Other."""
    safe_prefixes: list[str] = []
    for prefix in prefixes:
        if not re.fullmatch(r"[A-Za-z0-9_ -]+", prefix):
            raise ValueError(f"Unsafe vocabulary prefix in configuration: {prefix!r}")
        safe_prefixes.append(prefix)
    cases = "\n".join(
        "WHEN lower(code) = lower('{0}') OR lower(code) LIKE lower('{0}/%') THEN '{0}'".format(
            prefix.replace("'", "''")
        )
        for prefix in safe_prefixes
    )
    return f"""
        WITH classified AS (
            SELECT
                CASE
                    {cases}
                    ELSE 'Other'
                END AS vocabulary,
                code
            FROM {EVENT_SOURCE}
        )
        SELECT
            vocabulary,
            count(*)::BIGINT AS event_count,
            count(DISTINCT code)::BIGINT AS distinct_code_count
        FROM classified
        GROUP BY vocabulary
        ORDER BY event_count DESC, vocabulary
    """


def subject_aggregate_values_sql() -> str:
    """Return de-identified numeric subject aggregates for in-memory summarization only."""
    return f"""
        WITH per_subject AS (
            SELECT
                subject_id,
                count(*)::BIGINT AS events_per_subject,
                count(DISTINCT code)::BIGINT AS distinct_codes_per_subject,
                date_diff('second', min(time), max(time)) / 31556952.0
                    AS timeline_duration_years,
                count(DISTINCT CAST(time AS DATE))::BIGINT AS unique_event_dates
            FROM {EVENT_SOURCE}
            GROUP BY subject_id
        ),
        per_timestamp AS (
            SELECT subject_id, time, count(*)::BIGINT AS concurrency
            FROM {EVENT_SOURCE}
            GROUP BY subject_id, time
        ),
        max_concurrency AS (
            SELECT subject_id, max(concurrency)::BIGINT AS max_same_timestamp_concurrency
            FROM per_timestamp
            GROUP BY subject_id
        )
        SELECT
            events_per_subject,
            distinct_codes_per_subject,
            timeline_duration_years,
            unique_event_dates,
            max_same_timestamp_concurrency
        FROM per_subject
        JOIN max_concurrency USING (subject_id)
    """


def exact_duplicate_sql() -> str:
    columns = ", ".join(EVENT_COLUMNS)
    return f"""
        WITH groups AS (
            SELECT count(*)::BIGINT AS group_size
            FROM {EVENT_SOURCE}
            GROUP BY {columns}
            HAVING count(*) > 1
        )
        SELECT
            count(*)::BIGINT AS duplicate_groups,
            coalesce(sum(group_size), 0)::BIGINT AS rows_in_duplicate_groups,
            coalesce(sum(group_size - 1), 0)::BIGINT AS excess_duplicate_rows
        FROM groups
    """


def subject_date_code_duplicate_sql() -> str:
    return f"""
        WITH groups AS (
            SELECT count(*)::BIGINT AS group_size
            FROM {EVENT_SOURCE}
            GROUP BY subject_id, CAST(time AS DATE), code
            HAVING count(*) > 1
        )
        SELECT
            count(*)::BIGINT AS duplicate_subject_date_code_groups,
            coalesce(sum(group_size), 0)::BIGINT AS rows_in_subject_date_code_groups,
            coalesce(sum(group_size - 1), 0)::BIGINT AS excess_subject_date_code_rows
        FROM groups
    """


def recurrence_sql() -> str:
    return f"""
        WITH recurring AS (
            SELECT
                subject_id,
                code,
                count(DISTINCT CAST(time AS DATE))::BIGINT AS event_dates
            FROM {EVENT_SOURCE}
            GROUP BY subject_id, code
            HAVING count(DISTINCT CAST(time AS DATE)) > 1
        )
        SELECT
            count(*)::BIGINT AS recurring_subject_code_pairs,
            count(DISTINCT subject_id)::BIGINT AS subjects_with_repeated_codes_across_dates,
            coalesce(sum(event_dates - 1), 0)::BIGINT AS recurrence_dates_beyond_first
        FROM recurring
    """


def concurrency_sql() -> str:
    return f"""
        WITH concurrent AS (
            SELECT subject_id, time, count(*)::BIGINT AS group_size
            FROM {EVENT_SOURCE}
            GROUP BY subject_id, time
            HAVING count(*) > 1
        )
        SELECT
            count(*)::BIGINT AS same_timestamp_groups,
            count(DISTINCT subject_id)::BIGINT AS subjects_with_same_timestamp_events,
            coalesce(sum(group_size), 0)::BIGINT AS events_in_same_timestamp_groups,
            coalesce(sum(group_size - 1), 0)::BIGINT AS excess_same_timestamp_events
        FROM concurrent
    """


def subject_integrity_sql() -> str:
    return f"""
        WITH subject_checks AS (
            SELECT
                subject_id,
                count(DISTINCT CAST(time AS DATE))::BIGINT AS unique_dates,
                count(*) FILTER (
                    WHERE coalesce(omop_table, '') NOT IN
                        ('person', 'visit_occurrence', 'visit_detail')
                )::BIGINT AS non_person_or_administrative_events
            FROM {EVENT_SOURCE}
            GROUP BY subject_id
        )
        SELECT
            count(*) FILTER (WHERE unique_dates < 3)::BIGINT
                AS subjects_with_fewer_than_three_unique_event_dates,
            count(*) FILTER (WHERE non_person_or_administrative_events = 0)::BIGINT
                AS subjects_with_only_person_or_administrative_events
        FROM subject_checks
    """


def chronological_order_sql() -> str:
    return f"""
        WITH ordered AS (
            SELECT
                subject_id,
                time,
                lag(time) OVER (
                    PARTITION BY subject_id
                    ORDER BY time, code, numeric_value NULLS FIRST, text_value NULLS FIRST,
                        unit NULLS FIRST, omop_table NULLS FIRST
                ) AS prior_time
            FROM {EVENT_SOURCE}
        )
        SELECT count(DISTINCT subject_id)::BIGINT AS subjects_out_of_order_after_explicit_sort
        FROM ordered
        WHERE time < prior_time
    """


def range_check_sql() -> str:
    return f"""
        WITH bounds AS (
            SELECT min(time) AS minimum_time, max(time) AS maximum_time
            FROM {EVENT_SOURCE}
        )
        SELECT count(*)::BIGINT AS timestamps_outside_observed_global_range
        FROM {EVENT_SOURCE}, bounds
        WHERE time < minimum_time OR time > maximum_time
    """


def visit_audit_sql() -> str:
    return f"""
        SELECT
            count(*)::BIGINT AS visit_occurrence_events,
            count(DISTINCT code)::BIGINT AS distinct_visit_codes,
            count(*) FILTER (WHERE lower(code) LIKE 'visit/%')::BIGINT
                AS visit_prefixed_events,
            count(DISTINCT code) FILTER (WHERE lower(code) LIKE 'visit/%')::BIGINT
                AS distinct_visit_prefixed_codes,
            count(*) FILTER (
                WHERE lower(code) LIKE '%start%' OR lower(code) LIKE '%admit%'
            )::BIGINT AS code_text_suggesting_start,
            count(*) FILTER (
                WHERE lower(code) LIKE '%end%' OR lower(code) LIKE '%discharge%'
            )::BIGINT AS code_text_suggesting_end,
            count(numeric_value)::BIGINT AS visit_rows_with_numeric_value,
            count(text_value)::BIGINT AS visit_rows_with_text_value,
            count(unit)::BIGINT AS visit_rows_with_unit
        FROM {EVENT_SOURCE}
        WHERE omop_table = 'visit_occurrence'
    """


def visit_code_prefix_counts_sql() -> str:
    """Summarize visit-row code forms using public vocabulary-prefix names only."""
    return f"""
        SELECT
            split_part(code, '/', 1) AS vocabulary_prefix,
            count(*)::BIGINT AS event_count,
            count(DISTINCT code)::BIGINT AS distinct_code_count
        FROM {EVENT_SOURCE}
        WHERE omop_table = 'visit_occurrence'
        GROUP BY 1
        ORDER BY event_count DESC, vocabulary_prefix
    """
